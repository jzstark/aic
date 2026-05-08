import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

import numpy as np
import torch
import torch.nn as nn

from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench


# ---------------------------------------------------------------------------
# Hyper-parameters matching Isaac Lab training
# ---------------------------------------------------------------------------

# The 6 UR5e arm joints and their default positions (from ArticulationCfg.InitialStateCfg).
# Verify _ARM_JOINT_IDXS with inspect_joints.py — assumed to be 0-5 (arm before cable).
_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_ARM_JOINT_DEFAULT = np.array(
    [0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110], dtype=np.float32
)
# Indices of the 6 arm joints within the full robot joint array (arm + cable).
# Run inspect_joints.py inside the Isaac Lab container to verify.
_ARM_JOINT_IDXS = list(range(6))  # assumed 0-5; update after inspect_joints.py

# obs_dim is read from the checkpoint at load time (_OBS_DIM set in __init__).
# Training obs: joint_pos_rel(N) + joint_vel(N) + eef_pose(7) + port_rel(3) + last_action(6)
# where N = total robot joints (arm + cable, e.g. 46 → obs_dim=108).
_ACT_DIM = 6
_HIDDEN = [512, 256, 128]

# sim.dt=1/120, decimation=4 → control at 30 Hz; DifferentialIK scale=0.05.
# velocity = action × 0.05 × 30 Hz = action × 1.5 m/s (exact equivalence).
# SPEED_FACTOR < 1.0 adds a safety margin; tune empirically in Gazebo.
_ISAAC_VEL_SCALE = 1.5   # m/s per unit action (action * 0.05m * 30Hz)
_SPEED_FACTOR = 0.07      # conservative for insertion; max vel = clip * 1.5 * 0.07 ≈ 0.1 m/s
_CONTROL_HZ = 20.0


# ---------------------------------------------------------------------------
# Quaternion utilities (wxyz convention — matches _lookup_pos_quat output)
# ---------------------------------------------------------------------------

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate = inverse for unit quaternion [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Quaternion wxyz → rotation vector (axis × angle) in same reference frame."""
    w = float(np.clip(q[0], -1.0, 1.0))
    half = np.arccos(abs(w))
    s = np.sin(half)
    if s < 1e-7:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([q[1], q[2], q[3]], dtype=np.float32) / s
    if w < 0:
        axis = -axis
    return axis * (2.0 * half)


# ---------------------------------------------------------------------------
# Actor network (mirrors RSL-RL ActorCritic with per-actor obs normalizer)
# ---------------------------------------------------------------------------

class _EmpiricalNorm(nn.Module):
    """Matches RSL-RL's EmpiricalNormalization buffer shapes exactly.

    _mean/_var/_std are [1, shape] (batch dim kept); count is a scalar tensor.
    """

    def __init__(self, shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, shape))
        self.register_buffer("_var", torch.ones(1, shape))
        self.register_buffer("_std", torch.ones(1, shape))
        self.register_buffer("count", torch.tensor(0.0))  # scalar []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class _Actor(nn.Module):
    """Actor MLP — attribute name actor_obs_normalizer matches RSL-RL checkpoint keys."""

    def __init__(self, obs_dim: int, hidden: list, act_dim: int):
        super().__init__()
        self.actor_obs_normalizer = _EmpiricalNorm(obs_dim)
        layers: list = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.actor_obs_normalizer(obs))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RLPolicy(Policy):
    """RL policy that loads an RSL-RL PPO checkpoint and runs inference in Gazebo."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint from ROS 2 share directory
        ckpt_path = Path(get_package_share_directory("my_policy_node")) / "checkpoints" / "model_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Copy it with: docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/"
                "aic_task/<timestamp>/model_<iter>.pt "
                f"{ckpt_path}"
            )

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = raw.get("model_state_dict", raw)

        # Derive obs_dim and n_joints from checkpoint shape.
        obs_dim = int(state_dict["actor.0.weight"].shape[1])
        self._n_joints = (obs_dim - 16) // 2  # obs = joints*2 + eef(7) + port(3) + action(6)
        self.get_logger().info(
            f"Checkpoint obs_dim={obs_dim}, n_joints={self._n_joints}, "
            f"arm_idxs={_ARM_JOINT_IDXS}"
        )

        # Build actor with correct obs_dim.
        self._actor = _Actor(obs_dim, _HIDDEN, _ACT_DIM)

        # Strip 'actor_critic.' prefix; keep actor weights and obs normalizer.
        actor_state: dict = {}
        for k, v in state_dict.items():
            k2 = k.removeprefix("actor_critic.")
            if k2.startswith(("actor.", "actor_obs_normalizer.")):
                actor_state[k2] = v

        missing, unexpected = self._actor.load_state_dict(actor_state, strict=False)
        self.get_logger().info(f"Load result — missing: {missing}, unexpected: {unexpected}")

        self._actor.eval()
        self._actor.to(self.device)

        self._last_action = np.zeros(_ACT_DIM, dtype=np.float32)
        self.get_logger().info(
            f"RLPolicy ready on {self.device} | "
            f"vel_scale={_ISAAC_VEL_SCALE * _SPEED_FACTOR:.3f} m/s per unit action"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_tf(self, target: str, source: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(target, source, Time())
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for TF {source} → {target} "
                        "(need ground_truth:=true)"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(f"TF {source} → {target} not available after {timeout_sec}s")
        return False

    def _lookup_pos_quat(self, source: str, target: str = "base_link"):
        """Return (pos [3], quat_wxyz [4]) of *source* frame in *target* frame."""
        tf = self._parent_node._tf_buffer.lookup_transform(target, source, Time())
        t = tf.transform.translation
        r = tf.transform.rotation
        pos = np.array([t.x, t.y, t.z], dtype=np.float32)
        quat = np.array([r.w, r.x, r.y, r.z], dtype=np.float32)
        return pos, quat

    def _arm_indices_in_js(self, joint_names: list) -> list:
        """Indices of the 6 arm joints within Gazebo's joint_states.name list."""
        return [joint_names.index(n) for n in _ARM_JOINT_NAMES]

    def _build_obs(
        self,
        obs_msg: Observation,
        plug_pos: np.ndarray,
        plug_quat_wxyz: np.ndarray,
        entrance_pos: np.ndarray,
    ) -> torch.Tensor:
        """Build obs vector matching Isaac Lab training order.

        The training env included ALL robot joints (arm + cable) in joint_pos_rel and
        joint_vel, giving n_joints dims each. Cable joints are not observable in Gazebo
        so they are set to 0 (their training default relative value).

        Order: joint_pos_rel(n_joints) | joint_vel(n_joints) | eef_pose(7) | port_rel(3) | last_action(6)
        """
        js = obs_msg.joint_states
        js_names = list(js.name)
        try:
            arm_idxs_in_js = self._arm_indices_in_js(js_names)
            arm_pos = np.array([js.position[i] for i in arm_idxs_in_js], dtype=np.float32)
            arm_vel = np.array([js.velocity[i] for i in arm_idxs_in_js], dtype=np.float32)
        except ValueError:
            arm_pos = np.array(js.position[:6], dtype=np.float32)
            arm_vel = np.array(js.velocity[:6], dtype=np.float32)

        # Build full joint arrays; cable joints default to 0 (matching training default_rel=0).
        full_pos_rel = np.zeros(self._n_joints, dtype=np.float32)
        full_vel = np.zeros(self._n_joints, dtype=np.float32)
        for i, arm_idx in enumerate(_ARM_JOINT_IDXS):
            full_pos_rel[arm_idx] = arm_pos[i] - _ARM_JOINT_DEFAULT[i]
            full_vel[arm_idx] = arm_vel[i]

        # eef_pose: sfp_tip_link world pos + quat wxyz (7 dims)
        eef_pose = np.concatenate([plug_pos, plug_quat_wxyz])

        # port_rel: entrance_pos − sfp_tip_pos (3 dims)
        port_rel = entrance_pos - plug_pos

        obs = np.concatenate([
            full_pos_rel,       # n_joints
            full_vel,           # n_joints
            eef_pose,           # 7
            port_rel,           # 3
            self._last_action,  # 6
        ])

        return torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

    def _make_motion_update(self, twist: Twist) -> MotionUpdate:
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.velocity = twist
        msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten().tolist()
        msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten().tolist()
        msg.feedforward_wrench_at_tip = Wrench()
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.get_logger().info(f"RLPolicy.insert_cable() task={task}")
        send_feedback("RLPolicy: waiting for TF")

        # TF frame names (matches CheatCode convention, requires ground_truth:=true)
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        self.get_logger().info(
            f"Port frame: {port_frame} | Cable tip frame: {cable_tip_frame}"
        )

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                self.get_logger().error(f"Could not get TF for {frame}, aborting")
                return False

        self._last_action[:] = 0.0
        dt = 1.0 / _CONTROL_HZ
        vel_scale = _ISAAC_VEL_SCALE * _SPEED_FACTOR
        start_time = time.time()
        send_feedback("RLPolicy: running")

        while time.time() - start_time < 30.0:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.sleep_for(dt)
                continue

            try:
                # Use base_link frame for ALL obs: Isaac Lab world frame is aligned with
                # base_link (robot base has identity rotation). Gazebo "world" frame has X
                # and Z inverted relative to base_link, so using world frame would flip the
                # port_rel direction and break policy inference.
                plug_pos_b, plug_quat_b = self._lookup_pos_quat(cable_tip_frame)
                entrance_pos_b, q_port_wxyz = self._lookup_pos_quat(port_frame)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed: {ex}")
                self.sleep_for(dt)
                continue

            obs_tensor = self._build_obs(obs_msg, plug_pos_b, plug_quat_b, entrance_pos_b)

            with torch.inference_mode():
                action = self._actor(obs_tensor)[0].cpu().numpy()

            # Clip: policy learned to use large actions in sim (physics clamps naturally).
            action = np.clip(action, -1.0, 1.0)
            self._last_action = action.copy()

            # Target 5cm past the port face: port_link sits at the entrance; full SFP/SC
            # insertion seats the connector ~4-5cm deeper along −Z (vertical insertion axis).
            # Obs still use actual entrance_pos_b; only the velocity target is shifted.
            insertion_target_b = entrance_pos_b.copy()
            insertion_target_b[2] -= 0.06
            port_rel_b = insertion_target_b - plug_pos_b

            # Full P controller — caps at ≤0.05 m/s prevent tracking-error resets.
            k_p = 1.5
            k_p_z = 2.0
            vel_x = float(np.clip(port_rel_b[0] * k_p, -0.06, 0.06))
            vel_y = float(np.clip(port_rel_b[1] * k_p, -0.06, 0.06))
            # Constant bias ensures the arm keeps pushing past the target depth
            # rather than decelerating to zero at port_link face (P controller alone
            # stalls when connector friction > commanded force at target).
            # Equilibrium shifts to port_rel_b[2] = 0.015/2.0 = 0.75 cm past target.
            vel_z = float(np.clip(port_rel_b[2] * k_p_z, -0.05, 0.05))

            # Angular correction: rotate EE so plug frame aligns with port frame.
            # q_diff = q_port * q_plug_inv — same formula as CheatCode.
            # rotvec in base_link gives the angular velocity direction.
            q_diff = _quat_mul(q_port_wxyz, _quat_conj(plug_quat_b))
            rotvec = _quat_to_rotvec(q_diff)
            omega = np.clip(rotvec * 0.5, -0.15, 0.15)

            twist = Twist(
                linear=Vector3(x=vel_x, y=vel_y, z=vel_z),
                angular=Vector3(x=float(omega[0]), y=float(omega[1]), z=float(omega[2])),
            )
            move_robot(motion_update=self._make_motion_update(twist))

            dist = float(np.linalg.norm(entrance_pos_b - plug_pos_b))
            self.get_logger().info(
                f"dist={dist*100:.1f}cm  port_rel={port_rel_b.round(3)}  "
                f"eef_b={plug_pos_b.round(3)}  "
                f"vel=[{vel_x:.3f},{vel_y:.3f},{vel_z:.3f}]"
            )

            elapsed = time.time() - loop_start
            time.sleep(max(0, dt - elapsed))

        self.get_logger().info("RLPolicy.insert_cable() done")
        return True
