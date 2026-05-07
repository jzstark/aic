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

# Default joint positions from ArticulationCfg.InitialStateCfg in aic_task_env_cfg.py.
# joint_pos_rel = actual_pos - default_pos, so we subtract these here.
_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_JOINT_DEFAULT = np.array(
    [0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110], dtype=np.float32
)

# Network architecture from rsl_rl_ppo_cfg.py
_OBS_DIM = 28
_ACT_DIM = 6
_HIDDEN = [512, 256, 128]

# sim.dt=1/120, decimation=4 → control at 30 Hz; DifferentialIK scale=0.05.
# velocity = action × 0.05 × 30 Hz = action × 1.5 m/s (exact equivalence).
# SPEED_FACTOR < 1.0 adds a safety margin; tune empirically in Gazebo.
_ISAAC_VEL_SCALE = 1.5   # m/s (or rad/s) per unit action
_SPEED_FACTOR = 0.4       # conservative starting point; raise if too slow
_CONTROL_HZ = 20.0


# ---------------------------------------------------------------------------
# Actor network (mirrors RSL-RL EmpiricalNormalization + MLP)
# ---------------------------------------------------------------------------

class _EmpiricalNorm(nn.Module):
    """Running mean/var normalizer matching RSL-RL's EmpiricalNormalization."""

    def __init__(self, shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("running_count", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)


class _Actor(nn.Module):
    """Actor MLP with obs normalisation, matching Isaac Lab RSL-RL PPO checkpoint."""

    def __init__(self, obs_dim: int, hidden: list, act_dim: int):
        super().__init__()
        self.obs_normalizer = _EmpiricalNorm(obs_dim)
        layers: list = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.obs_normalizer(obs))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RLPolicy(Policy):
    """RL policy that loads an RSL-RL PPO checkpoint and runs inference in Gazebo."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build actor
        self._actor = _Actor(_OBS_DIM, _HIDDEN, _ACT_DIM)

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

        self.get_logger().info(f"Checkpoint keys: {list(state_dict.keys())[:20]}")

        # Build a mapping: strip any 'actor_critic.' prefix, keep actor & normalizer keys.
        actor_state: dict = {}
        for k, v in state_dict.items():
            k2 = k.removeprefix("actor_critic.")
            if k2.startswith(("actor.", "actor_mean.", "obs_normalizer.")):
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

    def _joint_indices(self, joint_names: list) -> list:
        """Map UR5e joint names to indices in joint_states.name list."""
        return [joint_names.index(n) for n in _JOINT_NAMES]

    def _build_obs(
        self,
        obs_msg: Observation,
        plug_pos: np.ndarray,
        plug_quat_wxyz: np.ndarray,
        entrance_pos: np.ndarray,
    ) -> torch.Tensor:
        """Build 28-dim observation matching Isaac Lab training order.

        Order: joint_pos(6) | joint_vel(6) | eef_pose(7) | port_rel(3) | last_action(6)
        """
        js = obs_msg.joint_states
        js_names = list(js.name)
        try:
            idxs = self._joint_indices(js_names)
            joint_pos = np.array([js.position[i] for i in idxs], dtype=np.float32)
            joint_vel = np.array([js.velocity[i] for i in idxs], dtype=np.float32)
        except ValueError:
            # Fallback: assume first 6 joints are the arm
            joint_pos = np.array(js.position[:6], dtype=np.float32)
            joint_vel = np.array(js.velocity[:6], dtype=np.float32)

        # joint_pos_rel = actual − default (matching Isaac Lab's joint_pos_rel)
        joint_pos_rel = joint_pos - _JOINT_DEFAULT

        # eef_pose: sfp_tip_link world pos + quat (7 dims)
        eef_pose = np.concatenate([plug_pos, plug_quat_wxyz])

        # port_rel: entrance_pos − sfp_tip_pos (3 dims)
        port_rel = entrance_pos - plug_pos

        obs = np.concatenate([
            joint_pos_rel,      # 6
            joint_vel,          # 6
            eef_pose,           # 7
            port_rel,           # 3
            self._last_action,  # 6
        ])  # = 28

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
                plug_pos, plug_quat = self._lookup_pos_quat(cable_tip_frame)
                entrance_pos, _ = self._lookup_pos_quat(port_frame)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed: {ex}")
                self.sleep_for(dt)
                continue

            obs_tensor = self._build_obs(obs_msg, plug_pos, plug_quat, entrance_pos)

            with torch.inference_mode():
                action = self._actor(obs_tensor)[0].cpu().numpy()

            self._last_action = action.copy()

            # Convert delta-pose action to Cartesian velocity (base_link frame)
            twist = Twist(
                linear=Vector3(
                    x=float(action[0] * vel_scale),
                    y=float(action[1] * vel_scale),
                    z=float(action[2] * vel_scale),
                ),
                angular=Vector3(
                    x=float(action[3] * vel_scale),
                    y=float(action[4] * vel_scale),
                    z=float(action[5] * vel_scale),
                ),
            )
            move_robot(motion_update=self._make_motion_update(twist))

            dist = float(np.linalg.norm(entrance_pos - plug_pos))
            self.get_logger().info(
                f"dist={dist*100:.1f}cm  action={action.round(3)}"
            )

            elapsed = time.time() - loop_start
            time.sleep(max(0, dt - elapsed))

        self.get_logger().info("RLPolicy.insert_cable() done")
        return True
