"""Print robot joint names and their default positions to map Isaac Lab → Gazebo obs indices."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from aic_task.tasks.manager_based.aic_task.aic_task_env_cfg import AICTaskEnvCfg  # noqa: E402

cfg = AICTaskEnvCfg()
cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()

robot = env.scene["robot"]
names = robot.joint_names
defaults = robot.data.default_joint_pos[0].tolist()

print(f"\nTotal joints: {len(names)}")
print(f"\n{'idx':>4}  {'joint_name':<45}  default_pos")
print("-" * 70)
for i, (n, d) in enumerate(zip(names, defaults)):
    marker = " <<<" if any(arm in n for arm in [
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3"
    ]) else ""
    print(f"{i:>4}  {n:<45}  {d:+.6f}{marker}")

env.close()
simulation_app.close()
