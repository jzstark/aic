"""Print body names and USD prim hierarchy to identify EE and port frames."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect body names and USD prims in AIC task scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from aic_task.tasks.manager_based.aic_task.aic_task_env_cfg import AICTaskEnvCfg  # noqa: E402


def world_pos(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    x = UsdGeom.Xformable(prim)
    if not x:
        return None
    t = x.ComputeLocalToWorldTransform(0).ExtractTranslation()
    return [round(t[0], 5), round(t[1], 5), round(t[2], 5)]


def dump_prims(stage, path, depth=0, max_depth=5):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    pos = world_pos(stage, path)
    pos_str = f"  {pos}" if pos else ""
    print(f"{'  ' * depth}[{prim.GetTypeName():12s}] {prim.GetName()}{pos_str}")
    if depth < max_depth:
        for child in prim.GetChildren():
            dump_prims(stage, str(child.GetPath()), depth + 1, max_depth)


def main():
    cfg = AICTaskEnvCfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    robot = env.scene["robot"]
    nic = env.scene["nic_card"]
    stage = omni.usd.get_context().get_stage()

    # ------------------------------------------------------------------ robot
    print("\n" + "=" * 70)
    print("KEY ROBOT BODIES")
    for name in ["wrist_3_link", "gripper_tcp", "sfp_tip_link",
                 "sfp_module_link", "lc_plug_link", "sc_tip_link"]:
        if name in robot.body_names:
            idx = robot.body_names.index(name)
            pos = robot.data.body_pos_w[0, idx].tolist()
            print(f"  {name:35s}  {[f'{v:.5f}' for v in pos]}")

    # ------------------------------------------------------------------ nic card root
    print("\n" + "=" * 70)
    print(f"NIC_CARD root_pos_w: {nic.data.root_pos_w[0].tolist()}")

    # ------------------------------------------------------------------ NIC card USD hierarchy
    print("\n" + "=" * 70)
    print("NIC_CARD USD prim hierarchy (env 0):")
    dump_prims(stage, "/World/envs/env_0/nic_card", max_depth=5)

    # ------------------------------------------------------------------ SC port (for reference)
    print("\n" + "=" * 70)
    print("SC_PORT USD prim hierarchy (env 0, for comparison):")
    dump_prims(stage, "/World/envs/env_0/sc_port", max_depth=5)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
