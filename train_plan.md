# AIC Isaac Lab RL 训练完整指南

> 从零开始，用 Isaac Lab PPO 训练一个能插入线缆的策略，部署到 Gazebo 评测环境并提交 v2。
> 截止日期：**2026-05-15**。

---

## 为什么用 RL 而非录制数据（IL）

评测有多维随机化（NIC 在 5 条 rail 任意一条、位置 ±2cm、任务板朝向随机、抓握偏差 ±2mm），覆盖全部变体需要 500+ 个 demo。RL 在训练时直接随机化这些参数，自然泛化。

**RL 核心逻辑**：机械臂做动作 → 获得奖励（离插口越近越高）→ PPO 算法更新网络 → 反复迭代直到策略收敛。

**整体流程**：
```
Isaac Lab（Isaac Sim 物理引擎）PPO 训练
    ↓ checkpoint（.pt 文件，PyTorch 神经网络）
封装成 AIC Policy 类（RLPolicy.py）
    ↓ 打包进 Docker 镜像
Gazebo 评测环境运行，提交 v2
```

---

## Phase 1：环境配置（Day 1，约 3 小时）

### 1.1 确认使用 Docker Engine（不是 Docker Desktop）

Isaac Lab 需要系统级 Docker Engine 的 NVIDIA 运行时。之前安装 nvidia-container-toolkit 时写入了
`/etc/docker/daemon.json`，需要切换到系统 Docker：

```bash
# 检查当前激活的 Docker
docker context ls
# 如果 active 是 desktop-linux，切换：
docker context use default

# 验证 NVIDIA runtime
docker info | grep -i nvidia
# 应该看到：Runtimes: nvidia runc
```

如果 nvidia runtime 不在，重新配置：
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i nvidia
```

### 1.2 克隆 IsaacLab v2.3.2

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd ~/IsaacLab
git checkout v2.3.2   # AIC 集成测试的版本
```

### 1.3 将 aic 代码放入 IsaacLab

在宿主机建立符号链接（方便路径引用）：
```bash
cd ~/IsaacLab
ln -s ~/Code/ws_aic/src/aic ./aic
ls aic/aic_utils/aic_isaac/  # 应看到 README.md 和 aic_isaaclab/
```

> **⚠️ 已知问题：Docker 不跟随指向容器挂载点之外的 symlink。**
> `~/IsaacLab/aic` 指向 `~/Code/ws_aic/src/aic`，而容器只挂载了 `~/IsaacLab`，
> 所以容器内 `aic/` 是死链接，任何 `pip install` 或脚本都会报路径不存在。

**解决方案：在 `docker-compose.yaml` 中添加 bind mount，直接挂载真实路径。**

编辑 `~/IsaacLab/docker/docker-compose.yaml`，在 `x-default-isaac-lab-volumes` 末尾追加：

```yaml
  - type: bind
    source: /home/stark/Code/ws_aic/src/aic
    target: /workspace/isaaclab/aic
```

（紧接在 `.isaac-lab-docker-history` 那行 bind mount 之后）

改完后需要重启容器才生效：
```bash
./docker/container.py stop base
./docker/container.py start base
./docker/container.py enter base
```

### 1.4 下载 NVIDIA 资产包

前往下载：`https://developer.nvidia.com/downloads/Omniverse/learning/Events/Hackathons/Intrinsic_assets.zip`

解压后将 `Intrinsic_assets/` 目录放到：
```
~/IsaacLab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/Intrinsic_assets/
```

验证：
```bash
ls .../aic_task/Intrinsic_assets/
# 应看到：aic_unified_robot_cable_sdf.usd  assets/  scene/
```

### 1.5 构建 Isaac Lab Docker 镜像（约 30-60 分钟）

```bash
cd ~/IsaacLab
./docker/container.py build base
# 看到 Successfully tagged isaac-lab-base:latest 则成功
```

构建过程中会询问是否启用 X11 forwarding：
```
Would you like to enable it? (y/N)
```
回答 **y**（本机有显示器，Phase 2 的 teleop 可视化需要它）。

### 1.6 启动容器并安装 Isaac Lab 包和 aic_task

```bash
cd ~/IsaacLab
./docker/container.py start base
./docker/container.py enter base   # 进入容器，下面都在容器内运行
```

> **⚠️ 已知问题：`isaaclab.sh --install` 和 `python -m pip install` 不能用于此容器。**
>
> - `isaaclab.sh --install` 会遇到大量依赖冲突（`psutil`、`rl-games` 等），
>   且最终 `pip show isaaclab` 仍返回 "not found"。
> - 裸 `python -m pip install` 使用的是系统 Python，不是 Isaac Sim 的 Python
>   (`_isaac_sim/python.sh`)，安装到了错误的环境。
> - 直接用 Isaac Sim Python 的 pip（`_isaac_sim/python.sh -m pip install`）时，
>   会报 `ModuleNotFoundError: No module named 'pkg_resources'`，
>   原因是 pip 的 build isolation 创建的临时环境缺少 setuptools。
>
> **解决方案：用 Isaac Sim Python 直接安装，并加 `--no-build-isolation` 跳过隔离环境。**

容器内执行（安装 Isaac Lab 全部包 + aic_task）：

```bash
# 安装 Isaac Lab 所有扩展包
for dir in \
    /workspace/isaaclab/source/isaaclab \
    /workspace/isaaclab/source/isaaclab_assets \
    /workspace/isaaclab/source/isaaclab_mimic \
    /workspace/isaaclab/source/isaaclab_rl \
    /workspace/isaaclab/source/isaaclab_tasks; do
    echo "Installing: $dir"
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-build-isolation --quiet -e "$dir"
done

# 安装 aic_task
/workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-build-isolation -e \
    /workspace/isaaclab/aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task
```

验证：
```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip show isaaclab
# 应显示 Name: isaaclab, Location: .../source/isaaclab

isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/list_envs.py
# 应看到 AIC-Task-v0 在列表中
```

> **注意：** `docker compose down` 会销毁容器的 overlay 文件系统，上面的安装会丢失。
> 每次执行 `container.py stop base` + `start base` 后需要重新运行安装命令。
> 若觉得麻烦，可以将安装命令写入 `~/IsaacLab/docker/Dockerfile.base` 的末尾，
> 重新 `build base` 一次，之后就永久有效。

---

## Phase 2：理解现有环境（Day 1 下午，约 1 小时）

### 2.1 运行随机策略（验证仿真可以跑）

```bash
# 容器内 — headless + enable_cameras（AIC-Task-v0 有相机配置，即使 headless 也必须加此 flag）
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/random_agent.py \
    --task AIC-Task-v0 --num_envs 1 --headless --enable_cameras
```
几秒后打印 episode stats 并退出，没有崩溃即表示物理仿真、资产加载、观测管道全部正常。

> **注意**：`--headless` 只是不弹出 GUI 窗口；相机渲染仍在 GPU 上离屏进行，
> 因此即使 headless 模式也必须加 `--enable_cameras`，否则报：
> `RuntimeError: A camera was spawned without the --enable_cameras flag`
>
> 另一个 WARNING `Not all actuators are configured! 6 != 46` 属于正常现象：
> 机器人 USD 有 46 个关节（含被动关节、线缆/夹爪附件），
> 实际被控制的只有 6 个臂关节，可忽略。

### 2.2 键盘遥操作（熟悉空间布局）

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
    --task AIC-Task-v0 --num_envs 1 --teleop_device keyboard --enable_cameras
```
用 W/S/A/D/Q/E 键控制机械臂，观察 NIC 卡和 SC 端口的位置。

### 2.3 关键认识：当前奖励函数训练的是什么

`aic_task_env_cfg.py` 中 `CommandsCfg` 使用 `UniformPoseCommandCfg`，每次 episode 随机生成一个
目标位姿。奖励是 EE 离这个**随机目标**越近越好。

**这不是插入任务**，需要在 Phase 3 修改成朝真实端口方向引导。

---

## Phase 3：奖励函数修改（Day 2，约 3 小时）

### 3.1 需要添加的奖励

| 奖励项 | 类型 | weight | 目的 |
|--------|------|--------|------|
| `dist_to_port`（L2 距离） | 密集负奖励 | -0.5 | 引导 EE 向 NIC 卡移动 |
| `dist_to_port_tanh` | 密集正奖励 | +2.0 | 接近时给强梯度信号 |
| `insertion_success_bonus`（距离 < 1cm） | 稀疏大正奖励 | +10.0 | 奖励实际完成插入 |
| 平滑惩罚（action_rate, joint_acc, joint_torques） | 惩罚 | 保持现有 | 避免抖动/大力 |

### 3.2 修改 rewards.py

文件：`aic/aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py`

在文件末尾追加三个函数：

```python
def dist_to_port(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    ee_body_name: str = "wrist_3_link",
) -> torch.Tensor:
    """L2 distance from EE to port (use with negative weight)."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :]
    port_pos = port.data.root_pos_w[:, :3]
    return torch.norm(ee_pos - port_pos, dim=1)


def dist_to_port_tanh(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    std: float = 0.05,
    ee_body_name: str = "wrist_3_link",
) -> torch.Tensor:
    """Tanh reward peaking at 1.0 when EE is on the port."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :]
    port_pos = port.data.root_pos_w[:, :3]
    dist = torch.norm(ee_pos - port_pos, dim=1)
    return 1.0 - torch.tanh(dist / std)


def insertion_success_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    threshold: float = 0.01,
    ee_body_name: str = "wrist_3_link",
) -> torch.Tensor:
    """Sparse +1 when EE is within threshold meters of port."""
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :]
    port_pos = port.data.root_pos_w[:, :3]
    dist = torch.norm(ee_pos - port_pos, dim=1)
    return (dist < threshold).float()
```

在 `mdp/__init__.py` 末尾添加导出：
```python
from .rewards import dist_to_port, dist_to_port_tanh, insertion_success_bonus
```

### 3.3 修改 aic_task_env_cfg.py 的 RewardsCfg

删除原有的 `end_effector_position_tracking*`、`end_effector_orientation_tracking*`、`reaching_bonus`，
替换为（保留平滑惩罚项不变）：

```python
@configclass
class RewardsCfg:
    dist_to_sfp = RewTerm(
        func=mdp.dist_to_port, weight=-0.5,
        params={"robot_cfg": SceneEntityCfg("robot"),
                "port_cfg": SceneEntityCfg("nic_card"),
                "ee_body_name": "wrist_3_link"},
    )
    dist_to_sfp_tanh = RewTerm(
        func=mdp.dist_to_port_tanh, weight=2.0,
        params={"robot_cfg": SceneEntityCfg("robot"),
                "port_cfg": SceneEntityCfg("nic_card"),
                "std": 0.05, "ee_body_name": "wrist_3_link"},
    )
    insertion_success = RewTerm(
        func=mdp.insertion_success_bonus, weight=10.0,
        params={"robot_cfg": SceneEntityCfg("robot"),
                "port_cfg": SceneEntityCfg("nic_card"),
                "threshold": 0.01, "ee_body_name": "wrist_3_link"},
    )
    # 平滑惩罚（保持不变）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel   = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    joint_acc   = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7,
                          params={"asset_cfg": SceneEntityCfg("robot")})
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6,
                            params={"asset_cfg": SceneEntityCfg("robot")})
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1,
                               params={"asset_cfg": SceneEntityCfg("robot")})
```

### 3.4 修改观测：加入端口相对位置，去掉相机

相机观测（ResNet18 特征）大幅拖慢训练速度，初次训练去掉。
策略需要知道端口在哪里 → 用地面真值位置（训练时可用）。

在 `observations.py` 末尾添加：
```python
def port_relative_to_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Port world pos minus EE world pos. Shape: (N, 3)."""
    from isaaclab.assets import Articulation, RigidObject
    robot: Articulation = env.scene[robot_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]
    ee_ids = robot_cfg.body_ids
    ee_pos = robot.data.body_pos_w[:, ee_ids[0], :]
    port_pos = port.data.root_pos_w[:, :3]
    return port_pos - ee_pos
```

在 `mdp/__init__.py` 添加：
```python
from .observations import port_relative_to_ee
```

在 `aic_task_env_cfg.py` 的 `PolicyCfg` 中：
- **删除** `center_rgb`, `left_rgb`, `right_rgb`
- **添加**：
```python
port_rel = ObsTerm(
    func=mdp.port_relative_to_ee,
    params={"robot_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "port_cfg": SceneEntityCfg("nic_card")},
)
```

修改后观测维度：
```
joint_pos(6) + joint_vel(6) + eef_pose(7) + port_rel(3) + last_action(6) = 28 维
```

---

## Phase 4：PPO 训练（Day 2-5）

### 4.1 第一次训练（100 次迭代，验证代码无误）

```bash
# 容器内
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
    --task AIC-Task-v0 \
    --num_envs 4 \
    --headless \
    --max_iterations 100
```

**看什么**：
- 没有报错 → 奖励/观测函数代码正确
- `mean_reward` 在前 50 次有上升趋势（哪怕从负值上升）→ 策略在学习
- 日志在：`logs/rsl_rl/aic_task/<timestamp>/`

### 4.2 正式训练（过夜，建议用 tmux）

```bash
tmux new -s rl_train

isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
    --task AIC-Task-v0 \
    --num_envs 16 \
    --headless \
    --max_iterations 1500

# 离开 tmux：Ctrl+B 然后 D
# 重新进入：tmux attach -t rl_train
```

- `--num_envs 16`：16 个并行仿真，数据吞吐量 16 倍
- 每 50 次迭代自动保存 checkpoint：`logs/rsl_rl/aic_task/<timestamp>/model_<iter>.pt`
- 时间预估：RTX 高端约 2-4 小时

### 4.3 监控训练日志

```
Iteration 200/1500
  mean_reward:    3.42    ← 应随时间上升
  value_loss:     0.008   ← 应随时间下降
  surrogate_loss: -0.002  ← 应在 0 附近浮动
```

**判断训练是否健康**：
- `mean_reward` 稳定上升 → 正常
- 停滞 300+ 次 → 可以停止，或调整权重（`dist_to_sfp_tanh` weight 从 2.0 → 5.0）
- 大幅震荡 → 降低所有奖励权重绝对值

### 4.4 可视化验证

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py \
    --task AIC-Task-v0 \
    --num_envs 4 \
    --enable_cameras \
    --load_run <timestamp>
```
Isaac Sim 窗口显示策略运行。确认机械臂能持续向 NIC 卡方向移动。

---

## Phase 5：部署到 Gazebo（Day 6-7）

### 5.1 从容器拷出 checkpoint

```bash
# 宿主机执行
mkdir -p ~/ws_aic/src/aic/my_policy_node/my_policy_node/checkpoints/
docker cp isaac-lab-base:/root/IsaacLab/logs/rsl_rl/aic_task/<timestamp>/model_1500.pt \
    ~/ws_aic/src/aic/my_policy_node/my_policy_node/checkpoints/model_best.pt
```

### 5.2 新建 RLPolicy.py

路径：`~/ws_aic/src/aic/my_policy_node/my_policy_node/RLPolicy.py`

**Actor 网络结构**（来自 `rsl_rl_ppo_cfg.py`，与训练时完全一致）：
```python
actor = torch.nn.Sequential(
    torch.nn.Linear(28, 512), torch.nn.ELU(),
    torch.nn.Linear(512, 256), torch.nn.ELU(),
    torch.nn.Linear(256, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 6),
)
```

**观测向量组装顺序**（必须与 Isaac Lab 训练时完全一致）：
```
[joint_pos(6), joint_vel(6), eef_pose(7), port_relative_pos(3), last_action(6)]
```

**Checkpoint 加载**（只取 actor 部分，去掉 critic）：
```python
checkpoint = torch.load("checkpoints/model_best.pt", map_location=device)
actor_state = {k.replace("actor.", ""): v
               for k, v in checkpoint["model_state_dict"].items()
               if k.startswith("actor.")}
actor.load_state_dict(actor_state)
actor.eval()
```

**TF 获取端口位置**（需要 `ground_truth:=true`）：
```python
transform = tf_buffer.lookup_transform("base_link", port_frame, rclpy.time.Time())
port_pos = [transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z]
```

**动作转换**（Isaac Lab 输出相对 delta，转成 Gazebo 速度命令）：
- 线速度缩放系数约 0.05-0.1（需实验调整）
- 角速度缩放系数约 0.02-0.05
- 使用 `MotionUpdate.trajectory_generation_mode = MODE_VELOCITY`，`frame_id = "gripper/tcp"`

**确认 Observation.msg 字段**（在写代码前先确认）：
```bash
cat ~/ws_aic/src/aic/aic_interfaces/aic_model_interfaces/msg/Observation.msg
```

### 5.3 在 Gazebo 中测试

```bash
# 终端 1（容器）
distrobox enter -r aic_eval
/entrypoint.sh ground_truth:=true start_aic_engine:=true

# 终端 2（宿主机）
cd ~/ws_aic/src/aic
pixi reinstall ros-kilted-my-policy-node
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p policy:=my_policy_node.RLPolicy
```

### 5.4 调试 sim-to-sim gap

| 症状 | 原因 | 对策 |
|------|------|------|
| 机械臂往错误方向走 | 坐标系差异（Isaac vs Gazebo） | 打印 port_pos_rel，对比两个仿真中的值 |
| 动作幅度太大/小 | 速度缩放系数不匹配 | 从小值（0.02）开始逐渐增大 |
| 到附近但不能插入 | EE ≠ 插头尖端，缺末端偏移 | 用 sample_config.yaml 中的 gripper_offset 值补偿 |
| 效果不稳定 | 训练时噪声不够 | 在 Isaac Lab 增大观测噪声范围后重新训练 |

---

## Phase 6：提交 v2（Day 10）

1. 更新 `docker/aic_model/Dockerfile`，在 `COPY` 段添加：
   ```dockerfile
   COPY my_policy_node/my_policy_node/checkpoints /ws_aic/src/aic/my_policy_node/my_policy_node/checkpoints
   ```
   更新 `CMD`：
   ```dockerfile
   CMD ["--ros-args", "-p", "policy:=my_policy_node.RLPolicy", "-p", "use_sim_time:=true"]
   ```

2. 构建并推送：
   ```bash
   cd ~/ws_aic/src/aic
   docker compose -f docker/docker-compose.yaml build model

   # 刷新 ECR token（12 小时有效期）
   aws ecr get-login-password --region us-east-1 | \
       docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

   docker tag my-solution:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team>:v2
   docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team>:v2
   ```

---

## 时间线

| 天 | 任务 | 产出 |
|----|------|------|
| Day 1 上午 | 切换 Docker Engine，克隆 IsaacLab，下载资产 | 文件就位 |
| Day 1 下午 | 构建镜像（等待），teleop 验证场景 | Isaac Sim 可运行 |
| Day 2 | 修改 rewards.py、observations.py、env_cfg.py | 奖励指向真实端口 |
| Day 2 晚 | 100 次迭代验证无报错 | 代码通过验证 |
| Day 3-4 | 1500 次迭代完整训练（过夜） | checkpoint 文件 |
| Day 5 | play.py 可视化，评估质量 | 策略能向端口移动 |
| Day 6 | 写 RLPolicy.py，确认 Observation msg 字段 | 部署代码完成 |
| Day 7-8 | Gazebo 测试，调整缩放系数 | 机械臂能移向目标 |
| Day 9 | 迭代调试 sim-to-sim gap | 基本插入成功 |
| Day 10 | 更新 Dockerfile，构建并推送 v2 | **提交新版本** |

---

## 关键文件索引

| 文件 | 操作 |
|------|------|
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/rewards.py` | 追加 3 个新奖励函数 |
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/observations.py` | 追加 `port_relative_to_ee` |
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/mdp/__init__.py` | 导出新函数 |
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/aic_task_env_cfg.py` | 替换 RewardsCfg，修改 ObservationsCfg（去相机，加 port_rel） |
| `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/agents/rsl_rl_ppo_cfg.py` | 基本不变（网络 [512,256,128]，1500 迭代） |
| `my_policy_node/my_policy_node/RLPolicy.py` | 新建：加载 checkpoint，包装成 AIC Policy |
| `my_policy_node/my_policy_node/checkpoints/model_best.pt` | 从 Isaac Lab 容器拷出的 checkpoint |
| `docker/aic_model/Dockerfile` | 更新 CMD 和 COPY checkpoints |
