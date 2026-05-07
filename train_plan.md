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

### 3.0 补充随机化：task board YAW（✅ 已实现）

**背景**：Phase 2 分析发现 `randomize_board_and_parts` 只随机化了 task board 的 XY 位置（±5mm），**没有随机化 YAW**。但 AIC 评测中 trial 1/2 的 yaw≈3.14 rad，trial 3 的 yaw≈3.0 rad，朝向明显不同，不加 YAW 随机化则策略无法泛化到 trial 3。

**已完成的改动：**

**`events.py`** — 在模块顶部（`_sample_axis` 之前）新增辅助函数，并扩展 `randomize_board_and_parts` 函数：

```python
# 新增：wxyz 四元数 Hamilton 乘积
def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)
```

在 `randomize_board_and_parts` 的 XY 随机化之后，board 写入 sim 之前插入：

```python
# YAW 随机化（board_range 无 "yaw" 键时自动跳过，lo==hi 时不做任何事）
yaw_lo, yaw_hi = board_range.get("yaw", (0.0, 0.0))
if yaw_hi != yaw_lo:
    yaw_delta = torch.empty(n, device=device).uniform_(yaw_lo, yaw_hi)
    half = yaw_delta * 0.5
    zeros = torch.zeros(n, device=device)
    dq = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
    board_rot = _quat_mul(dq, board_rot)
```

Parts 的 XY offset 也随 board yaw 同步旋转（保证 NIC 卡等零件在 board 转动后仍在正确位置）：

```python
# 在每个 part 的 offset 计算中替换原来的逐 idx 累加写法
local_offsets = torch.zeros(n, 3, device=device)
for idx in range(n):
    local_offsets[idx, 0] = ox + _sample_axis(pr, snap, "x")
    local_offsets[idx, 1] = oy + _sample_axis(pr, snap, "y")
    local_offsets[idx, 2] = oz

if yaw_hi != yaw_lo:
    cos_y = torch.cos(yaw_delta)
    sin_y = torch.sin(yaw_delta)
    rx = cos_y * local_offsets[:, 0] - sin_y * local_offsets[:, 1]
    ry = sin_y * local_offsets[:, 0] + cos_y * local_offsets[:, 1]
    local_offsets[:, 0] = rx
    local_offsets[:, 1] = ry

part_pos = board_world_pos.clone() + local_offsets
```

**`aic_task_env_cfg.py`** — `board_range` 已加入 `"yaw"` 键：

```python
"board_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "yaw": (-0.15, 0.15)},
```

> **如需暂时禁用 YAW 随机化**（例如先跑 100 次迭代验证奖励代码）：
> 只需将 `aic_task_env_cfg.py` 里的 `"yaw": (-0.15, 0.15)` 删掉即可，
> `events.py` 不需要改动——`yaw` 键不存在时 `lo==hi==0.0`，整个随机化逻辑自动跳过。

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
    --num_envs 16 \
    --headless \
    --max_iterations 100
```

**看什么**：
- 没有报错 → 奖励/观测函数代码正确
- `mean_reward` 在前 50 次有上升趋势（哪怕从负值上升）→ 策略在学习
- 日志在：`logs/rsl_rl/aic_task/<timestamp>/`


你现在运行的是 PPO 强化学习，可以想象成这样一个场景：

16 个虚拟房间，每个房间里有一条机械臂和一块任务板。每个房间每 10 秒重置一次（任务板位置随机偏移 ±5mm，随机旋转 ±8.6°）。机械臂每 1/30 秒做一次动作，然后根据结果拿到"评分"。一个神经网络控制所有 16 条手臂，每 128 步统一学习一次。

每一"轮"发生了什么（一次 iteration）

┌────────────────────────────────────────────────────────────────┐
│  16 个环境 × 128 步 = 2048 条数据                                 
│                                                                 
│  每步：                                                          
│  1. 读取观测（28维）                                              
│     joint_pos(6) + joint_vel(6) + eef_pose(7)                  
│     + port_rel(3) + last_action(6)                             
│                                                                 
│  2. 神经网络输出动作（6维 = EE delta pos + delta rot）          
│                                                                 
│  3. DifferentialIK 把 delta-pose 转成关节速度                  
│                                                                 
│  4. 物理引擎运行 4 个 sim 步（1/120s × 4 = 1/30s）            
│                                                                 
│  5. 计算奖励：                                                  
│     dist_to_sfp     = −2 × ‖EE − NIC卡‖      ← 线性惩罚      
│     dist_to_sfp_exp = 5 × exp(−dist/0.15)    ← 指数奖励      
│     joint_acc       = 惩罚抖动                                  
└─────────────────────────┬──────────────────────────────────────┘
                          │ 2048 条 (观测, 动作, 奖励, 下一观测)
                          ▼
┌────────────────────────────────────────────────────────────────┐
│  PPO 更新（重复 8 次）                                         
│                                                                 
│  1. 价值函数估计每个状态的"期望总回报"                         
│  2. 计算优势（这次实际拿到的奖励 比 预期 好多少？）            
│  3. 梯度上升：调整网络权重，让好动作概率升高                   
│  4. Clip 约束：单次更新幅度不超过 20%（PPO 的关键机制）       
└────────────────────────────────────────────────────────────────┘



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

---

## Debug 踩坑记录（Phase 3/4 实际遇到的问题）

> 以下是实际训练过程中遇到的 7 个坑，每个坑附有根因分析和解决方案，供之后遇到类似问题时参考。

---

### 坑 1：相机报错，但问题不在 ObservationsCfg

**症状**：
```
RuntimeError: A camera was spawned without the --enable_cameras flag.
```
训练启动即崩溃，即使已经把所有相机 `ObsTerm` 从 `PolicyCfg` 里删掉了。

**根因**：
Isaac Lab 的 `TiledCameraCfg` 是在 `AICTaskSceneCfg.__post_init__` 里用 `self.center_rgb = TiledCameraCfg(...)` 等语句**创建并注册**到 Scene 的。只删掉 `ObservationsCfg` 里的 `ObsTerm` 不会阻止相机资产被创建——Scene 里仍然存在这三个相机对象，Isaac Sim 看到就会报错。

**解决方案**：
在 `aic_task_env_cfg.py` 的 `AICTaskSceneCfg.__post_init__` 里，把三行相机创建代码注释掉：
```python
# self.center_rgb = TiledCameraCfg(...)
# self.left_rgb = TiledCameraCfg(...)
# self.right_rgb = TiledCameraCfg(...)
```
**规律**：Isaac Lab 中"不用某个 Scene 资产"的正确做法是不创建该资产（注释掉 `__post_init__` 里的赋值语句），而不是只删 Obs/Reward term。

---

### 坑 2：奖励函数静默返回 0（`SceneEntityCfg.body_ids` 的陷阱）

**症状**：
训练日志里 `Episode_Reward/dist_to_sfp` 等所有自定义奖励项均为 0.0，`mean_reward` 只由平滑惩罚项驱动，固定在 -0.13 附近，没有任何梯度信号。

**根因**：
Isaac Lab 的 `RewardManager` 在调用每个奖励函数时会 `try/except` 捕获所有异常并静默返回 0。当奖励函数里写了 `asset_cfg.body_ids[0]`，而 `SceneEntityCfg` 的 `body_ids` 在没有经过 `resolve()` 时默认为 `slice(None)`，`slice(None)[0]` 会抛出 `TypeError`，被静默吞掉，函数返回 0。

**解决方案**：
不使用 `body_ids[0]`，改为用 `find_bodies` 直接查询，并在模块级别缓存索引：
```python
_ee_body_idx: dict[str, int] = {}

def _get_ee_pos(robot: Articulation, body_name: str) -> torch.Tensor:
    if body_name not in _ee_body_idx:
        idx, _ = robot.find_bodies(body_name)
        _ee_body_idx[body_name] = idx[0]
    return robot.data.body_pos_w[:, _ee_body_idx[body_name], :]
```
`observations.py` 里的 `port_relative_to_ee` 也同理，直接用 `robot.find_bodies(ee_body_name)[0][0]` 而不依赖 `body_ids`。

**规律**：在自定义奖励/观测函数中，**永远不要用 `asset_cfg.body_ids[0]`**；用 `articulation.find_bodies("body_name")[0][0]` 代替，加模块级缓存避免每步重复查找。

---

### 坑 3：`RewTerm.params` 里的 float 参数可能无效

**症状**：
把 `dist_to_port_tanh` 加入 `RewardsCfg` 时，在 `params` 里传了 `"std": 0.25`，但奖励的行为跟 `std=0.05`（函数默认值）几乎一样——`tanh(0.26/0.25)≈0.89` 应有明显梯度，实测奖励接近 0。

**根因**（推测）：
Isaac Lab 的 `RewTerm` 参数分发逻辑对 `SceneEntityCfg` 对象有特殊处理（resolve、传 env 等）。当 `params` 字典里同时存在 `SceneEntityCfg` 和普通 float 时，float 参数可能不被正确转发，函数拿到的是默认值而非配置值。此行为没有报错，极难调试。

**解决方案**：
将需要 float 参数的奖励函数改为"无外部 float 配置"版本——把参数硬编码在函数体内，或采用不需要配置参数的数学形式。具体：
- `dist_to_port_exp`（`exp(-dist/0.15)`）完全不依赖外部 float，替换 `dist_to_port_tanh`
- 若确实需要可调参数，可以用不同 `sigma` 值各写一个独立函数（冗余但可靠）

**规律**：**在 `RewTerm.params` 里，只可靠传递 `SceneEntityCfg` 和 `str` 类型参数；普通 float 参数不可信，改为硬编码在函数内。**

---

### 坑 4：前 40 次迭代奖励完全不变——观测归一化预热

**症状**：
训练前 40 次迭代，所有 `Episode_Reward/X` 固定不变，`episode_length` 只有 9 步，`mean_reward` 也固定。第 41 次之后突然跳变到另一组固定值，`episode_length` 跳到 510 步。整个过程没有任何梯度上升。

**根因**：
`rsl_rl_ppo_cfg.py` 中设置了 `actor_obs_normalization=True` 和 `critic_obs_normalization=True`。归一化模块在初始阶段的 running mean/var 全为 0，对输入的缩放极端不稳定。前 ~40 次迭代网络输出的动作是大噪声，导致机械臂在物理引擎里崩溃，每个 episode 只有 9 步就结束（终止条件被触发）。约 40 次迭代后 running stats 稳定，动作恢复正常，episode 长度回到正常范围。

**这是正常行为，不是 bug**。只要之后奖励有上升趋势，不需要处理。

**规律**：开启 obs normalization 后，前 40-50 次迭代的数据不可靠；看训练是否健康，要从第 50 次迭代之后的趋势判断。

---

### 坑 5：解读训练日志的两个奖励数字

训练日志中同时出现两种奖励数字，含义不同：

| 字段 | 含义 | 如何看 |
|------|------|--------|
| `mean_reward` | 整个 rollout buffer 里所有 transition 的总奖励均值（每 step 平均值 × 很多步） | 绝对值大，负数正常 |
| `Episode_Reward/X` | **已完成 episode** 中，term X 每步平均值（只统计本次 rollout 里结束的 episode） | 绝对值小，用来诊断各项 |

**如何换算**：
- `dist_to_sfp_exp` 理论值：`weight × exp(-dist/0.15)`，EE 在 0.15m 处 = `5 × exp(-1) ≈ 1.84`
- 如果 `Episode_Reward/dist_to_sfp_exp` 从 0.009 涨到 1.8，对应 EE 从 ~25cm 移动到 ~15cm，是真实的学习信号，正常

**规律**：`Episode_Reward/X` 是诊断各奖励项是否正确的主要工具；`mean_reward` 用于看整体趋势。

---

### 坑 6：`episode_length_s` 太长 + `num_steps_per_env` 太小，价值函数学不好

**症状**：
奖励上升极慢，早期价值函数误差大。

**根因**：
- 原始配置：`episode_length_s = 200s`（6000 步），`num_steps_per_env = 24`
- 每次 rollout 只覆盖 24 步，是整个 episode 的 0.4%
- 价值函数几乎看不到 episode 结束，bootstrap error 积累严重

**解决方案**：
- `episode_length_s = 10.0`（300 步，允许机械臂有足够时间接近目标）
- `num_steps_per_env = 128`（一次 rollout 覆盖 ~43% 的 episode）
- 这两个参数的比例大致保持 `num_steps_per_env / (episode_length × control_freq) ≥ 20%`

---

### 坑 7：tanh 奖励在训练初期无梯度（std 校准问题）

**症状**：
使用 `dist_to_port_tanh(std=0.05)` 时，在整个训练过程中奖励接近 0，策略无法从这个信号学习。

**根因**：
训练开始时 EE 距离 NIC 卡约 25cm（0.25m）。`tanh(0.25/0.05) = tanh(5) ≈ 0.9999`，奖励 = `1 - 0.9999 ≈ 0.0001`，梯度几乎为 0。`std` 参数需要**接近 EE 的初始距离**才有梯度，而不是接近"理想精度"。

**解决方案**：
- 训练初期用 `dist_to_port_exp`（`exp(-dist/0.15)`）：dist=0.25m 时值 = `exp(-1.67)≈0.19`，有实际梯度
- 若用 tanh，`std` 应设为初始 EE-port 距离的 1/3 左右（~0.08-0.1m），而非最终精度（0.005-0.01m）
- 课程学习思路：先大 std，接近后再换小 std（需要手动切换或课程调度）

---

### 快速诊断 checklist

当奖励全为 0 或不变时，按顺序排查：

1. **相机报错？** → 检查 `__post_init__` 里是否有 `TiledCameraCfg` 创建语句
2. **Episode_Reward/X 全 0？** → 在奖励函数里加一行 `print(dist.mean())` 验证函数被调用且有非零输出；检查是否用了 `body_ids[0]`
3. **前 50 迭代数据异常？** → obs normalization 预热，正常，等到 50 迭代后再判断
4. **奖励有非零值但策略不动？** → 检查 `std` / `sigma` 是否匹配初始 EE-port 距离
5. **float 参数看起来没生效？** → 把参数硬编码进函数，排除 `RewTerm.params` 转发问题

---

## 重要修正：Phase 3 体坐标系错误（已在实际代码中修复）

Phase 3 文档中的代码片段用了错误的 body 名称和端口位置，这是实际训练初期 `insertion_success=0` 的根本原因。**以下是正确实现，与现在代码中实际存在的一致：**

### 正确的 EE frame：`sfp_tip_link`，不是 `wrist_3_link`

`wrist_3_link` 是机械臂法兰盘，`sfp_tip_link` 才是 SFP 线缆插头尖端，二者相差约 25cm。将 `wrist_3_link` 当做 EE 会导致策略学习错误的接近目标，训练完后看起来有收敛但实际永远不会插入。

所有 `RewardsCfg` 和 `ObservationsCfg` 中的 `body_names` 必须是 `"sfp_tip_link"`。

### 正确的端口位置：`sfp_port_0_link_entrance`，不是 NIC 卡 root

NIC 卡的 `root_pos_w`（Z≈0.074m）是卡的几何中心，SFP 端口入口（Z≈0.152m）在其上方 7.7cm。必须用 local offset + `quat_apply` 计算真实入口位置：

```python
_SFP_PORT_0_LOCAL_OFFSET = torch.tensor([0.01295, -0.07737, 0.00556])

def _get_sfp_entrance_pos(port: RigidObject) -> torch.Tensor:
    offset = _SFP_PORT_0_LOCAL_OFFSET.to(port.data.root_pos_w.device)
    offset_b = offset.unsqueeze(0).expand(port.data.root_pos_w.shape[0], -1)
    return port.data.root_pos_w[:, :3] + quat_apply(port.data.root_quat_w, offset_b)
```

**如何确认 body frame 和端口位置**：

使用 `scripts/inspect_bodies.py` 和 `scripts/inspect_joints.py` 脚本（已存在于代码库中），在 Isaac Lab 容器内运行：

```bash
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_bodies.py
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/inspect_joints.py
```

---

## 训练结果：实际观测维度（108 维，非 28 维）

训练用的 `aic_unified_robot_cable_sdf.usd` 是一体化 robot+cable 资产，包含 **46 个关节**（6 个臂关节 + 40 个线缆段关节）。`joint_pos_rel` 和 `joint_vel_rel` 没有指定 `joint_names`，所以默认对全部 46 个关节采样：

```
obs_dim = 46 + 46 + 7 + 3 + 6 = 108
         ↑    ↑    ↑   ↑   ↑
         all  all  eef port last
         jpos jvel pos  rel  act
```

关节顺序（由 `inspect_joints.py` 确认）：

| idx | 名称 | default |
|-----|------|---------|
| 0-5 | 6 个 UR5e 臂关节 | [0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110] |
| 6-45 | 40 个线缆段关节 (`joint_0_1:1`…) | 全部 0.0 |

**部署时的处理**：Gazebo 的 `joint_states` 只有 6 个臂关节，线缆关节不可观测。部署策略用零填充 idx 6-45（它们的训练默认值即为 0），只把臂关节数据填入 idx 0-5。实验表明此近似足以使策略正常运行。

---

## Phase 5 正确启动顺序

### 5.0 每次测试的启动步骤（顺序不能乱）

```
终端 1（Gazebo 评测环境）
  ↓
bash -x /entrypoint.sh ground_truth:=true start_aic_engine:=true
  等待 Gazebo 窗口和 RViz 窗口弹出（约 20-30 秒）
  ↓
终端 2（RL 策略节点）
  ↓
cd ~/Code/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=my_policy_node.RLPolicy
```

**关键顺序要求**：

1. **必须先启动 Gazebo，再启动策略节点**。策略节点在初始化时会等待 TF 树（10 秒超时）。如果 Gazebo 还没起来，TF 查找会超时，策略 abort。
2. **使用 `bash -x /entrypoint.sh`，不要用 `distrobox enter -r aic_eval && /entrypoint.sh`**。直接 `distrobox enter` 后执行 entrypoint.sh 会立即退出（distrobox 非交互式 shell 的 init 问题）；`bash -x` 方式可以正常启动 Gazebo 和 RViz。
3. 启动策略节点前**无需单独启动 Zenoh router**。`bash -x /entrypoint.sh` 已经在内部启动了 Zenoh，策略节点通过 pixi 的 ROS 2 环境自动接入同一 ROS 2 DDS 网络。

### 5.1（更新）从容器拷出 checkpoint 的正确路径

Isaac Lab 容器内的日志路径是 `/workspace/isaaclab/logs/...`，不是 `/root/IsaacLab/logs/...`：

```bash
# 先找到训练时间戳
docker exec isaac-lab-base ls /workspace/isaaclab/logs/rsl_rl/aic_task/

# 拷出最佳 checkpoint（用实际时间戳和迭代数替换）
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/aic_task/<timestamp>/model_3000.pt \
    ~/Code/ws_aic/src/aic/my_policy_node/my_policy_node/checkpoints/model_best.pt
```

**checkpoint 安装方式**：`setup.py` 的 `data_files` 包含了 `checkpoints/*.pt`，所以每次 `pixi reinstall ros-kilted-my-policy-node` 时，checkpoint 会自动被复制到 ROS 2 share 目录（`ament_index_python` 可以找到）：

```python
# setup.py
data_files=[
    ...
    ('share/my_policy_node/checkpoints', ['my_policy_node/checkpoints/model_best.pt']),
],
```

```python
# RLPolicy.py 加载路径
from ament_index_python.packages import get_package_share_directory
ckpt_path = Path(get_package_share_directory("my_policy_node")) / "checkpoints" / "model_best.pt"
```

**不要**把 checkpoint 手动复制到 `.pixi/envs/default/lib/.../site-packages/`——这是 hackish 的方式，下次 reinstall 就会丢失。

### 5.2（更新）RLPolicy.py 的正确结构

与 Phase 5 原始文档的差异：

| 项目 | 原始文档（错误） | 实际正确 |
|------|-----------------|---------|
| obs_dim | 28（硬编码） | 从 checkpoint `actor.0.weight.shape[1]` 自动读取（=108） |
| n_joints | 6（只有臂） | 46（臂+线缆），部署时线缆 idx 用 0 填充 |
| normalizer 类型 | 自定义 `running_mean/var/count` | RSL-RL 格式：`_mean/_var/_std` 形状 `[1, obs_dim]`，`count` 为标量 |
| normalizer 属性名 | `obs_normalizer` | `actor_obs_normalizer`（必须与 checkpoint key 完全一致） |
| 加载 key 过滤 | `startswith("actor.", "obs_normalizer.")` | `startswith("actor.", "actor_obs_normalizer.")` |

---

## Phase 5 部署踩坑记录

### 部署坑 1：checkpoint 路径 — `Path(__file__).parent` 在安装后失效

**症状**：
```
FileNotFoundError: Checkpoint not found: /home/stark/.../site-packages/my_policy_node/checkpoints/model_best.pt
```

**根因**：用 `Path(__file__).parent / "checkpoints"` 时，`__file__` 在 `pixi reinstall` 后指向 site-packages 里的已安装文件，但 `checkpoints/` 目录没有被复制过去（仅 `.py` 文件被安装）。

**解决方案**：
1. 在 `setup.py` 里把 `checkpoints/*.pt` 加入 `data_files`（不是 `package_data`）
2. 在代码里用 `ament_index_python.get_package_share_directory("my_policy_node")` 获取路径

```python
# setup.py
('share/my_policy_node/checkpoints', ['my_policy_node/checkpoints/model_best.pt']),
```

**规律**：ROS 2 包的运行时资产（模型权重、配置文件等）应放在 `share/` 下（`data_files`），用 `ament_index_python` 查找，而非依赖 Python 模块路径。

---

### 部署坑 2：obs_dim 硬编码导致 size mismatch

**症状**：
```
size mismatch for actor.0.weight: copying a param with shape torch.Size([512, 108])
into torch.Size([512, 28])
```

**根因**：训练环境的 `joint_pos_rel`/`joint_vel_rel` 没有 `joint_names` 过滤，包含了全部 46 个关节，导致实际 obs_dim=108，而 RLPolicy.py 里硬编码了 `_OBS_DIM = 28`。

**解决方案**：不硬编码 obs_dim，从 checkpoint 自动推断：

```python
obs_dim = int(state_dict["actor.0.weight"].shape[1])
n_joints = (obs_dim - 16) // 2  # obs = joints*2 + eef(7) + port(3) + action(6)
self._actor = _Actor(obs_dim, _HIDDEN, _ACT_DIM)
```

**规律**：网络输入维度永远从 checkpoint 读取，不要手动计算。手动计算容易因训练时的"细节"（如 env 包含了哪些关节）而出错。

---

### 部署坑 3：RSL-RL normalizer buffer 名称和形状不匹配

**症状**：
```
size mismatch for actor_obs_normalizer._mean:
  copying shape torch.Size([1, 108]) into torch.Size([108])
```

**根因**：RSL-RL 的 EmpiricalNormalization 使用的 buffer 名称和形状与自行实现时的直觉不同：
- 名称：`_mean`、`_var`、`_std`（预计算的标准差），以及标量 `count`
- 形状：`_mean/_var/_std` 是 `[1, obs_dim]`（带 batch 维），不是 `[obs_dim]`
- `count` 是标量 `torch.tensor(0.0)`，形状 `[]`，不是 `torch.zeros(1)`（形状 `[1]`）

**解决方案**：

```python
class _EmpiricalNorm(nn.Module):
    def __init__(self, shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, shape))   # [1, shape]
        self.register_buffer("_var",  torch.ones(1, shape))    # [1, shape]
        self.register_buffer("_std",  torch.ones(1, shape))    # [1, shape]
        self.register_buffer("count", torch.tensor(0.0))       # scalar []

    def forward(self, x):
        return (x - self._mean) / (self._std + self.eps)
```

**规律**：移植 RSL-RL checkpoint 时，先 `print(list(state_dict.keys()))` 和各 tensor 的 `.shape`，逐一对照自己实现的 buffer 名称和形状，再写加载代码。

---

### 部署坑 4：加载 key 过滤漏掉 normalizer

**症状**：checkpoint 加载显示 `missing: ['actor_obs_normalizer._mean', ...]`，normalizer 用随机初始值而非训练值，推理结果错误。

**根因**：加载过滤代码写的是 `startswith(("actor.", "obs_normalizer."))`，而 checkpoint 里的 key 是 `actor_obs_normalizer._mean`，它不以 `"actor."` 或 `"obs_normalizer."` 开头。

**解决方案**：

```python
for k, v in state_dict.items():
    k2 = k.removeprefix("actor_critic.")
    if k2.startswith(("actor.", "actor_obs_normalizer.")):
        actor_state[k2] = v
```

**规律**：写加载过滤前先把 `state_dict.keys()` 打印出来，确认实际 key 前缀，不要靠猜测。（代码里已有 `self.get_logger().info(f"Checkpoint keys: {list(state_dict.keys())[:20]}")` 正是为此设计的。）
