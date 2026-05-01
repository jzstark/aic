# AIC Policy 开发完整规划

## Context

Qualification 截止日期：**2026-05-15**（距今约 3 周）。GPU：RTX 高端。
策略：先快速拿到第一次提交，再用 IL + RL 迭代提升分数。

---

## 评分系统理解（每轮满分 100，共 3 轮）

```
Tier 1 (1分)：  模型能加载、能响应 action
Tier 2 (最高 +24, 最低 -36)：
  +6   轨迹平滑度（低加加速度，Savitzky-Golay 滤波）
  +12  完成时间（≤5秒=满分，≥60秒=0分）
  +6   轨迹效率（路径越短越好）
  -12  力超过 20N 且持续 >1秒
  -24  碰到禁止区域（enclosure、task board 外壁）
Tier 3 (最高 75)：
  75   成功插入正确端口
  -12  插入错误端口（!)
  38-50 部分插入（在端口 bounding box 内）
  0-25 接近分（plug 离 port 越近越高）
```

**关键洞察：**
- Tier 2 分数只有在 Tier 3 > 0（即 plug 到达 port 附近）时才给
- 光是"把 plug 移到 port 附近"就能拿 25 分接近分 + Tier 2 分
- 3 个 trial 类型：Trial 1/2 是 SFP→SFP（NIC卡位置不同），Trial 3 是 SC→SC（反向电缆）

---

## 三条技术路线对比

| 路线 | 原理 | 优点 | 缺点 | 适合时机 |
|------|------|------|------|---------|
| **A: 跑 pre-trained RunACT** | 加载 `grkw/aic_act_policy` HF模型 | 零开发成本，Day1 可提交 | 分数上限未知，不可控 | 立刻 |
| **B: LeRobot IL（Gazebo 遥操作）** | 在 Gazebo 里录制 demos → 训练 ACT | 数据来自真实评估环境，迁移性最好 | 遥操作学习曲线，每个 demo 需手动完成 | Week 1-2 |
| **C: Isaac Lab RL（PPO）** | 在 Isaac Sim 里并行强化学习 | 训练高效（多并行env），可域随机化 | 需配置 Isaac Lab Docker + 下载资产，sim-to-sim 迁移有风险 | Week 2-3 |

**推荐：A → B → C 顺序执行，每完成一步就提交。**

---

## Phase 0：Day 1 — 第一次提交（pre-trained RunACT）

**目标：今天就有东西在提交名单里。**

### Step 1：验证 RunACT 在本地能跑

```bash
# 终端1：启动 eval 容器
distrobox enter -r aic_eval && /entrypoint.sh ground_truth:=false start_aic_engine:=true

# 终端2：运行 RunACT（会自动从 HuggingFace 下载 grkw/aic_act_policy）
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunACT
```

记录 3 个 trial 的得分（保存在 `~/aic_results/`）。

### Step 2：构建 Docker 镜像并提交

```bash
# 构建 model 镜像（Dockerfile: docker/aic_model/Dockerfile）
docker compose -f docker/docker-compose.yaml build model

# 本地端到端测试（必须通过才能提交）
docker compose -f docker/docker-compose.yaml up

# 推送到 ECR + 在门户注册
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
docker tag my-solution:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team>:v1
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team>:v1
```

**关键文件：**
- [docker/aic_model/Dockerfile](../docker/aic_model/Dockerfile) — 提交镜像模板
- [docker/docker-compose.yaml](../docker/docker-compose.yaml) — 本地测试配置
- [aic_example_policies/aic_example_policies/ros/RunACT.py](../aic_example_policies/aic_example_policies/ros/RunACT.py) — 直接使用，不改

**里程碑：** 拿到 v1 提交后的实际分数作为基准线。

---

## Phase 1：Week 1 — 数据采集（LeRobot 遥操作）

**目标：采集 50-100 个高质量 demo，覆盖 3 种 trial 场景。**

### Step 1：学习遥操作

先用键盘控制理解任务空间：

```bash
cd ~/ws_aic/src/aic

# 在 eval 容器跑（开一个终端，ground_truth:=true 方便调试）
# 另一个终端：
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee \
  --robot.teleop_target_mode=cartesian \
  --robot.teleop_frame_id=base_link \
  --display_data=true
```

键盘映射：`w/s/a/d/r/f` = XYZ平移，`Shift+方向` = 旋转，`t` = 速度切换。
**目标：**能稳定手动完成插入，成功率 >70%。

### Step 2：录制 demos

每种 trial 类型录制 20-30 个：

```bash
# Trial 1/2 (SFP): eval 容器用 sample_config.yaml 里 trial_1 配置
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee \
  --robot.teleop_target_mode=cartesian \
  --dataset.repo_id=local/aic_sfp_demos \
  --dataset.single_task="Insert SFP cable" \
  --dataset.push_to_hub=false

# Trial 3 (SC): 切换 eval 容器配置到 trial_3
pixi run lerobot-record \
  --dataset.repo_id=local/aic_sc_demos \
  --dataset.single_task="Insert SC cable" \
  --dataset.push_to_hub=false
```

数据存储格式：HuggingFace dataset（3 RGB 摄像头 + 关节状态 + 动作）。

**关键文件：**
- [aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot.py](../aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot.py) — 摄像头/关节定义
- [aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py](../aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py) — 键盘映射配置
- [aic_engine/config/sample_config.yaml](../aic_engine/config/sample_config.yaml) — trial 场景配置，理解 3 个 trial 的差异

---

## Phase 2：Week 2 — 训练自己的 ACT 模型

**目标：用自己的 demos 训练，比 pre-trained 模型得分更高。**

### Step 1：训练 ACT

```bash
cd ~/ws_aic/src/aic

pixi run lerobot-train \
  --dataset.repo_id=local/aic_sfp_demos \
  --policy.type=act \
  --output_dir=outputs/train/act_aic_v1 \
  --policy.device=cuda \
  --training.num_epochs=100
```