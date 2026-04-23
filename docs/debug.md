# Policy 调试指南

## 1. `get_logger()` — 基本打印调试

输出出现在运行 `pixi run ros2 run aic_model ...` 的终端：

```python
def insert_cable(self, task, get_observation, move_robot, send_feedback):
    obs = get_observation()
    self.get_logger().info(f"wrench z: {obs.wrist_wrench.wrench.force.z:.3f}")
    self.get_logger().warn("还没到目标位置")
    self.get_logger().error("TF 查找失败")
```

三个级别：`info` / `warn` / `error`，按严重程度颜色不同。

---

## 2. `send_feedback()` — 双端可见的状态标记

发回给 aic_engine，eval 容器终端同步显示：

```python
send_feedback("phase 1: approaching port")
send_feedback(f"z_offset={z_offset:.4f}, force={force:.3f}N")
```

适合标记 policy 正在执行到哪个阶段，两个终端都能看，方便对照。

---

## 3. `ground_truth:=true` — 建立成功基准

```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

然后跑 CheatCode：
- 确认环境本身没问题（CheatCode 也失败说明是环境/配置问题）
- 观察端口的真实坐标范围，了解机械臂需要移动到哪里

---

## 4. ROS 2 CLI 工具 — 实时观察系统内部

在主机上另开终端（不用进容器）：

```bash
cd ~/ws_aic/src/aic && pixi shell

# 查看所有活跃话题
ros2 topic list

# 实时打印力矩数据
ros2 topic echo /aic_adapter/observations --field wrist_wrench.wrench.force

# 查看 Observation 发布频率（应为 20Hz）
ros2 topic hz /aic_adapter/observations

# 实时看关节状态
ros2 topic echo /joint_states

# 查看所有活跃节点
ros2 node list

# 查看 aic_model 参数
ros2 param list /aic_model
```

---

## 5. `ros2 bag` — 录制数据离线分析

录制一次完整 trial，之后反复回放，不用重新跑仿真：

```bash
# trial 运行时录制
ros2 bag record -o my_trial_debug \
  /aic_adapter/observations \
  /aic_controller/pose_commands \
  /aic_controller/state

# 回放
ros2 bag play my_trial_debug
```

---

## 6. PlotJuggler — 可视化时间序列

适合分析力矩曲线、关节轨迹等连续信号，找到"接触时力突变"的时刻：

```bash
sudo apt install ros-kilted-plotjuggler-ros
ros2 run plotjuggler plotjuggler
```

连接 ROS 2 source，实时查看 `wrist_wrench.force.z` 等曲线。

---

## 7. RViz — 可视化三维空间

eval 容器启动时已自动打开。在界面中点 "Add" 可添加：
- **TF 坐标系**（`ground_truth:=true` 后能看到端口位置）
- **左/右摄像头图像**（默认只显示中间摄像头）
- **机械臂位姿 Marker**

---

## 8. 手动遥操作 — 探索任务空间

用键盘手动控制机械臂，把夹爪移到端口附近，记下坐标值作为 policy 接近目标的参考：

```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_teleoperation aic_teleoperation
```

---

## 9. Build-Run-Debug 循环

改了 Python 文件后必须重新 reinstall，直接改文件不会生效：

```bash
pixi reinstall ros-kilted-my-policy-node

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=my_policy_node.MyPolicy
```

eval 容器不需要重启，只重启 model 这边即可。

---

## 常见坑

### 坑1：不能用 `time.sleep()`

仿真时钟与系统时钟不同步，务必用基类方法：

```python
# 错误
import time; time.sleep(1.0)

# 正确
self.sleep_for(1.0)
```

### 坑2：大型 import 不能放顶层

`torch` 等大库放在模块顶层会超过 30 秒发现超时，导致节点被杀。放进 `__init__`：

```python
# 错误
import torch  # 文件顶层

# 正确
class MyPolicy(Policy):
    def __init__(self, parent_node):
        import torch  # __init__ 里有 60 秒预算
        super().__init__(parent_node)
```
