import pybullet as p

# 连接物理引擎
physicsClient = p.connect(p.GUI)  # 使用 GUI 模式
p.setGravity(0, 0, -9.81)  # 设置重力
# 加载 URDF 文件
# robotId = p.loadURDF("/home/pengyang/codebase/H1_RL/HST/legged_gym/resources/robots/h1_lower_body/urdf/h1.urdf")
robotId = p.loadURDF("/home/pengyang/codebase/H1_RL/HST/legged_gym/resources/robots/h1_lower_body/urdf/h1.urdf", [0, 0, 0], [0, 0, 0, 1])
# 主循环
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)  # 控制仿真步长
except KeyboardInterrupt:
    pass

# 断开连接
p.disconnect()