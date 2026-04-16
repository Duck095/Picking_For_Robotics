import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

print("Num joints:", p.getNumJoints(robot))
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode("utf-8")
    link_name = info[12].decode("utf-8")
    print(f"{i:02d} joint={joint_name:20s} link={link_name}")

p.disconnect()