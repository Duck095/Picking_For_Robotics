import time
import math
import random
from dataclasses import dataclass

import pybullet as p
import pybullet_data

# =========================
# USER TUNABLE KNOBS (chỉnh ở đây)
# =========================

DT = 1.0 / 240.0                # bước mô phỏng (giữ nguyên)    

# Conveyor
CONVEYOR_SPEED = 0.30           # vận tốc băng tải (m/s)
SPAWN_Y = -0.60                 # vị trí spawn theo Y 
DESPAWN_Y = 0.90                # khi vật đi qua Y này sẽ xoá khỏi thế giới 
SPAWN_INTERVAL = 1.20           # tăng -> vật thưa hơn 
MIN_SPAWN_DY = 0.09             # tối thiểu khoảng cách theo Y ở vùng spawn giữa các vật (để tránh chồng vật)

# Robot speed (nhỏ hơn -> nhanh hơn)
SPEED_SCALE = 0.75              # 0.6 nhanh, 0.8 vừa, 1.0 chậm

# Steps (giảm -> nhanh hơn). SPEED_SCALE vẫn nhân vào steps.
MOVE_STEPS_HOVER = 80               # tới hover trên vật 
MOVE_STEPS_DESCEND = 70             # xuống để lấy vật    
MOVE_STEPS_LIFT = 70                # nâng lên sau khi kẹp vật 
MOVE_STEPS_TO_BIN = 100             # tới thùng (điểm hover) 
MOVE_STEPS_BIN_DESCEND = 70         # xuống thùng để thả vật 
MOVE_STEPS_HOME_SETTLE = 140        # khi về home, giữ yên vài bước để ổn định trước khi làm việc khác

# Debug
DEBUG_PRINT_EVERY = 80      # in debug mỗi N step
DEBUG_OVERLAY = True        # debug overlay text
DEBUG_LINES = True          # debug lines (pick path, etc)

# ROI pick box in world
PICK_ROI = {
    "xmin": 0.30, "xmax": 0.75,     # vùng an toàn hơn là 0.32-0.72
    "ymin": -0.10, "ymax": 0.25,    # vùng an toàn hơn là -0.08-0.22
    "z_pick": 0.018,                # z khi pick (bề mặt vật)
    "z_hover": 0.18,                # z khi hover (trên vật)
}

# Bins (đặt trong workspace an toàn hơn; nếu muốn xa hơn hãy tăng dần, đừng nhảy quá xa)
BINS = {
    "red":    (0.6, 0.4, 0.02),    
    "green":  (0.3, 0.4, 0.02),     
    "blue":   (0, 0.5, 0.02),
    "yellow": (-0.3, 0.2, 0.02),    # xa nhất -> xử lý bằng waypoint + hover cao hơn
}

# waypoint trung gian để tránh duỗi thẳng khi đi tới thùng xa
WAYPOINT_MID = (0.45, 0.35, 0.22)   # (x,y,z) bạn có thể chỉnh
YELLOW_EXTRA_HOVER = 0.06           # hover cao hơn cho thùng vàng


# =========================
# Robot constants
# =========================
EE_LINK = 11                                        # end-effector link index
ARM_JOINTS = list(range(7))                         # là các khớp cánh tay 0-6
GRIPPER_JOINTS = [9, 10]                            # khớp kẹp (cả hai bên)
HOME_Q = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]      # tư thế home
EE_DOWN_ORN = None                                  # sẽ khởi tạo sau   

COLORS = {
    "red":    (1, 0, 0, 1),    
    "green":  (0, 1, 0, 1),
    "blue":   (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
}


# =========================
# Data structures
# =========================
@dataclass
class Obj:
    body_id: int            # id của body trong pybullet
    color_name: str         # tên màu (key trong COLORS)
    sorted: bool = False    # đã được phân loại vào thùng chưa


# =========================
# Debug helper
# =========================
class Debugger:
    def __init__(self):
        self._text_ids = {}     # giữ id của các text để có thể xoá nếu cần
        self._line_ids = []     # giữ id của các line để có thể xoá nếu cần
        self._keys_order = []   # giữ thứ tự các key để xếp chồng text đúng

    def overlay(self, key, text, color=(1, 0, 1), size=1.2):
        if not DEBUG_OVERLAY:               # tắt debug overlay      
            return
        if key not in self._keys_order:     # lần đầu thêm key, lưu thứ tự 
            self._keys_order.append(key)    # thứ tự thêm key quan trọng để xếp chồng text đúng 
        idx = self._keys_order.index(key)   # vị trí trong thứ tự 

        anchor = [0.9, -0.6, 1.0]   # vị trí góc trên bên trái màn hình (đơn vị tỷ lệ), hiện thị từ dưới lên 
        pos = [anchor[0], anchor[1], anchor[2] - 0.03 * idx]    # tính vị trí text dựa trên thứ tự 

        if key in self._text_ids:   # xoá text cũ nếu có 
            try:
                p.removeUserDebugItem(self._text_ids[key]) # xoá text cũ nếu có 
            except Exception:
                pass
        # thêm text mới vào đúng vị trí và lưu id 
        self._text_ids[key] = p.addUserDebugText(
            text, pos, textColorRGB=color, textSize=size, lifeTime=0
        )

    def line(self, a, b, rgb=(1, 0, 1), width=2, life=0.15):
        if not DEBUG_LINES:     # tắt debug lines 
            return
        # xoá các line cũ (nếu có) để tránh đầy bộ nhớ 
        lid = p.addUserDebugLine(a, b, rgb, lineWidth=width, lifeTime=life)
        self._line_ids.append(lid)
    
    def draw_roi_and_bins(self):
        x0, x1 = PICK_ROI["xmin"], PICK_ROI["xmax"]     # vẽ hộp ROI 
        y0, y1 = PICK_ROI["ymin"], PICK_ROI["ymax"]     # vẽ hộp ROI 
        z = PICK_ROI["z_pick"]                          # vẽ hộp ROI 
        pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]   # 4 góc hộp ROI 
        for i in range(4): # vẽ các cạnh hộp ROI 
            # nối điểm i với điểm (i+1)%4 để tạo thành hình chữ nhật 
            p.addUserDebugLine(pts[i], pts[(i + 1) % 4], [1, 0, 1], lineWidth=2, lifeTime=0)

        for name, (bx, by, bz) in BINS.items():     # vẽ thùng 
            # vẽ hình chữ nhật thùng 
            p.addUserDebugText(
                name, (bx, by, bz + 0.06),          # dán nhãn thùng 
                textColorRGB=COLORS[name][:3],      # màu nhãn trùng với màu thùng
                textSize=1.4,                       # kích thước nhãn 
                lifeTime=0                          # vĩnh viễn 
            )


# =========================
# Panda robot controller
# =========================
class PandaRobot:
    def __init__(self):
        self.robot_id = p.loadURDF( 
            "franka_panda/panda.urdf",          # tải mô hình URDF của robot Panda 
            useFixedBase=True,                  # robot cố định trên mặt đất
            flags=p.URDF_USE_SELF_COLLISION     # bật tự va chạm giữa các phần của robot 
        )
        self.grasp_cid = None                   # constraint id khi kẹp vật 
        self.held_obj = None                    # id vật đang kẹp (nếu có)

        # cache joint limits for nullspace IK
        self.ll, self.ul, self.jr = [], [], []  # giới hạn dưới, giới hạn trên, phạm vi di chuyển
        for j in ARM_JOINTS:
            # lấy thông tin khớp
            info = p.getJointInfo(self.robot_id, j)                                         # trả về nhiều thông tin về khớp 
            low, high = info[8], info[9]                                                    # giới hạn dưới và trên 
            self.ll.append(low)                                                             # lưu giới hạn dưới 
            self.ul.append(high)                                                            # lưu giới hạn trên 
            # nếu phạm vi nhỏ hơn 10 radian, dùng phạm vi đó; nếu không, dùng 2π (để tránh giới hạn quá rộng) 
            rng = (high - low) if (high > low and abs(high - low) < 10) else 2 * math.pi    # tính phạm vi di chuyển 
            self.jr.append(rng)                                                             # lưu phạm vi di chuyển

        self.reset_home()                                                                   # đặt robot về tư thế home ban đầu

    def _step(self, n=1, realtime=False):   # bước mô phỏng 
        for _ in range(n):                  
            p.stepSimulation()              # bước mô phỏng một bước 
            if realtime:                    # nếu chế độ realtime, chờ đúng thời gian bước mô phỏng
                time.sleep(DT)              

    def open_gripper(self):
        for j in GRIPPER_JOINTS:    # mở kẹp
            # key fix: mở rộng hơn để chắc chắn kẹp được vật
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=80)

    def close_gripper(self):        # đóng kẹp
        for j in GRIPPER_JOINTS:
            # key fix: đóng chặt hơn để chắc chắn kẹp được vật
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.0, force=140)

    def get_ee_pose(self):
        # trả về (pos, orn) của end-effector
        ls = p.getLinkState(self.robot_id, EE_LINK)
        return ls[4], ls[5]

    # ---- Key fix: Nullspace IK biased to HOME_Q to avoid straight-arm singularities
    def ik(self, target_pos, target_orn):
        # sử dụng inverse kinematics với nullspace để tránh duỗi thẳng
        q = p.calculateInverseKinematics(
            self.robot_id,                  # id robot
            EE_LINK,                        # link end-effector
            targetPosition=target_pos,      # vị trí mục tiêu
            targetOrientation=target_orn,   # hướng mục tiêu
            lowerLimits=self.ll,            # giới hạn dưới
            upperLimits=self.ul,            # giới hạn trên
            jointRanges=self.jr,            # phạm vi di chuyển
            restPoses=HOME_Q,               # bias to HOME
            maxNumIterations=220,           # số lần lặp tối đa
            residualThreshold=1e-4          # ngưỡng dư
        )
        return [q[i] for i in range(7)]     # chỉ lấy 7 khớp cánh tay

    # ---- di chuyển đến tư thế khớp hoặc EE mục tiêu 
    def goto_joints(self, q, steps=100, realtime=True):
        steps = max(10, int(steps * SPEED_SCALE))       # điều chỉnh số bước theo SPEED_SCALE
        for _ in range(steps):                          # lặp qua số bước 
            for i, j in enumerate(ARM_JOINTS):          # lặp qua các khớp cánh tay 
                p.setJointMotorControl2(                # sử dụng điều khiển vị trí 
                    self.robot_id, j,                   # id robot và khớp 
                    p.POSITION_CONTROL,                 # chế độ điều khiển vị trí
                    targetPosition=q[i],                # vị trí mục tiêu 
                    force=250                           # lực tối đa 
                )
            self._step(1, realtime=realtime)            # bước mô phỏng một bước

    # ---- di chuyển đến vị trí và hướng EE mục tiêu
    def goto_ee(self, pos, orn, steps=100, realtime=True):
        q = self.ik(pos, orn)                                   # tính toán tư thế khớp từ vị trí và hướng EE mục tiêu    
        self.goto_joints(q, steps=steps, realtime=realtime)     # di chuyển đến tư thế khớp tính được

    # ---- Key fix: detach before home, and HOLD pose via motors
    def reset_home(self):
        self.detach()                                           # tháo vật nếu đang kẹp
        self.open_gripper()                                     # mở kẹp 
        for i, j in enumerate(ARM_JOINTS):                      # đặt lại trạng thái khớp về HOME_Q ngay lập tức    
            p.resetJointState(self.robot_id, j, HOME_Q[i])      # đặt trạng thái khớp về HOME_Q

        # HOLD home (this is what prevents drift after a few cycles)
        for i, j in enumerate(ARM_JOINTS):
            p.setJointMotorControl2(
                self.robot_id, j,                           # id robot và khớp
                p.POSITION_CONTROL,                         # chế độ điều khiển vị trí 
                targetPosition=HOME_Q[i],                   # vị trí mục tiêu là HOME_Q 
                force=250                                   # lực tối đa 
            )
        self._step(MOVE_STEPS_HOME_SETTLE, realtime=True)   # bước mô phỏng để ổn định 

    # ---- làm mượt về HOME_Q 
    def go_home_smooth(self, steps=180, realtime=True):
        """Move to HOME_Q smoothly using POSITION_CONTROL (no teleport)."""
        self.detach()                                               # tháo vật nếu đang kẹp
        self.open_gripper()                                         # mở kẹp 
        # set target home and just simulate for many steps
        for i, j in enumerate(ARM_JOINTS):          
            p.setJointMotorControl2(
                self.robot_id, j,                                   # id robot và khớp
                p.POSITION_CONTROL,                                 # chế độ điều khiển vị trí
                targetPosition=HOME_Q[i],                           # vị trí mục tiêu là HOME_Q
                force=250                                           # lực tối đa
            )
        self._step(int(steps * SPEED_SCALE), realtime=realtime)     # bước mô phỏng để về HOME_Q


    # ---- gắp và thả vật 
    def attach(self, obj_id):
        # tạo constraint cố định giữa end-effector và vật 
        if self.grasp_cid is not None:
            return
        # tạo constraint cố định giữa end-effector và vật 
        self.grasp_cid = p.createConstraint(
            parentBodyUniqueId=self.robot_id,       # id robot 
            parentLinkIndex=EE_LINK,                # link end-effector
            childBodyUniqueId=obj_id,               # id vật
            childLinkIndex=-1,                      # vật không có link con
            jointType=p.JOINT_FIXED,                # loại khớp cố định
            jointAxis=[0, 0, 0],                    # không cần thiết cho khớp cố định
            parentFramePosition=[0, 0, 0.03],       # vị trí khung cha (end-effector)
            childFramePosition=[0, 0, 0]            # vị trí khung con (vật)
        )
        self.held_obj = obj_id                      # lưu id vật đang kẹp

    # ---- tháo vật 
    def detach(self):
        # tháo constraint cố định giữa end-effector và vật
        if self.grasp_cid is None:
            return
        try:
            p.removeConstraint(self.grasp_cid)  # tháo constraint 
        except Exception:
            pass
        self.grasp_cid = None                   # reset constraint id
        self.held_obj = None                    # reset id vật đang kẹp


# =========================
# Conveyor World
# =========================
class ConveyorWorld:
    def __init__(self, dbg: Debugger):
        self.objects: list[Obj] = []        # danh sách vật trong thế giới
        self.last_spawn_t = time.time()     # thời gian spawn cuối cùng
        self.dbg = dbg                      # debugger để vẽ debug

    # ---- spawn vật nếu cần
    def spawn_if_needed(self):
        now = time.time()                               # thời gian hiện tại
        if now - self.last_spawn_t < SPAWN_INTERVAL:    # kiểm tra khoảng thời gian spawn   
            return
        self.last_spawn_t = now                         # cập nhật thời gian spawn cuối cùng
        self.spawn_object()                             # spawn vật mới 

    # ---- spawn một vật mới
    def spawn_object(self):
        # giữ khoảng cách tối thiểu giữa các vật ở vùng spawn
        for o in self.objects:
            if o.sorted:    # nếu vật đã được phân loại, bỏ qua
                continue
            # lấy vị trí vật 
            pos, _ = p.getBasePositionAndOrientation(o.body_id)
            if abs(pos[1] - SPAWN_Y) < MIN_SPAWN_DY:
                return

        color_name = random.choice(list(COLORS.keys()))     # chọn màu ngẫu nhiên 
        rgba = COLORS[color_name]                           # lấy giá trị RGBA của màu
        half = 0.02                                         # nửa kích thước hộp (20mm)     

        # tạo hình dạng va chạm và hình dạng hiển thị
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
        # tạo hình dạng hiển thị với màu sắc
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=rgba)

        x = random.uniform(PICK_ROI["xmin"], PICK_ROI["xmax"])  # vị trí X ngẫu nhiên trong vùng ROI
        y = SPAWN_Y                                             # vị trí Y cố định tại SPAWN_Y                                           
        z = 0.02                                                # vị trí Z để vật nằm trên băng tải (20mm)

        # tạo vật thể đa khối (multi-body) với khối lượng 0.05kg
        body = p.createMultiBody(
            baseMass=0.05,                      # khối lượng vật (50g)
            baseCollisionShapeIndex=col_shape,  # chỉ số hình dạng va chạm
            baseVisualShapeIndex=vis_shape,     # chỉ số hình dạng hiển thị
            basePosition=[x, y, z],             # vị trí ban đầu
        )
        # thiết lập ma sát cho vật 
        p.changeDynamics(body, -1, lateralFriction=0.9, rollingFriction=0.01, spinningFriction=0.01)

        # conveyor initial velocity (no teleport)
        p.resetBaseVelocity(body, linearVelocity=[0.0, CONVEYOR_SPEED, 0.0])

        # thêm vật vào danh sách vật trong băng chuyền
        self.objects.append(Obj(body, color_name, sorted=False))

    # ---- cập nhật băng tải
    def update_conveyor(self, held_obj_id=None):
        to_remove = []
        # duyệt qua tất cả vật trong băng chuyền
        for obj in self.objects:
            if obj.sorted:   # nếu vật đã được phân loại, bỏ qua
                continue
            # bỏ qua vật đang được kẹp
            if held_obj_id is not None and obj.body_id == held_obj_id:
                continue

            # kiểm tra nếu vật đã đi qua DESPAWN_Y
            pos, _ = p.getBasePositionAndOrientation(obj.body_id)
            if pos[1] > DESPAWN_Y:      # nếu vật vượt qua DESPAWN_Y
                to_remove.append(obj)   # đánh dấu vật để xoá
                continue

            # cập nhật vận tốc băng tải (giữ nguyên vận tốc)
            p.resetBaseVelocity(obj.body_id, linearVelocity=[0.0, CONVEYOR_SPEED, 0.0])

        # xóa các vật đã vượt qua DESPAWN_Y 
        for obj in to_remove:
            try:
                p.removeBody(obj.body_id)  # xoá vật khỏi băng chuyền
            except Exception:
                pass
            # xoá vật khỏi danh sách
            if obj in self.objects:
                self.objects.remove(obj)   

    # ---- lấy vật ứng viên để pick
    def get_pick_candidate(self):
        cx = 0.5 * (PICK_ROI["xmin"] + PICK_ROI["xmax"])    # tâm X của vùng ROI
        cy = 0.5 * (PICK_ROI["ymin"] + PICK_ROI["ymax"])    # tâm Y của vùng ROI

        best = None                         # vật ứng viên tốt nhất
        best_d2 = 1e9                       # khoảng cách bình phương tốt nhất (lớn ban đầu)
        # duyệt qua tất cả vật trong băng chuyền
        for obj in self.objects:
            if obj.sorted:
                continue
            # lấy vị trí vật 
            pos, _ = p.getBasePositionAndOrientation(obj.body_id)
            # kiểm tra nếu vật nằm trong vùng ROI
            x, y = pos[0], pos[1]
            # nếu vật nằm trong vùng ROI
            if (PICK_ROI["xmin"] <= x <= PICK_ROI["xmax"]) and (PICK_ROI["ymin"] <= y <= PICK_ROI["ymax"]):
                d2 = (x - cx) ** 2 + (y - cy) ** 2  # tính khoảng cách bình phương đến tâm ROI
                if d2 < best_d2:                    # nếu khoảng cách này nhỏ hơn khoảng cách tốt nhất hiện tại
                    best_d2 = d2                    # cập nhật khoảng cách tốt nhất
                    best = obj                      # cập nhật vật ứng viên tốt nhất
        return best                                 # trả về vật ứng viên tốt nhất (hoặc None nếu không có)


# =========================
# Pick & Place primitive (rule-based)
# =========================
def pick_and_place(robot: PandaRobot, world: ConveyorWorld, obj: Obj, ee_down_orn, dbg: Debugger):
    # ví trí của vật
    pos, _ = p.getBasePositionAndOrientation(obj.body_id)   # lấy vị trí và hướng của vật
    x, y, _ = pos                                           # lấy tọa độ X, Y, Z của vật

    # approach hover
    hover = [x, y, PICK_ROI["z_hover"]]                                         # vị trí hover trên vật
    dbg.line(hover, [x, y, PICK_ROI["z_pick"]], rgb=(1, 1, 0), width=3)         # vẽ đường từ hover đến vị trí pick

    robot.open_gripper()                                                        # mở kẹp
    robot.goto_ee(hover, ee_down_orn, steps=MOVE_STEPS_HOVER, realtime=True)    # di chuyển đến vị trí hover

    # resample right before descend (important because conveyor moves)
    pos2, _ = p.getBasePositionAndOrientation(obj.body_id)  # lấy lại vị trí vật
    x2, y2, _ = pos2                                        # lấy tọa độ X, Y, Z của vật

    hover2 = [x2, y2, PICK_ROI["z_hover"]]                  # vị trí hover trên vật (cập nhật)
    grasp2 = [x2, y2, PICK_ROI["z_pick"]]                   # vị trí pick (cập nhật)

    robot.goto_ee(hover2, ee_down_orn, steps=int(MOVE_STEPS_HOVER * 0.6), realtime=True)    # di chuyển đến vị trí hover cập nhật
    robot.goto_ee(grasp2, ee_down_orn, steps=MOVE_STEPS_DESCEND, realtime=True)             # di chuyển đến vị trí pick

    # close + attach
    robot.close_gripper()               # đóng kẹp
    robot._step(30, realtime=True)      # chờ một chút để kẹp chắc chắn
    robot.attach(obj.body_id)           # kẹp vật

    # nâng lên
    robot.goto_ee([x2, y2, PICK_ROI["z_hover"]], ee_down_orn, steps=MOVE_STEPS_LIFT, realtime=True) 

    # ---- chỗ đặt vật vào thùng
    bx, by, _ = BINS[obj.color_name]

    # đi qua một waypoint để tránh duỗi thẳng (đặc biệt là với thùng xa)
    robot.goto_ee(list(WAYPOINT_MID), ee_down_orn, steps=int(MOVE_STEPS_TO_BIN * 0.7), realtime=True)   

    extra_hover = YELLOW_EXTRA_HOVER if obj.color_name == "yellow" else 0.0     # hover cao hơn cho thùng vàng
    place_hover = [bx, by, PICK_ROI["z_hover"] + extra_hover]                   # vị trí hover trên thùng
    place = [bx, by, PICK_ROI["z_pick"]]                                        # vị trí đặt vật vào thùng

    robot.goto_ee(place_hover, ee_down_orn, steps=MOVE_STEPS_TO_BIN, realtime=True)     # di chuyển đến vị trí hover thùng
    robot.goto_ee(place, ee_down_orn, steps=MOVE_STEPS_BIN_DESCEND, realtime=True)      # di chuyển đến vị trí đặt vật

    # detach + open
    robot.detach()                  # tháo vật
    robot.open_gripper()            # mở kẹp
    robot._step(15, realtime=True)  # chờ một chút để vật rơi vào thùng

    # linearVelocity là vận tốc tuyến tính dùng để giữ vật trong thùng, 
    # thông số này rất quan trọng nó giúp vật không bị rơi ra ngoài thùng khi băng tải vẫn di chuyển
    
    # angularVelocity là vận tốc góc dùng để giữ vật trong thùng, 
    # thông số này giúp vật không bị quay tròn khi băng tải vẫn di chuyển
    
    # giữ vật trong thùng: dừng nó + đánh dấu đã phân loại để băng tải không kéo nó nữa
    p.resetBaseVelocity(obj.body_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    # thiết lập ma sát cao hơn để vật không trượt
    p.changeDynamics(obj.body_id, -1, lateralFriction=1.2, rollingFriction=0.02, spinningFriction=0.02)
    obj.sorted = True   # đánh dấu vật đã được phân loại

    # rút lui lên hover
    robot.goto_ee(place_hover, ee_down_orn, steps=int(MOVE_STEPS_TO_BIN * 0.6), realtime=True)

# draw world grid helper
def draw_world_grid(
    x_min=-0.6, x_max=0.9,
    y_min=-0.9, y_max=0.9,
    step=0.1,
    z=0.001,                 # nâng nhẹ lên khỏi mặt phẳng để không bị z-fighting
    major_every=5,           # cứ 5 ô (0.5m nếu step=0.1) vẽ đậm hơn
    show_axes=True,
    show_labels=True,
    label_every=0.2          # khoảng cách dán nhãn (m). None để tắt
):
    """
    Vẽ lưới XY trên plane (Z ~ 0).
    - Lines màu xám, line "major" dày hơn.
    - Trục X đỏ, Y xanh lá (tuỳ chọn).
    - Labels in-world (tuỳ chọn).
    """
    # Grid lines parallel to Y (vary X)
    i = 0
    x = x_min
    # vẽ các đường lưới song song với trục Y
    while x <= x_max + 1e-9:
        is_major = (i % major_every == 0)                                           # kiểm tra nếu là đường lưới chính
        color = [0.7, 0.7, 0.7] if is_major else [0.5, 0.5, 0.5]                    # màu sắc dựa trên loại đường lưới
        width = 2.5 if is_major else 1.0                                            # độ dày dựa trên loại đường lưới
        p.addUserDebugLine([x, y_min, z], [x, y_max, z], color, width, lifeTime=0)  # vẽ đường lưới
        i += 1                                                                      # tăng chỉ số đường lưới
        x += step                                                                   # tăng vị trí X

    # Grid lines parallel to X (vary Y)
    i = 0
    y = y_min
    # vẽ các đường lưới song song với trục X
    while y <= y_max + 1e-9:
        is_major = (i % major_every == 0)
        color = [0.7, 0.7, 0.7] if is_major else [0.5, 0.5, 0.5]
        width = 2.5 if is_major else 1.0
        p.addUserDebugLine([x_min, y, z], [x_max, y, z], color, width, lifeTime=0)
        i += 1
        y += step

    # Axes
    if show_axes:
        # X axis (red) 
        p.addUserDebugLine([0, 0, z], [0.5, 0, z], [1, 0, 0], 4, lifeTime=0)
        p.addUserDebugText("X+", [0.52, 0, z], [1, 0, 0], textSize=1.4, lifeTime=0)

        # Y axis (green)
        p.addUserDebugLine([0, 0, z], [0, 0.5, z], [0, 1, 0], 4, lifeTime=0)
        p.addUserDebugText("Y+", [0, 0.52, z], [0, 1, 0], textSize=1.4, lifeTime=0)

        # Origin
        p.addUserDebugText("(0,0)", [0, 0, z], [1, 1, 1], textSize=1.2, lifeTime=0)

    # vẽ nhãn dọc theo các trục(nếu được bật)
    if show_labels and label_every is not None and label_every > 0:
        # label along X at y=0
        x = x_min
        # vẽ nhãn dọc theo trục X tại y=0
        while x <= x_max + 1e-9:
            # kiểm tra nếu vị trí x là bội số của label_every
            if abs((x / label_every) - round(x / label_every)) < 1e-6:
                # dán nhãn tại vị trí (x, 0, z) 
                p.addUserDebugText(f"x={x:.1f}", [x, 0, z], [0.9, 0.9, 0.9], textSize=0.9, lifeTime=0)
            x += step  

        # label along Y at x=0
        y = y_min
        # vẽ nhãn dọc theo trục Y tại x=0
        while y <= y_max + 1e-9:
            # kiểm tra nếu vị trí y là bội số của label_every
            if abs((y / label_every) - round(y / label_every)) < 1e-6:
                # dán nhãn tại vị trí (0, y, z)
                p.addUserDebugText(f"y={y:.1f}", [0, y, z], [0.9, 0.9, 0.9], textSize=0.9, lifeTime=0)
            y += step

# =========================
# Main loop
# =========================
def main():
    global EE_DOWN_ORN

    p.connect(p.GUI)                                                # kết nối với PyBullet ở chế độ GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())          # thêm đường dẫn tìm kiếm dữ liệu bổ sung
    p.setGravity(0, 0, -9.81)                                       # đặt trọng lực trong mô phỏng
    p.setTimeStep(DT)                                               # đặt bước thời gian mô phỏng
    EE_DOWN_ORN = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])     # hướng end-effector hướng xuống dưới

    p.loadURDF("plane.urdf")                                        # tải mặt phẳng URDF

    # Ấn định vị trí camera để có góc nhìn tốt
    draw_world_grid(
        x_min=-0.6, x_max=0.9,  # vẽ lưới trục X từ -0.6m đến 0.9m
        y_min=-0.9, y_max=0.9,  # vẽ lưới trục Y từ -0.9m đến 0.9m
        step=0.1,               # khoảng cách giữa các đường lưới là 0.1m
        major_every=5,          # mỗi 5 đường lưới là đường lưới chính
        show_axes=True,         # hiển thị trục X và Y
        show_labels=True,       # hiển thị nhãn tọa độ
        label_every=0.2         # dán nhãn mỗi 0.2m
    )

    # thiết lập vị trí camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.25, cameraYaw=55, cameraPitch=-35, cameraTargetPosition=[0.42, 0.2, 0.15]
    )

    dbg = Debugger()            # khởi tạo debugger
    dbg.draw_roi_and_bins()     # vẽ vùng ROI và thùng phân loại

    robot = PandaRobot()        # khởi tạo robot Panda
    world = ConveyorWorld(dbg)  # khởi tạo thế giới băng tải

    sim_step = 0                # bước mô phỏng hiện tại
    busy = False                # trạng thái bận rộn của robot

    try:
        while True:
            sim_step += 1

            if not busy:
                world.spawn_if_needed()     # spawn vật nếu cần

            # cập nhật băng tải (bỏ qua vật đang kẹp nếu có)
            world.update_conveyor(held_obj_id=robot.held_obj)

            if not busy:
                cand = world.get_pick_candidate()   # lấy vật ứng viên để pick
                if cand is not None:
                    busy = True              # đặt trạng thái bận rộn
                    pick_and_place(robot, world, cand, EE_DOWN_ORN, dbg)    # thực hiện pick & place
                    robot.go_home_smooth(steps=220, realtime=True)          # về tư thế home mượt mà
                    busy = False                                            # đặt trạng thái không bận rộn

            # Debug
            if sim_step % DEBUG_PRINT_EVERY == 0:       # in thông tin debug mỗi DEBUG_PRINT_EVERY bước
                ee_pos, _ = robot.get_ee_pose()         # lấy vị trí end-effector
                cand = world.get_pick_candidate()       # lấy vật ứng viên để pick
                cand_str = "None"                       # khởi tạo chuỗi vật ứng viên
                if cand is not None:                    
                    # lấy vị trí vật ứng viên
                    cpos, _ = p.getBasePositionAndOrientation(cand.body_id)
                    # định dạng chuỗi thông tin vật ứng viên
                    cand_str = f"id={cand.body_id} color={cand.color_name} pos=({cpos[0]:.3f},{cpos[1]:.3f},{cpos[2]:.3f})"

                # lấy trạng thái khớp cánh tay
                qs = [p.getJointState(robot.robot_id, j)[0] for j in ARM_JOINTS]
                print(
                    f"[step={sim_step}] objs={len(world.objects)} busy={busy} held={robot.held_obj} "
                    f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) cand={cand_str} "
                    f"q={[round(v,3) for v in qs]}"
                )

                dbg.overlay("A", f"step: {sim_step}")                                               # hiển thị bước mô phỏng hiện tại
                dbg.overlay("B", f"objects: {len(world.objects)} | busy: {busy}")                   # hiển thị số lượng vật và trạng thái bận rộn
                dbg.overlay("C", f"held_obj: {robot.held_obj}")                                     # hiển thị id vật đang kẹp
                dbg.overlay("D", f"ee: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")        # hiển thị vị trí end-effector
                dbg.overlay("E", f"cand: {cand_str}")                                               # hiển thị thông tin vật ứng viên       
                dbg.overlay("F", f"SPEED_SCALE={SPEED_SCALE} | conveyor={CONVEYOR_SPEED}")          # hiển thị thông số tốc độ

            p.stepSimulation()      # bước mô phỏng một bước
            time.sleep(DT)          # chờ đúng thời gian bước mô phỏng

    except KeyboardInterrupt:   
        pass
    finally:
        p.disconnect()      # ngắt kết nối với PyBullet khi thoát


if __name__ == "__main__":
    main()
