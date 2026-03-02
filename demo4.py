import time
import math
import random
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# 1) QUICK PARAMS
# ============================================================

DT = 1.0 / 240.0

# Speed mode
REALTIME = True         # ✅ False = chạy nhanh (không sleep). True = chạy đúng realtime để nhìn
DETECT_EVERY = 8         # ✅ 240/8 = ~30Hz detection (đủ dùng, nhanh hơn nhiều)

# ---------- Conveyor ----------
CONVEYOR_SPEED = 0.30
SPAWN_Y = -0.60
DESPAWN_Y = 0.90
SPAWN_INTERVAL = 1.20
MIN_SPAWN_DY = 0.11

# ---------- Robot speed ----------
SPEED_SCALE = 0.80

# ---------- Motion steps ----------
MOVE_STEPS_HOVER = 85
MOVE_STEPS_DESCEND = 40
MOVE_STEPS_LIFT = 60
MOVE_STEPS_TO_BIN = 115
MOVE_STEPS_HOME_SETTLE = 140

# ---------- Early stop ----------
EE_TOL_HOVER = 0.016
EE_TOL_PLACE = 0.013
EE_SETTLE_STEPS = 8

# ---------- Prediction / Tracking ----------
TRACK_XY_GAIN = 1.0
PRED_TRACK_STEPS = 6
PRED_HOVER_STEPS = 14

# ---------- Gripper ----------
GRIP_HOLD_FORCE = 240
GRIP_CLOSE_STEP = 0.0015
GRIP_STUCK_EPS = 1e-4
GRIP_STUCK_COUNT = 4
REQUIRE_CONTACT = True

# ---------- Attach (still used but stricter) ----------
USE_ATTACH = True
ATTACH_MAX_EE_DIST = 0.025
ATTACH_MAX_FORCE = 3500

# ---------- Gripper yaw ----------
GRIP_YAW = math.pi / 2

# ---------- Place safety ----------
PLACE_CLEAR_Z = 0.14
PLACE_PRE_RELEASE_Z = 0.030
PLACE_RETRACT_Z = 0.10
PLACE_LINEAR_STEPS_DOWN = 55
PLACE_LINEAR_STEPS_UP = 35

# ---------- Debug ----------
DEBUG_PRINT_EVERY = 120
DEBUG_OVERLAY = True
DEBUG_LINES = True


# ============================================================
# 2) WORKSPACE / ROI / BINS
# ============================================================

# ✅ ROI (ô hồng / viền tím) = VÙNG GẮP CHÍNH
PICK_ROI = {
    "xmin": 0.30, "xmax": 0.75,
    "ymin": -0.10, "ymax": 0.25,
    "z_pick": 0.020,
    "z_hover": 0.18,
}

BINS = {
    "red":    (0.60, 0.40, 0.02),
    "green":  (0.30, 0.40, 0.02),
    "blue":   (0.00, 0.50, 0.02),
    "yellow": (-0.30, 0.20, 0.02),
}

WAYPOINT_MID = (0.45, 0.35, 0.22)


def in_pick_roi(x, y):
    return (PICK_ROI["xmin"] <= x <= PICK_ROI["xmax"]) and (PICK_ROI["ymin"] <= y <= PICK_ROI["ymax"])


# ============================================================
# 3) ROBOT CONSTANTS + OBJECT SIZE (pallet)
# ============================================================

EE_LINK = 11
ARM_JOINTS = list(range(7))
GRIPPER_JOINTS = [9, 10]
HOME_Q = [0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8]

EE_DOWN_ORN = None

OBJ_HALF = 0.02
OBJ_SIZE = 2.0 * OBJ_HALF

OBJ_GAP = 0.010
CELL = OBJ_SIZE + OBJ_GAP
LAYER_H = OBJ_SIZE + 0.005

PALLET_N = 4

COLORS = {
    "red":    (1, 0, 0, 1),
    "green":  (0, 1, 0, 1),
    "blue":   (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
}

COLORS_RGB255 = {
    "red":    np.array([255, 0, 0], dtype=np.float32),
    "green":  np.array([0, 255, 0], dtype=np.float32),
    "blue":   np.array([0, 0, 255], dtype=np.float32),
    "yellow": np.array([255, 255, 0], dtype=np.float32),
}


def ee_down_yaw(yaw_rad: float):
    return p.getQuaternionFromEuler([math.pi, 0.0, yaw_rad])


# ============================================================
# 4) DATA STRUCTURES
# ============================================================

@dataclass
class Obj:
    body_id: int
    true_color_name: str      # ground-truth (ENV only)
    sorted: bool = False
    entered_roi: bool = False # ✅ đã từng vào ROI chưa


# ============================================================
# 5) DEBUGGER
# ============================================================

class Debugger:
    def __init__(self):
        self._text_ids = {}
        self._keys_order = []

    def overlay(self, key, text, color=(1, 0, 1), size=1.2):
        if not DEBUG_OVERLAY:
            return
        if key not in self._keys_order:
            self._keys_order.append(key)
        idx = self._keys_order.index(key)
        anchor = [0.9, -0.6, 1.0]
        pos = [anchor[0], anchor[1], anchor[2] - 0.03 * idx]
        if key in self._text_ids:
            try:
                p.removeUserDebugItem(self._text_ids[key])
            except Exception:
                pass
        self._text_ids[key] = p.addUserDebugText(
            text, pos, textColorRGB=color, textSize=size, lifeTime=0
        )

    def line(self, a, b, rgb=(1, 0, 1), width=2, life=0.15):
        if not DEBUG_LINES:
            return
        p.addUserDebugLine(a, b, rgb, lineWidth=width, lifeTime=life)

    def draw_roi_and_bins(self):
        # ROI (magenta)
        x0, x1 = PICK_ROI["xmin"], PICK_ROI["xmax"]
        y0, y1 = PICK_ROI["ymin"], PICK_ROI["ymax"]
        z = PICK_ROI["z_pick"]
        pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        for i in range(4):
            p.addUserDebugLine(pts[i], pts[(i + 1) % 4], [1, 0, 1], lineWidth=3, lifeTime=0)
        p.addUserDebugText("PICK_ROI", (0.5*(x0+x1), 0.5*(y0+y1), z+0.02),
                           textColorRGB=[1, 0, 1], textSize=1.2, lifeTime=0)

        for name, (bx, by, bz) in BINS.items():
            p.addUserDebugText(
                name, (bx, by, bz + 0.06),
                textColorRGB=COLORS[name][:3],
                textSize=1.4,
                lifeTime=0
            )


# ============================================================
# 6) WORLD GRID (optional)
# ============================================================

def draw_world_grid(
    x_min=-0.6, x_max=0.9,
    y_min=-0.9, y_max=0.9,
    step=0.1,
    z=0.001,
    major_every=5,
    show_axes=True,
    show_labels=False,
    label_every=0.2
):
    i = 0
    x = x_min
    while x <= x_max + 1e-9:
        is_major = (i % major_every == 0)
        color = [0.7, 0.7, 0.7] if is_major else [0.5, 0.5, 0.5]
        width = 2.5 if is_major else 1.0
        p.addUserDebugLine([x, y_min, z], [x, y_max, z], color, width, lifeTime=0)
        i += 1
        x += step

    i = 0
    y = y_min
    while y <= y_max + 1e-9:
        is_major = (i % major_every == 0)
        color = [0.7, 0.7, 0.7] if is_major else [0.5, 0.5, 0.5]
        width = 2.5 if is_major else 1.0
        p.addUserDebugLine([x_min, y, z], [x_max, y, z], color, width, lifeTime=0)
        i += 1
        y += step

    if show_axes:
        p.addUserDebugLine([0, 0, z], [0.5, 0, z], [1, 0, 0], 4, lifeTime=0)
        p.addUserDebugText("X+", [0.52, 0, z], [1, 0, 0], textSize=1.4, lifeTime=0)
        p.addUserDebugLine([0, 0, z], [0, 0.5, z], [0, 1, 0], 4, lifeTime=0)
        p.addUserDebugText("Y+", [0, 0.52, z], [0, 1, 0], textSize=1.4, lifeTime=0)


# ============================================================
# 7) PALLETIZER (4x4, reset at 4 layers)
# ============================================================

class Palletizer:
    """
    4x4, multiple layers. Reset when reaches 4 layers (64 blocks) per color.
    fill row far -> near to reduce passing over.
    """
    def __init__(self):
        self.state = {c: {"count": 0} for c in BINS.keys()}

    def _layer_shift(self, layer_idx: int):
        hx = 0.5 * CELL
        hy = 0.5 * CELL
        k = layer_idx % 4
        if k == 0:
            return (0.0, 0.0)
        if k == 1:
            return (hx, 0.0)
        if k == 2:
            return (0.0, hy)
        return (hx, hy)

    def next_place_pose(self, color_name: str):
        cnt = self.state[color_name]["count"]
        layer = cnt // (PALLET_N * PALLET_N)
        idx_in_layer = cnt % (PALLET_N * PALLET_N)

        row_raw = idx_in_layer // PALLET_N
        col = idx_in_layer % PALLET_N
        row = (PALLET_N - 1) - row_raw  # far -> near

        ox, oy, oz = BINS[color_name]

        cx = (col - (PALLET_N - 1) / 2.0) * CELL
        cy = (row - (PALLET_N - 1) / 2.0) * CELL
        sx, sy = self._layer_shift(layer)

        px = ox + cx + sx
        py = oy + cy + sy
        pz = oz + OBJ_HALF + layer * LAYER_H

        place = [px, py, pz]
        hover = [px, py, pz + 0.16]
        return place, hover, (layer, row, col, cnt)

    def commit(self, color_name: str):
        self.state[color_name]["count"] += 1

    def is_full_4_layers(self, color_name: str):
        return self.state[color_name]["count"] >= (PALLET_N * PALLET_N * 4)

    def reset_color(self, color_name: str):
        self.state[color_name]["count"] = 0


# ============================================================
# 8) CONVEYOR WORLD (spawn + miss/remove based on ROI)
# ============================================================

class ConveyorWorld:
    def __init__(self, dbg: Debugger):
        self.objects: list[Obj] = []
        self.dbg = dbg
        self.sim_time = 0.0
        self.last_spawn_sim = -999.0

        # metrics
        self.spawn_count = 0
        self.miss_count = 0
        self.despawn_count = 0
        self.pick_attempt = 0
        self.pick_success = 0
        self.place_success = 0
        self.grasp_fail = 0

    def spawn_if_needed(self):
        if (self.sim_time - self.last_spawn_sim) < SPAWN_INTERVAL:
            return

        # ensure spacing
        for o in self.objects:
            if o.sorted:
                continue
            pos, _ = p.getBasePositionAndOrientation(o.body_id)
            if pos[1] < SPAWN_Y + MIN_SPAWN_DY:
                return

        self.last_spawn_sim = self.sim_time
        self._spawn_object()

    def _spawn_object(self):
        true_color = random.choice(list(COLORS.keys()))
        rgba = COLORS[true_color]

        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[OBJ_HALF, OBJ_HALF, OBJ_HALF])
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[OBJ_HALF, OBJ_HALF, OBJ_HALF], rgbaColor=rgba)

        x = random.uniform(PICK_ROI["xmin"], PICK_ROI["xmax"])
        y = SPAWN_Y
        z = OBJ_HALF

        body = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[x, y, z],
        )

        p.changeDynamics(body, -1, lateralFriction=1.0, rollingFriction=0.02, spinningFriction=0.02)
        p.resetBaseVelocity(body, linearVelocity=[0.0, CONVEYOR_SPEED, 0.0])

        self.objects.append(Obj(body, true_color, sorted=False, entered_roi=False))
        self.spawn_count += 1

    def update_conveyor(self, held_obj_id=None):
        self.sim_time += DT
        to_remove = []

        for obj in self.objects:
            if obj.sorted:
                continue
            if held_obj_id is not None and obj.body_id == held_obj_id:
                continue

            (x, y, _), _ = p.getBasePositionAndOrientation(obj.body_id)

            # mark entered ROI
            if in_pick_roi(x, y):
                obj.entered_roi = True

            # ✅ if entered ROI then exits ROI => miss/remove
            if obj.entered_roi and (not in_pick_roi(x, y)):
                to_remove.append(obj)
                self.miss_count += 1
                continue

            # safety despawn far away
            if y > DESPAWN_Y:
                to_remove.append(obj)
                self.despawn_count += 1
                continue

            p.resetBaseVelocity(obj.body_id, linearVelocity=[0.0, CONVEYOR_SPEED, 0.0])

        for obj in to_remove:
            try:
                p.removeBody(obj.body_id)
            except Exception:
                pass
            if obj in self.objects:
                self.objects.remove(obj)

    def is_valid_object(self, body_id: int):
        for o in self.objects:
            if o.body_id == body_id and (not o.sorted):
                return True
        return False

    def mark_sorted(self, body_id: int):
        for o in self.objects:
            if o.body_id == body_id:
                o.sorted = True
                return


# ============================================================
# 9) FIXED CAMERA PERCEPTION (FAST): seg unique + decode seg
# ============================================================

class FixedCameraPerception:
    """
    Nhanh: dùng segmentationMask để lấy object IDs.
    Không BFS/connected-components.
    Decode seg đúng để ra bodyUniqueId.
    """
    def __init__(self):
        cx = 0.5 * (PICK_ROI["xmin"] + PICK_ROI["xmax"])
        cy = 0.5 * (PICK_ROI["ymin"] + PICK_ROI["ymax"])
        self.cam = {
            "w": 320, "h": 240,
            "fov": 55,
            "near": 0.01, "far": 2.0,
            "pos": [cx, cy, 1.10],
            "target": [cx, cy, 0.02],
            "up": [0, 1, 0],
        }
        self.rgb_noise = 2.0
        self.max_objs = 10
        self.min_pixels = 80

    def render(self):
        view = p.computeViewMatrix(self.cam["pos"], self.cam["target"], self.cam["up"])
        proj = p.computeProjectionMatrixFOV(
            fov=self.cam["fov"], aspect=self.cam["w"] / self.cam["h"],
            nearVal=self.cam["near"], farVal=self.cam["far"]
        )
        img = p.getCameraImage(
            self.cam["w"], self.cam["h"],
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgba = np.reshape(img[2], (self.cam["h"], self.cam["w"], 4)).astype(np.uint8)
        rgb = rgba[:, :, :3].astype(np.float32)
        if self.rgb_noise > 0:
            rgb = np.clip(rgb + np.random.normal(0.0, self.rgb_noise, rgb.shape), 0, 255)
        rgb = rgb.astype(np.uint8)

        depth_buf = np.reshape(img[3], (self.cam["h"], self.cam["w"])).astype(np.float32)
        seg = np.reshape(img[4], (self.cam["h"], self.cam["w"])).astype(np.int32)
        return rgb, depth_buf, seg, view, proj

    def unproject_pixel_to_world(self, u, v, depth_buf_value, view, proj):
        view_m = np.array(view, dtype=np.float64).reshape(4, 4, order="F")
        proj_m = np.array(proj, dtype=np.float64).reshape(4, 4, order="F")
        inv_vp = np.linalg.inv(proj_m @ view_m)

        w, h = self.cam["w"], self.cam["h"]
        x_ndc = (2.0 * (u + 0.5) / w) - 1.0
        y_ndc = 1.0 - (2.0 * (v + 0.5) / h)
        z_ndc = 2.0 * depth_buf_value - 1.0

        ndc = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=np.float64)
        world = inv_vp @ ndc
        world /= world[3]
        return world[:3]

    def classify_color(self, mean_rgb):
        mean = np.array(mean_rgb, dtype=np.float32)
        best_name, best_d = None, 1e9
        for name, ref in COLORS_RGB255.items():
            d = np.linalg.norm(mean - ref)
            if d < best_d:
                best_d = d
                best_name = name
        return best_name, float(best_d)

    @staticmethod
    def decode_seg(seg_value: int):
        # seg_value is packed: (linkIndex+1)<<24 | objectUniqueId
        if seg_value < 0:
            return None, None
        obj_uid = seg_value & ((1 << 24) - 1)
        link_index = (seg_value >> 24) - 1
        return obj_uid, link_index

    def detect(self):
        rgb, depth_buf, seg, view, proj = self.render()

        uniq = np.unique(seg)
        uniq = uniq[uniq >= 0]

        dets = []
        for sv in uniq:
            mask = (seg == sv)
            cnt = int(mask.sum())
            if cnt < self.min_pixels:
                continue

            obj_uid, _ = self.decode_seg(int(sv))
            if obj_uid is None:
                continue

            ys, xs = np.nonzero(mask)
            v = int(np.mean(ys))
            u = int(np.mean(xs))

            mean_rgb = np.mean(rgb[ys, xs, :], axis=0)
            color_name, color_dist = self.classify_color(mean_rgb)

            depth_val = float(depth_buf[v, u])
            wxyz = self.unproject_pixel_to_world(u, v, depth_val, view, proj)
            xw, yw, zw = float(wxyz[0]), float(wxyz[1]), float(wxyz[2])

            if not in_pick_roi(xw, yw):
                continue

            y_exit = PICK_ROI["ymax"]
            t_exit = (y_exit - yw) / max(1e-6, CONVEYOR_SPEED)

            dets.append({
                "body_id": int(obj_uid),
                "x": xw, "y": yw, "z": zw,
                "color": color_name,
                "color_dist": color_dist,
                "t_exit": t_exit,
                "pix": cnt
            })
            if len(dets) >= self.max_objs:
                break

        dets.sort(key=lambda d: d["t_exit"])  # ưu tiên sắp ra khỏi ROI
        return dets


# ============================================================
# 10) PANDA ROBOT
# ============================================================

class PandaRobot:
    def __init__(self):
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        self.grasp_cid = None
        self.held_obj = None
        self.world: ConveyorWorld | None = None

        self.ll, self.ul, self.jr = [], [], []
        for j in ARM_JOINTS:
            info = p.getJointInfo(self.robot_id, j)
            low, high = info[8], info[9]
            self.ll.append(low)
            self.ul.append(high)
            rng = (high - low) if (high > low and abs(high - low) < 10) else 2 * math.pi
            self.jr.append(rng)

    def _step(self, n=1, realtime=False):
        for _ in range(n):
            if self.world is not None:
                self.world.update_conveyor(held_obj_id=self.held_obj)
            p.stepSimulation()
            if realtime:
                time.sleep(DT)

    def open_gripper(self):
        for j in GRIPPER_JOINTS:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=0.04, force=80)

    def gripper_q(self):
        return tuple(p.getJointState(self.robot_id, j)[0] for j in GRIPPER_JOINTS)

    def get_ee_pose(self):
        ls = p.getLinkState(self.robot_id, EE_LINK)
        return ls[4], ls[5]

    def ee_distance_to(self, target_pos):
        ee_pos, _ = self.get_ee_pose()
        return math.dist(ee_pos, target_pos)

    def get_arm_q(self):
        return [p.getJointState(self.robot_id, j)[0] for j in ARM_JOINTS]

    def ik(self, target_pos, target_orn, rest_poses=None):
        if rest_poses is None:
            rest_poses = self.get_arm_q()
        q = p.calculateInverseKinematics(
            self.robot_id,
            EE_LINK,
            targetPosition=target_pos,
            targetOrientation=target_orn,
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            restPoses=rest_poses,
            maxNumIterations=220,
            residualThreshold=1e-4
        )
        return [q[i] for i in range(7)]

    def goto_ee(self, pos, orn, steps=100, realtime=True, tol=0.01, settle=10, arm_force=420):
        q = self.ik(pos, orn)
        steps = max(10, int(steps * SPEED_SCALE))
        ok_count = 0
        for _ in range(steps):
            for i, j in enumerate(ARM_JOINTS):
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=q[i], force=arm_force)
            self._step(1, realtime=realtime)
            if self.ee_distance_to(pos) < tol:
                ok_count += 1
                if ok_count >= settle:
                    break
            else:
                ok_count = 0

    def goto_ee_linear(self, target_pos, orn, steps=60, realtime=True, arm_force=420):
        start_pos, _ = self.get_ee_pose()
        steps = max(10, int(steps * SPEED_SCALE))
        for k in range(steps):
            a = (k + 1) / steps
            pos = [
                start_pos[0] + a * (target_pos[0] - start_pos[0]),
                start_pos[1] + a * (target_pos[1] - start_pos[1]),
                start_pos[2] + a * (target_pos[2] - start_pos[2]),
            ]
            q = self.ik(pos, orn)
            for i, j in enumerate(ARM_JOINTS):
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=q[i], force=arm_force)
            self._step(1, realtime=realtime)

    def predict_obj_xy(self, obj_id, horizon_steps: int):
        (x, y, _), _ = p.getBasePositionAndOrientation(obj_id)
        y_pred = y + CONVEYOR_SPEED * DT * horizon_steps
        return x, y_pred

    def descend_track(self, obj_id, orn, z_start, z_end, steps=50, realtime=True, arm_force=450):
        steps = max(10, int(steps * SPEED_SCALE))
        for k in range(steps):
            alpha = (k + 1) / steps
            z = z_start + alpha * (z_end - z_start)

            remain = max(1, steps - (k + 1))
            horizon = min(PRED_TRACK_STEPS, remain)
            tx_obj, ty_obj = self.predict_obj_xy(obj_id, horizon)

            ee_pos, _ = self.get_ee_pose()
            tx = ee_pos[0] + TRACK_XY_GAIN * (tx_obj - ee_pos[0])
            ty = ee_pos[1] + TRACK_XY_GAIN * (ty_obj - ee_pos[1])

            q_arm = self.ik([tx, ty, z], orn)
            for i, j in enumerate(ARM_JOINTS):
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=q_arm[i], force=arm_force)

            self._step(1, realtime=realtime)

    def _contacts_lr(self, obj_id):
        cps_left = p.getContactPoints(bodyA=self.robot_id, bodyB=obj_id, linkIndexA=GRIPPER_JOINTS[0])
        cps_right = p.getContactPoints(bodyA=self.robot_id, bodyB=obj_id, linkIndexA=GRIPPER_JOINTS[1])
        return (len(cps_left), len(cps_right))

    def track_and_close_until_grasp(self, obj_id, orn, z_pick, max_loops=160, realtime=True):
        target = min(self.gripper_q())
        last_q = None
        stuck_count = 0

        for _ in range(max_loops):
            tx_obj, ty_obj = self.predict_obj_xy(obj_id, PRED_TRACK_STEPS)
            ee_pos, _ = self.get_ee_pose()
            tx = ee_pos[0] + TRACK_XY_GAIN * (tx_obj - ee_pos[0])
            ty = ee_pos[1] + TRACK_XY_GAIN * (ty_obj - ee_pos[1])

            q_arm = self.ik([tx, ty, z_pick], orn)
            for i, j in enumerate(ARM_JOINTS):
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=q_arm[i], force=470)

            target = max(0.0, target - GRIP_CLOSE_STEP)
            for j in GRIPPER_JOINTS:
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=target, force=GRIP_HOLD_FORCE)

            self._step(1, realtime=realtime)

            cl, cr = self._contacts_lr(obj_id)
            if REQUIRE_CONTACT and (cl > 0 or cr > 0):
                hold = min(self.gripper_q())
                for j in GRIPPER_JOINTS:
                    p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=hold, force=GRIP_HOLD_FORCE)
                return True, hold

            now_q = self.gripper_q()
            if last_q is not None:
                if abs(now_q[0] - last_q[0]) < GRIP_STUCK_EPS and abs(now_q[1] - last_q[1]) < GRIP_STUCK_EPS:
                    stuck_count += 1
                    if stuck_count >= GRIP_STUCK_COUNT:
                        hold = min(now_q)
                        for j in GRIPPER_JOINTS:
                            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=hold, force=GRIP_HOLD_FORCE)
                        return True, hold
                else:
                    stuck_count = 0
            last_q = now_q

        return False, min(self.gripper_q())

    def can_attach(self, obj_id):
        ee_pos, _ = self.get_ee_pose()
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
        dist = math.dist(ee_pos, obj_pos)

        if dist > ATTACH_MAX_EE_DIST:
            return False, dist
        cl, cr = self._contacts_lr(obj_id)
        if REQUIRE_CONTACT and (cl == 0 and cr == 0):
            return False, dist
        return True, dist

    def attach_no_snap(self, obj_id, max_force=ATTACH_MAX_FORCE):
        if self.grasp_cid is not None:
            return

        ee_pos, ee_orn = self.get_ee_pose()
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)

        inv_obj_pos, inv_obj_orn = p.invertTransform(obj_pos, obj_orn)
        child_pos, child_orn = p.multiplyTransforms(inv_obj_pos, inv_obj_orn, ee_pos, ee_orn)

        self.grasp_cid = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=EE_LINK,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFramePosition=child_pos,
            childFrameOrientation=child_orn
        )
        p.changeConstraint(self.grasp_cid, maxForce=max_force)
        self.held_obj = obj_id

    def detach(self):
        if self.grasp_cid is None:
            return
        try:
            p.removeConstraint(self.grasp_cid)
        except Exception:
            pass
        self.grasp_cid = None
        self.held_obj = None

    def reset_home(self):
        self.detach()
        self.open_gripper()
        for i, j in enumerate(ARM_JOINTS):
            p.resetJointState(self.robot_id, j, HOME_Q[i])
        for i, j in enumerate(ARM_JOINTS):
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=HOME_Q[i], force=420)
        self._step(MOVE_STEPS_HOME_SETTLE, realtime=REALTIME)

    def go_home_smooth(self, steps=220, realtime=True):
        self.detach()
        self.open_gripper()
        for i, j in enumerate(ARM_JOINTS):
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=HOME_Q[i], force=420)
        self._step(int(steps * SPEED_SCALE), realtime=realtime)


# ============================================================
# 11) PICK & PLACE (from camera detection)
# ============================================================

def fmt3(v):
    return f"({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})"


def pick_and_place_from_detection(robot: PandaRobot, det: dict, ee_orn, dbg: Debugger, pallet: Palletizer) -> bool:
    obj_id = det["body_id"]
    color_pred = det["color"]

    if (robot.world is None) or (not robot.world.is_valid_object(obj_id)):
        return False

    robot.world.pick_attempt += 1

    # ---------------- PICK ----------------
    xh, yh = robot.predict_obj_xy(obj_id, PRED_HOVER_STEPS)
    hover = [xh, yh, PICK_ROI["z_hover"]]
    dbg.line(hover, [xh, yh, PICK_ROI["z_pick"]], rgb=(1, 1, 0), width=3)

    robot.open_gripper()
    robot.goto_ee(hover, ee_orn, steps=MOVE_STEPS_HOVER, realtime=REALTIME, tol=EE_TOL_HOVER, settle=EE_SETTLE_STEPS)

    robot.descend_track(obj_id, ee_orn, PICK_ROI["z_hover"], PICK_ROI["z_pick"], steps=MOVE_STEPS_DESCEND, realtime=REALTIME)

    ok_grasp, _ = robot.track_and_close_until_grasp(obj_id, ee_orn, PICK_ROI["z_pick"], max_loops=180, realtime=REALTIME)
    if not ok_grasp:
        robot.world.grasp_fail += 1
        robot.open_gripper()
        robot._step(12, realtime=REALTIME)
        return False

    ok_attach, _ = robot.can_attach(obj_id)
    if USE_ATTACH and ok_attach:
        robot.attach_no_snap(obj_id)
    else:
        robot.world.grasp_fail += 1
        robot.open_gripper()
        robot._step(10, realtime=REALTIME)
        return False

    # lift
    ee_pos, _ = robot.get_ee_pose()
    robot.goto_ee([ee_pos[0], ee_pos[1], PICK_ROI["z_hover"]], ee_orn,
                 steps=MOVE_STEPS_LIFT, realtime=REALTIME, tol=EE_TOL_HOVER, settle=EE_SETTLE_STEPS)

    # lift confirm
    ee_pos2, _ = robot.get_ee_pose()
    obj_pos2, _ = p.getBasePositionAndOrientation(obj_id)
    if abs(obj_pos2[2] - ee_pos2[2]) > 0.10:
        robot.detach()
        robot.open_gripper()
        robot._step(10, realtime=REALTIME)
        robot.world.grasp_fail += 1
        return False

    # ---------------- PLACE ----------------
    if color_pred not in BINS:
        color_pred = "red"

    place, _, meta = pallet.next_place_pose(color_pred)

    # waypoint
    robot.goto_ee(list(WAYPOINT_MID), ee_orn, steps=int(MOVE_STEPS_TO_BIN * 0.65),
                 realtime=REALTIME, tol=0.02, settle=EE_SETTLE_STEPS)

    clear_above = [place[0], place[1], max(PICK_ROI["z_hover"], place[2] + PLACE_CLEAR_Z)]
    robot.goto_ee_linear(clear_above, ee_orn, steps=55, realtime=REALTIME, arm_force=420)

    pre_release = [place[0], place[1], place[2] + PLACE_PRE_RELEASE_Z]
    robot.goto_ee_linear(pre_release, ee_orn, steps=PLACE_LINEAR_STEPS_DOWN, realtime=REALTIME, arm_force=420)
    robot.goto_ee_linear(place, ee_orn, steps=max(18, int(PLACE_LINEAR_STEPS_DOWN * 0.55)), realtime=REALTIME, arm_force=420)

    # retract a bit then release
    robot.goto_ee_linear(pre_release, ee_orn, steps=18, realtime=REALTIME, arm_force=420)

    robot.detach()
    robot._step(2, realtime=REALTIME)
    robot.open_gripper()
    robot._step(12, realtime=REALTIME)

    # stabilize
    p.resetBaseVelocity(obj_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    p.changeDynamics(
        obj_id, -1,
        lateralFriction=1.2, rollingFriction=0.02, spinningFriction=0.02,
        linearDamping=0.7, angularDamping=0.9
    )

    robot.world.mark_sorted(obj_id)
    pallet.commit(color_pred)

    retract = [place[0], place[1], max(clear_above[2], place[2] + PLACE_RETRACT_Z)]
    robot.goto_ee_linear(retract, ee_orn, steps=PLACE_LINEAR_STEPS_UP, realtime=REALTIME, arm_force=420)

    robot.world.pick_success += 1
    robot.world.place_success += 1

    # pallet reset when full 4 layers
    if pallet.is_full_4_layers(color_pred):
        ox, oy, _ = BINS[color_pred]
        remove_ids = []
        for o2 in list(robot.world.objects):
            if o2.sorted:
                (x2, y2, _), _ = p.getBasePositionAndOrientation(o2.body_id)
                if (abs(x2 - ox) < 0.5) and (abs(y2 - oy) < 0.5):
                    remove_ids.append(o2.body_id)
        for bid in remove_ids:
            try:
                p.removeBody(bid)
            except Exception:
                pass
        robot.world.objects = [oo for oo in robot.world.objects if oo.body_id not in set(remove_ids)]
        pallet.reset_color(color_pred)
        print(f"[PALLET_RESET] color={color_pred} cleared={len(remove_ids)}")

    return True


# ============================================================
# 12) MAIN
# ============================================================

def main():
    global EE_DOWN_ORN

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)

    EE_DOWN_ORN = ee_down_yaw(GRIP_YAW)

    p.loadURDF("plane.urdf")

    draw_world_grid()

    p.resetDebugVisualizerCamera(
        cameraDistance=1.25,
        cameraYaw=55,
        cameraPitch=-35,
        cameraTargetPosition=[0.42, 0.2, 0.15]
    )

    dbg = Debugger()
    dbg.draw_roi_and_bins()

    world = ConveyorWorld(dbg)
    robot = PandaRobot()
    robot.world = world
    pallet = Palletizer()
    cam = FixedCameraPerception()

    robot.reset_home()

    sim_step = 0
    busy = False
    last_action_status = "idle"

    last_dets = []

    try:
        while True:
            sim_step += 1

            if not busy:
                world.spawn_if_needed()

                # ✅ detect theo tần suất + cache để nhanh
                if sim_step % DETECT_EVERY == 0:
                    last_dets = cam.detect()
                dets = last_dets
            else:
                dets = []

            if not busy:
                target = None
                for d in dets:
                    if world.is_valid_object(d["body_id"]):
                        target = d
                        break

                if target is not None:
                    busy = True
                    ok = pick_and_place_from_detection(robot, target, EE_DOWN_ORN, dbg, pallet)
                    last_action_status = "OK" if ok else "FAIL"
                    robot.go_home_smooth(steps=220, realtime=REALTIME)
                    busy = False
                # else: no detection => WAIT

            if sim_step % DEBUG_PRINT_EVERY == 0:
                ee_pos, _ = robot.get_ee_pose()
                gq = robot.gripper_q()
                det_str = "None"
                if last_dets:
                    d0 = last_dets[0]
                    det_str = f"id={d0['body_id']} xy=({d0['x']:.3f},{d0['y']:.3f}) color={d0['color']} t_exit={d0['t_exit']:.2f}s pix={d0['pix']}"
                print(
                    f"[step={sim_step}] sim_time={world.sim_time:.2f} objs={len(world.objects)} busy={busy} held={robot.held_obj} "
                    f"ee={fmt3(ee_pos)} gripper=({gq[0]:.4f},{gq[1]:.4f}) det={det_str} status={last_action_status}"
                )
                print(
                    f"metrics: spawn={world.spawn_count} miss={world.miss_count} despawn={world.despawn_count} "
                    f"attempt={world.pick_attempt} pick_ok={world.pick_success} grasp_fail={world.grasp_fail} place_ok={world.place_success}"
                )

                dbg.overlay("A", f"step: {sim_step} | sim_time: {world.sim_time:.2f}")
                dbg.overlay("B", f"objects: {len(world.objects)} | busy: {busy}")
                dbg.overlay("C", f"held_obj: {robot.held_obj}")
                dbg.overlay("D", f"ee: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
                dbg.overlay("E", f"gripper: ({gq[0]:.4f},{gq[1]:.4f})")
                dbg.overlay("F", f"dets_cached={len(last_dets)} miss={world.miss_count} DETECT_EVERY={DETECT_EVERY} REALTIME={REALTIME}")
                dbg.overlay("G", f"last: {last_action_status}",
                            color=(0, 1, 0) if last_action_status == "OK" else (1, 0.2, 0.2))

            robot._step(1, realtime=REALTIME)

    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()