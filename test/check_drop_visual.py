# test/check_drop_visual.py
import sys
import os
import time
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env.place_env import PlaceEnv

def main():
    print("="*70)
    print("CHẾ ĐỘ KIỂM TRA ĐỘ CHÍNH XÁC KHI THẢ VẬT (SLOW MOTION CAMERA)")
    print("="*70)

    # Khởi tạo môi trường
    env = PlaceEnv(use_gui=True, start_held=True)
    obs, info = env.reset()
    
    # Zoom camera sát vào target để nhìn cực rõ tay kẹp và vật
    target_pos = info["target_pos"]
    p.resetDebugVisualizerCamera(
        cameraDistance=0.35,  # Gần sát vào
        cameraYaw=45, 
        cameraPitch=-25, 
        cameraTargetPosition=target_pos,
        physicsClientId=env.cid
    )

    # Load model (Thay đường dẫn nếu Boss muốn test model cụ thể)
    model_path = "models/stage3_place_3A_latest.zip"
    if os.path.exists(model_path):
        from stable_baselines3 import PPO
        print(f"[INFO] Đã load model: {model_path}\n")
        model = PPO.load(model_path)
    else:
        print(f"[WARN] Không tìm thấy {model_path}. Chạy KỊCH BẢN TỰ ĐỘNG để Boss test góc nhìn Camera!\n")
        model = None

    step = 0
    try:
        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = action[0] if len(action.shape) > 1 else action
            else:
                # Chạy kịch bản di chuyển tự động (giống file place_env.py) để Boss ngắm mô phỏng
                ee_pos = np.array(info["ee_pos"])
                obj_pos_temp, _ = p.getBasePositionAndOrientation(env.object_id, physicsClientId=env.cid)
                bottom_z = obj_pos_temp[2] - 0.025 # Chiều cao khối hộp 5cm -> khoảng cách từ tâm đến đáy là 2.5cm
                
                dx, dy, dz, grip = 0.0, 0.0, 0.0, 1.0
                target_top_z = target_pos[2] + 0.025 # Mặt trên của hộp đích màu đỏ
                # Đặt mục tiêu thấp hơn 5mm để tay kẹp có đà đi xuống chạm hẳn mặt đích
                dest_lower = np.array([target_pos[0], target_pos[1], target_top_z + 0.020])
                diff = dest_lower - ee_pos
                
                if step < 40:
                    # Bay đến trên target
                    dest_hover = np.array([target_pos[0], target_pos[1], target_top_z + 0.15])
                    dx, dy, dz = (dest_hover - ee_pos) * 12.0
                elif bottom_z > target_top_z + 0.002 and step < 80: 
                    # Khom tay hạ thấp tới khi đáy vật vừa chạm sát mặt đỏ (sai số 2mm)
                    dx, dy, dz = diff * 6.0
                else:
                    grip = 0.0 # RA LỆNH THẢ VẬT
                action = np.array([dx, dy, dz, grip], dtype=np.float32)
                
            # Lệnh Grip AI gửi (<= 0.5 là mở, > 0.5 là đóng)
            grip_cmd = action[3]
            
            # --- KIỂM TRA TRẠNG THÁI THỰC TẾ TRONG PYBULLET ---
            # Độ mở của 1 ngón kẹp (mở tối đa là ~0.04m, đóng kín là 0.0m)
            grip_width = p.getJointState(env.robot, env.ctrl.GRIPPER_JOINTS[0], physicsClientId=env.cid)[0]
            obj_pos, _ = p.getBasePositionAndOrientation(env.object_id, physicsClientId=env.cid)
            
            # Tính khoảng cách theo XY và Z
            dist_to_target = np.linalg.norm(np.array(obj_pos) - np.array(target_pos))
            
            if grip_cmd <= 0.5:
                current_bottom_z = obj_pos[2] - 0.025
                print(f"[Step {step:03d}] 🔴 AI RA LỆNH NHẢ KẸP! | Z Đáy vật: {current_bottom_z*100:.1f}cm | Z Mặt đích: {(target_pos[2]+0.025)*100:.1f}cm | Sai số: {dist_to_target:.3f}m")
                time.sleep(0.08) # Chuyển sang SLOW MOTION để xem vật rớt
            else:
                if step % 10 == 0:
                    print(f"[Step {step:03d}] 🟢 Đang bay...        | Độ hở ngón tay: {grip_width*100:.1f}cm")
                time.sleep(1/30) # Tốc độ bình thường
                
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            
            if done or truncated:
                print(f"\n>>> KẾT THÚC EPISODE! Khoảng cách chung cuộc: {info.get('obj_target_dist', dist_to_target):.3f}m\n")
                time.sleep(1.5) # Dừng 1.5s cho Boss ngắm kết quả rồi mới reset
                obs, info = env.reset()
                step = 0
                
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    main()
