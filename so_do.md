# source D:/Picking-For-Robot/rl_robot/Scripts/activate

# 🧠 SƠ ĐỒ HUẤN LUYỆN CUỐI CÙNG

## 🎯 Mục tiêu cuối
Robot thực hiện pipeline:
```
reach → grasp (pick + lift) → place → sort màu → conveyor → pallet
```

---

# 🟢 GIAI ĐOẠN 1 — CORE SKILLS (bắt buộc)

## 🟢 Stage 1 — Reach

### Học gì
- Xác định vị trí object  
- Điều khiển end-effector (EE) tới hover position  

### Done khi dist(EE, object) < threshold


### Curriculum
- **1A:** object cố định  
- **1B:** random trong ROI  
- **1C:** thêm noise nhẹ (camera / ánh sáng)  

---

## 🟡 Stage 2 — Grasp (Pick + Lift) ⭐

### Học gì
- Descend đúng  
- Close gripper đúng timing  
- Attach object  
- 🔥 Lift object  

### Done khi object_z > LIFT_HEIGHT


### Curriculum
- **2A:** object fixed  
- **2B:** random vị trí  
- **2C:** giảm attach distance (khó dần)  

---

## 🔵 Stage 3 — Place

### Học gì
- Mang object tới target  
- Mở gripper đúng lúc  

### Done khi object nằm trong drop zone


### Curriculum
- **3A:** drop zone lớn  
- **3B:** nhỏ dần  
- **3C:** thả đúng độ cao (tránh ném)  

---

## ❌ Stage 4 — Pick+Place

👉 **Không train**

### Chỉ dùng cho
- Evaluation  
- Debug  
- Test pipeline  

---

# 🟡 GIAI ĐOẠN 2 — TASK LEARNING

## 🟡 Stage 5 — Sorting (color → bin)

### Học gì
- Phân biệt màu  
- Chọn đúng bin  

### Done khi đặt đúng theo màu bin


### Curriculum
- **5A:** 2 màu  
- **5B:** 4 màu  
- **5C:** lighting variation  

⚠️ **Lưu ý:** luôn random vị trí để tránh model học “vị trí” thay vì “màu”

---

# 🟣 GIAI ĐOẠN 3 — ROBUSTNESS

## 🟣 Stage 6 — Domain Randomization

### Học gì
- Chịu được:
  - Noise ảnh  
  - Lighting  
  - Physics variation  

### Curriculum
- **6A → 6C:** tăng dần độ khó  

---

# 🟠 GIAI ĐOẠN 4 — TRANSITION

## 🟠 Stage 6.5 — Fake Tracking ⭐

### Học gì
- Học “đuổi target” trước khi có conveyor  

### Cách làm
- Object đứng yên  
- Target = predicted future position  

---

# 🔴 GIAI ĐOẠN 5 — DYNAMIC WORLD

## 🔴 Stage 7 — Conveyor (1 object)

### Học gì
- Tracking object di chuyển  
- Timing để grasp  

### Curriculum
- **7A:** rất chậm  
- **7B:** tăng tốc  
- **7C:** spawn lệch  

---

# 🔴🔴 GIAI ĐOẠN 6 — SCALING

## 🔴 Stage 8 — Multi-object (throughput)

### Học gì
- Chọn object tối ưu  
- Tăng tốc độ xử lý  

### Reward objects_processed / time


---

# 🟤 GIAI ĐOẠN 7 — PALLET STACKING

## 🟤 Stage 9 — Pallet stacking

### Học gì
- Đặt đúng cell 4×4  
- Xếp tầng  
- Tránh va chạm  

### Curriculum
- **9A:** 1 tầng  
- **9B:** 2 tầng  
- **9C:** full 4 tầng  

---

# 🧠 LUỒNG TRAIN

```
Stage 1 → Stage 2 → Stage 3
            ↓
            Stage 5
            ↓
            Stage 6
            ↓
            Stage 6.5
            ↓
            Stage 7
            ↓
            Stage 8
            ↓
            Stage 9
```
