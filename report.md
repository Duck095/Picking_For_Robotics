<h1 align="center"> 🤖 Hệ thống robot gắp – thả và phân loại vật thể trên băng chuyền </h1>

# 🚀 CHƯƠNG 1. TỔNG QUAN DỰ ÁN

## 🧭 1.1. Bối cảnh và ý nghĩa

Trong các dây chuyền sản xuất hiện đại, bài toán **gắp – thả và phân loại vật thể** xuất hiện rất phổ biến (phân loại theo màu, kích thước, mã vạch, loại sản phẩm…). Một tình huống điển hình là các vật thể được đặt lên **băng chuyền di chuyển liên tục**, robot cần nhận biết vật thể và đưa về đúng vị trí tập kết.

Trong dự án này, mô hình được thiết kế nhằm giải quyết bài toán cốt lõi của công nghiệp:
- 📦 Băng chuyền chạy với tốc độ cố định.
- 🎲 Vật thể xuất hiện ngẫu nhiên theo thời gian.
- 🤖 Robot Franka Panda gắp vật trong một vùng cho phép và phân loại thả vào đúng thùng theo màu sắc.

Việc xây dựng hệ thống mô phỏng này mang lại ý nghĩa quan trọng:
- ⚡ Cho phép thử nghiệm nhanh các thuật toán điều khiển và **Học tăng cường (Reinforcement Learning - RL)**.
- 🔧 Dễ dàng thay đổi các tham số (tốc độ băng chuyền, mật độ sinh vật thể, vị trí thùng…).
- 🛡️ Đảm bảo an toàn và tối ưu hóa chi phí trước khi triển khai lên robot thực tế.

---

## 🎯 1.2. Mục tiêu dự án

Xây dựng một hệ thống mô phỏng robot gắp – thả, phân loại vật thể theo màu sắc trong môi trường **PyBullet** và chuẩn bị nền tảng hoàn chỉnh để huấn luyện mô hình **Học tăng cường (RL)** cho bài toán này.

### 🧩 Mục tiêu cụ thể

1. 🏗️ **Xây dựng môi trường mô phỏng:** Bao gồm không gian làm việc, robot Franka Panda (điều khiển bằng Inverse Kinematics), băng chuyền và các vật thể.
2. 🛤️ **Mô phỏng băng chuyền:** Đảm bảo băng chuyền vận hành với tốc độ ổn định và các vật thể di chuyển theo thời gian thực.
3. 🎨 **Sinh và quản lý vật thể:** Sinh vật thể ngẫu nhiên trên băng tải (cùng kích thước, khác màu) và quản lý vòng đời của chúng.
4. 🧮 **Điều khiển Robot (Baseline Rule-based):** Sử dụng động học ngược (IK) để thực hiện mượt mà chu trình gắp – thả, giải quyết các rủi ro như điểm kỳ dị (singularity) của tay máy.
5. 🗂️ **Phân loại vật thể:** Đưa vật thể đã gắp vào đúng thùng chứa tương ứng với màu sắc.
6. 🛠️ **Xây dựng công cụ Debug:** Cung cấp hệ thống hiển thị trực quan (overlay text, vẽ đường, lưới tọa độ) để hỗ trợ quá trình phát triển, theo dõi và mở rộng.

---

## 📐 1.3. Phạm vi và giả định của bài toán

Để tập trung vào thuật toán điều khiển và học tăng cường, dự án sử dụng các giả định sau:
- 📦 **Vật thể:** Có kích thước giống nhau (khối hộp), chỉ khác nhau về màu sắc.
- 👁️ **Nhận thức (Perception):** Sử dụng *ground-truth* (tọa độ không gian 3D trực tiếp từ mô phỏng) để lấy vị trí và màu sắc vật thể thay vì dùng xử lý ảnh (Computer Vision).
- 📍 **Không gian gắp (ROI):** Robot chỉ được phép gắp vật thể khi chúng đi vào một vùng không gian làm việc an toàn đã được giới hạn trước.
- 🧪 Giai đoạn hiện tại sử dụng baseline rule-based, chưa áp dụng học tăng cường.

---

## 🔄 1.4. Pipeline vận hành tổng quát

Hệ thống vận hành theo một luồng xử lý khép kín và liên tục như sau:

```text
🟢 Khởi tạo mô phỏng (PyBullet, Robot, Camera)
        ↓
📦 Sinh vật thể ngẫu nhiên trên băng chuyền di chuyển
        ↓
📍 Kiểm tra vùng gắp (ROI) & Lựa chọn vật thể tối ưu nhất
        ↓
🧮 Điều khiển tay máy tiếp cận vật thể (Hover & Descend)
        ↓
🤏 Đóng kẹp và gắp vật thể (Attach)
        ↓
🗂️ Di chuyển tới tọa độ thùng theo màu sắc của vật thể
        ↓
📥 Thả vật (Detach) & Định hình lại vật trong pallet
        ↓
🏠 Quay về tư thế chờ (Home) và lặp lại chu trình
```

# 🏗️ CHƯƠNG 2. THIẾT KẾ TỔNG THỂ HỆ THỐNG

## 🧩 2.1. Kiến trúc module
Hệ thống được chia thành bốn khối chính:

1. 🌍 **Environment & Conveyor**  
   - 🎲 Sinh vật thể  
   - ➡️ Cập nhật vận tốc băng chuyền  
   - ❌ Xoá vật ra khỏi vùng làm việc  

2. 🤖 **Robot Controller**  
   - 🧮 Inverse Kinematics  
   - 🎛️ Điều khiển khớp  
   - 🤏 Điều khiển kẹp  
   - 🔗 Gắn / tháo vật  

3. 🔄 **Task Logic (Pick & Place)**  
   - 📍 Chọn vật trong ROI  
   - ♻️ Chu trình gắp – thả  
   - 🧭 Waypoint trung gian  
   - 🗂️ Chọn thùng theo màu  

4. 🛠️ **Debug & Visualization**  
   - 📐 Vẽ lưới tọa độ  
   - 🧱 Hiển thị ROI và thùng  
   - 📊 Overlay thông tin trạng thái  

---

## 🔀 2.2. Sơ đồ luồng xử lý hệ thống

```text
        +---------------------+
        | 🌍 ConveyorWorld   |
        | (spawn, move)      |
        +----------+----------+
                   |
                   v
        +---------------------+
        | 📍 ROI Filtering   |
        | Candidate Select   |
        +----------+----------+
                   |
                   v
        +---------------------+
        | 🤖 PandaRobot      |
        | (IK + control)     |
        +----------+----------+
                   |
                   v
        +---------------------+
        | 🔄 Pick & Place    |
        +----------+----------+
                   |
                   v
        +---------------------+
        | 🛠️ Debug & Visual  |
        +---------------------+
```

---

## 🛤️ 2.3. Thiết kế môi trường và băng chuyền

### ⏱️ Thời gian mô phỏng
- ⏲️ Bước thời gian cố định: `DT = 1/240` giây  
- 🔁 Mỗi vòng lặp gọi `stepSimulation()` để cập nhật trạng thái  

---

### 🎲 Sinh vật thể ngẫu nhiên
- 🎯 Vật thể được sinh theo chu kỳ `SPAWN_INTERVAL`  
- ↔️ Trục X được lấy ngẫu nhiên trong ROI  
- ⬆️ Trục Y cố định tại vị trí spawn  
- 📏 Cơ chế giãn cách theo Y (`MIN_SPAWN_DY`) để tránh chồng vật  

---

### ➡️ Mô phỏng chuyển động băng chuyền
- 🛤️ Băng chuyền được mô phỏng bằng vận tốc tuyến tính theo trục Y  
- ⚡ Tốc độ cố định `CONVEYOR_SPEED`  
- 🗑️ Vật vượt quá `DESPAWN_Y` sẽ bị xoá khỏi mô phỏng  

---

## 📍 2.4. Vùng gắp (ROI) và chiến lược chọn vật
- 🟦 ROI là hình chữ nhật giới hạn không gian gắp an toàn  
- 🤖 Robot chỉ gắp vật nằm trong ROI  
- 🎯 Nếu có nhiều vật, chọn vật gần tâm ROI nhất  

Chiến lược này giúp:
- 🔍 Giảm không gian tìm kiếm  
- 🦾 Tránh các cấu hình robot khó hoặc gần suy biến  
- 📈 Tăng độ ổn định của quá trình gắp  

---

## 🗂️ 2.5. Thiết kế thùng (Bins) và waypoint
- 🎨 Mỗi màu có một thùng tương ứng  
- 🧱 Thùng được đặt trong workspace an toàn  
- 🧭 Waypoint trung gian được sử dụng khi di chuyển tới thùng xa  
- ⬆️ Một số thùng có vị trí hover cao hơn để tăng độ ổn định  

---

## 🛠️ 2.6. Trực quan hóa và debug
- 📐 Vẽ lưới tọa độ trên mặt phẳng làm việc  
- 🏷️ Hiển thị ROI và nhãn thùng  
- 📊 In trạng thái robot, vật thể và end-effector    

---

# 🤖 CHƯƠNG 3. PHƯƠNG PHÁP ĐIỀU KHIỂN ROBOT

## 🦾 3.1. Robot Franka Panda
- 🤖 Robot có **7 bậc tự do**  
- 🤏 Trang bị **2 khớp kẹp**  
- 🎯 End-effector được xác định bằng **chỉ số link** trong mô hình  

---

## 🧮 3.2. Điều khiển bằng Inverse Kinematics
- 📐 Điều khiển robot theo **không gian làm việc** `(x, y, z)`  
- 🧠 Tính toán **góc khớp** thông qua Inverse Kinematics  
- 🎛️ Sau đó điều khiển từng khớp bằng **Position Control**  

Cách tiếp cận này giúp:
- 🎯 Điều khiển trực quan theo vị trí  
- 🧩 Giảm độ phức tạp khi lập trình  
- 🔄 Dễ mở rộng sang các chiến lược điều khiển cao cấp hơn  

---

## 🧠 3.3. Nullspace IK và rest pose
- 🔒 Cung cấp **giới hạn góc khớp** để tránh cấu hình không hợp lệ  
- 🏠 Sử dụng **rest pose** làm tư thế home mặc định  
- ⚠️ Tránh cấu hình tay duỗi thẳng (**singularity**)  

Nullspace IK giúp:
- 🦾 Robot duy trì cấu hình tay ổn định  
- 📉 Hạn chế dao động và rung lắc  
- 🔁 Dễ kiểm soát quỹ đạo trong không gian hẹp  

---

## 🌀 3.4. Điều khiển chuyển động mượt
- 🔢 Chia chuyển động thành **nhiều bước nhỏ**  
- 🎚️ Các tham số `MOVE_STEPS_*` dùng để điều chỉnh tốc độ  
- ⚡ `SPEED_SCALE` dùng để thay đổi tốc độ tổng thể của robot  

Cơ chế này giúp:
- 🚫 Tránh hiện tượng teleport  
- 🎢 Chuyển động mượt và tự nhiên  
- 🛡️ Tăng độ an toàn trong quá trình gắp  

---

## 🔄 3.5. Chu trình gắp – thả (Pick & Place)

```text
🟦 Hover
        ↓
🔄 Resample vị trí vật
        ↓
⬇️ Descend
        ↓
🤏 Grasp
        ↓
⬆️ Lift
        ↓
🧭 Waypoint
        ↓
📦 Move to bin
        ↓
📥 Place
        ↓
↩️ Retreat
        ↓
🏠 Return home

```

---

## 🔗 3.6. Gắn và thả vật trong mô phỏng
- 🔗 Vật thể được **gắn** bằng constraint cố định giữa **end-effector (EE)** và vật thể  
- ❌ Khi thả, constraint được **xoá** để giải phóng vật  
- 🧲 Sau khi thả, vận tốc vật thể được đặt về `0` để vật đứng yên trong thùng  

Cơ chế này giúp:
- 🤝 Mô phỏng hành vi gắp **chắc chắn và ổn định**  
- 📉 Tránh hiện tượng **rơi vật** trong quá trình di chuyển  
- 🧪 Đơn giản hoá việc xử lý **va chạm và tiếp xúc**  

---

## 🧭 3.7. Quản lý trạng thái hệ thống
- 🚦 Sử dụng cờ `busy` để phân biệt robot **đang rảnh** hay **đang thực hiện gắp – thả**  
- 🔁 Đảm bảo luồng điều khiển **không bị xung đột** giữa các hành động  
- 🧩 Chuẩn bị nền tảng tốt cho việc **chuyển sang môi trường học tăng cường (RL)**  

Cách quản lý trạng thái này giúp:
- 🛡️ Hệ thống vận hành **ổn định và nhất quán**  
- 🧠 Tránh các lỗi logic do lệnh điều khiển chồng chéo  
- 🚀 Dễ dàng mở rộng sang các chiến lược điều khiển thông minh hơn  


# 📊 CHƯƠNG 4. TRIỂN KHAI VÀ KẾT QUẢ HỆ THỐNG BASELINE

## 🎯 4.1. Mục tiêu của giai đoạn triển khai

Chương này trình bày chi tiết cách hệ thống gắp – thả **baseline rule-based** được triển khai và vận hành trong môi trường mô phỏng, đồng thời đánh giá các kết quả đạt được.

Mục tiêu chính bao gồm:
- 🤖 Mô tả cách robot **xác định vật thể**, **lựa chọn vật để gắp** và **ra quyết định hành động**
- 🔄 Trình bày chi tiết **chu trình di chuyển, gắp và thả vật**
- 📈 Đánh giá **tính ổn định và độ tin cậy** của hệ thống
- ⚠️ Xác định **hạn chế** của phương pháp baseline để làm cơ sở cho học tăng cường  

---

## 🧪 4.2. Thiết lập thực nghiệm

### ⚙️ Cấu hình mô phỏng
Hệ thống được triển khai trong môi trường PyBullet với các thiết lập sau:

- ⏱️ Bước thời gian mô phỏng: `DT = 1/240` giây  
- 🌍 Trọng lực: `(0, 0, -9.81)`  
- 🤖 Robot sử dụng: Franka Panda (base cố định)  
- 🧱 Mặt phẳng làm việc: `plane.urdf`  

Toàn bộ mô phỏng được cập nhật theo **thời gian thực**, cho phép quan sát trực quan quá trình robot thao tác.

---

### 🛤️ Cấu hình băng chuyền và vật thể

Băng chuyền được mô phỏng bằng cách áp **vận tốc tuyến tính không đổi** cho các vật thể theo trục Y:

- ⚡ Tốc độ băng chuyền: `CONVEYOR_SPEED = 0.30 m/s`  
- ⏲️ Chu kỳ sinh vật thể: `SPAWN_INTERVAL = 1.20 s`  
- 📍 Vị trí sinh vật thể: `SPAWN_Y = -0.60`  
- 🗑️ Ngưỡng loại bỏ vật thể: `DESPAWN_Y = 0.90`  

Mỗi vật thể:
- 📦 Có hình dạng khối hộp với kích thước cố định  
- 🎨 Được gán màu sắc ngẫu nhiên (đỏ, xanh lá, xanh dương, vàng)  
- 🔁 Được quản lý vòng đời rõ ràng (chưa phân loại / đã phân loại)  

Cơ chế giãn cách theo trục Y (`MIN_SPAWN_DY`) giúp:
- 🚫 Tránh hiện tượng chồng lấn vật thể  
- 📉 Giảm va chạm không mong muốn  

---

## 🔍 4.3. Cơ chế xác định và lựa chọn vật thể

### 📌 Xác định vật thể trong môi trường
Tại mỗi vòng lặp, hệ thống duyệt qua danh sách các vật thể đang tồn tại:

1. 📍 Lấy vị trí `(x, y, z)` từ mô phỏng  
2. 🏷️ Kiểm tra trạng thái vật thể (đã phân loại hay chưa)  
3. ⏭️ Bỏ qua các vật thể đã được thả vào thùng  

Việc xác định vị trí và trạng thái sử dụng **ground-truth trực tiếp từ mô phỏng**, đảm bảo độ chính xác tuyệt đối trong giai đoạn baseline.

---

### 🟦 Vùng gắp (ROI)

Hệ thống định nghĩa một vùng gắp (Region of Interest – ROI) hình chữ nhật trên mặt phẳng làm việc.

Một vật thể chỉ được xem là ứng viên gắp nếu:
- `xmin ≤ x ≤ xmax`
- `ymin ≤ y ≤ ymax`

Lợi ích:
- 🔍 Giảm không gian tìm kiếm  
- 🦾 Tránh cấu hình robot khó hoặc gần suy biến  
- 📈 Tăng độ ổn định của quá trình gắp  

---

### 🎯 Chiến lược chọn vật thể để gắp

Khi có nhiều vật thể nằm trong ROI, hệ thống áp dụng chiến lược:

- 📐 Tính tâm ROI  
- 📏 Tính khoảng cách từ mỗi vật thể tới tâm ROI  
- 🥇 Chọn vật thể có khoảng cách nhỏ nhất  

Chiến lược này giúp robot:
- 🎯 Ưu tiên vị trí gắp an toàn  
- 🚶 Giảm quãng đường di chuyển  
- 🛡️ Hạn chế rủi ro suy biến  

---

## 🧭 4.4. Cơ chế điều hướng và tiếp cận vật thể

### 🪂 Tiếp cận theo hai pha (Hover – Descend)

Robot tiếp cận vật thể theo hai pha:

1. 🟦 **Hover**: di chuyển end-effector tới vị trí phía trên vật (`z_hover`)  
2. ⬇️ **Descend**: hạ end-effector xuống vị trí gắp (`z_pick`)  

Do vật thể chuyển động trên băng chuyền, ngay trước pha hạ xuống, hệ thống **resample lại vị trí vật thể** để giảm sai lệch.

---

### 🎛️  khiển chuyển động

Robot được điều khiển bằng:
- 🧮 Inverse Kinematics để tính góc khớp  
- 🎚️ Position Control với nhiều bước nhỏ  

Các tham số `MOVE_STEPS_*` và `SPEED_SCALE` cho phép:
- ⚖️ Điều chỉnh tốc độ  
- 🎢 Đảm bảo chuyển động mượt  

---

## 🤏 4.5. Cơ chế gắp vật thể

Khi end-effector đạt vị trí gắp:

1. 🤏 Robot đóng kẹp  
2. 🔗 Tạo constraint cố định giữa EE và vật  
3. ⬆️ Nâng vật khỏi băng chuyền  

Cách làm này giúp:
- 🤝 Gắp vật chắc chắn  
- 📉 Tránh rơi vật  
- 🛡️ Đảm bảo ổn định khi di chuyển  

---

## 🗂️ 4.6. Cơ chế phân loại và thả vật thể

### 🎨 Xác định vị trí thả
- 🧠 Thùng thả được xác định dựa trên **màu sắc vật thể**  
- 🧭 Sử dụng waypoint trung gian cho thùng xa  

---

### 📥 Thả và cố định vật thể

Quá trình thả vật:
```text
1. 🟦 Hover trên thùng
2. ⬇️ Hạ EE xuống vị trí thả
3. ❌ Gỡ constraint
4. 🤏 Mở kẹp
5. 🧲 Đặt vận tốc vật = 0

```
Sau khi thả, vật thể được đánh dấu là **đã phân loại**, không còn chịu ảnh hưởng của băng chuyền.

---

## 🔁 4.7. Chu trình hoàn chỉnh gắp – thả

Toàn bộ chu trình xử lý một vật thể có thể tóm tắt như sau:

```
📍 Xác định vật trong ROI
        ↓
🎯 Chọn vật gần tâm ROI
        ↓
🟦 Hover trên vật
        ↓
🔄 Resample vị trí vật
        ↓
⬇️ Descend và gắp
        ↓
⬆️ Nâng vật
        ↓
🧭 Waypoint trung gian
        ↓
📦 Di chuyển tới thùng
        ↓
📥 Thả vật
        ↓
🏠 Quay về tư thế home

```

Chu trình này được lặp lại liên tục cho các vật thể xuất hiện trên băng chuyền.

---

## 📈 4.8. Kết quả quan sát và đánh giá

### ✅ Kết quả đạt được
Qua thực nghiệm, hệ thống cho thấy:

✔️ Robot gắp và thả đúng vật thể theo màu
✔️ Chuyển động mượt, không teleport
✔️ Không xảy ra hiện tượng rơi vật
✔️ Vật thể sau khi phân loại không bị băng chuyền kéo đi
✔️ Hệ thống chạy ổn định qua nhiều chu kỳ liên tiếp

---

### 🛡️ Đánh giá tính ổn định
Tính ổn định của hệ thống đạt được nhờ:

🧠 Sử dụng Nullspace IK với rest pose
🔢 Chia chuyển động thành nhiều bước nhỏ
🧭 Sử dụng waypoint trung gian
🚦 Quản lý trạng thái hệ thống rõ ràng

---

## ⚠️ 4.9. Hạn chế của phương pháp baseline

Mặc dù hệ thống hoạt động ổn định, phương pháp baseline vẫn tồn tại các hạn chế:

📏 Hành vi robot hoàn toàn dựa trên luật cố định
🔄 Không có khả năng thích nghi khi môi trường thay đổi
🎯 Không tối ưu chiến lược gắp – thả theo thời gian
🧩 Khó mở rộng sang các tình huống phức tạp hơn

---

## 🌟 4.10. Cải tiến hệ thống: Bám sát động và Xếp Pallet (Advanced Baseline - demo5.py)

Để giải quyết các hạn chế cơ bản và mô phỏng sát hơn với yêu cầu trong công nghiệp, một phiên bản baseline nâng cao (`demo5.py`) đã được phát triển với các cơ chế đột phá:

### 🎯 Dự đoán quỹ đạo và bám sát mục tiêu (Dynamic Tracking)
Thay vì chờ vật thể di chuyển đến một điểm cố định, robot nay có khả năng **vừa di chuyển vừa dự đoán**:
- 🧮 **Tính toán trước (Prediction):** Sử dụng vận tốc băng chuyền để tính toán vị trí của vật thể trong tương lai (`predict_obj_xy`).
- 🦅 **Bám sát (Tracking):** Vừa hạ thấp (descend) vừa tinh chỉnh liên tục tọa độ X, Y theo vật thể.
- 🤏 **Đóng kẹp động:** Quá trình đóng ngón kẹp diễn ra từ từ ngay trong lúc tracking thay vì đợi robot dừng hẳn, giúp gắp mượt mà và tiết kiệm thời gian.

### 📦 Thuật toán xếp lớp (Palletizer)
Thay vì thả rơi tự do vào thùng (Bin), hệ thống đã tích hợp module **Palletizer** để xếp gọn gàng vật thể:
- 📏 Hỗ trợ xếp thành **lưới 4x4** và có thể **xếp chồng nhiều tầng**.
- 🧠 **Chiến lược an toàn:** Thuật toán tự động ưu tiên xếp từ hàng xa nhất đến hàng gần nhất, giúp robot không phải bay ngang qua các vật thể đã xếp trước đó.
- 🏗️ Tự động tính toán độ lệch tọa độ (`layer_shift`) để xếp so le giữa các tầng nếu cần thiết.

### 📏 Quỹ đạo tuyến tính (Linear Motion) an toàn
Trong không gian hẹp của pallet, chuyển động vòng cung của tay máy có thể vô tình gạt ngã các vật đã xếp. Khắc phục điều này, cơ chế **Nội suy tuyến tính (Linear IK)** được áp dụng:
- ⬇️ Robot bay đến một điểm cao an toàn (Clearance Z), sau đó **đi thẳng đứng xuống** vị trí đặt vật.
- ⬆️ Sau khi nhả kẹp, tay máy **rút thẳng đứng lên** trước khi di chuyển sang quỹ đạo khác.

### 🔗 Cơ chế Attach "No-snap" (Khóa tiếp xúc thực tế)
Thay vì dịch chuyển tức thời vật thể (teleport/snap) vào đúng tâm tay kẹp ngay khi có va chạm (như `demo3.py`), hệ thống mới tính toán và duy trì khoảng cách tương đối hiện tại:
- 🤝 Khóa vật thể (Constraint) ngay tại vị trí ngón kẹp vừa chạm vào.
- Mô phỏng chính xác hơn các tình huống kẹp lệch, kẹp mép ngoài của vật thể trong thực tế.

Những cải tiến này tạo ra một hệ thống Baseline cực kỳ vững chắc, phản ánh đúng độ phức tạp của bài toán sản xuất, đồng thời là bàn đạp xuất sắc để đánh giá khả năng của thuật toán Học tăng cường ở giai đoạn sau.


# 🧠 CHƯƠNG 5. ĐỊNH HƯỚNG HỌC TĂNG CƯỜNG VÀ MỞ RỘNG HỆ THỐNG

## 🚀 5.1. Động cơ áp dụng học tăng cường

Hệ thống gắp – thả hiện tại được xây dựng theo phương pháp **baseline rule-based**, trong đó toàn bộ hành vi của robot được xác định trước bằng các luật cố định. Mặc dù phương pháp này hoạt động ổn định trong môi trường mô phỏng, nó vẫn tồn tại nhiều hạn chế:

- 🔄 Robot không có khả năng **tự thích nghi** khi điều kiện môi trường thay đổi  
- 📉 Chiến lược gắp – thả **không được tối ưu theo thời gian**  
- 🎛️ Các tham số như tốc độ, quỹ đạo, thứ tự gắp đều được **thiết kế thủ công**  
- 🧩 Khó mở rộng sang các tình huống **phức tạp hơn** (nhiều vật, nhiều mục tiêu, nhiễu)

Do đó, **Học tăng cường (Reinforcement Learning – RL)** được lựa chọn như một hướng tiếp cận phù hợp, cho phép robot **tự học chiến lược điều khiển thông qua tương tác với môi trường**.

---

## 🧩 5.2. Mô hình học tăng cường áp dụng cho bài toán

Trong khuôn khổ học tăng cường, bài toán gắp – thả được mô hình hóa theo cấu trúc chuẩn:

- 🤖 **Agent**: Robot Franka Panda  
- 🌍 **Environment**: Môi trường mô phỏng gồm băng chuyền, vật thể và thùng  
- 👁️ **State (Observation)**: Trạng thái robot và vật thể tại mỗi thời điểm  
- 🎮 **Action**: Hành động điều khiển robot  
- ⭐ **Reward**: Phần thưởng đánh giá chất lượng hành động  

Cấu trúc này cho phép robot cải thiện hành vi thông qua **quá trình thử – sai**.

---

## 🏗️ 5.3. Định nghĩa môi trường học tăng cường (RL Environment)

### 🔄 Thiết kế hàm `reset()`

Hàm `reset()` có nhiệm vụ:
- 🏠 Đưa robot về tư thế ban đầu (**home**)  
- 🗑️ Xoá các vật thể cũ trong môi trường  
- 🛤️ Khởi tạo lại băng chuyền và sinh vật thể mới  
- 🔁 Đặt lại bộ đếm thời gian và trạng thái hệ thống  

Hàm này đảm bảo mỗi **episode huấn luyện** bắt đầu từ một trạng thái xác định.

---

### ▶️ Thiết kế hàm `step(action)`

Hàm `step(action)` thực hiện các nhiệm vụ:
- 🎮 Nhận hành động từ agent  
- 🤖 Điều khiển robot tương ứng với hành động đó  
- ⏱️ Tiến mô phỏng một khoảng thời gian xác định  
- ⭐ Tính toán reward  
- 📤 Trả về trạng thái mới, reward và cờ kết thúc episode  

Cấu trúc tổng quát:
```text
state, reward, done = step(action)
```

---

### 👁️ 5.4. Thiết kế không gian trạng thái (Observation Space)

Không gian trạng thái dự kiến bao gồm:

- 📍 Tọa độ end-effector `(x, y, z)`  
- 🔢 Góc các khớp robot  
- 🤏 Trạng thái kẹp (mở / đóng)  
- 📦 Vị trí vật thể gần nhất trong ROI  
- ➡️ Vận tốc vật thể trên băng chuyền  
- 🎨 Thông tin màu sắc vật thể (one-hot encoding)  

Việc lựa chọn các thành phần trạng thái này nhằm:
- ✅ Đảm bảo **đủ thông tin** để agent ra quyết định  
- 🚫 Tránh **dư thừa dữ liệu** không cần thiết  
- ⚖️ Giữ cân bằng giữa **khả năng học** và **độ phức tạp**  

---

## 🎮 5.5. Thiết kế không gian hành động (Action Space)

Hai hướng tiếp cận chính được xem xét:

### 🧭 Hành động cấp cao (High-level actions)
- ⏱️ Chọn **thời điểm gắp**  
- ⏸️ Chọn **hành động chờ**  
- 🗂️ Chọn **thùng thả**  

Ưu điểm:
- 📦 Không gian hành động **nhỏ**  
- ⚡ Dễ học, **hội tụ nhanh**  
- 🧠 Phù hợp cho giai đoạn học ban đầu  

---

### 🕹️ Hành động cấp thấp (Low-level actions)
- ↔️ Thay đổi vị trí end-effector `(Δx, Δy, Δz)`  
- 🤏 Điều khiển **mở / đóng kẹp**  

Ưu điểm:
- 🔄 Linh hoạt cao  
- 📈 Có khả năng học **chiến lược tối ưu hơn**  
- 🤖 Phù hợp với các thuật toán điều khiển liên tục  

---

## ⭐ 5.6. Thiết kế hàm thưởng (Reward Function)

Hàm thưởng dự kiến được thiết kế dựa trên các tiêu chí sau:

- ✅ **+1.0** khi thả vật **đúng thùng theo màu**  
- ❌ **-1.0** khi thả **sai thùng**  
- ⏱️ **-0.01** cho mỗi bước thời gian (khuyến khích hoàn thành nhanh)  
- ⚠️ **-0.5** khi làm rơi vật hoặc thao tác lỗi  

Thiết kế reward đóng vai trò then chốt vì:
- 🎯 Định hướng hành vi học của agent  
- ⚖️ Cân bằng giữa **độ chính xác** và **tốc độ**  
- 🚀 Ảnh hưởng trực tiếp đến **tốc độ hội tụ**  

---

## 🧠 5.7. Thuật toán học tăng cường dự kiến

Các thuật toán RL dự kiến áp dụng gồm:

- 📘 **PPO (Proximal Policy Optimization)**  
- 🔥 **SAC (Soft Actor-Critic)**  

Các thuật toán này phù hợp với:
- 🔢 Không gian trạng thái **liên tục**  
- 🎛️ Điều khiển robot trong **môi trường mô phỏng**  
- ⚖️ Bài toán yêu cầu cân bằng giữa ổn định và tối ưu  

---

## 📊 5.8. So sánh giữa baseline rule-based và học tăng cường

```text
| Tiêu chí            | Rule-based | Reinforcement Learning |
|---------------------|----------- |------------------------|
| Tính ổn định        | Cao        | Phụ thuộc quá trình học|
| Khả năng thích nghi | Thấp       | Cao                    |
| Công thiết kế       | Cao        | Thấp sau khi học       |
| Tối ưu chiến lược   | Không      | Có                     |
| Khả năng mở rộng    | Hạn chế    | Tốt                    |
```

**Baseline rule-based đóng vai trò là:**

🧱 Mốc so sánh về độ ổn định
📐 Chuẩn tham chiếu để đánh giá hiệu quả của RL

---

## 🛣️ 5.9. Lộ trình triển khai học tăng cường

Lộ trình triển khai RL dự kiến gồm các bước:

1. 🧩 Đóng gói hệ thống hiện tại thành môi trường RL chuẩn
2. 🧠 Xác định state, action và reward
3. 🎓 Huấn luyện agent trong mô phỏng
4. 📈 Đánh giá hiệu suất và so sánh với baseline
5. 🚀 Tối ưu và mở rộng kịch bản

# CHƯƠNG 6. KẾT QUẢ HUẤN LUYỆN STAGE 1 VÀ STAGE 2

---

## 6.1. Tổng quan quy trình huấn luyện

Dự án sử dụng học tăng cường để huấn luyện robot Panda thực hiện quy trình gắp vật theo từng giai đoạn. Thay vì yêu cầu robot học toàn bộ thao tác pick-and-place ngay từ đầu, bài toán được chia thành các stage nhỏ hơn để robot học dần từng kỹ năng.

Quy trình tổng thể:

```text
Home
  ↓
Stage 1: Reach / Pre-grasp Learning
  ↓
Stage 2: Grasp Mastery
  ↓
Stage 3: Place / Drop
  ↓
Sorting / Palletizing
```

Trong phạm vi chương này, báo cáo tập trung vào hai giai đoạn đầu:

| Stage | Mục tiêu chính | Vai trò trong hệ thống |
|---|---|---|
| Stage 1 | Di chuyển từ home đến vị trí pre-grasp | Chuẩn bị vị trí trước khi gắp |
| Stage 2 | Hạ xuống, đóng gripper, giữ, nhấc và đưa vật về home | Học kỹ năng gắp vật |

Các biểu đồ trong chương này được tạo từ các file log sau khi train:

| Loại file | Vai trò |
|---|---|
| `debug.log` | Ghi chi tiết từng step và từng episode |
| `summary.log` | Ghi thống kê trung bình theo cửa sổ episode |
| `events.out.tfevents...` | Ghi chỉ số TensorBoard của PPO và custom metrics |

> Lưu ý: File `.md` này giả định đang nằm ở thư mục gốc `D:/Picking-For-Robot`. Vì vậy, đường dẫn ảnh bắt đầu bằng `training_report_output/figures/...`.

---

# 6.2. Stage 1 – Reach / Pre-grasp Learning

## Mục tiêu của Stage 1

Stage 1 là giai đoạn đầu tiên trong quá trình huấn luyện robot. Mục tiêu chính của Stage 1 là giúp robot học cách di chuyển từ vị trí ban đầu `home` đến vị trí `pre-grasp`, tức là vị trí nằm phía trên vật thể trước khi thực hiện thao tác gắp ở Stage 2.

Ở giai đoạn này, robot chưa thực hiện đóng gripper, chưa gắp vật và chưa nâng vật. Robot chỉ tập trung vào kỹ năng tiếp cận vật thể chính xác và ổn định.

Các mục tiêu chính của Stage 1 gồm:

- Di chuyển end-effector từ vị trí home đến gần vật thể.
- Căn chỉnh vị trí theo trục X, Y so với tâm vật.
- Điều chỉnh độ cao Z để đạt vị trí pre-grasp.
- Ở các substage sau, tinh chỉnh thêm hướng quay yaw.
- Giữ ổn định tại vị trí mục tiêu trong một số bước để được tính là thành công.

Stage 1 đóng vai trò nền tảng cho Stage 2. Nếu robot chưa thể tiếp cận đúng vị trí phía trên vật thì thao tác hạ xuống, đóng kẹp và nâng vật ở Stage 2 sẽ dễ thất bại.

---

## Cấu trúc substage của Stage 1

Stage 1 được chia thành 6 substage từ 1A đến 1F. Mỗi substage tăng dần độ khó, giúp robot học từ hành vi đơn giản đến hành vi chính xác hơn.

| Substage | Mục tiêu | Kết quả mong muốn |
|---|---|---|
| 1A | Reach cơ bản | Robot đi được đến vùng gần vật |
| 1B | Ổn định chuyển động | Robot giảm dao động khi tiếp cận |
| 1C | Căn chỉnh XY tốt hơn | End-effector nằm đúng phía trên vật |
| 1D | Cải thiện độ cao Z | Robot đạt độ cao pre-grasp chính xác hơn |
| 1E | Căn chỉnh yaw | Robot học hướng quay phù hợp với vật |
| 1F | Hoàn thiện pre-grasp | Kết hợp XY, Z, yaw và giữ ổn định |

Quy trình phase của Stage 1 có thể mô tả như sau:

```text
far → descend → settle → success
```

Trong đó:

- `far`: robot còn xa mục tiêu, ưu tiên giảm sai số tổng và sai số XY.
- `descend`: robot đã gần về XY và bắt đầu điều chỉnh độ cao Z.
- `settle`: robot đã gần target và cần giữ ổn định.
- `success`: robot đạt điều kiện thành công.

---

## Observation, Action và Reward trong Stage 1

Observation của Stage 1 gồm các thông tin chính:

- Vị trí end-effector.
- Vị trí vật thể.
- Vị trí target pre-grasp.
- Vector sai lệch giữa end-effector và target.
- Sai số tổng `dist`.
- Sai số `xy`, `z`, `yaw`.
- Phase hiện tại và số bước ổn định.

Action của agent trong Stage 1 chủ yếu điều khiển chuyển động end-effector:

```text
dx, dy, dz, dyaw
```

Trong đó:

- `dx`: dịch chuyển theo trục X.
- `dy`: dịch chuyển theo trục Y.
- `dz`: dịch chuyển theo trục Z.
- `dyaw`: điều chỉnh góc quay quanh trục Z.

Reward của Stage 1 được thiết kế để khuyến khích robot:

- Giảm khoảng cách đến target.
- Giảm sai số XY.
- Giảm sai số Z.
- Giảm sai số yaw ở các substage có xét yaw.
- Giữ ổn định tại vị trí mục tiêu.
- Hoàn thành episode với success.

Dạng tổng quát:

```text
reward = progress_reward + alignment_bonus + success_bonus - penalty
```

---

## Quy trình huấn luyện Stage 1

Quy trình huấn luyện Stage 1:

```text
Train 1A
  ↓
Load model 1A để train 1B
  ↓
Load model 1B để train 1C
  ↓
Load model 1C để train 1D
  ↓
Load model 1D để train 1E
  ↓
Load model 1E để train 1F
```

Cách huấn luyện này là curriculum learning, tức là robot học từ nhiệm vụ dễ đến nhiệm vụ khó.

---

## Kết quả huấn luyện từng substage của Stage 1

### Substage 1A – Reach cơ bản

Substage 1A là bước đầu tiên của Stage 1. Mục tiêu của 1A là giúp robot học hành vi di chuyển cơ bản từ home đến vùng gần vị trí pre-grasp phía trên vật.

Kết quả mong muốn:

- Reward tăng dần.
- Success rate bắt đầu xuất hiện và tăng theo thời gian.
- Sai số XY, Z giảm dần.
- Episode length giảm khi robot hoàn thành nhiệm vụ nhanh hơn.

<p align="center">
  <img src="training_report_output/figures/stage1/1A/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.1. Learning curve của Stage 1A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1A/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.2. Sai số vị trí của Stage 1A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1A/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.3. Success rate của Stage 1A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1A/04_episode_length_report.png" width="760">
</p>
<p align="center"><em>Hình 6.4. Episode length của Stage 1A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1A/tb_train_explained_variance_report.png" width="760">
</p>
<p align="center"><em>Hình 6.5. Explained variance của Stage 1A.</em></p>

Nhận xét: 1A giúp robot hình thành chính sách điều khiển ban đầu để tiếp cận vùng pre-grasp. Đây là nền tảng để các substage sau tiếp tục cải thiện độ chính xác.

---

### Substage 1B – Cải thiện ổn định

Substage 1B kế thừa từ 1A và tập trung vào việc làm cho chuyển động ổn định hơn. Robot không chỉ cần đi đến gần target mà còn phải giảm dao động khi tiếp cận.

Kết quả mong muốn:

- Reward ổn định hơn so với 1A.
- Success rate tăng và ít dao động hơn.
- Episode length giảm hoặc ổn định.
- Sai số tổng giảm đều hơn.

<p align="center">
  <img src="training_report_output/figures/stage1/1B/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.6. Learning curve của Stage 1B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1B/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.7. Sai số vị trí của Stage 1B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1B/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.8. Success rate của Stage 1B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1B/04_episode_length_report.png" width="760">
</p>
<p align="center"><em>Hình 6.9. Episode length của Stage 1B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1B/tb_train_explained_variance_report.png" width="760">
</p>
<p align="center"><em>Hình 6.10. Explained variance của Stage 1B.</em></p>

Nhận xét: 1B giúp chính sách điều khiển mượt hơn, giảm các hành vi dao động không cần thiết khi robot tiến đến mục tiêu.

---

### Substage 1C – Căn chỉnh XY

Substage 1C tập trung vào khả năng căn chỉnh end-effector theo mặt phẳng XY. Đây là kỹ năng quan trọng vì robot phải nằm đúng phía trên vật trước khi hạ xuống ở các bước sau.

Kết quả mong muốn:

- XY error giảm rõ rệt.
- Robot ít lệch ngang so với tâm vật.
- Success rate ổn định hơn.

<p align="center">
  <img src="training_report_output/figures/stage1/1C/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.11. Learning curve của Stage 1C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1C/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.12. Sai số vị trí của Stage 1C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1C/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.13. Success rate của Stage 1C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1C/04_episode_length_report.png" width="760">
</p>
<p align="center"><em>Hình 6.14. Episode length của Stage 1C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1C/tb_train_explained_variance_report.png" width="760">
</p>
<p align="center"><em>Hình 6.15. Explained variance của Stage 1C.</em></p>

Nhận xét: 1C là substage quan trọng để cải thiện độ chính xác theo mặt phẳng ngang. Khi XY error giảm tốt, Stage 2 sẽ dễ học thao tác hạ xuống và đóng gripper hơn.

---

### Substage 1D – Cải thiện độ cao Z

Substage 1D tập trung vào việc điều chỉnh độ cao của end-effector để đạt vị trí pre-grasp chính xác hơn. Robot cần vừa giữ XY tốt vừa đưa Z về đúng độ cao mục tiêu.

Kết quả mong muốn:

- Z error giảm rõ rệt.
- Robot không hạ quá thấp gây va chạm với vật.
- Robot giữ được độ cao pre-grasp ổn định.

<p align="center">
  <img src="training_report_output/figures/stage1/1D/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.16. Learning curve của Stage 1D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1D/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.17. Sai số vị trí của Stage 1D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1D/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.18. Success rate của Stage 1D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1D/04_episode_length_report.png" width="760">
</p>
<p align="center"><em>Hình 6.19. Episode length của Stage 1D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1D/tb_train_explained_variance_report.png" width="760">
</p>
<p align="center"><em>Hình 6.20. Explained variance của Stage 1D.</em></p>

Nhận xét: 1D giúp robot học cách kiểm soát chiều cao trước khi gắp. Đây là bước chuẩn bị quan trọng để tránh hạ sai độ cao khi bước sang Stage 2.

---

### Substage 1E – Căn chỉnh yaw

Substage 1E bổ sung yêu cầu căn chỉnh yaw. Robot cần điều chỉnh hướng quay của end-effector sao cho phù hợp với hướng của vật thể.

Kết quả mong muốn:

- Yaw error giảm dần.
- Robot duy trì được XY, Z ổn định trong khi điều chỉnh yaw.
- Success rate không giảm khi thêm yêu cầu orientation.

<p align="center">
  <img src="training_report_output/figures/stage1/1E/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.21. Learning curve của Stage 1E.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1E/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.22. Sai số vị trí của Stage 1E.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1E/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.23. Success rate của Stage 1E.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1E/04_episode_length_report.png" width="760">
</p>
<p align="center"><em>Hình 6.24. Episode length của Stage 1E.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1E/05_yaw_error_report.png" width="760">
</p>
<p align="center"><em>Hình 6.25. Yaw error của Stage 1E.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1E/tb_train_explained_variance_report.png" width="760">
</p>
<p align="center"><em>Hình 6.26. Explained variance của Stage 1E.</em></p>

Nhận xét: 1E giúp robot không chỉ đến đúng vị trí mà còn quay đúng hướng. Điều này hỗ trợ việc đóng gripper chính xác hơn ở Stage 2.

---

### Substage 1F – Hoàn thiện pre-grasp

Substage 1F là bước hoàn thiện của Stage 1. Robot cần kết hợp các kỹ năng đã học: căn chỉnh XY, điều chỉnh Z, căn chỉnh yaw và giữ ổn định tại vị trí pre-grasp.

Kết quả mong muốn:

- Sai số tổng thấp.
- XY, Z, yaw đều đạt ngưỡng mục tiêu.
- Stable steps đạt yêu cầu.
- Success rate cao và ổn định.

<p align="center">
  <img src="training_report_output/figures/stage1/1F/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.27. Learning curve của Stage 1F.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1F/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.28. Sai số vị trí của Stage 1F.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1F/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.29. Success rate của Stage 1F.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1F/04_episode_length_report.png" width="760">
</p>
<p align="center"><em>Hình 6.30. Episode length của Stage 1F.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1F/05_yaw_error_report.png" width="760">
</p>
<p align="center"><em>Hình 6.31. Yaw error của Stage 1F.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/1F/tb_train_explained_variance_report.png" width="760">
</p>
<p align="center"><em>Hình 6.32. Explained variance của Stage 1F.</em></p>

Nhận xét: 1F là substage tổng hợp, cho thấy robot đã hoàn thiện kỹ năng tiếp cận vị trí pre-grasp. Đây là đầu ra quan trọng để nối sang Stage 2.

---

## So sánh tổng hợp các substage Stage 1

Các biểu đồ tổng hợp giúp đánh giá toàn bộ quá trình curriculum learning trong Stage 1.

<p align="center">
  <img src="training_report_output/figures/stage1/all_substages_reward_comparison_report.png" width="760">
</p>
<p align="center"><em>Hình 6.33. So sánh reward giữa các substage Stage 1.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/all_substages_success_comparison_report.png" width="760">
</p>
<p align="center"><em>Hình 6.34. So sánh success rate giữa các substage Stage 1.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage1/all_substages_distance_comparison_report.png" width="760">
</p>
<p align="center"><em>Hình 6.35. So sánh final distance giữa các substage Stage 1.</em></p>

Đánh giá chung Stage 1: qua các substage từ 1A đến 1F, robot học dần từ kỹ năng tiếp cận cơ bản đến pre-grasp chính xác. Khi các biểu đồ cho thấy reward và success rate tăng, đồng thời sai số giảm, có thể kết luận curriculum của Stage 1 phù hợp và đủ làm nền tảng cho Stage 2.

---

# 6.3. Stage 2 – Grasp Mastery

## Mục tiêu của Stage 2

Sau khi Stage 1 hoàn thành nhiệm vụ đưa end-effector đến vị trí pre-grasp, Stage 2 tập trung vào việc huấn luyện robot thực hiện thao tác gắp vật. Mục tiêu của Stage 2 là giúp robot học chuỗi hành động từ căn chỉnh vị trí, hạ xuống vật, đóng gripper, giữ vật ổn định, nhấc vật lên và đưa vật về vị trí home.

Stage 2 là bước chuyển từ bài toán di chuyển đến vị trí mục tiêu sang bài toán tương tác vật lý với vật thể. Robot không chỉ cần đến đúng vị trí mà còn phải tiếp xúc đúng với vật, đóng gripper đúng thời điểm, giữ vật sau khi gắp và tránh làm rơi vật khi nâng hoặc di chuyển.

---

## Cấu trúc substage của Stage 2

Stage 2 được chia thành 4 substage:

```text
2A → 2B → 2C → 2D
```

| Substage | Phase chính | Mục tiêu | Kết quả mong muốn |
|---|---|---|---|
| 2A | `xy_align → descend → close` | Học hạ xuống và đóng gripper | Tạo grasp ban đầu |
| 2B | `xy_align → descend → close → hold` | Giữ grasp ổn định | Duy trì grasp trong nhiều step |
| 2C | `xy_align → descend → close → hold → lift` | Nhấc vật lên | Vật tăng độ cao đủ ngưỡng |
| 2D | `xy_align → descend → close → hold → lift → return_home` | Đưa vật về home | Giữ vật và đưa về home/safe |

---

## Phase logic của Stage 2

Quy trình phase của Stage 2:

```text
xy_align → descend → close → hold → lift → return_home
```

Ý nghĩa từng phase:

| Phase | Ý nghĩa |
|---|---|
| `xy_align` | Căn gripper đúng phía trên vật theo XY |
| `descend` | Hạ end-effector xuống độ cao grasp |
| `close` | Đóng gripper để kẹp vật |
| `hold` | Giữ vật ổn định sau khi kẹp |
| `lift` | Nhấc vật lên khỏi mặt bàn |
| `return_home` | Mang vật về vị trí home/safe |

---

## Observation, Action và Reward trong Stage 2

Action của Stage 2 gồm 4 thành phần:

```text
dx, dy, dz, dgrip
```

Trong đó:

- `dx`, `dy`, `dz`: điều khiển dịch chuyển end-effector.
- `dgrip`: điều khiển độ mở/đóng gripper.

Yaw không nằm trực tiếp trong action của Stage 2. Môi trường điều chỉnh yaw theo vật hoặc home tùy phase để policy tập trung hơn vào thao tác gắp và điều khiển gripper.

Reward của Stage 2 được thiết kế theo từng phase:

- `xy_align`: thưởng khi giảm sai số XY.
- `descend`: thưởng khi hạ xuống đúng độ cao grasp.
- `close`: thưởng khi đóng gripper đúng thời điểm.
- `hold`: thưởng khi giữ grasp ổn định.
- `lift`: thưởng khi vật được nâng lên.
- `return_home`: thưởng khi đưa vật về gần home.

Penalty được dùng để hạn chế các hành vi sai:

- Action quá lớn.
- Đứng yên quá lâu.
- Ra ngoài workspace.
- Đóng gripper quá sớm.
- Đóng gripper nhưng không có contact.
- Contact giả.
- Làm rơi vật.
- Timeout.

---

## Kết quả huấn luyện từng substage của Stage 2

### Substage 2A – Hạ xuống và đóng gripper

Substage 2A là bước đầu của Stage 2. Robot học cách căn chỉnh XY, hạ xuống vị trí grasp và đóng gripper để tạo grasp ban đầu.

Kết quả mong muốn:

- Gripper đóng đúng thời điểm.
- Có contact giữa hai ngón gripper và vật.
- Grasp established rate tăng.
- Success rate tăng khi robot tạo được grasp cơ bản.

<p align="center">
  <img src="training_report_output/figures/stage2/2A/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.36. Learning curve của Stage 2A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2A/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.37. Sai số vị trí của Stage 2A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2A/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.38. Success rate của Stage 2A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2A/06_grasp_contact_report.png" width="760">
</p>
<p align="center"><em>Hình 6.39. Grasp/contact quality của Stage 2A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2A/07_grip_width_report.png" width="760">
</p>
<p align="center"><em>Hình 6.40. Grip width của Stage 2A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2A/S01_summary_learning_report.png" width="760">
</p>
<p align="center"><em>Hình 6.41. Summary learning của Stage 2A.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2A/S03_summary_grasp_contact_report.png" width="760">
</p>
<p align="center"><em>Hình 6.42. Summary grasp/contact của Stage 2A.</em></p>

Nhận xét: 2A tập trung vào việc tạo grasp ban đầu. Nếu dual contact rate và grasp established rate tăng, robot đã học được cách tiếp xúc và đóng gripper hiệu quả hơn.

---

### Substage 2B – Giữ grasp ổn định

Substage 2B kế thừa từ 2A và yêu cầu robot giữ grasp trong nhiều step liên tiếp. Đây là bước phân biệt giữa việc chỉ chạm vào vật và việc thật sự giữ được vật ổn định.

Kết quả mong muốn:

- Grasp rate cao hơn 2A.
- Dual contact rate ổn định.
- Hold steps tăng.
- Success rate ổn định hơn.

<p align="center">
  <img src="training_report_output/figures/stage2/2B/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.43. Learning curve của Stage 2B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2B/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.44. Sai số vị trí của Stage 2B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2B/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.45. Success rate của Stage 2B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2B/06_grasp_contact_report.png" width="760">
</p>
<p align="center"><em>Hình 6.46. Grasp/contact quality của Stage 2B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2B/07_grip_width_report.png" width="760">
</p>
<p align="center"><em>Hình 6.47. Grip width của Stage 2B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2B/S01_summary_learning_report.png" width="760">
</p>
<p align="center"><em>Hình 6.48. Summary learning của Stage 2B.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2B/S03_summary_grasp_contact_report.png" width="760">
</p>
<p align="center"><em>Hình 6.49. Summary grasp/contact của Stage 2B.</em></p>

Nhận xét: 2B đánh giá độ ổn định của grasp. Khi grasp/contact rate ổn định hơn, robot đã học được cách duy trì vật sau khi đóng gripper.

---

### Substage 2C – Nhấc vật lên

Substage 2C bổ sung phase lift. Robot không chỉ cần giữ vật mà còn phải nhấc vật lên khỏi mặt bàn.

Kết quả mong muốn:

- Lift delta tăng.
- Vật được nâng lên đủ ngưỡng.
- Grasp vẫn được duy trì trong quá trình lift.
- Drop rate thấp.

<p align="center">
  <img src="training_report_output/figures/stage2/2C/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.50. Learning curve của Stage 2C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2C/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.51. Sai số vị trí của Stage 2C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2C/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.52. Success rate của Stage 2C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2C/06_grasp_contact_report.png" width="760">
</p>
<p align="center"><em>Hình 6.53. Grasp/contact quality của Stage 2C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2C/08_lift_delta_report.png" width="760">
</p>
<p align="center"><em>Hình 6.54. Lift delta của Stage 2C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2C/S01_summary_learning_report.png" width="760">
</p>
<p align="center"><em>Hình 6.55. Summary learning của Stage 2C.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2C/S04_summary_lift_home_report.png" width="760">
</p>
<p align="center"><em>Hình 6.56. Summary lift/home của Stage 2C.</em></p>

Nhận xét: 2C kiểm tra robot có thật sự gắp được vật hay không. Nếu lift delta tăng trong khi grasp/contact vẫn ổn định, robot đã học được thao tác nâng vật.

---

### Substage 2D – Nhấc vật và đưa về home

Substage 2D là substage đầy đủ nhất của Stage 2. Robot cần gắp vật, giữ vật, nhấc lên và đưa vật về vị trí home/safe.

Kết quả mong muốn:

- Grasp ổn định sau khi lift.
- Lift delta đạt ngưỡng.
- Home error giảm.
- Success rate tăng khi robot đưa vật về home thành công.

<p align="center">
  <img src="training_report_output/figures/stage2/2D/01_learning_curve_report.png" width="760">
</p>
<p align="center"><em>Hình 6.57. Learning curve của Stage 2D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2D/02_error_reduction_report.png" width="760">
</p>
<p align="center"><em>Hình 6.58. Sai số vị trí của Stage 2D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2D/03_success_rate_report.png" width="760">
</p>
<p align="center"><em>Hình 6.59. Success rate của Stage 2D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2D/06_grasp_contact_report.png" width="760">
</p>
<p align="center"><em>Hình 6.60. Grasp/contact quality của Stage 2D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2D/08_lift_delta_report.png" width="760">
</p>
<p align="center"><em>Hình 6.61. Lift delta của Stage 2D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2D/09_home_error_report.png" width="760">
</p>
<p align="center"><em>Hình 6.62. Home error của Stage 2D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2D/S01_summary_learning_report.png" width="760">
</p>
<p align="center"><em>Hình 6.63. Summary learning của Stage 2D.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/2D/S04_summary_lift_home_report.png" width="760">
</p>
<p align="center"><em>Hình 6.64. Summary lift/home của Stage 2D.</em></p>

Nhận xét: 2D là bước nối giữa grasp và các stage place/drop sau này. Khi home error giảm và success rate tăng, robot đã học được cách giữ vật và đưa vật về vị trí an toàn.

---

## So sánh tổng hợp các substage Stage 2

<p align="center">
  <img src="training_report_output/figures/stage2/all_substages_reward_comparison_report.png" width="760">
</p>
<p align="center"><em>Hình 6.65. So sánh reward giữa các substage Stage 2.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/all_substages_success_comparison_report.png" width="760">
</p>
<p align="center"><em>Hình 6.66. So sánh success rate giữa các substage Stage 2.</em></p>

<p align="center">
  <img src="training_report_output/figures/stage2/all_substages_grasp_comparison_report.png" width="760">
</p>
<p align="center"><em>Hình 6.67. So sánh grasp rate giữa các substage Stage 2.</em></p>

Đánh giá chung Stage 2: thông qua curriculum 2A đến 2D, robot học dần từ việc tạo grasp ban đầu, giữ grasp ổn định, nhấc vật lên và đưa vật về home. Các chỉ số quan trọng để đánh giá Stage 2 gồm success rate, grasp/contact rate, grip width, lift delta và home error.

---

# 6.4. Kết luận chung Stage 1 và Stage 2

Stage 1 và Stage 2 tạo thành hai bước nền tảng của pipeline pick-and-place.

Stage 1 giúp robot học cách tiếp cận vật thể ở vị trí pre-grasp. Qua các substage 1A đến 1F, robot dần cải thiện khả năng căn chỉnh XY, điều chỉnh Z, căn chỉnh yaw và giữ ổn định tại target.

Stage 2 giúp robot học kỹ năng gắp vật. Qua các substage 2A đến 2D, robot dần học cách hạ xuống, đóng gripper, giữ vật, nhấc vật và đưa vật về home.

Tổng kết:

- Stage 1 đảm bảo robot đến đúng vị trí trước khi gắp.
- Stage 2 đảm bảo robot có thể thực hiện thao tác gắp thật.
- Curriculum learning giúp giảm độ khó và tăng tính ổn định trong quá trình huấn luyện.
- Các biểu đồ reward, success rate, error reduction, grasp/contact, lift delta và home error là bằng chứng trực quan cho quá trình học của agent.

Sau khi hoàn thành hai stage này, hệ thống có thể tiếp tục mở rộng sang Stage 3 để học thao tác place/drop, sau đó phát triển thêm sorting theo màu và palletizing.
