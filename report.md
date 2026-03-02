# 🚀 CHƯƠNG 1. TỔNG QUAN DỰ ÁN

## 🧭 1.1. Bối cảnh và ý nghĩa

Trong các dây chuyền sản xuất hiện đại, bài toán **gắp – thả và phân loại vật thể** xuất hiện rất phổ biến (phân loại theo màu, kích thước, mã vạch, loại sản phẩm…). Một tình huống điển hình là các vật thể được đặt lên **băng chuyền di chuyển liên tục**, robot cần nhận biết vật thể và đưa về đúng vị trí tập kết.

Trong dự án này, mô hình được đơn giản hóa nhưng vẫn giữ được bản chất của bài toán công nghiệp:

- 📦 Băng chuyền chạy với tốc độ cố định  
- 🎲 Vật thể xuất hiện ngẫu nhiên theo thời gian  
- 🤖 Robot gắp vật trong một vùng cho phép và thả vào đúng thùng theo màu  

Việc xây dựng mô phỏng có ý nghĩa quan trọng vì:

- ⚡ Cho phép thử nghiệm nhanh các thuật toán điều khiển và học tăng cường  
- 🔧 Dễ thay đổi tham số (tốc độ băng chuyền, mật độ vật, vị trí thùng…)  
- 🛡️ An toàn và tiết kiệm chi phí so với robot thật  

---

## 🎯 1.2. Mục tiêu dự án

### 🎯 1.2.1. Mục tiêu tổng quát

Xây dựng một hệ thống mô phỏng robot gắp – thả phân loại theo màu trong PyBullet và chuẩn bị nền tảng để triển khai **Học tăng cường (Reinforcement Learning – RL)** cho bài toán này.

### 🧩 1.2.2. Mục tiêu cụ thể

1. 🏗️ Tạo môi trường mô phỏng gồm: mặt phẳng, robot Franka Panda, băng chuyền và vật thể  
2. 🛤️ Mô phỏng băng chuyền chạy ổn định với tốc độ không đổi  
3. 🎨 Sinh vật thể ngẫu nhiên (kích thước giống nhau, khác màu) và quản lý vòng đời vật thể  
4. 🧮 Điều khiển robot bằng Inverse Kinematics (IK) để thực hiện chu trình gắp – thả ổn định  
5. 🗂️ Phân loại vật thể theo màu và thả đúng thùng  
6. 🛠️ Cung cấp công cụ debug và trực quan để hỗ trợ phát triển và mở rộng  

---

## 📐 1.3. Phạm vi và giả định của bài toán

Dự án sử dụng các giả định sau:

- 📦 Vật thể có kích thước giống nhau, khác nhau về màu sắc  
- 👁️ Việc nhận biết vị trí và màu vật thể sử dụng ground-truth từ mô phỏng  
- 📍 Robot chỉ gắp vật trong một vùng làm việc giới hạn (ROI)  
- 🧪 Giai đoạn hiện tại sử dụng baseline rule-based, chưa áp dụng học tăng cường  

---

## 🔄 1.4. Pipeline vận hành tổng quát

```text
🟢 Khởi tạo mô phỏng
        ↓
📦 Sinh vật thể trên băng chuyền
        ↓
📍 Kiểm tra vùng gắp (ROI)
        ↓
🧮 Điều khiển robot bằng IK
        ↓
🤏 Gắp vật
        ↓
🗂️ Di chuyển tới thùng theo màu
        ↓
📥 Thả vật
        ↓
🏠 Quay về trạng thái sẵn sàng
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

### ⏱️ 2.3.1. Thời gian mô phỏng
- ⏲️ Bước thời gian cố định: `DT = 1/240` giây  
- 🔁 Mỗi vòng lặp gọi `stepSimulation()` để cập nhật trạng thái  

---

### 🎲 2.3.2. Sinh vật thể ngẫu nhiên
- 🎯 Vật thể được sinh theo chu kỳ `SPAWN_INTERVAL`  
- ↔️ Trục X được lấy ngẫu nhiên trong ROI  
- ⬆️ Trục Y cố định tại vị trí spawn  
- 📏 Cơ chế giãn cách theo Y (`MIN_SPAWN_DY`) để tránh chồng vật  

---

### ➡️ 2.3.3. Mô phỏng chuyển động băng chuyền
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

### ⚙️ 4.2.1. Cấu hình mô phỏng
Hệ thống được triển khai trong môi trường PyBullet với các thiết lập sau:

- ⏱️ Bước thời gian mô phỏng: `DT = 1/240` giây  
- 🌍 Trọng lực: `(0, 0, -9.81)`  
- 🤖 Robot sử dụng: Franka Panda (base cố định)  
- 🧱 Mặt phẳng làm việc: `plane.urdf`  

Toàn bộ mô phỏng được cập nhật theo **thời gian thực**, cho phép quan sát trực quan quá trình robot thao tác.

---

### 🛤️ 4.2.2. Cấu hình băng chuyền và vật thể

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

### 📌 4.3.1. Xác định vật thể trong môi trường
Tại mỗi vòng lặp, hệ thống duyệt qua danh sách các vật thể đang tồn tại:

1. 📍 Lấy vị trí `(x, y, z)` từ mô phỏng  
2. 🏷️ Kiểm tra trạng thái vật thể (đã phân loại hay chưa)  
3. ⏭️ Bỏ qua các vật thể đã được thả vào thùng  

Việc xác định vị trí và trạng thái sử dụng **ground-truth trực tiếp từ mô phỏng**, đảm bảo độ chính xác tuyệt đối trong giai đoạn baseline.

---

### 🟦 4.3.2. Vùng gắp (ROI)

Hệ thống định nghĩa một vùng gắp (Region of Interest – ROI) hình chữ nhật trên mặt phẳng làm việc.

Một vật thể chỉ được xem là ứng viên gắp nếu:
- `xmin ≤ x ≤ xmax`
- `ymin ≤ y ≤ ymax`

Lợi ích:
- 🔍 Giảm không gian tìm kiếm  
- 🦾 Tránh cấu hình robot khó hoặc gần suy biến  
- 📈 Tăng độ ổn định của quá trình gắp  

---

### 🎯 4.3.3. Chiến lược chọn vật thể để gắp

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

### 🪂 4.4.1. Tiếp cận theo hai pha (Hover – Descend)

Robot tiếp cận vật thể theo hai pha:

1. 🟦 **Hover**: di chuyển end-effector tới vị trí phía trên vật (`z_hover`)  
2. ⬇️ **Descend**: hạ end-effector xuống vị trí gắp (`z_pick`)  

Do vật thể chuyển động trên băng chuyền, ngay trước pha hạ xuống, hệ thống **resample lại vị trí vật thể** để giảm sai lệch.

---

### 🎛️ 4.4.2. Điều khiển chuyển động

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

### 🎨 4.6.1. Xác định vị trí thả
- 🧠 Thùng thả được xác định dựa trên **màu sắc vật thể**  
- 🧭 Sử dụng waypoint trung gian cho thùng xa  

---

### 📥 4.6.2. Thả và cố định vật thể

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

### ✅ 4.8.1. Kết quả đạt được
Qua thực nghiệm, hệ thống cho thấy:

✔️ Robot gắp và thả đúng vật thể theo màu
✔️ Chuyển động mượt, không teleport
✔️ Không xảy ra hiện tượng rơi vật
✔️ Vật thể sau khi phân loại không bị băng chuyền kéo đi
✔️ Hệ thống chạy ổn định qua nhiều chu kỳ liên tiếp

---

### 🛡️ 4.8.2. Đánh giá tính ổn định
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

### 🔄 5.3.1. Thiết kế hàm `reset()`

Hàm `reset()` có nhiệm vụ:
- 🏠 Đưa robot về tư thế ban đầu (**home**)  
- 🗑️ Xoá các vật thể cũ trong môi trường  
- 🛤️ Khởi tạo lại băng chuyền và sinh vật thể mới  
- 🔁 Đặt lại bộ đếm thời gian và trạng thái hệ thống  

Hàm này đảm bảo mỗi **episode huấn luyện** bắt đầu từ một trạng thái xác định.

---

### ▶️ 5.3.2. Thiết kế hàm `step(action)`

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

### 🧭 5.5.1. Hành động cấp cao (High-level actions)
- ⏱️ Chọn **thời điểm gắp**  
- ⏸️ Chọn **hành động chờ**  
- 🗂️ Chọn **thùng thả**  

Ưu điểm:
- 📦 Không gian hành động **nhỏ**  
- ⚡ Dễ học, **hội tụ nhanh**  
- 🧠 Phù hợp cho giai đoạn học ban đầu  

---

### 🕹️ 5.5.2. Hành động cấp thấp (Low-level actions)
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