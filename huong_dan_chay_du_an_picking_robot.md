# Hướng dẫn sử dụng và chạy dự án Picking-For-Robot

Dự án sử dụng robot **Franka Panda** trong môi trường mô phỏng **PyBullet** để huấn luyện robot thực hiện thao tác gắp – thả vật thể bằng học tăng cường. Môi trường được xây dựng bằng Python, PyBullet, Gymnasium và Stable-Baselines3.

Trong dự án này, robot Panda được dùng làm cánh tay thao tác chính. PyBullet đóng vai trò mô phỏng vật lý, bao gồm robot, bàn, vật thể, va chạm, tiếp xúc giữa gripper và vật, cũng như chuyển động của robot trong không gian 3D.

---

## 1. Yêu cầu môi trường

Dự án được chạy bằng Python trong virtual environment riêng tên là `rl_robot`.

Phiên bản Python đang sử dụng:

```bash
Python 3.9.13
```

Các thư viện chính:

```text
pybullet
gymnasium
stable-baselines3
torch
numpy
matplotlib
tensorboard
pandas
tqdm
rich
```

Vai trò của các thư viện chính:

| Thư viện | Vai trò |
|---|---|
| `pybullet` | Mô phỏng vật lý, robot Panda, vật thể và va chạm |
| `gymnasium` | Chuẩn hóa môi trường RL theo dạng Env |
| `stable-baselines3` | Huấn luyện agent bằng thuật toán PPO |
| `torch` | Backend học sâu cho PPO |
| `numpy` | Xử lý vector, trạng thái, tọa độ |
| `matplotlib` | Vẽ biểu đồ kết quả |
| `tensorboard` | Theo dõi quá trình huấn luyện |
| `pandas` | Xử lý dữ liệu log nếu cần |

---

## 2. Cấu trúc thư mục chính

Cấu trúc dự án có thể tổ chức như sau:

```text
Picking-For-Robot/
│
├── config/
│   └── reach_env_config.py
│   └── grasp_env_config.py
│
├── env/
│   ├── grasp_env.py
│   ├── reach_env.py
│   ├── panda_controller.py
│   ├── reward_grasp.py
│   ├── reward_reach.py
│   └── camera.py
│
├── train_model/
│   └── train_stage1_grasp_sb3.py
│   └── train_stage2_grasp_sb3.py
│
├── test/
│   └── test_reach_env.py
│   └── test_grasp_env.py
│
├── script/
│   └── stage1_reach/
│       ├── control_callbacks.py
│       ├── reach_debug_step_callback.py
│       ├── reach_debug_summary_callback.py
│       └── reach_tensorboard_callback.py
│   └── stage2_grasp/
│       ├── control_callbacks.py
│       ├── grasp_debug_step_callback.py
│       ├── grasp_debug_summary_callback.py
│       └── grasp_tensorboard_callback.py
│
├── models/
├── debug_logs/
├── logs_stage1/
├── logs_stage2/
├── training_report_output/
├── plot_advanced.py
├── debug_links.py
│── demo5.py
└── requirements.txt
```

Ý nghĩa các thư mục:

| Thư mục/File | Vai trò |
|---|---|
| `config/` | Chứa cấu hình môi trường, reward, target, substage |
| `env/` | Chứa môi trường PyBullet, controller robot Panda, reward |
| `train_model/` | Chứa file huấn luyện model |
| `test/` | Chứa file chạy thử model đã train |
| `script/stage/` | Chứa callback ghi log, summary và TensorBoard |
| `models/` | Lưu model đã train |
| `debug_logs/` | Lưu log chi tiết và log tổng hợp |
| `logs_stage/` | Lưu TensorBoard log |
| `training_report_output/` | Lưu biểu đồ và file báo cáo tự động |
| `requirements.txt` | Danh sách thư viện của môi trường Python |

---

## 3. Kích hoạt môi trường Python

Mở Git Bash tại thư mục dự án:

```bash
cd 'đường dẫn dự án/Picking-For-Robot' 
```

Kích hoạt virtual environment:

```bash
source rl_robot/Scripts/activate
```

Sau khi kích hoạt thành công, terminal sẽ có dạng:

```bash
(rl_robot)
user 'đường dẫn dự án/Picking-For-Robot' (main)
```

Kiểm tra Python đang dùng:

```bash
which python
python --version
```

Kết quả đúng là Python nằm trong môi trường `rl_robot` và phiên bản Python là:

```bash
Python 3.9.13
```

Có thể kiểm tra kỹ hơn bằng:

```bash
python -c "import sys; print(sys.executable)"
```

Nếu đường dẫn trỏ đến:

```text
đường dẫn dự án\Picking-For-Robot\rl_robot\Scripts\python.exe
```

thì nghĩa là dự án đang dùng đúng Python trong virtual environment.

---

## 4. Cài thư viện cần thiết

Nếu đã có file `requirements.txt`, chạy:

```bash
python -m pip install -r requirements.txt
```

Nếu cần cài thủ công các thư viện chính:

```bash
python -m pip install numpy matplotlib gymnasium pybullet stable-baselines3 tensorboard torch pandas tqdm rich
```

Sau khi cài xong, lưu lại danh sách thư viện:

```bash
python -m pip freeze > requirements.txt
```

Kiểm tra các thư viện quan trọng:

```bash
python -c "import pybullet; print('pybullet OK')"
python -c "import stable_baselines3; print('stable-baselines3 OK')"
python -c "import torch; print('torch OK')"
python -c "import gymnasium; print('gymnasium OK')"
python -c "import tensorboard; print('tensorboard OK')"
python -c "import matplotlib; print('matplotlib OK')"
```

Nếu tất cả đều hiện `OK` thì môi trường đã sẵn sàng.

---

## 5. Chạy kiểm tra môi trường PyBullet

Trước khi train, nên chạy test môi trường để kiểm tra robot Panda và PyBullet hoạt động đúng.

```bash
python ./test/file stage muốn test
```

Khi chạy file này, chương trình cho phép chọn substage Stage  cần test:

```text
Substage
Ví dụ:
1A
1B
1C
....
```

Nếu có model đã train, chương trình sẽ tự tìm model trong thư mục `models/`. Nếu chưa có model, chương trình có thể chạy bằng random policy.

Khi bật GUI, PyBullet sẽ hiển thị robot Panda, bàn và vật thể. Robot sẽ thực hiện các hành vi tương ứng với substage được chọn.

---

## 6. Chạy huấn luyện Stage 2

Để huấn luyện Stage , chạy:

```bash
python ./train_model/stage muốn train 
```

File này sẽ tự động huấn luyện theo thứ tự:

```text
Substage được set trước đó
ví dụ: 1A -> 1B -> 1C -> 1D -> 1E -> 1F
```


Quá trình train sử dụng thuật toán PPO của Stable-Baselines3. Mỗi substage sau sẽ kế thừa model từ substage trước.


Đây là cách huấn luyện theo **curriculum learning**, giúp robot học từ nhiệm vụ đơn giản đến nhiệm vụ phức tạp hơn.

---

## 7. Các file được tạo sau khi train

### 7.1. Model

Model được lưu trong thư mục:

```text
models/
```

Ví dụ:

```text
stage2_grasp_mastery_2A.zip
stage2_grasp_mastery_2B.zip
stage2_grasp_mastery_2C.zip
stage2_grasp_mastery_2D.zip
```

Nếu quá trình train bị dừng giữa chừng, sẽ có file:

```text
stage2_grasp_mastery_2A_latest.zip
```

File `_latest` dùng để resume training.

### 7.2. Debug log

Debug log được lưu trong thư mục:

```text
debug_logs/
```

Ví dụ:

```text
stage2_2A_debug.log
stage2_2A_summary.log
stage2_2B_debug.log
stage2_2B_summary.log
stage2_2C_debug.log
stage2_2C_summary.log
stage2_2D_debug.log
stage2_2D_summary.log
```

Ý nghĩa:

| File | Ý nghĩa |
|---|---|
| `debug.log` | Ghi chi tiết từng step |
| `summary.log` | Ghi thống kê theo window, ví dụ success rate, reward mean, grasp rate |

### 7.3. TensorBoard log

TensorBoard log được lưu trong thư mục:

```text
logs_stage2/
```

Ví dụ:

```text
logs_stage2/stage2_grasp_mastery_2A_0/
logs_stage2/stage2_grasp_mastery_2B_0/
logs_stage2/stage2_grasp_mastery_2C_0/
logs_stage2/stage2_grasp_mastery_2D_0/
```

Mỗi thư mục sẽ chứa file:

```text
events.out.tfevents...
```

File này dùng để xem biểu đồ TensorBoard.

---

## 8. Xem TensorBoard

Chạy lệnh:

```bash
tensorboard --logdir logs_stage2
```

Sau đó mở trình duyệt tại địa chỉ:

```text
http://localhost:6006
```

TensorBoard sẽ hiển thị các chỉ số như:

```text
train/explained_variance
train/value_loss
train/policy_gradient_loss
train/approx_kl
train/entropy_loss
episode/success_rate_100
episode/reward_mean_100
grasp/dual_contact_rate_100
grasp/grasp_established_rate_100
grasp/lift_delta_mean_100
grasp/home_error_mean_100
```

Các chỉ số quan trọng:

| Chỉ số | Ý nghĩa |
|---|---|
| `success_rate_100` | Tỷ lệ thành công trong 100 episode gần nhất |
| `reward_mean_100` | Reward trung bình |
| `grasp_established_rate_100` | Tỷ lệ grasp thật sự được thiết lập |
| `dual_contact_rate_100` | Tỷ lệ hai ngón gripper cùng chạm vật |
| `lift_delta_mean_100` | Độ nâng trung bình của vật |
| `home_error_mean_100` | Sai số khi đưa vật về home |
| `explained_variance` | Độ ổn định của critic/value function |

---

## 9. Vẽ biểu đồ cho báo cáo

Sau khi train xong, có thể chạy script vẽ biểu đồ tự động.

Ví dụ:

```bash
python plot_training_auto_report_stage2.py
```

Script sẽ quét các thư mục:

```text
debug_logs/
logs_stage2/
```

và tạo output tại:

```text
training_report_output/
```

Cấu trúc output:

```text
training_report_output/
│
├── training_report_auto.md
│
└── figures/
    └── stage2/
        ├── 2A/
        ├── 2B/
        ├── 2C/
        └── 2D/
```

---

## 10. Test model sau khi train

Sau khi train xong, kiểm tra model bằng:

```bash
python ./test/test_grasp_env.py
```

Chương trình sẽ hỏi substage cần test:

```text
2A
2B
2C
2D
```

Ví dụ chọn `2A`, robot sẽ thực hiện:

```text
xy_align → descend → close
```

Chọn `2D`, robot sẽ thực hiện:

```text
xy_align → descend → close → hold → lift → return_home
```

Trong quá trình test, terminal sẽ in ra:

```text
phase
reward
xy
z
yaw
grip
contact
grasp
success
```

Các thông tin này giúp kiểm tra robot có thực hiện đúng từng phase hay không.

---

## 11. Resume training khi bị dừng

Nếu quá trình train bị dừng giữa chừng, chương trình sẽ lưu model `_latest`.

Ví dụ:

```text
models/stage2_grasp_mastery_2A_latest.zip
```

Khi chạy lại:

```bash
python ./train_model/train_stage2_grasp_sb3.py
```

script sẽ tự phát hiện `_latest` và tiếp tục train từ substage đó.

---

## 12. Một số lỗi thường gặp

### 12.1. Lỗi không import được PyBullet

Nếu gặp lỗi:

```text
ImportError: DLL load failed while importing pybullet
```

Kiểm tra lại môi trường Python:

```bash
which python
python -c "import sys; print(sys.executable)"
```

Sau đó cài lại PyBullet:

```bash
python -m pip install --upgrade --force-reinstall pybullet
```

### 12.2. Lỗi thiếu thư viện

Nếu thiếu thư viện, cài lại bằng:

```bash
python -m pip install -r requirements.txt
```

Hoặc cài riêng:

```bash
python -m pip install tên_thư_viện
```

### 12.3. TensorBoard không chạy

Cài lại TensorBoard:

```bash
python -m pip install tensorboard
```

Chạy lại:

```bash
tensorboard --logdir logs_stage2
```

### 12.4. Không thấy GUI PyBullet

Trong file config hoặc file test, cần bật:

```python
USE_GUI = True
```

Khi train thường để:

```python
USE_GUI = False
```

vì train bằng GUI sẽ rất chậm.

---

## 13. Quy trình chạy đầy đủ từ đầu

Một quy trình chạy đầy đủ có thể thực hiện như sau:

```bash
cd /d/Picking-For-Robot
source rl_robot/Scripts/activate

python -m pip install -r requirements.txt

python ./test/test_grasp_env.py

python ./train_model/train_stage2_grasp_sb3.py

tensorboard --logdir logs_stage2

python plot_training_auto_report_stage2.py
```

Sau đó mở file:

```text
training_report_output/training_report_auto.md
```

để xem báo cáo tự động kèm biểu đồ.

---

## 14. Tóm tắt cách sử dụng

Để chạy dự án:

1. Mở Git Bash trong thư mục `Picking-For-Robot`.
2. Kích hoạt môi trường `rl_robot`.
3. Cài thư viện từ `requirements.txt`.
4. Chạy test để kiểm tra PyBullet và Panda.
5. Chạy train Stage.
6. Theo dõi kết quả bằng TensorBoard.
7. Vẽ biểu đồ và đưa vào báo cáo.
8. Test model sau khi train.

Quy trình lệnh ngắn gọn:

```bash
cd /d/Picking-For-Robot
source rl_robot/Scripts/activate
python -m pip install -r requirements.txt
python ./train_model/train_stage..._grasp_sb3.py
tensorboard --logdir logs_stage...
python plot_training_auto_report_stage....py
python ./test/test_grasp_env.py
```

---

## 15. Ghi chú về PyBullet và Panda

Dự án sử dụng robot **Franka Panda** có sẵn trong bộ dữ liệu URDF của PyBullet. Khi môi trường được khởi tạo, PyBullet sẽ load robot Panda, mặt phẳng, bàn và khối lập phương. Robot được điều khiển thông qua inverse kinematics để di chuyển end-effector đến vị trí mong muốn.

Trong quá trình train, GUI thường được tắt để tăng tốc độ huấn luyện:

```python
USE_GUI = False
```

Khi cần quan sát trực tiếp robot, nên chạy file test với GUI:

```python
USE_GUI = True
```

PyBullet giúp mô phỏng các yếu tố quan trọng như:

- chuyển động robot Panda,
- va chạm giữa gripper và vật thể,
- tiếp xúc trái/phải của gripper,
- trọng lực,
- lực ma sát,
- trạng thái vật sau khi grasp/lift.

Nhờ đó, hệ thống có thể kiểm tra được robot có thật sự gắp được vật hay chỉ chạm vào vật.
