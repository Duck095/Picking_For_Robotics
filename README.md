# Picking-For-Robot

![Python](https://img.shields.io/badge/Python-3.9.13-blue.svg) ![PyBullet](https://img.shields.io/badge/PyBullet-3.2.5-green.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg) ![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg) ![Status](https://img.shields.io/badge/Status-Research-yellow.svg)

Picking-For-Robot là dự án huấn luyện robot **Franka Panda** thực hiện thao tác gắp, nâng và chuẩn bị thả vật thể trong môi trường mô phỏng **PyBullet** bằng **Reinforcement Learning**.

Dự án sử dụng **PyBullet** để mô phỏng robot, bàn, vật thể, va chạm và tiếp xúc vật lý; sử dụng **Gymnasium** để xây dựng môi trường huấn luyện; và sử dụng **Stable-Baselines3 PPO** để train policy cho robot.

## Mục tiêu dự án

Mục tiêu của dự án là xây dựng pipeline huấn luyện robot theo từng stage, giúp robot học từ các kỹ năng đơn giản đến phức tạp hơn:

```text
Home
  ↓
Stage 1: Reach / Pre-grasp
  ↓
Stage 2: Grasp / Lift / Return Home
  ↓
Stage 3: Place / Drop
  ↓
Sorting / Palletizing
```

Hiện tại dự án tập trung vào hai stage chính:

| Stage | Mục tiêu |
|---|---|
| Stage 1 | Robot học di chuyển từ vị trí home đến vị trí pre-grasp phía trên vật |
| Stage 2 | Robot học căn chỉnh, hạ xuống, đóng gripper, giữ vật, nâng vật và đưa về home |

## Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python 3.9.13 | Ngôn ngữ chính của dự án |
| PyBullet | Mô phỏng vật lý, robot Panda, vật thể và va chạm |
| Gymnasium | Chuẩn hóa môi trường RL theo dạng Env |
| Stable-Baselines3 | Huấn luyện agent bằng PPO |
| PyTorch | Backend học sâu cho PPO |
| NumPy | Xử lý tọa độ, vector, observation và action |
| Matplotlib | Vẽ biểu đồ kết quả huấn luyện |
| TensorBoard | Theo dõi quá trình huấn luyện |
| Pandas | Xử lý dữ liệu log nếu cần |

## Yêu cầu môi trường

Dự án được chạy trong virtual environment tên là `rl_robot`.

Phiên bản Python khuyến nghị:

```bash
Python 3.9.13
```

Các thư viện chính được quản lý trong file:

```text
requirements.txt
```

## Cấu trúc thư mục

```text
Picking-For-Robot/
│
├── config/
│   ├── reach_env_config.py
│   └── grasp_env_config.py
│
├── env/
│   ├── reach_env.py
│   ├── grasp_env.py
│   ├── panda_controller.py
│   ├── reward_reach.py
│   ├── reward_grasp.py
│   └── camera.py
│
├── train_model/
│   ├── train_stage1_grasp_sb3.py
│   └── train_stage2_grasp_sb3.py
│
├── test/
│   ├── test_reach_env.py
│   └── test_grasp_env.py
│
├── script/
│   ├── stage1_reach/
│   │   ├── control_callbacks.py
│   │   ├── reach_debug_step_callback.py
│   │   ├── reach_debug_summary_callback.py
│   │   └── reach_tensorboard_callback.py
│   │
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
├── requirements.txt
└── README.md
```

Ý nghĩa các thư mục chính:

| Thư mục/File | Vai trò |
|---|---|
| `config/` | Chứa cấu hình môi trường, workspace, target, reward và substage |
| `env/` | Chứa môi trường PyBullet, controller Panda, reward và camera |
| `train_model/` | Chứa các file train model |
| `test/` | Chứa các file test model đã train |
| `script/` | Chứa callback lưu model, ghi log, summary và TensorBoard |
| `models/` | Lưu model `.zip` sau khi train |
| `debug_logs/` | Lưu debug log và summary log |
| `logs_stage1/`, `logs_stage2/` | Lưu TensorBoard log |
| `training_report_output/` | Lưu biểu đồ và báo cáo tự động |

## Cài đặt dự án

### 1. Clone repository

```bash
git clone <repository-url>
cd Picking-For-Robot
```

### 2. Tạo virtual environment

```bash
python -m venv rl_robot
```

### 3. Kích hoạt virtual environment

Trên Git Bash / Windows:

```bash
source rl_robot/Scripts/activate
```

Sau khi kích hoạt thành công, terminal sẽ hiển thị:

```bash
(rl_robot)
```

### 4. Kiểm tra Python đang dùng

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

Python phải trỏ vào thư mục:

```text
Picking-For-Robot/rl_robot/Scripts/python.exe
```

### 5. Cài thư viện

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu muốn cài thủ công các thư viện chính:

```bash
python -m pip install numpy matplotlib gymnasium pybullet stable-baselines3 tensorboard torch pandas tqdm rich
```

### 6. Kiểm tra thư viện

```bash
python -c "import pybullet; print('pybullet OK')"
python -c "import stable_baselines3; print('stable-baselines3 OK')"
python -c "import torch; print('torch OK')"
python -c "import gymnasium; print('gymnasium OK')"
python -c "import tensorboard; print('tensorboard OK')"
python -c "import matplotlib; print('matplotlib OK')"
```

## Chạy test môi trường

Trước khi train, nên chạy test để kiểm tra PyBullet, robot Panda và môi trường hoạt động đúng.

### Test Stage 1

```bash
python ./test/test_reach_env.py
```

### Test Stage 2

```bash
python ./test/test_grasp_env.py
```

Khi chạy test, chương trình có thể yêu cầu chọn substage, ví dụ:

```text
1A, 1B, 1C, 1D, 1E, 1F
2A, 2B, 2C, 2D
```

Nếu có model trong thư mục `models/`, chương trình sẽ load model tương ứng. Nếu chưa có model, có thể chạy bằng random policy để kiểm tra môi trường.

## Huấn luyện Stage 1

Stage 1 huấn luyện robot học cách di chuyển từ home đến vị trí pre-grasp phía trên vật.

Các substage của Stage 1:

```text
1A → 1B → 1C → 1D → 1E → 1F
```

Ý nghĩa:

| Substage | Mục tiêu |
|---|---|
| 1A | Reach cơ bản đến vùng gần vật |
| 1B | Cải thiện độ ổn định khi tiếp cận |
| 1C | Căn chỉnh XY tốt hơn |
| 1D | Cải thiện độ cao Z tại pre-grasp |
| 1E | Bắt đầu căn chỉnh yaw |
| 1F | Hoàn thiện pre-grasp với XY, Z, yaw và giữ ổn định |

Chạy train Stage 1:

```bash
python ./train_model/train_stage1_grasp_sb3.py
```

Sau khi train, các file được tạo gồm:

```text
models/stage1_pregrasp_mastery_1A.zip
debug_logs/stage1_1A_debug.log
debug_logs/stage1_1A_summary.log
logs_stage1/stage1_pregrasp_mastery_1A_0/events.out.tfevents...
```

## Huấn luyện Stage 2

Stage 2 huấn luyện robot học thao tác grasp, gồm căn chỉnh, hạ xuống, đóng gripper, giữ vật, nâng vật và đưa về home.

Các substage của Stage 2:

```text
2A → 2B → 2C → 2D
```

Ý nghĩa:

| Substage | Mục tiêu |
|---|---|
| 2A | Căn chỉnh, hạ xuống và đóng gripper |
| 2B | Giữ grasp ổn định |
| 2C | Nhấc vật lên khỏi mặt bàn |
| 2D | Nhấc vật và đưa về vị trí home/safe |

Chạy train Stage 2:

```bash
python ./train_model/train_stage2_grasp_sb3.py
```

Quá trình train sử dụng curriculum learning:

```text
Train 2A
  ↓
Load model 2A để train 2B
  ↓
Load model 2B để train 2C
  ↓
Load model 2C để train 2D
```

Sau khi train, các file model được lưu trong:

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

Nếu quá trình train bị dừng giữa chừng, dự án sẽ lưu file `_latest` để resume:

```text
stage2_grasp_mastery_2A_latest.zip
```

## Xem TensorBoard

Stage 1:

```bash
tensorboard --logdir logs_stage1
```

Stage 2:

```bash
tensorboard --logdir logs_stage2
```

Sau đó mở trình duyệt tại:

```text
http://localhost:6006
```

Các chỉ số TensorBoard quan trọng:

| Metric | Ý nghĩa |
|---|---|
| `episode/success_rate_100` | Tỷ lệ thành công trong 100 episode gần nhất |
| `episode/reward_mean_100` | Reward trung bình |
| `episode/ep_len_mean_100` | Độ dài trung bình của episode |
| `grasp/dual_contact_rate_100` | Tỷ lệ hai ngón gripper cùng chạm vật |
| `grasp/grasp_established_rate_100` | Tỷ lệ grasp thật sự được thiết lập |
| `grasp/lift_delta_mean_100` | Độ nâng trung bình của vật |
| `grasp/home_error_mean_100` | Sai số khi đưa vật về home |
| `train/explained_variance` | Độ ổn định của critic/value function |
| `train/value_loss` | Loss của value function |
| `train/policy_gradient_loss` | Loss của policy |
| `train/approx_kl` | Độ thay đổi policy sau mỗi lần update |
| `train/entropy_loss` | Mức độ exploration của policy |

## Vẽ biểu đồ cho báo cáo

Sau khi train xong, có thể chạy script vẽ biểu đồ tự động.

Ví dụ:

```bash
python plot_training_auto_report_stage2.py
```

Script sẽ quét:

```text
debug_logs/
logs_stage1/
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
    ├── stage1/
    │   ├── 1A/
    │   ├── 1B/
    │   ├── 1C/
    │   ├── 1D/
    │   ├── 1E/
    │   └── 1F/
    │
    └── stage2/
        ├── 2A/
        ├── 2B/
        ├── 2C/
        └── 2D/
```

Các biểu đồ nên dùng trong báo cáo:

| Biểu đồ | Ý nghĩa |
|---|---|
| `01_learning_curve_report.png` | Reward và success rate |
| `02_error_reduction_report.png` | Sai số vị trí |
| `03_success_rate_report.png` | Tỷ lệ thành công |
| `04_episode_length_report.png` | Số bước mỗi episode |
| `05_yaw_error_report.png` | Sai số yaw |
| `06_grasp_contact_report.png` | Chất lượng grasp/contact, dùng cho Stage 2 |
| `07_grip_width_report.png` | Độ mở gripper, dùng cho Stage 2 |
| `08_lift_delta_report.png` | Độ nâng vật, dùng cho Stage 2C/2D |
| `09_home_error_report.png` | Sai số về home, dùng cho Stage 2D |
| `tb_train_explained_variance_report.png` | Độ ổn định của PPO |

## Resume training

Nếu quá trình huấn luyện bị dừng, chạy lại file train:

```bash
python ./train_model/train_stage2_grasp_sb3.py
```

Script sẽ tự phát hiện model `_latest` và tiếp tục từ substage đang dở.

Ví dụ:

```text
models/stage2_grasp_mastery_2A_latest.zip
```

## Một số lỗi thường gặp

### PyBullet không import được

Lỗi thường gặp:

```text
ImportError: DLL load failed while importing pybullet
```

Cách kiểm tra:

```bash
which python
python -c "import sys; print(sys.executable)"
```

Cài lại PyBullet:

```bash
python -m pip install --upgrade --force-reinstall pybullet
```

### Thiếu thư viện

```bash
python -m pip install -r requirements.txt
```

Hoặc cài riêng:

```bash
python -m pip install ten_thu_vien
```

### TensorBoard không chạy

```bash
python -m pip install tensorboard
tensorboard --logdir logs_stage2
```

### Không thấy GUI PyBullet

Khi train nên để:

```python
USE_GUI = False
```

Khi muốn quan sát robot bằng GUI, chạy file test và bật:

```python
USE_GUI = True
```

## Quy trình chạy nhanh

```bash
cd /d/Picking-For-Robot
source rl_robot/Scripts/activate

python -m pip install -r requirements.txt

python ./test/test_grasp_env.py
python ./train_model/train_stage2_grasp_sb3.py

tensorboard --logdir logs_stage2
python plot_training_auto_report_stage2.py
```

## Ghi chú về PyBullet và Panda

Dự án sử dụng robot **Franka Panda** có sẵn trong bộ dữ liệu URDF của PyBullet. Khi môi trường được khởi tạo, PyBullet sẽ load robot Panda, mặt phẳng, bàn và khối lập phương.

Robot được điều khiển thông qua **inverse kinematics** để đưa end-effector đến vị trí mong muốn. Trong Stage 2, gripper được điều khiển để đóng/mở, tạo tiếp xúc với vật và kiểm tra trạng thái grasp.

PyBullet giúp mô phỏng các yếu tố quan trọng như:

- chuyển động của robot Panda,
- trọng lực,
- va chạm giữa gripper và vật thể,
- tiếp xúc trái/phải của gripper,
- lực ma sát,
- trạng thái vật sau khi grasp/lift.

Nhờ đó, dự án có thể kiểm tra robot có thật sự gắp được vật hay chỉ chạm vào vật.

