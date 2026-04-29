# Face Recognition with Partial Facial Visibility

**Nhóm 12** - Dự Án Sinh Trắc Học (Biometric Project)

Hệ thống nhận dạng khuôn mặt có khả năng xác định các cá nhân ngay cả khi một phần khuôn mặt của họ bị che phủ (mặt nạ, kính mát, mũ). Dựa trên fine-tuning **ArcFace** với backbone **MobileNetV2/V3**.

---

## 👥 Thành Viên Nhóm

| STT | Tên | MSSV |
|-----|-----|------|
| 1 | Trần Anh Tuấn | 20235628 |
| 2 | Nguyễn Thiện Khải | 20235595 |
| 3 | Nguyễn Đức Anh | 20235585 |
| 4 | Phạm Nguyễn Huy Tuấn | 20235627 |

---

## 📋 Mục Lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn nhanh](#hướng-dẫn-nhanh)
- [Chạy các script](#chạy-các-script)
- [Tập dữ liệu](#tập-dữ-liệu)
- [Phương pháp tiếp cận](#phương-pháp-tiếp-cận)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Kết quả dự kiến](#kết-quả-dự-kiến)
- [Sử dụng và tùy chỉnh](#sử-dụng-và-tùy-chỉnh)
- [Đánh giá mô hình](#đánh-giá-mô-hình)
- [Ghi chú quan trọng](#ghi-chú-quan-trọng)
- [License](#license)

---

## 🎯 Tổng Quan Dự Án

Dự án này phát triển một hệ thống nhận dạng khuôn mặt mạnh mẽ có khả năng xử lý các tình huống trong đó một phần của khuôn mặt bị che phủ. Mục tiêu chính là:

- ✅ Nhận dạng chính xác trên khuôn mặt đầy đủ (Full Face): ~99% accuracy
- ✅ Nhận dạng trên khuôn mặt bị che phủ (Masked Face): ~85-92% accuracy
- ✅ Model nhẹ, có thể triển khai trên thiết bị di động
- ✅ Thời gian suy luận nhanh (< 50ms trên CPU)

---

## 📁 Cấu Trúc Dự Án

```
biometric-project/
├── configs/                          # Cấu hình YAML cho các thí nghiệm
│   ├── data/
│   │   └── lfw.yaml                 # Config tập dữ liệu LFW
│   ├── model/
│   │   ├── mobilenet_v2_arcface.yaml
│   │   └── mobilenet_v3_arcface.yaml
│   └── train/
│       └── exp_001_baseline.yaml    # Config training cơ bản
│
├── data/                            # Dữ liệu
│   ├── raw/
│   │   └── lfw/                     # Tập dữ liệu thô
│   ├── processed/                   # Dữ liệu đã xử lý
│   ├── embeddings/                  # Embedding đã trích xuất
│   └── splits/                      # Train/val/test splits
│
├── src/                             # Mã nguồn chính
│   ├── data/
│   │   ├── dataset.py              # FaceDataset, DataLoader
│   │   ├── preprocessing.py        # Xử lý ảnh
│   │   └── mask_augment.py         # Tạo mặt nạ tổng hợp
│   ├── models/
│   │   ├── backbone.py             # MobileNet backbone
│   │   ├── arcface_head.py         # ArcFace head
│   │   └── face_recognizer.py      # Model hoàn chỉnh
│   ├── losses/
│   │   └── arcface_loss.py         # Hàm loss ArcFace
│   ├── metrics/
│   │   └── verification.py         # Độ chính xác, EER, ROC
│   ├── training/
│   │   └── trainer.py              # Vòng lặp training
│   ├── inference/
│   │   └── embedding_extractor.py  # Trích xuất embedding
│   └── utils/
│       ├── config.py               # Quản lý config
│       ├── logging.py              # Logging
│       └── visualization.py        # Trực quan hóa dữ liệu
│
├── scripts/                         # Script chính để chạy
│   ├── train.py                    # Huấn luyện mô hình
│   ├── evaluate.py                 # Đánh giá mô hình
│   ├── extract_embeddings.py       # Trích xuất embedding
│   ├── preprocess_data.py          # Tiền xử lý dữ liệu
│   └── create_splits.py            # Tạo train/val/test splits
│
├── experiments/                     # Kết quả thí nghiệm
│   └── exp_001_arcface_mobilenetv2/
│       ├── checkpoints/             # Checkpoint mô hình
│       ├── logs/                    # TensorBoard logs
│       └── results/                 # Kết quả đánh giá
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_embedding_visualization.ipynb
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   └── test_models.py
│
├── outputs/                         # Output cuối cùng
├── Dockerfile                       # Docker configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # Tài liệu này
```

---

## 🔧 Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8+
- CUDA 11.8+ (tùy chọn, cho GPU training)
- 8GB+ RAM (tối thiểu), 16GB+ khuyên dùng
- Disk space: ~50GB cho LFW dataset

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd biometric-project
```

### Bước 2: Tạo Virtual Environment

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cài Đặt Với Docker (Tùy Chọn)

```bash
docker build -t biometric-project .
docker run -it --gpus all biometric-project
```

---

## 🚀 Hướng Dẫn Nhanh

Dưới đây là quy trình nhanh để chạy dự án. Để có hướng dẫn chi tiết cho từng script, xem phần [Chạy các script](#chạy-các-script).

### 1. Tải Tập Dữ Liệu

Tải LFW dataset từ: http://vis-www.cs.umass.edu/lfw/

```bash
# Giải nén vào data/raw/lfw/
cd data/raw/lfw/
# Bạn sẽ có cấu trúc như:
# lfw/
#   ├── lfw-deepfunneled/
#   │   ├── Aaron_Eckhart/
#   │   ├── Aaron_Guiel/
#   │   └── ...
#   └── pairs.txt
```


### 2. Tiền Xử Lý Dữ Liệu

```bash
python scripts/preprocess_data.py --config configs/data/lfw.yaml
```

Chi tiết: Xem [Tiền Xử Lý Dữ Liệu](#1-tiền-xử-lý-dữ-liệu-preprocess_datappy)

### 3. Tạo Splits (Train/Val/Test)

```bash
python scripts/create_splits.py --config configs/data/lfw.yaml
```

Chi tiết: Xem [Tạo Train/Val/Test Splits](#2-tạo-trainvaltest-splits-create_splitspy)

### 4. Huấn Luyện Mô Hình

```bash
# Huấn luyện cơ bản
python scripts/train.py --config configs/train/exp_001_baseline.yaml

# Theo dõi quá trình training
tensorboard --logdir experiments/exp_001_arcface_mobilenetv2/logs
```

Chi tiết: Xem [Huấn Luyện Mô Hình](#3-huấn-luyện-mô-hình-trainpy)

### 5. Đánh Giá Mô Hình

```bash
python scripts/evaluate.py --exp experiments/exp_001_arcface_mobilenetv2
```

Chi tiết: Xem [Đánh Giá Mô Hình](#4-đánh-giá-mô-hình-evaluatepy)

Output: Metrics như Accuracy, EER, TAR@FAR, ROC curves

### 6. Trích Xuất Embedding

```bash
python scripts/extract_embeddings.py \
    --model-path experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth \
    --input-dir data/processed/ \
    --output-dir data/embeddings/
```

---

## 🚀 Chạy Các Script

Phần này cung cấp hướng dẫn chi tiết cho từng script chính trong dự án, bao gồm các tham số, ví dụ sử dụng, và output mong đợi.

### 1. Tiền Xử Lý Dữ Liệu (`preprocess_data.py`)

**Mục đích**: Đọc ảnh gốc, resize, chuẩn hóa, lưu vào thư mục processed

**Cú pháp cơ bản**:
```bash
python scripts/preprocess_data.py --config configs/data/lfw.yaml
```

**Các tham số**:

| Tham số | Mô tả | Ví dụ |
|--------|-------|-------|
| `--config` | Đường dẫn file config | `configs/data/lfw.yaml` |
| `--input-dir` | Thư mục chứa ảnh gốc | `data/raw/lfw/` |
| `--output-dir` | Thư mục lưu ảnh đã xử lý | `data/processed/` |
| `--size` | Kích thước ảnh output | `112` (default) |
| `--num-workers` | Số workers xử lý song song | `4` |

**Ví dụ nâng cao**:
```bash
# Sử dụng 8 workers, resize ảnh 128x128
python scripts/preprocess_data.py \
    --config configs/data/lfw.yaml \
    --size 128 \
    --num-workers 8

# Từ thư mục khác
python scripts/preprocess_data.py \
    --config configs/data/lfw.yaml \
    --input-dir data/raw/casia/ \
    --output-dir data/processed_casia/
```

**Output**: 
- `data/processed/` - Ảnh đã xử lý (112x112 hoặc tuỳ chỉnh)
- `data/processed/info.json` - Metadata (tổng số ảnh, người, etc.)

---

### 2. Tạo Train/Val/Test Splits (`create_splits.py`)

**Mục đích**: Chia dữ liệu thành train/val/test set

**Cú pháp cơ bản**:
```bash
python scripts/create_splits.py --config configs/data/lfw.yaml
```

**Các tham số**:

| Tham số | Mô tả | Default |
|--------|-------|---------|
| `--config` | Đường dẫn file config | - |
| `--data-dir` | Thư mục dữ liệu | `data/processed/` |
| `--output-dir` | Nơi lưu splits | `data/splits/` |
| `--train-ratio` | Tỷ lệ train | `0.7` |
| `--val-ratio` | Tỷ lệ validation | `0.15` |
| `--test-ratio` | Tỷ lệ test | `0.15` |
| `--seed` | Random seed | `42` |

**Ví dụ**:
```bash
# Tạo splits với tỷ lệ 80/10/10
python scripts/create_splits.py \
    --config configs/data/lfw.yaml \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# Với seed khác (để có splits khác)
python scripts/create_splits.py \
    --config configs/data/lfw.yaml \
    --seed 2024
```

**Output**:
- `data/splits/train/` - Các thư mục người dùng cho training
- `data/splits/val/` - Validation set
- `data/splits/test/` - Test set

---

### 3. Huấn Luyện Mô Hình (`train.py`)

**Mục đích**: Huấn luyện mô hình từ đầu hoặc từ checkpoint

**Cú pháp cơ bản**:
```bash
python scripts/train.py --config configs/train/exp_001_baseline.yaml
```

**Các tham số chính**:

| Tham số | Mô tả | Ví dụ |
|--------|-------|-------|
| `--config` | Đường dẫn training config | `configs/train/exp_001_baseline.yaml` |
| `--experiment-name` | Tên experiment | `exp_002_test` |
| `--resume` | Checkpoint để resume từ | `experiments/exp_001/checkpoints/last.pth` |
| `--num-epochs` | Số epochs | `50` |
| `--batch-size` | Batch size | `128` |
| `--lr` | Learning rate | `0.01` |
| `--device` | GPU hoặc CPU | `cuda:0` hoặc `cpu` |
| `--num-workers` | Số data loader workers | `4` |
| `--fp16` | Sử dụng mixed precision | (flag) |
| `--debug` | Chế độ debug | (flag) |

**Ví dụ cơ bản**:
```bash
# Huấn luyện mô hình cơ bản
python scripts/train.py --config configs/train/exp_001_baseline.yaml
```

**Ví dụ nâng cao**:
```bash
# Huấn luyện với batch size lớn, mixed precision, 50 epochs
python scripts/train.py \
    --config configs/train/exp_001_baseline.yaml \
    --experiment-name exp_002_large_batch \
    --batch-size 256 \
    --num-epochs 50 \
    --fp16

# Resume từ checkpoint
python scripts/train.py \
    --config configs/train/exp_001_baseline.yaml \
    --experiment-name exp_001_resumed \
    --resume experiments/exp_001_arcface_mobilenetv2/checkpoints/last.pth

# Sử dụng learning rate khác
python scripts/train.py \
    --config configs/train/exp_001_baseline.yaml \
    --lr 0.05 \
    --experiment-name exp_003_high_lr

# Chỉ dùng CPU (chậm nhưng không cần GPU)
python scripts/train.py \
    --config configs/train/exp_001_baseline.yaml \
    --device cpu \
    --batch-size 32  # Giảm batch size vì thiếu VRAM

# Chế độ debug (xem 1 batch)
python scripts/train.py \
    --config configs/train/exp_001_baseline.yaml \
    --debug
```

**Monitoring Training**:

Sử dụng TensorBoard để theo dõi training:

```bash
# Terminal 1: Chạy training
python scripts/train.py --config configs/train/exp_001_baseline.yaml

# Terminal 2: Chạy TensorBoard
tensorboard --logdir experiments/exp_001_arcface_mobilenetv2/logs

# Mở browser: http://localhost:6006
```

**Output**:
- `experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth` - Mô hình tốt nhất
- `experiments/exp_001_arcface_mobilenetv2/checkpoints/last.pth` - Checkpoint cuối cùng
- `experiments/exp_001_arcface_mobilenetv2/logs/` - TensorBoard logs
- `experiments/exp_001_arcface_mobilenetv2/config.yaml` - Config đã sử dụng

---

### 4. Đánh Giá Mô Hình (`evaluate.py`)

**Mục đích**: Tính các metrics (Accuracy, EER, TAR@FAR, ROC) trên test set

**Cú pháp cơ bản**:
```bash
python scripts/evaluate.py --exp experiments/exp_001_arcface_mobilenetv2
```

**Các tham số**:

| Tham số | Mô tả | Default |
|--------|-------|---------|
| `--exp` | Đường dẫn experiment folder | - |
| `--checkpoint` | Tên checkpoint (`best` hoặc `last`) | `best` |
| `--test-pairs` | File chứa test pairs | `data/splits/test_pairs.txt` |
| `--batch-size` | Batch size để extract embedding | `256` |
| `--device` | GPU hoặc CPU | `cuda:0` |
| `--output-dir` | Thư mục lưu results | `<exp>/results/` |

**Ví dụ**:
```bash
# Đánh giá mô hình tốt nhất
python scripts/evaluate.py --exp experiments/exp_001_arcface_mobilenetv2

# Dùng checkpoint last
python scripts/evaluate.py \
    --exp experiments/exp_001_arcface_mobilenetv2 \
    --checkpoint last

# Đánh giá trên tập test khác
python scripts/evaluate.py \
    --exp experiments/exp_001_arcface_mobilenetv2 \
    --test-pairs data/splits/custom_test_pairs.txt

# Dùng CPU (chậm)
python scripts/evaluate.py \
    --exp experiments/exp_001_arcface_mobilenetv2 \
    --device cpu

# Batch size lớn (nếu GPU có đủ VRAM)
python scripts/evaluate.py \
    --exp experiments/exp_001_arcface_mobilenetv2 \
    --batch-size 512
```

**Output**:
- `experiments/exp_001_arcface_mobilenetv2/results/metrics.json` - Metrics (Accuracy, EER, TAR@FAR)
- `experiments/exp_001_arcface_mobilenetv2/results/roc_curve.png` - ROC curve plot
- `experiments/exp_001_arcface_mobilenetv2/results/embeddings.npy` - Embeddings của test set

**Xem kết quả**:
```bash
# Xem metrics
cat experiments/exp_001_arcface_mobilenetv2/results/metrics.json | jq .

# Hoặc dùng Python
python -c "import json; print(json.load(open('experiments/exp_001_arcface_mobilenetv2/results/metrics.json')))"
```

---

### 5. Trích Xuất Embedding (`extract_embeddings.py`)

**Mục đích**: Trích xuất embedding từ tập ảnh lớn, lưu vào file

**Cú pháp cơ bản**:
```bash
python scripts/extract_embeddings.py \
    --model-path experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth \
    --input-dir data/processed/ \
    --output-dir data/embeddings/
```

**Các tham số**:

| Tham số | Mô tả | Default |
|--------|-------|---------|
| `--model-path` | Đường dẫn checkpoint | - |
| `--input-dir` | Thư mục chứa ảnh | - |
| `--output-dir` | Nơi lưu embeddings | `data/embeddings/` |
| `--batch-size` | Batch size | `256` |
| `--device` | GPU hoặc CPU | `cuda:0` |
| `--num-workers` | Data loader workers | `4` |
| `--save-format` | Format lưu (`npy`, `h5`, `pt`) | `npy` |

**Ví dụ**:
```bash
# Trích xuất cơ bản
python scripts/extract_embeddings.py \
    --model-path experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth \
    --input-dir data/processed/

# Với batch size lớn, GPU
python scripts/extract_embeddings.py \
    --model-path experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth \
    --input-dir data/processed/ \
    --batch-size 512 \
    --device cuda:0

# Trích xuất từ tập CASIA-WebFace
python scripts/extract_embeddings.py \
    --model-path experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth \
    --input-dir data/processed_casia/ \
    --output-dir data/embeddings_casia/

# Lưu dưới dạng HDF5 (nén, tiết kiệm disk)
python scripts/extract_embeddings.py \
    --model-path experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth \
    --input-dir data/processed/ \
    --save-format h5
```

**Output**:
- `data/embeddings/embeddings.npy` - Embedding matrix (N, 512)
- `data/embeddings/image_paths.txt` - Danh sách ảnh tương ứng

---

### 6. Xử Lý Dữ Liệu Với Mask Augmentation

**Trong huấn luyện**, mask augmentation tự động được áp dụng. Nhưng bạn có thể tạo masked dataset riêng:

```bash
# Ví dụ: Tạo synthetic masked faces
python -c "
from src.data.mask_augment import MaskAugmenter
import cv2
from pathlib import Path

augmenter = MaskAugmenter()

for img_path in Path('data/processed/').glob('*/*.jpg'):
    img = cv2.imread(str(img_path))
    masked_img = augmenter(img)
    
    # Lưu ảnh masked
    output_path = str(img_path).replace('processed', 'processed_masked')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, masked_img)
"
```

---

## �📊 Tập Dữ Liệu

### LFW (Labeled Faces in the Wild)

- **Kích thước**: ~13,000 ảnh từ hơn 5,700 người
- **Đặc điểm**: Ảnh thực tế, đa dạng môi trường chiếu sáng, tư thế
- **Mục đích chính**: Baseline và evaluation

### Các Tập Dữ Liệu Bổ Sung (Tuỳ Chọn)

- **CASIA-WebFace**: Large-scale (~500K ảnh từ 10,575 người)
- **RMFRD/MFR2**: Thực tế với mặt nạ
- **Synthetic Masked**: Tạo bằng MaskTheFace library

### Tỷ Lệ Chia Dữ Liệu

```
Train: 70%
Val:   15%
Test:  15%
```

---

## 🧠 Phương Pháp Tiếp Cận

### 1. Thiết Kế Kiến Trúc

```
Input Image
    ↓
[MobileNetV2/V3 Backbone] → Trích xuất features (512-d)
    ↓
[L2 Normalization] → Normalize embeddings
    ↓
[ArcFace Head] → Classification layer với ArcFace loss
    ↓
Class Logits
```

### 2. Loss Function: ArcFace

ArcFace là một metric learning loss function được thiết kế để tối ưu hóa các embedding cho face recognition.

**Công thức**:
```
Loss = -log( exp(s*cos(θ + m)) / (exp(s*cos(θ + m)) + Σ exp(s*cos(θ_j))) )
```

Các tham số:
- **margin (m)**: 0.5 (khoảng cách margin giữa các lớp)
- **scale (s)**: 64 (độ nhạy của logit)

### 3. Chiến Lược Huấn Luyện

**Giai đoạn 1 - Khởi tạo (Epoch 1-10)**:
- Freeze backbone (pretrained ImageNet)
- Chỉ training ArcFace head
- Learning rate: 0.1

**Giai đoạn 2 - Fine-tune (Epoch 11-30)**:
- Unfreeze backbone
- Full training với mask augmentation
- Learning rate: Cosine annealing từ 0.1 đến 1e-5

### 4. Data Augmentation

Áp dụng augmentation để tăng robustness:

```python
# Các augmentation được sử dụng
transforms = [
    Resize(112, 112),           # Chuẩn hóa kích thước
    RandomHorizontalFlip(0.5),  # Lật ngang
    ColorJitter(0.2, 0.2, 0.2), # Thay đổi màu sắc
    MaskAugmentation(),         # Thêm mặt nạ tổng hợp
    RandomRotation(10),         # Xoay nhẹ
    Normalize(),                # Chuẩn hóa
]
```

### 5. Verification Protocol

Để so sánh hai khuôn mặt:

1. **Trích xuất embedding**: Input ảnh → Embedding 512-d
2. **Tính khoảng cách**: Cosine distance hoặc L2 distance
3. **Quyết định**: Nếu distance < threshold → Cùng người

```python
# Ví dụ
embedding1 = model.extract_embedding(image1)  # shape: (512,)
embedding2 = model.extract_embedding(image2)  # shape: (512,)

# Cosine similarity
similarity = torch.nn.functional.cosine_similarity(
    embedding1.unsqueeze(0),
    embedding2.unsqueeze(0)
)  # output: 0.0 ~ 1.0

if similarity > 0.5:
    print("Cùng người")
else:
    print("Khác người")
```

---

## 🏗️ Kiến Trúc Mô Hình

### Backbone Options

#### MobileNetV2
- **Tham số**: ~3.5M
- **Kích thước mô hình**: ~14MB
- **Tốc độ**: Nhanh nhất, phù hợp mobile
- **Accuracy**: Tốt

#### MobileNetV3
- **Tham số**: ~5.4M
- **Kích thước mô hình**: ~22MB
- **Tốc độ**: Nhanh
- **Accuracy**: Cao hơn MobileNetV2

### Embedding Dimension

- Được trích xuất từ layer cuối cùng của backbone
- Kích thước: **512 dimensions**
- Format: L2-normalized vector

### Classification Head (Training Only)

```python
# Chỉ được dùng trong training
class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=5000):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.loss = ArcFaceLoss(margin=0.5, scale=64)
```

---

## 📈 Kết Quả Dự Kiến

### Hiệu Suất Trên LFW

| Metric | Full Face | Masked Face |
|--------|-----------|-------------|
| **Accuracy** | ~99% | ~85-92% |
| **EER** | <1% | 5-10% |
| **TAR@FAR=0.1%** | >99% | >85% |

### Hiệu Năng Mô Hình

| Tiêu Chí | MobileNetV2 | MobileNetV3 |
|----------|------------|------------|
| **Kích thước** | ~14MB | ~22MB |
| **Tham số** | 3.5M | 5.4M |
| **Thời gian suy luận (CPU)** | ~40ms | ~50ms |
| **Thời gian suy luận (GPU)** | ~5ms | ~6ms |

---

## 🛠️ Sử Dụng và Tùy Chỉnh

### Thay Đổi Cấu Hình Huấn Luyện

Chỉnh sửa file `configs/train/exp_001_baseline.yaml`:

```yaml
training:
  num_epochs: 50          # Tăng epochs
  batch_size: 256         # Tăng batch size (cần GPU tốt)
  
  optimizer:
    lr: 0.05              # Giảm learning rate
    weight_decay: 1e-4
    
  scheduler:
    warmup_epochs: 5      # Tăng warmup
    min_lr: 5e-6
```

### Sử Dụng Backbone Khác

```bash
# Sử dụng MobileNetV3 thay vì MobileNetV2
python scripts/train.py \
    --config configs/train/exp_001_baseline.yaml \
    --model-config configs/model/mobilenet_v3_arcface.yaml
```

### Resume Training Từ Checkpoint

```bash
python scripts/train.py \
    --config configs/train/exp_001_baseline.yaml \
    --resume experiments/exp_001_arcface_mobilenetv2/checkpoints/last.pth
```

### Inference Trên Ảnh Mới

```python
from pathlib import Path
from src.models.face_recognizer import build_model
from src.data.preprocessing import preprocess_image
import torch

# Load model
model = build_model(config, num_classes=None)
model.load_state_dict(torch.load('checkpoints/best.pth'))
model.eval()

# Load ảnh
image = preprocess_image('path/to/image.jpg')

# Trích xuất embedding
with torch.no_grad():
    embedding = model.extract_embedding(image)  # (1, 512)

print(f"Embedding shape: {embedding.shape}")
```

---

## 📊 Đánh Giá Mô Hình

### Các Metric Chính

1. **Accuracy**: Tỷ lệ các cặp được phân loại đúng
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **EER (Equal Error Rate)**: Điểm mà FAR = FRR
   - FAR (False Accept Rate): Tỷ lệ chấp nhận sai
   - FRR (False Reject Rate): Tỷ lệ từ chối sai

3. **TAR@FAR**: True Accept Rate tại một mức FAR cụ thể
   - TAR@FAR=0.1%: Tỷ lệ nhận dạng đúng khi FAR=0.1%

4. **ROC Curve**: Plot giữa TPR vs FPR

### Xem Kết Quả Chi Tiết

```bash
# Tất cả mô hình
ls -la experiments/exp_001_arcface_mobilenetv2/results/

# Metrics
cat experiments/exp_001_arcface_mobilenetv2/results/metrics.json

# Plots
# ROC curve, Confusion matrix, Distribution plots...
```

---

## � Hướng Dẫn Sử Dụng Chi Tiết

### 1. Verification - So Sánh Hai Khuôn Mặt

Xác định xem hai ảnh khuôn mặt có cùng người không.

**Code ví dụ**:

```python
import cv2
import torch
import numpy as np
from pathlib import Path
from src.models.face_recognizer import build_model
from src.data.preprocessing import preprocess_image

class FaceVerifier:
    def __init__(self, model_path, config, threshold=0.5):
        """
        Args:
            model_path: Đường dẫn đến checkpoint mô hình
            config: Configuration dict
            threshold: Cosine similarity threshold (0.0 ~ 1.0)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(config, num_classes=None)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold

    def extract_embedding(self, image_path):
        """Trích xuất embedding từ ảnh"""
        # Load và preprocess ảnh
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Assume preprocess_image trả về tensor (1, 3, 112, 112)
        image_tensor = preprocess_image(image).to(self.device)
        
        # Trích xuất embedding
        with torch.no_grad():
            embedding = self.model.extract_embedding(image_tensor)
        
        return embedding.cpu().numpy()

    def verify(self, image1_path, image2_path):
        """So sánh hai ảnh
        
        Returns:
            {
                'match': bool,
                'similarity': float (0.0 ~ 1.0),
                'distance': float
            }
        """
        emb1 = self.extract_embedding(image1_path)
        emb2 = self.extract_embedding(image2_path)
        
        # Tính cosine similarity
        similarity = np.dot(emb1, emb2.T) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        
        return {
            'match': float(similarity[0, 0]) > self.threshold,
            'similarity': float(similarity[0, 0]),
            'distance': float(1 - similarity[0, 0])
        }

# Sử dụng
verifier = FaceVerifier(
    model_path='experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth',
    config=config,
    threshold=0.5
)

result = verifier.verify('image1.jpg', 'image2.jpg')
print(f"Match: {result['match']}")
print(f"Similarity: {result['similarity']:.4f}")
print(f"Distance: {result['distance']:.4f}")
```

### 2. Identification - Tìm Người Tương Tự Trong Database

Tìm người giống nhất trong danh sách mẫu.

**Code ví dụ**:

```python
class FaceIdentifier:
    def __init__(self, model_path, config):
        """Khởi tạo face identifier"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(config, num_classes=None)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Database: {person_id: embedding}
        self.database = {}

    def add_person(self, person_id, image_path, name=None):
        """Thêm người vào database"""
        embedding = self.extract_embedding(image_path)
        self.database[person_id] = {
            'embedding': embedding,
            'name': name,
            'image_path': image_path
        }

    def identify(self, query_image_path, top_k=5):
        """Tìm người giống nhất
        
        Returns:
            List of (person_id, name, similarity)
        """
        query_emb = self.extract_embedding(query_image_path)
        
        similarities = []
        for person_id, data in self.database.items():
            sim = np.dot(query_emb, data['embedding'].T) / (
                np.linalg.norm(query_emb) * np.linalg.norm(data['embedding']) + 1e-8
            )
            similarities.append({
                'person_id': person_id,
                'name': data['name'],
                'similarity': float(sim[0, 0])
            })
        
        # Sắp xếp theo similarity giảm dần
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def extract_embedding(self, image_path):
        """Trích xuất embedding"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.extract_embedding(image_tensor)
        
        return embedding.cpu().numpy()

# Sử dụng
identifier = FaceIdentifier(
    model_path='experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth',
    config=config
)

# Thêm người vào database
identifier.add_person(1, 'person1_face1.jpg', 'Person 1')
identifier.add_person(2, 'person2_face1.jpg', 'Person 2')
identifier.add_person(3, 'person3_face1.jpg', 'Person 3')

# Tìm người giống nhất
results = identifier.identify('query_image.jpg', top_k=3)
for i, result in enumerate(results, 1):
    print(f"{i}. {result['name']} (ID: {result['person_id']})")
    print(f"   Similarity: {result['similarity']:.4f}\n")
```

### 3. Enrollment - Thêm Người Mới Vào Hệ Thống

Đăng ký khuôn mặt mới với nhiều ảnh.

**Code ví dụ**:

```python
from pathlib import Path

class FaceEnroller:
    def __init__(self, model_path, config, db_path='face_database.pkl'):
        """
        Args:
            model_path: Đường dẫn checkpoint
            config: Configuration
            db_path: Đường dẫn lưu database
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(config, num_classes=None)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.db_path = db_path
        self.database = self.load_database()

    def enroll_person(self, person_id, image_paths, name=None, metadata=None):
        """Đăng ký khuôn mặt mới với nhiều ảnh
        
        Args:
            person_id: ID duy nhất cho người
            image_paths: List đường dẫn ảnh
            name: Tên người
            metadata: Thông tin bổ sung (tuổi, giới tính, etc.)
        """
        embeddings = []
        
        for image_path in image_paths:
            try:
                embedding = self.extract_embedding(image_path)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Lỗi xử lý ảnh {image_path}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("Không thể trích xuất embedding từ bất kỳ ảnh nào")
        
        # Tính embedding trung bình
        mean_embedding = np.mean(embeddings, axis=0)
        
        self.database[person_id] = {
            'name': name,
            'embeddings': embeddings,  # Tất cả embeddings
            'mean_embedding': mean_embedding,  # Embedding trung bình
            'num_samples': len(embeddings),
            'metadata': metadata or {},
            'enrolled_date': str(Path.ctime(Path(image_paths[0])))
        }
        
        # Lưu database
        self.save_database()
        
        print(f"✓ Đã đăng ký {name} (ID: {person_id}) với {len(embeddings)} ảnh")

    def verify_enrollment(self, person_id, test_image_path, threshold=0.5):
        """Xác minh enrollment bằng ảnh mới
        
        Returns:
            {
                'is_same': bool,
                'similarity': float,
                'confidence': float
            }
        """
        if person_id not in self.database:
            raise ValueError(f"Người ID {person_id} không tồn tại")
        
        test_emb = self.extract_embedding(test_image_path)
        mean_emb = self.database[person_id]['mean_embedding']
        
        similarity = np.dot(test_emb, mean_emb.T) / (
            np.linalg.norm(test_emb) * np.linalg.norm(mean_emb) + 1e-8
        )
        
        return {
            'is_same': float(similarity[0, 0]) > threshold,
            'similarity': float(similarity[0, 0]),
            'confidence': float(similarity[0, 0])
        }

    def extract_embedding(self, image_path):
        """Trích xuất embedding"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.extract_embedding(image_tensor)
        
        return embedding.cpu().numpy()

    def save_database(self):
        """Lưu database"""
        import pickle
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)

    def load_database(self):
        """Tải database"""
        import pickle
        if Path(self.db_path).exists():
            with open(self.db_path, 'rb') as f:
                return pickle.load(f)
        return {}

# Sử dụng
enroller = FaceEnroller(
    model_path='experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth',
    config=config
)

# Đăng ký người mới
enroller.enroll_person(
    person_id=1,
    image_paths=['person1_photo1.jpg', 'person1_photo2.jpg', 'person1_photo3.jpg'],
    name='Nguyễn Văn A',
    metadata={'age': 25, 'gender': 'M'}
)

# Xác minh với ảnh mới
result = enroller.verify_enrollment(
    person_id=1,
    test_image_path='person1_new_photo.jpg'
)
print(f"Xác minh: {result['is_same']}, Độ tin cậy: {result['confidence']:.4f}")
```

### 4. Batch Processing - Xử Lý Nhiều Ảnh

**Code ví dụ**:

```python
from pathlib import Path
from tqdm import tqdm

class FaceBatchProcessor:
    def __init__(self, model_path, config):
        """Khởi tạo batch processor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(config, num_classes=None)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings_batch(self, image_dir, output_file='embeddings.npy'):
        """Trích xuất embedding cho tất cả ảnh trong thư mục
        
        Returns:
            {
                'embeddings': numpy array (N, 512),
                'image_paths': list đường dẫn ảnh
            }
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        embeddings = []
        image_paths = []
        
        for image_path in tqdm(image_files, desc="Extracting embeddings"):
            try:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = preprocess_image(image).to(self.device)
                
                with torch.no_grad():
                    embedding = self.model.extract_embedding(image_tensor)
                
                embeddings.append(embedding.cpu().numpy())
                image_paths.append(str(image_path))
            except Exception as e:
                print(f"Lỗi xử lý {image_path}: {e}")
                continue
        
        embeddings = np.vstack(embeddings)
        
        # Lưu kết quả
        np.save(output_file, embeddings)
        with open(output_file.replace('.npy', '_paths.txt'), 'w') as f:
            f.write('\n'.join(image_paths))
        
        print(f"✓ Đã xử lý {len(embeddings)} ảnh, lưu vào {output_file}")
        
        return {
            'embeddings': embeddings,
            'image_paths': image_paths
        }

    def compute_similarity_matrix(self, embeddings_file):
        """Tính ma trận similarity giữa tất cả ảnh"""
        embeddings = np.load(embeddings_file)
        
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Tính similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        return similarity_matrix

# Sử dụng
processor = FaceBatchProcessor(
    model_path='experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth',
    config=config
)

# Trích xuất embedding cho tất cả ảnh
result = processor.extract_embeddings_batch(
    image_dir='data/processed/',
    output_file='data/embeddings/all_embeddings.npy'
)

# Tính similarity matrix
sim_matrix = processor.compute_similarity_matrix('data/embeddings/all_embeddings.npy')
print(f"Similarity matrix shape: {sim_matrix.shape}")
```

### 5. Sử Dụng Với Flask API (Web Service)

**Code ví dụ** - `app.py`:

```python
from flask import Flask, request, jsonify
from pathlib import Path
import cv2
import numpy as np
import torch
from src.models.face_recognizer import build_model

app = Flask(__name__)

class FaceAPI:
    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(config, num_classes=None)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def extract_embedding(self, image_data):
        """Trích xuất embedding từ dữ liệu ảnh"""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        image_tensor = preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.extract_embedding(image_tensor)
        
        return embedding.cpu().numpy()

# Khởi tạo API
face_api = FaceAPI(
    model_path='experiments/exp_001_arcface_mobilenetv2/checkpoints/best.pth',
    config=config
)

@app.route('/api/verify', methods=['POST'])
def verify():
    """Verify hai ảnh
    
    Request:
        {
            'image1': <binary>,
            'image2': <binary>
        }
    
    Response:
        {
            'match': bool,
            'similarity': float,
            'distance': float
        }
    """
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing images'}), 400
    
    image1_data = request.files['image1'].read()
    image2_data = request.files['image2'].read()
    
    emb1 = face_api.extract_embedding(image1_data)
    emb2 = face_api.extract_embedding(image2_data)
    
    similarity = np.dot(emb1, emb2.T) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
    )
    
    return jsonify({
        'match': float(similarity[0, 0]) > 0.5,
        'similarity': float(similarity[0, 0]),
        'distance': float(1 - similarity[0, 0])
    })

@app.route('/api/embedding', methods=['POST'])
def get_embedding():
    """Trích xuất embedding từ ảnh
    
    Response:
        {
            'embedding': [float, ...],  # 512 values
            'shape': [1, 512]
        }
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image'}), 400
    
    image_data = request.files['image'].read()
    embedding = face_api.extract_embedding(image_data)
    
    return jsonify({
        'embedding': embedding.tolist(),
        'shape': embedding.shape
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Sử dụng API**:

```bash
# Chạy Flask server
python app.py

# Test verify endpoint
curl -X POST -F "image1=@image1.jpg" -F "image2=@image2.jpg" \
  http://localhost:5000/api/verify

# Test embedding endpoint
curl -X POST -F "image=@image.jpg" \
  http://localhost:5000/api/embedding
```

---

## �📝 Ghi Chú Quan Trọng

### Cải Thiện Hiệu Năng

1. **Tăng Accuracy Trên Masked Face**:
   - Sử dụng RMFRD hoặc MFR2 dataset (real masked faces)
   - Tăng cường data augmentation với mặt nạ
   - Tuning margin và scale của ArcFace
   - Sử dụng backbone nặng hơn (ResNet50)

2. **Tối Ưu Hóa Cho Mobile**:
   - Sử dụng MobileNetV2 (nhẹ nhất)
   - Áp dụng quantization (INT8)
   - Dùng ONNX format cho inference
   - Export sang TFLite cho thiết bị Android

3. **Thực Hiện Inference Nhanh**:
   - Batch processing
   - GPU inference (CUDA)
   - Model quantization
   - Convert sang ONNX/TensorRT

### Khắc Phục Sự Cố

**Lỗi CUDA out of memory:**
```bash
# Giảm batch size
python scripts/train.py ... --batch-size 64
```

**Model underfitting:**
```bash
# Tăng số epochs hoặc learning rate
# Thêm data augmentation
# Sử dụng backbone lớn hơn
```

**Model overfitting:**
```bash
# Tăng regularization (weight_decay)
# Giảm số epochs
# Thêm dropout
# Giảm capacity của model
```

### GPU vs CPU

```bash
# Buộc sử dụng CPU
CUDA_VISIBLE_DEVICES="" python scripts/train.py ...

# Sử dụng GPU (mặc định)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py ...  # GPU 0
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py ... # GPU 0 và 1
```

---

## 📚 Tài Liệu Tham Khảo

### Các Paper Liên Quan

1. **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**
   - https://arxiv.org/abs/1801.07698

2. **MobileNetV2: Inverted Residuals and Linear Bottlenecks**
   - https://arxiv.org/abs/1801.04381

3. **Searching for MobileNetV3**
   - https://arxiv.org/abs/1905.02175

4. **LFW Dataset**
   - http://vis-www.cs.umass.edu/lfw/

### Thư Viện Sử Dụng

- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **scikit-learn**: Metrics & utilities
- **TensorBoard**: Visualization

---

## 📄 License

MIT License - Xem file LICENSE để chi tiết

---

## 🤝 Đóng Góp

Nếu bạn tìm thấy bug hoặc có đề xuất cải thiện, vui lòng:

1. Tạo Issue mô tả vấn đề
2. Fork repository
3. Tạo branch: `git checkout -b feature/your-feature`
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature/your-feature`
6. Tạo Pull Request

---

## 📞 Liên Hệ

Nếu có câu hỏi, vui lòng liên hệ:

- **Trần Anh Tuấn**: [email]
- **Nguyễn Thiện Khải**: [email]
- **Nguyễn Đức Anh**: [email]
- **Phạm Nguyễn Huy Tuấn**: [email]

---

**Cập nhật lần cuối**: 27/04/2026

Educational project - not for commercial use.
