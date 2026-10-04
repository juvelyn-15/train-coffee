# Tiền xử lý dữ liệu USK-Coffee

`DataPreprocessor` chuyển cấu trúc thư mục trong `data/` thành manifest và các mảng dữ liệu `X`, `y` để sử dụng cho ba mô hình: CNN đơn giản, CNN phức tạp và transfer learning.

Các bước augmentation, normalization và class weights không được thực hiện trong `DataPreprocessor`; chúng thuộc bước training.

## 1. Đặc điểm của dataset

Nguồn: Febriana, Muchtar, Dawood, Lin (2022), *USK-COFFEE Dataset: A Multi-Class Green Arabica Coffee Bean Dataset for Deep Learning*. File gốc chỉ ghi nhận những nội dung có thể xác định từ các trích đoạn của bài báo.

| Thành phần | Dataset / nghiên cứu gốc | `DataPreprocessor` |
|---|---|---|
| Ảnh | 8.000 ảnh, 4 lớp: peaberry, longberry, premium, defect; mỗi lớp 2.000 ảnh | Giữ nguyên dữ liệu và ánh xạ thành bài toán nhị phân Normal vs Defect |
| Nhãn | 4 lớp | Normal = 0: longberry, peaberry, premium; Defect = 1 |
| Kích thước | 256×256 RGB | Đọc ảnh ở dạng RGB, kỳ vọng kích thước 256×256 |
| Chia dữ liệu | Train/validation/test | Sử dụng các split có sẵn trong `data/` |
| Mô hình tham chiếu | ResNet-18 và MobileNetV2, input 3×256×256 | Giữ kích thước 256×256 cho cả ba mô hình |
| Normalization / augmentation | Không được xác định từ các trích đoạn đã đọc | Không thực hiện trong `DataPreprocessor` |

EDA của dataset hiện tại cho thấy các file đã có kích thước 256×256 RGB. Vì vậy, bước preprocessing không cần resize lại ảnh.

## 2. Thiết kế preprocessing

Pipeline của `DataPreprocessor`:

```text
data/
  ↓
remove DUPLICATE_PATHS
  ↓
build manifest
  ↓
load RGB 256×256
  ↓
X_train, y_train
X_val, y_val
X_test, y_test
```

### 2.1. Loại bỏ các file được chỉ định

`DataPreprocessor` loại bỏ các file duplicates xác định trong EDA, được khai báo trực tiếp trong `DUPLICATE_PATHS`:

```python
DUPLICATE_PATHS = {
    "data/train/peaberry/14.jpg",
    "data/test/peaberry/1916.jpg",
}
```

## 3. Xây dựng manifest

`build_file_table()` duyệt qua ba split `train`, `val` và `test`, sau đó kiểm tra cấu trúc class folder và tạo một `pandas.DataFrame` mô tả dataset.

Manifest gồm bốn trường:

| Trường | Ý nghĩa |
|---|---|
| `path` | Đường dẫn đến file ảnh |
| `split` | `train`, `val` hoặc `test` |
| `class_name` | Tên class gốc |
| `label` | Nhãn nhị phân: Normal = 0, Defect = 1 |

Các class hợp lệ:

```text
longberry
peaberry
premium
defect
```

Ba class `longberry`, `peaberry` và `premium` được gộp thành Normal:

```text
longberry → 0
peaberry  → 0
premium   → 0
defect    → 1
```

`defect` là positive class, vì vậy nhãn 1 được sử dụng để tính precision, recall và F1 trong bước đánh giá mô hình.

`build_file_table()` cũng kiểm tra:

- split folder có tồn tại hay không;
- class folder có thuộc danh sách class hợp lệ hay không;
- dataset có chứa ảnh hay không;
- phần mở rộng của file có thuộc nhóm được hỗ trợ hay không.

## 4. Load ảnh RGB

`load_rgb()` mở từng ảnh bằng PIL và chuyển ảnh sang RGB:

```python
with Image.open(path) as image:
    return np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )
```

Ảnh sau khi load có dạng:

```text
(height, width, channels)
= (256, 256, 3)
```

Các giá trị pixel vẫn giữ nguyên ở dạng `uint8` trong khoảng:

```text
0–255
```

`DataPreprocessor` không thực hiện:

- chia pixel cho 255;
- mean/std normalization;
- ImageNet normalization;
- augmentation.

Các bước này được thực hiện ở training pipeline.

## 5. Output

`preprocess()` trả về:

```python
(X_train, y_train), (X_val, y_val), (X_test, y_test)
```

Trong đó:

```text
X_train: uint8, shape (N_train, 256, 256, 3)
X_val:   uint8, shape (N_val, 256, 256, 3)
X_test:  uint8, shape (N_test, 256, 256, 3)
```

Label:

```text
y_train: integer vector gồm 0 và 1
y_val:   integer vector gồm 0 và 1
y_test:  integer vector gồm 0 và 1
```


## 5. Các quyết định thiết kế

| Thành phần | Quyết định |
|---|---|
| Bài toán | Binary classification: Normal vs Defect |
| Normal | longberry, peaberry, premium |
| Defect | defect |
| Label | Normal = 0, Defect = 1 |
| Duplicate handling | Chỉ loại các path có trong `DUPLICATE_PATHS` |
| Resize | Không cần vì dataset đã là 256×256 |
| Channels | RGB |
| Pixel type | `uint8` |
| Pixel range | 0–255 |
| Normalization | Thực hiện ở training |
| Augmentation | Thực hiện ở training |
| Class weights | Tính ở training |
| One-hot encoding | Không sử dụng |


