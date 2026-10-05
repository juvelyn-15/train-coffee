# Augmentation cho phân loại hạt cà phê lỗi

## 1. Mục tiêu và dữ liệu đầu vào

File `augmentation.py` cung cấp class `Augmentation`, dùng trực tiếp với `datagen.flow(...)` và `model.fit(...)` như code mẫu.
Module chỉ tạo biến thể ảnh train.
Việc đọc dữ liệu và gán nhãn thuộc `DataPreprocessor` hiện tại; preprocessing đầu vào model, xây dựng model và điều phối huấn luyện thuộc pipeline train.
Repository hiện có notebook EDA, chưa có module train để tích hợp trực tiếp.

Theo `usk_coffee_eda.ipynb`, bài toán ở đây là USK-Coffee với ảnh RGB 256×256, gồm hai nhãn: `defect` và `normal`.
Nhóm `normal` gộp ba thư mục `longberry`, `peaberry`, `premium`.
Train có 1.200 ảnh lỗi và 3.600 ảnh bình thường.
Đoạn CIFAR-10 được dùng làm mẫu cách tổ chức code, không dùng nguyên số lớp 10 hoặc kích thước 32×32 cho dữ liệu cà phê.

Đầu vào của class là mảng NumPy `(N, H, W, 3)`, RGB, pixel trong `[0, 255]`, ưu tiên kiểu `uint8`.
Đầu ra là batch `float32` trong `[0, 255]`, giữ nguyên kích thước ảnh và nhãn tương ứng.
Chuyển sang `float32` phục vụ nội suy và tạo batch, không đồng nghĩa với chia pixel cho 255.
Không truyền ảnh đã chia 255 hoặc chuẩn hóa mean/std vào class này, vì hợp đồng đầu vào là `[0, 255]` và bước brightness của Keras xử lý qua ảnh PIL.
Class không rescale hoặc chuẩn hóa mean/std.
Nếu đọc bằng OpenCV, phải đổi BGR sang RGB trước.

## 2. Nghiên cứu đã dùng phép biến đổi nào?

### Nguồn trực tiếp từ nhóm xây dựng USK-Coffee

Trang dữ liệu xác nhận USK-Coffee gồm bốn loại hạt và có thể gộp thành bài toán normal/defect.
Nguồn: [USK-Coffee của nhóm tác giả](https://coffee.comvislab-usk.org/).

Trong cell `image_transforms` của [notebook công khai của tác giả](https://github.com/cvitlab/USK-COFFEE-DATASET-A-multi-class-dataset-composed-of-the-various-green-bean-arabica/blob/main/USK_Coffee_Code.ipynb), pipeline train gồm `Resize((256, 256))`, `RandomHorizontalFlip()`, `ToTensor()` và `Normalize(mean, std)`.
Val/test không có lật ngẫu nhiên.
Trong đó, lật ngang là augmentation; resize, chuyển tensor và normalization là tiền xử lý.
Đây là nội dung xác minh từ code công khai, không phải khẳng định toàn bộ thí nghiệm trong bài báo gốc chỉ dùng đúng một phép augmentation.

### Nghiên cứu bổ sung trên cùng bộ dữ liệu

Pereira Neto và cộng sự, trong [Enhancing green coffee quality assessment through deep learning, mục III.C](https://sol.sbc.org.br/index.php/wvc/article/download/27537/27349), dùng lật ngang, xoay từ -15° đến 15°, random resized crop 80-100%, thay đổi brightness/contrast/saturation với hệ số 0,8-1,2 và random erasing 2-20%.
Tác giả so sánh huấn luyện có và không có augmentation nhằm đánh giá ảnh hưởng đến khả năng phân loại.
Đây là thí nghiệm bốn lớp, nên không thể coi cấu hình đó là cấu hình tối ưu đã được chứng minh cho bài toán hai lớp hiện tại.

Izza và Kusuma, trong [Image Classification of Green Arabica Coffee Using Transformer-Based Architecture, mục 3.4.4](https://ijettjournal.org/Volume-72/Issue-6/IJETT-V72I6P128.pdf), dùng augmentation ngay trong lúc train: lật ngang/dọc, thay đổi tương phản và xoay ngẫu nhiên.
Mục này nêu các phép biến đổi nhưng không cung cấp biên độ cụ thể để chép lại.

Các nguồn trên hỗ trợ việc chọn nhóm phép biến đổi.
Lý do cho từng lựa chọn và các mức điều chỉnh dưới đây là lập luận thiết kế cho repository này, không gán cho tác giả nếu nguồn không nêu.

## 3. Tham số được chọn và cơ sở lựa chọn

| Tham số | Mặc định | Ý nghĩa và cơ sở |
|---|---|---|
| `rotation_range` | `15` | Mỗi ảnh lấy góc ngẫu nhiên trong [-15°, 15°], theo khoảng góc được báo cáo bởi Pereira Neto và cộng sự; giúp giảm phụ thuộc hướng đặt hạt, đồng thời dùng mức xoay nhẹ để hạn chế cắt biên. |
| `width_shift_range` | `0.05` | Dịch ngang tối đa 5% chiều rộng, khoảng 12,8 pixel ở ảnh 256×256; đây là mức đề xuất của module để mô phỏng vị trí hạt thay đổi, không phải con số trích từ các bài trên. |
| `height_shift_range` | `0.05` | Dịch dọc tối đa 5% chiều cao; cùng cơ sở với dịch ngang. |
| `horizontal_flip` | `True` | Keras lật ngang với xác suất 50%; phép này xuất hiện trong code gốc và các nghiên cứu bổ sung, còn nhãn chất lượng hạt không phụ thuộc bên trái/phải của ảnh. |
| `vertical_flip` | `True` | Keras lật dọc với xác suất 50%, độc lập với lật ngang; được dùng trong nghiên cứu của Izza và Kusuma, phù hợp giả định hạt không có hướng trên/dưới bắt buộc. |
| `brightness_range` | `(0.9, 1.1)` | Nhân độ sáng với hệ số ngẫu nhiên 0,9-1,1; đề xuất nhẹ hơn mức 0,8-1,2 của Pereira Neto và cộng sự nhằm hạn chế biến đổi dấu hiệu màu. |
| `fill_mode` | `"nearest"` | Lấp phần trống sau xoay/dịch bằng pixel biên gần nhất; lựa chọn triển khai để tránh tự tạo viền đen trên nền sáng. |
| `data_format` | `"channels_last"` | Quy định mảng theo `(N, H, W, 3)` để khớp cách dùng Keras trong ví dụ. |
| `dtype` | `"float32"` | Batch đầu ra dùng số thực cho nội suy; pixel vẫn trong [0, 255]. |

Lật và xoay thay đổi cách nhìn hạt, còn dịch chuyển giảm sự phụ thuộc vào vị trí cố định.
Brightness mô phỏng dao động ánh sáng khi chụp; mức ±10% là điểm khởi đầu để thử nghiệm, không phải kết quả tối ưu hóa.
EDA cũng gợi ý biến đổi ánh sáng nhẹ, nhưng không dùng tập test để chọn biên độ cuối cùng.
Giữ hoặc đổi tham số phải dựa trên train/validation và kiểm tra ảnh trực quan.

Không bật zoom, shear, hue shift, channel shift, blur, crop hoặc random erasing trong cấu hình này.
Crop/erasing có thể làm mất vùng lỗi nhỏ; shear và zoom độc lập theo hai trục có thể làm thay đổi hình dáng; blur có thể xóa texture; đổi màu mạnh có thể làm sai dấu hiệu chất lượng.
Đây là lý do chọn một cấu hình đơn giản, không có nghĩa những phép đó luôn gây hại.
Contrast được một số nghiên cứu sử dụng nhưng không được thêm vì `ImageDataGenerator` không có tham số contrast trực tiếp và bản này ưu tiên code ngắn theo mẫu.

Xoay và dịch vẫn có thể cắt hạt nằm sát mép; `nearest` có thể kéo dài màu hoặc phần hạt ở biên.
Cần xem batch thực tế, giảm dịch về `0` hoặc giảm góc nếu xuất hiện ảnh không còn hợp lý.
Không khẳng định augmentation chắc chắn cải thiện độ chính xác trước khi có thí nghiệm so sánh.

## 4. Ranh giới trách nhiệm

| Thành phần | Trách nhiệm | Hợp đồng |
|---|---|---|
| `DataPreprocessor` hiện tại | Đọc RGB, tạo manifest, giữ split và gán nhãn | `X`: `uint8 [0, 255]`; `y`: vector số nguyên, normal=0, defect=1 |
| `Augmentation` | Xoay, dịch, lật và thay đổi độ sáng ngẫu nhiên cho train | Nhận RGB [0, 255], trả batch `float32 [0, 255]`; giữ nguyên nhãn |
| Preprocessing đầu vào model | Biến đổi cố định theo yêu cầu model | Áp dụng cùng cấu hình cho train, validation, test và inference |
| Pipeline train | Tạo batch, xây dựng model, chọn loss và tính class weights | Điều phối các thành phần trên |

Khi tạo `Augmentation()`, chưa có ảnh nào được biến đổi.
Khi `.flow(...)` được model yêu cầu cung cấp batch, Keras chọn ảnh và lấy tham số ngẫu nhiên riêng cho từng ảnh.
Với cấu hình này, Keras áp dụng phép hình học xoay/dịch, lật ngang/dọc và brightness; không chia pixel cho 255.
Phép hình học dùng nội suy bậc 1 theo mặc định của generator.
Ảnh gốc trên đĩa và mảng đầu vào không bị ghi đè bởi augmentation.
Ở các lượt lấy dữ liệu sau, cùng một ảnh có thể xuất hiện dưới biến thể khác; không tạo thêm file hoặc tự tăng số mẫu mỗi epoch.
Nhãn được giữ nguyên vì các biến đổi được giả định không làm thay đổi normal/defect.

## 5. Cách sử dụng trong pipeline train

```text
Train:     đọc RGB -> augmentation -> preprocessing đầu vào -> model
Val/test:  đọc RGB -----------------> preprocessing đầu vào -> model
Inference: đọc RGB -----------------> preprocessing đầu vào -> model
```

Ví dụ CNN tự xây dưới đây đặt `Rescaling(1/255)` ở đầu model, tách khỏi augmentation.
Cả model và generator đều nhận RGB trong `[0, 255]`; không chia 255 bên ngoài model.
Lớp preprocessing được lưu cùng model và áp dụng cho cả train/validation/test/predict.

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from augmentation import Augmentation
from preprocessing import DataPreprocessor

(x_train, y_train), (x_val, y_val), (x_test, y_test) = (
    DataPreprocessor().preprocess()
)
# Nhãn số nguyên: normal=0, defect=1; không one-hot.
datagen = Augmentation()
train_batches = datagen.flow(
    x_train, y_train, batch_size=64, shuffle=True, seed=42
)

# Preprocessing cố định cho CNN tự xây, độc lập với augmentation.
input_preprocessing = keras.Sequential(
    [layers.Rescaling(1.0 / 255)], name="input_preprocessing"
)
model = keras.Sequential([
    keras.Input(shape=(256, 256, 3)),
    input_preprocessing,
    layers.Conv2D(16, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9, nesterov=True
    ),
    metrics=["accuracy"],
)

counts = np.bincount(y_train, minlength=2)
if np.any(counts == 0):
    raise ValueError("Train must contain both normal and defect images")
class_weight = {
    label: len(y_train) / (2 * int(count))
    for label, count in enumerate(counts)
}
H = model.fit(
    train_batches,
    validation_data=(x_val, y_val),
    epochs=50,
    class_weight=class_weight,
    verbose=1,
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# predict trả xác suất defect, vì defect là nhãn 1.
p_defect = float(model.predict(x_test[:1], verbose=0)[0, 0])
class_names = ["normal", "defect"]
print("Giá trị dự đoán:", class_names[int(p_defect >= 0.5)])
```

Đây là ví dụ tích hợp tối giản, không phải kiến trúc CNN cuối cùng hoặc kết quả thí nghiệm.
Ví dụ sử dụng API preprocessing hiện tại; lời gọi `preprocess()` hiện có hành vi xóa các file duplicate được chỉ định, độc lập với augmentation.
Hành vi đó là vấn đề tồn đọng nêu ở mục 7.

Không cần `datagen.fit(x_train)` vì cấu hình này không tính thống kê mean/std hoặc ZCA.
Không cần đặt `steps_per_epoch`: Keras lấy độ dài iterator, gồm cả batch cuối chưa đủ 64 ảnh.
Nếu muốn ghi rõ, dùng `steps_per_epoch=len(train_batches)` thay cho chia nguyên `len(x_train)//64`, vốn có thể bỏ batch cuối.
`seed=42` hỗ trợ lặp lại chuỗi augmentation khi giữ cùng cách gọi; không đảm bảo toàn bộ quá trình huấn luyện trên mọi phần cứng cho kết quả giống hệt nhau.

Augmentation không cân bằng lại tỷ lệ giữa hai lớp.
Class weights được tính từ nhãn train thực tế, không hard-code theo số mẫu trước khi loại duplicate.
Khi đánh giá, cần thêm precision, recall, F1 của lớp defect và confusion matrix; accuracy đơn lẻ có thể che khuất việc bỏ sót hạt lỗi.
Ngưỡng 0,5 trong ví dụ là điểm khởi đầu; nếu điều chỉnh ngưỡng, dùng validation và chốt trước khi đánh giá test.

Với backbone pretrained, thay phần `input_preprocessing` bằng preprocessing phù hợp với backbone và kiểm tra model có tích hợp preprocessing hay chưa.
Không mặc định ghép `Rescaling(1/255)` với `preprocess_input` hoặc với backbone đã có rescaling.
Nếu dùng mean/std, chỉ tính thống kê từ train đã loại duplicate rồi áp dụng cùng thống kê cho mọi split và inference.

Có thể giảm biến đổi hoặc chỉ dùng lật ngang bằng cách truyền lại tham số:

```python
datagen = Augmentation(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=None,
)
```

## 6. Kiểm tra và giới hạn

Module giữ API `.flow()` hiện có để thay đổi trách nhiệm mà không thay cơ chế augmentation.
TensorFlow đánh dấu `ImageDataGenerator` là deprecated; việc thay API là công việc riêng.
Nguồn API: [tài liệu TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).
Lớp `Rescaling` áp dụng trong cả training và inference: [tài liệu TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling).
Môi trường cần TensorFlow, NumPy, Pillow và SciPy; module không tự cài thư viện.

Sau khi bỏ `rescale`, code sử dụng generator cũ phải chuyển việc chuẩn hóa sang preprocessing đầu vào model.
Nếu CNN trước đây nhận batch [0, 1], dùng generator mới mà chưa thêm preprocessing sẽ thay đổi miền pixel đầu vào CNN.
Validation/test/predict phải truyền ảnh trong cùng miền [0, 255] như train khi model có `Rescaling`.

Kiểm tra tích hợp cần xác nhận miền pixel trước/sau preprocessing, nhãn, batch cuối, seed và việc không sửa mảng đầu vào.
Với augmentation tắt, pixel 255 phải còn là 255 ở đầu ra generator và trở thành 1 sau `Rescaling(1/255)`.
Smoke test với ảnh tổng hợp xác minh đường đi kỹ thuật; không thay thế việc kiểm tra ảnh hạt cà phê thực tế hoặc đo chất lượng mô hình.

Thí nghiệm tiếp theo nên so sánh cùng kiến trúc, split và lịch train với ba cấu hình: không augmentation, chỉ lật ngang và cấu hình mặc định.
Chọn cấu hình trên validation bằng F1/recall lớp defect, rồi đánh giá test sau khi chốt.
Báo cáo không đưa ra mức tăng accuracy vì chưa chạy thí nghiệm đó.

## 7. Các vấn đề preprocessing tồn đọng

Các vấn đề sau chưa được sửa trong phạm vi thay đổi augmentation:

- `DataPreprocessor.preprocess()` gọi `remove_duplicates()` và xóa file gốc bằng `unlink()`; nên loại đường dẫn khỏi manifest thay vì xóa trên đĩa.
- `load_rgb()` chưa kiểm tra kích thước 256×256; `np.stack()` chỉ yêu cầu ảnh cùng shape, không đảm bảo kích thước đã cam kết.
- `load_split()` chưa báo lỗi rõ ràng cho split rỗng; việc kiểm tra thư mục split và tổng số ảnh chưa bảo đảm từng split có ảnh.
- Tên `DataPreprocessor` hiện bao gồm trách nhiệm chuẩn bị dataset; việc đổi tên hoặc tách thành `DatasetLoader` chưa được thực hiện.
- Tài liệu `preprocessing.md` mô tả normalization thuộc bước training; cần làm rõ preprocessing đầu vào model cũng phải áp dụng cho validation/test/inference khi cập nhật module đó.
