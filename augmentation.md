# Augmentation cho phân loại hạt cà phê lỗi

## 1. Mục tiêu và dữ liệu đầu vào

File `augmentation.py` cung cấp class `Augmentation`, dùng trực tiếp với `datagen.flow(...)` và `model.fit(...)` như code mẫu.
Module chỉ tạo biến thể ảnh train và chia giá trị pixel cho 255; việc đọc dữ liệu, gán nhãn và xây dựng model thuộc module train.
Repository hiện có notebook EDA, chưa có module train để tích hợp trực tiếp.

Theo `usk_coffee_eda.ipynb`, bài toán ở đây là USK-Coffee với ảnh RGB 256×256, gồm hai nhãn: `defect` và `normal`.
Nhóm `normal` gộp ba thư mục `longberry`, `peaberry`, `premium`.
Train có 1.200 ảnh lỗi và 3.600 ảnh bình thường.
Đoạn CIFAR-10 được dùng làm mẫu cách tổ chức code, không dùng nguyên số lớp 10 hoặc kích thước 32×32 cho dữ liệu cà phê.

Đầu vào của class là mảng NumPy `(N, H, W, 3)`, RGB, pixel trong `[0, 255]`, ưu tiên kiểu `uint8`.
Đầu ra là batch `float32` trong `[0, 1]`, giữ nguyên kích thước ảnh và nhãn tương ứng.
Không truyền ảnh đã chia 255 hoặc chuẩn hóa mean/std vào class này, vì bước brightness của Keras xử lý qua ảnh PIL và class còn chia 255 ở cuối.
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
| `rescale` | `1.0 / 255` | Chuyển pixel về [0, 1] sau biến đổi; đây là tiền xử lý, không phải augmentation. |
| `data_format` | `"channels_last"` | Quy định mảng theo `(N, H, W, 3)` để khớp cách dùng Keras trong ví dụ. |
| `dtype` | `"float32"` | Batch đầu ra dùng kiểu số thực thông dụng cho huấn luyện. |

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

## 4. Giải thích từng dòng trong augmentation.py

Số dòng dưới đây ứng với phiên bản file đi kèm báo cáo.

| Dòng | Nội dung | Giải thích |
|---|---|---|
| 1 | Docstring module | Nêu mục đích và miền giá trị đầu vào [0, 255]. |
| 2, 4, 5, 12 | Dòng trống | Tách các phần để dễ đọc, không thực hiện xử lý. |
| 3 | Import `ImageDataGenerator` | Dùng bộ tạo batch và biến đổi ảnh có sẵn của TensorFlow/Keras. |
| 6 | `class Augmentation(ImageDataGenerator)` | Kế thừa generator để dùng sẵn `.flow(...)`, không cần viết lại vòng lặp batch. |
| 7-11 | Docstring class | Mô tả kiểu đầu ra, hình dạng đầu vào và việc chỉ sử dụng cho train; dòng 8 trống trong docstring. |
| 13 | `def __init__(` | Định nghĩa cấu hình khi tạo một đối tượng augmentation. |
| 14 | `self` | Đối tượng đang được khởi tạo; người dùng không truyền tham số này. |
| 15 | `rotation_range=15` | Góc xoay tối đa theo hai chiều, tính bằng độ. |
| 16 | `width_shift_range=0.05` | Mức dịch ngang tính theo tỷ lệ chiều rộng. |
| 17 | `height_shift_range=0.05` | Mức dịch dọc tính theo tỷ lệ chiều cao. |
| 18 | `horizontal_flip=True` | Bật khả năng lật ngang ngẫu nhiên. |
| 19 | `vertical_flip=True` | Bật khả năng lật dọc ngẫu nhiên. |
| 20 | `brightness_range=(0.9, 1.1)` | Khoảng hệ số độ sáng; truyền `None` để tắt. |
| 21 | `):` | Kết thúc danh sách tham số và bắt đầu thân hàm. |
| 22 | `super().__init__(` | Khởi tạo `ImageDataGenerator` bằng cấu hình đã chọn. |
| 23 | `rotation_range=rotation_range` | Chuyển giá trị góc xoay xuống Keras. |
| 24 | `width_shift_range=width_shift_range` | Chuyển giá trị dịch ngang xuống Keras. |
| 25 | `height_shift_range=height_shift_range` | Chuyển giá trị dịch dọc xuống Keras. |
| 26 | `horizontal_flip=horizontal_flip` | Chuyển lựa chọn lật ngang xuống Keras. |
| 27 | `vertical_flip=vertical_flip` | Chuyển lựa chọn lật dọc xuống Keras. |
| 28 | `brightness_range=brightness_range` | Chuyển khoảng độ sáng xuống Keras. |
| 29 | `fill_mode="nearest"` | Chọn cách lấp phần ảnh trống sau phép hình học. |
| 30 | `rescale=1.0 / 255` | Chia pixel cho 255 sau augmentation. |
| 31 | `data_format="channels_last"` | Đặt kênh RGB ở trục cuối. |
| 32 | `dtype="float32"` | Chọn kiểu dữ liệu cho generator. |
| 33 | `)` | Kết thúc lời gọi khởi tạo lớp cha. |

Khi tạo `Augmentation()`, chưa có ảnh nào được biến đổi.
Khi `.flow(...)` được model yêu cầu cung cấp batch, Keras chọn ảnh và lấy tham số ngẫu nhiên riêng cho từng ảnh.
Với cấu hình này, Keras áp dụng phép hình học xoay/dịch, lật ngang/dọc, brightness rồi rescale.
Phép hình học dùng nội suy bậc 1 theo mặc định của generator.
Ảnh gốc trên đĩa và mảng đầu vào không bị ghi đè.
Ở các lượt lấy dữ liệu sau, cùng một ảnh có thể xuất hiện dưới biến thể khác; không tạo thêm file hoặc tự tăng số mẫu mỗi epoch.
Nhãn được giữ nguyên vì các biến đổi được giả định không làm thay đổi normal/defect.

## 5. Cách sử dụng trong pipeline train

Pipeline: đọc ảnh RGB và gán nhãn -> lấy batch train -> augmentation -> chia 255 -> CNN -> loss -> cập nhật trọng số.
Validation/test/predict chỉ tiền xử lý cố định bằng chia 255, không đi qua `Augmentation`.

Ví dụ dưới đây giả sử module train đã đọc ba tập `x_train`, `x_val`, `x_test` thành mảng RGB `uint8` trong [0, 255].
Các mảng nhãn `y_train`, `y_val`, `y_test` phải là one-hot hai phần tử, với `defect=0`, `normal=1`.
Khi đọc thư mục, ánh xạ tên `defect` thành 0 và ba tên `longberry`, `peaberry`, `premium` thành 1; gọi `flow_from_directory` mặc định trên bốn thư mục sẽ tạo bốn lớp và không đúng bài toán này.

```python
import numpy as np
from tensorflow.keras.optimizers import SGD
from augmentation import Augmentation

# Model CNN đã được xây dựng với input_shape=(256, 256, 3).
# Lớp cuối là Dense(2, activation="softmax"), thay cho Dense(10).
# y_* có dạng one-hot, ví dụ defect=[1, 0], normal=[0, 1].
datagen = Augmentation()
train_batches = datagen.flow(
    x_train, y_train, batch_size=64, shuffle=True, seed=42
)

# Chỉ chuẩn hóa cố định cho validation/test.
x_val_scaled = x_val.astype("float32") / 255.0
x_test_scaled = x_test.astype("float32") / 255.0

opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)
H = model.fit(
    train_batches,
    validation_data=(x_val_scaled, y_val),
    epochs=50,
    class_weight={0: 2.0, 1: 2.0 / 3.0},
    verbose=1,
)

score = model.evaluate(x_test_scaled, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Giữ nguyên H, W và thêm trục batch bằng slicing.
y_pred = model.predict(x_test_scaled[:1], verbose=0)
class_names = ["defect", "normal"]
print("Giá trị dự đoán:", class_names[int(np.argmax(y_pred[0]))])
```

Không cần `datagen.fit(x_train)` vì cấu hình này không tính thống kê mean/std hoặc ZCA.
Không cần đặt `steps_per_epoch`: Keras lấy độ dài iterator, gồm cả batch cuối chưa đủ 64 ảnh.
Nếu muốn ghi rõ, dùng `steps_per_epoch=len(train_batches)` thay cho chia nguyên `len(x_train)//64`, vốn có thể bỏ batch cuối khi số ảnh không chia hết.
`seed=42` hỗ trợ lặp lại chuỗi augmentation khi giữ cùng cách gọi; không đảm bảo toàn bộ quá trình huấn luyện trên mọi phần cứng đều cho kết quả giống hệt nhau.

Augmentation không cân bằng lại tỷ lệ 3:1 giữa hai lớp.
Ví dụ dùng trọng số `N / (2 * N_class)`, tức `4800/(2*1200)=2` cho defect và `4800/(2*3600)=2/3` cho normal.
Nếu số mẫu hoặc quy ước nhãn thay đổi, module train phải tính lại trọng số theo train.
Khi đánh giá, cần thêm precision, recall, F1 của lớp defect và confusion matrix; accuracy đơn lẻ có thể che khuất việc bỏ sót hạt lỗi.

Nếu đã dùng `Rescaling(1/255)` trong model, hãy bỏ bước đó khi dùng class này để không chia hai lần.
Ví dụ đang chọn chia 255 làm tiền xử lý cho CNN tự xây, chưa áp dụng mean/std được tính trong EDA.
Nếu muốn dùng mean/std của train, có thể thêm lớp `Normalization` với các thống kê đó ở đầu model, sau đầu vào [0, 1], để áp dụng nhất quán cho cả train/val/test/predict.
Với backbone pretrained, cần điều chỉnh toàn bộ tiền xử lý theo yêu cầu của backbone, không ghép thêm `preprocess_input` tùy tiện.

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

Module dùng API `ImageDataGenerator` để giữ cách gọi của code mẫu; TensorFlow hiện đánh dấu API này là deprecated.
Nguồn API: [tài liệu TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).
Môi trường cần TensorFlow, NumPy, Pillow và SciPy; module không tự cài thư viện.

Kiểm tra kỹ thuật sử dụng ảnh train thật từ cả bốn thư mục, xác nhận kích thước, kiểu dữ liệu, miền pixel, nhãn, batch cuối, seed và việc không sửa ảnh đầu vào.
Một CNN nhỏ được dùng để kiểm tra đường đi `flow -> fit -> validation -> evaluate -> predict`; đây là smoke test tích hợp, không phải thí nghiệm đo chất lượng mô hình phân loại.
Ảnh biến đổi được xuất để kiểm tra trực quan trước khi chốt cấu hình.

Thí nghiệm tiếp theo nên so sánh cùng kiến trúc, split và lịch train với ba cấu hình: không augmentation, chỉ lật ngang và cấu hình mặc định của module.
Chọn cấu hình trên validation bằng F1/recall lớp defect, rồi mới đánh giá test sau khi chốt.
Báo cáo này không đưa ra mức tăng accuracy vì chưa chạy thí nghiệm đó.
