### Objective

Mục đích là cài đặt một phương pháp phát hiện đối tượng sử dụng đặc trưng cục bộ.

- Đầu vào: ảnh đối tượng cần tìm và ảnh có thể hoặc không chứa đối tượng đó
- Đầu ra: vị trí của đối tượng trong ảnh nếu tồn tại
  Các kỹ thuật trích chọn đặc trưng và so khớp ảnh được tự do lựa chọn bởi mỗi nhóm

Phương pháp dựa trên 4 bước chính: Trích chọn đặc trưng cục bộ, So khớp đặc trưng, Xác minh hình học và Xác định
vị trí đối tượng.

### Bước 1: Trích chọn Đặc trưng Cục bộ (Local Feature Extraction)

Mục đích là tìm ra các điểm đặc trưng (keypoints/interest points) và tính toán bộ mô tả (descriptor) cho chúng trên cả *
*ảnh mẫu (template)** và **ảnh đầu vào (input image)**.

**Lựa chọn Kỹ thuật:**

1. **Bộ phát hiện điểm đặc trưng (Interest Point Detector):**
   Cần một bộ phát hiện có tính **lặp lại (repeatable)** và **bất biến (invariant)** với các thay đổi về tỷ lệ (scale)
   và xoay (rotation).
    * **SIFT Detector (Difference-of-Gaussians - DoG):** Phương pháp này phát hiện các cực trị cục bộ trên miền không
      gian-scale (space – scale).
    * **Harris-Laplacian:** Tìm cực đại cục bộ của bộ phát hiện góc Harris trong không gian và Laplacian theo tỷ lệ (
      scale).

2. **Bộ mô tả cục bộ (Local Descriptor):**
   Bộ mô tả cần phải **gọn (Compact)** và **bất biến** với các phép biến đổi hình học cơ bản (như dịch chuyển, xoay, tỷ
   lệ) cũng như các thay đổi về điều kiện chiếu sáng (illumination).
    * **SIFT (Scale-Invariant Feature Transform):** SIFT là một lựa chọn mạnh mẽ, tạo ra vector 128 chiều. SIFT hiệu quả
      khi điều kiện chiếu sáng thay đổi nhờ sử dụng đạo hàm bậc 1 và chuẩn hóa vector (độ lớn vector = 1.0).
    * **SURF (Speeded Up Robust Features):** Một lựa chọn khác, thường nhanh hơn SIFT.

*(Các kỹ thuật khác có thể tham khảo bao gồm Harris corner detector, LBP, BRISK, MSER, FREAK, GLOH...).*

### Bước 2: So khớp Đặc trưng (Feature Matching)

Sau khi trích chọn đặc trưng, ta tìm kiếm các cặp đặc trưng tương ứng giữa ảnh mẫu ($I_1$) và ảnh đầu vào ($I_2$).

1. **Tính khoảng cách:** Sử dụng độ đo khoảng cách (distance metric) như **L2, L1, cosine** hoặc **Mahalanobis** giữa
   các vector đặc trưng.
2. **Tìm điểm gần nhất (Nearest Neighbor):** Với mỗi đặc trưng trong $I_1$, tìm đặc trưng gần nhất trong $I_2$.
3. **Lọc sơ bộ bằng Tỷ lệ Khoảng cách (Nearest Neighbor Distance Ratio - NNDR):** Để loại bỏ các cặp so khớp không tốt (
   false matches) ngay từ đầu, sử dụng tỷ lệ khoảng cách giữa điểm gần nhất ($f_2$) và điểm gần thứ hai ($f_2'$). Chỉ
   giữ lại những cặp có tỷ lệ này nhỏ.

### Bước 3: Xác minh Hình học (Geometric Verification) và Lọc Điểm Ngoại lai (Outlier Removal)

Các cặp so khớp sau Bước 2 vẫn có thể chứa các điểm ngoại lai (outliers) do nhiễu hoặc các vùng không liên quan trong
ảnh. Cần một phương pháp mạnh mẽ để ước lượng phép biến đổi hình học (ví dụ: phép biến đổi affine hoặc homography) khớp
ảnh mẫu vào ảnh đích.

**Sử dụng RANSAC (RANdom SAmple Consensus):**

RANSAC là một phương pháp khớp mô hình mạnh mẽ, được sử dụng để tìm phép biến đổi giữa hai tập dữ liệu và đặc biệt **rất
mạnh mẽ với các điểm ngoại lai (robust to outliers)**.

* **Quy trình RANSAC cơ bản:**
    * **Lặp:** Lặp lại $k$ lần.
    * **Lấy mẫu ngẫu nhiên:** Chọn ngẫu nhiên một nhóm tối thiểu các cặp điểm so khớp (seed group) (ví dụ: cần ít nhất *
      *ba điểm** để ước lượng ma trận biến đổi 2D, vì mỗi điểm cung cấp hai phương trình).
    * **Tính toán Biến đổi:** Tính toán phép biến đổi ($T$) dựa trên nhóm điểm mẫu này.
    * **Tìm Inliers:** Tìm tất cả các điểm so khớp khác trong tập dữ liệu nằm trong ngưỡng sai số (threshold) khi áp
      dụng phép biến đổi $T$ (các điểm này gọi là "inliers").
    * **Giữ kết quả tốt nhất:** Giữ lại phép biến đổi có số lượng inliers lớn nhất.

Nếu RANSAC tìm thấy một mô hình biến đổi với đủ số lượng điểm inliers (inliers là các điểm so khớp chính xác), ta xác
định đối tượng đã được tìm thấy.

### Bước 4: Xác định Vị trí Đối tượng (Object Localization)

Sau khi có phép biến đổi hình học $T_{best}$ được xác nhận bởi RANSAC:

1. **Áp dụng Biến đổi:** Áp dụng phép biến đổi $T_{best}$ lên bốn góc của **hộp giới hạn (bounding box)** của ảnh mẫu (
   template).
2. **Đầu ra:** Vị trí mới của bốn góc này trong ảnh đầu vào sẽ xác định vị trí của đối tượng (location of the object in
   images).
