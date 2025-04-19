# 🌱 FirmFuture

## 📌 Giới thiệu

**FirmFuture** là một dự án Machine Learning sử dụng thuật toán **Random Forest Regression** để dự đoán **mức độ bền vững (sustainability score)** của một công ty. Mô hình được xây dựng từ dữ liệu ESG (Môi trường, Xã hội, Quản trị) và các chỉ số tài chính, nhằm hỗ trợ các nhà đầu tư và tổ chức đánh giá khả năng phát triển bền vững của doanh nghiệp.

## 📊 Nguồn dữ liệu

- **Kaggle Dataset:**
- **Định dạng:** CSV
## 🎯 Mục tiêu

- Dự đoán điểm số bền vững của công ty dưới dạng **giá trị số liên tục** 
- Xây dựng mô hình dễ triển khai, trực quan hóa và đánh giá kết quả tốt.
- Phục vụ các tổ chức ESG, quỹ đầu tư xanh, nhà phân tích tài chính.

## ⚙️ Mô hình & Công nghệ sử dụng

- **Thuật toán:** `RandomForestRegressor` 
- **Ngôn ngữ:** Python
- **Thư viện:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`
- **Kỹ thuật áp dụng:**
  - Xử lý giá trị thiếu (missing values)
  - One-hot encoding cho biến phân loại
  - Chuẩn hóa dữ liệu
  - Đánh giá mô hình bằng  R²
