# 🩺 Hệ thống Dự đoán Bệnh Tiểu Đường

**Thuật toán chính:** K-Nearest Neighbors (KNN, K=5)  
**Dataset:** Pima Indians Diabetes Dataset (768 mẫu, 8 đặc trưng)  
**Test Accuracy:** 90.26%

---

## 📁 Cấu trúc thư mục

```
diabetes_project/
├── train.py            ← Huấn luyện 5 thuật toán, phân tích K, learning curve
├── app.py              ← Flask web server + API endpoints + tạo biểu đồ
├── requirements.txt    ← Thư viện cần cài
├── diabetes.csv        ← Dữ liệu (tải từ Kaggle)
├── model.pkl           ← Mô hình KNN (tạo sau khi train)
├── scaler.pkl          ← StandardScaler (tạo sau khi train)
├── accuracy.json       ← Accuracy 5 thuật toán (tạo sau khi train)
├── all_results.json    ← Kết quả chi tiết (tạo sau khi train)
├── k_results.json      ← Phân tích K=1..20 (tạo sau khi train)
├── learning_curve.json ← Dữ liệu learning curve (tạo sau khi train)
├── history.csv         ← Lịch sử dự đoán (tạo tự động)
└── templates/
    └── index.html      ← Giao diện web đầy đủ
```

---

## 🚀 Cài đặt và chạy

### Bước 1: Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Tải dataset
Tải file `diabetes.csv` từ Kaggle:  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
Đặt vào cùng thư mục với `train.py`

### Bước 3: Huấn luyện mô hình
```bash
python train.py
```
Kết quả: tạo ra `model.pkl`, `scaler.pkl`, `accuracy.json`, `all_results.json`, `k_results.json`, `learning_curve.json`

### Bước 4: Chạy ứng dụng web
```bash
python app.py
```
Mở trình duyệt: http://localhost:5000

---

## 🔌 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Trang chính |
| `/predict` | POST | Dự đoán với 8 chỉ số đầu vào |
| `/results/table` | GET | Bảng so sánh 5 thuật toán |
| `/chart/k-analysis` | GET | Biểu đồ phân tích tham số K |
| `/chart/learning-curve` | GET | Biểu đồ learning curve |
| `/chart/confusion-matrix` | GET | Confusion matrix 5 thuật toán |

---

## 📊 Tính năng giao diện

| Tab | Nội dung |
|-----|---------|
| 🔮 Dự đoán | Nhập 8 chỉ số, KNN dự đoán, lưu lịch sử |
| 📊 So sánh | Bảng số liệu Train/Test/CV/Precision/Recall/F1/AUC |
| 📈 Phân tích K | Biểu đồ K=1→20 + bảng chi tiết |
| 📉 Learning Curve | Accuracy theo kích thước tập huấn luyện |
| 🎯 Confusion Matrix | Ma trận nhầm lẫn 5 thuật toán |
| 🗄 Dữ liệu | Thống kê mô tả, phân bố nhãn |

---

## 📥 Đầu vào / Đầu ra

**Input (8 chỉ số y tế):**
- Pregnancies, Glucose, BloodPressure, SkinThickness
- Insulin, BMI, DiabetesPedigreeFunction, Age

**Output:**
- Xác suất mắc bệnh (%)
- Kết luận: Mắc bệnh / Không mắc bệnh
- 3 biểu đồ: nguy cơ, so sánh thuật toán, lịch sử

---

## 📈 Kết quả thực nghiệm

| Thuật toán | Train Acc | Test Acc | Recall | F1 |
|-----------|-----------|----------|--------|----|
| **KNN (K=5)** | 89.25% | **90.26%** | 75.00% | 83.87% |
| Logistic Regression | 88.44% | 90.91% | 84.62% | 86.27% |
| Random Forest | 100.0% | 92.21% | 84.62% | 88.00% |
| SVM | 91.69% | 90.91% | 84.62% | 86.27% |
| Decision Tree | 100.0% | 82.47% | 84.62% | 76.52% |
