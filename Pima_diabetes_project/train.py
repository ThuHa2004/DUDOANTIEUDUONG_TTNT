"""
train.py - Huấn luyện mô hình dự đoán bệnh tiểu đường
Dataset: Pima Indians Diabetes Dataset
Thuật toán chính: KNN (K-Nearest Neighbors, K=13)
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =============================================================
# 1. LOAD & TIỀN XỬ LÝ DỮ LIỆU
# =============================================================
print("=" * 60)
print("  DỰ ĐOÁN BỆNH TIỂU ĐƯỜNG - KNN MODEL")
print("=" * 60)

data = pd.read_csv("diabetes.csv")
print(f"\n[1] Dữ liệu: {data.shape[0]} mẫu, {data.shape[1]-1} đặc trưng")
print(f"    Outcome=0 (Không mắc): {(data['Outcome']==0).sum()} mẫu")
print(f"    Outcome=1 (Mắc bệnh): {(data['Outcome']==1).sum()} mẫu")

# Thay thế giá trị 0 bất thường bằng mean
cols_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_fix:
    data[col] = data[col].replace(0, data[col].mean())
print(f"\n[2] Đã xử lý giá trị 0 bất thường cho: {cols_fix}")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# =============================================================
# 2. CHUẨN HÓA & PHÂN CHIA DỮ LIỆU
# =============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Phân chia: Train={len(X_train)} | Test={len(X_test)}")

# =============================================================
# 3. MÔ HÌNH CHÍNH: KNN (K=13)
# =============================================================
print("\n[4] Huấn luyện mô hình KNN (K=13)...")
model = KNeighborsClassifier(n_neighbors=13)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

knn_metrics = {
    "train_acc":  round(accuracy_score(y_train, model.predict(X_train)) * 100, 2),
    "test_acc":   round(accuracy_score(y_test, y_pred) * 100, 2),
    "precision":  round(precision_score(y_test, y_pred) * 100, 2),
    "recall":     round(recall_score(y_test, y_pred) * 100, 2),
    "f1":         round(f1_score(y_test, y_pred) * 100, 2),
    "auc":        round(roc_auc_score(y_test, y_prob) * 100, 2),
    "cm":         confusion_matrix(y_test, y_pred).tolist(),
}
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
knn_metrics["cv_mean"] = round(cv_scores.mean() * 100, 2)
knn_metrics["cv_std"]  = round(cv_scores.std() * 100, 2)

print(f"    Train Accuracy : {knn_metrics['train_acc']}%")
print(f"    Test  Accuracy : {knn_metrics['test_acc']}%")
print(f"    CV (5-fold)    : {knn_metrics['cv_mean']} ± {knn_metrics['cv_std']}%")
print(f"    Precision      : {knn_metrics['precision']}%")
print(f"    Recall         : {knn_metrics['recall']}%")
print(f"    F1-Score       : {knn_metrics['f1']}%")
print(f"    AUC-ROC        : {knn_metrics['auc']}%")

# Lưu mô hình chính
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n    ✅ Đã lưu model.pkl và scaler.pkl")

# =============================================================
# 4. SO SÁNH 5 THUẬT TOÁN
# =============================================================
print("\n[5] So sánh 5 thuật toán...")
print("-" * 60)

models_compare = {
    "KNN":           model,   # Dùng lại model đã train ở bước 3, không train lại
    "Logistic Reg.": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":           SVC(probability=True, random_state=42),
}

accuracy_data   = {}
all_results     = {}

for name, m in models_compare.items():
    if name != "KNN":
        m.fit(X_train, y_train)
    y_pred_tr = m.predict(X_train)
    y_pred_te = m.predict(X_test)
    y_prob_te = m.predict_proba(X_test)[:, 1]
    cv        = cross_val_score(m, X_scaled, y, cv=5, scoring='accuracy')

    res = {
        "train_acc":  round(accuracy_score(y_train, y_pred_tr) * 100, 2),
        "test_acc":   round(accuracy_score(y_test,  y_pred_te) * 100, 2),
        "cv_mean":    round(cv.mean() * 100, 2),
        "cv_std":     round(cv.std()  * 100, 2),
        "precision":  round(precision_score(y_test, y_pred_te) * 100, 2),
        "recall":     round(recall_score(y_test,    y_pred_te) * 100, 2),
        "f1":         round(f1_score(y_test,         y_pred_te) * 100, 2),
        "auc":        round(roc_auc_score(y_test, y_prob_te)   * 100, 2),
        "cm":         confusion_matrix(y_test, y_pred_te).tolist(),
    }
    accuracy_data[name] = round(accuracy_score(y_test, y_pred_te), 4)
    all_results[name]   = res

    marker = " ◀ CHÍNH" if name == "KNN" else ""
    print(f"  {name:<16} Train:{res['train_acc']:>6}%  "
          f"Test:{res['test_acc']:>6}%  CV:{res['cv_mean']:>5}%±{res['cv_std']:>4}%{marker}")

print("-" * 60)

# =============================================================
# 5. PHÂN TÍCH ẢNH HƯỞNG CỦA K
# =============================================================
print("\n[6] Phân tích tham số K (K=1..20)...")
k_results = {}
for k in range(1, 21):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    cv_k = cross_val_score(knn_k, X_scaled, y, cv=5, scoring='accuracy')
    k_results[str(k)] = {
        "train_acc": round(accuracy_score(y_train, knn_k.predict(X_train)) * 100, 2),
        "test_acc":  round(accuracy_score(y_test,  knn_k.predict(X_test))  * 100, 2),
        "cv_mean":   round(cv_k.mean() * 100, 2),
        "cv_std":    round(cv_k.std()  * 100, 2),
    }

# =============================================================
# 6. LEARNING CURVE
# =============================================================
print("[7] Tính learning curve...")
MAIN_K = 13
train_sizes, train_scores, test_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=MAIN_K),
    X_scaled, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)
learning_curve_data = {
    "train_sizes":      [int(s) for s in train_sizes],
    "train_mean":       [round(s.mean() * 100, 2) for s in train_scores],
    "train_std":        [round(s.std()  * 100, 2) for s in train_scores],
    "test_mean":        [round(s.mean() * 100, 2) for s in test_scores],
    "test_std":         [round(s.std()  * 100, 2) for s in test_scores],
}

# =============================================================
# 7. LƯU TẤT CẢ KẾT QUẢ
# =============================================================
with open("accuracy.json", "w", encoding="utf-8") as f:
    json.dump(accuracy_data, f, indent=2, ensure_ascii=False)

with open("all_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

with open("k_results.json", "w", encoding="utf-8") as f:
    json.dump(k_results, f, indent=2, ensure_ascii=False)

with open("learning_curve.json", "w", encoding="utf-8") as f:
    json.dump(learning_curve_data, f, indent=2, ensure_ascii=False)

print("\n✅ Đã lưu: accuracy.json | all_results.json | k_results.json | learning_curve.json")
print("\n" + "=" * 60)
print("  TRAIN HOÀN TẤT — Chạy: python app.py")
print("=" * 60)