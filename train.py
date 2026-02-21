import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =========================
# 1. LOAD DATA
# =========================
data = pd.read_csv("diabetes.csv")

# Thay 0 bằng mean cho các cột không thể = 0
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    data[col] = data[col].replace(0, data[col].mean())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# =========================
# 2. SCALE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 4. MODEL CHÍNH — KNN
# =========================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred_main = model.predict(X_test)
acc_main = accuracy_score(y_test, y_pred_main)
print(f"[KNN] Accuracy: {acc_main:.4f}")

# Lưu model + scaler
pickle.dump(model,  open("model.pkl",  "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("✅ Đã lưu model.pkl và scaler.pkl")

# =========================
# 5. SO SÁNH THUẬT TOÁN
# =========================
models = {
    "Logistic":    LogisticRegression(max_iter=1000),
    "Dec.Tree":    DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM":         SVC(probability=True),
    "KNN":         KNeighborsClassifier(n_neighbors=5)
}

results = {}
print("\n📊 So sánh thuật toán:")
print("-" * 35)

for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    results[name] = float(acc)
    print(f"  {name:<15} → {acc*100:.2f}%")

print("-" * 35)

# Lưu accuracy để Flask dùng
with open("accuracy.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Đã lưu accuracy.json")
print("✅ TRAIN XONG — Sẵn sàng chạy app.py")