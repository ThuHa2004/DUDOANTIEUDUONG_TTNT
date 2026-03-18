"""
app.py - Flask Web Application
Hệ thống dự đoán bệnh tiểu đường với KNN
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64, json, os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ─── Load model & dữ liệu đã huấn luyện ──────────────────────────
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("accuracy.json",  encoding="utf-8") as f:
    accuracy_data = json.load(f)

with open("all_results.json", encoding="utf-8") as f:
    all_results = json.load(f)

with open("k_results.json", encoding="utf-8") as f:
    k_results = json.load(f)

with open("learning_curve.json", encoding="utf-8") as f:
    lc_data = json.load(f)

HISTORY_FILE = "history.csv"

# ─── Dark style cho matplotlib ────────────────────────────────────
DARK   = "#0a0e1a"
PANEL  = "#111827"
CARD   = "#1a2235"
BORDER = "#2a3550"
TEXT   = "#e2e8f0"
MUTED  = "#64748b"
GREEN  = "#4ade80"
RED    = "#f87171"
BLUE   = "#38bdf8"
YELLOW = "#fbbf24"
PURPLE = "#a78bfa"
ORANGE = "#fb923c"
PALETTE = [BLUE, GREEN, YELLOW, RED, PURPLE]

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor': DARK,
        'axes.facecolor':   PANEL,
        'axes.edgecolor':   BORDER,
        'axes.labelcolor':  MUTED,
        'axes.titlecolor':  TEXT,
        'xtick.color':      MUTED,
        'ytick.color':      MUTED,
        'text.color':       TEXT,
        'grid.color':       BORDER,
        'grid.alpha':       0.6,
        'font.family':      'DejaVu Sans',
        'axes.titlesize':   13,
        'axes.labelsize':   11,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
    })

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=DARK)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return encoded


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ─── Dự đoán ──────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[
        float(data["pregnancies"]),
        float(data["glucose"]),
        float(data["bloodpressure"]),
        float(data["skin"]),
        float(data["insulin"]),
        float(data["bmi"]),
        float(data["dpf"]),
        float(data["age"]),
    ]])
    features_scaled = scaler.transform(features)
    prob    = model.predict_proba(features_scaled)[0][1]
    percent = round(prob * 100, 2)
    outcome = 1 if percent >= 50 else 0
    level   = "Mắc tiểu đường" if outcome == 1 else "Không mắc tiểu đường"

    # Lưu history
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    new_row = pd.DataFrame([[now, percent, outcome]], columns=["time", "risk", "outcome"])
    try:
        old = pd.read_csv(HISTORY_FILE)
        if "outcome" not in old.columns:
            old["outcome"] = old["risk"].apply(lambda x: 1 if float(x) >= 50 else 0)
        all_data = pd.concat([old, new_row], ignore_index=True)
    except Exception:
        all_data = new_row
    all_data.to_csv(HISTORY_FILE, index=False)

    # Chart 1: Nguy cơ
    chart1 = make_risk_chart(percent, outcome)
    # Chart 2: So sánh thuật toán
    chart2 = make_accuracy_chart()
    # Chart 3: Lịch sử
    chart3 = make_history_chart(all_data)

    return jsonify({
        "percent": percent,
        "outcome": outcome,
        "level":   level,
        "chart1":  chart1,
        "chart2":  chart2,
        "chart3":  chart3,
        "history": all_data.tail(10).to_dict(orient="records"),
    })


# ─── Biểu đồ phân tích K ──────────────────────────────────────────
@app.route("/chart/k-analysis")
def chart_k_analysis():
    ks        = [int(k) for k in k_results.keys()]
    train_acc = [k_results[str(k)]["train_acc"] for k in ks]
    test_acc  = [k_results[str(k)]["test_acc"]  for k in ks]
    cv_acc    = [k_results[str(k)]["cv_mean"]    for k in ks]

    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, train_acc, color=BLUE,   lw=2.5, marker='o', markersize=5,  label="Train Accuracy")
    ax.plot(ks, test_acc,  color=GREEN,  lw=2.5, marker='s', markersize=5,  label="Test Accuracy")
    ax.plot(ks, cv_acc,    color=ORANGE, lw=2,   marker='^', markersize=4,
            linestyle='--', label="CV Accuracy (5-fold)")
    best_k_int = max(k_results.keys(), key=lambda k: k_results[k]["cv_mean"] - k_results[k]["cv_std"])
    best_k_int = int(best_k_int)
    ax.axvline(best_k_int, color=YELLOW, lw=2, linestyle='--', alpha=0.9, label=f"K = {best_k_int} (đã chọn)")
    ax.fill_between(ks, train_acc, test_acc, alpha=0.07, color=RED)
    ax.set_xlim(0.5, max(ks) + 0.5)
    ax.set_ylim(55, 105)
    ax.set_xticks(ks)
    ax.set_xlabel("Giá trị K")
    ax.set_ylabel("Độ chính xác (%)")
    ax.set_title("Ảnh Hưởng Tham Số K đến Hiệu Năng KNN")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=MUTED, fontsize=10)
    fig.tight_layout()
    return jsonify({"chart": fig_to_base64(fig)})


# ─── Biểu đồ learning curve ───────────────────────────────────────
@app.route("/chart/learning-curve")
def chart_learning_curve():
    sizes      = lc_data["train_sizes"]
    train_mean = lc_data["train_mean"]
    train_std  = lc_data["train_std"]
    test_mean  = lc_data["test_mean"]
    test_std   = lc_data["test_std"]
    tm  = np.array(train_mean)
    ts  = np.array(train_std)
    vm  = np.array(test_mean)
    vs  = np.array(test_std)

    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sizes, tm, color=BLUE,  lw=2.5, marker='o', markersize=5, label="Train Accuracy")
    ax.fill_between(sizes, tm - ts, tm + ts, alpha=0.15, color=BLUE)
    ax.plot(sizes, vm, color=GREEN, lw=2.5, marker='s', markersize=5, label="CV Accuracy (Validation)")
    ax.fill_between(sizes, vm - vs, vm + vs, alpha=0.15, color=GREEN)
    ax.axhline(y=vm[-1], color=YELLOW, lw=1.5, linestyle=':', alpha=0.8)
    ax.set_xlabel("Số mẫu huấn luyện")
    ax.set_ylabel("Độ chính xác (%)")
    # Lấy K tối ưu từ k_results
    best_k_lc = max(k_results.keys(), key=lambda k: k_results[k]["cv_mean"] - k_results[k]["cv_std"])
    best_k_lc = int(best_k_lc)
    ax.set_title(f"Learning Curve — KNN (K={best_k_lc})")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=MUTED, fontsize=11)
    fig.tight_layout()
    return jsonify({"chart": fig_to_base64(fig)})


# ─── Confusion Matrix ─────────────────────────────────────────────
@app.route("/chart/confusion-matrix")
def chart_confusion_matrix():
    algo_names = list(all_results.keys())
    set_dark_style()
    fig, axes = plt.subplots(1, len(algo_names), figsize=(16, 4))
    fig.patch.set_facecolor(DARK)

    colors_algo = [BLUE, GREEN, YELLOW, PURPLE, ORANGE]
    for i, (name, res) in enumerate(all_results.items()):
        cm  = np.array(res["cm"])
        ax  = axes[i]
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        total = cm.sum()
        cell_labels = [["TN","FP"],["FN","TP"]]
        cell_colors = [["#052e16","#450a0a"],["#3b0000","#052e16"]]
        border_colors= [[GREEN, RED],[RED, GREEN]]
        for r in range(2):
            for c in range(2):
                val = cm[r][c]
                pct = val / total * 100
                rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1,
                                     facecolor=cell_colors[r][c],
                                     edgecolor=border_colors[r][c], lw=2)
                ax.add_patch(rect)
                fc = GREEN if r == c else RED
                ax.text(c, r,    f"{val}",             ha='center', va='center',
                        fontsize=18, fontweight='bold', color=fc)
                ax.text(c, r+0.28, f"({pct:.1f}%)",   ha='center', va='center',
                        fontsize=9, color=fc)
                ax.text(c, r-0.30, cell_labels[r][c],  ha='center', va='center',
                        fontsize=9, fontweight='bold', color=border_colors[r][c])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: 0\n(Không)", "Pred: 1\n(Mắc)"], fontsize=9, color=MUTED)
        ax.set_yticklabels(["Act: 0\n(Không)", "Act: 1\n(Mắc)"],   fontsize=9, color=MUTED)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        acc = res["test_acc"]; rec = res["recall"]
        ax.set_title(f"{name}\nAcc:{acc}% | Recall:{rec}%",
                     color=colors_algo[i], fontsize=11, fontweight='bold')
        ax.tick_params(colors=MUTED, length=0)

    fig.suptitle("Confusion Matrix — 5 Thuật Toán (Tập Test, n=154)",
                 color=TEXT, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return jsonify({"chart": fig_to_base64(fig)})


# ─── Bảng kết quả đầy đủ ─────────────────────────────────────────
@app.route("/results/table")
def results_table():
    rows = []
    for name, res in all_results.items():
        rows.append({
            "name":      name,
            "train_acc": res["train_acc"],
            "test_acc":  res["test_acc"],
            "cv_mean":   res["cv_mean"],
            "cv_std":    res["cv_std"],
            "precision": res["precision"],
            "recall":    res["recall"],
            "f1":        res["f1"],
            "auc":       res["auc"],
            "cm":        res["cm"],
        })
    return jsonify({"rows": rows})


# ─── Bảng K từ 1 đến 20 ──────────────────────────────────────────
@app.route("/results/k-table")
def results_k_table():
    rows = []
    for k_str, v in k_results.items():
        rows.append({
            "k":         int(k_str),
            "train_acc": v["train_acc"],
            "test_acc":  v["test_acc"],
            "cv_mean":   v["cv_mean"],
            "cv_std":    v["cv_std"],
        })
    rows.sort(key=lambda x: x["k"])
    return jsonify({"rows": rows})


# ─── Learning curve summary ───────────────────────────────────────
@app.route("/results/learning-curve")
def results_learning_curve():
    return jsonify({
        "train_mean_last": lc_data["train_mean"][-1],
        "test_mean_last":  lc_data["test_mean"][-1],
    })


# ─── Helper charts ────────────────────────────────────────────────
def make_risk_chart(percent, outcome):
    set_dark_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals   = [round(100 - percent, 2), percent]
    labels = ["Không mắc (0)", "Mắc bệnh (1)"]
    clrs   = ([GREEN, PANEL] if outcome == 0 else [PANEL, RED])
    bars   = ax.bar(labels, vals, color=clrs, width=0.45, zorder=3)
    ax.set_ylim(0, 115); ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.set_ylabel("Xác suất (%)")
    ax.set_title("Nguy Cơ Mắc Tiểu Đường")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.1f}%", ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=TEXT)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_accuracy_chart():
    set_dark_style()
    names  = list(accuracy_data.keys())
    scores = [v * 100 for v in accuracy_data.values()]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(names, scores, color=PALETTE[:len(names)], height=0.5, zorder=3)
    ax.set_xlim(0, 108); ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("Độ chính xác (%)")
    ax.set_title("So Sánh Các Thuật Toán")
    for bar, val in zip(bars, scores):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va='center', fontsize=10,
                fontweight='bold', color=TEXT)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_history_chart(df):
    set_dark_style()
    pcts = df["risk"].astype(float).tolist()
    x    = list(range(len(pcts)))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axhspan(0,  50, color=GREEN, alpha=0.04)
    ax.axhspan(50, 105, color=RED,  alpha=0.04)
    ax.plot(x, pcts, color=GREEN, lw=2.5, marker='o', markersize=6,
            markerfacecolor=DARK, markeredgecolor=GREEN, markeredgewidth=2, zorder=4)
    ax.fill_between(x, pcts, alpha=0.12, color=GREEN)
    ax.axhline(50, color=YELLOW, lw=1.5, linestyle='--', alpha=0.8)
    ax.scatter([len(pcts)-1], [pcts[-1]], color=YELLOW, s=80, zorder=5)
    ax.set_ylim(0, 105); ax.set_xlim(-0.3, max(len(pcts)-1, 0)+0.3)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    if len(df) <= 15:
        ax.set_xticks(x)
        ax.set_xticklabels(df["time"].tolist(), rotation=30, ha='right', fontsize=8)
    else:
        step = max(1, len(df) // 10)
        ticks = list(range(0, len(df), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([df["time"].iloc[i] for i in ticks], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("% Nguy cơ")
    ax.set_title("Lịch Sử Các Lần Kiểm Tra")
    patch = mpatches.Patch(color=YELLOW, alpha=0.7, label="Ngưỡng 50%")
    ax.legend(handles=[patch], loc='upper left', fontsize=9,
              facecolor=CARD, edgecolor=BORDER, labelcolor=MUTED)
    fig.tight_layout()
    return fig_to_base64(fig)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)