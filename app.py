from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load accuracy từ train
with open("accuracy.json") as f:
    accuracy_data = json.load(f)

HISTORY_FILE = "history.csv"

# ==================== STYLE CHART ====================
def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  '#161b22',
        'axes.facecolor':    '#1c2330',
        'axes.edgecolor':    '#30363d',
        'axes.labelcolor':   '#7d8590',
        'axes.titlecolor':   '#e6edf3',
        'xtick.color':       '#7d8590',
        'ytick.color':       '#7d8590',
        'text.color':        '#e6edf3',
        'grid.color':        '#30363d',
        'grid.alpha':        0.5,
        'font.family':       'DejaVu Sans',
        'axes.titlesize':    13,
        'axes.labelsize':    10,
    })

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")


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
        float(data["age"])
    ]])

    features_scaled = scaler.transform(features)

    prob    = model.predict_proba(features_scaled)[0][1]
    percent = round(prob * 100, 2)

    # Dùng ngưỡng 50% — chuẩn phân loại nhị phân
    outcome = 1 if percent >= 50 else 0

    if outcome == 1:
        level = "Mắc tiểu đường"
    else:
        level = "Không mắc tiểu đường"

    # =====================
    # LƯU HISTORY
    # =====================
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    new_row = pd.DataFrame([[now, percent, outcome]], columns=["time", "risk", "outcome"])

    try:
        old = pd.read_csv(HISTORY_FILE)
        # Đảm bảo cột outcome tồn tại nếu file cũ chưa có
        if "outcome" not in old.columns:
            old["outcome"] = old["risk"].apply(lambda x: 1 if float(x) >= 50 else 0)
        all_data = pd.concat([old, new_row], ignore_index=True)
    except:
        all_data = new_row

    all_data.to_csv(HISTORY_FILE, index=False)

    # =====================
    # BIỂU ĐỒ 1: NGUY CƠ (BAR)
    # =====================
    set_dark_style()

    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    fig1.patch.set_facecolor('#161b22')

    vals   = [100 - percent, percent]
    labels = ['Không mắc (0)', 'Mắc bệnh (1)']
    bar_colors = ['#00d4aa', '#ff6b6b' if outcome == 1 else '#30363d']
    bar_colors[0] = '#30363d' if outcome == 1 else '#00d4aa'

    bars = ax1.bar(labels, vals, color=bar_colors, width=0.45, zorder=3)
    ax1.set_ylim(0, 115)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_ylabel("Xác suất (%)")
    ax1.set_title("Nguy Cơ Mắc Tiểu Đường", pad=12)

    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='#e6edf3')

    fig1.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=120, facecolor='#161b22')
    buf1.seek(0)
    chart1 = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig1)

    # =====================
    # BIỂU ĐỒ 2: SO SÁNH THUẬT TOÁN
    # =====================
    set_dark_style()

    names  = list(accuracy_data.keys())
    scores = [v * 100 for v in accuracy_data.values()]
    palette = ['#00d4aa','#4facfe','#ffd166','#ff6b6b','#a78bfa']

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    fig2.patch.set_facecolor('#161b22')

    bars2 = ax2.barh(names, scores, color=palette[:len(names)], height=0.55, zorder=3)
    ax2.set_xlim(0, 108)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("Độ chính xác (%)")
    ax2.set_title("So Sánh Các Thuật Toán", pad=12)

    for bar, val in zip(bars2, scores):
        ax2.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=10,
                 fontweight='bold', color='#e6edf3')

    fig2.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=120, facecolor='#161b22')
    buf2.seek(0)
    chart2 = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig2)

    # =====================
    # BIỂU ĐỒ 3: LỊCH SỬ (LINE CHART)
    # =====================
    set_dark_style()

    hist = pd.read_csv(HISTORY_FILE)

    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    fig3.patch.set_facecolor('#161b22')

    x   = range(len(hist))
    pcts = hist["risk"].astype(float).tolist()

    # Vùng màu nền theo outcome
    ax3.axhspan(0,  50, color='#00d4aa', alpha=0.05, zorder=0)
    ax3.axhspan(50, 100, color='#ff6b6b', alpha=0.05, zorder=0)

    # Đường
    ax3.plot(list(x), pcts, color='#00d4aa', linewidth=2.5, zorder=4, marker='o',
             markersize=6, markerfacecolor='#161b22', markeredgecolor='#00d4aa',
             markeredgewidth=2)

    # Fill dưới đường
    ax3.fill_between(list(x), pcts, alpha=0.12, color='#00d4aa', zorder=2)

    # Đường ngưỡng quyết định 50%
    ax3.axhline(50, color='#ffd166', linestyle='--', linewidth=1.5, alpha=0.8, label='Ngưỡng quyết định (50%)')

    ax3.set_ylim(0, 105)
    ax3.set_xlim(-0.3, max(len(hist)-1, 0) + 0.3)
    ax3.yaxis.grid(True, zorder=0, alpha=0.4)
    ax3.set_axisbelow(True)

    # Label trục X — hiện ngày giờ
    if len(hist) <= 15:
        ax3.set_xticks(list(x))
        ax3.set_xticklabels(hist["time"].tolist(), rotation=30, ha='right', fontsize=9)
    else:
        step = max(1, len(hist) // 10)
        ticks = list(range(0, len(hist), step))
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([hist["time"].iloc[i] for i in ticks], rotation=30, ha='right', fontsize=9)

    ax3.set_ylabel("% Nguy cơ")
    ax3.set_title("Lịch Sử Các Lần Kiểm Tra", pad=12)

    # Đánh dấu điểm hiện tại
    ax3.scatter([len(hist)-1], [pcts[-1]], color='#ffd166', s=80, zorder=5)

    # Legend
    l1 = mpatches.Patch(color='#ffd166', alpha=0.7, label='Ngưỡng quyết định (50%)')
    ax3.legend(handles=[l1], loc='upper left', fontsize=9,
               facecolor='#1c2330', edgecolor='#30363d', labelcolor='#7d8590')

    fig3.tight_layout()
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', dpi=120, facecolor='#161b22')
    buf3.seek(0)
    chart3 = base64.b64encode(buf3.getvalue()).decode()
    plt.close(fig3)

    # =====================
    # HISTORY DATA cho frontend
    # =====================
    history_list = all_data.to_dict(orient='records')

    return jsonify({
        "percent": percent,
        "outcome": outcome,
        "level":   level,
        "chart1":  chart1,
        "chart2":  chart2,
        "chart3":  chart3,
        "history": history_list
    })

import os

if __name__ == "__main__":
    port = int (os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    #app.run(debug=True)