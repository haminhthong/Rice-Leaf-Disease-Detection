"""Streamlit Web Dashboard ứng dụng Nhận Diện & Định Vị Bệnh Lá Lúa (RiceGuard AI).

Giao diện trực quan cao cấp dành cho bài toán Computer Vision trong Nông nghiệp Thông minh:
1. 🎯 Chẩn Đoán & Định Vị: Upload ảnh hoặc chọn ảnh mẫu 1-click, soi chi tiết vết bệnh (Zoom Crop),
   tính toán diện tích tổn thương (Infection Severity Index) và tải báo cáo.
2. ⚡ Chẩn Đoán Hàng Loạt: Tải lên nhiều ảnh cùng lúc, thống kê phân bổ bệnh theo lô và xuất CSV.
3. 💡 Khuyến Cáo Nông Học: Biện pháp xử lý thực địa chuyên sâu cho từng bệnh và cẩm nang 4 giai đoạn vụ lúa.
4. 📈 Đánh Giá Mô Hình: Trực quan hóa mAP@0.5, Precision, Recall, ma trận nhầm lẫn & đường cong PR từ evaluate.py.
5. 🔬 Phân Tích Lỗi & Chống Rò Rỉ: Phân loại TP/FP/FN theo error_analysis.py, khử trùng lặp SHA-256 & gom cụm pHash.
6. 📖 Model Card & Kiến Trúc: Đặc tả kỹ thuật mạng YOLOv8s, pipeline 6 bước và hướng dẫn an toàn thực địa.
"""

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.settings import get_settings
from rice_leaf_detection.error_analysis import classify_lesion_size
from rice_leaf_detection.inference import RiceLeafDetector

logger = logging.getLogger(__name__)

# ==============================================================================
# CẤU HÌNH TRANG VÀ DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="RiceGuard AI | Nhận Diện Bệnh Lá Lúa",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container chung */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 60%, #031c16 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 1.25rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 12px 30px -8px rgba(6, 78, 59, 0.45);
        margin-bottom: 1.8rem;
        color: #f0fdf4;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.18) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #a7f3d0;
        margin-bottom: 1rem;
        max-width: 900px;
        line-height: 1.5;
    }

    .badge-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #ecfdf5;
        backdrop-filter: blur(8px);
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.75rem;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.1);
        border-color: #cbd5e1;
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }

    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* Banners chẩn đoán kết quả */
    .diag-banner {
        border-radius: 1rem;
        padding: 1.25rem 1.6rem;
        margin: 1.1rem 0;
        border-left: 6px solid;
        box-shadow: 0 4px 14px -3px rgba(0, 0, 0, 0.05);
    }

    .diag-safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #10b981;
        color: #065f46;
    }

    .diag-blb {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-color: #f59e0b;
        color: #92400e;
    }

    .diag-brown-spot {
        background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%);
        border-color: #c084fc;
        color: #6b21a8;
    }

    .diag-mixed {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #ef4444;
        color: #991b1b;
    }

    /* Hộp khuyến cáo nông học */
    .advice-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.95rem;
        padding: 1.2rem 1.4rem;
        margin-top: 0.9rem;
        border-left: 4px solid #10b981;
    }

    .advice-header {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Thẻ crop tổn thương */
    .lesion-crop-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 0.75rem;
        text-align: center;
        box-shadow: 0 2px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.75rem;
    }

    .lesion-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    .lesion-blb {
        background: #fef3c7;
        color: #b45309;
    }

    .lesion-bs {
        background: #f3e8ff;
        color: #7e22ce;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# HÀM TẢI MÔ HÌNH VÀ DỮ LIỆU CACHED
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_detector(
    weights_path: str, image_size: int, confidence: float, iou: float
) -> RiceLeafDetector:
    """Khởi tạo và lưu cache singleton cho mô hình RiceLeafDetector."""
    return RiceLeafDetector(
        Path(weights_path),
        image_size=image_size,
        confidence=confidence,
        iou=iou,
    )


@st.cache_data(show_spinner=False)
def load_evaluation_data() -> tuple[dict | None, pd.DataFrame | None]:
    """Tải báo cáo số liệu đánh giá mô hình từ thư mục runs/evaluate."""
    val_dir = Path("runs/evaluate/_val")
    metrics_file = val_dir / "metrics.json"
    per_class_file = val_dir / "per_class_metrics.csv"

    metrics_data = None
    df_per_class = None

    if metrics_file.exists():
        try:
            metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Không thể đọc file metrics.json: %s", exc)

    if per_class_file.exists():
        try:
            df_per_class = pd.read_csv(per_class_file)
        except Exception as exc:
            logger.warning("Không thể đọc per_class_metrics.csv: %s", exc)

    return metrics_data, df_per_class


@st.cache_data(show_spinner=False)
def load_data_manifest() -> tuple[pd.DataFrame | None, dict | None]:
    """Tải manifest và báo cáo chất lượng dữ liệu data_report.json."""
    data_dir = Path("data/processed/rice_leaf_detection")
    manifest_file = data_dir / "manifest.csv"
    report_file = data_dir / "data_report.json"

    df_manifest = None
    report_data = None

    if manifest_file.exists():
        try:
            df_manifest = pd.read_csv(manifest_file)
        except Exception as exc:
            logger.warning("Không thể đọc manifest.csv: %s", exc)

    if report_file.exists():
        try:
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Không thể đọc data_report.json: %s", exc)

    return df_manifest, report_data


# ==============================================================================
# HERO BANNER
# ==============================================================================
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">🌾 RiceGuard AI — Nhận Diện & Định Vị Bệnh Lá Lúa</div>
    <div class="hero-subtitle">
        Hệ thống Trí tuệ Nhân tạo Computer Vision định vị tổn thương thực địa:
        <b>Bạc lá lúa</b> (<i>Bacterial Leaf Blight</i>) và <b>Đốm nâu</b> (<i>Brown Spot</i>).
        Áp dụng kiến trúc YOLOv8s kết hợp phương pháp chống rò rỉ dữ liệu (Group-aware Split)
        và phân tích mức độ xâm nhiễm phục vụ nông nghiệp chính xác.
    </div>
    <div class="badge-wrap">
        <span class="hero-badge">🎯 Object Detection (Lesion Localization)</span>
        <span class="hero-badge">⚡ YOLOv8s PyTorch Engine</span>
        <span class="hero-badge">🛡️ Leakage-Aware pHash Split</span>
        <span class="hero-badge">🔬 Micro-Lesion Crop Inspector</span>
        <span class="hero-badge">🌱 Precision AgTech</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# SIDEBAR: THAM SỐ SUY LUẬN & HỆ THỐNG
# ==============================================================================
st.sidebar.markdown("## ⚙️ Thiết Lập Suy Luận")

runtime_settings = get_settings()
default_weights = os.getenv("RICE_MODEL_PATH", runtime_settings.weights.as_posix())

# Tự động tìm trọng số khả dụng nếu đường dẫn mặc định chưa đúng
if not Path(default_weights).exists():
    found = list(Path("runs/train").glob("**/weights/best.pt"))
    if found:
        default_weights = str(found[0])
    elif Path("artifacts/model.pt").exists():
        default_weights = "artifacts/model.pt"

weights_input = st.sidebar.text_input(
    "Đường dẫn file trọng số (.pt):",
    default_weights,
    help="Đường dẫn đến file weights huấn luyện YOLOv8 (.pt).",
)
weights_path = Path(weights_input)

if weights_path.exists():
    st.sidebar.success(f"✅ Sẵn sàng ({weights_path.name})")
else:
    st.sidebar.error("⚠️ Chưa tìm thấy file trọng số")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Ngưỡng Lọc Dò Tìm")

confidence = st.sidebar.slider(
    "Độ tin cậy tối thiểu (Confidence):",
    min_value=0.10,
    max_value=0.95,
    value=float(os.getenv("RICE_CONFIDENCE", str(runtime_settings.confidence))),
    step=0.05,
    help="Chỉ giữ lại các vùng tổn thương có điểm tin cậy đạt ngưỡng này.",
)

iou = st.sidebar.slider(
    "Ngưỡng NMS IoU (Loại trùng lặp):",
    min_value=0.10,
    max_value=0.90,
    value=float(os.getenv("RICE_IOU", str(runtime_settings.iou))),
    step=0.05,
    help="Ngưỡng Non-Maximum Suppression để gộp các box chồng lấn trên cùng một vết bệnh.",
)

image_size = st.sidebar.select_slider(
    "Kích thước chuẩn hóa ảnh đầu vào:",
    options=[480, 640, 768, 800],
    value=int(os.getenv("RICE_IMAGE_SIZE", str(runtime_settings.image_size))),
    help="Kích thước resize trước khi đưa qua mạng nơ-ron (mặc định 640px).",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ Tùy Chọn Hiển Thị")
show_lesion_crop = st.sidebar.checkbox(
    "Kính lúp soi chi tiết từng vết bệnh (Zoom Crop)",
    value=True,
    help="Tự động cắt và hiển thị phóng to từng vùng tổn thương phát hiện được.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌾 2 Bệnh Mục Tiêu")
st.sidebar.markdown(
    """
- 🟡 **Bạc lá lúa** (*Xanthomonas oryzae*): Vết sọc dài dọc mép lá, ban đầu xanh tái úng nước,
  sau chuyển vàng rơm rồi bạc trắng, làm suy kiệt quang hợp nghiêm trọng.
- 🟤 **Đốm nâu** (*Bipolaris oryzae*): Đốm hình bầu dục hoặc mắt tròn màu nâu sẫm,
  tâm xám nhạt; chỉ thị đất phèn, thiếu kẽm hoặc thiếu dinh dưỡng trầm trọng.
"""
)

st.sidebar.markdown("---")
st.sidebar.caption("RiceGuard AI • Engine YOLOv8s • Native Python (No Docker)")


# ==============================================================================
# HỆ THỐNG 6 TABS CHỨC NĂNG
# ==============================================================================
(
    tab_single,
    tab_batch,
    tab_advice,
    tab_evaluation,
    tab_error_analysis,
    tab_architecture,
) = st.tabs(
    [
        "🎯 Chẩn Đoán & Định Vị",
        "⚡ Chẩn Đoán Hàng Loạt",
        "💡 Khuyến Cáo Nông Học",
        "📈 Đánh Giá Mô Hình",
        "🔬 Phân Tích Lỗi & Dữ Liệu",
        "📖 Model Card & Kỹ Thuật",
    ]
)


# ==============================================================================
# TAB 1: CHẨN ĐOÁN & ĐỊNH VỊ ĐƠN ẢNH
# ==============================================================================
with tab_single:
    col_input, col_view = st.columns([1, 1.2], gap="large")

    image_bgr: np.ndarray | None = None
    source_name = ""

    with col_input:
        st.markdown("### 📥 1. Chọn Nguồn Ảnh Thực Địa")

        input_mode = st.radio(
            "Phương thức chọn ảnh:",
            [
                "🖼️ Chọn ảnh mẫu có sẵn (1-Click)",
                "📁 Tải ảnh từ thiết bị (JPG, PNG, WebP)",
            ],
            horizontal=True,
        )

        if "🖼️ Chọn ảnh mẫu có sẵn (1-Click)" in input_mode:
            sample_dir = Path("data/sample")
            sample_options: dict[str, Path] = {}

            if (sample_dir / "bacterial_leaf_blight_sample.jpg").exists():
                sample_options["Mẫu 1: Bạc lá lúa (Bacterial Leaf Blight)"] = (
                    sample_dir / "bacterial_leaf_blight_sample.jpg"
                )
            if (sample_dir / "brown_spot_sample.jpg").exists():
                sample_options["Mẫu 2: Đốm nâu (Brown Spot)"] = sample_dir / "brown_spot_sample.jpg"
            if (sample_dir / "mixed_infection_sample.jpg").exists():
                sample_options["Mẫu 3: Nhiễm đồng thời cả 2 loại bệnh"] = (
                    sample_dir / "mixed_infection_sample.jpg"
                )

            if sample_options:
                chosen_sample = st.selectbox(
                    "Chọn mẫu bệnh điển hình:",
                    list(sample_options.keys()),
                )
                sample_path = sample_options[chosen_sample]
                image_bgr = cv2.imread(str(sample_path))
                source_name = sample_path.name
            else:
                st.warning("Chưa tìm thấy ảnh trong thư mục `data/sample/`.")

        else:
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc duyệt ảnh chụp phiến lá lúa:",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                source_name = uploaded_file.name

        if image_bgr is not None:
            st.markdown("##### 🔍 Xem Trước Ảnh Gốc:")
            img_h, img_w = image_bgr.shape[:2]
            st.image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                caption=f"{source_name} • Kích thước: {img_w}x{img_h}px",
                width="stretch",
            )

    with col_view:
        st.markdown("### 🔍 2. Kết Quả Định Vị & Phân Tích")

        if image_bgr is None:
            st.info("👈 Hãy chọn ảnh mẫu hoặc tải ảnh lá lúa ở cột bên trái để bắt đầu phân tích.")

        elif not weights_path.exists():
            st.error(
                f"❌ Chưa tìm thấy file mô hình tại `{weights_path}`.\n\n"
                "Vui lòng kiểm tra lại đường dẫn ở Sidebar hoặc huấn luyện mô hình bằng lệnh:\n"
                "```bash\n"
                "python scripts/train.py --config configs/default.yaml\n"
                "```"
            )

        else:
            start_time = time.perf_counter()
            with st.spinner("Đang chạy mô hình YOLOv8s định vị tổn thương..."):
                try:
                    detector = load_detector(str(weights_path), image_size, confidence, iou)
                    prediction, result = detector.predict(
                        image_bgr,
                        confidence=confidence,
                        iou=iou,
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Phân loại số lượng từng bệnh
                    blb_detections = [d for d in prediction.detections if d.class_id == 0]
                    bs_detections = [d for d in prediction.detections if d.class_id == 1]
                    total_count = len(prediction.detections)

                    # Tính toán tổng diện tích tổn thương và chỉ số nghiêm trọng
                    img_h, img_w = image_bgr.shape[:2]
                    total_leaf_pixels = img_h * img_w
                    total_lesion_pixels = 0.0

                    for det in prediction.detections:
                        x1, y1, x2, y2 = det.box_xyxy
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        total_lesion_pixels += bw * bh

                    severity_ratio = (
                        (total_lesion_pixels / total_leaf_pixels * 100)
                        if total_leaf_pixels > 0
                        else 0.0
                    )

                    # Xếp loại mức độ nghiêm trọng
                    if total_count == 0:
                        severity_label = "0.0% (Lành lặn)"
                        severity_color = "#10b981"
                    elif severity_ratio < 5.0:
                        severity_label = f"{severity_ratio:.1f}% (Cấp 1: Nhẹ)"
                        severity_color = "#f59e0b"
                    elif severity_ratio <= 20.0:
                        severity_label = f"{severity_ratio:.1f}% (Cấp 2: Trung bình)"
                        severity_color = "#ea580c"
                    else:
                        severity_label = f"{severity_ratio:.1f}% (Cấp 3: Nghiêm trọng)"
                        severity_color = "#dc2626"

                    # 5 Thẻ KPI Tương Tác
                    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                    with kpi1:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">Bạc Lá Lúa</div>
                            <div class="metric-value" style="color: #d97706;">{len(blb_detections)}</div>
                            <div class="metric-sub">vùng tổn thương</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with kpi2:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">Đốm Nâu</div>
                            <div class="metric-value" style="color: #9333ea;">{len(bs_detections)}</div>
                            <div class="metric-sub">vết đốm</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with kpi3:
                        max_conf = (
                            max((d.confidence for d in prediction.detections), default=0.0)
                            if total_count > 0
                            else 0.0
                        )
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">Tin Cậy Max</div>
                            <div class="metric-value" style="color: #0284c7;">{max_conf:.1%}</div>
                            <div class="metric-sub">ngưỡng ≥ {confidence:.2f}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with kpi4:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">Tổn Thương</div>
                            <div class="metric-value" style="color: {severity_color};">{severity_ratio:.1f}%</div>
                            <div class="metric-sub">{severity_label.split("(")[-1].replace(")", "")}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with kpi5:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">Thời Gian</div>
                            <div class="metric-value" style="color: #10b981;">{latency_ms:.0f}ms</div>
                            <div class="metric-sub">kích thước {image_size}px</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Banner chẩn đoán tổng thể
                    if total_count == 0:
                        st.markdown(
                            """
                        <div class="diag-banner diag-safe">
                            <h4 style="margin:0 0 0.3rem 0;">✅ KHÔNG PHÁT HIỆN DẤU HIỆU BỆNH MỤC TIÊU</h4>
                            Phiến lá không có vùng tổn thương Bạc lá lúa hoặc Đốm nâu vượt qua ngưỡng tin cậy. Cây lúa phát triển bình thường.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif len(blb_detections) > 0 and len(bs_detections) > 0:
                        st.markdown(
                            f"""
                        <div class="diag-banner diag-mixed">
                            <h4 style="margin:0 0 0.3rem 0;">⚠️ CẢNH BÁO: NHIỄM ĐỒNG THỜI CẢ 2 BỆNH</h4>
                            Phát hiện <b>{len(blb_detections)} vùng Bạc lá</b> do vi khuẩn <i>Xanthomonas oryzae</i> và
                            <b>{len(bs_detections)} vết Đốm nâu</b> do nấm <i>Bipolaris oryzae</i>.
                            Cần áp dụng phác đồ xử lý tích hợp ngay để tránh thất thu năng suất.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif len(blb_detections) > 0:
                        st.markdown(
                            f"""
                        <div class="diag-banner diag-blb">
                            <h4 style="margin:0 0 0.3rem 0;">🟡 PHÁT HIỆN BỆNH BẠC LÁ LÚA (Bacterial Leaf Blight)</h4>
                            Định vị được <b>{len(blb_detections)} vùng tổn thương</b>. Mầm bệnh là vi khuẩn <i>Xanthomonas oryzae</i>.
                            Cần dừng ngay bón đạm và giữ thông thoáng mặt ruộng.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div class="diag-banner diag-brown-spot">
                            <h4 style="margin:0 0 0.3rem 0;">🟤 PHÁT HIỆN BỆNH ĐỐM NÂU (Brown Spot)</h4>
                            Định vị được <b>{len(bs_detections)} vết đốm nâu</b>. Tác nhân là nấm <i>Bipolaris oryzae</i>.
                            Dấu hiệu cảnh báo đất chua phèn hoặc thiếu hụt kẽm/kali.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Ảnh kết quả gán bounding box
                    annotated_bgr = result.plot()
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    st.markdown("##### 🎯 Ảnh Định Vị Bounding Box:")
                    st.image(
                        annotated_rgb,
                        caption="Vùng tổn thương được đóng khung bởi mô hình YOLOv8s",
                        width="stretch",
                    )

                    # Tải ảnh về máy
                    is_success, buffer = cv2.imencode(".png", annotated_bgr)
                    if is_success:
                        st.download_button(
                            label="📥 Tải Ảnh Đã Gán Nhãn (PNG)",
                            data=buffer.tobytes(),
                            file_name=f"detected_{source_name}.png",
                            mime="image/png",
                        )

                    # ==============================================================
                    # KÍNH LÚP SOI TỔN THƯƠNG (LESION ZOOM CROPS)
                    # ==============================================================
                    if show_lesion_crop and total_count > 0:
                        st.markdown("---")
                        st.markdown("#### 🔬 Kính Lúp Soi Chi Tiết Vết Bệnh (Lesion Gallery):")
                        st.caption(
                            "Thu phóng trực tiếp các vùng tổn thương để quan sát rìa vết bệnh, "
                            "màu sắc và mức độ hoại tử tế bào lá:"
                        )

                        # Giới hạn tối đa 8 vết bệnh tiêu biểu
                        crop_cols = st.columns(min(4, total_count))
                        for idx, det in enumerate(prediction.detections[:8]):
                            col_target = crop_cols[idx % min(4, total_count)]
                            x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
                            # Mở rộng nhẹ vùng biên để người dùng dễ nhìn bối cảnh lá
                            pad = 4
                            cx1 = max(0, x1 - pad)
                            cy1 = max(0, y1 - pad)
                            cx2 = min(img_w, x2 + pad)
                            cy2 = min(img_h, y2 + pad)

                            cropped_bgr = image_bgr[cy1:cy2, cx1:cx2]
                            if cropped_bgr.size > 0:
                                cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                                size_cat = classify_lesion_size(det.box_xyxy, img_w, img_h)

                                with col_target:
                                    badge_class = "lesion-blb" if det.class_id == 0 else "lesion-bs"
                                    st.markdown(
                                        f"""
                                    <div class="lesion-crop-card">
                                        <div style="font-size: 0.75rem; font-weight: 700; color: #475569;">Vết bệnh #{idx + 1}</div>
                                        <span class="lesion-badge {badge_class}">{det.class_name_vi}</span>
                                        <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.2rem;">
                                            Tin cậy: <b>{det.confidence:.1%}</b> • Size: <i>{size_cat}</i>
                                        </div>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )
                                    st.image(
                                        cropped_rgb,
                                        width="stretch",
                                    )

                    # Bảng chi tiết tọa độ Bounding Box
                    if total_count > 0:
                        st.markdown("---")
                        st.markdown("##### 📋 Bảng Chi Tiết Vị Trí & Tọa Độ:")
                        rows = []
                        for idx, det in enumerate(prediction.detections, 1):
                            x1, y1, x2, y2 = det.box_xyxy
                            bw = max(0.0, x2 - x1)
                            bh = max(0.0, y2 - y1)
                            area_pct = (bw * bh) / total_leaf_pixels * 100
                            size_cat = classify_lesion_size(det.box_xyxy, img_w, img_h)

                            rows.append(
                                {
                                    "#": idx,
                                    "Bệnh (Tiếng Việt)": det.class_name_vi,
                                    "Tên Khoa Học": det.class_name,
                                    "Độ Tin Cậy": f"{det.confidence:.2%}",
                                    "Tọa Độ [x1, y1, x2, y2]": (
                                        f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
                                    ),
                                    "Diện Tích (% Lá)": f"{area_pct:.2f}%",
                                    "Phân Loại Kích Thước": size_cat.capitalize(),
                                }
                            )

                        df_boxes = pd.DataFrame(rows)
                        st.dataframe(df_boxes, width="stretch")

                        # Nút tải bảng kết quả
                        csv_data = df_boxes.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Xuất Báo Cáo Chẩn Đoán (CSV)",
                            data=csv_data,
                            file_name=f"report_{source_name}.csv",
                            mime="text/csv",
                        )

                except Exception as exc:
                    logger.exception("Lỗi khi suy luận trên Streamlit")
                    st.error(f"Đã xảy ra lỗi khi phân tích ảnh: {exc}")


# ==============================================================================
# TAB 2: CHẨN ĐOÁN HÀNG LOẠT (BATCH INFERENCE PIPELINE)
# ==============================================================================
with tab_batch:
    st.markdown("### ⚡ Chẩn Đoán Hàng Loạt Nhiều Mẫu Lá (Batch Inference)")
    st.markdown(
        """
    Tính năng cho phép nạp một lô ảnh thực địa cùng lúc để đánh giá mức độ lây lan
    trên toàn diện tích khảo sát, xuất báo cáo CSV phục vụ quản lý dịch bệnh diện rộng.
    """
    )

    batch_col1, batch_col2 = st.columns([1, 1], gap="medium")

    with batch_col1:
        uploaded_batch = st.file_uploader(
            "Tải lên danh sách nhiều ảnh cùng lúc:",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_uploader",
        )

    with batch_col2:
        st.markdown("#### Hoặc kiểm tra nhanh trên tập ảnh mẫu:")
        btn_run_samples = st.button(
            "🧪 Chạy Thử Trên Toàn Bộ Mẫu Mặc Định",
            use_container_width=True,
        )

    images_to_process: list[tuple[str, np.ndarray]] = []

    if uploaded_batch:
        for f in uploaded_batch:
            file_bytes = np.frombuffer(f.getvalue(), dtype=np.uint8)
            decoded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if decoded is not None:
                images_to_process.append((f.name, decoded))

    elif btn_run_samples:
        sample_dir = Path("data/sample")
        for p in sample_dir.glob("*.jpg"):
            dec = cv2.imread(str(p))
            if dec is not None:
                images_to_process.append((p.name, dec))

    if images_to_process:
        if not weights_path.exists():
            st.error("Chưa có file weights để chạy suy luận batch.")
        else:
            st.markdown(f"**Đang xử lý lô gồm {len(images_to_process)} ảnh...**")
            progress_bar = st.progress(0)

            detector = load_detector(str(weights_path), image_size, confidence, iou)

            batch_results = []
            total_blb = 0
            total_bs = 0
            infected_images = 0

            for idx, (img_name, img_mat) in enumerate(images_to_process):
                pred, _ = detector.predict(img_mat, confidence=confidence, iou=iou)
                n_blb = sum(1 for d in pred.detections if d.class_id == 0)
                n_bs = sum(1 for d in pred.detections if d.class_id == 1)
                n_total = len(pred.detections)

                total_blb += n_blb
                total_bs += n_bs
                if n_total > 0:
                    infected_images += 1

                if n_total == 0:
                    diag = "Lành lặn (Không có triệu chứng)"
                elif n_blb > 0 and n_bs > 0:
                    diag = "Nhiễm đồng thời (Bạc lá + Đốm nâu)"
                elif n_blb > 0:
                    diag = "Bạc lá lúa (BLB)"
                else:
                    diag = "Đốm nâu (Brown Spot)"

                max_c = max((d.confidence for d in pred.detections), default=0.0)

                batch_results.append(
                    {
                        "Tên Tệp": img_name,
                        "Kích Thước": f"{img_mat.shape[1]}x{img_mat.shape[0]}",
                        "Vùng Bạc Lá": n_blb,
                        "Vết Đốm Nâu": n_bs,
                        "Tổng Tổn Thương": n_total,
                        "Chẩn Đoán Ưu Thế": diag,
                        "Tin Cậy Cao Nhất": f"{max_c:.1%}" if n_total > 0 else "-",
                        "Trạng Thái": pred.status,
                    }
                )
                progress_bar.progress((idx + 1) / len(images_to_process))

            st.success(f"✅ Hoàn tất xử lý {len(images_to_process)} ảnh thực địa!")

            # Thống kê nhanh toàn lô
            b_kpi1, b_kpi2, b_kpi3, b_kpi4 = st.columns(4)
            b_kpi1.metric("Tổng Ảnh Xử Lý", len(images_to_process))
            b_kpi2.metric("Ảnh Có Triệu Chứng", f"{infected_images}/{len(images_to_process)}")
            b_kpi3.metric("Tổng Vùng Bạc Lá", total_blb)
            b_kpi4.metric("Tổng Vết Đốm Nâu", total_bs)

            df_batch = pd.DataFrame(batch_results)
            st.dataframe(df_batch, width="stretch")

            # Biểu đồ phân bổ bệnh
            st.markdown("##### 📊 Tỷ Lệ Phát Hiện Mầm Bệnh Trong Lô:")
            chart_data = pd.DataFrame(
                {
                    "Số lượng tổn thương": [total_blb, total_bs],
                },
                index=["Bạc lá lúa (Xanthomonas)", "Đốm nâu (Bipolaris)"],
            )
            st.bar_chart(chart_data)

            # Nút xuất kết quả batch
            csv_batch = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Tải Xuống Báo Cáo Toàn Bộ Lô (CSV)",
                data=csv_batch,
                file_name=f"batch_diagnosis_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ==============================================================================
# TAB 3: KHUYẾN CÁO NÔNG HỌC CHUYÊN SÂU
# ==============================================================================
with tab_advice:
    st.markdown("### 💡 Cẩm Nang Khuyến Cáo Kỹ Thuật Nông Nghiệp & Bảo Vệ Thực Vật")
    st.markdown(
        """
    Hệ thống tổng hợp các biện pháp canh tác bền vững theo tài liệu của Cục Bảo vệ Thực vật
    và Viện Nghiên cứu Lúa Quốc tế (IRRI), kết hợp quản lý dịch hại tổng hợp (IPM).
    """
    )

    adv_col1, adv_col2 = st.columns([1, 1], gap="large")

    with adv_col1:
        st.markdown(
            """
        <div class="advice-card" style="border-left-color: #f59e0b;">
            <div class="advice-header" style="color: #b45309;">
                🌾 BỆNH BẠC LÁ LÚA (Xanthomonas oryzae pv. oryzae)
            </div>
            <p><strong>Cơ chế lây lan:</strong> Vi khuẩn xâm nhập qua khí khổng hoặc vết thương rách lá
            do giông bão, gió mạnh cọ xát. Lây truyền cực nhanh qua giọt dịch và nguồn nước tưới.</p>
            <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #e2e8f0;">
            <p><strong>Biện pháp can thiệp cấp bách:</strong></p>
            <ul>
                <li><strong>Quản lý phân bón:</strong> NGƯNG NGAY việc bón thúc phân đạm (N) hoặc phun phân bón lá có đạm. Tăng cường Kali (K) và Silic để thúc đẩy hình thành lớp cutin dày cứng trên biểu bì lá.</li>
                <li><strong>Quản lý nguồn nước:</strong> Tháo cạn nước ruộng đến mức tối thiểu (giữ ẩm chân ruộng), tuyệt đối không để nước ngập sâu hoặc luân chuyển nước từ ruộng bệnh sang ruộng lành.</li>
                <li><strong>Tác động cơ học:</strong> Không lội ruộng bón phân, làm cỏ khi lá còn đọng sương sớm để tránh phát tán vi khuẩn cơ học.</li>
                <li><strong>Hoạt chất BVTV đăng ký:</strong> Ưu tiên các hoạt chất kháng sinh sinh học như <i>Ningnanmycin</i>, <i>Oxolinic acid</i>, hoặc hợp chất đồng sinh học; luân phiên để tránh kháng thuốc.</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with adv_col2:
        st.markdown(
            """
        <div class="advice-card" style="border-left-color: #a855f7;">
            <div class="advice-header" style="color: #7e22ce;">
                🌾 BỆNH ĐỐM NÂU (Bipolaris oryzae)
            </div>
            <p><strong>Cơ chế phát sinh:</strong> Bệnh đốm nâu là bệnh "chỉ thị sinh thái".
            Thường bùng phát khi cây lúa bị stress sinh lý: đất ngộ độc phèn, ngộ độc hữu cơ,
            hoặc thiếu hụt nghiêm trọng vi lượng kẽm (Zn), silic (Si) và lân (P).</p>
            <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #e2e8f0;">
            <p><strong>Biện pháp can thiệp bền vững:</strong></p>
            <ul>
                <li><strong>Cải tạo thổ nhưỡng:</strong> Rửa phèn, bón vôi bột (300-500 kg/ha) hạ độc phèn nhôm/sắt. Bổ sung phân lân nung chảy để kích thích hệ rễ ăn sâu.</li>
                <li><strong>Bổ sung vi lượng:</strong> Phun bổ sung kẽm (ZnSO4) hoặc kẽm chelate qua lá nhằm tái thiết lập enzyme phòng vệ của cây lúa.</li>
                <li><strong>Duy trì ẩm độ:</strong> Tránh để mặt ruộng khô nứt nẻ trong các giai đoạn mẫn cảm (đẻ nhánh rộ và làm đòng).</li>
                <li><strong>Xử lý hạt giống vụ sau:</strong> Ngâm hạt giống bằng nước ấm 3 sôi 2 lạnh (54°C trong 15 phút) trước khi ủ để diệt bào tử nấm ngủ ngầm trên vỏ trấu.</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📅 Lịch Trình Bảo Vệ Cây Lúa Theo 4 Giai Đoạn Vụ:")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Giai Đoạn 1</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">Gieo Sạ & Mạ Non</div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.4rem;">
                Xử lý giống sạch khuẩn, bón lót lân và vôi. Tránh sạ quá dày (tiêu chuẩn 80-100 kg/ha).
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Giai Đoạn 2</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">Đẻ Nhánh Rộ</div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.4rem;">
                Tưới ngập - khô xen kẽ (AWD) để kích rễ sâu. Quan sát mép lá khi có mưa dông đầu mùa.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Giai Đoạn 3</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">Làm Đòng</div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.4rem;">
                Bón đón đòng cân đối N-P-K theo bảng so màu lá lúa (LCC). Phòng ngừa đốm nâu tấn công lá đòng.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Giai Đoạn 4</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">Trổ & Chín Sáp</div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.4rem;">
                Bảo vệ 3 lá công năng; tuân thủ thời gian cách ly (PHI) khi dùng bất kỳ chế phẩm BVTV nào.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 4: ĐÁNH GIÁ MÔ HÌNH (EVALUATION & PERFORMANCE)
# ==============================================================================
with tab_evaluation:
    st.markdown("### 📈 Báo Cáo Hiệu Năng & Độ Đo Đánh Giá (Object Detection Metrics)")
    st.markdown(
        """
    Kết quả đánh giá khách quan dựa trên bộ dữ liệu kiểm định độc lập (Validation/Test Split),
    sử dụng module đánh giá chuẩn `src/rice_leaf_detection/evaluate.py`.
    """
    )

    metrics_data, df_per_class = load_evaluation_data()

    if metrics_data is not None:
        p_val = metrics_data.get("precision", 0.0)
        r_val = metrics_data.get("recall", 0.0)
        map50 = metrics_data.get("mAP50", 0.0)
        map50_95 = metrics_data.get("mAP50-95", 0.0)
        split_name = metrics_data.get("split", "val").upper()

        m_kpi1, m_kpi2, m_kpi3, m_kpi4 = st.columns(4)
        m_kpi1.metric("Precision (Độ Chính Xác)", f"{p_val:.2%}")
        m_kpi2.metric("Recall (Độ Thu Hồi)", f"{r_val:.2%}")
        m_kpi3.metric("mAP @ IoU 0.50", f"{map50:.2%}")
        m_kpi4.metric("mAP @ IoU 0.50:0.95", f"{map50_95:.2%}")

        if df_per_class is not None:
            st.markdown(f"#### 📋 Bảng Đánh Giá Chi Tiết Từng Lớp Bệnh (Tập {split_name}):")
            st.dataframe(df_per_class, width="stretch")

    else:
        st.info("Chưa tìm thấy file `runs/evaluate/_val/metrics.json`.")

    st.markdown("---")
    st.markdown("#### 🖼️ Đồ Thị Đánh Giá & Ma Trận Nhầm Lẫn (Evaluation Curves):")

    val_dir = Path("runs/evaluate/_val")
    cf_norm = val_dir / "confusion_matrix_normalized.png"
    cf_raw = val_dir / "confusion_matrix.png"
    pr_curve = val_dir / "BoxPR_curve.png"
    f1_curve = val_dir / "BoxF1_curve.png"

    tab_cf, tab_pr, tab_f1, tab_batches = st.tabs(
        [
            "📊 Confusion Matrix (Ma Trận Nhầm Lẫn)",
            "📉 Precision-Recall Curve (PR)",
            "🎯 F1-Confidence Curve",
            "🖼️ Mẫu Đánh Giá Lô Thực Nghiệm",
        ]
    )

    with tab_cf:
        c1, c2 = st.columns(2)
        if cf_norm.exists():
            with c1:
                st.image(str(cf_norm), caption="Confusion Matrix Chuẩn Hóa (%)", width="stretch")
        if cf_raw.exists():
            with c2:
                st.image(
                    str(cf_raw), caption="Confusion Matrix Giá Trị Thô (Counts)", width="stretch"
                )

    with tab_pr:
        if pr_curve.exists():
            st.image(
                str(pr_curve),
                caption="Đường cong Precision-Recall (BoxPR Curve) ở IoU=0.5",
                width="stretch",
            )
        else:
            st.caption("Chưa có đồ thị BoxPR_curve.png")

    with tab_f1:
        if f1_curve.exists():
            st.image(
                str(f1_curve),
                caption="Đường cong F1-Score theo ngưỡng Confidence",
                width="stretch",
            )
        else:
            st.caption("Chưa có đồ thị BoxF1_curve.png")

    with tab_batches:
        batch_imgs = list(val_dir.glob("val_batch*_pred.jpg"))
        if batch_imgs:
            st.markdown("Các mẻ ảnh Validation được mô hình dự đoán tự động:")
            b_cols = st.columns(min(3, len(batch_imgs)))
            for idx, b_path in enumerate(batch_imgs[:3]):
                with b_cols[idx]:
                    st.image(str(b_path), caption=b_path.name, width="stretch")
        else:
            st.caption("Chưa có ảnh val_batch*_pred.jpg")


# ==============================================================================
# TAB 5: PHÂN TÍCH LỖI & CHỐNG RÒ RỈ DỮ LIỆU
# ==============================================================================
with tab_error_analysis:
    st.markdown("### 🔬 Phân Tích Lỗi Mô Hình & Quản Trị Dữ Liệu Chống Rò Rỉ")
    st.markdown(
        """
    Phương pháp luận kiểm tra chất lượng từ `src/rice_leaf_detection/error_analysis.py`
    và chiến lược chia tách dữ liệu an toàn chống rò rỉ (Data Leakage Prevention).
    """
    )

    err_col1, err_col2 = st.columns([1.1, 1], gap="large")

    with err_col1:
        st.markdown("#### 🎯 Phân Loại Lỗi Phát Hiện (Error Breakdown Taxonomy)")
        st.markdown(
            """
        - **True Positive (TP):** Bounding box khớp nhãn Ground Truth (IoU ≥ 0.50) đúng lớp bệnh.
        - **False Positive (FP - Báo nhầm):**
          - *FP Duplicate:* Dự đoán lặp lại nhiều box trên cùng 1 tổn thương (do NMS chưa đủ chặt).
          - *FP Localization:* Phát hiện trúng mầm bệnh nhưng IoU lệch vị trí (IoU < 0.50).
          - *FP Background:* Nhận nhầm gân lá tự nhiên hoặc giọt nước/đất bám thành mầm bệnh.
          - *FP Classification:* Bạc lá bị nhầm sang Đốm nâu hoặc ngược lại.
        - **False Negative (FN - Bỏ sót):** Tổn thương thật có trên lá nhưng mô hình bỏ qua
          (thường gặp ở các vết đốm nâu mới nhú có kích thước quá bé).
        """
        )

        st.markdown("#### 📐 Thách Thức Theo Kích Cỡ Tổn Thương")
        st.markdown(
            """
        - **Nhỏ (Small < 5% diện tích lá):** Vết chấm kim đốm nâu. Khó nhất đối với mạng nơ-ron vì độ phân giải đặc trưng thấp.
        - **Trung bình (Medium 5% - 20%):** Vết đốm nâu tròn hoặc vệt bạc lá chớm phát triển.
        - **Lớn (Large > 20%):** Vết sọc bạc lá cháy toàn bộ chóp lá, phân bố rộng.
        """
        )

    with err_col2:
        st.markdown("#### 🛡️ Thống Kê Chống Rò Rỉ Dữ Liệu (Anti-Leakage)")
        df_manifest, report_data = load_data_manifest()

        if df_manifest is not None and report_data is not None:
            summary = report_data.get("summary", {})
            st.metric("Tổng Số Ảnh Hợp Lệ", len(df_manifest))

            c_dup1, c_dup2 = st.columns(2)
            c_dup1.metric(
                "Khử Trùng Tuyệt Đối (SHA-256)",
                summary.get("exact_duplicates_removed", 0),
                help="Loại bỏ các ảnh copy giống hệt nhau về byte.",
            )
            c_dup2.metric(
                "Gom Cụm Tương Tự (pHash)",
                summary.get("near_duplicate_links", 0),
                help="Gom nhóm ảnh chụp cùng phiến lá bằng Perceptual Hash Hamming.",
            )

            st.markdown("##### Phân Bố Theo Tập Dữ Liệu:")
            split_stats = df_manifest.groupby("split").agg(
                Tong_Anh=("output_image", "count"),
                Bac_La_Lua=("instances_class_0", "sum"),
                Dom_Nau=("instances_class_1", "sum"),
            )
            st.dataframe(split_stats, width="stretch")

        else:
            st.info("Chưa tìm thấy dữ liệu `data/processed/rice_leaf_detection/manifest.csv`.")

    if df_manifest is not None:
        st.markdown("---")
        with st.expander("📄 Khám phá Manifest Explorer (Lọc & Tìm kiếm dữ liệu)"):
            filter_split = st.selectbox(
                "Lọc theo tập phân chia (Split):",
                ["Tất cả", "train", "val", "test"],
            )
            filtered_df = df_manifest
            if filter_split != "Tất cả":
                filtered_df = filtered_df[filtered_df["split"] == filter_split]

            st.dataframe(
                filtered_df[
                    [
                        "image_id",
                        "split",
                        "group_id",
                        "instances_class_0",
                        "instances_class_1",
                        "width",
                        "height",
                    ]
                ].head(50),
                width="stretch",
            )


# ==============================================================================
# TAB 6: MODEL CARD & KIẾN TRÚC KỸ THUẬT
# ==============================================================================
with tab_architecture:
    st.markdown("### 📖 Model Card & Kiến Trúc Kỹ Thuật Hệ Thống")

    arch_col1, arch_col2 = st.columns([1, 1], gap="large")

    with arch_col1:
        st.markdown("#### 🎯 So Sánh Detection vs Classification")
        st.markdown(
            """
        | Tiêu Chí | Phân Loại Ảnh (Classification) | Định Vị Vết Bệnh (Object Detection - YOLOv8s) |
        | :--- | :--- | :--- |
        | **Đầu ra** | 1 nhãn duy nhất cho cả ảnh | Danh sách tọa độ Bounding Box + Từng loại bệnh |
        | **Nhiễm đồng thời** | Bất lực (phải đoán mò 1 trong 2) | Nhận diện chính xác cả Bạc lá và Đốm nâu cùng lúc |
        | **Đo lường mức độ** | Không định lượng được | Tính % diện tích tổn thương (Infection Severity) |
        | **Ý nghĩa nông học** | Chỉ biết ruộng có bệnh chung chung | Hỗ trợ tính toán liều lượng thuốc BVTV chính xác |
        """
        )

        st.markdown("#### ⚙️ Cấu Hình Huấn Luyện (Hyperparameters)")
        st.markdown(
            """
        - **Kiến trúc:** YOLOv8 Small (YOLOv8s) anchor-free decoupled head.
        - **Kích thước chuẩn hóa:** 640 x 640 pixels.
        - **Bộ tối ưu hóa:** AdamW, Learning Rate 0.001, Weight Decay 0.0005.
        - **Augmentation chỉ trên Train (Train-only):** HSV shift, xoay nhẹ, lật ngang, mosaic nhẹ;
          **Val và Test tắt hoàn toàn data augmentation** để bảo toàn tính trung thực.
        """
        )

    with arch_col2:
        st.markdown("#### 🛡️ Giới Hạn Triển Khai & Đạo Đức AI")
        st.markdown(
            """
        - **Hiện tượng Domain Shift:** Mô hình được huấn luyện dựa trên ảnh chụp thực địa
          điều kiện ánh sáng tự nhiên. Ảnh chụp có đèn flash gắt hoặc ảnh vệ tinh chụp quá xa
          có thể làm giảm độ tin cậy.
        - **Tổn thương siêu nhỏ:** Các chấm đốm nâu có đường kính dưới 10 pixels có nguy cơ bị bỏ sót.
        - **Khuyến cáo an toàn:** Hệ thống mang tính chất **trợ lý trinh sát thực địa**,
          không thay thế vai trò của kỹ sư bảo vệ thực vật.
        - **Chính sách môi trường:** Không lạm dụng thuốc bảo vệ thực vật hóa học;
          luôn ưu tiên các biện pháp sinh học và cân đối dinh dưỡng ruộng lúa.
        """
        )

    st.markdown("---")
    st.markdown("#### 📋 Sơ Đồ Quy Trình 6 Bước Khép Kín (End-to-End Pipeline):")
    st.code(
        """
1. Thu Thập Dữ Liệu Thực Địa (Ảnh phiến lá kèm tọa độ nhãn YOLO)
                 │
                 ▼
2. Làm Sạch & Thẩm Định Nhãn (Missing Annotation ≠ Negative Image)
                 │
                 ▼
3. Chống Rò Rỉ Dữ Liệu (Khử trùng SHA-256 & Gom cụm pHash Hamming)
                 │
                 ▼
4. Huấn Luyện YOLOv8s (Train-only Augmentation, kiểm soát Overfitting)
                 │
                 ▼
5. Đánh Giá Khách Quan (mAP@0.5, Precision, Recall, Confusion Matrix)
                 │
                 ▼
6. Đóng Gói Phục Vụ Nông Nghiệp Thực Địa (FastAPI RESTful & Streamlit UI)
""",
        language="text",
    )
