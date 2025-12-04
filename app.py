import streamlit as st
from utils.video import process_video_with_preview, detect_video_realtime
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# FORCE RELOAD
import importlib
import sys
if 'utils.video' in sys.modules:
    importlib.reload(sys.modules['utils.video'])
    from utils.video import process_video_with_preview, detect_video_realtime

st.set_page_config(
    page_title="Đếm vật thể - Nhóm 12", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    .main-header {
        text-align: center;
        color: #1e3a8a;
        padding: 1.5rem 0;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .group-title {
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem 0;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>🎯 HỆ THỐNG ĐẾM VẬT THỂ</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("""
        <div style="text-align: center; padding: 20px 0; background-color: #0e1a2f; border-radius: 15px; margin-bottom: 20px;">
            <img src="https://tools1s.com/images/dkmh/vaa-logo.png" width="140">
            <p style="color: white; margin: 15px 0 0 0; font-size: 1.35rem; font-weight: bold; letter-spacing: 1px;">
                Nhóm 12 _ Lập trình Python
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tiêu đề nhóm
    st.markdown("""
        <div class='group-title'>
            📚 Nhóm 12<br>
            <span style='font-size: 0.9rem;'>ĐẾM vật thể</span>
        </div>
    """, unsafe_allow_html=True)
    
    # CHỌN MODEL
    st.markdown("<p style='color: white; font-weight: bold; font-size: 1.1rem; margin-top: 1rem;'>🤖 CHỌN MODEL</p>", unsafe_allow_html=True)
    
    model_folder = "models"
    if os.path.exists(model_folder):
        model_files = glob.glob(os.path.join(model_folder, "*.pt"))
        model_names = [os.path.basename(f) for f in model_files]
        
        if model_names:
            selected_model = st.selectbox(
                "Model:",
                model_names,
                index=model_names.index("best.pt") if "best.pt" in model_names else 0,
                label_visibility="collapsed"
            )
            model_path = os.path.join(model_folder, selected_model)
            
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;'>
                    <small style='color: white;'>
                    📦 Kích thước: {model_size:.1f} MB<br>
                    📁 {model_path}
                    </small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Không tìm thấy file model (.pt)")
            model_path = None
    else:
        st.error(f"⚠️ Thư mục '{model_folder}' không tồn tại")
        model_path = None
    
    st.markdown("---")
    
    # Navigation
    st.markdown("<p style='color: white; font-weight: bold; font-size: 1.1rem;'>🧭 CHỨC NĂNG</p>", unsafe_allow_html=True)
    
    option = st.selectbox(
        "Chọn chức năng:",
        ["🖼️ Đếm từ ảnh", "🎥 Đếm từ video", "📈 Visualize Training Results", "🧪 Test & Validation Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Thông tin nhóm
    with st.expander("👥 Thành viên nhóm", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        • Trần Thanh Đạt(Lead)<br>
        • Nguyễn Minh Phúc (Thành Viên)<br>
        • Trần Thanh Trúc (Thành Viên)<br>
        • Đồng Đức Mạnh (Thành Viên)<br>
        • Nguyễn Trần Duy Khánh (Thành Viên)
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📖 Hướng dẫn", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        <b>🖼️ Đếm từ ảnh:</b><br>
        Upload ảnh để Đếm vật thể<br><br>
        <b>🎥 Đếm từ video:</b><br>
        Upload video để Đếm và đếm vật thể<br><br>
        <b>📈 Visualize:</b><br>
        vật thểm kết quả training model
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# ẢNH
# -------------------------
if option == "🖼️ Đếm từ ảnh":
    st.header("📷 Đếm vật thể từ ảnh")
    
    if model_path is None or not os.path.exists(model_path):
        st.error("❌ Vui lòng chọn model hợp lệ từ sidebar")
        st.stop()
    
    @st.cache_resource
    def load_model(path):
        return YOLO(path)
    
    try:
        model = load_model(model_path)
        st.success(f"✅ Đã load model: {selected_model}")
    except Exception as e:
        st.error(f"❌ Lỗi load model: {str(e)}")
        st.stop()
    
    with st.expander("⚙️ Cài đặt thông số", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "🎯 Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Ngưỡng độ tin cậy"
            )
        
        with col2:
            iou_threshold = st.slider(
                "📦 IoU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Ngưỡng IoU cho NMS"
            )
        
        st.info(f"**Cài đặt:** Confidence ≥ {confidence_threshold:.2f} | IoU ≤ {iou_threshold:.2f}")
    
    upload_files = st.file_uploader(
        "🖼️ Chọn ảnh", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="JPG, JPEG, PNG"
    )
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🖼️ Ảnh {idx + 1}: {upload.name}")
            
            col_left, col_right = st.columns(2)
            
            try:
                file_bytes = upload.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if img is None:
                    st.error(f"❌ Không đọc được ảnh")
                    continue
                
                with col_left:
                    st.markdown("**Ảnh gốc**")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with st.spinner("🔍 Đang Đếm..."):
                    results = model(img, conf=confidence_threshold, iou=iou_threshold)[0]
                    annotated = results.plot()
                    
                    class_count = {}
                    for box in results.boxes:
                        cls_id = int(box.cls.item())
                        class_name = model.names[cls_id]
                        class_count[class_name] = class_count.get(class_name, 0) + 1
                
                with col_right:
                    st.markdown("**Kết quả**")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if class_count:
                    st.success("✅ Đếm thành công!")
                    with st.expander("📊 Thống kê", expanded=True):
                        cols = st.columns(len(class_count))
                        for idx, (name, count) in enumerate(class_count.items()):
                            with cols[idx]:
                                st.metric(str(name).capitalize(), count)
                        st.bar_chart(class_count)
                else:
                    st.warning("⚠️ Không Đếm được vật thể")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    else:
        st.info("👆 Upload ảnh để bắt đầu")

# -------------------------
# VIDEO
# -------------------------
elif option == "🎥 Đếm từ video":
    st.header("🎥 Đếm vật thể từ video")
    
    if model_path is None or not os.path.exists(model_path):
        st.error("❌ Vui lòng chọn model từ sidebar")
        st.stop()
    
    st.success(f"✅ Model: {selected_model}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        upload_files = st.file_uploader(
            "📹 Chọn video", 
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True
        )
    
    with col2:
        st.markdown("**⚙️ Cài đặt:**")
        show_preview = st.checkbox("Preview", value=True)
        save_output = st.checkbox("Lưu video", value=True)
        use_tracking = st.checkbox("Tracking", value=True, help="Đếm unique objects")
    
    with st.expander("🎯 Ngưỡng Đếm", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider("🎯 Confidence", 0.0, 1.0, 0.25, 0.05)
        
        with col2:
            iou_threshold = st.slider("📦 IoU", 0.0, 1.0, 0.45, 0.05)
        
        st.info(f"Conf ≥ {confidence_threshold:.2f} | IoU ≤ {iou_threshold:.2f} | Tracking: {'✅' if use_tracking else '❌'}")
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🎬 Video {idx + 1}: {upload.name}")
            
            try:
                temp_input = f"temp_input_{idx}.mp4"
                with open(temp_input, "wb") as f:
                    f.write(upload.read())
                
                with st.expander("📹 Video gốc", expanded=False):
                    st.video(temp_input)
                
                st.markdown("#### 🔍 Đang xử lý...")
                
                if save_output:
                    output_path = f"output_{idx}_{upload.name}"
                    output_path, class_count = process_video_with_preview(temp_input, output_path, show_preview,conf=confidence_threshold, iou=iou_threshold,model_path=model_path, use_tracking=use_tracking)
                else:
                    class_count = detect_video_realtime(
                        temp_input,
                        conf=confidence_threshold, iou=iou_threshold,
                        model_path=model_path, use_tracking=use_tracking
                    )
                    output_path = None
                
                st.success("✅ Hoàn thành!")
                
                if save_output and output_path and os.path.exists(output_path):
                    st.markdown("#### 🎥 Video đã xử lý")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            "⬇️ Tải video",
                            file,
                            f"detected_{upload.name}",
                            "video/mp4",
                            use_container_width=True
                        )
                
                if class_count and isinstance(class_count, dict):
                    with st.expander("📊 Thống kê", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Số lượng {'unique' if use_tracking else 'MAX'}:**")
                            for name, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
                                st.metric(str(name).capitalize(), count)
                        
                        with col2:
                            import pandas as pd
                            df = pd.DataFrame(list(class_count.items()), columns=['Class', 'Count'])
                            st.bar_chart(df.set_index('Class'))
                else:
                    st.warning("⚠️ Không đếm được vật thể")
                
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                import traceback
                with st.expander("Chi tiết"):
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload video")

# -------------------------
# VISUALIZE
# -------------------------
elif option == "📈 Visualize":
    st.header("📈 Kết quả Training")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        results_path = st.text_input(
            "📁 Đường dẫn:",
            value="run/detect/train",
            help="Ví dụ: run/detect/train"
        )
    
    with col2:
        refresh = st.button("🔄 Tải lại", use_container_width=True)
    
    if os.path.exists(results_path):
        st.success(f"✅ Tìm thấy: `{results_path}`")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📉 Curves", "🎯 Predictions", "📂 All Files"])
        
        with tab1:
            st.subheader("Ma trận nhầm lẫn")
            col1, col2 = st.columns(2)
            
            with col1:
                cm_path = os.path.join(results_path, "confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                else:
                    st.warning("⚠️ Không có confusion_matrix.png")
            
            with col2:
                cm_norm = os.path.join(results_path, "confusion_matrix_normalized.png")
                if os.path.exists(cm_norm):
                    st.image(cm_norm, caption="Normalized", use_container_width=True)
                else:
                    st.warning("⚠️ Không có confusion_matrix_normalized.png")
        
        with tab2:
            results_img = os.path.join(results_path, "results.png")
            if os.path.exists(results_img):
                st.image(results_img, caption="Training Results", use_container_width=True)
            else:
                st.warning("⚠️ Không có results.png")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                pr_path = os.path.join(results_path, "PR_curve.png")
                if os.path.exists(pr_path):
                    st.image(pr_path, caption="PR Curve", use_container_width=True)
            
            with col2:
                f1_path = os.path.join(results_path, "F1_curve.png")
                if os.path.exists(f1_path):
                    st.image(f1_path, caption="F1 Curve", use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                labels_path = os.path.join(results_path, "labels.jpg")
                if os.path.exists(labels_path):
                    st.image(labels_path, caption="Labels", use_container_width=True)
                
                train_batch = os.path.join(results_path, "train_batch0.jpg")
                if os.path.exists(train_batch):
                    st.image(train_batch, caption="Train Batch", use_container_width=True)
            
            with col2:
                val_labels = os.path.join(results_path, "val_batch0_labels.jpg")
                if os.path.exists(val_labels):
                    st.image(val_labels, caption="Val Labels", use_container_width=True)
                
                val_pred = os.path.join(results_path, "val_batch0_pred.jpg")
                if os.path.exists(val_pred):
                    st.image(val_pred, caption="Val Predictions", use_container_width=True)
        
        with tab4:
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(glob.glob(os.path.join(results_path, ext)))
            
            if image_files:
                st.write(f"**{len(image_files)}** files")
                cols = st.columns(3)
                for idx, img_path in enumerate(sorted(image_files)):
                    with cols[idx % 3]:
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.warning("⚠️ Không có file ảnh")
    else:
        st.error(f"❌ Không tìm thấy: `{results_path}`")



# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎯 Obj detection - Nhóm 12</p>
</div>
""", unsafe_allow_html=True)