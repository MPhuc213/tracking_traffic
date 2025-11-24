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
    page_title="Đếm xe cộ - Nhóm 2", 
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
st.markdown("<h1 class='main-header'>🎯 HỆ THỐNG ĐẾM XE CỘ</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("""
        <div style="text-align: center; padding: 20px 0; background-color: #0e1a2f; border-radius: 15px; margin-bottom: 20px;">
            <img src="https://tools1s.com/images/dkmh/vaa-logo.png" width="140">
            <p style="color: white; margin: 15px 0 0 0; font-size: 1.35rem; font-weight: bold; letter-spacing: 1px;">
                NHÓM 2 _ XỬ LÍ THỊ GIÁC
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tiêu đề nhóm
    st.markdown("""
        <div class='group-title'>
            📚 NHÓM 2<br>
            <span style='font-size: 0.9rem;'>ĐẾM XE CỘ</span>
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
        • Trần Kim Minh (Lead)<br>
        • Nguyễn Minh Phúc (Thành Viên)<br>
        • Vũ Thị Kim Loan (Thành Viên)<br>
        • Huỳnh Chí Danh (Thành Viên)<br>
        • Nguyễn Triệu Thiên Anh (Thành Viên)
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📖 Hướng dẫn", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        <b>🖼️ Đếm từ ảnh:</b><br>
        Upload ảnh để Đếm xe cộ<br><br>
        <b>🎥 Đếm từ video:</b><br>
        Upload video để Đếm và đếm xe<br><br>
        <b>📈 Visualize:</b><br>
        Xem kết quả training model
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# ẢNH
# -------------------------
if option == "🖼️ Đếm từ ảnh":
    st.header("📷 Đếm xe cộ từ ảnh")
    
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
    
    with st.expander("⚙️ Cài đặt Đếm", expanded=True):
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
                    st.warning("⚠️ Không Đếm được xe")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    else:
        st.info("👆 Upload ảnh để bắt đầu")

# -------------------------
# VIDEO
# -------------------------
elif option == "🎥 Đếm từ video":
    st.header("🎥 Đếm xe cộ từ video")
    
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
                    output_path, class_count = process_video_with_preview(
                        temp_input, output_path, show_preview,
                        conf=confidence_threshold, iou=iou_threshold,
                        model_path=model_path, use_tracking=use_tracking
                    )
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
                    st.warning("⚠️ Không đếm được xe")
                
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
elif option == "📈 Visualize Training Results":
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

# -------------------------
# TEST & VALIDATION RESULTS
# -------------------------
elif option == "🧪 Test & Validation Results":
    st.header("🧪 Kết quả Test/Validation Model")
    
    st.info("""
    📂 **Hiển thị kết quả test/validation đã chạy sẵn**
    
    Nhập đường dẫn đến thư mục chứa kết quả (confusion matrix, curves, predictions)
    """)
    
    # Nhập đường dẫn
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_results_path = st.text_input(
            "📁 Đường dẫn kết quả test:",
            value="run/detect/test",
            help="Ví dụ: run/detect/val, run/detect/test, run/detect/val2"
        )
    
    with col2:
        refresh_btn = st.button("🔄 Tải lại", use_container_width=True)
    
    # Kiểm tra thư mục
    if os.path.exists(test_results_path):
        st.success(f"✅ Tìm thấy: `{test_results_path}`")
        
        # Tabs hiển thị kết quả
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Confusion Matrix", 
            "📈 Performance Curves", 
            "🎯 Predictions",
            "📉 Additional Plots",
            "📂 All Files"
        ])
        
        # TAB 1: Confusion Matrix
        with tab1:
            st.subheader("Ma trận nhầm lẫn")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_path = os.path.join(test_results_path, "confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                    
                    # Download button
                    with open(cm_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            f,
                            "confusion_matrix.png",
                            "image/png",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Không tìm thấy confusion_matrix.png")
            
            with col2:
                cm_norm_path = os.path.join(test_results_path, "confusion_matrix_normalized.png")
                if os.path.exists(cm_norm_path):
                    st.image(cm_norm_path, caption="Normalized Confusion Matrix", use_container_width=True)
                    
                    with open(cm_norm_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download",
                            f,
                            "confusion_matrix_normalized.png",
                            "image/png",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Không tìm thấy confusion_matrix_normalized.png")
        
        # TAB 2: Performance Curves
        with tab2:
            st.subheader("Các đường cong đánh giá")
            
            # Row 1: PR và F1
            col1, col2 = st.columns(2)
            
            with col1:
                pr_path = os.path.join(test_results_path, "PR_curve.png")
                if os.path.exists(pr_path):
                    st.image(pr_path, caption="Precision-Recall Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có PR_curve.png")
            
            with col2:
                f1_path = os.path.join(test_results_path, "F1_curve.png")
                if os.path.exists(f1_path):
                    st.image(f1_path, caption="F1 Score Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có F1_curve.png")
            
            # Row 2: P và R
            col3, col4 = st.columns(2)
            
            with col3:
                p_path = os.path.join(test_results_path, "P_curve.png")
                if os.path.exists(p_path):
                    st.image(p_path, caption="Precision Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có P_curve.png")
            
            with col4:
                r_path = os.path.join(test_results_path, "R_curve.png")
                if os.path.exists(r_path):
                    st.image(r_path, caption="Recall Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có R_curve.png")
        
        # TAB 3: Predictions
        with tab3:
            st.subheader("Ví dụ predictions")
            
            # Tìm ảnh predictions
            pred_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                pred_images.extend(glob.glob(os.path.join(test_results_path, ext)))
            
            # Loại bỏ các file plot/curve
            exclude_keywords = ['confusion', 'curve', 'Curve', 'PR_', 'F1_', 'P_', 'R_', 'labels']
            pred_images = [img for img in pred_images if not any(kw in os.path.basename(img) for kw in exclude_keywords)]
            
            if pred_images:
                st.write(f"Tìm thấy **{len(pred_images)}** ảnh predictions")
                
                # Điều chỉnh số cột
                cols_per_row = st.slider("Số cột hiển thị:", 2, 4, 3, key="pred_cols")
                
                # Tìm kiếm
                search_term = st.text_input("🔍 Tìm kiếm ảnh:", placeholder="Nhập tên file...")
                
                # Lọc
                if search_term:
                    pred_images = [img for img in pred_images if search_term.lower() in os.path.basename(img).lower()]
                    st.write(f"Hiển thị **{len(pred_images)}** ảnh")
                
                # Hiển thị grid
                cols = st.columns(cols_per_row)
                for idx, img_path in enumerate(pred_images):
                    with cols[idx % cols_per_row]:
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                        
                        # Info
                        file_size = os.path.getsize(img_path) / 1024
                        st.caption(f"📦 {file_size:.1f} KB")
            else:
                st.info("ℹ️ Không tìm thấy ảnh predictions")
        
        # TAB 4: Additional Plots
        with tab4:
            st.subheader("Các biểu đồ khác")
            
            # Labels
            labels_path = os.path.join(test_results_path, "labels.jpg")
            labels_correlogram = os.path.join(test_results_path, "labels_correlogram.jpg")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(labels_path):
                    st.image(labels_path, caption="Labels Distribution", use_container_width=True)
                
                # Val batch
                val_batch = os.path.join(test_results_path, "val_batch0_labels.jpg")
                if os.path.exists(val_batch):
                    st.image(val_batch, caption="Validation Batch Labels", use_container_width=True)
            
            with col2:
                if os.path.exists(labels_correlogram):
                    st.image(labels_correlogram, caption="Labels Correlogram", use_container_width=True)
                
                # Val predictions
                val_pred = os.path.join(test_results_path, "val_batch0_pred.jpg")
                if os.path.exists(val_pred):
                    st.image(val_pred, caption="Validation Predictions", use_container_width=True)
            
            # Tìm thêm val batch khác
            other_val_batches = glob.glob(os.path.join(test_results_path, "val_batch*_pred.jpg"))
            if len(other_val_batches) > 1:
                st.markdown("---")
                st.markdown("#### 📸 Các validation batch khác")
                
                cols = st.columns(3)
                for idx, batch_path in enumerate(other_val_batches[1:6]):  # Hiển thị tối đa 5 batch
                    with cols[idx % 3]:
                        st.image(batch_path, caption=os.path.basename(batch_path), use_container_width=True)
        
        # TAB 5: All Files
        with tab5:
            st.subheader("📂 Tất cả file trong thư mục")
            
            # Liệt kê tất cả file
            all_files = []
            for root, dirs, files in os.walk(test_results_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            if all_files:
                st.write(f"Tổng số file: **{len(all_files)}**")
                
                # Phân loại file
                images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                jsons = [f for f in all_files if f.lower().endswith('.json')]
                txts = [f for f in all_files if f.lower().endswith('.txt')]
                others = [f for f in all_files if f not in images + jsons + txts]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🖼️ Images", len(images))
                with col2:
                    st.metric("📄 JSON", len(jsons))
                with col3:
                    st.metric("📝 TXT", len(txts))
                with col4:
                    st.metric("📦 Khác", len(others))
                
                # Hiển thị danh sách
                with st.expander("📋 Danh sách file chi tiết", expanded=False):
                    import pandas as pd
                    
                    file_data = []
                    for f in all_files:
                        file_data.append({
                            'Tên': os.path.basename(f),
                            'Đường dẫn': f,
                            'Kích thước': f"{os.path.getsize(f)/1024:.1f} KB",
                            'Loại': os.path.splitext(f)[1]
                        })
                    
                    df = pd.DataFrame(file_data)
                    st.dataframe(df, use_container_width=True)
                
                # Hiển thị tất cả ảnh
                if images:
                    st.markdown("---")
                    st.markdown("#### 🖼️ Tất cả ảnh")
                    
                    cols = st.columns(3)
                    for idx, img_path in enumerate(images):
                        with cols[idx % 3]:
                            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.warning("⚠️ Thư mục trống")
        
        # Download all
        st.markdown("---")
        st.markdown("### 📥 Download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download JSON nếu có
            json_files = glob.glob(os.path.join(test_results_path, "*.json"))
            if json_files:
                with open(json_files[0], "rb") as f:
                    st.download_button(
                        "📄 Download JSON Results",
                        f,
                        os.path.basename(json_files[0]),
                        "application/json",
                        use_container_width=True
                    )
        
        with col2:
            # Download ZIP toàn bộ
            import zipfile
            import io
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(test_results_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, test_results_path)
                        zip_file.write(file_path, arcname)
            
            st.download_button(
                "📦 Download All (ZIP)",
                zip_buffer.getvalue(),
                f"{os.path.basename(test_results_path)}_results.zip",
                "application/zip",
                use_container_width=True
            )
    
    else:
        st.error(f"❌ Không tìm thấy thư mục: `{test_results_path}`")
        
        st.markdown("### 💡 Hướng dẫn:")
        st.markdown("""
        Sau khi chạy test/validation với YOLOv8, kết quả thường lưu tại:
        
        ```
        run/detect/val/          # Lần chạy đầu tiên
        run/detect/val2/         # Lần chạy thứ 2
        run/detect/test/         # Nếu chạy trên test set
        ```
        
        **Cấu trúc thư mục kết quả:**
        ```
        run/detect/val/
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── PR_curve.png
        ├── F1_curve.png
        ├── P_curve.png
        ├── R_curve.png
        ├── predictions.json
        ├── labels.jpg
        └── [ảnh predictions...]
        ```
        
        Nhập đường dẫn chính xác vào ô trên để xem kết quả!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎯 Traffic Detection - Nhóm 2 | YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)