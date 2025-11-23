import streamlit as st
from utils.detect import detect_image
from utils.video import process_video_with_preview, detect_video_realtime
import os
import glob

# FORCE RELOAD - Thêm đoạn này
import importlib
import sys
if 'utils.detect' in sys.modules:
    importlib.reload(sys.modules['utils.detect'])
    from utils.detect import detect_image
if 'utils.video' in sys.modules:
    importlib.reload(sys.modules['utils.video'])
    from utils.video import process_video_with_preview, detect_video_realtime

st.set_page_config(
    page_title="Đếm vật thể - Nhóm 2", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        padding: 1.5rem 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    /* Title styling */
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
    
    /* Main header */
    .main-header {
        text-align: center;
        color: #1e3a8a;
        padding: 1.5rem 0;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        width: 100%;
    }
    
    /* Selectbox styling */
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
                NHÓM 2 _ XỬ LÍ THỊ GIÁC
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tiêu đề nhóm
    st.markdown("""
        <div class='group-title'>
            📚 NHÓM 12<br>
            <span style='font-size: 0.9rem;'>ĐẾM vật thể</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    st.markdown("<p style='color: white; font-weight: bold; font-size: 1.1rem; margin-top: 1rem;'>🧭 CHỨC NĂNG</p>", unsafe_allow_html=True)
    
    option = st.selectbox(
        "Chọn chức năng:",
        ["🖼️ Phát hiện từ ảnh", "🎥 Phát hiện từ video", "📈 Visualize Training Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Thông tin nhóm
    with st.expander("👥 Thành viên nhóm", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        • Trần Kim Minh    (Lead)<br>
        • Nguyễn Minh Phúc  (Thành Viên)<br>
        • Vũ Thị Kim Loan     (Thành Viên)<br>
        • Huỳnh Chí Danh     (Thành Viên)<br>
        • Nguyễn Triệu Thiên Anh (Thành Viên)
        </div>
        """, unsafe_allow_html=True)
    
    # Hướng dẫn
    with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        <b>🖼️ Phát hiện từ ảnh:</b><br>
        Upload một hoặc nhiều ảnh để phát hiện vật thể<br><br>
        
        <b>🎥 Phát hiện từ video:</b><br>
        Upload video để phát hiện và theo dõi vật thể<br><br>
        
        <b>📈 Visualize Training Results:</b><br>
        Xem kết quả training từ thư mục runs/detect/train
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# ẢNH
# -------------------------
if option == "🖼️ Phát hiện từ ảnh":
    st.header("📷 Phát hiện vật thể từ ảnh")
    
    # Thanh cài đặt
    with st.expander("⚙️ Cài đặt phát hiện", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "🎯 Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Ngưỡng độ tin cậy tối thiểu (0-1). Giá trị càng cao, kết quả càng chắc chắn nhưng có thể bỏ sót."
            )
        
        with col2:
            iou_threshold = st.slider(
                "📦 IoU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Ngưỡng IoU cho NMS (Non-Maximum Suppression). Giá trị càng thấp, loại bỏ box trùng lặp càng nhiều."
            )
        
        st.info(f"**Cài đặt hiện tại:** Confidence ≥ {confidence_threshold:.2f} | IoU ≤ {iou_threshold:.2f}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Upload ảnh")
        upload_files = st.file_uploader(
            "Chọn một hoặc nhiều ảnh", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Hỗ trợ định dạng: JPG, JPEG, PNG"
        )
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🖼️ Ảnh {idx + 1}: {upload.name}")
            
            col_left, col_right = st.columns(2)
            
            try:
                file_bytes = upload.read()
                import numpy as np
                import cv2
                
                # Đọc ảnh
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if img is None:
                    st.error(f"❌ Không thể đọc ảnh {upload.name}")
                    continue
                
                with col_left:
                    st.markdown("**Ảnh gốc**")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Detect với confidence và iou
                with st.spinner(f"🔍 Đang phát hiện vật thể trong {upload.name}..."):
                    annotated, class_count = detect_image(img, conf=confidence_threshold, iou=iou_threshold)
                
                with col_right:
                    st.markdown("**Kết quả phát hiện**")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Thống kê
                if class_count:
                    st.success("✅ Phát hiện thành công!")
                    
                    if isinstance(class_count, dict) and class_count:
                        with st.expander("📊 Thống kê phát hiện", expanded=True):
                            stats_col1, stats_col2 = st.columns(2)
                            with stats_col1:
                                for animal, count in class_count.items():
                                    st.metric(label=str(animal).capitalize(), value=count)
                            with stats_col2:
                                st.bar_chart(class_count)
                    elif isinstance(class_count, (int, float)):
                        st.info(f"📊 Tổng số đối tượng phát hiện: {class_count}")
                else:
                    st.warning("⚠️ Không phát hiện được vật thể nào trong ảnh")
                    st.info("💡 Thử giảm Confidence Threshold để phát hiện nhiều hơn")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi xử lý ảnh {upload.name}: {str(e)}")
    else:
        st.info("👆 Vui lòng upload ảnh để bắt đầu phát hiện")

# -------------------------
# VIDEO
# -------------------------
elif option == "🎥 Phát hiện từ video":
    st.header("🎥 Phát hiện vật thể từ video")
    
    # Tùy chọn xử lý
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Upload video")
        upload_files = st.file_uploader(
            "Chọn một hoặc nhiều video", 
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True,
            help="Hỗ trợ định dạng: MP4, AVI, MOV"
        )
    
    with col2:
        st.subheader("⚙️ Cài đặt")
        show_preview = st.checkbox("Hiển thị preview", value=True, help="Hiển thị frame mẫu khi xử lý")
        save_output = st.checkbox("Lưu video", value=True, help="Lưu video để tải xuống")
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🎬 Video {idx + 1}: {upload.name}")
            
            try:
                # Lưu video tạm
                temp_input = f"temp_input_{idx}.mp4"
                with open(temp_input, "wb") as f:
                    f.write(upload.read())
                
                # Hiển thị video gốc
                with st.expander("📹 Xem video gốc", expanded=False):
                    st.video(temp_input)
                
                # Phát hiện
                st.markdown("#### 🔍 Đang xử lý video...")
                
                if save_output:
                    output_path = f"output_{idx}_{upload.name}"
                    output_path, class_count = process_video_with_preview(temp_input, output_path, show_preview)
                else:
                    class_count = detect_video_realtime(temp_input)
                    output_path = None
                
                st.success("✅ Xử lý video thành công!")
                
                # Video kết quả
                if save_output and output_path and os.path.exists(output_path):
                    st.markdown("#### 🎥 Video sau khi phát hiện")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Tải video",
                            data=file,
                            file_name=f"detected_{upload.name}",
                            mime="video/mp4",
                            use_container_width=True
                        )
                
                # Thống kê
                if class_count:
                    if isinstance(class_count, dict) and class_count:
                        with st.expander("📊 Thống kê phát hiện", expanded=True):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("**Số lượng:**")
                                for animal, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
                                    st.metric(label=str(animal).capitalize(), value=count)
                            
                            with col2:
                                st.markdown("**Biểu đồ:**")
                                import pandas as pd
                                df = pd.DataFrame(list(class_count.items()), columns=['Class', 'Count'])
                                st.bar_chart(df.set_index('Class'))
                    elif isinstance(class_count, (int, float)):
                        st.info(f"📊 Tổng: {class_count} đối tượng")
                else:
                    st.warning("⚠️ Không phát hiện được vật thể")
                
                # Cleanup
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                import traceback
                with st.expander("Chi tiết"):
                    st.code(traceback.format_exc())
                
                if os.path.exists(temp_input):
                    os.remove(temp_input)
    else:
        st.info("👆 Vui lòng upload video")

# -------------------------
# VISUALIZE TRAINING RESULTS
# -------------------------
elif option == "📈 Visualize Training Results":
    st.header("📈 Kết quả Training Model")
    
    # Nhập đường dẫn thư mục
    st.markdown("### 📁 Chọn thư mục kết quả training")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        results_path = st.text_input(
            "Đường dẫn thư mục:",
            value="run/detect/train",
            help="Đường dẫn đến thư mục chứa kết quả training (vd: run/detect/train)"
        )
    
    with col2:
        refresh = st.button("🔄 Tải lại", use_container_width=True)
    
    # Kiểm tra thư mục tồn tại
    if os.path.exists(results_path):
        st.success(f"✅ Tìm thấy thư mục: `{results_path}`")
        
        # Tab để tổ chức nội dung
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Confusion Matrix", 
            "📉 Training Curves", 
            "🎯 Predictions", 
            "📂 Tất cả"
        ])
        
        # Tab 1: Confusion Matrix
        with tab1:
            st.subheader("Ma trận nhầm lẫn")
            
            col1, col2 = st.columns(2)
            
            # Confusion matrix thông thường
            with col1:
                cm_path = os.path.join(results_path, "confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                else:
                    st.warning("⚠️ Không tìm thấy confusion_matrix.png")
            
            # Normalized confusion matrix
            with col2:
                cm_norm_path = os.path.join(results_path, "confusion_matrix_normalized.png")
                if os.path.exists(cm_norm_path):
                    st.image(cm_norm_path, caption="Normalized Confusion Matrix", use_container_width=True)
                else:
                    st.warning("⚠️ Không tìm thấy confusion_matrix_normalized.png")
        
        # Tab 2: Training Curves
        with tab2:
            st.subheader("Đường cong Training")
            
            # Results.png - tổng hợp
            results_img = os.path.join(results_path, "results.png")
            if os.path.exists(results_img):
                st.image(results_img, caption="Training Results Overview", use_container_width=True)
                
                # Giải thích
                with st.expander("📖 Giải thích các metrics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        **📊 Metrics:**
                        - **mAP50**: Độ chính xác @ IoU 0.5
                        - **mAP50-95**: Độ chính xác trung bình
                        - **Precision**: Độ chính xác dự đoán
                        - **Recall**: Khả năng phát hiện
                        """)
                    
                    with col2:
                        st.markdown("""
                        **📉 Loss:**
                        - **Box Loss**: Lỗi vị trí bounding box
                        - **Class Loss**: Lỗi phân loại
                        - **DFL Loss**: Distribution Focal Loss
                        """)
                    
                    with col3:
                        st.markdown("""
                        **✅ Model tốt khi:**
                        - Loss giảm dần
                        - mAP tăng và ổn định
                        - Val loss ~ Train loss
                        - Không overfitting
                        """)
            else:
                st.warning("⚠️ Không tìm thấy results.png")
            
            st.markdown("---")
            
            # PR và F1 curves
            col1, col2 = st.columns(2)
            
            with col1:
                pr_path = os.path.join(results_path, "PR_curve.png")
                if os.path.exists(pr_path):
                    st.image(pr_path, caption="Precision-Recall Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có PR_curve.png")
            
            with col2:
                f1_path = os.path.join(results_path, "F1_curve.png")
                if os.path.exists(f1_path):
                    st.image(f1_path, caption="F1 Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có F1_curve.png")
        
        # Tab 3: Predictions
        with tab3:
            st.subheader("Ví dụ dự đoán")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Labels (Ground Truth)")
                labels_path = os.path.join(results_path, "labels.jpg")
                if os.path.exists(labels_path):
                    st.image(labels_path, caption="Labels Distribution", use_container_width=True)
                
                # Train batch
                train_batch = os.path.join(results_path, "train_batch0.jpg")
                if os.path.exists(train_batch):
                    st.image(train_batch, caption="Train Batch Example", use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Predictions")
                
                # Val batch labels
                val_labels = os.path.join(results_path, "val_batch0_labels.jpg")
                if os.path.exists(val_labels):
                    st.image(val_labels, caption="Validation Labels", use_container_width=True)
                
                # Val batch predictions
                val_pred = os.path.join(results_path, "val_batch0_pred.jpg")
                if os.path.exists(val_pred):
                    st.image(val_pred, caption="Validation Predictions", use_container_width=True)
            
            # Tìm thêm các batch khác
            st.markdown("---")
            st.markdown("#### 📸 Các batch khác")
            
            other_batches = glob.glob(os.path.join(results_path, "val_batch*_pred.jpg"))
            if len(other_batches) > 1:
                cols = st.columns(3)
                for idx, batch_path in enumerate(other_batches[1:]):  # Bỏ qua batch0 đã hiển thị
                    with cols[idx % 3]:
                        st.image(batch_path, caption=os.path.basename(batch_path), use_container_width=True)
            else:
                st.info("ℹ️ Không có batch validation khác")
        
        # Tab 4: Tất cả file
        with tab4:
            st.subheader("📂 Tất cả file trong thư mục")
            
            # Lấy tất cả file ảnh
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(glob.glob(os.path.join(results_path, ext)))
            
            if image_files:
                st.write(f"Tìm thấy **{len(image_files)}** file ảnh")
                
                # Hiển thị dạng grid
                cols = st.columns(3)
                for idx, img_path in enumerate(sorted(image_files)):
                    with cols[idx % 3]:
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.warning("⚠️ Không tìm thấy file ảnh nào")
    
    else:
        st.error(f"❌ Không tìm thấy thư mục: `{results_path}`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎯 Object Detection System - Nhóm 12 | Powered by YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)