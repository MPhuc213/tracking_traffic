import cv2
from ultralytics import YOLO
import streamlit as st

def process_video_with_preview(video_path, output_path="output.mp4", show_preview=True, conf=0.25, iou=0.45, model_path="models/best.pt", use_tracking=True):
    """
    Xử lý video với tracking và preview
    """
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or fps > 120:
        fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # ĐỔI: Dùng 2 biến riêng cho tracking và non-tracking
    unique_ids_tracking = {}  # {class_name: set of track_ids} - cho tracking
    max_count_no_tracking = {}  # {class_name: int} - cho không tracking
    
    frame_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if show_preview:
        preview_placeholder = st.empty()
        preview_interval = max(1, total_frames // 20)
    
    # Kiểm tra tracking
    tracking_available = use_tracking
    if use_tracking:
        try:
            test_frame = cap.read()[1]
            if test_frame is not None:
                _ = model.track(test_frame, conf=conf, iou=iou, persist=True, tracker="bytetrack.yaml")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as e:
            st.warning(f"⚠️ Tracking không khả dụng: {str(e)[:100]}... Dùng detection thông thường")
            tracking_available = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect hoặc Track
        try:
            if tracking_available:
                results = model.track(frame, conf=conf, iou=iou, persist=True, tracker="bytetrack.yaml")[0]
            else:
                results = model(frame, conf=conf, iou=iou)[0]
        except Exception as e:
            if tracking_available:
                st.warning(f"⚠️ Tracking lỗi, chuyển sang detection")
                tracking_available = False
            results = model(frame, conf=conf, iou=iou)[0]
        
        annotated = results.plot()
        
        # Đếm theo mode
        if tracking_available and hasattr(results.boxes, 'id') and results.boxes.id is not None:
            # MODE 1: TRACKING - Đếm unique IDs
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for track_id, cls_id in zip(track_ids, classes):
                class_name = model.names[cls_id]
                
                if class_name not in unique_ids_tracking:
                    unique_ids_tracking[class_name] = set()
                unique_ids_tracking[class_name].add(track_id)
            
            class_count = {k: len(v) for k, v in unique_ids_tracking.items()}
            
        else:
            # MODE 2: NO TRACKING - Đếm MAX trong frame
            current_frame_count = {}
            for box in results.boxes:
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                current_frame_count[class_name] = current_frame_count.get(class_name, 0) + 1
            
            # Update MAX
            for class_name, count in current_frame_count.items():
                if class_name not in max_count_no_tracking:
                    max_count_no_tracking[class_name] = count
                else:
                    max_count_no_tracking[class_name] = max(max_count_no_tracking[class_name], count)
            
            class_count = max_count_no_tracking
        
        out.write(annotated)
        
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            count_str = ", ".join([f"{k}: {v}" for k, v in class_count.items()]) if class_count else "0"
            mode = "Unique" if tracking_available else "MAX"
            status_text.text(f"⏳ {frame_count}/{total_frames} ({progress*100:.1f}%) | {mode}: {count_str}")
            
            if show_preview and frame_count % preview_interval == 0:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(annotated_rgb, caption=f"Frame {frame_count}", use_container_width=True)
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    if show_preview:
        preview_placeholder.empty()
    
    return output_path, class_count


def detect_video_realtime(video_path, conf=0.25, iou=0.45, model_path="models/best.pt", use_tracking=True):
    """
    Phát hiện video realtime
    """
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    # 2 biến riêng
    unique_ids_tracking = {}
    max_count_no_tracking = {}
    
    frame_count = 0
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Kiểm tra tracking
    tracking_available = use_tracking
    if use_tracking:
        try:
            test_frame = cap.read()[1]
            if test_frame is not None:
                _ = model.track(test_frame, conf=conf, iou=iou, persist=True, tracker="bytetrack.yaml")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception as e:
            st.warning(f"⚠️ Tracking không khả dụng, dùng detection")
            tracking_available = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        try:
            if tracking_available:
                results = model.track(frame, conf=conf, iou=iou, persist=True, tracker="bytetrack.yaml")[0]
            else:
                results = model(frame, conf=conf, iou=iou)[0]
        except:
            results = model(frame, conf=conf, iou=iou)[0]
            tracking_available = False
        
        annotated = results.plot()
        
        # Đếm
        if tracking_available and hasattr(results.boxes, 'id') and results.boxes.id is not None:
            # TRACKING MODE
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for track_id, cls_id in zip(track_ids, classes):
                class_name = model.names[cls_id]
                if class_name not in unique_ids_tracking:
                    unique_ids_tracking[class_name] = set()
                unique_ids_tracking[class_name].add(track_id)
            
            class_count = {k: len(v) for k, v in unique_ids_tracking.items()}
            
        else:
            # NO TRACKING MODE
            current_frame_count = {}
            for box in results.boxes:
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                current_frame_count[class_name] = current_frame_count.get(class_name, 0) + 1
            
            for class_name, count in current_frame_count.items():
                if class_name not in max_count_no_tracking:
                    max_count_no_tracking[class_name] = count
                else:
                    max_count_no_tracking[class_name] = max(max_count_no_tracking[class_name], count)
            
            class_count = max_count_no_tracking
        
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        
        mode = "Unique" if tracking_available else "MAX"
        stats_text = f"Frame: {frame_count}/{total_frames} | {mode}"
        if class_count:
            stats_text += " | " + ", ".join([f"{k}: {v}" for k, v in class_count.items()])
        stats_placeholder.text(stats_text)
        
        if frame_count % 3 == 0:
            continue
    
    cap.release()
    
    return class_count