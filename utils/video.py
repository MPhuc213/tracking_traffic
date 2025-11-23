import cv2
from ultralytics import YOLO
import streamlit as st

model = YOLO("models/best.pt")

def detect_video(video_path, output_path="output.mp4", conf=0.25, iou=0.45):
    """
    Phát hiện vật thể trong video với tracking
    """
    cap = cv2.VideoCapture(video_path)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or fps > 120:
        fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Tracking: Lưu các unique ID đã gặp
    unique_ids = {}  # {class_name: set of track_ids}
    max_class_count = {}
    frame_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # TRACKING: Dùng track() thay vì predict()
        results = model.track(
            frame, 
            conf=conf, 
            iou=iou,
            persist=True,  # Giữ tracking ID giữa các frame
            tracker="bytetrack.yaml"  # Hoặc "botsort.yaml"
        )[0]
        
        annotated = results.plot()
        
        # Đếm unique objects qua tracking ID
        if results.boxes.id is not None:
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for track_id, cls_id in zip(track_ids, classes):
                class_name = model.names[cls_id]
                
                # Lưu unique ID
                if class_name not in unique_ids:
                    unique_ids[class_name] = set()
                unique_ids[class_name].add(track_id)
        
        # Đếm số lượng unique
        max_class_count = {k: len(v) for k, v in unique_ids.items()}
        
        out.write(annotated)
        
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Đã xử lý: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    
    return output_path, max_class_count


def detect_video_realtime(video_path, conf=0.25, iou=0.45):
    """
    Phát hiện video theo thời gian thực với tracking
    """
    cap = cv2.VideoCapture(video_path)
    
    unique_ids = {}
    frame_count = 0
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # TRACKING
        results = model.track(
            frame, 
            conf=conf, 
            iou=iou,
            persist=True,
            tracker="bytetrack.yaml"
        )[0]
        
        annotated = results.plot()
        
        # Đếm unique objects
        if results.boxes.id is not None:
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for track_id, cls_id in zip(track_ids, classes):
                class_name = model.names[cls_id]
                
                if class_name not in unique_ids:
                    unique_ids[class_name] = set()
                unique_ids[class_name].add(track_id)
        
        class_count = {k: len(v) for k, v in unique_ids.items()}
        
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        
        stats_text = f"Frame: {frame_count}/{total_frames}"
        if class_count:
            stats_text += " | Unique: " + ", ".join([f"{k}: {v}" for k, v in class_count.items()])
        stats_placeholder.text(stats_text)
        
        if frame_count % 3 == 0:
            continue
    
    cap.release()
    
    return class_count


def process_video_with_preview(video_path, output_path="output.mp4", show_preview=True, conf=0.25, iou=0.45):
    """
    Xử lý video với tracking và preview
    """
    cap = cv2.VideoCapture(video_path)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or fps > 120:
        fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    unique_ids = {}  # Tracking unique objects
    frame_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if show_preview:
        preview_placeholder = st.empty()
        preview_interval = max(1, total_frames // 20)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # TRACKING với YOLOv8
        results = model.track(
            frame, 
            conf=conf, 
            iou=iou,
            persist=True,
            tracker="bytetrack.yaml"
        )[0]
        
        annotated = results.plot()
        
        # Đếm unique objects qua tracking ID
        if results.boxes.id is not None:
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for track_id, cls_id in zip(track_ids, classes):
                class_name = model.names[cls_id]
                
                if class_name not in unique_ids:
                    unique_ids[class_name] = set()
                unique_ids[class_name].add(track_id)
        
        class_count = {k: len(v) for k, v in unique_ids.items()}
        
        out.write(annotated)
        
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Hiển thị số lượng unique
            count_str = ", ".join([f"{k}: {v}" for k, v in class_count.items()]) if class_count else "0"
            status_text.text(f"⏳ Đang xử lý: {frame_count}/{total_frames} frames ({progress*100:.1f}%) | Unique: {count_str}")
            
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