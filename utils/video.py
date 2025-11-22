import cv2
from ultralytics import YOLO
import streamlit as st

model = YOLO("models/best.pt")

def detect_video(video_path, output_path="output.mp4", conf=0.25, iou=0.45):
    """
    Phát hiện vật thể trong video với progress bar
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
    
    max_class_count = {}
    frame_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect với confidence và iou
        results = model(frame, conf=conf, iou=iou)[0]
        annotated = results.plot()
        
        # Đếm trong frame hiện tại
        current_frame_count = {}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            current_frame_count[class_name] = current_frame_count.get(class_name, 0) + 1
        
        # Update MAX count
        for class_name, count in current_frame_count.items():
            if class_name not in max_class_count:
                max_class_count[class_name] = count
            else:
                max_class_count[class_name] = max(max_class_count[class_name], count)
        
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
    Phát hiện video theo thời gian thực
    """
    cap = cv2.VideoCapture(video_path)
    
    max_class_count = {}
    frame_count = 0
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect với conf và iou
        results = model(frame, conf=conf, iou=iou)[0]
        annotated = results.plot()
        
        # Đếm
        current_frame_count = {}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            current_frame_count[class_name] = current_frame_count.get(class_name, 0) + 1
        
        # Update MAX
        for class_name, count in current_frame_count.items():
            if class_name not in max_class_count:
                max_class_count[class_name] = count
            else:
                max_class_count[class_name] = max(max_class_count[class_name], count)
        
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        
        stats_text = f"Frame: {frame_count}/{total_frames}"
        if max_class_count:
            stats_text += " | MAX: " + ", ".join([f"{k}: {v}" for k, v in max_class_count.items()])
        stats_placeholder.text(stats_text)
        
        if frame_count % 3 == 0:
            continue
    
    cap.release()
    
    return max_class_count


def process_video_with_preview(video_path, output_path="output.mp4", show_preview=True, conf=0.25, iou=0.45):
    """
    Xử lý video với tùy chọn preview
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
    
    max_class_count = {}
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
        
        # Detect với confidence và iou
        results = model(frame, conf=conf, iou=iou)[0]
        annotated = results.plot()
        
        # Đếm
        current_frame_count = {}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            current_frame_count[class_name] = current_frame_count.get(class_name, 0) + 1
        
        # Update MAX
        for class_name, count in current_frame_count.items():
            if class_name not in max_class_count:
                max_class_count[class_name] = count
            else:
                max_class_count[class_name] = max(max_class_count[class_name], count)
        
        out.write(annotated)
        
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"⏳ Đang xử lý: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
            
            if show_preview and frame_count % preview_interval == 0:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(annotated_rgb, caption=f"Frame {frame_count}", use_container_width=True)
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    if show_preview:
        preview_placeholder.empty()
    
    return output_path, max_class_count