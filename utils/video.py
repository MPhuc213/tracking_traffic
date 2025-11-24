import cv2
from ultralytics import YOLO
import streamlit as st

def detect_video_realtime(video_path, conf=0.25, iou=0.45,
                          model_path="models/best.pt", use_tracking=True):
    """
    Xử lý video realtime hiển thị trực tiếp lên Streamlit
    Không ghi file, không preview nặng – tối ưu tốc độ.
    """
    st.title("🔍 Realtime Video Detection")

    # Load model
    model = YOLO(model_path)

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Không mở được video!")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Placeholder hiển thị
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    # Đếm object
    unique_ids_tracking = {}
    max_count_no_tracking = {}

    # Kiểm tra tracking
    tracking_available = use_tracking
    if use_tracking:
        try:
            ok, test_frame = cap.read()
            if ok:
                _ = model.track(test_frame, conf=conf, iou=iou,
                                persist=True, tracker="bytetrack.yaml")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except:
            tracking_available = False

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 Giảm kích thước nếu video quá lớn (tăng tốc)
        h, w = frame.shape[:2]
        if max(h, w) > 1280:  
            frame = cv2.resize(frame, (w // 2, h // 2))

        # Run model
        try:
            if tracking_available:
                results = model.track(frame, conf=conf, iou=iou,
                                      persist=True, tracker="bytetrack.yaml")[0]
            else:
                results = model(frame, conf=conf, iou=iou)[0]
        except:
            results = model(frame, conf=conf, iou=iou)[0]
            tracking_available = False

        annotated = results.plot()

        # Đếm object
        if tracking_available and hasattr(results.boxes, "id") and results.boxes.id is not None:
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for tid, cid in zip(track_ids, cls_ids):
                name = model.names[cid]
                if name not in unique_ids_tracking:
                    unique_ids_tracking[name] = set()
                unique_ids_tracking[name].add(tid)

            class_count = {k: len(v) for k, v in unique_ids_tracking.items()}

        else:
            current_frame_count = {}
            for box in results.boxes:
                cid = int(box.cls.item())
                name = model.names[cid]
                current_frame_count[name] = current_frame_count.get(name, 0) + 1

            for name, cnt in current_frame_count.items():
                if name not in max_count_no_tracking:
                    max_count_no_tracking[name] = cnt
                else:
                    max_count_no_tracking[name] = max(max_count_no_tracking[name], cnt)

            class_count = max_count_no_tracking

        # Chuyển màu và hiển thị
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, use_container_width=True)

        # Hiện thông tin
        mode = "Unique Tracking" if tracking_available else "MAX Detection"
        stats_placeholder.text(
            f"Frame {frame_count}/{total_frames} | {mode} | " +
            ", ".join([f"{k}: {v}" for k, v in class_count.items()])
        )

        # Giảm tần suất refresh để tăng FPS
        if frame_count % 3 == 0:
            continue

    cap.release()
    return class_count
