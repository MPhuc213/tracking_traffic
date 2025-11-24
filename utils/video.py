import cv2
from ultralytics import YOLO
import streamlit as st
import os
import tempfile
from pathlib import Path

# Load model một lần duy nhất (tránh reload nhiều lần)
@st.cache_resource
def load_model(model_path: str):
    if not Path(model_path).exists():
        st.error(f"Không tìm thấy model: {model_path}")
        return None
    return YOLO(model_path)

model = load_model("model_path")  # Thay bằng đường dẫn thực tế hoặc dùng st.file_uploader


def process_video_with_preview(
    video_path: str,
    output_path: str = None,
    show_preview: bool = True,
    conf: float = 0.25,
    iou: float = 0.45,
    use_tracking: bool = True,
    max_preview_frames: int = 20
):
    """
    Xử lý video với YOLO + Tracking, lưu file output và hiển thị preview
    Trả về: (output_path, final_count_dict) hoặc (None, count_dict) nếu lỗi
    """
    if model is None:
        return None, {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không mở được video")
        return None, {}

    # Lấy thông tin video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # --- Tối ưu codec để tương thích browser ---
    fourcc_options = [
        ('avc1', '.mp4'),   # H.264 - tốt nhất cho web
        ('mp4v', '.mp4'),
        ('H264', '.mp4'),
        ('XVID', '.avi'),
    ]

    out = None
    final_output_path = output_path
    for fourcc_code, ext in fourcc_options:
        final_output_path = Path(output_path).with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        out = cv2.VideoWriter(str(final_output_path), fourcc, fps, (w, h))
        if out.isOpened():
            break
    else:
        st.error("Không thể khởi tạo VideoWriter với bất kỳ codec nào")
        cap.release()
        return None, {}

    # --- Kiểm tra tracking có hoạt động không ---
    tracking_available = use_tracking
    tracker_config = "bytetrack.yaml"
    if use_tracking:
        ret, test_frame = cap.read()
        if ret:
            try:
                _ = model.track(test_frame, conf=conf, iou=iou, persist=True, tracker=tracker_config)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset về đầu
            except Exception as e:
                st.warning(f"Tracking không khả dụng ({e}). Chuyển sang detection thường.")
                tracking_available = False
        else:
            tracking_available = False

    # --- Khởi tạo biến đếm ---
    unique_tracker = {}      # Dùng khi có tracking: đếm unique ID
    max_detector = {}        # Dùng khi không tracking: lấy max từng khung hình

    frame_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_placeholder = st.empty() if show_preview else None

    preview_interval = max(1, total_frames // max_preview_frames) if total_frames > 0 else 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Inference
            try:
                if tracking_available:
                    results = model.track(
                        frame,
                        conf=conf,
                        iou=iou,
                        persist=True,
                        tracker=tracker_config,
                        verbose=False
                    )[0]
                else:
                    results = model(frame, conf=conf, iou=iou, verbose=False)[0]
            except Exception as e:
                st.warning(f"Lỗi inference frame {frame_count}: {e}")
                results = model(frame, conf=conf, iou=iou, verbose=False)[0]
                tracking_available = False

            # Vẽ kết quả
            annotated_frame = results.plot()

            # Resize nếu cần (tránh lỗi kích thước)
            if annotated_frame.shape[:2] != (h, w):
                annotated_frame = cv2.resize(annotated_frame, (w, h))

            # Ghi frame vào file output
            out.write(annotated_frame)

            # --- ĐẾM OBJECT ---
            if tracking_available and results.boxes.id is not None:
                ids = results.boxes.id.cpu().numpy().astype(int)
                classes = results.boxes.cls.cpu().numpy().astype(int)
                for obj_id, cls_id in zip(ids, classes):
                    cls_name = model.names[int(cls_id)]
                    unique_tracker.setdefault(cls_name, set()).add(obj_id)
                current_count = {k: len(v) for k, v in unique_tracker.items()}
            else:
                # Không có tracking → lấy số lượng lớn nhất từng thấy
                frame_count_dict = {}
                for box in results.boxes:
                    cls_name = model.names[int(box.cls.item())]
                    frame_count_dict[cls_name] = frame_count_dict.get(cls_name, 0) + 1
                for k, v in frame_count_dict.items():
                    max_detector[k] = max(max_detector.get(k, 0), v)
                current_count = max_detector.copy()

            # --- Cập nhật UI ---
            if total_frames > 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)

                mode = "Unique IDs" if tracking_available else "Max/frame"
                count_str = ", ".join(f"{k}: {v}" for k, v in current_count.items()) or "0"
                status_text.text(
                    f"Đang xử lý... {frame_count}/{total_frames} "
                    f"({progress*100:.1f}%) | {mode} → {count_str}"
                )

            # Preview
            if show_preview and frame_count % preview_interval == 0:
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(
                    rgb_frame,
                    caption=f"Frame {frame_count}/{total_frames}",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý: {e}")
    finally:
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        if show_preview:
            preview_placeholder.empty()

    # --- Kiểm tra file output ---
    if not Path(final_output_path).exists() or Path(final_output_path).stat().st_size < 1024:
        st.error("File output bị lỗi hoặc quá nhỏ!")
        if Path(final_output_path).exists():
            Path(final_output_path).unlink(missing_ok=True)
        return None, current_count

    file_size_mb = Path(final_output_path).stat().st_size / (1024 * 1024)
    st.success(f"Hoàn thành! Đã lưu video ({file_size_mb:.1f} MB)")

    return str(final_output_path), current_count


def detect_video_realtime(
    video_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    use_tracking: bool = True
):
    """
    Xem realtime không lưu file (nhẹ hơn, nhanh hơn)
    """
    if model is None:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không mở được video")
        return {}

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    unique_tracker = {}
    max_detector = {}
    tracking_available = use_tracking

    # Test tracking
    if use_tracking:
        ret, frame = cap.read()
        if ret:
            try:
                _ = model.track(frame, persist=True, tracker="bytetrack.yaml")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except:
                tracking_available = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(frame, conf=conf, iou=iou, persist=True, tracker="bytetrack.yaml", verbose=False)[0] \
                  if tracking_available else \
                  model(frame, conf=conf, iou=iou, verbose=False)[0]

        annotated = results.plot()

        # Đếm
        if tracking_available and results.boxes.id is not None:
            for tid, cls_id in zip(results.boxes.id.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                name = model.names[int(cls_id)]
                unique_tracker.setdefault(name, set()).add(int(tid))
            count_dict = {k: len(v) for k, v in unique_tracker.items()}
        else:
            temp = {}
            for cls_id in results.boxes.cls.cpu().numpy():
                name = model.names[int(cls_id)]
                temp[name] = temp.get(name, 0) + 1
            for k, v in temp.items():
                max_detector[k] = max(max_detector.get(k, 0), v)
            count_dict = max_detector.copy()

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, use_container_width=True)

        mode = "Unique" if tracking_available else "Max"
        stats_placeholder.info(f"Frame {frame_idx} | {mode}: {count_dict}")

    cap.release()
    st.success("Xem realtime hoàn tất!")
    return count_dict