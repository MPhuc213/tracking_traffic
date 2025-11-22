from ultralytics import YOLO
import cv2

# Load model (chỉ load 1 lần)
model = YOLO("models/best.pt")

def detect_image(image, conf=0.25, iou=0.45):
    """
    Phát hiện vật thể trong ảnh (dùng cho Streamlit, Flask, v.v.)
    
    Args:
        image: numpy array (BGR) từ cv2.imread() hoặc uploaded file
        conf: ngưỡng confidence (0.0 - 1.0)
        iou: ngưỡng IoU cho NMS
    
    Returns:
        annotated_img: ảnh đã vẽ box + label
        class_count: dict đếm số lượng từng class (vd: {'cat': 2, 'dog': 1})
    """
    # QUAN TRỌNG: thêm verbose=False để không in log dài dòng
    results = model(image, conf=conf, iou=iou, verbose=False)[0]
    
    # Dùng results.plot() → siêu đẹp, tự động vẽ box + tên + conf
    annotated = results.plot()

    # Đếm số lượng từng class
    class_count = {}
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            class_count[class_name] = class_count.get(class_name, 0) + 1

    return annotated, class_count