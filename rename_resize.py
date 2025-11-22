import os
import cv2
from tqdm import tqdm
# =========================
# CONFIG
# =========================
input_folder = "input_img"       # thư mục chứa ảnh gốc
output_folder = "dataset"     # thư mục lưu ảnh sau khi rename + resize
custom_name = "car"            # tên custom
target_size = 640                   # kích thước resize: 640x640

# =========================
# HÀM RESIZE GIỮ TỈ LỆ
# =========================
def resize_square(image, size=640):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Tạo canvas vuông
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    canvas = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT
    )

    return canvas


# =========================
# MAIN
# =========================
def process_images():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lấy danh sách ảnh
    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    counter = 0

    # Thanh tiến trình
    for file in tqdm(files, desc="Processing images", unit="img"):

        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize nếu cần
        h, w = img.shape[:2]
        if h > target_size or w > target_size:
            img = resize_square(img, target_size)

        # Tạo tên file mới
        if counter == 0:
            new_name = f"{custom_name}.jpg"
        else:
            new_name = f"{custom_name}_{counter}.jpg"

        out_path = os.path.join(output_folder, new_name)
        cv2.imwrite(out_path, img)

        # XÓA ảnh gốc
        os.remove(img_path)

        counter += 1


if __name__ == "__main__":
    process_images()