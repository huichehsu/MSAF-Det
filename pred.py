from ultralytics import YOLO
import cv2
import os

# 載入模型
# model = YOLO(r'./ARIS/yolo11n_trim_visdrone/weights/best.pt')
# model = YOLO(r'./ARIS/yolo11n_trim+enhance_suppress/weights/best.pt')  # load a custom model
# model = YOLO(r'./graduation_6000ada_new/yolo11n_trim+TDSA/weights/best.pt')  # load a custom model
# model = YOLO(r'./ARIS/yolo11n_trim+DEFM+FSAM_visdrone/weights/best.pt')  # load a custom model

# model = YOLO(r'./ARIS/yolo11n_trim_uavdt/weights/best.pt')  # load a custom model
model = YOLO(r'./ARIS/yolo11n_trim+DEFM+FSAM_uavdt/weights/best.pt')  # load a custom model


# folder = '/home/hsu/Desktop/study/visdrone/VisDrone2019-DET-val/images'
folder = '/home/hsu/Desktop/study/UAVDT/UAVDT-train/images'

import os
import cv2
import numpy as np # Import numpy for color definitions

## visdrone
colors = [
    (255, 0, 0),    # Blue for pedestrian (class 0)
    (0, 255, 0),    # Green for people (class 1)
    (0, 0, 255),    # Red for bicycle (class 2)
    (255, 255, 0),  # Cyan for car (class 3)
    (255, 0, 255),  # Magenta for van (class 4)
    (0, 255, 255),  # Yellow for truck (class 5)
    (128, 0, 0),    # Dark Blue for tricycle (class 6)
    (0, 128, 0),    # Dark Green for awning-tricycle (class 7)
    (0, 0, 128),    # Dark Red for bus (class 8)
    (128, 128, 0)   # Teal for motor (class 9)
]

class_color_map = {
    "pedestrian": colors[0],
    "people": colors[1],
    "bicycle": colors[2],
    "car": colors[3],
    "van": colors[4],
    "truck": colors[5],
    "tricycle": colors[6],
    "awning-tricycle": colors[7],
    "bus": colors[8],
    "motor": colors[9]
}



## uavdt
# colors = [
#     (255, 255, 0),  # Cyan for car (class 3)
#     (0, 255, 255),  # Yellow for truck (class 5)
#     (255, 0, 255),    # Dark Red for bus (class 8)
# ]

# class_color_map = {
#     "car": colors[0],
#     "truck": colors[1],
#     "bus": colors[2],
# }

for file in os.listdir(folder):
    # Check if the file is an image (you can add more extensions if needed)
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    # 讀取圖片
    img_path = os.path.join(folder, file)
    img = cv2.imread(img_path)

    # Check if image was loaded successfully
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue

    # Create a directory for feature_map if it doesn't exist
    output_dir = 'feature_map'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save a copy of the original image (optional, if you want to save the raw input)
    # cv2.imwrite(os.path.join(output_dir, f'original_{file}'), img)


    # 推論
    # Ensure your model.predict function can handle a numpy array directly
    # If model.predict strictly needs a path, you might need to save a temporary file
    # or check if it accepts a pre-loaded image. Many YOLO versions accept a numpy array.
    # Assuming 'result = model.predict(source=img, ...)' works if source can be a numpy array.
    # If not, stick to 'img_path'.
    # For consistency with the original code, let's use img_path for prediction
    # but draw on the 'img' numpy array.

    result = model.predict(source=img_path, conf=0.5, save=False)[0] # Assuming model.predict is Ultralytics YOLOv8 style

    # 取得預測結果
    boxes = result.boxes  # 檢測框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f'{model.names[cls]} {conf:.2f}'
        
        # 取得該類別的顏色
        box_color = colors[cls % len(colors)] # Use modulo in case cls > number of defined colors

        # 畫框
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Save the image with detections
    # It's better to save each processed image with a unique name rather than overwriting 'image.jpg'
    cv2.imwrite(os.path.join(output_dir, f'detected_{file}'), img)

    # 顯示圖像
    cv2.imshow('ours', img)
    if cv2.waitKey(0) & 0xFF == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
