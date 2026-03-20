from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
weight = r'./Trans/yolo11n_trim***/weights/best.pt'

if __name__ == '__main__':

    model = YOLO(weight)  # load a custom model
    metrics = model.val(data='visdrone.yaml', iou=0.5, conf=0.001, half=False, device=0, save_json=True, name=r'val\vidsrone')
    # Validate the model
    # metrics = model.val(data='ultralytics/datasets/coco.yaml', iou=0.7, conf=0.001, half=False, device=0, save_json=True)
