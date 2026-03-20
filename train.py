from ultralytics import YOLO

model = YOLO(model='ultralytics/cfg/models/Ours/Ours.yaml')
model.train(**{'cfg': 'ultralytics/cfg/default.yaml'},data='visdrone.yaml',device='0', epochs=400, batch=4, project='Trans', name='Visdrone')  # 配置

# model = YOLO("./Trans/yolo11n_wfpn/weights/last.pt")
# model.train(resume=True)

metrics = model.val()

