from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  


results = model.train(data="C:\\Users\\Bharathi\\yolov8\\dataset\\config.yaml", epochs=2)  

