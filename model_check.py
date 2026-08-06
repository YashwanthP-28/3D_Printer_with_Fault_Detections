from ultralytics import YOLO

model = YOLO(r"C:\Users\punith p\OneDrive\Desktop\runs\detect\3d_printing_training\yolov8s_custom-3\weights\best.pt")

print(model.names)
