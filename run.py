from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\punith p\OneDrive\Desktop\runs\detect\3d_printing_training\yolov8s_custom-3\weights\best.pt")

cap = cv2.VideoCapture(r"C:\Users\punith p\Downloads\EPIC3DPRINTERFAILS.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("YOLO", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()