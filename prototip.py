import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
            break
    
    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]
        if class_name == 'person':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 0, 0), 5)
                conf = float(box.conf[0])
                label = f'insan  {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if class_name == 'bottle':
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 100), 10)
                conf = float(box.conf[0])
                label = f'Beypazari  {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)




    cv2.imshow('Simurgh-X Ödev Prototipi', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("end")