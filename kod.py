from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture("kayit.mp4")

target_class_id = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, verbose=False) 

    person_count_in_frame = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0]) 
            if cls == target_class_id:
                person_count_in_frame += 1 

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                
               
                if model.names[cls] == 'person':
                    display_label = "insan"
                else:
                    display_label = model.names[cls] 
                
                label = f"{display_label} {conf:.2f}" 
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if person_count_in_frame > 0:
        print(f"Tespit Edilen İnsan Sayisi: {person_count_in_frame}")

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()