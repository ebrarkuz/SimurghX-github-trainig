import cv2
from ultralytics import YOLO

# --- MODELİ YÜKLE ---
model = YOLO("yolov8n.pt")

# --- VİDEOYU AÇ ---
cap = cv2.VideoCapture("input.mp4")

# video bilgilerini al
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# çıktı videosu oluştur
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# --- VİDEOYU KARE KARE İŞLE ---
while True:
    ret, frame = cap.read()        # videodan bir kare al
    if not ret:                    # kare yoksa video bitmiş
        break

    results = model(frame)         # kareyi modele ver

    for r in results:              # sonuçları gez
        for box in r.boxes:        # bulunan kutuları gez

            # kutu koordinatları
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls_id = int(box.cls[0])         # hangi sınıf? (0 = insan)
            score = float(box.conf[0])       # güven skoru

            # sadece insan ve güven > 0.4 ise
            if cls_id == 0 and score > 0.4:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # kutu çiz
                cv2.putText(
                    frame,
                    f"Person {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    out.write(frame)  # işlenen kareyi çıktı videosuna ekle

cap.release()
out.release()
cv2.destroyAllWindows()
