from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train16/weights/best.pt")

cap = cv2.VideoCapture("traffic_video.mp4")

no_helmet_frame = 0
violation_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break
    results = model(frame)
    motorcycle_detected = False
    helmet_detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # just a person
            if cls == 1:  
                label = f"No Helmet ({conf:.2f})"
                color = (0, 0, 255)
                if conf >0.5:
                    motorcycle_detected = True
           # a helmet
            elif cls == 0:  
                label = f"Helmet ({conf:.2f})"
                color = (0, 255, 0)
                if conf > 0.5:
                    helmet_detected = True
            else:
                continue


            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    no_helmet_detected = motorcycle_detected and not helmet_detected

    if no_helmet_detected:
        no_helmet_frame += 1
    else:
        no_helmet_frame = 0

    if no_helmet_frame == 30:
        violation_count += 1
        no_helmet_frame = 0


    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break
fps = cap.get(cv2.CAP_PROP_FPS)
num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

video_length = (num_frame/fps)/60
cap.release()
cv2.destroyAllWindows()
print(f"Total violations: {violation_count}")
print(f"Duration (minutes): {video_length:.2f}")
print(f"Violation rate: {violation_count/video_length:.2f}")