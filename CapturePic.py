import cv2
import os


cap = cv2.VideoCapture("traffic_video.mp4")
frame_id = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Capturing Video", frame)

    if frame_id % 7 == 0:
        cv2.imwrite(f"./data/train/frame_{frame_id}.jpg", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()