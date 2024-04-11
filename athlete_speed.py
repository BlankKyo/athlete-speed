import cv2
import numpy as np
from ultralytics import YOLO
from speed_scale import*
from datetime import datetime


    

# opening the file in read mode
my_file = open("coco.txt", "r")

# reading the file
data = my_file.read()

# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list

detection_color =(100, 50, 10)

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480
previous_time = datetime.now()
current_time = 0
previous_xcenter = 0
current_xcenter = 0
vitesse = 0
vector = []
# video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    current_time = datetime.now()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # resize the frame | small frame optimise the run
    #frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    print("a")
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    print("b")
    boxes = detect_params[0].boxes
    for i in range(len(detect_params[0])):
        box = boxes[i]  
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]
        if (clsID == 0):
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_color,
                3,
            )
            current_xcenter = center(int(bb[0]), int(bb[2]))   

    # Display vitesse
    font = cv2.FONT_HERSHEY_COMPLEX
    if (previous_time == 0):
        vitesse = 0;
    else:
        vitesse = speed(previous_xcenter, current_xcenter, previous_time, current_time)
    cv2.putText(
            frame,
            "speed : " + str(round(vitesse, 3)) + " pixel/s",
            (30, 30),
            font,
            1,
            (255, 255, 255),
            2,
        )
    vector.append(float((current_time - previous_time).total_seconds()))
    previous_time = current_time
    previous_xcenter = current_xcenter
    
    # Display the resulting frame
    display("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break
print("temps d'execution pour chaque frame avec ultalytics library")
for i in vector:
    print(i)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
