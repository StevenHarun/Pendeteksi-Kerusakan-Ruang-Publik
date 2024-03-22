from ultralytics import YOLO
from playsound import playsound
import cv2
import math


model=YOLO("YOLO-Weights/best.pt")
classNames = ['Accident', 'Graffiti', 'Pothole']
sound_file = "static/notifikasi.mp3"
notifikasi_kecelakaan = "static/accident.mp3"
notifikasi_jalanlubang = "static/pothole.mp3"
notifikasi_grafiti = "static/graffiti.mp3"

def video_detection(path_x):
    video_capture = path_x
    
    cap=cv2.VideoCapture(video_capture)

    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'Accident':
                    color=(0, 204, 255)
                elif class_name == "Graffiti":
                    color = (222, 82, 175)
                elif class_name == "Pothole":
                    color = (0, 149, 255)
                else:
                    color = (85,45,255)
                if conf>0.5:
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                    cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA) 
                    if class_name == 'Accident':
                        playsound(notifikasi_kecelakaan, True)
                        
                    elif class_name == 'Graffiti':
                        playsound(notifikasi_grafiti, True)
                        
                    elif class_name == 'Pothole':
                        playsound(notifikasi_jalanlubang, True)

                    else : 
                        pass

        yield img
cv2.destroyAllWindows()

if __name__ == "__main__":
    video_detection(0)
