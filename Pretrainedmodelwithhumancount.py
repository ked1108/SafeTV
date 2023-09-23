from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import * 

names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

#cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture(0) #for video detection :-

# cap.set(3,1280)
# cap.set(4,720)

model = YOLO("yolov8n.pt")

#mask = cv2.imread("/home/parougv/sih_project/Untitled.png")
class ProcessedOP(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 1280)
        self.video.set(4, 720)

    def get_frames(self):
        tracker = Sort(max_age=20 , min_hits=3 , iou_threshold=0.3)
        success , img = self.video.read()
        #img_region = cv2.bitwise_and(img , mask)
        results = model(img , stream = True)

        detections = np.empty((0,5))

        for r in results :
            boxes = r.boxes
            for box in boxes :
                #bonding box
                x1, y1 , x2 , y2 = box.xyxy[0]
                x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
                print(x1 , y1 , x2 , y2)
                #cv2.rectangle(img,(x1,y1),(x2,y2),(255 , 0 , 255),)
                #w , h = x2 - x1 , y2 - y1 
                #cvzone.cornerRect(img(x1 , y1 , w , h))

                #Confindence
                conf = (math.ceil(box.conf[0]*100))/100

                #class Name :-
                cls = int(box.cls[0])
                checker = names[cls]
                #cvzone.putTextRect(img,f'{checker} {conf}',(max(0,x1) , max(35,y1)),scale = 0.6 , thickness = 1 , offset = 3)
                currentarray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentarray))
                #if (checker == "person" or checker == "thief" or checker == "threat" or checker == "liscence"  or checker == "train") and conf >= 0.3 :
                    #if checker != "liscence plate" :
                        #cvzone.putTextRect(img , f'{checker} {conf}',(max(0,x1) , max(35 ,y1)),scale = 0.6 ,thickness = 1,offset = 3)
                    #else:
                        #read text from the number plate and display it on the box 
                #      cvzone.putTextRect(img , f'{checker} {conf}',(max(0,x1)  , max(35,y1)),scale=0.6 , thickness = 1,offset = 3)
        
        resultstracker = tracker.update(detections)
        for result in resultstracker :
            if checker == "person" or checker == "Person" :
                x1 , y1 , x2 , y2 , id = result
                x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
                print(result)
                w , h = x2 - x1 , y2 - y1
                cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (255 , 0 , 255))
                #cvzone.cornerRect(img , (x1 , y1 , w , h) , l = 9 , rt = 2 , colorR = (255,0,0))
                cvzone.putTextRect(img,f'{id} {checker} {conf}',(max(0,x1) , max(35,y1)),scale = 0.8 , thickness = 1 , offset =3)
            else:
                x1 , y1 , x2 , y2 , id = result
                x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
                print(result)
                w , h = x2 - x1 , y2 - y1
                cv2.rectangle(img , (x1 , y1) , (x2 , y2) , (255 , 0 , 255))
                #cvzone.cornerRect(img , (x1 , y1 , w , h) , l = 9 , rt = 2 , colorR = (255,0,0))
                cvzone.putTextRect(img,f'{checker} {conf}',(max(0,x1) , max(35,y1)),scale = 0.8 , thickness = 1 , offset =3)            

        # ret, jpeg = cv2.imencode('.jpg', img)
        return cv2.imencode('.jpg', img)[1].tobytes()







