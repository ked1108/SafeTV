#This is a trained model on custom data sets of HUMANS , KNIFE , RIFLE , PISTOL , (These are the threats to common people) 
from turbojpeg import TurboJPEG
from ultralytics import YOLO
import cv2
import cvzone
import numpy
import math

names = {
    0:'Knife',
    1:'Person',
    2:'Person-',
    3:'knife',
    4:'person',
    5:'pistol',
    6:'rifle'
}

jpeg = TurboJPEG()
model = YOLO("safetv.pt")

#mask = cv2.imread("/home/parougv/sih_project/Untitled.png")

class ProcessedOP(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def get_frames(self):
        success , img = self.video.read()
        #img_region = cv2.bitwise_and(img , mask)
        results = model(img , stream = True)
        for r in results :
            boxes = r.boxes
            for box in boxes :
                #bonding box
                x1, y1 , x2 , y2 = box.xyxy[0]
                x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
                print(x1 , y1 , x2 , y2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(255 , 0 , 255),1)
                #w , h = x2 - x1 , y2 - y1 
                #cvzone.cornerRect(img(x1 , y1 , w , h))

                #Confindence
                conf = (math.ceil(box.conf[0]*100))/100

                #class Name :-
                cls = int(box.cls[0])
                checker = names[cls]
                cvzone.putTextRect(img,f'{checker} {conf}',(max(0,x1) , max(35,y1)) , scale = 0.6 , thickness = 1 , offset = 3)
                #if (checker == "person" or checker == "thief" or checker == "threat" or checker == "liscence"  or checker == "train" or checker=="gun" or checker=="rifle" or checker=="knife" or checker=="human") and conf >= 0.3 :
                #   if checker != "liscence plate" :
                #      cvzone.putTextRect(img , f'{checker} {conf}',(max(0,x1) , max(35 ,y1)),scale = 0.6 ,thickness = 1,offset = 3)
                # else:
                        #read text from the number plate and display it on the box 
                    #    cvzone.putTextRect(img , f'{checker} {conf}',(max(0,x1)  , max(35,y1)),scale=0.6 , thickness = 1,offset = 3)


        return jpeg.encode(img)










