from flask import Flask, render_template, Response
from Pretrained_ml_model import ProcessedOP
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture('rtsp://<ip>:<port>/')

def gen_frames(camera):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(ProcessedOP()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
