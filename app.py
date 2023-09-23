from flask import Flask, render_template, Response, request, redirect
from Pretrainedmodelwithhumancount import ProcessedOP
import cv2

app = Flask(__name__, static_url_path="/static")
# camera = cv2.VideoCapture('rtsp://<ip>:<port>/')

def gen_frames(camera):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        frame = camera.get_frames()  # read the camera frame
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(ProcessedOP()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/contact', methods=["POST", "GET"])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(name, email, message)
        return redirect('/')
    else:
        return render_template("contacts.html")

@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
