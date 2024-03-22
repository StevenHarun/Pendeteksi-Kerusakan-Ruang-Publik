from flask import Flask, render_template, Response, session
from YOLO_Video import video_detection
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'masihpemula'

def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])

@app.route('/dashboard', methods = ['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

@app.route("/stream", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('stream.html')

@app.route("/report", methods = ['GET', 'POST'])
def report():
    return render_template('report.html')

@app.route('/webapp')
def webapp():
    return Response(generate_frames(path_x=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
