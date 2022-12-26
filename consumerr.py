from flask import Flask, Response, render_template
from kafka import KafkaConsumer
import numpy as np
import cv2

# Fire up the Kafka Consumer
topic = "kafka-video-topic"
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['localhost:9092'],
    # value_deserializer=lambda x: loads(x.decode('utf-8'))
)

# Set the consumer in a Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/read_video', methods=['GET'])
def read_video():
    """
    This is the heart of our video display. Notice we set the mimetype to 
    multipart/x-mixed-replace. This tells Flask to replace any old images with 
    new values streaming through the pipeline.
    """
    return Response(
        get_video_stream(), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def detect_objects(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the Haar cascade classifier
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    # Detect objects in the frame
    objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw a bounding box around the detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

def get_video_stream():
    for msg in consumer:
        # Decode the message value as a JPEG image
        img = cv2.imdecode(np.frombuffer(msg.value, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Perform object detection on the frame
        frame = detect_objects(img)
        
        # Encode the frame as a JPEG image and yield it
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

