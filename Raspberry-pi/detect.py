import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from tflite_runtime.interpreter import Interpreter
import json
from gpiozero import Servo

servo1 = Servo(23)
servo2 = Servo(24)

servo1.max()
servo2.min()


class VideoStream:
    def __init__(self, resolution):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',default=0.9)

parser.add_argument('--distance', help='maximum minimum distance between detected face and faces in data',default=0.1)

parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',default='480x360')

parser.add_argument('--view', help='Display video. This reduces FPS',default='True')

parser.add_argument('allowed', help='Allowed dogs',default='Muchu')

parser.add_argument('time-open', help='Time the door will be open',default='10')

args = parser.parse_args()

min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
min_dist_threshold = float(args.distance)
view_stream = args.view 
allowed_dogs = args.allowed.split(',')
time_open = int(args.time-open)



detectionmodel = os.path.join(os.getcwd(),'models/detection.tflite')
recognitionmodel = os.path.join(os.getcwd(),'models/recognition.tflite')


# load recognition model and databases
interpreterRecognition = Interpreter(model_path=recognitionmodel)
interpreterRecognition.allocate_tensors()

def verify(embedding, database, min_dist_threshold):
    min_dist = float('inf')
    identity = None
    for name, db_enc in database.items():
        dist = np.linalg.norm(embedding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name 
    if min_dist > min_dist_threshold:
        return min_dist, 'unknown'        
    else:        
        return min_dist, identity

def readDatabase(file):
    data = {}
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def classify(embedding):
    distances = {}
    for database in [databaseEllie, databaseMarley, databaseMuchu]:
        dist, identity = verify(embedding, database, min_dist_threshold)
        distances[identity] = dist
    name = ''.join([i for i in min(distances, key=distances.get) if not i.isdigit()]).split("_")[0]
    return name

databaseFolder = "databases/"
databaseEllie = readDatabase(databaseFolder + "databaseEllie.json")
databaseMarley = readDatabase(databaseFolder + "databaseMarley.json")
databaseMuchu = readDatabase(databaseFolder + "databaseMuchu.json")

def check_dog(xyxy, image):
    xmin, ymin, xmax, ymax = xyxy
    crop_img = image[ymin:ymax, xmin:xmax]
    crop_img = cv2.resize(crop_img, (256, 256))
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img = crop_img.reshape(1,256,256,3)
    crop_img = crop_img.astype('float32')/255.0

    interpreterRecognition.set_tensor(interpreterRecognition.get_input_details()[0]['index'], crop_img)
    interpreterRecognition.invoke()
    embedding = interpreterRecognition.get_tensor(interpreterRecognition.get_output_details()[0]['index'])
    name = classify(embedding)
    return name

def openDoor():
    servo1.value = -0.53
    servo2.max()

def closeDoor():
    servo1.max()
    servo2.min()

# Load the Tensorflow Lite model.
interpreterDetection = Interpreter(model_path=detectionmodel)
interpreterDetection.allocate_tensors()

# Get model details
input_details = interpreterDetection.get_input_details()
output_details = interpreterDetection.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models

boxes_idx, classes_idx, scores_idx = 1, 3, 0


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH)).start()
time.sleep(1)

name = None
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    if name is not None and name.lower() in allowed_dogs:
        openDoor()
        time.sleep(time_open)
        name = None

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreterDetection.set_tensor(input_details[0]['index'],input_data)
    interpreterDetection.invoke()

    # Retrieve detection results
    boxes = interpreterDetection.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreterDetection.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreterDetection.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects










    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            xyxy = [xmin, ymin, xmax, ymax]
            name = check_dog(xyxy, frame)

            if view_stream:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                label = '%s: %d%%' % (name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # Draw framerate in corner of frame
    if view_stream:
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

    if name is None or name.lower() not in allowed_dogs:
        closeDoor()
     
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    print(frame_rate_calc)


    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()


