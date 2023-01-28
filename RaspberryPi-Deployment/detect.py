import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
from tflite_runtime.interpreter import Interpreter
import json
from gpiozero import Servo



# ----- parser arguments for the script -----#
# -- threshold: minimum confidence threshold for displaying detected objects (detecting face)
# -- distance: maximum minimum distance between detected face and faces in data (recognizing face)
# -- resolution: resolution shown on desktop if view enabled
# -- view: Display video. This reduces speed
# -- allowed: Allowed dogs (comma separated) example: Muchu,Marley,Ellie
# -- time-open: Time the door will be open (seconds) example: 10 recommend more than 10 for dog to enter    

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',default=0.9)
parser.add_argument('--distance', help='maximum minimum distance between detected face and faces in data',default=0.1)
parser.add_argument('--resolution', help='resolution shown on desktop if view enabled',default='576x432')
parser.add_argument('--view', help='Display video.',action='store_true')
parser.add_argument('--allowed', help='Allowed dogs',default='Muchu,Marley,Ellie')
parser.add_argument('--opentime', help='Time the door will be open',default='10')

args = parser.parse_args()
min_conf_threshold = float(args.threshold)
res_width, res_height = args.resolution.split('x')
image_width, image_height = int(res_width), int(res_height)
min_dist_threshold = float(args.distance)
view_stream = args.view 
allowed_dogs = args.allowed.split(',')
time_open = int(args.opentime)

#lowercase allowed dogs
allowed_dogs = [x.lower() for x in allowed_dogs]

#----- Loading TFLite models and allocating tensors. -----#
detectionmodel = os.path.join(os.getcwd(),'models/detection.tflite')
recognitionmodel = os.path.join(os.getcwd(),'models/recognition.tflite')


# load recognition model 
interpreter_recognition = Interpreter(model_path=recognitionmodel)
interpreter_recognition.allocate_tensors()

# load detection model
interpreter_detection = Interpreter(model_path=detectionmodel)
interpreter_detection.allocate_tensors()

# Get detection model details
input_details = interpreter_detection.get_input_details()
output_details = interpreter_detection.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

boxes_idx, classes_idx, scores_idx = 1, 3, 0


#----- Load databases used for recognition -----#

def read_database(file):
    data = {}
    with open(file) as json_file:
        data = json.load(json_file)
    return data

database_folder = "databases/"
database_ellie = read_database(database_folder + "databaseEllie.json")
database_marley = read_database(database_folder + "databaseMarley.json")
database_muchu = read_database(database_folder + "databaseMuchu.json")


# ----- Servos ----- #

# Initialize servo pins and set them to the initial position
servo1 = Servo(23)
servo2 = Servo(24)

servo1.max()
servo2.min()

# servo functions

def open_door():
    servo1.value = -0.53
    servo2.max()

def close_door():
    servo1.max()
    servo2.min()

#----- Recognition functions -----#

# Verify if the detected face is close enough to the faces in the database
# if the distance is less than the threshold, the name of the dog is returned
# if the distance is greater than the threshold, the name "unknown" is returned
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

# classify the detected face 
# the name of the dog with the smallest distance is returned 
def classify(embedding):
    distances = {}
    for database in [database_ellie, database_marley, database_muchu]:
        dist, identity = verify(embedding, database, min_dist_threshold)
        distances[identity] = dist
    name = ''.join([i for i in min(distances, key=distances.get) if not i.isdigit()]).split("_")[0]
    return name

#process the detected face to a image that can be used by the recognition model
def check_dog(xyxy, image):
    xmin, ymin, xmax, ymax = xyxy
    crop_img = image[ymin:ymax, xmin:xmax]
    crop_img = cv2.resize(crop_img, (256, 256))
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img = crop_img.reshape(1,256,256,3)
    crop_img = crop_img.astype('float32')/255.0

    interpreter_recognition.set_tensor(interpreter_recognition.get_input_details()[0]['index'], crop_img)
    interpreter_recognition.invoke()
    embedding = interpreter_recognition.get_tensor(interpreter_recognition.get_output_details()[0]['index'])
    name = classify(embedding)
    return name


# ----- Camera ----- #

# camera class to handle streaming of video from raspberry pi camera 
class Camera:
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





# ----- Main ----- #

# Initialize frame rate calculation
if view_stream:
    frame_rate_calc = 1
    frequency = cv2.getTickFrequency()

# Initialize camera stream
camera = Camera(resolution=(image_width,image_height)).start()
time.sleep(1)

#initialize variables
name = None


while True:
    # Check if the door should be opened by looking at the predicted name of the last frame if any
    if name is not None and name.lower() in allowed_dogs:
        open_door()
        time.sleep(time_open)
        name = None

    if view_stream:
        # Start timer (for calculating frame rate)
        start_time_frame = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = camera.read()

    # process frame for detection model
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127


    # Perform the detection 
    interpreter_detection.set_tensor(input_details[0]['index'],input_data)
    interpreter_detection.invoke()

    # results
    boxes = interpreter_detection.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter_detection.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter_detection.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over detections 
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            ymin = int(max(1,(boxes[i][0] * image_height)))
            xmin = int(max(1,(boxes[i][1] * image_width)))
            ymax = int(min(image_height,(boxes[i][2] * image_height)))
            xmax = int(min(image_width,(boxes[i][3] * image_width)))
            xyxy = [xmin, ymin, xmax, ymax]
            name = check_dog(xyxy, frame)

            if view_stream:
                #draw bounding box with detection confidence and predicted name
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                label = '%s: %d%%' % (name, int(scores[i]*100)) 
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                label_ymin = max(ymin, label_size[1] + 10) 
                cv2.rectangle(frame, (xmin, label_ymin-label_size[1]-10), (xmin+label_size[0], label_ymin+base_line-10), (255, 255, 255), cv2.FILLED) 
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 

    # Draw framerate in corner of frame
    if view_stream:
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Dog cam', frame)
        # Calculate framerate to display on next frame
        end_time_frame = cv2.getTickCount()
        time_frame = (end_time_frame-start_time_frame)/frequency
        frame_rate_calc= 1/time_frame

    #close the door if the dog is not in the allowed list or if no dog is detected
    if name is None or name.lower() not in allowed_dogs:
        close_door()
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
camera.stop()


