from tflite_runtime.interpreter import Interpreter
import json
import numpy as np
import os
import cv2


class DetectionModel:
    def __init__(self):
        self.detectionmodel_path = os.path.join(os.getcwd(), "models/detection.tflite")
        self.interpreter_detection = Interpreter(self.detectionmodel_path)
        self.interpreter_detection.allocate_tensors()
        (
            self.input_details,
            self.output_details,
            self.height,
            self.width,
        ) = self.__get_input_details()
        self.boxes_idx, self.scores_idx = 1, 0

    def __get_input_details(self):
        input_details = self.interpreter_detection.get_input_details()
        output_details = self.interpreter_detection.get_output_details()
        print(output_details)
        height = input_details[0]["shape"][1]
        width = input_details[0]["shape"][2]
        return input_details, output_details, height, width

    def detect_face(self, input_data):
        self.interpreter_detection.set_tensor(
            self.input_details[0]["index"], input_data
        )
        self.interpreter_detection.invoke()
        boxes = self.interpreter_detection.get_tensor(
            self.output_details[self.boxes_idx]["index"]
        )[0]
        scores = self.interpreter_detection.get_tensor(
            self.output_details[self.scores_idx]["index"]
        )[0]
        return boxes, scores

    def resise_tensor(self, input_size, batch_size):
        self.interpreter_detection.resize_tensor_input(
            self.input_details[0]["index"], [batch_size, input_size, input_size, 3]
        )
        self.interpreter_detection.resize_tensor_input(
            self.output_details[0]["index"], [batch_size, 10, 4]
        )
        self.interpreter_detection.allocate_tensors()

    def get_bb_coords(self, boxes, i, image_width, image_height):
        ymin = int(max(1, (boxes[i][0] * image_height)))
        xmin = int(max(1, (boxes[i][1] * image_width)))
        ymax = int(min(image_height, (boxes[i][2] * image_height)))
        xmax = int(min(image_width, (boxes[i][3] * image_width)))
        xyxy = [xmin, ymin, xmax, ymax]
        return ymin, xmin, ymax, xmax, xyxy

    def proces_frame_for_model(self, interpreter_detection, frame):
        frame_resized = cv2.resize(
            frame, (interpreter_detection.width, interpreter_detection.height)
        )
        input_data = np.expand_dims(frame_resized, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5
        return input_data


class RecognitionModel:
    def __init__(self, min_dist_threshold):
        self.min_dist_threshold = min_dist_threshold
        self.recognitionmodel_path = os.path.join(
            os.getcwd(), "models/recognition.tflite"
        )
        self.interpreter_recognition = Interpreter(self.recognitionmodel_path)
        self.interpreter_recognition.allocate_tensors()
        self.__load_databases("databases/")

    def __load_databases(self, path):
        self.database_ellie = self.__read_database(path + "databaseEllie.json")
        self.database_marley = self.__read_database(path + "databaseMarley.json")
        self.database_muchu = self.__read_database(path + "databaseMuchu.json")

    def __read_database(self, file):
        data = {}
        with open(file) as json_file:
            data = json.load(json_file)
        return data

    def __verify(self, embedding, database):
        min_dist = float("inf")
        identity = None
        for name, db_enc in database.items():
            dist = np.linalg.norm(embedding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > self.min_dist_threshold:
            return min_dist, "unknown"
        else:
            return min_dist, identity

    def __classify(self, embedding):
        distances = {}
        for database in [
            self.database_ellie,
            self.database_marley,
            self.database_muchu,
        ]:
            dist, identity = self.__verify(embedding, database)
            distances[identity] = dist
        name = "".join(
            [i for i in min(distances, key=distances.get) if not i.isdigit()]
        ).split("_")[0]
        return name

    def __predict_embedding(self, crop_img):
        self.interpreter_recognition.set_tensor(
            self.interpreter_recognition.get_input_details()[0]["index"], crop_img
        )
        self.interpreter_recognition.invoke()
        embedding = self.interpreter_recognition.get_tensor(
            self.interpreter_recognition.get_output_details()[0]["index"]
        )
        return embedding

    def __crop_img_from_bounding_boxes(self, xyxy, image):
        xmin, ymin, xmax, ymax = xyxy
        crop_img = image[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (256, 256))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_img = crop_img.reshape(1, 256, 256, 3)
        crop_img = crop_img.astype("float32") / 255.0
        return crop_img

    def check_dog(self, xyxy, image):
        crop_img = self.__crop_img_from_bounding_boxes(xyxy, image)
        embedding = self.__predict_embedding(crop_img)
        name = self.__classify(embedding)
        return name
