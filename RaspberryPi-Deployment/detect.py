import argparse
import cv2
import numpy as np
from camera import Camera
from models import DetectionModel, RecognitionModel

from gpiozero import Servo


def initialize_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        help="Minimum confidence threshold for displaying detected objects",
        default=0.9,
    )
    parser.add_argument(
        "--distance",
        help="maximum minimum distance between detected face and faces in data",
        default=0.1,
    )
    parser.add_argument(
        "--resolution",
        help="resolution shown on desktop if view enabled",
        default="1280x720",
    )
    parser.add_argument("--view", help="Display video.", action="store_true")
    parser.add_argument("--allowed", help="Allowed dogs",
                        default="Muchu,Marley,Ellie")
    parser.add_argument(
        "--opentime", help="Time the door will be open", default="10")
    return parser


def read_args_parser(parser):
    args = parser.parse_args()
    min_conf_threshold = float(args.threshold)
    res_width, res_height = args.resolution.split("x")
    image_width, image_height = int(res_width), int(res_height)
    min_dist_threshold = float(args.distance)
    view_stream = True  # args.view
    time_open = int(args.opentime)
    allowed_dogs = [x.lower() for x in args.allowed.split(",")]
    return (
        min_conf_threshold,
        image_width,
        image_height,
        min_dist_threshold,
        view_stream,
        time_open,
        allowed_dogs,
    )


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


def main():
    if view_stream:
        frame_rate_calc = 1
        frequency = cv2.getTickFrequency()
    name = None
    frame_rate_calcs = []

    camera = Camera(resolution=(image_width, image_height)).start()
    interpreter_recognition = RecognitionModel(min_dist_threshold)
    interpreter_detection = DetectionModel()
    frame_i = 0
    while True:
        try:
            frame_i += 1
            if name is not None and name.lower() in allowed_dogs:
                open_door()
                time.sleep(time_open)
                name = None

            if view_stream:
                start_time_frame = cv2.getTickCount()

            frame = camera.read()
            if frame_i == 25:
                frame_i = 0
                input_data = interpreter_detection.proces_frame_for_model(
                    interpreter_detection, frame
                )
                boxes, scores = interpreter_detection.detect_face(input_data)

                for i in range(10):
                    if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                        (
                            ymin,
                            xmin,
                            ymax,
                            xmax,
                            xyxy,
                        ) = interpreter_detection.get_bb_coords(
                            boxes, i, image_width, image_height
                        )
                        name = interpreter_recognition.check_dog(xyxy, frame)
                        print(name)
                        if view_stream:
                            camera.draw_bounding_boxes_with_name(
                                name, frame, ymin, xmin, ymax, xmax
                            )
                frame_rate_calc = np.average(frame_rate_calcs)
                frame_rate_calcs = []

            if view_stream:
                camera.show_stream(frame)
                camera.show_fps(frame_rate_calc, frame)
                frame_rate_calcs.append(
                    camera.calc_framerate(frequency, start_time_frame)
                )

            if name is None or name.lower() not in allowed_dogs:
                close_door()

            if cv2.waitKey(1) == ord("q"):
                break

        except KeyboardInterrupt:
            break

        except Exception as e:
            print(e)
            break

    camera.stop()


if __name__ == "__main__":
    parser = initialize_parser()
    (
        min_conf_threshold,
        image_width,
        image_height,
        min_dist_threshold,
        view_stream,
        time_open,
        allowed_dogs,
    ) = read_args_parser(parser)
    main()
