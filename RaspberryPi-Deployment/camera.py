import cv2
from threading import Thread


class Camera:
    def __init__(self, resolution):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        cv2.destroyAllWindows()
        self.stopped = True

    def show_fps(self, frame_rate_calc, frame):
        cv2.putText(
            frame,
            "FPS: {0:.2f}".format(frame_rate_calc),
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def calc_framerate(self, frequency, start_time_frame):
        end_time_frame = cv2.getTickCount()
        time_frame = (end_time_frame - start_time_frame) / frequency
        return 1 / time_frame

    def show_stream(self, frame):
        cv2.imshow("Dog cam", frame)

    def draw_bounding_boxes_with_name(self, name, frame, ymin, xmin, ymax, xmax):
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        label_size, base_line = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (xmin, label_ymin - label_size[1] - 10),
            (xmin + label_size[0], label_ymin + base_line - 10),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            name,
            (xmin, label_ymin - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
