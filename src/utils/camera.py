from pathlib import Path
from typing import Union

import cv2

from utils.log import logger

HAAR_CASCADE_PATH = Path(__file__).parent / "detectors" / "haarcascade_frontalface_alt2.xml"
class Camera:
    def __init__(self, video_stream: Union[int, str]):
        self.video_stream = video_stream

    async def check_camera(self):
        stream = cv2.VideoCapture(self.video_stream)
        logger.debug(f"Stream Fps: {round(stream.get(cv2.CAP_PROP_FPS), 2)}")
        logger.debug(f"Stream Backend: {stream.getBackendName()}")
        logger.debug(f"Stream Size(W*H): {stream.get(cv2.CAP_PROP_FRAME_WIDTH)}*"
                     f"{stream.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        logger.info("Camera Check Start, press ESC to exit")
        while stream.isOpened():
            ret, frame = stream.read()
            if not ret:
                logger.error("Camera is not working")
                break
            cv2.imshow("Camera Checking", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        stream.release()
        cv2.destroyAllWindows()

    async def face_detector(self):
        stream = cv2.VideoCapture(self.video_stream)
        face_classifier = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
        increment_num = 0

        while True:
            ret, img = stream.read()
            if not ret:
                break
            # detect faces using haar cascade detector
            faces = face_classifier.detectMultiScale(img, 1.0485258, 6)

            for (x, y, w, h) in faces:
                increment_num += 1

                # saving the captured face in the <id> folder under static/images/dataset
                # cv2.imwrite(f"{id_path}{os.sep}{increment_num}.jpg", img)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow("Capturing Face", img)
            if cv2.waitKey(1) & 0xFF == 27 or increment_num > 15:
                break
        stream.release()
        cv2.destroyAllWindows()
