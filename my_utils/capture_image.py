
import time
import cv2
from my_utils import cvdraw

working_cameraID = [
    105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124
]
camera_info_lists, url_lists = cvdraw.get_cali_data_from_file(working_cameraID)


for key, value in url_lists:

    cap = cv2.VideoCapture(value)
    frame = cap.read()[1]
    cv2.imwrite(key+'.jpg', frame)

    cap.release()
    time.sleep(0.2)
