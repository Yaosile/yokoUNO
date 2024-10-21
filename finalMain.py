import numpy as np
import myOwnLibrary as myJazz
from numpy import asanyarray as ana
import cv2

cameraWidth = 3264
cameraHeight = 2464

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=cameraWidth,
    capture_height=cameraHeight,
    display_width=cameraWidth,
    display_height=cameraHeight,
    framerate=1,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def cameraCalibration():
    dist = ana([-0.0639733628476694, -0.059022840140777, 0, 0, 0.0238818089164303])
    mtx = ana([
        [1.734239392051136E3,0,1.667798059392088E3],
        [0,1.729637617052701E3,1.195682065165660E3],
        [0,0,1],
    ])

    yu, xu = myJazz.distortionMap(dist, mtx, cameraWidth, cameraHeight)
    output = np.zeros((cameraHeight, cameraWidth, 3))
    window_title = "CSI Camera"

    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                output = frame[yu,xu]
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, output[::4, ::4, :])
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == '__main__':
    cameraCalibration()
    