import numpy as np
import myOwnLibrary as myJazz
from numpy import asanyarray as ana
from PIL import Image
import cv2

cameraWidth = 3264
cameraHeight = 2464

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=cameraWidth,
    capture_height=cameraHeight,
    display_width=cameraWidth,
    display_height=cameraHeight,
    framerate=21,
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

"""
1219, 616
1072, 1851
2276, 1851
2115, 609
"""
dist = ana([-0.0639733628476694, -0.059022840140777, 0, 0, 0.0238818089164303])
mtx = ana([
    [1.734239392051136E3,0,1.667798059392088E3],
    [0,1.729637617052701E3,1.195682065165660E3],
    [0,0,1],
])
src = [
    [1219, 616],
    [1072, 1851],
    [2276, 1851],
    [2115, 609],
]

boardSize = (517*2, 605*2)
def cameraCalibration():

    yu, xu = myJazz.distortionMap(dist, mtx, cameraWidth, cameraHeight)
    yw, xw = myJazz.unwarpMap(src, *boardSize, cameraHeight, cameraHeight)
    yuw, xuw = myJazz.getFinalTransform(yw,xw,yu,xu)

    output = np.zeros((*boardSize, 3))
    window_title = "CSI Camera"

    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                output = frame[yuw,xuw]
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title,output[::2, ::2, :])
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w'):
                    Image.fromarray(output.astype(np.uint8)).save('Images/Screenshot.png')
                    print('Kachow')
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == '__main__':
    cameraCalibration()
    