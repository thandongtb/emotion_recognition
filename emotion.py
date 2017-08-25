# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
from head_pose_estimation import CnnHeadPoseEstimator
import tensorflow as tf
import os
import time

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mountStart, mountEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

sess = tf.Session()  # Launch the graph in a session.
print("[INFO] loading head pose predictor...")

estimator = CnnHeadPoseEstimator(sess)
estimator.print_allocated_variables()
estimator.load_roll_variables(os.path.realpath("head_pose/roll/cnn_cccdd_30k"))
estimator.load_pitch_variables(os.path.realpath("head_pose/pitch/cnn_cccdd_30k"))
estimator.load_yaw_variables(os.path.realpath("head_pose/yaw/cnn_cccdd_30k"))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def drawFace(rects, frame):
    for i, d in enumerate(rects):
        if d.top() > 0 and d.bottom() > 0 and d.left() > 0 and d.right() > 0:
            crop_face = frame[d.top():d.bottom(), d.left():d.right()]
            crop_face = cv2.resize(crop_face, (200, 200))
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 3)
            return crop_face

    return None

def detect_pose(roi_face):
    roll = estimator.return_roll(roi_face)
    pitch = estimator.return_pitch(roi_face)
    yaw = estimator.return_yaw(roi_face)

    print "Estimated [roll, pitch, yaw] ..... [" + str(roll[0, 0, 0]) + "," + str(
        pitch[0, 0, 0]) + "," + str(yaw[0, 0, 0]) + "]"

    if yaw[0, 0, 0] > 10:
        print "Lac sang phai"
    if yaw[0, 0, 0] < -10:
        print "Lac sang trai"

    if roll[0, 0, 0] > 10:
        print "Nghieng sang phai"
    if roll[0, 0, 0] < -10:
        print "Nghieng sang trai"

    if pitch[0, 0, 0] > 8:
        print "Ngua mat len"
    if pitch[0, 0, 0] < -8:
        print "Cui mat xuong"

def show_icon(frame, icon_path):
    icon = cv2.imread(icon_path)
    frame[:icon.shape[0], :icon.shape[1]] = icon

    return frame

def detect_close_eye(rects, frame):
    EYE_AR_THRESH = 0.2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mount = shape[mountStart:mountEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # leftEyeHull = cv2.convexHull(leftEye)
        # rightEyeHull = cv2.convexHull(rightEye)
        # mountHull = cv2.convexHull(mount)
        #
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [mountHull], -1, (0, 0, 255), 1)

        # if leftEAR < EYE_AR_THRESH:
        #     print "LEFT EYE CLOSED"
        #
        # if rightEAR < EYE_AR_THRESH:
        #     print "RIGHT EYE CLOSED"

def main():
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=0).start()
    time.sleep(1)

    # loop over frames from the video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 1)

        detect_close_eye(rects=rects, frame=frame)

        roi_face = drawFace(rects, frame)

        if roi_face != None:
            detect_pose(roi_face)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    main()
