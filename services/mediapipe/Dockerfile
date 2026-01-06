FROM ubuntu:20.04

WORKDIR /opt/app

RUN apt-get update -y && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 \
    pip \
    wget

# install python packages
RUN pip install \
    numpy \
    opencv-python \
    mediapipe[all] \
    fastapi \
    uvicorn

# download models
RUN wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O /opt/app/hand_landmarker.task
RUN wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task -O /opt/app/face_landmarker.task
RUN wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -O /opt/app/pose_landmarker_full.task
RUN wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task -O /opt/app/pose_landmarker_heavy.task
RUN wget -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite -O /opt/app/blaze_face_short_range.tflite
RUN wget -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task -O /opt/app/gesture_recognizer.task


# copy files and run
COPY . .
CMD [ "uvicorn", "routs:app", "--host", "0.0.0.0", "--port", "8080"]