import io
import cv2
import mediapipe as mp
import numpy as np

from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException

from mediapipe.tasks import python as python_utils
from mediapipe.tasks.python import vision

# CONSTANTS --------------------------------------------------------

#todo: move to environment variables and make accessible in docker
RUNNING_MODE = vision.RunningMode.IMAGE

NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5

NUM_FACES = 1
MIN_FACE_DETECTION_CONFIDENCE = 0.5
OUTPUT_FACE_BLEND_SHAPES = True
OUTPUT_FACIAL_TRANSFORMATION_MATRIXES = True

NUM_POSES = 1
MIN_POSE_DETECTION_CONFIDENCE = 0.5
OUTPUT_POSE_SEGMENTATION_MASKS = True
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
POSE_USE_HEAVY_MODEL = True


# HELPERS --------------------------------------------------------

async def file_to_mediapipe_image(file: Union[UploadFile, bytes]):
    
    # load image from buffer
    if isinstance(file, UploadFile):
        contents = await file.read()
    elif isinstance(file, bytes):
        contents = file
    else:
        raise HTTPException(status_code=400, detail="Invalid file format!")
    
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = mp.Image(mp.ImageFormat.SRGB, image)
    return image

# PARSE KEYPOINTS --------------------------------------------------------

def parse_hand_keypoints_readable(keypoints):
    return {
        "wrist": keypoints[0],
        "thumb": keypoints[1:5],
        "index": keypoints[5:9],
        "middle": keypoints[9:13],
        "ring": keypoints[13:17],
        "pinky": keypoints[17:21],
    }

def parse_pose_keypoints_readable(keypoints):
    return {
        "head": {
            "nose": keypoints[0],
            "leftEyeInner": keypoints[1],
            "leftEye": keypoints[2],
            "leftEyeOuter": keypoints[3],
            "rightEyeInner": keypoints[4],
            "rightEye": keypoints[5],
            "rightEyeOuter": keypoints[6],
            "leftEar": keypoints[7],
            "rightEar": keypoints[8],
            "mouthLeft": keypoints[9],
            "mouthRight": keypoints[10]
        },
        "rightArm": {
            "shoulder": keypoints[12],
            "elbow": keypoints[14],
            "wrist": keypoints[16],
            "palmPinky": keypoints[18],
            "palmIndex": keypoints[20],
            "thumb": keypoints[22],
        },
        "leftArm": {
            "shoulder": keypoints[11],
            "elbow": keypoints[13],
            "wrist": keypoints[15],
            "palmPinky": keypoints[17],
            "palmIndex": keypoints[19],
            "thumb": keypoints[21],
        },
        "rightLeg": {
            "hip": keypoints[24],
            "knee": keypoints[26],
            "ankle": keypoints[28],
            "heel": keypoints[30],
            "toes": keypoints[32],
        },
        "leftLeg": {
            "hip": keypoints[23],
            "knee": keypoints[25],
            "ankle": keypoints[27],
            "heel": keypoints[29],
            "toes": keypoints[31],
        }
    }

# GLOBAL OBJECTS --------------------------------------------------------

app = FastAPI()

hand_tracker = None
hand_gesture = None
face_tracker = None
face_detector = None
pose_tracker = None

# STARTUP --------------------------------------------------------

@app.on_event("startup")
async def initialize_hand_tracker():
    global hand_tracker
    hand_tracker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python_utils.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode = RUNNING_MODE,
            num_hands = NUM_HANDS,
            min_hand_detection_confidence = MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence = MIN_HAND_PRESENCE_CONFIDENCE
        )
    )
    
    global hand_gesture
    hand_gesture = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=python_utils.BaseOptions(model_asset_path="gesture_recognizer.task"),
            running_mode = RUNNING_MODE,
            min_hand_detection_confidence = MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence = MIN_HAND_PRESENCE_CONFIDENCE
        )
    )

@app.on_event("startup")
async def initialize_face_tracker():
    global face_tracker
    face_tracker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python_utils.BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode = RUNNING_MODE,
            num_faces = NUM_FACES,
            min_face_detection_confidence = MIN_FACE_DETECTION_CONFIDENCE,
            output_face_blendshapes = OUTPUT_FACE_BLEND_SHAPES,
            output_facial_transformation_matrixes = OUTPUT_FACIAL_TRANSFORMATION_MATRIXES,
        )
    )
    
    global face_detector
    face_detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=python_utils.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
        )
    )

@app.on_event("startup")
async def initialize_pose_tracker():
    global pose_tracker
    
    if POSE_USE_HEAVY_MODEL:
        pose_tracker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python_utils.BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
                running_mode = RUNNING_MODE,
                num_poses = NUM_POSES,
                min_pose_detection_confidence = MIN_POSE_DETECTION_CONFIDENCE,
                min_pose_presence_confidence = MIN_POSE_PRESENCE_CONFIDENCE,
                output_segmentation_masks = OUTPUT_POSE_SEGMENTATION_MASKS,
            )
        )
    else:
        pose_tracker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python_utils.BaseOptions(model_asset_path="pose_landmarker_full.task"),
                running_mode = RUNNING_MODE,
                num_poses = NUM_POSES,
                min_pose_detection_confidence = MIN_POSE_DETECTION_CONFIDENCE,
                min_pose_presence_confidence = MIN_POSE_PRESENCE_CONFIDENCE,
                output_segmentation_masks = OUTPUT_POSE_SEGMENTATION_MASKS,
            )
        )


# ROUTES --------------------------------------------------------

@app.post("/hands/gesture")
async def hands_gesture(file: Union[UploadFile, bytes] = File(...)):
    
    assert hand_gesture != None, "Model not loaded yet!"
    
    image = await file_to_mediapipe_image(file)
    
    # predict keypionts and process results
    result = hand_gesture.recognize(image)
    
    if not result.handedness or len(result.handedness) == 0:
        return { "status": "error", "return": "No hands detected!" }
    
    hands = []
    for i in range(len(result.handedness)):
        hand = {
            "handedness": result.handedness[i][0].category_name,
            "score": result.handedness[i][0].score,
            "gesture": {
                result.gestures[i].gesture.categoryName: result.gestures[i].score
            }
        }
        hands.append(hand)
        
    return { 
        "status": "success", 
        "return": {
            "hands": hands,
            "image_size": [image.width, image.height]
        }
    }

@app.post("/hands/detect")
async def hands(file: Union[UploadFile, bytes] = File(...)):
    
    assert hand_tracker != None, "Model not loaded yet!"
    
    image = await file_to_mediapipe_image(file)
    
    # assert (image.width == 192 and image.height == 192) or (image.width == 224 and image.height == 224), "Image size must be 192x192 or 224x224!"
    
    # predict keypionts and process results
    result = hand_tracker.detect(image)
    
    if not result.handedness or len(result.handedness) == 0:
        return { "status": "error", "return": "No hands detected!" }
    
    hands = []
    for i in range(len(result.handedness)):
        hand = {}
        
        # extract information from result
        hand = {
            "handedness": result.handedness[i][0].category_name,
            "score": result.handedness[i][0].score,
            "imageSpace": [
                {
                    "keypoint": [kpt.x * image.width, kpt.y * image.height, kpt.z],
                    "visibility": kpt.visibility,
                    "presence": kpt.presence
                } for kpt in result.hand_landmarks[i]
            ],
            "worldSpace": [
                {
                    "keypoint": [kpt.x, kpt.y, kpt.z],
                    "visibility": kpt.visibility,
                    "presence": kpt.presence
                } for kpt in result.hand_world_landmarks[i]
            ]
        }
        
        # compute bounding box
        bbox_xy = np.min([kpt["keypoint"] for kpt in hand["imageSpace"]], axis=0)[:2]
        bbox_wh = np.max([kpt["keypoint"] for kpt in hand["imageSpace"]], axis=0)[:2] - bbox_xy
        hand["bbox"] = [*bbox_xy, *bbox_wh]
        
        # # parse keypoints to readable format
        # hand["imageSpace"] = parse_hand_keypoints_readable(hand["imageSpace"])
        # hand["worldSpace"] = parse_hand_keypoints_readable(hand["worldSpace"])
        
        hands.append(hand)
        
    return { 
        "status": "success", 
        "return": {
            "hands": hands,
            "image_size": [image.width, image.height]
        }
    }

@app.post("/face/detect")
async def face_keypoints(file: Union[UploadFile, bytes] = File(...)):
    
    assert face_detector != None, "Model not loaded yet!"
    
    image = await file_to_mediapipe_image(file)
    
    # predict keypionts, transformations and blendshapes and process results
    result = face_detector.detect(image)
    
    if len(result.detections) == 0:
        return { "status": "error", "return": "No faces detected!" }
    
    faces = []
    for i in range(len(result.detections)):
        
        bbox = result.detections[i].bounding_box
        
        face = {
            "score": result.detections[i].categories[0].score,
            "bbox": [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
            "keypoints": [[kpt.x * image.width, kpt.y * image.height] for kpt in result.detections[i].keypoints]
        }
        
        faces.append(face)
        
    return {
        "status": "success",
        "return": {
            "faces": faces,
            "image_size": [image.width, image.height]
        }
    }


@app.post("/face/keypoints")
async def face_keypoints(file: Union[UploadFile, bytes] = File(...)):
    
    assert face_tracker != None, "Model not loaded yet!"
    
    image = await file_to_mediapipe_image(file)
    
    # predict keypionts, transformations and blendshapes and process results
    result = face_tracker.detect(image)
    
    if len(result.face_landmarks) == 0:
        return { "status": "error", "return": "No faces detected!" }
    
    faces = []
    
    for i in range(len(result.face_landmarks)):
        
        transformation_matrix = list(map(list, result.facial_transformation_matrixes[i]))
        keypoints = [[res.x * image.width, res.y * image.height, res.z] for res in result.face_landmarks[i]]
        blendshapes = {res.category_name: res.score for res in result.face_blendshapes[i]}
        
        bbox_xy = np.min(keypoints, axis=0)[:2]
        bbox_wh = np.max(keypoints, axis=0)[:2] - bbox_xy
        
        face = {
            "bbox": [*bbox_xy, *bbox_wh],
            "transformationMatrix": transformation_matrix,
            "keypoints": keypoints,
            "visibility": [res.visibility for res in result.face_landmarks[i]],
            "presence": [res.presence for res in result.face_landmarks[i]],
            "blendshapes": blendshapes
        }
        
        faces.append(face)
    
    return {
        "status": "success",
        "return": {
            "faces": faces,
            "image_size": [image.width, image.height]
        }
    }


@app.post("/pose")
async def pose(file: Union[UploadFile, bytes] = File(...)):
    
    assert pose_tracker != None, "Model not loaded yet!"
    
    image = await file_to_mediapipe_image(file)
    
    # predict keypionts and process results
    result = pose_tracker.detect(image)
    
    if len(result.pose_landmarks) == 0:
        return { "status": "error", "return": "No poses detected!" }
    
    poses = []
    for i in range(len(result.pose_landmarks)):
        
        pose = {
            "imageSpace": [{
                "keypoint": [kpt.x * image.width, kpt.y * image.height, kpt.z],
                "visibility": kpt.visibility,
                "presence": kpt.presence
            } for kpt in result.pose_landmarks[i]],
            "worldSpace": [{
                "keypoint": [kpt.x * image.width, kpt.y * image.height, kpt.z],
                "visibility": kpt.visibility,
                "presence": kpt.presence
            } for kpt in result.pose_landmarks[i]],
            "segmentation_mask": result.segmentation_masks[i]
        }
            
        human_readable_keypoints_image_space = parse_pose_keypoints_readable(pose["imageSpace"])

        torso_kpts = [
            human_readable_keypoints_image_space["leftArm"]["shoulder"]["keypoint"],
            human_readable_keypoints_image_space["rightArm"]["shoulder"]["keypoint"],
            human_readable_keypoints_image_space["leftLeg"]["hip"]["keypoint"],
            human_readable_keypoints_image_space["rightLeg"]["hip"]["keypoint"],
            ]

        torso_bbox_xy = np.min(torso_kpts, axis=0)[:2]
        torso_bbox_wh = np.max(torso_kpts, axis=0)[:2] - torso_bbox_xy
        

        left_arm_bbox_xy = np.min([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["leftArm"].values()], axis=0)[:2]
        left_arm_bbox_wh = np.max([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["leftArm"].values()], axis=0)[:2] - left_arm_bbox_xy
        
        right_arm_bbox_xy = np.min([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["rightArm"].values()], axis=0)[:2]
        right_arm_bbox_wh = np.max([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["rightArm"].values()], axis=0)[:2] - right_arm_bbox_xy
        
        left_leg_bbox_xy = np.min([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["leftLeg"].values()], axis=0)[:2]
        left_leg_bbox_wh = np.max([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["leftLeg"].values()], axis=0)[:2] - left_leg_bbox_xy
        
        right_leg_bbox_xy = np.min([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["rightLeg"].values()], axis=0)[:2]
        right_leg_bbox_wh = np.max([kpt["keypoint"] for kpt in human_readable_keypoints_image_space["rightLeg"].values()], axis=0)[:2] - right_leg_bbox_xy
        
        pose["bboxes"] = {
            "torso": [*torso_bbox_xy, *torso_bbox_wh],
            "leftArm": [*left_arm_bbox_xy, *left_arm_bbox_wh],
            "rightArm": [*right_arm_bbox_xy, *right_arm_bbox_wh],
            "leftLeg": [*left_leg_bbox_xy, *left_leg_bbox_wh],
            "rightLeg": [*right_leg_bbox_xy, *right_leg_bbox_wh]
        }

        poses.append(pose)


        # # parse human readable format
        # pose["imageSpace"] = parse_pose_keypoints_readable(pose["imageSpace"])
        # pose["worldSpace"] = parse_pose_keypoints_readable(pose["worldSpace"])

        poses.append(pose)
