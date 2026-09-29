import queue
import threading
import time
import traceback

import cv2
import numpy as np
import pygame
import timm
import torch as T
from torchvision.transforms import (
    CenterCrop,
    InterpolationMode,
    Normalize,
    Resize,
)
from ultralytics import YOLO
from webcam import Webcam


ENCODER_MODEL = "vit_pe_spatial_tiny_patch16_512.fb"
FRAME_SIZE = (512, 512)
DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")

YOLO_MODEL = "yolov9t.pt"
YOLO_CONF = 0.25

INFERENCE_BUTTON = 8
START_BUTTON = 7  # Change if your controller reports START at another index.

BUTTON_NAME_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

# OpenCV uses BGR colors.
CLASS_COLORS = {
    0: (70, 200, 70),      # Green
    1: (65, 65, 235),      # Red
    2: (235, 125, 55),     # Blue
    3: (40, 215, 245),     # Yellow
}

CLASS_COLOR_NAMES = {
    0: "GREEN",
    1: "RED",
    2: "BLUE",
    3: "YELLOW",
}

WINDOW_NAME = "Webcam Classification"
HIGHLIGHT_SECONDS = 1.5
MAX_PENDING_SAMPLES = 64
UI_FPS = 30
CONTROLLER_SCAN_INTERVAL = 1.0

# The original program displays webcam frames as RGB.
CAMERA_RETURNS_RGB = True


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.generation = 0
        self.counts = {button: 0 for button in BUTTON_NAME_MAP}
        self.ready = False
        self.status = "Loading models..."
        self.camera_error = None
        self.prediction = None
        self.highlight_until = 0.0
        self.controller_count = 0

    def snapshot(self):
        with self.lock:
            return {
                "generation": self.generation,
                "counts": dict(self.counts),
                "ready": self.ready,
                "status": self.status,
                "camera_error": self.camera_error,
                "prediction": self.prediction,
                "highlight_until": self.highlight_until,
                "controller_count": self.controller_count,
            }


def _init_model():
    model = timm.create_model(
        ENCODER_MODEL,
        pretrained=True,
        num_classes=0,
    ).eval().to(DEVICE)

    transforms = T.nn.Sequential(
        Resize(
            size=FRAME_SIZE,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        CenterCrop(size=FRAME_SIZE),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ).to(DEVICE)

    return model, transforms


def _crop_largest_object(yolo_model, frame_rgb):
    # Ultralytics interprets NumPy image inputs as BGR.
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = yolo_model.predict(
        frame_bgr,
        conf=YOLO_CONF,
        verbose=False,
        device=str(DEVICE),
    )[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return cv2.resize(frame_rgb, FRAME_SIZE)

    xyxy = boxes.xyxy.detach().cpu().numpy()
    areas = (
        np.maximum(0, xyxy[:, 2] - xyxy[:, 0])
        * np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
    )
    largest = xyxy[int(np.argmax(areas))]

    h, w = frame_rgb.shape[:2]
    x1 = int(np.clip(np.floor(largest[0]), 0, w))
    y1 = int(np.clip(np.floor(largest[1]), 0, h))
    x2 = int(np.clip(np.ceil(largest[2]), 0, w))
    y2 = int(np.clip(np.ceil(largest[3]), 0, h))

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_rgb

    return cv2.resize(crop, FRAME_SIZE)


def _infer_model(model, transforms, frame_rgb):
    tensor = T.from_numpy(
        np.ascontiguousarray(frame_rgb)
    ).permute(2, 0, 1)

    tensor = tensor.to(device=DEVICE, dtype=T.float32)
    tensor = tensor.unsqueeze(0) / 255.0

    with T.inference_mode():
        features = model(transforms(tensor))

    if not T.is_tensor(features) or features.ndim != 2:
        raise RuntimeError("Expected encoder output with shape [batch, features].")

    return features[0].detach().float().cpu()


def camera_worker(frame_queue, state, stop_event):
    webcam = None
    try:
        webcam = Webcam(src=0, w=640)

        for frame in webcam:
            if stop_event.is_set():
                break

            frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3:
                continue

            if not CAMERA_RETURNS_RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = np.ascontiguousarray(frame).copy()

            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

        if not stop_event.is_set():
            with state.lock:
                state.camera_error = "Camera stopped. Restart the application."

    except Exception as exc:
        traceback.print_exc()
        with state.lock:
            state.camera_error = f"Camera error: {exc}"

    finally:
        if webcam is not None:
            try:
                webcam.release()
            except Exception:
                pass


def model_worker(work_queue, state, stop_event):
    try:
        model, transforms = _init_model()

        with state.lock:
            state.status = "Loading object detector..."

        yolo_model = YOLO(YOLO_MODEL)

        with state.lock:
            state.ready = True
            state.status = "Ready. Press A-D to collect images."

    except Exception as exc:
        traceback.print_exc()
        with state.lock:
            state.ready = False
            state.status = f"Model initialization failed: {exc}"
        return

    # Running sums preserve the mean of every collected embedding without
    # storing an ever-growing tensor or keeping the original images.
    feature_sums = {}
    feature_counts = {}
    active_generation = -1

    while not stop_event.is_set():
        try:
            generation, button, frame = work_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            with state.lock:
                if generation != state.generation:
                    continue

                if generation != active_generation:
                    feature_sums.clear()
                    feature_counts.clear()
                    active_generation = generation

                if button == INFERENCE_BUTTON and not feature_counts:
                    state.status = "Collect images before running inference."
                    continue

                state.status = (
                    "Classifying..."
                    if button == INFERENCE_BUTTON
                    else f"Collecting image for {BUTTON_NAME_MAP[button]}..."
                )

            crop = _crop_largest_object(yolo_model, frame)
            feature = _infer_model(model, transforms, crop)

            with state.lock:
                # Ignore work that finished after START reset the dataset.
                if generation != state.generation:
                    continue

                if button == INFERENCE_BUTTON:
                    distances = {
                        class_button: T.dist(
                            feature_sums[class_button] / count,
                            feature,
                        ).item()
                        for class_button, count in feature_counts.items()
                    }

                    selected = min(distances, key=distances.get)
                    state.prediction = selected
                    state.highlight_until = (
                        time.monotonic() + HIGHLIGHT_SECONDS
                    )
                    state.status = (
                        f"Prediction: {BUTTON_NAME_MAP[selected]} "
                        f"({CLASS_COLOR_NAMES[selected]})"
                    )
                else:
                    if button not in feature_sums:
                        feature_sums[button] = feature.clone()
                        feature_counts[button] = 1
                    else:
                        feature_sums[button] += feature
                        feature_counts[button] += 1

                    state.counts[button] = feature_counts[button]
                    state.status = (
                        f"Collected image for {BUTTON_NAME_MAP[button]}."
                    )

        except Exception as exc:
            traceback.print_exc()
            with state.lock:
                if generation == state.generation:
                    state.status = f"Processing error: {exc}"

        finally:
            work_queue.task_done()


def reset_collected_images(state, work_queue):
    with state.lock:
        state.generation += 1
        state.counts = {button: 0 for button in BUTTON_NAME_MAP}
        state.prediction = None
        state.highlight_until = 0.0
        state.status = (
            "All collected images removed."
            if state.ready
            else "Dataset cleared. Waiting for models..."
        )

    # Remove pending samples. In-flight results are invalidated by generation.
    while True:
        try:
            work_queue.get_nowait()
        except queue.Empty:
            break
        else:
            work_queue.task_done()


def submit_sample(button, frame, state, work_queue):
    if button == START_BUTTON:
        reset_collected_images(state, work_queue)
        return

    if button not in BUTTON_NAME_MAP and button != INFERENCE_BUTTON:
        return

    with state.lock:
        if not state.ready:
            return

        if state.camera_error:
            state.status = "Camera unavailable. Cannot capture an image."
            return

        if frame is None:
            state.status = "Waiting for the camera..."
            return

        generation = state.generation

    try:
        work_queue.put_nowait((generation, button, frame.copy()))
    except queue.Full:
        with state.lock:
            state.status = "Processing queue full. Please wait."


def update_controller_count(state, joysticks):
    with state.lock:
        previous_count = state.controller_count
        state.controller_count = len(joysticks)
        current_count = state.controller_count

    if previous_count > 0 and current_count == 0:
        print(
            "[INPUT] No controller connected. "
            "Searching automatically. Keyboard controls enabled."
        )


def disconnect_controller(instance_id, joysticks, state):
    joystick = joysticks.pop(instance_id, None)

    if joystick is not None:
        try:
            joystick.quit()
        except pygame.error:
            pass

        print(f"[INPUT] Controller disconnected: instance {instance_id}")

    update_controller_count(state, joysticks)


def refresh_controllers(joysticks, state):
    """Discover and initialize OS-visible controllers on the main thread."""
    try:
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        pygame.event.pump()
        device_count = pygame.joystick.get_count()
    except pygame.error:
        # Keep the UI running and retry during the next scheduled scan.
        return

    detected_ids = set()
    scan_complete = True

    for device_index in range(device_count):
        try:
            joystick = pygame.joystick.Joystick(device_index)

            if not joystick.get_init():
                joystick.init()

            instance_id = joystick.get_instance_id()
            detected_ids.add(instance_id)

            if instance_id not in joysticks:
                joysticks[instance_id] = joystick

                try:
                    name = joystick.get_name()
                except pygame.error:
                    name = "Controller"

                print(
                    f"[INPUT] Controller connected: {name} "
                    f"(instance {instance_id})"
                )

        except pygame.error:
            # A device can disappear while scanning. Retry next time.
            scan_complete = False

    # Do not remove working controllers based on an incomplete scan.
    if scan_complete:
        for instance_id in list(joysticks):
            if instance_id not in detected_ids:
                disconnect_controller(instance_id, joysticks, state)

    update_controller_count(state, joysticks)


def fit_text(
    image,
    text,
    x,
    y,
    max_width,
    scale=0.7,
    color=(235, 235, 235),
    thickness=1,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = cv2.getTextSize(text, font, scale, thickness)[0][0]

    if width > max_width and width > 0:
        scale *= max_width / width

    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def centered_text(
    image,
    text,
    rect,
    scale=0.7,
    color=(255, 255, 255),
    thickness=1,
):
    x1, y1, x2, y2 = rect
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = max(1, x2 - x1 - 24)

    (text_width, text_height), _ = cv2.getTextSize(
        text, font, scale, thickness
    )

    if text_width > max_width:
        scale *= max_width / text_width
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, scale, thickness
        )

    x = x1 + (x2 - x1 - text_width) // 2
    y = y1 + (y2 - y1 + text_height) // 2

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_banner(canvas, lines, color, top, video_width, video_height, margin, scale):
    line_height = max(23, int(32 * scale))
    banner_height = len(lines) * line_height + margin

    x1 = margin
    y1 = top
    x2 = video_width - margin
    y2 = min(video_height - margin, y1 + banner_height)

    if x2 <= x1 or y2 <= y1:
        return top

    roi = canvas[y1:y2, x1:x2]
    dark = np.full_like(roi, (16, 19, 24))
    cv2.addWeighted(dark, 0.86, roi, 0.14, 0, dst=roi)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    for index, line in enumerate(lines):
        baseline = y1 + line_height * (index + 1)

        if baseline >= y2:
            break

        fit_text(
            canvas,
            line,
            x1 + margin,
            baseline,
            x2 - x1 - 2 * margin,
            scale=(0.7 if index == 0 else 0.57) * scale,
            color=color if index == 0 else (245, 245, 245),
            thickness=2 if index == 0 else 1,
        )

    return y2 + margin


def draw_interface(frame_rgb, snapshot, screen_size):
    width, height = screen_size
    canvas = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)

    # The classification rectangles occupy approximately the right third.
    panel_width = width // 3
    video_width = width - panel_width
    scale = max(0.5, min(width / 1440.0, height / 900.0))
    margin = max(10, int(18 * scale))
    footer_height = max(104, int(140 * scale))
    video_height = height - footer_height

    if frame_rgb is not None:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_h, frame_w = frame_bgr.shape[:2]
        ratio = min(video_width / frame_w, video_height / frame_h)

        resized_w = max(1, int(frame_w * ratio))
        resized_h = max(1, int(frame_h * ratio))
        resized = cv2.resize(frame_bgr, (resized_w, resized_h))

        x = (video_width - resized_w) // 2
        y = (video_height - resized_h) // 2
        canvas[y:y + resized_h, x:x + resized_w] = resized
    else:
        centered_text(
            canvas,
            "Waiting for camera...",
            (0, 0, video_width, video_height),
            scale=0.9 * scale,
        )

    banner_top = margin

    # Keep this notification visible until a controller is connected.
    if snapshot["controller_count"] == 0:
        banner_top = draw_banner(
            canvas,
            [
                "NO CONTROLLER CONNECTED",
                "Searching every second. Connect or pair a controller.",
                "Keyboard controls remain available.",
            ],
            (70, 210, 255),
            banner_top,
            video_width,
            video_height,
            margin,
            scale,
        )

    populated_classes = sum(
        count > 0 for count in snapshot["counts"].values()
    )

    banner_lines = None
    banner_color = (70, 210, 255)

    if snapshot["camera_error"]:
        banner_lines = [
            "CAMERA UNAVAILABLE",
            snapshot["camera_error"],
        ]
        banner_color = (90, 100, 255)
    elif populated_classes == 1:
        banner_lines = [
            "ONLY ONE CLASS HAS IMAGES",
            "Predictions can only select that class.",
            "Collect images for another class to compare.",
        ]
    elif populated_classes == 0 and snapshot["ready"]:
        banner_lines = [
            "NO COLLECTED IMAGES",
            "Press A-D to collect examples for each class.",
        ]

    if banner_lines:
        draw_banner(
            canvas,
            banner_lines,
            banner_color,
            banner_top,
            video_width,
            video_height,
            margin,
            scale,
        )

    cv2.rectangle(
        canvas,
        (0, video_height),
        (video_width, height),
        (24, 27, 33),
        -1,
    )

    status_width = video_width - 2 * margin
    line_spacing = max(20, int(29 * scale))
    baseline = video_height + line_spacing

    fit_text(
        canvas,
        snapshot["status"],
        margin,
        baseline,
        status_width,
        scale=0.65 * scale,
    )

    controller_count = snapshot["controller_count"]

    if controller_count:
        controller_text = (
            "Controller connected"
            if controller_count == 1
            else f"{controller_count} controllers connected"
        )
        controller_color = (100, 220, 100)
    else:
        controller_text = "No controller connected - searching automatically..."
        controller_color = (70, 210, 255)

    fit_text(
        canvas,
        controller_text,
        margin,
        baseline + line_spacing,
        status_width,
        scale=0.52 * scale,
        color=controller_color,
    )
    fit_text(
        canvas,
        "A-D: collect   |   INFERENCE / SPACE: classify",
        margin,
        baseline + 2 * line_spacing,
        status_width,
        scale=0.52 * scale,
        color=(175, 180, 190),
    )
    fit_text(
        canvas,
        "START / ENTER: clear all images   |   Q / ESC: quit",
        margin,
        baseline + 3 * line_spacing,
        status_width,
        scale=0.52 * scale,
        color=(175, 180, 190),
    )

    panel_x1 = video_width + margin
    panel_x2 = width - margin
    available_height = height - 5 * margin
    card_height = available_height // 4

    highlight_active = time.monotonic() < snapshot["highlight_until"]

    for index, button in enumerate(BUTTON_NAME_MAP):
        y1 = margin + index * (card_height + margin)
        y2 = y1 + card_height
        color = CLASS_COLORS[button]

        selected = (
            highlight_active and snapshot["prediction"] == button
        )

        fill = color if selected else tuple(
            int(channel * 0.38) for channel in color
        )

        cv2.rectangle(
            canvas,
            (panel_x1, y1),
            (panel_x2, y2),
            fill,
            -1,
        )

        if selected:
            border_width = max(4, int(6 * scale))
            cv2.rectangle(
                canvas,
                (panel_x1, y1),
                (panel_x2, y2),
                (255, 255, 255),
                border_width,
            )

            inset = border_width + 3
            cv2.rectangle(
                canvas,
                (panel_x1 + inset, y1 + inset),
                (panel_x2 - inset, y2 - inset),
                (20, 23, 28),
                max(1, int(2 * scale)),
            )
        else:
            cv2.rectangle(
                canvas,
                (panel_x1, y1),
                (panel_x2, y2),
                color,
                max(1, int(2 * scale)),
            )

        text_color = (15, 20, 25) if selected else (245, 245, 245)

        centered_text(
            canvas,
            f"{BUTTON_NAME_MAP[button]}  /  {CLASS_COLOR_NAMES[button]}",
            (
                panel_x1,
                y1 + int(card_height * 0.08),
                panel_x2,
                y1 + int(card_height * 0.37),
            ),
            scale=0.9 * scale,
            color=text_color,
            thickness=2,
        )

        count = snapshot["counts"][button]
        centered_text(
            canvas,
            str(count),
            (
                panel_x1,
                y1 + int(card_height * 0.32),
                panel_x2,
                y1 + int(card_height * 0.72),
            ),
            scale=1.7 * scale,
            color=text_color,
            thickness=3,
        )

        centered_text(
            canvas,
            "SELECTED" if selected else (
                "collected image" if count == 1 else "collected images"
            ),
            (
                panel_x1,
                y1 + int(card_height * 0.72),
                panel_x2,
                y2 - int(card_height * 0.05),
            ),
            scale=0.55 * scale,
            color=text_color,
            thickness=2 if selected else 1,
        )

    return canvas


def main():
    state = AppState()
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=1)
    work_queue = queue.Queue(maxsize=MAX_PENDING_SAMPLES)

    workers = []
    joysticks = {}
    latest_frame = None

    try:
        pygame.init()

        display_info = pygame.display.Info()
        screen_size = (
            display_info.current_w or 1280,
            display_info.current_h or 720,
        )

        # Initialization failures are retried by subsequent scans.
        refresh_controllers(joysticks, state)
        next_controller_scan = (
            time.monotonic() + CONTROLLER_SCAN_INTERVAL
        )

        if not joysticks:
            print(
                "[INPUT] No controller connected. "
                "Searching every second. Keyboard controls enabled."
            )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

        # Paint the fullscreen interface before loading models.
        cv2.imshow(
            WINDOW_NAME,
            draw_interface(None, state.snapshot(), screen_size),
        )
        cv2.waitKey(1)

        workers = [
            threading.Thread(
                target=camera_worker,
                args=(frame_queue, state, stop_event),
                daemon=True,
                name="camera",
            ),
            threading.Thread(
                target=model_worker,
                args=(work_queue, state, stop_event),
                daemon=True,
                name="models",
            ),
        ]

        for worker in workers:
            worker.start()

        keyboard_buttons = {
            ord("a"): 0,
            ord("b"): 1,
            ord("c"): 2,
            ord("d"): 3,
            ord("A"): 0,
            ord("B"): 1,
            ord("C"): 2,
            ord("D"): 3,
            ord(" "): INFERENCE_BUTTON,
            10: START_BUTTON,
            13: START_BUTTON,
        }

        clock = pygame.time.Clock()

        while not stop_event.is_set():
            try:
                latest_frame = frame_queue.get_nowait()
            except queue.Empty:
                pass

            # Keep controller event handling and GUI operations on the main thread.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()

                elif event.type == pygame.JOYDEVICEADDED:
                    # Scan current devices instead of trusting an event index
                    # that may have changed during rapid hot-plugging.
                    refresh_controllers(joysticks, state)

                elif event.type == pygame.JOYDEVICEREMOVED:
                    disconnect_controller(
                        event.instance_id,
                        joysticks,
                        state,
                    )
                    next_controller_scan = 0.0

                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.instance_id not in joysticks:
                        refresh_controllers(joysticks, state)

                    if event.instance_id in joysticks:
                        submit_sample(
                            event.button,
                            latest_frame,
                            state,
                            work_queue,
                        )

            if stop_event.is_set():
                break

            # Periodic discovery also runs when no controller is connected.
            # OS-visible controllers are initialized automatically. Bluetooth
            # pairing, if required, must be completed in the operating system.
            now = time.monotonic()
            if now >= next_controller_scan:
                refresh_controllers(joysticks, state)
                next_controller_scan = (
                    time.monotonic() + CONTROLLER_SCAN_INTERVAL
                )

            canvas = draw_interface(
                latest_frame,
                state.snapshot(),
                screen_size,
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if key in keyboard_buttons:
                submit_sample(
                    keyboard_buttons[key],
                    latest_frame,
                    state,
                    work_queue,
                )

            try:
                if cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
            except cv2.error:
                break

            clock.tick(UI_FPS)

    except KeyboardInterrupt:
        pass

    except Exception:
        traceback.print_exc()

    finally:
        stop_event.set()

        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=2.0)

        for joystick in list(joysticks.values()):
            try:
                joystick.quit()
            except pygame.error:
                pass

        joysticks.clear()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()