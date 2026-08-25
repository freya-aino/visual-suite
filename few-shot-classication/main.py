import pygame
import time
from webcam import Webcam
import keyboard
import numpy as np
import cv2
import asyncio
import timm
from urllib.request import urlopen
from PIL import Image
import torch as T
from torch import nn

from functools import partial
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, ToTensor

# is the deployment environment dev (for debugging)
# TODO - set by env var or something
IS_DEV = False
# ENCODER_MODEL = "vit_base_patch16_dinov3.lvd1689m"
# FRAME_SIZE = (256, 256)
ENCODER_MODEL = "vit_pe_spatial_tiny_patch16_512.fb"
FRAME_SIZE = (512, 512)
MAX_NUM_VECTORS = 32
INFERENCE_BUTTON = 8
BUTTON_NAME_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}
DEVICE = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
# DEVICE = T.device("cpu")


def put_save(loop, q, ell):
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(q.put(ell), loop)


def _init_model():
    model = timm.create_model(
        ENCODER_MODEL,
        pretrained=True,
        num_classes=0,
    ).eval()

    # print("\n".join([f"{n} => {m.shape}" for n, m in model.named_parameters()]))
    # print("\n".join([f"{n} => {m}" for n, m in model.named_modules()]))

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(model)
    # transforms = timm.data.create_transform(**data_config, is_training=False)
    transforms = T.jit.script(
        T.nn.Sequential(
            Resize(size=FRAME_SIZE, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            CenterCrop(size=FRAME_SIZE),
            Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        )
    )
    return model, transforms

def _to_chw(frame):
    """
    Convert one frame to a float CHW tensor in the range [0, 1].
    Supports NumPy arrays, PIL images, and torch tensors.
    """
    if T.is_tensor(frame):
        tensor = frame

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        elif tensor.ndim == 3:
            # Convert HWC to CHW when the channel dimension is last.
            if tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)

        else:
            raise ValueError(
                f"Expected one image with 2 or 3 dimensions, got {tensor.shape}"
            )

        tensor = tensor.contiguous().to(device=DEVICE)

        if tensor.dtype == T.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()
            if tensor.numel() > 0 and tensor.max() > 1.0:
                tensor = tensor / 255.0

        return tensor

    # PIL image or NumPy array
    return ToTensor()(frame).to(device=DEVICE)


def _infer_model(model, transforms, frames):
    """
    Convert one or more frames to BCHW and run inference.
    """
    if T.is_tensor(frames):
        if frames.ndim == 3:
            # Single CHW or HWC image
            batch = _to_chw(frames).unsqueeze(0)

        elif frames.ndim == 4:
            # Either BCHW or BHWC
            if frames.shape[-1] in (1, 3, 4):
                batch = frames.permute(0, 3, 1, 2)
            else:
                batch = frames

            batch = batch.contiguous().to(device=DEVICE)

            if batch.dtype == T.uint8:
                batch = batch.float() / 255.0
            else:
                batch = batch.float()
                if batch.numel() > 0 and batch.max() > 1.0:
                    batch = batch / 255.0

        else:
            raise ValueError(
                f"Expected frame tensor with 3 or 4 dimensions, got {frames.shape}"
            )

    elif isinstance(frames, (list, tuple)):
        batch = T.stack([_to_chw(frame) for frame in frames])

    else:
        # Single PIL image or NumPy array
        batch = _to_chw(frames).unsqueeze(0)

    batch = transforms(batch)

    with T.inference_mode():
        return model(batch)
    
# def _infer_model(model, transforms, frame):
#     # features = await loop.run_in_executor(None, _run_inference, model, transform, frame)
#     frame = ToTensor()(frame).to(dtype=T.float32, device=DEVICE)
#     frame = transforms(frame)
#     return model(frame)


def input_thread(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):

    try:

        pygame.init()
        joysticks = []

        if pygame.joystick.get_count() <= 0:
            print("[INPUT] WARNING - no controler detected")

        for i in range(0, pygame.joystick.get_count()):
            joysticks.append(pygame.joystick.Joystick(i))
            joysticks[-1].init()

        print("[INPUT] - started")

        while not stop_event.is_set():
            state = {}
            
            for event in pygame.event.get():
                
                # print(event.type, pygame.JOYBUTTONDOWN, pygame.KEYDOWN)

                if event.type == pygame.JOYBUTTONDOWN:
                    state[event.type] = event.dict
                # if event.type == pygame.KEYDOWN:
                #     state[event.type] = event.dict

            if state != {}:
                
                print(f"[INPUT] recorded: {state}")

                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                put_save(loop, queue, state)
             
    except Exception as e:
        print(f"[INPUT] ERROR - {e}")
    finally:
        if not stop_event.is_set():
            stop_event.set()
        print("[INPUT] stopped")


def camera_thread(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):

    try:

        webcam = Webcam(src=0, w=640)
        print("[CAMERA] started")

        for frame in webcam:
            if stop_event.is_set():
                break
            if queue.full():
                # non-blocking drop of oldest frame to avoid unbounded growth
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            put_save(loop, queue, frame)
    except Exception as e:
        print(f"[CAMERA] error: {e}")
    finally:
        webcam.release()
        if not stop_event.is_set():
            stop_event.set()
        print("[CAMERA] stopped")

async def camera_task(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):
    await loop.run_in_executor(None, camera_thread, queue, loop, stop_event)

async def input_task(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):
    await loop.run_in_executor(None, input_thread, queue, loop, stop_event)

async def process_task(frame_queue: asyncio.Queue, input_queue: asyncio.Queue, result_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):

    print("[PROCESS] start")
    try:
        while not stop_event.is_set():

            # get latest frame
            frame = await frame_queue.get()
            while not frame_queue.empty():
                try:
                    frame = frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            cv2.imshow("webcam frames", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if input_queue.empty():
                continue

            user_input = await input_queue.get()
            
            button_input = user_input.get(pygame.JOYBUTTONDOWN, None)

            if button_input is None:
                continue
                
            if result_queue.full():
                # put_save(loop, result_queue, (frame, button_input))
                try:
                    result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            put_save(loop, result_queue, (frame, button_input))

    except Exception as e:
        print(f"[PROCESS] failed - main loop: {e}")
    finally:
        cv2.destroyAllWindows()
        if not stop_event.is_set():
            stop_event.set()
        print("[PROCESS] stopped")


async def training_task(result_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):

    button_vectors = {}
    is_inference = False

    try:

        print("[INFERENCE]: initializing model")
        model, transforms = await loop.run_in_executor(None, _init_model)
        model = model.to(device=DEVICE)
        transforms = transforms.to(device=DEVICE)


        print("[INFERENCE] - starting")
        while not stop_event.is_set():

            try:
                # This yields control to the other asyncio tasks.
                first_item = await asyncio.wait_for(
                    result_queue.get(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue


            pending_items = [first_item]

            # Drain currently available items without blocking.
            while True:
                try:
                    pending_items.append(result_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            frame_buffer = {}

            # Build frame groups
            frame_buffer = {}

            for frame, button_input in pending_items:
                button = button_input.get("button")

                if button is not None:
                    frame_buffer.setdefault(button, []).append(frame)

            if not frame_buffer:
                continue

            # Inference mode
            if INFERENCE_BUTTON in frame_buffer:
                if not button_vectors:
                    print("[inference] no trained button vectors available")
                    continue

                features = await loop.run_in_executor(
                    None,
                    _infer_model,
                    model,
                    transforms,
                    frame_buffer[INFERENCE_BUTTON],
                )

                query = features.detach().cpu().mean(dim=0)

                prototypes = {
                    button: vectors.mean(dim=0)
                    for button, vectors in button_vectors.items()
                }

                distances = {
                    BUTTON_NAME_MAP[button]: T.dist(prototype, query).item()
                    for button, prototype in prototypes.items()
                }

                prediction = min(distances.items(), key=lambda item: item[1])
                print(f"[inference] prediction: {prediction}")
                continue

            # Training mode
            for button, frames in frame_buffer.items():
                if button not in BUTTON_NAME_MAP:
                    continue

                print(
                    f"[training] processing {len(frames)} frames "
                    f"for button {button}"
                )

                features = await loop.run_in_executor(
                    None,
                    _infer_model,
                    model,
                    transforms,
                    frames,
                )

                features_cpu = features.detach().cpu()

                if button in button_vectors:
                    button_vectors[button] = T.cat(
                        [button_vectors[button], features_cpu],
                        dim=0,
                    )
                else:
                    button_vectors[button] = features_cpu

                print(
                    f"[training] {BUTTON_NAME_MAP[button]}: "
                    # f"{button_vectors[button].shape}"
                )

            frame_buffer = {}
            if not result_queue.empty():
                while not result_queue.empty(): 
                    frame, button_input = await result_queue.get()

                    if button_input["button"] == INFERENCE_BUTTON:
                        break

                    # time.sleep(0.3)               
                    await asyncio.sleep(0.3)
                    
                    # if button_input["button"] in frame_buffer:
                    #     frame_buffer[button_input["button"]].stack(T.from_numpy(frame), dim=0)


                    # else:
                    #     frame_buffer[button_input["button"]] = T.from_numpy(frame).unsqueeze(0)

                    button = button_input["button"]
                    # Keep frames as NumPy arrays.
                    # _infer_model() will convert them into a batch later.
                    frame_buffer.setdefault(button, []).append(frame)
            
            if INFERENCE_BUTTON in frame_buffer.keys():

                # inference
                # features = await loop.run_in_executor(None, _infer_model, model, transforms, frame_buffer[INFERENCE_BUTTON].unsqueeze())
                features = await loop.run_in_executor(None, _infer_model, model, transforms, frame_buffer[INFERENCE_BUTTON])

                # compute tensors
                print("[inference] stacking tensors")
                # if not ojb_tensors_computed:
                #     for k, v in button_vectors.items():
                #         obj_tensors[k] = T.stack(v, dim=1).to(device=DEVICE)
                #     ojb_tensors_computed = True
                prototypes = {
                    button: vectors.mean(dim=0)
                    for button, vectors in button_vectors.items()
                }

                # distances = {BUTTON_NAME_MAP[k]: T.dist(obj_tensors[k].mean(dim=0).squeeze(), features.squeeze()).item() for k in obj_tensors}
                query = features.detach().cpu().mean(dim=0)

                distances = {
                    BUTTON_NAME_MAP[button]: T.dist(prototype, query).item()
                    for button, prototype in prototypes.items()
                }

                print(
                    f"prediction: {min(distances.items(), key=lambda item: item[1])}"
                )
                print(f"prediction: {min(distances.items(), key = lambda a: a[1])}")
                continue

            else:

                for button, frames in frame_buffer.items():
                    # print(f"[inference] processing frame for button {button} - {frames.shape}")

                    features = await loop.run_in_executor(None, _infer_model, model, transforms, frames)

                    print("------- TRAINING -------")

                    # Skip if button is not named
                    if button not in BUTTON_NAME_MAP.keys():
                        continue

                    # if button in button_vectors:
                    #     print("appending")
                    #     button_vectors[button].stack(features.to("cpu"), dim=0)
                    # else:
                    #     print("new vector")
                    #     button_vectors[button] = features.to("cpu")
                    features_cpu = features.detach().cpu()

                    if button in button_vectors:
                        button_vectors[button] = T.cat(
                            [button_vectors[button], features_cpu],
                            dim=0,
                        )
                    else:
                        button_vectors[button] = features_cpu

                    # ojb_tensors_computed = False

                    # print(f"[inference] shape={features.shape} mean={features.mean():.4f}")
                    print("\n".join([f"{BUTTON_NAME_MAP[k]}: {e.shape}" for k, e in button_vectors.items()]))

            # reset frame buffer
            frame_buffer = {}

    except Exception as exp:
        print(f"[INFERENCE] ERROR - {exp}")
    finally:
        if not stop_event.is_set():
            stop_event.set()
        print("[INFERENCE] - stopped")



async def amain(loop: asyncio.AbstractEventLoop):
    
    frame_queue = asyncio.Queue(maxsize=2)
    input_queue = asyncio.Queue(maxsize=2)
    result_queue = asyncio.Queue(maxsize=10)
    stop_event = asyncio.Event()
    
    result = await asyncio.gather(
        asyncio.create_task(input_task(input_queue, loop, stop_event)),
        asyncio.create_task(camera_task(frame_queue, loop, stop_event)), 
        asyncio.create_task(process_task(frame_queue, input_queue, result_queue, loop, stop_event)),
        asyncio.create_task(training_task(result_queue, loop, stop_event)),
        return_exceptions=True
    )
    
    for r in result:
        if isinstance(r, Exception):
            print(f"[AMAIN] ERROR - task raised {r}")

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(amain(loop))
    except Exception as e:
        print(e.with_traceback)
    finally:
        print("[MAIN] - trying to shut down ")
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(asyncio.sleep(1.0))
        loop.close()
        print("[MAIN] - loop closed")

if __name__ == "__main__":
    main()