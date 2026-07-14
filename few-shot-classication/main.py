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
ENCODER_MODEL = "vit_base_patch16_dinov3.lvd1689m"
FRAME_SIZE = (256, 256)
# ENCODER_MODEL = "vit_pe_spatial_tiny_patch16_512.fb"
# FRAME_SIZE = (512, 512)
MAX_NUM_VECTORS = 32
INFERENCE_BUTTON = 8
BUTTON_NAME_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}
DEVICE = T.device("cuda") if T.cuda.is_available() else T.device("cpu")


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

def _infer_model(model, transforms, frame):
    # features = await loop.run_in_executor(None, _run_inference, model, transform, frame)
    frame = ToTensor()(frame).to(dtype=T.float32, device=DEVICE)
    frame = transforms(frame)
    return model(frame)


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
    obj_tensors = {}
    ojb_tensors_computed = False
    is_inference = False

    try:

        print("[INFERENCE]: initializing model")
        model, transforms = await loop.run_in_executor(None, _init_model)
        model = model.to(device=DEVICE)
        transforms = transforms.to(device=DEVICE)


        print("[INFERENCE] - starting")
        while not stop_event.is_set():

            frame_buffer = {}
            if not result_queue.empty():
                while not result_queue.empty(): 
                    frame, button_input = await result_queue.get()

                    if button_input["button"] == INFERENCE_BUTTON:
                        break

                    time.sleep(0.3)               
                    
                    if button_input["button"] in frame_buffer:
                        frame_buffer[button_input["button"]].stack(T.from_numpy(frame), dim=0)
                    else:
                        frame_buffer[button_input["button"]] = T.from_numpy(frame).unsqueeze(0)
            
            if INFERENCE_BUTTON in frame_buffer.keys():

                # inference
                features = await loop.run_in_executor(None, _infer_model, model, transforms, frame_buffer[INFERENCE_BUTTON].unsqueeze())

                # compute tensors
                print("[inference] stacking tensors")
                if not ojb_tensors_computed:
                    for k, v in button_vectors.items():
                        obj_tensors[k] = T.stack(v, dim=1).to(device=DEVICE)
                    ojb_tensors_computed = True

                distances = {BUTTON_NAME_MAP[k]: T.dist(obj_tensors[k].mean(dim=0).squeeze(), features.squeeze()).item() for k in obj_tensors}
                print(f"prediction: {min(distances.items(), key = lambda a: a[1])}")
                continue

            else:

                for button, frames in frame_buffer.items():
                    print(f"[inference] processing frame for button {button} - {frames.shape}")

                    features = await loop.run_in_executor(None, _infer_model, model, transforms, frames)

                    print("------- TRAINING -------")

                    # Skip if button is not named
                    if button not in BUTTON_NAME_MAP.keys():
                        continue

                    if button in button_vectors:
                        print("appending")
                        button_vectors[button].stack(features.to("cpu"), dim=0)
                    else:
                        print("new vector")
                        button_vectors[button] = features.to("cpu")
                    ojb_tensors_computed = False

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