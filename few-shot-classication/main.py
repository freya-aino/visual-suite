import pygame
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

# is the deployment environment dev (for debugging)
# TODO - set by env var or something
IS_DEV = False

image_vector_collections = {
    "A": [],
    "B": [],
    "C": [],
    "D": []
}

def _init_model():

    print("[_INIT_MODEL] initializing models")
    
    model = timm.create_model(
        'vit_base_patch16_dinov3.lvd1689m',
        pretrained=True,
        num_classes=0,
    ).eval()

    # print("\n".join([f"{n} => {m.shape}" for n, m in model.named_parameters()]))
    # print("\n".join([f"{n} => {m}" for n, m in model.named_modules()]))

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # output = model(transforms(img).unsqueeze(0))
    return model, transforms

def _run_inference(model, transforms, frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = Image.fromarray(frame)
    user_input = transforms(frame).unsqueeze(0)
    return model(user_input)


def input_thread(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):

    pygame.init()
    joysticks = []

    for i in range(0, pygame.joystick.get_count()):
        joysticks.append(pygame.joystick.Joystick(i))
        joysticks[-1].init()

    def put(a):
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(queue.put(a), loop)

    print("[INPUT] started")
    try:
        while not stop_event.is_set():
            state = {}
            for event in pygame.event.get():
                state[event.type] = event.dict

            if state != {}:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                put(state)
             
    except Exception as e:
        print(f"[INPUT] error: {e}")
    finally:
        print("[INPUT] stopped")


def camera_thread(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):

    webcam = Webcam(src=0, w=640)
    print("[CAMERA] started")

    def put(frame):
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(queue.put(frame), loop)

    try:
        for frame in webcam:
            if stop_event.is_set():
                break
            if queue.full():
                # non-blocking drop of oldest frame to avoid unbounded growth
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            put(frame)
    except Exception as e:
        print(f"[CAMERA] error: {e}")
    finally:
        webcam.release()
        print("[CAMERA] stopped")


async def camera_task(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):
    await loop.run_in_executor(None, camera_thread, queue, loop, stop_event)

async def input_task(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):
    await loop.run_in_executor(None, input_thread, queue, loop, stop_event)

async def inference_task(frame_queue: asyncio.Queue, input_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event):

    button_vectors = {
        0: [],
        1: [],
        2: [],
        3: []
    }
    def handle_features(future, button_input):
        try:
            
            features = future.result()
            button_vectors[button_input["button"]].append(features)

            print(f"[inference] shape={features.shape} mean={features.mean():.4f}")
            print("\n".join([f"{k}: {len(e)}" for k, e in button_vectors.items()]))

        except Exception as e:
            print(f"[inference] error - {e}")


    try:
        model, transform = await loop.run_in_executor(None, _init_model)
    except Exception as e:
        print("[INFERENCE] failed : model initalization")
        stop_event.set()
        return

    print("[INFERENCE] start")
    try:
        while True:
            frame = await frame_queue.get()

            cv2.imshow("webcam frames", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            if input_queue.empty():
                continue

            user_input = await input_queue.get()
            
            button_input = user_input.get(pygame.JOYBUTTONDOWN, None)

            if button_input is None:
                continue
                
            if button_input["button"] not in button_vectors.keys():
                continue

            infer = partial(_run_inference, model, transform, frame)
            try:
                future = loop.run_in_executor(None, infer)
                future.add_done_callback(partial(handle_features, button_input=button_input))
            except Exception as e:
                print(f"[inference] failed - frame: {e}")

    except Exception as e:
        print(f"[INFERENCE] failed - main loop: {e}")
    finally:
        cv2.destroyAllWindows()
        stop_event.set()
        print("[INFERENCE] stopped")


async def amain(loop: asyncio.AbstractEventLoop):
    
    frame_queue = asyncio.Queue(maxsize=2)
    input_queue = asyncio.Queue(maxsize=2)
    stop_event = asyncio.Event()
    
    result = await asyncio.gather(
        asyncio.create_task(camera_task(frame_queue, loop, stop_event)), 
        asyncio.create_task(inference_task(frame_queue, input_queue, loop, stop_event)),
        asyncio.create_task(input_task(input_queue, loop, stop_event)),
        return_exceptions=True
    )
    
    for r in result:
        if isinstance(r, Exception):
            print(f"[amain] failed - task raised {r}")

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(amain(loop))
    finally:
        print("[MAIN] trying to shut down ")
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(asyncio.sleep(1.0))
        loop.close()
        print("[MAIN] loop closed")

if __name__ == "__main__":
    main()