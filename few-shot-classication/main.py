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

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

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
    input = transforms(frame).unsqueeze(0)
    return model(input)

def camera_thread(frame_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    from webcam import Webcam

    webcam = Webcam(src=0, w=640)
    print("[camera] started")

    def put(frame):
        asyncio.run_coroutine_threadsafe(frame_queue.put(frame), loop)

    try:
        for frame in webcam:
            if frame_queue.full():
                # non-blocking drop of oldest frame to avoid unbounded growth
                try:
                    frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            put(frame)
    except Exception as e:
        print(f"[camera] error: {e}")
    finally:
        # sentinel
        put(None)
        webcam.release()
        print("[camera] stopped")


async def camera_task(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    print("[camera] launching thread...")
    await loop.run_in_executor(None, camera_thread, queue, loop)


async def inference_task(q: asyncio.Queue, loop: asyncio.AbstractEventLoop):

    try:
        model, transform = await loop.run_in_executor(None, _init_model)
    except Exception as e:
        print("[inference] failed : model initalization")
        return

    print("[inference] start")
    try:
        while True:
            frame = await q.get()
            if frame is None:
                break

            infer = partial(_run_inference, model, transform, frame)
            try:
                features = await loop.run_in_executor(None, infer)
                # TODO: route features into image_vector_collections
                print(f"[inference] shape={features.shape} mean={features.mean():.4f}")
            except Exception as e:
                print(f"[inference] failed - frame: {e}")

            # cv2.imshow("webcam frames", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"[INFERENCE] failed - main loop: {e}")


async def amain(loop: asyncio.AbstractEventLoop):
    
    queue = asyncio.Queue(maxsize=2)
    
    cam = asyncio.create_task(camera_task(queue, loop))
    inference = asyncio.create_task(inference_task(queue, loop))
    
    result = await asyncio.gather(cam, inference, return_exceptions=True)
    for r in result:
        if isinstance(r, Exception):
            print(f"[amain] failed - task raised {r}")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(amain(loop))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == "__main__":
    main()