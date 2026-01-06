import io
import os
import json
import base64
import torch as T

from numpy import array as np_array
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from torchvision import transforms

from backbones import get_model
from face_alignment import mtcnn

# TorchServe Handler
class ModelHandler(BaseHandler):
    
    def initialize(self, context):
        
        self.model_name = "edgeface_s_gamma_05" # or edgeface_xs_gamma_06
        self.max_number_of_faces = 1
        self.device = "cuda:0"
        self.crop_size = (112, 112)
        
        assert self.model_name in ["edgeface_s_gamma_05", "edgeface_xs_gamma_06"], "model_name_ should be edgeface_s_gamma_05 or edgeface_xs_gamma_06"
        assert f"{self.model_name}.pt" in os.listdir("checkpoints"), f"{self.model_name}.pt not found in ./checkpoints"
        assert self.device in ["cuda:0", "cuda:1", "cuda", "cpu"], "device should be cuda or cpu"
        assert self.max_number_of_faces > 0, "max_number_of_faces should be greater than 0"
        assert len(self.crop_size) == 2, "crop_size should be a tuple of 2 integers"
        
        # load mtcnn model
        self.mtcnn_model = mtcnn.MTCNN(device=self.device, crop_size=self.crop_size)
        
        # load face recognition model
        self.model = get_model(self.model_name)
        self.model.load_state_dict(T.load(f"./checkpoints/{self.model_name}.pt"))
        self.model.eval()
        self.model.to(self.device)
        
        # preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
    def preprocess(self, data):
        data = data[0]["body"]["data"]
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        image = image.convert("RGB")
        image = image.resize(self.crop_size, Image.BILINEAR)
        return image

    def inference(self, image):
        with T.no_grad():
            try:
                _, faces = self.mtcnn_model.align_multi(image, limit=1)
            except Exception as e:
                raise e
            
            print(f"Number of faces detected: {len(faces)}")
            if len(faces) == 0:
                raise Exception("No faces detected")
            
            try:
                faces = T.stack([self.transform(face) for face in faces][:self.max_number_of_faces], dim=0)
                results = self.model(faces)
            except Exception as e:
                raise e
            
            return results
        
    def postprocess(self, data):
        return data.detach().to("cpu").numpy().tolist()

