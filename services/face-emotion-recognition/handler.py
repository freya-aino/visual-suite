import io
import base64
import torch as T

from PIL import Image
from networks.DDAM import DDAMNet
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler



class ModelServer_AffectNet8(BaseHandler):
    def initialize(
        self, 
        model_path = "./checkpoints_ver2.0/affecnet8_epoch25_acc0.6469.pth", 
        num_head = 2,
        class_names: list = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt'],
        device: str = "cuda:0"):
        
        self.device = device
        
        self.num_head = num_head
        self.num_classes = len(class_names)
        
        self.model = DDAMNet(num_class=self.num_classes, num_head=self.num_head)
        checkpoint = T.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.data_transforms_val = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

    def preprocess(self, data):
        image = data[0].get("data")
        image = Image.open(io.BytesIO(base64.decode(image)))
        image = self.data_transforms_val(image)
        image = T.Tensor(image, device=self.device, dtype=T.float32)
        return image