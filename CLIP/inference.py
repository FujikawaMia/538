import argparse
import sys
import time
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from data import create_dataloader
import random
from transformers import CLIPModel
import warnings
warnings.filterwarnings('ignore')
import torch
from PIL import Image
from torchvision import transforms

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(123)

class C2P_CLIP(nn.Module):
    def __init__(self, name='openai/clip-vit-large-patch14', num_classes=1):
        super(C2P_CLIP, self).__init__()
        self.model        = CLIPModel.from_pretrained(name)
        del self.model.text_model
        del self.model.text_projection
        del self.model.logit_scale
        
        self.model.vision_model.requires_grad_(True)
        self.model.visual_projection.requires_grad_(True)
        self.model.fc = nn.Linear( 768, num_classes )
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)

    def encode_image(self, img):
        vision_outputs = self.model.vision_model(
            pixel_values=img,
            output_attentions    = self.model.config.output_attentions,
            output_hidden_states = self.model.config.output_hidden_states,  
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features    

    def forward(self, img):
        # tmp = x; print(f'x: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}, mean: {tmp.mean()}')
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return self.model.fc(image_embeds)

if __name__ == '__main__':
    # state_dict = torch.hub._legacy_zip_load( 'C2P_CLIP_release_20240901.zip', './', map_location = "cpu", weights_only= False)
    state_dict = torch.hub.load_state_dict_from_url('https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip' , map_location = "cpu", progress = True )
    model      = C2P_CLIP( name='openai/clip-vit-large-patch14', num_classes=1 )
    model.load_state_dict(state_dict, strict=True);model.cuda(); model.eval()



    transform = transforms.Compose([
        transforms.Resize( 224 ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    img = Image.open('mixed_hybrid222.png').convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    #C2P-CLIP-DeepfakeDetection

    with torch.no_grad():
        y_pred = []
        preds = model(img_tensor.cuda()).sigmoid().flatten().tolist()
        y_pred.extend(preds)

    y_pred = np.array(y_pred)

    for p, in zip(y_pred):
        print(f"pred={p:.4f}")

