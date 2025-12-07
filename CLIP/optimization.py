import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from pytorch_wavelets import DWTForward
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
from torchvision.utils import save_image

mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]

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

# -------------------- HighFreqLossGPU --------------------
class MultiScaleHF_Loss(nn.Module):
    """
    Multi-scale high-frequency loss using pytorch_wavelets DWT.
    - Supports multi-layer decomposition
    - Combines all high-frequency subbands (LH, HL, HH) at each scale
    - Can weight different scales
    """
    def __init__(self, J=1, wave='db3', mode='zero', scale_weights=None):
        """
        Args:
            J : int, decomposition levels
            wave : str, wavelet type
            mode : str, boundary mode
            scale_weights : list of floats, weight for each DWT level (len=J)
        """
        super(MultiScaleHF_Loss, self).__init__()
        self.J = J
        self.xfm = DWTForward(J=J, wave=wave, mode=mode)
        self.mse = nn.MSELoss()
        if scale_weights is None:
            self.scale_weights = [1.0]*J
        else:
            assert len(scale_weights) == J
            self.scale_weights = scale_weights

    def forward(self, generated, target):
        """
        Args:
            generated : [B, C, H, W]
            target    : [B, C, H, W]
        Returns:
            scalar loss
        """
        # DWT decomposition
        Yl_g, Yh_g = self.xfm(generated)
        Yl_t, Yh_t = self.xfm(target)

        total_loss = 0.0
        for j in range(self.J):
            HF_g = torch.abs(Yh_g[j]).sum(dim=2) 
            HF_t = torch.abs(Yh_t[j]).sum(dim=2)
            
            level_loss = self.mse(HF_g, HF_t)
            total_loss += self.scale_weights[j] * level_loss

        return total_loss

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean

if __name__ == '__main__':
    # ----------------------------------------
    max_watermark_strength = 0.05
    hf_loss_weight = 5



    state_dict = torch.hub.load_state_dict_from_url('https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip' , map_location = "cpu", progress = True )
    model      = C2P_CLIP( name='openai/clip-vit-large-patch14', num_classes=1 )
    model.load_state_dict(state_dict, strict=True);model.cuda(); model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # LPIPS and HighFreqLoss
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    # Multi-scale high-frequency loss
    hf_fn = MultiScaleHF_Loss(J=3, wave='db3', scale_weights=[1.0, 1.0 , 0.5]).to(device)

    transform = transforms.Compose([
        transforms.Resize( 224 ),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])


    # transform_full = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
    #                      std=[0.26862954, 0.26130258, 0.27577711]),
    # ])
    img = Image.open('1.jpg').convert("RGB")
    img_tensor_full =  transforms.ToTensor()(img).to(device)


    watermark = torch.randn_like(img_tensor_full, requires_grad=True, device=device)
    optimizer = optim.Adam([watermark], lr=0.1)

    for step in range(100):
        optimizer.zero_grad()
        wm = max_watermark_strength * torch.tanh(watermark)
        watermarked_full = img_tensor_full + wm

        watermarked_224 = transform(watermarked_full).unsqueeze(0)

        preds = model(watermarked_224).sigmoid().flatten()
        loss = -preds.mean()
        
        if(step % 10 == 0):
            print(f"step {step}: preds={preds.item()}")
        
        loss.backward()
        optimizer.step()

    to_pil = transforms.ToPILImage()
    img_pil = to_pil(watermarked_full.squeeze(0).cpu().clamp(0,1))  
    img_pil.save("watermarked11.png")
