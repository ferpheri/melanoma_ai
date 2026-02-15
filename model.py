import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import gdown

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Google Drive model URLs (replace with your file IDs)
MODEL_URLS = {
    "mobilenet": "https://drive.google.com/uc?id=1s1hpAVDJtRNJNrkLfcHEPTLzZVXrOqQr",
    "resnet": "https://drive.google.com/uc?id=1GBXi0aEIp2wK9vtbPNGCwk9ELcXNE8VL",
    "efficientnet": "https://drive.google.com/uc?id=1Vc-VjmK2MV3f40fGBxaaeMPrgt8CdOeU"
}


def download_model(model_name):
    """Download model from Google Drive if not present locally."""
    path = os.path.join(BASE_DIR, f"{model_name}_model.pt")
    if not os.path.exists(path):
        print(f"[INFO] Downloading {model_name} model from Google Drive...")
        gdown.download(MODEL_URLS[model_name], path, quiet=False)
    return path

def load_model(model_type):
    """Load a model given its type."""
    if model_type == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
    elif model_type == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_type == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_path = download_model(model_type)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Pre-load models into memory
models_dict = {
    "mobilenet": load_model("mobilenet"),
    "resnet": load_model("resnet"),
    "efficientnet": load_model("efficientnet")
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image, model_name: str) -> float:
    """
    Predict melanoma probability for a given PIL image and model.
    
    Args:
        image: PIL.Image object (RGB)
        model_name: 'mobilenet', 'resnet', or 'efficientnet'
        
    Returns:
        probability of melanoma (0-1)
    """
    model = models_dict[model_name]
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        melanoma_prob = probs[0][1].item()  # index 1 = melanoma
    return melanoma_prob