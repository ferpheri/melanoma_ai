import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gdown

# =========================
# Device configuration
# =========================
DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# Google Drive model files
# =========================
MODEL_FILES = {
    "mobilenet_v2_model.pt": "1s1hpAVDJtRNJNrkLfcHEPTLzZVXrOqQr",
    "resnet18_model.pt": "1GBXi0aEIp2wK9vtbPNGCwk9ELcXNE8VL",
    "efficientnet_b0_model.pt": "1Vc-VjmK2MV3f40fGBxaaeMPrgt8CdOeU",
}

def ensure_models_exist():
    """
    Download model files from Google Drive if they do not exist locally.
    This runs ONCE when the app starts.
    """
    for filename, file_id in MODEL_FILES.items():
        file_path = os.path.join(BASE_DIR, filename)

        if not os.path.exists(file_path):
            print(f"[INFO] Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"[INFO] {filename} already exists. Skipping download.")

# Run once at startup
ensure_models_exist()

# =========================
# Model loader
# =========================
def load_model(model_type):
    if model_type == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        path = "mobilenet_v2_model.pt"

    elif model_type == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        path = "resnet18_model.pt"

    elif model_type == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 2
        )
        path = "efficientnet_b0_model.pt"

    else:
        raise ValueError("Invalid model type")

    model_path = os.path.join(BASE_DIR, path)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# Pre-load models (fast inference)
# =========================
models_dict = {
    "mobilenet": load_model("mobilenet"),
    "resnet": load_model("resnet"),
    "efficientnet": load_model("efficientnet"),
}

# =========================
# Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Prediction function
# =========================
def predict(image: Image.Image, model_name: str) -> float:
    if model_name not in models_dict:
        raise ValueError("Invalid model selected")

    model = models_dict[model_name]
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        melanoma_prob = probs[0, 1].item()  # index 1 = melanoma

    return melanoma_prob
