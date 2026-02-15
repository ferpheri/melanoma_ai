import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        path = "efficientnet_b0_model.pt"
    
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Pre-load models to memory for fast inference
models_dict = {
    "mobilenet": load_model("mobilenet"),
    "resnet": load_model("resnet"),
    "efficientnet": load_model("efficientnet")
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image, model_name):
    if model_name not in models_dict:
        raise ValueError("Invalid model selected")

    model = models_dict[model_name]
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        # Assuming index 1 is 'Melanoma'
        melanoma_prob = probs[0][1].item()
    return melanoma_prob