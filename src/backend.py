import torch
import torch.nn as nn
from torchvision.models import swin_t, efficientnet_b0
from torchvision import transforms
from huggingface_hub import hf_hub_download

# Cyclone model
class CycloneModel(nn.Module):

    def __init__(self):
        super().__init__()

        # BT image model
        self.swin_stream = swin_t(weights=None)
        self.swin_stream.head = nn.Identity()

        # Raw satellite image model
        self.effnet_stream = efficientnet_b0(weights=None)
        self.effnet_stream.classifier = nn.Identity()

        # Combine both models
        self.fusion_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, bt_image, raw_image):

        bt_features = self.swin_stream(bt_image)
        raw_features = self.effnet_stream(raw_image)

        combined = torch.cat((bt_features, raw_features), dim=1)

        return self.fusion_head(combined)


# Create the model
model = CycloneModel()


# Load the trained model
model_path = hf_hub_download(
    repo_id="AkshatBansal123/SIH26",
    filename="best_two_stream_model_mae0.02_latest.pt"
)
model.load_state_dict(
    torch.load(
        #"model/best_two_stream_model_mae0.02_latest.pt",
        model_path,
        
        map_location="cpu",
        weights_only=True
    )
) 

model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Cyclone classification
def classify_cyclone(msw):

    if msw < 31:
        return "Low Pressure"

    elif msw <= 49:
        return "Depression"

    elif msw <= 61:
        return "Deep Depression"

    elif msw <= 88:
        return "Cyclonic Storm"

    elif msw <= 117:
        return "Severe Cyclonic Storm"

    else:
        return "Very Severe Cyclonic Storm"


# Predict cyclone wind speed
def predict_cyclone(bt_image, raw_image):

    bt_image = transform(bt_image).unsqueeze(0)
    raw_image = transform(raw_image).unsqueeze(0)

    with torch.no_grad():
        output = model(bt_image, raw_image)

    wind_speed = output.item() * 150

    category = classify_cyclone(wind_speed)

    return wind_speed, category


print("Trained model loaded successfully!")