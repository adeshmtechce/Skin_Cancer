import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ViT_B_16_Weights
from PIL import Image
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os 
import gdown 
MODEL_PATH = "model.pth" 
if not os.path.exists(MODEL_PATH): 
  print("Downloading model...") 
  gdown.download( id="1wpEyrRSdNxOzb5qAVnWUjlvU-A6LEqwy", output=MODEL_PATH, quiet=False ) 
else: 
  print(f"Model already exists: {MODEL_PATH}")

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanocytic Nevus",
    "Melanoma",
    "Vascular Lesion",
    "Warts_Molluscum"
]

# ==========================================
# SeparableConv2D
# ==========================================
class SeparableConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ==========================================
# DenseNet Feature Extractor
# ==========================================
class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        base = torchvision.models.densenet121(weights=None)

        self.features = base.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_features = base.classifier.in_features

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x


# ==========================================
# Hybrid Model
# ==========================================
class HybridViTDensNet(nn.Module):
    def __init__(self, vit_model, dens_model, num_classes=8):

        super().__init__()

        self.vit = vit_model

        self.dens = dens_model

        vit_feat_dim = 768
        dens_feat_dim = dens_model.out_features

        fusion_dim = vit_feat_dim + dens_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_classes)
        )

    def forward(self, x):

        vit_feats = self.vit(x)

        dens_feats = self.dens(x)

        fused = torch.cat(
            [vit_feats, dens_feats],
            dim=1
        )

        return self.classifier(fused)


# ==========================================
# BUILD MODEL
# ==========================================
def build_model():

    vit = torchvision.models.vit_b_16(
        weights=None
    )

    orig_proj = vit.conv_proj

    vit.conv_proj = SeparableConv2D(
        in_channels=orig_proj.in_channels,
        out_channels=orig_proj.out_channels,
        kernel_size=orig_proj.kernel_size,
        stride=orig_proj.stride,
        padding=orig_proj.padding,
        bias=(orig_proj.bias is not None)
    )

    vit.heads = nn.Identity()

    densnet = DenseNetFeatureExtractor()

    model = HybridViTDensNet(
        vit,
        densnet,
        num_classes=8
    )

    return model


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():

    model = build_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model


# ==========================================
# LOAD MODEL
# ==========================================
model = load_model()

weights = ViT_B_16_Weights.IMAGENET1K_V1
transform = weights.transforms()

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="Skin Disease Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Skin Disease Classification")
st.markdown(
    "Upload a skin lesion image and the model will predict the disease category."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    pred_idx = pred.item()

    with col2:

        st.success(
            f"Prediction: {CLASS_NAMES[pred_idx]}"
        )

        st.info(
            f"Confidence: {confidence.item()*100:.2f}%"
        )

    st.subheader("Class Probabilities")

    prob_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probs[0].cpu().numpy()
    })

    st.bar_chart(
        prob_df.set_index("Class")
    )

    st.dataframe(
        prob_df.style.format({
            "Probability": "{:.4f}"
        }),
        use_container_width=True
    )
