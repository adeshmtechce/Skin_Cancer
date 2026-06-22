import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import DenseNet121_Weights

# ==========================
# CONFIG
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
import os
import gdown

MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(
        id="1CB0gMpfLF09Rulvl6qxUBDL7sD72XA1R",
        output=MODEL_PATH,
        quiet=False
    )
else:
    print(f"Model already exists: {MODEL_PATH}")

# ==========================
# MODEL
# ==========================
@st.cache_resource
def load_model():

    model = models.densenet121(weights=None)

    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, len(CLASS_NAMES))
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# ==========================
# PREPROCESS
# ==========================
weights = DenseNet121_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# ==========================
# TITLE
# ==========================
st.title("Skin Disease Classification with Grad-CAM")

st.write("Upload a skin lesion image to classify.")

uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # ======================
    # PREDICTION
    # ======================
    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_idx = predicted_class.item()

    st.success(
        f"Prediction: {CLASS_NAMES[predicted_idx]}"
    )

    st.info(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    # ======================
    # GRAD-CAM
    # ======================
    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    rgb_img = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    targets = [
        ClassifierOutputTarget(predicted_idx)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # ======================
    # DISPLAY
    # ======================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image.resize((224, 224)),
            use_container_width=True
        )

    with col2:
        st.subheader("Grad-CAM")
        st.image(
            visualization,
            use_container_width=True
        )

    # ======================
    # CLASS PROBABILITIES
    # ======================
    st.subheader("Class Probabilities")

    probs = probabilities[0].cpu().numpy()

    for cls, prob in zip(CLASS_NAMES, probs):
        st.write(f"{cls}: {prob*100:.2f}%")
