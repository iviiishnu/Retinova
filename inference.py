# inference.py
import json
import torch
from PIL import Image
from torchvision import transforms

from src.fusion_model import FusionEffNetTabular

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Risk helpers
# -------------------------------
def get_prs_proxy(ml_risk: float):
    if ml_risk < 0.30:
        return "Low"
    elif ml_risk < 0.60:
        return "Moderate"
    else:
        return "High"


def apply_frs(base_ml_risk, age, smoker, diabetes, sbp, bp_meds):
    risk = base_ml_risk
    explanations = []

    if smoker:
        risk += 0.08
        explanations.append("Smoking increases cardiovascular risk")

    if diabetes:
        risk += 0.10
        explanations.append("Diabetes increases cardiovascular risk")

    if sbp >= 140:
        risk += 0.07
        explanations.append("High systolic blood pressure increases risk")

    if bp_meds:
        risk += 0.03
        explanations.append("Use of blood pressure medication indicates elevated risk")

    if age >= 55:
        risk += 0.05
        explanations.append("Age above 55 increases baseline risk")

    return min(risk, 0.99), explanations


# -------------------------------
# Image transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -------------------------------
# Model loader
# -------------------------------
def load_model(weights_path: str):
    model = FusionEffNetTabular(pretrained=True)

    ckpt = torch.load(weights_path, map_location=DEVICE)

    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# -------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------
def run_inference(
    image_path: str,
    age: int,
    sex: int,
    smoker: bool,
    diabetes: bool,
    sbp: int,
    bp_meds: bool,
    weights_path: str = "models/baseline_best.pth",
):
    model = load_model(weights_path)

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    tabular = torch.tensor([[age, sex]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logit = model(img, tabular)
        base_ml_risk = torch.sigmoid(logit).item()

    prs_proxy = get_prs_proxy(base_ml_risk)

    final_risk, explanations = apply_frs(
        base_ml_risk,
        age,
        smoker,
        diabetes,
        sbp,
        bp_meds,
    )

    return {
        "base_ml_risk": round(base_ml_risk, 3),
        "prs_proxy": prs_proxy,
        "final_adjusted_risk": round(final_risk, 3),
        "risk_delta": round(final_risk - base_ml_risk, 3),
        "clinical_explanations": explanations,
    }