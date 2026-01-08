import sys
import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
SRC_PATH = "/kaggle/input/retinova-1/Retina_ai/src"
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from fusion_model import FusionEffNetTabular

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# PRS PROXY (ASCENDING, DEMO)
# -------------------------------------------------
def get_prs_proxy(ml_risk: float):
    if ml_risk < 0.30:
        return "Low"
    elif ml_risk < 0.60:
        return "Moderate"
    else:
        return "High"

# -------------------------------------------------
# FRS POST-PROCESSING
# -------------------------------------------------
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

# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# MODEL LOADER
# -------------------------------------------------
def load_model(weights_path):
    model = FusionEffNetTabular(pretrained=True)
    ckpt = torch.load(weights_path, map_location=DEVICE)

    filtered_state = {
        k: v for k, v in ckpt["model_state"].items()
        if not k.startswith("net.")
    }

    model.load_state_dict(filtered_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Retinova CVD Inference CLI")

    parser.add_argument("--image", required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", type=int, choices=[0, 1], required=True)

    parser.add_argument("--smoker", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--diabetes", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--sbp", type=int, required=True)
    parser.add_argument("--bp_meds", type=lambda x: x.lower() == "true", required=True)

    parser.add_argument(
        "--weights",
        default="/kaggle/input/retinova-1/Retina_ai/models/baseline_best.pth"
    )

    args = parser.parse_args()

    model = load_model(args.weights)

    img = Image.open(args.image).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    tab = torch.tensor([[args.age, args.sex]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logit = model(img, tab)
        base_ml_risk = torch.sigmoid(logit).item()

    prs_proxy = get_prs_proxy(base_ml_risk)

    final_risk, explanations = apply_frs(
        base_ml_risk,
        args.age,
        args.smoker,
        args.diabetes,
        args.sbp,
        args.bp_meds
    )

    output = {
        "base_ml_risk": round(base_ml_risk, 3),
        "prs_proxy": prs_proxy,
        "final_adjusted_risk": round(final_risk, 3),
        "risk_delta": round(final_risk - base_ml_risk, 3),
        "clinical_explanations": explanations
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
