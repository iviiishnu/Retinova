import argparse
import json
from inference import run_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", type=int, choices=[0, 1], required=True)
    parser.add_argument("--smoker", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--diabetes", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--sbp", type=int, required=True)
    parser.add_argument("--bp_meds", type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--weights", default="models/baseline_best.pth")

    args = parser.parse_args()

    result = run_inference(
        image_path=args.image,
        age=args.age,
        sex=args.sex,
        smoker=args.smoker,
        diabetes=args.diabetes,
        sbp=args.sbp,
        bp_meds=args.bp_meds,
        weights_path=args.weights
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()