import argparse
from PIL import Image

from src.inference.bundle import load_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True, help="Path to model bundle directory")
    ap.add_argument("--image", required=True, help="Path to image file")
    ap.add_argument("--device", default=None, help="Device override (cpu, cuda, mps)")
    args = ap.parse_args()

    predictor = load_bundle(args.bundle_dir, device=args.device)
    img = Image.open(args.image).convert("RGB")
    result = predictor.predict_pil(img)

    print(f"Prediction : {result['pred_label']}")
    print(f"Confidence : {result['positive_prob']:.4f}")
    print(f"Is minifig : {result['is_positive']}")
    print(f"All probs  : {result['probs']}")


if __name__ == "__main__":
    main()
