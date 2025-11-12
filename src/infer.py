import argparse, json, pathlib
from utils import load_model, load_meta, predict

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", default="../models/best_model.keras")
parser.add_argument("--meta",  default="../models/model_meta.json")
args = parser.parse_args()

m = load_model(args.model)
meta = load_meta(args.meta)
res = predict(m, meta["classes"], args.image, tuple(meta["img_size"]))
print(json.dumps(res, indent=2))