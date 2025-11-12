import json, pathlib, tensorflow as tf
from tensorflow import keras
from typing import List, Tuple

def load_model(model_path: str):
    return keras.models.load_model(model_path)

def load_meta(meta_path: str):
    with open(meta_path) as f: return json.load(f)

def predict(model, class_names: List[str], img_path: str, img_size=(224,224)) -> List[Tuple[str,float]]:
    img = keras.utils.load_img(img_path, target_size=img_size)
    x = keras.utils.img_to_array(img)
    x = tf.keras.applications.efficientnet.preprocess_input(x[None,...])
    probs = model.predict(x, verbose=0)[0]
    idxs = probs.argsort()[::-1][:3]
    return [(class_names[i], float(probs[i])) for i in idxs]