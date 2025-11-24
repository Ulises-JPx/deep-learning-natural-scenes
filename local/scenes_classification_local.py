import pathlib
import zipfile
from pathlib import Path
import shutil
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ================================================================
# 1) Configuración de rutas para entorno local
# ================================================================
# Carpetas de trabajo (Algunas descartadas por GitHub ya que no se pudo probar este código)
BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "intel_scenes"
MODELS_DIR = BASE_DIR / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2) Descomprimir dataset local
# ================================================================
zip_path = RAW_DIR / "intel-image-classification.zip"
extract_path = RAW_DIR / "intel_raw"
extract_path.mkdir(parents=True, exist_ok=True)

if not zip_path.exists():
    print("ERROR: No se encontró el dataset intel-image-classification.zip.")
    print("Coloca el archivo en data/raw/")
    exit()

print("Descomprimiendo dataset...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_path)

print("Dataset descomprimido en:", extract_path)


# ================================================================
# 3) Crear splits train / val / test
# ================================================================
TRAIN_DST = PROCESSED_DIR / "train"
VAL_DST = PROCESSED_DIR / "val"
TEST_DST = PROCESSED_DIR / "test"

for d in [TRAIN_DST, VAL_DST, TEST_DST]:
    d.mkdir(parents=True, exist_ok=True)

root = extract_path

seg_train = root / "seg_train"
if (seg_train / "seg_train").exists():
    seg_train = seg_train / "seg_train"

seg_test = root / "seg_test"
if (seg_test / "seg_test").exists():
    seg_test = seg_test / "seg_test"

classes = sorted([p.name for p in seg_train.iterdir() if p.is_dir()])
print("Clases detectadas:", classes)

# Copiar test completo
for cls in classes:
    src = seg_test / cls
    dst = TEST_DST / cls
    dst.mkdir(exist_ok=True)
    for img in src.glob("*.*"):
        shutil.copy2(img, dst / img.name)

# Split 80/20
random.seed(42)
for cls in classes:
    src = seg_train / cls
    imgs = [p for p in src.glob("*.*") if p.is_file()]
    random.shuffle(imgs)

    cut = int(0.8 * len(imgs))
    train_imgs = imgs[:cut]
    val_imgs = imgs[cut:]

    (TRAIN_DST / cls).mkdir(exist_ok=True)
    (VAL_DST / cls).mkdir(exist_ok=True)

    for img in train_imgs:
        shutil.copy2(img, TRAIN_DST / cls / img.name)
    for img in val_imgs:
        shutil.copy2(img, VAL_DST / cls / img.name)

print("Split train/val/test completado.")


# ================================================================
# 4) Cargar datos con TensorFlow
# ================================================================
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DST, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DST, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DST, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Clases:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)


# ================================================================
# 5) Modelo EfficientNetB0 con data augmentation
# ================================================================
print("Construyendo modelo EfficientNetB0...")

data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
)
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augment(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()


# ================================================================
# 6) Entrenamiento fase 1 (base congelada)
# ================================================================
BEST_PATH = MODELS_DIR / "intel_efficientnet_best.keras"

callbacks = [
    ModelCheckpoint(str(BEST_PATH), monitor="val_accuracy", save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5)
]

print("Entrenando fase 1...")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)


# ================================================================
# 7) Fine-tuning (descongelar capas superiores)
# ================================================================
print("Fine-tuning fase 2...")

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)


# ================================================================
# 8) Evaluación en test
# ================================================================
print("Evaluando modelo en test...")

best = tf.keras.models.load_model(BEST_PATH)
loss, acc = best.evaluate(test_ds)

print("Accuracy final en test:", acc)


# ================================================================
# 9) Matriz de confusión y reporte
# ================================================================
y_true, y_pred = [], []

for imgs, labels in test_ds:
    probs = best.predict(imgs)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(probs, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.colorbar()
plt.xticks(range(num_classes), class_names, rotation=45)
plt.yticks(range(num_classes), class_names)
plt.tight_layout()
plt.show()


# ================================================================
# 10) Guardar modelo final
# ================================================================
FINAL_PATH = MODELS_DIR / "intel_efficientnet_final.keras"
best.save(FINAL_PATH)
print("Modelo final guardado en:", FINAL_PATH)


# ================================================================
# 11) Inferencia con imagen personalizada
# ================================================================
from tensorflow.keras.preprocessing import image

img_path = "C:\\Users\\Ulises Jaramillo\\Desktop\\TEC-25\\deep-learning-natural-scenes\\local\\images\\bellas_artes.jpeg"

if not Path(img_path).exists():
    print("No se encontró la imagen:", img_path)
else:
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = tf.expand_dims(x, axis=0)

    pred = best.predict(x)[0]
    cls = class_names[np.argmax(pred)]
    conf = float(np.max(pred)) * 100

    print("Predicción:", cls, f"({conf:.2f}%)")

    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(f"{cls} ({conf:.2f}%)")
    plt.show()