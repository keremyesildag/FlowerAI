import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import json
import pathlib
import os

GORSEL_BOYUTU = (96, 96)
BATCH_SIZE = 32
EPOCH = 10

# Otomatik indir — 218MB, 5 çiçek sınıfı
print("📂 Veri seti indiriliyor...")
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
veri_yolu = pathlib.Path(r"C:\Users\Prime\.keras\datasets\flower_photos\flower_photos")
veri_yolu = pathlib.Path(veri_yolu)

print("📂 Veri seti yükleniyor...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    veri_yolu,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=GORSEL_BOYUTU,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    veri_yolu,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=GORSEL_BOYUTU,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

sinif_isimleri = train_ds.class_names
print(f"✅ Sınıflar: {sinif_isimleri}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("🧠 Model oluşturuluyor...")
base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(sinif_isimleri), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Eğitim başlıyor...")
class EgitimTakip(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n{'='*50}")
        print(f"🚀 Epoch {epoch+1}/{EPOCH} başlıyor...")
        print(f"{'='*50}")

    def on_batch_end(self, batch, logs=None):
        print(f"  📸 Batch {batch+1} | "
              f"Loss: {logs['loss']:.4f} | "
              f"Doğruluk: %{logs['accuracy']*100:.1f}", end="\r")

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n✅ Epoch {epoch+1} bitti!")
        print(f"  🎯 Eğitim Doğruluğu : %{logs['accuracy']*100:.2f}")
        print(f"  🧪 Test Doğruluğu   : %{logs['val_accuracy']*100:.2f}")
        print(f"  📉 Eğitim Loss      : {logs['loss']:.4f}")
        print(f"  📉 Test Loss        : {logs['val_loss']:.4f}")
        if logs['val_accuracy'] > 0.90:
            print(f"  🔥 Harika! %90 üzeri doğruluk!")
        elif logs['val_accuracy'] > 0.75:
            print(f"  👍 İyi gidiyor!")
        else:
            print(f"  ⏳ Model hâlâ öğreniyor...")

model.fit(
    train_ds,
    epochs=EPOCH,
    validation_data=test_ds,
    callbacks=[EgitimTakip()],
    verbose=0
)

model.save("cicek_modeli.keras")
with open("sinif_isimleri.json", "w", encoding="utf-8") as f:
    json.dump(sinif_isimleri, f, ensure_ascii=False, indent=2)

loss, acc = model.evaluate(test_ds)
print(f"🎯 Test Doğruluğu: %{acc*100:.2f}")
print("✅ Model kaydedildi!")