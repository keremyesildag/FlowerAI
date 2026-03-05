import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go

# Sayfa ayarları
st.set_page_config(page_title="🌸 Çiçek Tanıyıcı", layout="wide")

# Model ve sınıfları yükle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cicek_modeli.keras")
    with open("sinif_isimleri.json") as f:
        siniflar = json.load(f)
    return model, siniflar

model, siniflar = load_model()

# Başlık
st.title("🌸 Çiçek Tanıyıcı AI")
st.write("Kendi eğittiğim model ile çiçek türünü tahmin ediyor!")

# İki kolon
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Fotoğraf Yükle")
    uploaded = st.file_uploader("Bir çiçek fotoğrafı seç", type=["jpg","jpeg","png"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Yüklenen fotoğraf", use_column_width=True)

with col2:
    if uploaded:
        st.subheader("🤖 AI Analizi")

        # Tahmin
        img = image.convert("RGB").resize((96, 96))
        arr = tf.keras.utils.img_to_array(img)
        arr = tf.expand_dims(arr, 0)

        predictions = model.predict(arr, verbose=0)[0]
        top_idx = np.argmax(predictions)
        top_sinif = siniflar[top_idx]
        top_skor = predictions[top_idx]

        # Sonuç kutusu
        st.success(f"✅ Tahmin: **{top_sinif.upper()}**")
        st.metric("Güven Oranı", f"%{top_skor*100:.1f}")

        # Tüm sınıflar için bar chart
        st.subheader("📊 Tüm Tahminler")
        fig = go.Figure(go.Bar(
            x=[f"%{p*100:.1f}" for p in predictions],
            y=siniflar,
            orientation="h",
            marker_color=["#FF4B4B" if i == top_idx else "#4B9EFF"
                         for i in range(len(siniflar))]
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Güven Oranı"
        )
        st.plotly_chart(fig, use_container_width=True)

# Model bilgileri
st.divider()
st.subheader("🧠 Model Hakkında")
m1, m2, m3 = st.columns(3)
m1.metric("Mimari", "MobileNetV2")
m2.metric("Sınıf Sayısı", len(siniflar))
m3.metric("Görsel Boyutu", "96x96")