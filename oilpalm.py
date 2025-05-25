import streamlit as st
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from collections import Counter
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections

# ------------------------#
# Konfigurasi halaman
# ------------------------#
st.set_page_config(page_title="Deteksi Buah Sawit", layout="wide")
st.title("🌴 Deteksi dan Klasifikasi Kematangan Buah Sawit")
st.markdown("Gunakan model YOLOv8 untuk mendeteksi dan mengklasifikasikan buah sawit berdasarkan tingkat kematangannya.")

# ------------------------#
# Load model
# ------------------------#
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Pastikan best.pt sudah ada di direktori yang sama

model = load_model()

# ------------------------#
# Warna label dan anotator
# ------------------------#
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}
label_annotator = LabelAnnotator()

# ------------------------#
# Fungsi prediksi & anotasi
# ------------------------#
def predict_image(image: Image.Image):
    img_array = np.array(image.convert("RGB"))
    results = model(img_array)
    return results

def draw_results(image: Image.Image, results):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    for result in results:
        boxes = result.boxes
        names = result.names
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(xyxy, class_ids, confidences):
            class_name = names[class_id]
            label = f"{class_name}: {conf:.2f}"
            color = label_to_color.get(class_name, Color.WHITE)
            class_counts[class_name] += 1

            detection = Detections(
                xyxy=np.array([box]),
                confidence=np.array([conf]),
                class_id=np.array([class_id])
            )

            box_annotator = BoxAnnotator(color=color)
            img = box_annotator.annotate(scene=img, detections=detection)
            img = label_annotator.annotate(scene=img, detections=detection, labels=[label])

    return img, class_counts

# ------------------------#
# Input Gambar
# ------------------------#
col1, col2 = st.columns(2)
with col1:
    input_method = st.radio("Pilih metode input:", ["📁 Upload Gambar", "📷 Kamera"], index=0)

image = None

if input_method == "📁 Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📥 Gambar diunggah", use_column_width=True)

elif input_method == "📷 Kamera":
    st.markdown("### Kamera Environment (Belakang)")
    st.components.v1.html(
        """
        <div style="text-align:center;">
            <video id="video" autoplay playsinline style="width:100%; border:1px solid gray;"></video>
            <br/>
            <button onclick="takePhoto()" style="margin-top:10px; padding:10px 20px;">📸 Ambil Gambar</button>
            <canvas id="canvas" style="display:none;"></canvas>
        </div>

        <script>
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" } },
                    audio: false
                });
                document.getElementById('video').srcObject = stream;
            } catch (err) {
                alert("Gagal mengakses kamera: " + err.message);
            }
        }

        function takePhoto() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const dataURL = canvas.toDataURL('image/png');
            const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
            input.value = dataURL;
            input.dispatchEvent(new Event("input", { bubbles: true }));
        }

        document.addEventListener("DOMContentLoaded", startCamera);
        </script>
        """, height=600
    )

    base64_img = st.text_input("Gambar dari Kamera", label_visibility="collapsed")
    if base64_img.startswith("data:image"):
        try:
            header, encoded = base64_img.split(",", 1)
            decoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(decoded))
            st.image(image, caption="📸 Gambar dari Kamera", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Gagal memproses gambar: {e}")

# ------------------------#
# Proses Deteksi
# ------------------------#
if image:
    with st.spinner("🚀 Memproses gambar..."):
        results = predict_image(image)
        img_out, class_counts = draw_results(image, results)

        st.image(img_out, caption="✅ Hasil Deteksi", use_column_width=True)
        st.markdown("### 📊 Ringkasan Deteksi:")
        for name, count in class_counts.items():
            st.markdown(f"- **{name}**: {count}")
