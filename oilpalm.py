import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
from io import BytesIO
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Buah Sawit", layout="wide")

# Foto heading (pastikan file foto_heading.jpg tersedia di folder ini)
st.image("Buah-Kelapa-Sawit.jpg", width=150, caption="Deteksi Buah Sawit - by Team", use_column_width=False)

# Sidebar pengaturan
with st.sidebar:
    st.title("⚙️ Pengaturan")
    input_method = st.radio("Metode Input Gambar", ["📁 Upload", "📷 Kamera"])

# Load model (cache agar tidak dimuat ulang)
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ganti path model jika berbeda

# Warna bounding box sesuai label
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}
label_annotator = LabelAnnotator()

# Fungsi prediksi gambar
def predict_image(model, image):
    image = np.array(image.convert("RGB"))
    results = model(image)
    return results

# Fungsi menggambar hasil deteksi
def draw_results(image, results):
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

            box_annotator = BoxAnnotator(color=color)
            detection = Detections(
                xyxy=np.array([box]),
                confidence=np.array([conf]),
                class_id=np.array([class_id])
            )

            img = box_annotator.annotate(scene=img, detections=detection)
            img = label_annotator.annotate(scene=img, detections=detection, labels=[label])

    return img, class_counts

# Inisialisasi session state untuk kamera
if "camera_image" not in st.session_state:
    st.session_state["camera_image"] = ""

# Judul aplikasi
st.title("🌴 Deteksi dan Klasifikasi Kematangan Buah Sawit")

image = None

# Upload Gambar
if input_method == "📁 Upload":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Gambar yang Diunggah", use_container_width=True)

# Kamera Langsung
elif input_method == "📷 Kamera":
    st.markdown("### Kamera Belakang (Environment)")

    camera_html = """
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
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/png');
            const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
            if (input) {
                input.value = dataURL;
                input.dispatchEvent(new Event("input", { bubbles: true }));
            }
        }
        document.addEventListener("DOMContentLoaded", startCamera);
    </script>
    """
    st.components.v1.html(camera_html, height=600)
    base64_img = st.text_input("Gambar dari Kamera", type="default", label_visibility="collapsed")

    if base64_img.startswith("data:image"):
        st.session_state["camera_image"] = base64_img
        try:
            header, encoded = base64_img.split(",", 1)
            decoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(decoded))
            st.image(image, caption="📷 Gambar dari Kamera", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memproses gambar: {e}")

# Proses Deteksi
if image:
    with st.spinner("🔍 Memproses gambar..."):
        model = load_model()
        results = predict_image(model, image)
        img_out, class_counts = draw_results(image, results)
        st.image(img_out, caption="📊 Hasil Deteksi", use_container_width=True)

        st.subheader("Jumlah Objek Terdeteksi:")
        cols = st.columns(len(class_counts))
        for i, (label, count) in enumerate(class_counts.items()):
            with cols[i]:
                st.metric(label=label, value=count)

        if st.button("💾 Unduh Hasil Deteksi"):
    # Simpan ke objek memori
    img_pil = Image.fromarray(img_out)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Klik untuk Mengunduh Gambar",
        data=byte_im,
        file_name=f"hasil_deteksi_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
        mime="image/png"
    )

