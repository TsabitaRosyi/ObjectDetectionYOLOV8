import streamlit as st
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections
from io import BytesIO
import base64
import tempfile
import plotly.express as px  # ==== TAMBAHAN BARU UNTUK DIAGRAM ====



# =============================
# Fungsi konversi gambar → base64
# =============================
def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# =============================
# Konfigurasi halaman
# =============================
st.set_page_config(page_title="Deteksi Buah Sawit", layout="wide")


# =============================
# Load Model YOLO
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ganti sesuai model kamu

model = load_model()


# =============================
# Warna label
# =============================
label_to_color = {
    "matang": Color.RED,
    "mengkal": Color.YELLOW,
    "mentah": Color.BLACK
}
label_annotator = LabelAnnotator(
    text_scale=5,       #  perbesar ukuran teks (default kecil)
    text_thickness=3,     #  tebalkan huruf
    text_padding=7       #  beri jarak biar tidak mepet box
)



# =============================
# Fungsi anotasi YOLO
# =============================
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

            if class_id not in names:
                continue

            class_name = names[class_id].strip().lower()
            label = f"{class_name}"

            color = label_to_color.get(class_name, Color.WHITE)
            class_counts[class_name] += 1

            box_annotator = BoxAnnotator(
                color=color,
                thickness=3   # KETEBALAN BOUNDING BOX
            )

            detection = Detections(
                xyxy=np.array([box]),
                confidence=np.array([conf]),
                class_id=np.array([class_id])
            )

            img = box_annotator.annotate(
                scene=img,
                detections=detection
            )

            # 🔥 pakai label_annotator GLOBAL (yang text_scale=7)
            img = label_annotator.annotate(
                scene=img,
                detections=detection,
                labels=[label]
            )

    return Image.fromarray(img), class_counts



# =============================
# Fungsi crop foto
# =============================
def crop_center_square(img):
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    return img.crop((left, top, right, bottom))


# =============================
# Load foto profil
# =============================
profile_img = Image.open("foto.jpg")
profile_img = crop_center_square(profile_img)


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.image("logo.png", width=150)
    st.markdown("<h4>Pilih metode input:</h4>", unsafe_allow_html=True)
    option = st.radio("", ["Upload Gambar", "Upload Video"], label_visibility="collapsed")

    st.markdown(
        f"""
        <style>
            .created-by-container {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: 20px;
                padding-top: 10px;
                border-top: 1px solid #ccc;
            }}
            .created-by-img {{
                width: 45px;
                height: 45px;
                border-radius: 50%;
                border: 2px solid #444;
                object-fit: cover;
            }}
            .created-by-text {{
                font-size: 14px;
                color: #555;
                font-style: italic;
            }}
        </style>

        <div class="created-by-container">
            <img class="created-by-img" src="data:image/png;base64,{image_to_base64(profile_img)}" />
            <div class="created-by-text">Created by : Tsabit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================
# Judul Halaman
# =============================
st.markdown("<h1 style='text-align:center;'>🌴 Deteksi Kematangan Buah Sawit</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; font-size:16px; max-width:800px; margin:auto;">
    Sistem ini menggunakan teknologi YOLOv8 untuk mendeteksi kematangan buah kelapa sawit 
    secara otomatis berdasarkan gambar atau video input. 
</div>
""", unsafe_allow_html=True)


# ==========================================================
# ======================= MODE GAMBAR ======================
# ==========================================================
if option == "Upload Gambar":

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        with st.spinner("🔍 Memproses gambar..."):
            results = model(image)
            result_img, class_counts = draw_results(image, results)

        # ======================================================
        # AREA INPUT & OUTPUT
        # ======================================================
        st.markdown("<br>", unsafe_allow_html=True)
        col_input, col_output = st.columns(2)

        with col_input:
            st.markdown("""
            <div style="
                padding:10px;
                height:10px;
                margin-bottom:15px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-weight:bold;
                font-size:20px;">
                AREA INPUT FOTO
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_output:
            st.markdown("""
            <div style="
                padding:10px;
                height:10px;
                margin-bottom:15px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-weight:bold;
                font-size:20px;">
                AREA HASIL FOTO
            </div>
            """, unsafe_allow_html=True)
            st.image(result_img, use_container_width=True)

        # =================== DOWNLOAD ===================
        buf = BytesIO()
        result_img.save(buf, format="PNG")

        st.download_button(
            "⬇️ Download Hasil Deteksi",
            buf.getvalue(),
            "hasil_deteksi.png",
            "image/png"
        )

        # ==================== REKAP DETEKSI ======================
        total = sum(class_counts.values())

        mentah = class_counts.get("mentah", 0)
        mengkal = class_counts.get("mengkal", 0)
        matang = class_counts.get("matang", 0)

        colA, colB = st.columns([1,1])

        with colA:
            # Total Deteksi
            st.markdown("<br><h4> 🔢 Jumlah Total Deteksi </h4>", unsafe_allow_html=True)
        
            # Angka total deteksi
            st.markdown(
                f"<h1 style='text-align:center; font-size:60px; margin-top:10px;'>{total}</h1>",
                unsafe_allow_html=True,
            )
        
        with colB:
            # Judul (dikasih jarak biar sejajar sama colA)
            st.markdown("<br><h4> 📊 Detail Kematangan </h4>", unsafe_allow_html=True)
        
            # Isi data
            st.markdown(f"""
            <div style='font-size:22px; margin-top:20px;'>
                🌱 Mentah&nbsp;&nbsp;&nbsp;: <b>{mentah}</b><br>
                🟡 Mengkal : <b>{mengkal}</b><br>
                🌾 Matang&nbsp;&nbsp;&nbsp;: <b>{matang}</b>
            </div>
            """, unsafe_allow_html=True)


        # ===================== STATUS PANEN ======================
        st.markdown("<br><h4>🌾 Status Kesiapan Panen</h4>", unsafe_allow_html=True)
        
        # Ambil kelas yang paling banyak terdeteksi
        counts = {"mentah": mentah, "mengkal": mengkal, "matang": matang}
        max_count = max(counts.values())
        
        # Kalau tidak ada deteksi sama sekali
        if max_count == 0:
            status_text = "❌ <b>Tidak ada objek terdeteksi</b>"
            status_color = "#7F8C8D"
        else:
            # Ambil semua kelas yang jumlahnya sama-sama paling besar (antisipasi seri)
            top_classes = [k for k, v in counts.items() if v == max_count]
        
            # Kalau seri, pakai prioritas: matang > mengkal > mentah
            priority = {"matang": 3, "mengkal": 2, "mentah": 1}
            dominant = sorted(top_classes, key=lambda k: priority[k], reverse=True)[0]
        
            if dominant == "matang":
                status_text = "✔ <b>Siap Dipanen</b> (Matang)"
                status_color = "#1FA63A"
            elif dominant == "mengkal":
                status_text = "⚠ <b>Belum Siap</b> (Mengkal)"
                status_color = "#E0A800"
            else:  # dominant == "mentah"
                status_text = "❌ <b>Belum Siap</b> (Mentah)"
                status_color = "#C0392B"
        
        st.markdown(
            f"""
            <div style="
                margin-top:15px;
                padding:20px;
                border-radius:15px;
                border:3px solid {status_color};
                background-color:#FAFAFA;
                font-size:22px;
                font-weight:bold;
                color:{status_color};
                text-align:center;
            ">
                {status_text}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ===================== DIAGRAM KECIL (PIE CHART) ======================
        st.markdown("<br><h4>📊 Diagram Deteksi (Pie Chart)</h4>", unsafe_allow_html=True)
        
        data_chart = {
            "Kategori": ["Mentah", "Mengkal", "Matang"],
            "Jumlah": [mentah, mengkal, matang]
        }
        
        fig = px.pie(
            data_chart,
            names="Kategori",
            values="Jumlah",
            title="Persentase Deteksi per Kategori",
            color="Kategori",
            color_discrete_map={
                "Mentah": "#E74C3C",    # Merah
                "Mengkal": "#F1C40F",  # Kuning
                "Matang": "#27AE60"   # Hijau
            },
            hole=0.3   # donut kecil biar lebih bagus
        )
        
        fig.update_layout(height=320)  # ukuran pie chart normal
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# ======================= MODE VIDEO =======================
# ==========================================================
elif option == "Upload Video":

    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov"])

    if uploaded_video:

        # Simpan video input sementara
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        # File output video
        output_path = "hasil_deteksi_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()

        with st.spinner("🔍 Memproses video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)

                annotated_img, _ = draw_results(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    results
                )

                annotated_bgr = cv2.cvtColor(
                    np.array(annotated_img), cv2.COLOR_RGB2BGR
                )

                out.write(annotated_bgr)
                stframe.image(annotated_bgr, channels="BGR", use_container_width=True)

        cap.release()
        out.release()

        st.success("✅ Video selesai diproses!")

        # ================= DOWNLOAD VIDEO =================
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Video Hasil Deteksi",
                data=f,
                file_name="hasil_deteksi_sawit.mp4",
                mime="video/mp4"
            )
