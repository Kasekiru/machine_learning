!pip install gradio

import gradio as gr
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Definisikan daftar kelas
class_names = ["anjing", "ayam", "sapi"]
class_indices = {i: class_name for i, class_name in enumerate(class_names)}

# Fungsi untuk melakukan prediksi dan mencocokkan hasilnya
def predict_image(image):
    # Path model yang akan digunakan untuk prediksi
    model_file = 'CNN_Anjing_Ayam_Sapi.keras'

    # Parameter gambar
    img_height, img_width = 240, 240

    # Load dan preprocess gambar
    img = image.resize((img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalisasi (0-1)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Load model
    model = load_model(model_file)

    # Prediksi dengan model
    predictions = model.predict(img_array)

    # Ambil indeks prediksi tertinggi dan nilai probabilitasnya
    predicted_class_index = np.argmax(predictions[0])  # Ambil indeks prediksi tertinggi
    predicted_class = class_indices[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100  # Tingkat kepercayaan dalam persen

    # Hasil prediksi
    return f"Gambar tersebut termasuk kategori '{predicted_class}' dengan kepercayaan {confidence:.2f}%"

# Antarmuka Gradio
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),  # Menggunakan tipe "pil" untuk mengelola gambar
    outputs=gr.Textbox(),
    title="Klasifikasi Gambar Anjing, Ayam, dan Sapi",
    description="Unggah gambar anjing, ayam, atau sapi, dan model akan mengklasifikasikannya."
)

# Jalankan aplikasi
if __name__ == "__main__":
    demo.launch(share=True)  # Set "share=True" untuk mendapatkan URL publik