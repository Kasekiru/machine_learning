import os
import zipfile
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path ke file zip yang diunggah
zip_path = '/content/dataset.zip'  # Pastikan file ini bernama dataset.zip setelah diunggah

# Ekstrak file zip ke dalam folder /content/dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

# Tentukan folder dataset yang sudah diekstraksi di langkah pertama
dataset_path = '/content/dataset'
output_folder = '/content/preprocessing_dataset'
os.makedirs(output_folder, exist_ok=True)

# Inisialisasi ImageDataGenerator untuk augmentasi
datagen = ImageDataGenerator(
    shear_range=0.2,        # Shearing
    zoom_range=0.2,         # Random zoom
    horizontal_flip=True     # Flip horizontal
)

# Fungsi untuk mengubah gambar menjadi grayscale
def convert_to_grayscale(img):
    return img.convert('L')

# Fungsi untuk melakukan Gaussian Blur
def gaussian_blur(img, kernel_size=(5, 5)):
    img_array = np.array(img)
    blurred_img = cv2.GaussianBlur(img_array, kernel_size, 0)
    return blurred_img

# Proses setiap gambar dalam folder dataset, dengan struktur yang sama di folder output
for root, dirs, files in os.walk(dataset_path):
    category = os.path.basename(root)  # Ambil nama kategori dari nama folder
    if category in ['anjing', 'ayam', 'sapi']:  # Pastikan folder kategori sesuai
        output_category_folder = os.path.join(output_folder, category)
        os.makedirs(output_category_folder, exist_ok=True)

        for img_file in files:
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, img_file)
                img = Image.open(img_path)

                # Convert to RGB if image mode is not RGB (to ensure 3 channels)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize gambar
                img_resized = img.resize((224, 224))

                # Normalisasi dan tambahkan dimensi batch
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)  # Shape: (1, 224, 224, 3)

                # Augmentasi gambar
                aug_iter = datagen.flow(img_array, batch_size=1)

                for i in range(5):  # Buat 5 variasi augmentasi
                    aug_img = next(aug_iter)[0]
                    aug_img_uint8 = np.clip(aug_img * 255, 0, 255).astype('uint8')

                    # Simpan gambar augmentasi ke folder kategori
                    aug_img_output_path = os.path.join(output_category_folder, f'aug_{img_file.split(".")[0]}_{i}.jpg')
                    Image.fromarray(aug_img_uint8).save(aug_img_output_path)

                # Terapkan Gaussian Blur
                img_blurred = gaussian_blur(img_resized)

                # Simpan gambar yang sudah diblur ke folder kategori
                blurred_img_output_path = os.path.join(output_category_folder, f'blurred_{img_file.split(".")[0]}.jpg')
                Image.fromarray(img_blurred).save(blurred_img_output_path)

                # Konversi ke grayscale dan simpan ke folder kategori
                img_gray = convert_to_grayscale(img_resized)
                gray_img_output_path = os.path.join(output_category_folder, f'gray_{img_file.split(".")[0]}.jpg')
                img_gray.save(gray_img_output_path)

print("Preprocessing selesai! Hasil disimpan di folder:", output_folder)