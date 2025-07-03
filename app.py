import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Memuat model yang telah disimpan
model = tf.keras.models.load_model('model.h5')

# Fungsi untuk memproses gambar input
def preprocess_image(image):
    image = image.resize((224, 224))  # Ukuran yang sesuai dengan input model
    image_array = np.array(image) / 255.0  # Normalisasi gambar
    image_array = np.expand_dims(image_array, axis=0)  # Menambah dimensi batch
    return image_array

# Fungsi untuk memvisualisasikan hasil prediksi dengan Bar Chart
def plot_bar_chart(prediction):
    fig, ax = plt.subplots()
    categories = ['Land Disaster', 'Water Disaster']
    ax.bar(categories, prediction[0])
    ax.set_ylabel('Probabilitas')
    ax.set_title('Hasil Prediksi Bencana Alam')
    st.pyplot(fig)

# Fungsi untuk memvisualisasikan hasil prediksi dengan Pie Chart
def plot_pie_chart(prediction):
    fig, ax = plt.subplots()
    categories = ['Land Disaster', 'Water Disaster']
    ax.pie(prediction[0], labels=categories, autopct='%1.1f%%', startangle=90, colors=["lightblue", "lightgreen"])
    ax.set_title('Distribusi Prediksi Bencana Alam')
    st.pyplot(fig)

# Tampilan utama
st.title('Prediksi Bencana Alam: Land Disaster atau Water Disaster')

# Input gambar dari file chooser
uploaded_file = st.file_uploader("Pilih gambar bencana dari komputer", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_column_width=True)

    # Proses gambar
    image_array = preprocess_image(image)

    # Prediksi menggunakan model
    prediction = model.predict(image_array)

    # Memeriksa hasil prediksi untuk memastikan tidak ada nilai yang sama
    land_prob = prediction[0][0]  # Probabilitas untuk Land Disaster
    water_prob = prediction[0][1]  # Probabilitas untuk Water Disaster

    # Menampilkan hasil prediksi
    if land_prob > water_prob:
        predicted_class = "Land Disaster"
    else:
        predicted_class = "Water Disaster"

    st.write(f"Kategori Prediksi: {predicted_class}")

    # Visualisasi grafik hasil prediksi - Bar Chart
    plot_bar_chart(prediction)

    # Visualisasi grafik hasil prediksi - Pie Chart
    plot_pie_chart(prediction)
