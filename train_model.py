import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Menyiapkan ImageDataGenerator untuk memproses gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi gambar
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Memuat data gambar dari folder dataset
train_generator = train_datagen.flow_from_directory(
    'C:/Users/Biyan/Dataset/',  # Path ke dataset Anda
    target_size=(224, 224),  # Ukuran gambar yang digunakan oleh model
    batch_size=32,  # Ukuran batch
    class_mode='categorical'  # Klasifikasi biner (Gempa atau Tsunami)
)

# Membangun model CNN untuk 2 kelas
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Menghindari overfitting
    Dense(2, activation='softmax')  # Output layer untuk 2 kelas (Land Disaster atau Water Disaster)
])


# Mengkompilasi model dengan learning_rate, bukan lr
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(train_generator, epochs=10)

# Menyimpan model yang sudah dilatih
model.save('model.h5')
