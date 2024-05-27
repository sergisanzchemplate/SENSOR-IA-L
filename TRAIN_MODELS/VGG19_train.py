import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Cargar el dataset de imágenes
data_dir = 'dataset'
batch_size = 32
img_height = 224
img_width = 224

# Crear generadores de datos para entrenamiento y validación
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = data_generator.flow_from_directory(data_dir, target_size=(img_height, img_width), batch_size=batch_size, subset='training')
val_data = data_generator.flow_from_directory(data_dir, target_size=(img_height, img_width), batch_size=batch_size, subset='validation')

# Crear el modelo base a partir de VGG19
base_model = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_height, img_width, 3)))
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(4, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_data, validation_data=val_data, epochs=15)

# Graficar la precisión y la pérdida durante el entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig('accuracy_loss_VGG19.png')




# Guardar el modelo en formato keras
model.save('model_saved_VGG19.keras')

# Cargar el modelo guardado
saved_model = tf.keras.models.load_model('model_saved_VGG19.keras')

# Crear un convertidor
converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)

# Convertir el modelo
tflite_model = converter.convert()

# Guardar el modelo .tflite
with open('model_VGG19.tflite', 'wb') as f:
    f.write(tflite_model)
