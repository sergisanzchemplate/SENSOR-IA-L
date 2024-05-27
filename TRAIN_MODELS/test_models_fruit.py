import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# Función para predecir la clase de una imagen
def predict_image_class(image_array, model):
    # Realizar la predicción
    prediction = model.predict(image_array)
    
    # Obtener la clase predicha
    predicted_class = np.argmax(prediction)
    
    return predicted_class

# Ruta de la imagen que deseas probar

#image_path = 'dataset_fruta/(0)Apple/Apple (483).jpeg'
image_path = 'dataset_fruta/(1)Banana/Banana (2496).jpeg'
image_path = 'dataset_fruta/(2)Mango/Mango (500).jpeg'
image_path = 'dataset_fruta/(3)Strawberry/Strawberry (517).jpeg'


# Cargar el modelo guardado
loaded_model = tf.keras.models.load_model('model_saved_Xception.keras')

# Preprocesamiento de la imagen
image = load_img(image_path, target_size=(229, 229))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array /= 255.0  # Normalizar la imagen

# Predicción de la clase de la imagen seleccionada
predicted_class = predict_image_class(image_array, loaded_model)

# Imprime el resultado
print(f'Clase predicha para la imagen {image_path}: {predicted_class}')