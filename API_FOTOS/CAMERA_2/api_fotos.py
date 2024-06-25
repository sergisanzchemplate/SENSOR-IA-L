from flask import Flask,url_for,render_template, jsonify, Response, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from tflite_runtime.interpreter import Interpreter
import cv2
import os
import time
import threading
import schedule
import subprocess
from collections import defaultdict
import logging
import csv
from datetime import datetime
import numpy as np
import psycopg2
import base64



app = Flask(__name__)



cap = None
capturing = False

def capture_images():
    global cap
    global capturing

    # Obtener la fecha actual
    current_date = time.strftime("%d_%m")

    # Crear el nombre de la carpeta para guardar las imágenes
    folder_name = os.path.join(os.path.dirname(__file__), "static", "imagenes_cam2", "dia_" + current_date)

    # Crear la carpeta si no existe
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    while capturing:
        # Obtener la hora actual
        current_time = time.strftime("%H-%M-%S")

        # Crear el nombre de archivo con la hora actual
        file_name = "img_" + current_time + ".jpg"

        # Ruta completa para guardar la imagen
        file_path = os.path.join(folder_name, file_name)

        # Capturar un fotograma
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el fotograma")
            break

        # Guardar la imagen en la ruta especificada
        cv2.imwrite(file_path, frame)
        print(f"Imagen guardada como {file_path}")

        # Esperar 5 segundos (ajusta según sea necesario)
        time.sleep(5)

    cap.release()




@app.route('/start_capture')
def start_capture():
    global cap
    global capturing
    capturing = True
    cap = cv2.VideoCapture(0)
    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    # Configurar la resolución de la cámara (opcional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    #cap = cv2.VideoCapture(0) Iniciar el hilo para capturar imágenes
    threading.Thread(target=capture_images).start()
    return 'Captura de imágenes iniciada, si quieres parar la captura, vuelve al menú y presiona el botón Detener Captura'




@app.route('/stop_capture')
def stop_capture():
    global capturing
    capturing = False
    while threading.active_count() > 1:
        time.sleep(1)
    #sync_with_server()
    return 'Captura de imágenes detenida, Se enviaran las imagenes al servidor a las 21h de manera automatica.'
    



def generar_frames():
    # Abre la cámara USB conectada a la Raspberry Pi
    cap = cv2.VideoCapture(0)  # Puedes cambiar el número según el dispositivo de la cámara

    while True:
        # Lee un frame de la cámara
        ret, frame = cap.read()

        if not ret:
            break

        # Convierte el frame a un formato que se pueda transmitir
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Renderea el frame como un objeto de respuesta
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Libera los recursos de la cámara
    cap.release()




@app.route('/abrir_camara')
def abrir_camara():
    return render_template('camera.html')




@app.route('/video_feed')
def video_feed():
    # Renderea el flujo de video
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/cerrar_camara')
def cerrar_camara():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    return 'Camera released and window closed'




@app.route('/sync_with_server')
def sync_with_server():
    # Ruta local de la carpeta a sincronizar en la Raspberry Pi
    local_folder = '/home/proves/Desktop/API_FOTOS/static/imagenes_cam2'

    # Parámetros de la sincronización
    remote_folder = '/mnt/nvme/chemplate/static/imagenes_cam2'
    remote_ip = '192.168.50.107'
    remote_user = 'chemplate'

    # Comando rsync para sincronizar las carpetas
    command = f'rsync -avz {local_folder}/ {remote_user}@{remote_ip}:{remote_folder}'

    try:
        # Ejecutar el comando rsync
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        # Verificar si el comando se ejecutó correctamente
        if result.returncode == 0:
            return jsonify({'message': 'Sincronización exitosa', 'output': result.stdout})
        else:
            return jsonify({'error': 'Error durante la sincronización', 'details': result.stderr}), 500

    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Error durante la sincronización', 'details': str(e)}), 500





def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image, input_shape):
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    image = preprocess_image(image, input_shape)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def gen_frames():
    camera_id = 0

    # Rutas de los archivos y otras configuraciones
    DATA_FOLDER = "/home/proves/Desktop/API_FOTOS/tensorFlowLite/"
    model_path = DATA_FOLDER + "model2_cubeta3.tflite"
    labels_path = DATA_FOLDER + "labels.txt"

    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    labels = load_labels(labels_path)

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("ERROR: Unable to access the camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        output_data = classify_image(interpreter, frame)

        top_index = np.argmax(output_data[0])
        top_label = labels[top_index]
        top_score = output_data[0][top_index]

        text = f"{top_label}: {top_score:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()



@app.route('/model')
def model():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    



@app.route('/historial')
def historial():
    with open('historial.csv', 'r') as file:
        data = file.read()
    return Response(data, mimetype='text/plain')





processing_thread = None
stop_event = threading.Event()

def load_labels1(labels_path):
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image1(image, input_shape):
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_image1(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    image = preprocess_image1(image, input_shape)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def save_to_database(timestamp, estacio, estat, confidence, image_base64):
    try:
        conn = psycopg2.connect(
            host="192.168.50.111",
            port="5432",
            dbname="chemplate_img",
            user="chemplate",
            password="Chemplate123"
        )
        cur = conn.cursor()

        confidence_float = float(confidence)

        cur.execute("""
            INSERT INTO images_cubeta3 (timestamp, estacio, estat, confidence, imatge)
            VALUES (%s, %s, %s, %s, %s)
        """, (timestamp, estacio, estat, confidence_float, image_base64))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error al guardar en la base de datos: {e}")

def process_images1():
    DATA_FOLDER = "/home/proves/Desktop/API_FOTOS/tensorFlowLite/"
    model_path = DATA_FOLDER + "model2_cubeta3.tflite"
    labels_path = DATA_FOLDER + "labels.txt"
    camera_id = 0

    try:
        interpreter = Interpreter(model_path)
        interpreter.allocate_tensors()
        labels = load_labels1(labels_path)
    except Exception as e:
        print(f"Error al cargar el modelo o etiquetas: {e}")
        return

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("ERROR: Unable to access the camera")
        return

    last_save_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        current_time = time.time()

        if current_time - last_save_time >= 5:
            try:
                output_data = classify_image1(interpreter, frame)

                top_index = np.argmax(output_data[0])
                top_label = labels[top_index]
                top_score = output_data[0][top_index]

                timestamp = datetime.now()
                estacio = 3  # Estación fija
                estat = top_label
                confidence = top_score

                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                save_to_database(timestamp, estacio, estat, confidence, image_base64)

                last_save_time = current_time
            except Exception as e:
                print(f"Error en el procesamiento de la imagen: {e}")

        time.sleep(1)  # Pequeña pausa para evitar un bucle ocupado

    cap.release()



@app.route('/enviar_clasificacion_bd', methods=['GET'])
def enviar_clasificacion_bd():
    global processing_thread
    if processing_thread is None or not processing_thread.is_alive():
        stop_event.clear()
        processing_thread = threading.Thread(target=process_images1)
        processing_thread.start()
        return jsonify({'message': 'Started processing images'}), 200
    else:
        return jsonify({'message': 'Processing already started'}), 200




@app.route('/stop_envio', methods=['GET'])
def stop_envio():
    stop_event.set()
    if processing_thread:
        processing_thread.join()
    return jsonify({'message': 'Stopped processing images'}), 200





@app.route('/')
def menu():
    return render_template('menu.html')



if __name__ == '__main__':
    app.run(debug=True, host='192.168.50.114')

