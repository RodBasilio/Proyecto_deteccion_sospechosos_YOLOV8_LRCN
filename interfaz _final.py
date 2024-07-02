import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from collections import deque
from keras.models import load_model
from ultralytics import YOLO
import time
from sort import Sort
from threading import Thread

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 30
CLASSES_LIST = ["peligroso", "sospechoso", "normal"]
model = load_model('last_lrcn.h5')
model2 = YOLO('last_pose_200.pt')

video_writer = None
start_time = 0

def predict_single_action(frames_list, model):
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    return predicted_class_name, predicted_labels_probabilities[predicted_label]

def iniciarCamara():
    global cameraObject
    cameraObject = cv2.VideoCapture(0)
    capturarImagen()


def capturarImagen():
    global cameraObject
    if cameraObject is not None:
        retval, imagen = cameraObject.read()
        if retval == True:
            img = Image.fromarray(imagen)
            img = img.resize((640,480))
            imgTk = ImageTk.PhotoImage(image = img)
            captureLabel.configure(image = imgTk)
            captureLabel.image = imgTk
            captureLabel.after(10, capturarImagen)

            # Procesa la imagen con el modelo YOLO
            results = model2(imagen, show=True, conf=0.2)
        else:
            captureLabel.image = ""
            cameraObject.release()

def iniciarCamaraYOLO():
    global cameraObject, frames_queue
    cameraObject = cv2.VideoCapture(0)
    capturarImagenYOLO()

def capturarImagenYOLO():
    global cameraObject, frames_queue
    if cameraObject is not None:
        retval, imagen = cameraObject.read()
        if retval == True:
            resized_frame = cv2.resize(imagen, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)
            if len(frames_queue) == SEQUENCE_LENGTH:
                predicted_class_name, confidence = predict_single_action(frames_queue, model)
                cv2.putText(imagen, f'Action: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(imagen, f'Confidence: {confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Utiliza el modelo YOLO para la detección de objetos en cada fotograma
            results = model2(source=imagen, show=True)
            # Convertir de BGR a RGB antes de crear la imagen PIL
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(imagen)
            img = img.resize((640,480))
            imgTk = ImageTk.PhotoImage(image = img)
            captureLabel.configure(image = imgTk)
            captureLabel.image = imgTk
            captureLabel.after(10, capturarImagenYOLO)
        else:
            captureLabel.image = ""
            cameraObject.release()

def iniciarSeguimiento():
    cap = cv2.VideoCapture(0)
    model_yolo = YOLO("models\last_arma_100.pt")
    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break
        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.3)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                cv2.putText(img=frame, text=f"Sospechoso Id: {track_id}", org=(xmin, ymin - 10),
                            fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

def iniciarCamaraLRCN():
    global cameraObject, frames_queue, video_writer, start_time

    cameraObject = cv2.VideoCapture(0)

    def iniciarGrabacionVideo():
        global video_writer, start_time

        # Obtener la hora y la fecha actual
        current_time = time.strftime("%Y%m%d%H%M%S")
        video_filename = f"video_{current_time}.avi"  # Nombre del archivo de video

        # Carpeta de destino para los videos
        output_folder = "videos"  # Cambia esto a la carpeta deseada
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Configurar el objeto VideoWriter
        frame_width, frame_height = 640, 480  # Tamaño del fotograma
        video_writer = cv2.VideoWriter(os.path.join(output_folder, video_filename),
                                      cv2.VideoWriter_fourcc(*'XVID'),
                                      30, (frame_width, frame_height))

        # Iniciar el tiempo de grabación
        start_time = time.time()

    def detenerGrabacionVideo():
        global video_writer, start_time
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            start_time = 0

    def procesarYGrabarImagen():
        global cameraObject, frames_queue, video_writer, start_time

        if cameraObject is not None:
            retval, imagen = cameraObject.read()

            if retval:
                resized_frame = cv2.resize(imagen, (IMAGE_HEIGHT, IMAGE_WIDTH))
                normalized_frame = resized_frame / 255
                frames_queue.append(normalized_frame)
                if len(frames_queue) == SEQUENCE_LENGTH:
                    predicted_class_name, confidence = predict_single_action(frames_queue, model)
                    cv2.putText(imagen, f'Action: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(imagen, f'Confidence: {confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if video_writer is not None:
                    video_writer.write(imagen)

                # Convertir de BGR a RGB antes de crear la imagen PIL
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(imagen)
                img = img.resize((640, 480))
                imgTk = ImageTk.PhotoImage(image=img)
                captureLabel.configure(image=imgTk)
                captureLabel.image = imgTk

            captureLabel.after(10, procesarYGrabarImagen)

    # Crear un botón para iniciar la grabación de video
    btnIniciarVideo = Button(btnFrame, text="Iniciar Grabación LRCN", command=iniciarGrabacionVideo)
    btnIniciarVideo.place(x=100, y=60)

    # Crear un botón para detener la grabación de video
    btnDetenerVideo = Button(btnFrame, text="Detener Grabación LRCN", command=detenerGrabacionVideo)
    btnDetenerVideo.place(x=300, y=60)

    procesarYGrabarImagen()

def cerrarCamara():
    global cameraObject
    if video_writer is not None:
        video_writer.release()
    captureLabel.image = ""
    cameraObject.release()

def cerrarVentana():
    raiz.destroy()

raiz = Tk()
raiz.geometry("640x580")
raiz.title("Capturar imagen")

captureFrame = Frame()
captureFrame.config(width="640", heigh="480")
captureFrame.place(x=0, y=0)

btnFrame = Frame()
btnFrame.config(width="640", heigh="100")
btnFrame.place(x=0, y=480)

captureLabel = Label(captureFrame)
captureLabel.place(x=0, y=0)

btnCaptureLRCN = Button(btnFrame, text="Camara LRCN", command=iniciarCamaraLRCN)
btnCaptureLRCN.place(x=20, y=20)

btnCapture = Button(btnFrame, text="Camara YOLO", command=iniciarCamara)
btnCapture.place(x=140, y=20)

btnCaptureYOLO = Button(btnFrame, text="Camara LRCN-YOLO", command=iniciarCamaraYOLO)
btnCaptureYOLO.place(x=250, y=20)

btnSeguimiento = Button(btnFrame, text="Seguimiento", command=iniciarSeguimiento)
btnSeguimiento.place(x=380, y=20)

btnCerrarVideo = Button(btnFrame, text="Cerrar Camara", command=cerrarCamara)
btnCerrarVideo.place(x=480, y=20)

btnCerrar = Button(btnFrame, text="Cerrar", command=cerrarVentana)
btnCerrar.place(x=580, y=20)

frames_queue = deque(maxlen=SEQUENCE_LENGTH)

raiz.mainloop()
