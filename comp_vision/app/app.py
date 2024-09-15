import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Función para extraer el primer frame del video
def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if success:
        # Guarda el primer frame en el directorio local
        frame_path = os.path.join("primer_frame.jpg")
        cv2.imwrite(frame_path, frame)
        return frame_path
    else:
        return None

# Función para encontrar los centroides
def find_objects_centroids(class_numbers, image_path):
    centroid = []
    image = cv2.imread(image_path)

    # Detectar mesas (o cualquier clase por class_numbers)
    results = model(image, classes=class_numbers, conf=0.5)  

    # Extraer las cajas de detección
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas delimitadoras
    scores = results[0].boxes.conf.cpu().numpy()  # Confianza de las detecciones

    # Dibujar las cajas en la imagen
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        center_x = int(((x2 - x1) // 2) + x1)
        center_y = int(((y2 - y1) // 2 ) + y1)
        centroid.append((center_x, center_y))  
    
    return centroid

# Función para rastrear personas
def person_tracker(video_path, centroid, threshold):
    cap = cv2.VideoCapture(video_path)
    
    # Diccionario para almacenar el historial completo de las posiciones (x, y) por cada ID
    track_history = defaultdict(lambda: [])
    
    # Loop a través de los frames del video
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Ejecuta el seguimiento con YOLOv8 solo para personas (clase 0)
            results = model.track(frame, persist=True, classes=[0], conf=0.5)

            # Verificar si hay detecciones antes de acceder a las cajas e IDs
            if results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu()  # Coordenadas de las cajas delimitadoras

                # Verificar si hay track IDs
                track_ids = results[0].boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()  # Convertir a lista si los IDs existen

                    # Guarda las posiciones de cada ID
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # Guardar la posición del centro del cuadro (x, y))

        else:
            # Finaliza cuando el video se haya procesado completamente
            break

    # Libera el video
    cap.release()

    # Guardar el historial completo de posiciones en un archivo CSV
    track_data = []
    for track_id, positions in track_history.items():
        for frame_num, (x, y) in enumerate(positions):
            track_data.append([track_id, frame_num, x, y])

    # Crear un DataFrame para organizar los datos
    df = pd.DataFrame(track_data, columns=['ID', 'Frame', 'X', 'Y'])

    def inbound(row, x, y, threshold):
        # Euclidean distance formula
        distance = np.sqrt((row['X'] - x) ** 2 + (row['Y'] - y) ** 2)
        return 1 if distance < threshold else 0

    for point in centroid:
        df[str(point)] = df.apply(inbound, axis=1, args=(point[0], point[1], threshold))

    column_names = [str(point) for point in centroid]

    # Add 'ID' to the column names
    column_names = ['ID'] + column_names

    # Create an empty DataFrame with the dynamic column names
    df_summarize = pd.DataFrame(columns=column_names)


    df_summarize = pd.DataFrame(columns=df_summarize.columns)  # Initialize the summarized DataFrame

    for client in df['ID'].unique():
        new_row = [client]
        df_copy = df
        filter_df = df_copy[df_copy ['ID'] == client]
        total_screentime = filter_df.shape[0]
        
        # Iterate through the columns and calculate the average for relevant columns
        for column in df.columns:
            if '(' in column:
                new_row.append(filter_df[column].sum() / total_screentime)
        
        # Create a new DataFrame row from the new_row list
        new_row_df = pd.DataFrame([new_row], columns=df_summarize.columns)
        
        # Concatenate the new row to the df_summarize DataFrame
        df_summarize = pd.concat([df_summarize, new_row_df], ignore_index=True)



    return df, df_summarize

# Función para generar un heatmap
def plot_heatmap(image_path, df_general):
    # Lee la imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte de BGR a RGB

    # Tamaño del heatmap
    heatmap_size = (image.shape[1], image.shape[0])

    # Define el número de bins (ajusta según necesidad)
    bins_x = 25
    bins_y = 25

    # Crea el histograma 2D (heatmap data)
    heatmap_data, xedges, yedges = np.histogram2d(df_general['X'], df_general['Y'], bins=(bins_x, bins_y))

    # Normaliza los datos del heatmap
    heatmap_data = np.clip(heatmap_data / np.max(heatmap_data), 0, 1)

    # Aplica un mapa de color
    heatmap_colormap = plt.get_cmap('hot')
    heatmap_color = heatmap_colormap(heatmap_data)
    heatmap_color = (heatmap_color[:, :, :3] * 255).astype(np.uint8)

    # Asegúrate de que el tamaño del heatmap sea el mismo que la imagen
    heatmap_color = cv2.resize(heatmap_color, (heatmap_size[0], heatmap_size[1]))

    # Fusiona el heatmap con la imagen original
    alpha = 0.7  # Ajusta la transparencia
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    # Muestra el resultado
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.axis('off')  # Elimina el eje
    st.pyplot(plt)

# ------------------------------
# Streamlit App
# ------------------------------

# Título de la aplicación
st.title("Seguimiento de Personas y Heatmap")

# Sidebar para cargar el archivo
st.sidebar.title("Carga de Video")
uploaded_file = st.sidebar.file_uploader("Sube un archivo MP4", type=["mp4"])

# Parámetros adicionales
threshold = st.sidebar.slider("Distancia Umbral para Seguimiento", min_value=50, max_value=500, value=150)
class_numbers = [2]  # Por defecto las mesas o cualquier objeto de interés

# Ruta para guardar el archivo
if uploaded_file is not None:
    video_path = os.path.join("videos_subidos", uploaded_file.name)

    # Guardar el archivo
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("Video subido correctamente.")

    # Botón de procesamiento
    if st.sidebar.button("Procesar"):
        # Extraer primer frame
        first_frame_path = extract_first_frame(video_path)
        st.image(first_frame_path, caption="Primer Frame Extraído", use_column_width=True)

        # Calcular centroides de objetos en el primer frame
        centroids = find_objects_centroids(class_numbers, first_frame_path)
        #st.write("Centroides calculados:", centroids)

        # Realizar el análisis de seguimiento de personas
        df_tracking, df_summarize = person_tracker(video_path, centroids, threshold)
        st.write("DataFrame con el seguimiento de personas:")
        st.dataframe(df_summarize)

        # Generar el heatmap basado en el análisis
        st.write("Heatmap del seguimiento:")
        plot_heatmap(first_frame_path, df_tracking)

else:
    st.info("Por favor, sube un archivo de tipo MP4.")
