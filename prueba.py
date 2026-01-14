import pyrealsense2 as rs
import torch
import numpy as np
import cv2
import math
import os

# --- FUNCIÓN MODULAR DE NAVEGACIÓN ---
def obtener_distancia_angulo(px, py, depth_frame, intrinsics):
    """
    Recibe coordenadas de píxel y devuelve (distancia, angulo).
    """
    # 1. Obtener Distancia Real (Z)
    distancia = depth_frame.get_distance(px, py)
    
    # Si la lectura falla (cero), intentamos un pequeño promedio de 
    # píxeles adyacentes para evitar zonas ciegas por reflejo.
    if distancia == 0:
        for offset in [5, -5, 10, -10]:
            distancia = depth_frame.get_distance(px, py + offset)
            if distancia != 0: break

    # 2. Calcular Ángulo Horizontal (Yaw)
    # Usamos la fórmula: angulo = atan2(distancia_al_centro_x, focal_x)
    offset_x = px - intrinsics.ppx
    angulo = math.degrees(math.atan2(offset_x, intrinsics.fx))
    
    return distancia, angulo

# --- SCRIPT PRINCIPAL ---
def iniciar_robot():
    # Inicialización de Modelo
    base_path = os.path.dirname(os.path.abspath(__file__))
    model = torch.hub.load(os.path.join(base_path, 'yolov5'), 'custom', 
                           path=os.path.join(base_path, 'best.pt'), source='local')
    model.conf = 0.4

    # Inicialización de Cámara
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            results = model(img)
            detecciones = results.pandas().xyxy[0]

            lata_cercana = None
            min_dist = 99.0

            for _, row in detecciones.iterrows():
                # Calculamos centro de la detección
                px = int((row['xmin'] + row['xmax']) / 2)
                py = int((row['ymin'] + row['ymax']) / 2)

                # --- USO DE LA FUNCIÓN ---
                dist, ang = obtener_distancia_angulo(px, py, depth_frame, intrinsics)

                # Lógica para encontrar la más cercana
                if 0.1 < dist < min_dist:
                    min_dist = dist
                    lata_cercana = {"distancia": dist, "angulo": ang, "box": row}

            # Si encontramos un objetivo, imprimimos los valores que irán al motor
            if lata_cercana:
                d = lata_cercana["distancia"]
                a = lata_cercana["angulo"]
                print(f"OBJETIVO >> Distancia: {d:.2f}m | Ángulo: {a:.1f}°")
                
                # Visualización
                box = lata_cercana["box"]
                cv2.rectangle(img, (int(box['xmin']), int(box['ymin'])), 
                              (int(box['xmax']), int(box['ymax'])), (0, 255, 0), 2)

            cv2.imshow('Robot Playa - Deteccion Cercana', img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        pipeline.stop()

if __name__ == "__main__":
    iniciar_robot()