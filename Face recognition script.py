import os
import cv2
import numpy as np
from pathlib import Path

# Configuración
UMBRAL_SIMILITUD = 0.75  # Ajustar según necesidad (0-1)
MODELO_DNN = True  # True para DNN (OpenCV + OpenFace), False para face_recognition

if not MODELO_DNN:
    import face_recognition

def cargar_modelos():
    """Carga los modelos de detección y extracción de características"""
    if MODELO_DNN:
        # Modelo de detección facial
        prototxt = "deploy.prototxt"
        modelo_det = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        if not (os.path.exists(prototxt) and os.path.exists(modelo_det)):
            raise FileNotFoundError("Archivos del modelo DNN no encontrados")
        
        # Modelo de extracción de características (OpenFace)
        modelo_emb = "nn4.small2.v1.t7"
        if not os.path.exists(modelo_emb):
            raise FileNotFoundError("Modelo de embeddings no encontrado")
        
        det_model = cv2.dnn.readNetFromCaffe(prototxt, modelo_det)
        emb_model = cv2.dnn.readNetFromTorch(modelo_emb)
        return det_model, emb_model
    return None, None

def extraer_rostros(img, det_model):
    """Extrae todos los rostros de una imagen"""
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, 
                               (300, 300), (104.0, 177.0, 123.0))
    det_model.setInput(blob)
    detecciones = det_model.forward()
    
    rostros = []
    for i in range(detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]
        if confianza > 0.7:  # Umbral de confianza mínimo
            box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            rostro = img[y1:y2, x1:x2]
            if rostro.size > 0:  # Asegurar que el rostro es válido
                rostros.append(rostro)
    return rostros

def obtener_codigo_facial(rostro, emb_model):
    """Obtiene el vector de características faciales"""
    if MODELO_DNN:
        # Preprocesamiento para OpenFace
        rostro = cv2.resize(rostro, (96, 96))
        blob = cv2.dnn.blobFromImage(rostro, 1./255, (96, 96), 
                                   (0, 0, 0), swapRB=True, crop=False)
        emb_model.setInput(blob)
        return emb_model.forward().flatten()
    else:
        return face_recognition.face_encodings(rostro)[0]

def comparar_rostros(codigo_ref, codigo_comparar):
    """Compara dos rostros usando similitud coseno"""
    codigo_ref_norm = codigo_ref / np.linalg.norm(codigo_ref)
    codigo_comp_norm = codigo_comparar / np.linalg.norm(codigo_comparar)
    return np.dot(codigo_ref_norm, codigo_comp_norm)

def buscar_coincidencias(referencia, dir_busqueda):
    """Busca coincidencias faciales en un directorio"""
    det_model, emb_model = cargar_modelos() if MODELO_DNN else (None, None)
    
    # Procesar imagen de referencia
    img_ref = cv2.imread(referencia)
    if img_ref is None:
        print("Error: No se pudo cargar la imagen de referencia")
        return []
    
    if MODELO_DNN:
        rostros_ref = extraer_rostros(img_ref, det_model)
        if len(rostros_ref) != 1:
            print("Error: Imagen de referencia debe contener exactamente un rostro")
            return []
        codigo_ref = obtener_codigo_facial(rostros_ref[0], emb_model)
    else:
        rgb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        codigos_ref = face_recognition.face_encodings(rgb_ref)
        if len(codigos_ref) != 1:
            print("Error: Imagen de referencia debe contener exactamente un rostro")
            return []
        codigo_ref = codigos_ref[0]
    
    # Buscar en directorio
    coincidencias = []
    extensiones = ('jpg', 'jpeg', 'png', 'bmp')
    
    for ext in extensiones:
        for img_path in Path(dir_busqueda).rglob(f'*.{ext}'):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            try:
                if MODELO_DNN:
                    rostros = extraer_rostros(img, det_model)
                    for rostro in rostros:
                        codigo = obtener_codigo_facial(rostro, emb_model)
                        similitud = comparar_rostros(codigo_ref, codigo)
                        if similitud > UMBRAL_SIMILITUD:
                            coincidencias.append((str(img_path), similitud))
                else:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    codigos = face_recognition.face_encodings(rgb)
                    for codigo in codigos:
                        similitud = 1 - face_recognition.face_distance([codigo_ref], codigo)[0]
                        if similitud > UMBRAL_SIMILITUD:
                            coincidencias.append((str(img_path), similitud))
            except Exception as e:
                print(f"Error procesando {img_path}: {str(e)}")
                continue
    
    # Ordenar por similitud (mayor primero)
    coincidencias.sort(key=lambda x: x[1], reverse=True)
    return coincidencias

def main():
    print("=== Sistema de Comparación Facial ===")
    print(f"Modo: {'OpenCV + OpenFace' if MODELO_DNN else 'face_recognition'}")
    
    # Solicitar imagen de referencia
    while True:
        ref_path = input("\nRuta completa de la imagen de referencia (1 rostro): ").strip()
        if not os.path.exists(ref_path):
            print("Error: Archivo no encontrado")
            continue
            
        # Verificar que tenga exactamente 1 rostro
        img = cv2.imread(ref_path)
        if MODELO_DNN:
            det_model, _ = cargar_modelos()
            rostros = extraer_rostros(img, det_model)
            num_rostros = len(rostros)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            num_rostros = len(face_recognition.face_locations(rgb))
            
        if num_rostros == 1:
            break
        print(f"Error: La imagen debe contener exactamente un rostro (encontrados: {num_rostros})")
    
    # Solicitar directorio de búsqueda
    while True:
        dir_busqueda = input("\nDirectorio para buscar coincidencias: ").strip()
        if os.path.isdir(dir_busqueda):
            break
        print("Error: Directorio no válido")
    
    # Procesar
    print("\nBuscando coincidencias...")
    resultados = buscar_coincidencias(ref_path, dir_busqueda)
    
    # Mostrar resultados
    print("\n" + "-"*50)
    if resultados:
        print(f"{len(resultados)} coincidencias encontradas (umbral: {UMBRAL_SIMILITUD*100:.0f}%):")
        for i, (ruta, similitud) in enumerate(resultados, 1):
            print(f"{i}. [{similitud*100:.1f}%] {ruta}")
    else:
        print("No se encontraron coincidencias")
    print("-"*50)

    input("\nPresione Enter para salir...")

if __name__ == "__main__":
    main()