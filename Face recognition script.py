import collections # defaultdict para agrupar archivos por carpeta
import pathlib # Manejo moderno de rutas de archivos y directorios
import cv2 # OpenCV: lectura de imágenes, detección de rostros, guardado
import numpy # Operaciones con arrays numéricos (coordenadas de rostros)

# VARIABLES GLOBALES
umbral_similitud = 0.75 # Umbral de similitud para coincidencias faciales (0-1)

# FUNCIONES
# Compara rostros encontrados con imagen de referencia y recopila coincidencias
def comparar_rostros(dir_imagen, rostros_imagen, vec_img_ref, modelo_embeddings):
    coincidencias_encontradas = [] # Lista para almacenar coincidencias con similitud
    
    # Comparar cada rostro detectado con la referencia
    for indice_rostro, rostro_iter in enumerate(rostros_imagen, 1):
        try:
            # Obtener código de características del rostro actual
            vector_rostro = calc_vector_embedding(rostro_iter, modelo_embeddings)
            
            # Compara rostros usando similitud coseno y devuelve nivel de coincidencia
            similitud_val = numpy.dot(vec_img_ref / numpy.linalg.norm(vec_img_ref), vector_rostro / numpy.linalg.norm(vector_rostro))
            
            # Verificar si supera el umbral de similitud
            if similitud_val > umbral_similitud:
                coincidencias_encontradas.append((str(dir_imagen), similitud_val, indice_rostro))
        except Exception as e:
            continue # Ignorar errores en rostros individuales
    
    return coincidencias_encontradas

# Procesa una imagen individual: carga, detecta rostros y recorta
def recortar_rostros(ruta_imagen, modelo_caffe):
    # Cargar imagen desde archivo usando OpenCV
    imagen_val = cv2.imread(str(ruta_imagen))
    
    try:
        # Ejecutar detección de rostros en la imagen
        rostros_coordenadas = coordenadas_rostros(imagen_val, modelo_caffe)
    except Exception as e:
        return []

    # Lista para almacenar imágenes recortadas de rostros
    rostros_imagenes = []
    
    # Recortar cada rostro detectado sin márgenes adicionales
    for (x_val, y_val, ancho_val, altura_val) in rostros_coordenadas:
        # Recortar región exacta del rostro detectado
        rostro_recortado = imagen_val[y_val : y_val + altura_val, x_val : x_val + ancho_val]
        
        # Agregar imagen al array de imágenes de rostros
        rostros_imagenes.append(rostro_recortado)
    
    return rostros_imagenes

# Procesamiento recursivo de imágenes y comparación con referencia
def procesar_directorio_imagenes(directorio_entrada, extensiones_lista, vec_img_ref, modelo_caffe, modelo_embeddings):
    total_imagenes_procesadas = 0
    contador_coincidencias = 0
    
    # Mostrar separador visual para inicio de resultados
    print("-" * 36)
    
    # Procesar recursivamente cada imagen en el directorio
    for extension_val in extensiones_lista:
        for archivo_iter in pathlib.Path(directorio_entrada).rglob(f'*.{extension_val}'):
            if archivo_iter.is_file():
                # Extraer rostros de la imagen actual
                rostros_imagen = recortar_rostros(archivo_iter, modelo_caffe)
                
                # Comparar rostros si se encontró alguno
                if rostros_imagen:
                    coincidencias_imagen = comparar_rostros(archivo_iter, rostros_imagen, vec_img_ref, modelo_embeddings)
                    
                    # Mostrar coincidencias inmediatamente
                    if coincidencias_imagen:
                        # Contar cuántas coincidencias hay en esta imagen
                        for ruta_imagen, similitud_val, numero_rostro in coincidencias_imagen:
                            contador_coincidencias += 1
                            
                            if len(coincidencias_imagen) > 1:
                                print(f"{contador_coincidencias}. (Face {numero_rostro}) [{similitud_val*100:.1f}%] {ruta_imagen}")
                            else:
                                print(f"{contador_coincidencias}. [{similitud_val*100:.1f}%] {ruta_imagen}")
                
                total_imagenes_procesadas += 1
    
    # Mostrar mensaje si no se encontraron coincidencias
    if contador_coincidencias == 0:
        if total_imagenes_procesadas > 0:
            print("No matches found")
        else:
            print("No images found")
    
    # Mostrar separador final
    print("-" * 36 + "\n")

# Extrae vector de características faciales usando modelo de embeddings
def calc_vector_embedding(rostro_imagen, modelo_embeddings):
    # Redimensionar rostro a tamaño requerido por OpenFace (96x96)
    rostro_redimensionado = cv2.resize(rostro_imagen, (96, 96))
    
    # Crear blob con normalización y configuración específica para OpenFace
    blob_rostro = cv2.dnn.blobFromImage(rostro_redimensionado, 1./255, (96, 96), (0, 0, 0), swapRB = True, crop = False)
    
    modelo_embeddings.setInput(blob_rostro) # Establecer blob como entrada del modelo
    
    # Ejecutar inferencia y obtener vector de características aplanado
    return modelo_embeddings.forward().flatten()

# Detecta rostros en una imagen usando red neuronal DNN con umbral de confianza
def coordenadas_rostros(imagen_val, modelo_caffe):
    alto_val, ancho_val = imagen_val.shape[:2] # Obtener dimensiones de la imagen original
    
    # Crear blob: redimensionar imagen a 300x300 y normalizar valores de píxeles
    blob_val = cv2.dnn.blobFromImage(cv2.resize(imagen_val, (300, 300)), 1.0, (300, 300), [104, 117, 123])
    
    modelo_caffe.setInput(blob_val) # Establecer blob como entrada de la red neuronal
    
    detecciones_val = modelo_caffe.forward() # Ejecutar inferencia y obtener detecciones
    
    rostros_coords = [] # Lista para almacenar coordenadas de rostros válidos
    
    # Procesar cada detección encontrada por la red neuronal
    for iter_val in range(detecciones_val.shape[2]):
        # Extraer nivel de confianza de la detección actual
        umbral_confianza = detecciones_val[0, 0, iter_val, 2]
        
        # Filtrar detecciones con confianza mayor al umbral (70%)
        if umbral_confianza > 0.7: # Umbral de confianza para reducir falsos positivos
            # Extraer coordenadas del rostro y escalar a dimensiones originales
            caja_val = detecciones_val[0, 0, iter_val, 3:7] * numpy.array([ancho_val, alto_val, ancho_val, alto_val])
            
            # Convertir coordenadas flotantes a enteros
            (x1, y1, x2, y2) = caja_val.astype("int")
            
            # Asegurar que las coordenadas estén dentro de los límites de la imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho_val, x2), min(alto_val, y2)
            
            # Convertir formato de coordenadas (x1,y1,x2,y2) a (x, y, ancho, alto)
            rostros_coords.append((x1, y1, x2 - x1, y2 - y1))
    
    return rostros_coords

# Procesa imagen de referencia y extrae características del primer rostro encontrado
def procesar_imagen_referencia(ruta_imagen_referencia, modelo_caffe, modelo_embeddings):
    # Cargar imagen de referencia desde archivo
    vec_img_ref = cv2.imread(str(ruta_imagen_referencia))
    
    if vec_img_ref is None:
        return None # Terminar función, retornar valor nulo
    
    # Detectar rostros en imagen de referencia
    rostros_coordenadas = coordenadas_rostros(vec_img_ref, modelo_caffe)
    
    # Verificar que se encontró exactamente un rostro
    if len(rostros_coordenadas) != 1:
        return None
    
    # Extraer coordenadas del único rostro encontrado
    (x_val, y_val, ancho_val, altura_val) = rostros_coordenadas[0]
    
    # Recortar rostro de referencia sin márgenes adicionales
    rostro_referencia = vec_img_ref[y_val : y_val + altura_val, x_val : x_val + ancho_val]
    
    # Calcular vector de características del rostro de referencia
    imagen_referencia = calc_vector_embedding(rostro_referencia, modelo_embeddings)
    
    return imagen_referencia # Retornar embedding de rostro

# Buscar y procesar imagen de referencia en carpeta "ref"
def cargar_imagen_referencia(extensiones_lista, modelo_caffe, modelo_embeddings):
    carpeta_referencia = pathlib.Path("ref") # Carpeta fija para imagen de referencia
    
    # Verificar si la carpeta "ref" existe
    if not carpeta_referencia.exists():
        # No existe carpeta: crear y solicitar imagen
        carpeta_referencia.mkdir(exist_ok = True)
        
        input('Place reference image in "ref" folder and press Enter...')
    
    # Buscar primera imagen válida en carpeta "ref"
    while True:
        ruta_imagen_referencia = None
        
        # Buscar archivos con extensiones válidas
        for extension_val in extensiones_lista:
            for archivo_iter in carpeta_referencia.glob(f'*.{extension_val}'):
                if archivo_iter.is_file():
                    ruta_imagen_referencia = archivo_iter
                    
                    break # Cierra el primer ciclo For
            
            if ruta_imagen_referencia:
                break # Cierra el segundo ciclo For
        
        # Verificar si se encontró imagen de referencia
        if not ruta_imagen_referencia:
            # Carpeta existe pero sin imagen
            input('Place reference image in "ref" folder and press Enter...')
            
            continue # Volver al inicio del ciclo While
        
        # Procesar imagen de referencia encontrada
        vec_img_ref = procesar_imagen_referencia(ruta_imagen_referencia, modelo_caffe, modelo_embeddings)
        
        if vec_img_ref is not None:
            # Imagen válida encontrada: continuar
            return vec_img_ref
        else:
            # Imagen inválida: solicitar reemplazo
            input('Place valid image in "ref" folder and press Enter...')

# Inicializa el modelo de red neuronal DNN para detección de rostros
def cargar_modelos():
    # Buscar el primer archivo prototxt en el directorio actual (no recursivo)
    archivo_configuracion = None
    
    for archivo_iter in pathlib.Path('.').glob('*.prototxt'):
        archivo_configuracion = str(archivo_iter)
        
        break
    
    # Buscar el primer archivo caffemodel solo en el directorio actual (no recursivo)
    archivo_pesos_modelo = None
    
    for archivo_iter in pathlib.Path('.').glob('*.caffemodel'):
        archivo_pesos_modelo = str(archivo_iter)
        
        break
    
    # Buscar archivo del modelo de embeddings OpenFace
    archivo_embeddings = None
    
    for archivo_iter in pathlib.Path('.').glob('*.t7'):
        archivo_embeddings = str(archivo_iter)
        
        break
    
    # Cargar y retornar modelos
    return cv2.dnn.readNetFromCaffe(archivo_configuracion, archivo_pesos_modelo), cv2.dnn.readNetFromTorch(archivo_embeddings)

# PUNTO DE PARTIDA
# Lista de formatos de imagen soportados por OpenCV
extensiones_lista = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif', 'heic']

try:
    # Cargar modelos necesarios
    modelo_caffe, modelo_embeddings = cargar_modelos()
    
    # Obtener imagen de referencia al inicio
    vec_img_ref = cargar_imagen_referencia(extensiones_lista, modelo_caffe, modelo_embeddings)
    
    # Bucle principal del programa
    while True:
        # Solicitar directorio de entrada
        while True:
            directorio_entrada = input("Enter directory: ").strip('"\'')
            
            # Verificar que el directorio exista
            if not pathlib.Path(directorio_entrada).exists():
                print("Wrong directory\n")
            else:
                break
        
        # Ejecutar detección y comparación de rostros
        procesar_directorio_imagenes(directorio_entrada, extensiones_lista, vec_img_ref, modelo_caffe, modelo_embeddings)
        
except Exception as e:
    print("Model files not found")
        
    # Detener el programa
    input()
