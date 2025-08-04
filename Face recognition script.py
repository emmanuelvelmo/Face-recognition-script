import collections # defaultdict para agrupar archivos por carpeta
import pathlib # Manejo moderno de rutas de archivos y directorios
import cv2 # OpenCV: lectura de imágenes, detección de rostros, guardado
import numpy # Operaciones con arrays numéricos (coordenadas de rostros)

# CONFIGURACIÓN GLOBAL
UMBRAL_SIMILITUD = 0.75 # Umbral de similitud para coincidencias faciales (0-1)

# FUNCIONES
# Compara rostros usando similitud coseno y devuelve nivel de coincidencia
def comparar_rostros_similitud(codigo_referencia, codigo_comparar):
    # Normalizar vectores de características faciales
    codigo_ref_normalizado = codigo_referencia / numpy.linalg.norm(codigo_referencia)
    codigo_comp_normalizado = codigo_comparar / numpy.linalg.norm(codigo_comparar)
    
    # Calcular similitud coseno entre vectores normalizados
    return numpy.dot(codigo_ref_normalizado, codigo_comp_normalizado)

# Extrae vector de características faciales usando modelo de embeddings
def obtener_codigo_facial(rostro_imagen, modelo_embeddings):
    # Redimensionar rostro a tamaño requerido por OpenFace (96x96)
    rostro_redimensionado = cv2.resize(rostro_imagen, (96, 96))
    
    # Crear blob con normalización y configuración específica para OpenFace
    blob_rostro = cv2.dnn.blobFromImage(rostro_redimensionado, 1./255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    
    modelo_embeddings.setInput(blob_rostro) # Establecer blob como entrada del modelo
    
    # Ejecutar inferencia y obtener vector de características aplanado
    return modelo_embeddings.forward().flatten()

# Procesa imagen de referencia y extrae características del primer rostro encontrado
def procesar_imagen_referencia(ruta_imagen_ref, modelo_dnn, modelo_embeddings):
    # Cargar imagen de referencia desde archivo
    imagen_referencia = cv2.imread(str(ruta_imagen_ref))
    
    if imagen_referencia is None:
        return None
    
    try:
        # Detectar rostros en imagen de referencia
        rostros_coordenadas = coordenadas_rostros(imagen_referencia, modelo_dnn)
        
        # Verificar que se encontró exactamente un rostro
        if len(rostros_coordenadas) != 1:
            return None
        
        # Extraer coordenadas del único rostro encontrado
        (x, y, w, h) = rostros_coordenadas[0]
        
        # Recortar rostro de referencia sin márgenes adicionales
        rostro_referencia = imagen_referencia[y:y+h, x:x+w]
        
        # Obtener vector de características del rostro de referencia
        codigo_referencia = obtener_codigo_facial(rostro_referencia, modelo_embeddings)
        
        return codigo_referencia
        
    except Exception as e:
        return None

# Busca y procesa imagen de referencia en carpeta "ref"
def obtener_imagen_referencia(extensiones_lista, modelo_dnn, modelo_embeddings):
    carpeta_referencia = pathlib.Path("ref") # Carpeta fija para imagen de referencia
    
    # Verificar si la carpeta "ref" existe
    if not carpeta_referencia.exists():
        # Caso 1: No existe carpeta - crear y solicitar imagen
        carpeta_referencia.mkdir(exist_ok=True)
        print('Reference folder created. Place reference image and press Enter...')
        input()
    
    # Buscar primera imagen válida en carpeta "ref"
    while True:
        imagen_referencia_encontrada = None
        
        # Buscar archivos con extensiones válidas
        for extension_val in extensiones_lista:
            for archivo_iter in carpeta_referencia.glob(f'*.{extension_val}'):
                if archivo_iter.is_file():
                    imagen_referencia_encontrada = archivo_iter
                    break
            if imagen_referencia_encontrada:
                break
        
        # Verificar si se encontró imagen de referencia
        if not imagen_referencia_encontrada:
            # Caso 2: Carpeta existe pero sin imagen válida
            print('No reference image found. Place image in "ref" folder and press Enter...')
            input()
            continue
        
        # Procesar imagen de referencia encontrada
        codigo_referencia = procesar_imagen_referencia(imagen_referencia_encontrada, modelo_dnn, modelo_embeddings)
        
        if codigo_referencia is not None:
            # Caso 3: Imagen válida encontrada - continuar
            return codigo_referencia
        else:
            # Imagen inválida - solicitar reemplazo
            print('Invalid reference image (must contain exactly one face). Replace and press Enter...')
            input()

# Compara rostros encontrados con imagen de referencia y recopila coincidencias
def comparar_rostros_directorio(dir_imagen, rostros_imgs, codigo_referencia, modelo_embeddings):
    coincidencias_encontradas = [] # Lista para almacenar coincidencias con similitud
    
    # Comparar cada rostro detectado con la referencia
    for rostro_iter in rostros_imgs:
        try:
            # Obtener código de características del rostro actual
            codigo_rostro = obtener_codigo_facial(rostro_iter, modelo_embeddings)
            
            # Calcular similitud con rostro de referencia
            similitud = comparar_rostros_similitud(codigo_referencia, codigo_rostro)
            
            # Verificar si supera el umbral de similitud
            if similitud > UMBRAL_SIMILITUD:
                coincidencias_encontradas.append((str(dir_imagen), similitud))
                
        except Exception as e:
            continue # Ignorar errores en rostros individuales
    
    return coincidencias_encontradas

# Detecta rostros en una imagen usando red neuronal DNN con umbral de confianza
def coordenadas_rostros(imagen_val, modelo_dnn):
    alto_val, ancho_val = imagen_val.shape[:2] # Obtener dimensiones de la imagen original
    
    # Crear blob: redimensionar imagen a 300x300 y normalizar valores de píxeles
    blob_val = cv2.dnn.blobFromImage(cv2.resize(imagen_val, (300, 300)), 1.0, (300, 300), [104, 117, 123])
    
    modelo_dnn.setInput(blob_val) # Establecer blob como entrada de la red neuronal
    
    detecciones_val = modelo_dnn.forward() # Ejecutar inferencia y obtener detecciones
    
    rostros_coords = [] # Lista para almacenar coordenadas de rostros válidos
    
    # Procesar cada detección encontrada por la red neuronal
    for iter_val in range(detecciones_val.shape[2]):
        # Extraer nivel de confianza de la detección actual
        umbral_confianza = detecciones_val[0, 0, iter_val, 2]
        
        # Filtrar detecciones con confianza mayor al umbral (70%)
        if umbral_confianza > 0.7: # Umbral de confianza para reducir falsos positivos
            # Extraer coordenadas del rostro y escalar a dimensiones originales
            caja = detecciones_val[0, 0, iter_val, 3:7] * numpy.array([ancho_val, alto_val, ancho_val, alto_val])
            
            # Convertir coordenadas flotantes a enteros
            (x1, y1, x2, y2) = caja.astype("int")
            
            # Asegurar que las coordenadas estén dentro de los límites de la imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ancho_val, x2), min(alto_val, y2)
            
            # Convertir formato de coordenadas (x1,y1,x2,y2) a (x, y, ancho, alto)
            rostros_coords.append((x1, y1, x2-x1, y2-y1))
    
    return rostros_coords

# Procesa una imagen individual: carga, detecta rostros y recorta
def recortar_rostros(ruta_imagen, modelo_dnn):
    # Cargar imagen desde archivo usando OpenCV
    imagen_val = cv2.imread(str(ruta_imagen))
    
    try:
        # Ejecutar detección de rostros en la imagen
        rostros_coordenadas = coordenadas_rostros(imagen_val, modelo_dnn)
    except Exception as e:
        return []

    # Lista para almacenar imágenes recortadas de rostros
    rostros_imagenes = []
    
    # Recortar cada rostro detectado sin márgenes adicionales
    for (x, y, w, h) in rostros_coordenadas:
        # Recortar región exacta del rostro detectado
        rostro_recortado = imagen_val[y:y+h, x:x+w]
        
        # Agregar imagen al array de imágenes de rostros
        rostros_imagenes.append(rostro_recortado)
    
    return rostros_imagenes

# Procesamiento de imágenes y comparación con referencia
def procesar_directorio_imagenes(directorio_entrada, lista_carpetas_archivos, codigo_referencia, modelo_dnn, modelo_embeddings):
    # Lista para almacenar todas las coincidencias
    todas_coincidencias = []
    total_imagenes_procesadas = 0
    
    # Procesar cada carpeta y sus archivos de imagen
    for iter_carpeta, lista_archivos in lista_carpetas_archivos.items():
        # Procesar cada imagen individual de la carpeta actual
        for dir_imagen in lista_archivos:
            # Extraer rostros de la imagen actual
            rostros_encontrados = recortar_rostros(dir_imagen, modelo_dnn)
            
            # Comparar rostros si se encontró alguno
            if rostros_encontrados:
                coincidencias_imagen = comparar_rostros_directorio(dir_imagen, rostros_encontrados, codigo_referencia, modelo_embeddings)
                todas_coincidencias.extend(coincidencias_imagen)
                
            total_imagenes_procesadas += 1
    
    # Ordenar coincidencias por similitud (mayor primero)
    todas_coincidencias.sort(key=lambda x: x[1], reverse=True)
    
    return total_imagenes_procesadas, todas_coincidencias

# Organiza archivos de imagen agrupándolos por carpeta (búsqueda recursiva)
def agrupar_archivos_carpetas(directorio_origen, extensiones_lista):
    # Diccionario para carpetas y archivos de la misma
    dicc_carpetas_archivos = collections.defaultdict(list)
    
    # Buscar archivos para cada extensión de imagen soportada
    for extension_val in extensiones_lista:
        # Búsqueda recursiva en todas las subcarpetas usando patrón glob
        for archivo_iter in pathlib.Path(directorio_origen).rglob(f'*.{extension_val}'):
            # Verificar que sea un archivo válido y no un directorio
            if archivo_iter.is_file():
                carpeta_contenedora = archivo_iter.parent
                
                dicc_carpetas_archivos[carpeta_contenedora].append(archivo_iter)
    
    return dicc_carpetas_archivos

# Inicializa el modelo de red neuronal DNN para detección de rostros
def cargar_modelo_dnn():
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
    
    # Cargar y retornar modelo
    return cv2.dnn.readNetFromCaffe(archivo_configuracion, archivo_pesos_modelo)

# Inicializa el modelo de embeddings para extracción de características faciales
def cargar_modelo_embeddings():
    # Buscar archivo del modelo de embeddings OpenFace
    archivo_embeddings = None
    
    for archivo_iter in pathlib.Path('.').glob('*.t7'):
        archivo_embeddings = str(archivo_iter)
        
        break
    
    # Cargar y retornar modelo de embeddings
    return cv2.dnn.readNetFromTorch(archivo_embeddings)

# PUNTO DE PARTIDA
# Lista de formatos de imagen soportados por OpenCV
extensiones_lista = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif', 'heic']

try:
    # Cargar modelos necesarios
    modelo_dnn = cargar_modelo_dnn()
    modelo_embeddings = cargar_modelo_embeddings()
    
    # Obtener imagen de referencia al inicio
    codigo_referencia = obtener_imagen_referencia(extensiones_lista, modelo_dnn, modelo_embeddings)
    
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
        
        # Generar lista de directorios de imágenes en carpeta de entrada
        lista_carpetas_archivos = agrupar_archivos_carpetas(directorio_entrada, extensiones_lista)
        
        # Ejecutar detección, comparación de rostros
        total_imagenes, lista_coincidencias = procesar_directorio_imagenes(directorio_entrada, lista_carpetas_archivos, codigo_referencia, modelo_dnn, modelo_embeddings)
        
        # Mostrar separador visual para resultados
        print("-" * 36)
        
        # Mostrar resultados de procesamiento
        if total_imagenes > 0:
            # Mostrar coincidencias encontradas con formato detallado
            if lista_coincidencias:
                for indice_val, (ruta_imagen, similitud) in enumerate(lista_coincidencias, 1):
                    print(f"{indice_val}. [{similitud*100:.1f}%] {ruta_imagen}")
            else:
                print("No matches found")
        else:
            print("No images found")
        
        print("-" * 36 + "\n")
        
except Exception as e:
    print("Model files not found")
        
    # Detener el programa
    input()
