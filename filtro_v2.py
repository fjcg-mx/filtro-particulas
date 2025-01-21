
import cv2
import numpy as np
import time
import random

#------------------ Funciones de Color utiles -------------------

def rgb2hsv(color):
    # Aseguramos que el color esté en formato np.uint8
    rgb_color = np.array(color, dtype=np.uint8)
    
    # Convertir RGB a HSV utilizando OpenCV
    hsv_color = cv2.cvtColor(np.reshape(rgb_color, (1, 1, 3)), cv2.COLOR_RGB2HSV)
    
    # Devuelvo el valor en HSV como un array
    return np.array(hsv_color[0][0])

#-------------- 1. Inicialización de partículas -----------------------

def crear_particulas(num_particulas, max1, max2):
  """
  Crea las partículas en posiciones aleatorias.

  Parametros:
  - num_particulas: Numero de particulas a crear.
  - max1: Valor maximo de la posicion x.
  - max2: Valor maximo de la posicion y.
  Retorno:
  - lista de partículas.
  """
  # Validar que max1 y max2 sean mayores que 0
  if max1 <= 0 or max2 <= 0:
    raise ValueError("Los valores de max1 y max2 deben ser mayores que 0.")
  
  # Generar las partículas aleatorias
  particulas = np.random.randint(0, [max1, max2], size=(num_particulas, 2))
  return particulas

#--------- 2. Propagación de partículas (predicción) -----------------
def particula_obtener_color(particula, imagen, tam_ventana=3):
  """
  Obtiene la región alrededor de una partícula en la imagen.

  Parametros:
  - particula: Posición de la partícula (x, y)
  - imagen: Imagen actual
  - tam_ventana: Tamaño de la ventana alrededor de la partícula

  Retorno:
  - region: Subimagen de la región alrededor de la partícula
  """
  x, y = particula
  mitad_ventana = tam_ventana // 2
  # Extraer la región alrededor de la partícula
  region = imagen[max(0, y - mitad_ventana):min(imagen.shape[0], y + mitad_ventana),
                  max(0, x - mitad_ventana):min(imagen.shape[1], x + mitad_ventana)]
  return region

def particulas_actualizar(imagen, particulas, prediccion, radio = 3, color_objetivo_hsv = None, tolerancia_color=None):
  """
  Actualiza las partículas agregando movimiento aleatorio.

  Parametros:
  - imagen: Imagen actual.
  - particulas: Lista de posiciones de las partículas.
  - prediccion: Rango de movimiento aleatorio.
  - radio: Radio de visualización de las partículas.
  - color_objetivo_hsv: Color objetivo en HSV (opcional, si quieres cambiar el color basado en eso).
  - tolerancia_color: Tolerancia al color
  """
  largo, ancho, canales = imagen.shape
  for i in range(len(particulas)):
    particulas[i][0] += random.randint(-prediccion, prediccion) # Movimiento aleatorio en x
    particulas[i][1] += random.randint(-prediccion, prediccion) # Movimiento aleatorio en y

    # Asegurarse de que las partículas no se salgan de la imagen
    particulas[i][0] = np.clip(particulas[i][0], 0, ancho - 1)
    particulas[i][1] = np.clip(particulas[i][1], 0, largo - 1)

    # Extraer la región alrededor de la partícula
    region = particula_obtener_color(particulas[i], imagen)

    # Calcular el peso del color para esta partícula (también devuelve la distancia)
    peso_color = calcular_peso_color(region, color_objetivo_hsv, tolerancia_color)
    
    # Si el peso de color es alto (distancia pequeña), cambiar el color de la partícula
    if peso_color > 0.3:  # Este umbral depende de la tolerancia que se use
        # Cambiar el color a verde, por ejemplo
        cv2.circle(imagen, tuple(particulas[i]), radio, (0, 255, 0), 2)  # Verde (BGR)
    else:
        # Si la partícula no está cerca del color objetivo, mantén el color original (rojo, por ejemplo)
        cv2.circle(imagen, tuple(particulas[i]), radio, (0, 0, 255), 2)  # Rojo (BGR)


#-------------------- 3. Medición de la probabilidad -------------------

def calcular_peso_color(region, color_objetivo_hsv, tolerancia):
  """
  Calcula el peso de una partícula basada en el color de la región que la rodea,
  comparado con el color objetivo y utilizando una tolerancia.

  Parametros:
  - region: La región de la imagen que rodea la partícula.
  - color_objetivo_hsv: El color objetivo en espacio HSV.
  - tolerancia: La tolerancia para la diferencia de color. Por defecto 30.

  Retorno:
  - Peso calculado basado en la distancia entre los colores.
  """
  region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV) # Convertir la región a espacio HSV
  mean_color = np.mean(region_hsv, axis=(0, 1))  # Promedio de la región (media de los valores HSV)
  
  # Calcular la distancia Euclidiana entre el color promedio y el color objetivo
  distance = np.linalg.norm(mean_color - color_objetivo_hsv)
  peso = peso = np.exp(-distance / tolerancia)  # Decaimiento exponencial
  return  peso

# Funciones de Movimiento y Pesos
def calcular_peso_movimiento(mapa_calor, particula):
  """
  Calcula el peso de la partícula basado en el mapa de calor.

  Parametros:
  - mapa_calor: Mapa de calor que muestra las diferencias entre fotogramas.
  - particula: Posición de la partícula (x, y)

  Retorno:
  - Peso de movimiento normalizado
  """
  x, y = particula
  # Asegurarse de que las partículas estén dentro de los límites de la imagen
  y = np.clip(y, 0, mapa_calor.shape[0] - 1)
  x = np.clip(x, 0, mapa_calor.shape[1] - 1)
  peso_movimiento = mapa_calor[y, x]  # Obtener la intensidad en la posición de la partícula
  return peso_movimiento / 255.0  # Normalizar el peso

#--------------------- 4. Actualización de pesos: ---------------------

def particulas_calcular_peso(imagen, particula, mapa_calor, color_objetivo_hsv, alpha, beta, tolerancia_color):
  """
  Calcula el peso final de una partícula basado en movimiento y color.

  Parametros:
  - imagen: Imagen actual.
  - particula: Posición de la partícula.
  - mapa_calor: Mapa de calor.
  - target_color_hsv: Color objetivo.
  - alpha: Coeficiente de peso para el movimiento.
  - beta: Coeficiente de peso para el color.
  - tolerancia_color: Tolerancia al color
  Retorno:
  - Peso final de la partícula.
  """
  # Peso basado en el movimiento
  peso_movimiento = calcular_peso_movimiento(mapa_calor, particula)

  # Peso basado en el color
  region = particula_obtener_color(particula, imagen)
  peso_color = calcular_peso_color(region, color_objetivo_hsv, tolerancia_color)

  # Combinamos ambos pesos 
  #final_weight = peso_movimiento * color_weight
  # Combinar los pesos con ponderación ajustada
  peso_final = alpha * peso_movimiento + beta * peso_color

  return peso_final

#------------------- 5. Remuestreo de partículas ----------------------

def particulas_remuestreo(particulas, pesos):
  """
  Realiza el re-muestreo de partículas basado en sus pesos.

  Parametros:
  - particulas: Lista de posiciones de las partículas.
  - pesos: Pesos de las partículas.

  Retorno:
  - Nueva lista de partículas re-muestreadas.
  """
  # Normalizar los pesos para que sumen 1
  suma_pesos = np.sum(pesos)
  pesos_normalizados = 0
  if suma_pesos == 0:
    # Si todos los pesos son cero, asignar pesos uniformes (esto puede ser un caso raro)
    pesos_normalizados = np.ones(len(pesos)) / len(pesos)
  else:
    pesos_normalizados = pesos / suma_pesos
  
  # Asegurarse de que los pesos sumen exactamente 1 (debido a errores de redondeo)
  pesos_normalizados = np.clip(pesos_normalizados, 0, 1)
  pesos_normalizados /= np.sum(pesos_normalizados)  # Re-normalizamos para que sumen exactamente 1
  
  # Selección aleatoria de partículas basada en sus pesos
  indices = np.random.choice(len(particulas), size=len(particulas), p=pesos_normalizados)
  nuevas_particula = particulas[indices]
  return nuevas_particula


# ---------------------- Filtro de partículas V2--------------------------

"""
Procesa el video - seguimiento de objeto con filtro de particulas.

Parametro:
- archivo: Ruta/Nombre del archivo de video.
"""
def filtros_particulas_v2(archivo):
  
  # Configuración de colores, para el color objetivo en HSV 
  azulBajo = np.array([100, 100, 20], np.uint8) #Azul en un rango
  azulAlto = np.array([140, 255, 255], np.uint8)
  color_objetivo_hsv_rojo = np.array([0, 255, 255]) # (Para el rojo, por ejemplo)
  color_objetivo_hsv_azul = np.array([120, 255, 255]) # En nustro caso sera azul puro
  # 29, 44, 123
  # 91,93,175
  # 75,168,194
  # 19,48,118
  color_rgb_azul_muestra = (19,48,118)  # Color RGB
  
  color_objetivo_hsv_azul_muestra = rgb2hsv(color_rgb_azul_muestra)
  
  color_objetivo_hsv = color_objetivo_hsv_azul_muestra
  print(color_objetivo_hsv)

  tolerancia_color = 73 # Tolerancia para la diferencia de color.

  # Para el filtro de particulas
  num_particulas = 200  # Número de partículas a usar en el filtro.
  radio_particula = 1 # Radio de cada partícula.
  # Ajustar los coeficientes alpha y beta para ponderar la importancia de movimiento y color
  alpha = 0.0  # Peso para el movimiento
  beta = 0.5  # Peso para el color
  prediccion = 13 # Rango de movimiento aleatorio de las partículas.
  
  # Captura de video y sus caracteristicas
  video = cv2.VideoCapture(archivo)
  if not video.isOpened():
    raise ValueError("No se pudo abrir el archivo de video.")
  
  ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  largo = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps= int(video.get(cv2.CAP_PROP_FPS))
  print(f"Imagen: {ancho} x {largo}")
  f = 0

  # Inicializar partículas (posición aleatoria)
  particulas = crear_particulas(num_particulas, ancho, largo)
  
  # Leer el primer fotograma
  ret, imagen = video.read()
  if not ret:
    print("Error al leer el primer fotograma")
    return  # Detener el proceso si no se puede leer el primer fotograma
  
  # Inicializar con el primer fotograma
  previous_frame = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

  while video.isOpened():

    ret, imagen = video.read()

    if ret: # Solo proceder si se ha leído correctamente el fotograma

      frame = imagen.copy()
      cv2.putText(imagen,str(f),(ancho-100,largo-50),cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
      
      #frameHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
      #maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
      #maskAzulvis = cv2.bitwise_and(imagen, imagen, mask= maskAzul)

      cv2.imshow('Video', imagen) # Muestra el fotograma
      #cv2.imshow('maskAzul', maskAzul)
      #cv2.imshow('maskAzulvis', maskAzulvis)

      # Convertir el frame a escala de grises
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      #cv2.imshow('GRIS', gray)

      # Detectar el movimiento usando la diferencia entre el cuadro actual y el anterior
      frame_diff = np.abs(previous_frame.astype(np.int16) - gray.astype(np.int16))  # Diferencia de cuadros
      frame_diff = np.clip(frame_diff, 0, 255).astype(np.uint8)  # Limitar el rango a 0-255
      
      # Crear un mapa de calor sumando las diferencias de las áreas
      mapa_calor = np.uint8(frame_diff)
      #cv2.imshow('Mapa de Calor', mapa_calor)

      # Filtrar las partículas según el mapa de calor
      # Calcular el peso de cada partícula basado en la intensidad del mapa de calor y color
      weights = np.array([particulas_calcular_peso(imagen, p, mapa_calor, color_objetivo_hsv, alpha, beta, tolerancia_color) for p in particulas])
      
      # Re-muestrear las partículas en función de sus pesos
      particulas = particulas_remuestreo(particulas, weights)
              
      # Actualizar las partículas con el movimiento
      particulas_actualizar(frame, particulas, prediccion, radio_particula, color_objetivo_hsv, tolerancia_color)

      
      # Mostrar el video con las partículas y la detección de movimiento
      cv2.imshow('Deteccion de movimiento con filtro de particulas v2', frame)
      
      # Actualizar la imagen anterior
      previous_frame = gray

      # delay
      time.sleep(0+1/fps)

      if cv2.waitKey(1) & 0xFF == ord('s'): # Termina cuando se apriete s(salir)
        break
      
      # incremento del conteo de fotogramas
      f += 1

    else:
      video.release()
      break
  print("Finalizado...")
  cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------
# Llamar a la función main
if __name__ == "__main__":
  nombre_archivo = "walking.mp4"
  filtros_particulas_v2(nombre_archivo)