import numpy as np
import cv2
import time

#-------------- 1. Inicialización de partículas -----------------------
def inicializar_particulas(num_particulas, image_shape):
  # Validar que image_shape tenga valores mayores que 0
  max2, max1, otro = image_shape
  # Generar las partículas aleatorias
  particulas = np.random.randint(0, [max1, max2], size=(num_particulas, 2))
  print(particulas)
  return particulas

#--------- 2. Propagación de partículas (predicción) -----------------

# Función para propagar las partículas (simulación de movimiento)
def propagar_particulas(particulas, image_shape, prediccion=5):
  nuevas_particulas = []
  for (x, y) in particulas:
    # Desplazamos las partículas aleatoriamente dentro de la imagen
    nuevo_x = np.clip(x + np.random.randint(-prediccion, prediccion), 0, image_shape[1]-1)
    nuevo_y = np.clip(y + np.random.randint(-prediccion, prediccion), 0, image_shape[0]-1)
    nuevas_particulas.append((nuevo_x, nuevo_y))
  return nuevas_particulas

#-------------------- 3. Medición de la probabilidad -------------------

# Función de pertenencia
def funcion_pertenencia(rango, valor_actual):
  if rango[0] <= valor_actual <= rango[1]:
    return 1
  else:
    return 0

# Calcular el número de componentes dentro de sus rangos (variable aleatoria X)
def calcular_variable_aleatoria(matiz, saturacion, valor):
  # tolerancias
  desviacion_H = 20
  desviacion_S = 50
  desviacion_V = 50
  #color_objetivo = (111,214,118)
  color_objetivo = (115,214,118)
  matiz_obj, saturacion_obj, valor_obj = color_objetivo

  # Rango para el color azul
  rango_hue = [matiz_obj - desviacion_H, matiz_obj + desviacion_H]  # Matiz: 111° ± 20°
  rango_saturacion = [saturacion_obj - desviacion_S, saturacion_obj + desviacion_S]  # Saturación: 214 ± 50
  rango_value = [valor_obj - desviacion_V, valor_obj + desviacion_V]  # Valor: 118 ± 50
  
  # Calcular la pertenencia para cada componente (H, S, V)
  matiz_pertenencia = funcion_pertenencia(rango_hue, matiz)
  saturation_pertenencia = funcion_pertenencia(rango_saturacion, saturacion)
  valor_pertenencia = funcion_pertenencia(rango_value, valor)
  
  # La variable aleatoria X es la suma de los valores de pertenencia
  X = matiz_pertenencia + saturation_pertenencia + valor_pertenencia
  return X

def calcular_peso_particula(particula, imagen_hsv, tam_ventana=5):
  # Probabilidad de cada valor de X (número de componentes dentro del rango)
  probabilidades_X = {0: 1/8, 1: 3/8, 2: 3/8, 3: 1/8}  # Valores determinados en la documentacion.
  probabilidades_acumuladas = [1/8, 4/8, 7/8, 1]  # Distribución acumulada de probabilidades

  probabilidad = 0
  x,y = particula
  mitad_ventana = tam_ventana // 2
  
  # Extraer la región alrededor de la partícula
  region = imagen_hsv[max(0, y - mitad_ventana):min(imagen_hsv.shape[0], y + mitad_ventana),
                  max(0, x - mitad_ventana):min(imagen_hsv.shape[1], x + mitad_ventana)]
  color_promedio = np.mean(region, axis=(0, 1))  # Promedio de la región (media de los valores HSV)
  
  hue, saturation, value = color_promedio
  x = calcular_variable_aleatoria(hue, saturation, value)
  # Definir los pesos según el valor de X
  #peso = probabilidades_acumuladas[x]
  if x == 0:
    peso = 1/8  # Ningún componente dentro del rango
  elif x == 1:
    peso = 4/8  # 1 componente dentro del rango
  elif x == 2:
    peso = 7/8  # 2 componentes dentro del rango
  elif x == 3:
    peso = 1  # Todos los componentes dentro del rango
  # CDF de la variable aleatoria x
  return peso

#--------------------- 4. Actualización de pesos: ---------------------

def actualizar_pesos(particulas, imagen_hsv, radio):
  pesos = []
  
  # Probabilidades acumuladas para los valores de X (0, 1, 2, 3)
  # CDF de la variable aleatoria X
  #cdf = np.cumsum(list(probabilidades_X.values()))

  # Evaluar el peso de cada partícula basado en X (variable aleatoria)
  for particula in particulas:
    peso = calcular_peso_particula(particula, imagen_hsv, radio)
    pesos.append(peso)
  
  # Normalizar los pesos para que sumen 1    
  total_pesos = sum(pesos)
  pesos_normalizados = [peso / total_pesos for peso in pesos]
  
  return pesos_normalizados

#------------------- 5. Remuestreo de partículas ----------------------

# Función de remuestreo
def remuestrear_particulas(particulas, pesos):
  particulas_remuestreadas = []
  # Selección de partículas basada en los pesos normalizados
  for _ in range(len(particulas)):
    seleccion = np.random.choice(range(len(particulas)), p=pesos)
    particulas_remuestreadas.append(particulas[seleccion])
  
  return particulas_remuestreadas

#------------------- 6. Visualizacion de partículas ----------------------

def dibujar_cuadrado_por_dispersión(particulas, imagen):
    # Obtener las coordenadas mínimas y máximas para cada dimensión
    min_x = min(p[0] for p in particulas)
    max_x = max(p[0] for p in particulas)
    min_y = min(p[1] for p in particulas)
    max_y = max(p[1] for p in particulas)

    # Determinar el tamaño del cuadrado (el mayor rango de dispersión)
    lado = max(max_x - min_x, max_y - min_y)

    # Calcular las coordenadas del rectángulo que cubre la dispersión
    punto_inicial = (min_x, min_y)
    punto_final = (min_x + lado, min_y + lado)

    # Dibujar el cuadrado en la imagen (color verde, grosor 2)
    cv2.rectangle(imagen, punto_inicial, punto_final, (0, 255, 0), 2)

def centroide(particulas):
  # Calcular el promedio de las coordenadas para cada dimensión
  num_particulas = len(particulas)
  suma_x = sum(p[0] for p in particulas)
  suma_y = sum(p[1] for p in particulas)

  centroide = (suma_x / num_particulas, suma_y / num_particulas)
  print(f"La posición central (centroide) es: {centroide}")

  return centroide

# Visualizar las partículas
def visualizar_particulas(particulas, imagen, radio):
  # Convertir la imagen de BGR a HSV, en este se realizan el procesamiento
  imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
  for particula in particulas:
    peso = calcular_peso_particula(particula, imagen_hsv, radio)
    # Si el peso de color es alto (distancia pequeña), cambiar el color de la partícula
    if peso >= 0.875:  # Este umbral depende de la tolerancia que se use
        # Cambiar el color a verde, por ejemplo
        cv2.circle(imagen, particula, radio, (0, 255, 0), 2)  # Verde (BGR)
    else:
        # Si la partícula no está cerca del color objetivo, mantén el color original (rojo, por ejemplo)
        cv2.circle(imagen, particula, radio, (0, 0, 255), 2)  # Rojo (BGR)
  dibujar_cuadrado_por_dispersión(particulas, imagen)

# ----------------- Filtro de partículas v1 ---------------------
def filtros_particulas_v1(video_path, num_particulas=100):
    cap = cv2.VideoCapture(video_path)
    
    radio_particula = 2 # Radio de cada partícula.

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    largo = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = 0
    particulas = []  # Inicializamos las partículas
    
    while cap.isOpened():
        ret, frame  = cap.read()
        if not ret:
            break
        # video original
        cv2.imshow('Video original en formato RGB', frame ) # Muestra la imagen actual

        # Convertir la imagen de BGR a HSV, en este se realizan el procesamiento
        imagen_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # copia de la imagen para mostrar los resultados
        imagen = frame .copy()
        cv2.putText(frame ,str(f),(ancho-100,largo-50),cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))
        
        if len(particulas) == 0:  # Inicialización de partículas en el primer fotograma
            particulas = inicializar_particulas(num_particulas, imagen_hsv.shape)
            print(f"Imagen: {imagen_hsv.shape[0]} x {imagen_hsv.shape[1]}")
        # Propagamos las partículas
        particulas = propagar_particulas(particulas, imagen_hsv.shape, prediccion=10)
        
        # Actualizamos los pesos de las partículas
        pesos = actualizar_pesos(particulas, imagen_hsv, radio_particula)
        
        # Remuestreo de partículas
        particulas = remuestrear_particulas(particulas, pesos)
        
        # Visualizar las partículas
        visualizar_particulas(particulas, imagen, radio_particula)
        
        cv2.imshow('Video en formato HSV', imagen_hsv)
        cv2.imshow('Deteccion de movimiento con filtro de particulas', imagen)
        time.sleep(1/30)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        # incremento del conteo de fotogramas
        f += 1
    cap.release()
    cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------
# Llamar a la función main
if __name__ == "__main__":
  nombre_archivo = "walking.mp4"
  # Probando el filtro
  filtros_particulas_v1('walking.mp4', num_particulas=400)
