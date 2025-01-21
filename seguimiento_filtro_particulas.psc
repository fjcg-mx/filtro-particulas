Funcion posicion_particulas <- inicializar_particulas ( num_particulas, imagen_hsv_shape )
	// generar las posiciones de las particulas 
	// de manera uniforme en toda la imagen
Fin Funcion

Funcion re_particulas <- remuestrear_particulas ( particulas, pesos )
	// seleccion de las particulas mas probables
Fin Funcion

Funcion nuevos_pesos <- actualizar_pesos ( partiulas, imagen_hsv, radio_particula )
	// asignacion de pesos de acuerdo a lo observado
Fin Funcion

Funcion propagar_particulas <- propagar_particulas ( particulas, imagen_hsv_shape, prediccion )
	// mover las particulas 
Fin Funcion

Algoritmo filtro_particulas
	captura = captura_video
	radio_particula<-1 // radio de la partic
	particulas<-vacio
	num_particulas<-0
	Mientras captura_esta_abierta Hacer
		frame<-captura_leer_frame
		imagen_hsv<-convertir_a_formato_HSV
		Si len_particulas=0 Entonces
			particulas<-inicializar_particulas(num_particulas, imagen_hsv_shape)
		Fin Si
		particulas<-propagar_particulas(particulas, imagen_hsv_shape, prediccion=5)
		pesos<-actualizar_pesos(particulas, imagen_hsv, radio_particula)
		particulas<-remuestrear_particulas(particulas, pesos)
		Escribir particulas_imagen
	Fin Mientras
FinAlgoritmo
