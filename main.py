import cv2
import matplotlib.pyplot as plt
from morfologia import *

# Cargar imagen
imagen = cv2.imread('imgs/img_bin3.jpg', 0)

# Mostrar imagen original
mostrarIMG(imagen,'Imagen Original')

# Aplicar apertura tradicional

imagen_fr = frontera(imagen)

# Mostrar
mostrarIMG(imagen_fr,'Frontera Morfológica')

mostrarComparacion(imagen, imagen_fr, 'Imagen Original', 'Frontera Morfológica')
