import numpy as np
import Loader
from matplotlib import pyplot as plt


def find_next_white(pos: tuple, img):
    (shapeY, shapeX) = img.shape
    n = []
    for y in np.arange(pos[0], shapeY - 5):
        if img[y][pos[0]] > 0:
            n.append((y, pos[0]))
    return n


def trim_error_measures(measures, positions):
    d2 = []
    p2 = []
    it2 = 0
    for d in measures:
        if d > (np.sum(measures) / len(measures) * 0.5):
            d2.append(d)
            p2.append(positions[it2])
        it2 = it2 + 1

    return d2, p2


def find_vertical_lines(edgeImage):
    inverse = Loader.inverse_img(edgeImage)
    Loader.print_image(inverse)
    (shapeY, shapeX) = edgeImage.shape
    lineas = []
    for x in np.arange(1, shapeX - 5):
        y = shapeY-5
        # Busco el primer pixel blanco
        while inverse[y][x] == 0:
            y = y - 1
        # Estoy en el primer punto vertical blanco
        # Avanzo hacia arriba hasta el primer hueco negro
        while inverse[y][x] != 0:
            y = y - 1
        # Estoy en el principio de cornea en negro
        while inverse[y][x] == 0:
            y = y - 1
        # He acabado la cornea, empieza lo que quiero guardar, blanco
        vertical = []
        while inverse[y][x] != 0:
            vertical.append((y, x))
            y = y - 1
        # Acabo el trozo que me interesa, lo añado a lineas
        lineas.append(vertical.copy())
    return lineas


def calculate_differences(lineas):
    diferencias = []
    posiciones = []
    for linea in lineas:
        distancia = linea[0][0] - linea[len(linea) - 1][0]
        posX = linea[0][1]
        diferencias.append(distancia)
        posiciones.append(posX)

    return trim_error_measures(diferencias, posiciones)


def lines_image(lines, img):
    #DIUJA LAS LINEAS, DEJANDO ALGUNOS HUECOS
    #TODO PINTAR SOBRE IMAGEN ORIGINAL COMO FONDO
    output = img.copy()
    it = 0
    for linea in lines:
        if it % 2 == 0:
            for dot in linea:
                output[dot[0]][dot[1]] = 255
        it = it+1
    return output


def draw_graph_distance(measures, positions):
    plt.plot(positions, measures)
    plt.xticks(np.arange(min(positions)-2, max(positions), 100.0)), plt.yticks(np.arange(min(measures), max(measures), 1.0))
    plt.show()
