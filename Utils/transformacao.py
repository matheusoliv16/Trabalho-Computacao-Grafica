import numpy as np

def transformar(vertices, escala=1.0, rotacao=None, translacao=None):
    # vertices: (N, 3)
    v = np.array(vertices)
    if escala != 1.0:
        v = v * escala
    if rotacao is not None:
        v = v @ rotacao.T
    if translacao is not None:
        v = v + np.array(translacao)
    return v
