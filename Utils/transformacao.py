import numpy as np

# Escala todos os vértices pelo fator especificado
def escala(vertices, fator):
    v = np.array(vertices)            # Garante que é um array numpy
    return v * fator                  # Multiplica cada coordenada pelo fator de escala

# Realiza a translação (deslocamento) dos vértices
def translacao(vertices, deslocamento):
    v = np.array(vertices)            # Converte para array numpy
    deslocamento = np.array(deslocamento)  # Converte deslocamento para array
    return v + deslocamento           # Soma o vetor de deslocamento a cada vértice

# Rotaciona os vértices em torno de um eixo ('x', 'y' ou 'z') por um ângulo em graus
def rotacao(vertices, eixo, angulo_graus):
    v = np.array(vertices)            # Converte para array numpy
    angulo_rad = np.radians(angulo_graus)  # Converte o ângulo de graus para radianos
    # Monta a matriz de rotação correspondente ao eixo
    if eixo == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angulo_rad), -np.sin(angulo_rad)],
                      [0, np.sin(angulo_rad), np.cos(angulo_rad)]])
    elif eixo == 'y':
        R = np.array([[np.cos(angulo_rad), 0, np.sin(angulo_rad)],
                      [0, 1, 0],
                      [-np.sin(angulo_rad), 0, np.cos(angulo_rad)]])
    elif eixo == 'z':
        R = np.array([[np.cos(angulo_rad), -np.sin(angulo_rad), 0],
                      [np.sin(angulo_rad), np.cos(angulo_rad), 0],
                      [0, 0, 1]])
    # Multiplica cada vértice pela matriz de rotação transposta
    return v @ R.T
