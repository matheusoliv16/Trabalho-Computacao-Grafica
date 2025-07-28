import matplotlib.pyplot as plt
import numpy as np
import os

# Gera os vértices e arestas de uma linha reta no espaço 3D (usando coordenada homogênea)
def constroi_vertices_linha(comprimento=4.0, direcao=np.array([1, 0, 0]), origem=np.array([0, 0, 0])):
    """
    Gera os vértices e arestas para uma linha reta em 3D.

    Args:
        comprimento (float): O comprimento da linha.
        direcao (np.ndarray): Vetor 3D indicando a direção da linha (será normalizado).
        origem (np.ndarray): Ponto 3D onde a linha começa.

    Returns:
        tuple: Uma tupla contendo:
            - vertices (np.ndarray): Array (2, 4) com os vértices (ponto inicial e final) em coordenadas homogêneas.
            - arestas (np.ndarray): Array (1, 2) com os índices dos vértices que formam a aresta.
    """
    # Normaliza o vetor direção (garante comprimento unitário)
    direcao = direcao / np.linalg.norm(direcao)
    # Ponto inicial (origem, em coordenada homogênea [x, y, z, 1])
    p0 = np.append(origem, 1)
    # Ponto final (origem + direção * comprimento, em coordenada homogênea)
    p1 = np.append(origem + direcao * comprimento, 1)
    # Cria o array de vértices (2 pontos) e as arestas que ligam esses pontos
    vertices = np.array([p0, p1])
    arestas = np.array([[0, 1]])
    return vertices, arestas

# Plota uma linha reta 3D usando Matplotlib. Salva em /imagens se 'salvar_como' for especificado.
def plotar_linha(comprimento=4.0, direcao=np.array([1, 0, 0]), origem=np.array([0, 0, 0]), cor="black", salvar_como=None):
    """
    Plota uma linha reta em 3D.

    Args:
        comprimento (float): O comprimento da linha.
        direcao (np.ndarray): Vetor 3D indicando a direção da linha (será normalizado).
        origem (np.ndarray): Ponto 3D onde a linha começa.
        cor (str): Cor da linha.
        salvar_como (str, optional): Nome do arquivo para salvar a imagem na pasta 'imagens/'.
                                     Se None, a imagem não é salva.
    """
    vertices, arestas = constroi_vertices_linha(comprimento, direcao, origem)
    v = vertices[:, :3]  # Desconsidera coordenada homogênea para plotagem

    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    # Plota a linha conectando os dois vértices
    ax.plot([v[0, 0], v[1, 0]], [v[0, 1], v[1, 1]], [v[0, 2], v[1, 2]], color=cor, linewidth=3)

    # Ajusta limites dos eixos com folga
    ax.set_xlim(v[:, 0].min() - 1, v[:, 0].max() + 1)
    ax.set_ylim(v[:, 1].min() - 1, v[:, 1].max() + 1)
    ax.set_zlim(v[:, 2].min() - 1, v[:, 2].max() + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Linha Reta")

    # Se solicitado, salva a imagem em /imagens com o nome especificado
    if salvar_como:
        caminho_pasta = os.path.join(os.getcwd(), "imagens")
        os.makedirs(caminho_pasta, exist_ok=True)
        caminho_arquivo = os.path.join(caminho_pasta, salvar_como)
        plt.savefig(caminho_arquivo, bbox_inches="tight")
    plt.show()

# Função alternativa para criar uma linha reta (sem coordenada homogênea)
def constroi_linha_reta(comprimento=4.0, direcao=np.array([1, 0, 0]), origem=np.array([0, 0, 0])):
    """
    Gera os vértices e arestas para uma linha reta em 3D (sem coordenada homogênea).

    Args:
        comprimento (float): O comprimento da linha.
        direcao (np.ndarray): Vetor 3D indicando a direção da linha (será normalizado).
        origem (np.ndarray): Ponto 3D onde a linha começa.

    Returns:
        tuple: Uma tupla contendo:
            - vertices (list): Lista de listas com os vértices (ponto inicial e final) [x, y, z].
            - arestas (list): Lista de listas com os índices dos vértices que formam a aresta.
    """
    # Normaliza o vetor direção
    direcao = direcao / np.linalg.norm(direcao)
    # Define pontos inicial e final da linha (sem coordenada homogênea)
    p0 = origem
    p1 = origem + direcao * comprimento
    vertices = [p0.tolist(), p1.tolist()]
    arestas = [[0, 1]]
    return vertices, arestas