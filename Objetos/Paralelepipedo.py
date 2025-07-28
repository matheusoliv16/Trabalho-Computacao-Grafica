import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import os

# Gera um grid regular de vértices para o paralelepípedo subdividido
def constroi_vertices_paralelepipedo(comprimento, largura, altura, n_div=1):
    """
    Gera um grid de vértices 3D subdivididos igualmente nas três dimensões.
    Retorna array (N, 3) com todos os vértices do cubo, em ordem de grade.

    Args:
        comprimento (float): O comprimento do paralelepípedo no eixo X.
        largura (float): A largura do paralelepípedo no eixo Y.
        altura (float): A altura do paralelepípedo no eixo Z.
        n_div (int): Número de divisões em cada dimensão. n_div=1 cria um paralelepípedo simples.

    Returns:
        np.ndarray: Array de vértices (N, 3).
    """
    # Cria pontos igualmente espaçados em cada eixo
    x = np.linspace(-comprimento/2, comprimento/2, n_div+1)
    y = np.linspace(-largura/2,    largura/2,    n_div+1)
    z = np.linspace(-altura/2,     altura/2,     n_div+1)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    # Empacota os pontos em uma lista de vértices (N, 3)
    vertices = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    return vertices

# Gera as faces triangulares do paralelepípedo subdividido
def constroi_faces_paralelepipedo(n_div=1):
    """
    Gera as faces triangulares da malha subdividida do paralelepípedo.
    Retorna array (M, 3) com índices de vértices para cada triângulo.

    Args:
        n_div (int): Número de divisões em cada dimensão, usado para calcular os índices dos vértices.

    Returns:
        np.ndarray: Array de faces (M, 3).
    """
    faces = []
    n = n_div + 1  # Número de pontos por eixo

    # Função lambda que calcula o índice linear no grid 3D a partir dos índices i, j, k
    idx = lambda i, j, k: i * n * n + j * n + k

    # Faces onde z é constante (bases inferior e superior)
    for k in [0, n_div]:
        for i in range(n_div):
            for j in range(n_div):
                v0 = idx(i,   j,   k)
                v1 = idx(i+1, j,   k)
                v2 = idx(i+1, j+1, k)
                v3 = idx(i,   j+1, k)
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

    # Faces onde y é constante (frontal e traseira)
    for j in [0, n_div]:
        for i in range(n_div):
            for k in range(n_div):
                v0 = idx(i,   j, k)
                v1 = idx(i+1, j, k)
                v2 = idx(i+1, j, k+1)
                v3 = idx(i,   j, k+1)
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

    # Faces onde x é constante (lateral esquerda e direita)
    for i in [0, n_div]:
        for j in range(n_div):
            for k in range(n_div):
                v0 = idx(i, j,   k)
                v1 = idx(i, j+1, k)
                v2 = idx(i, j+1, k+1)
                v3 = idx(i, j,   k+1)
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

    return np.array(faces)

# Plota o paralelepípedo em 3D e salva a imagem se solicitado
def plotar_paralelepipedo(comprimento, largura, altura, n_div=1, cor="lightblue", salvar_como=None):
    """
    Plota o paralelepípedo em 3D usando Matplotlib.

    Args:
        comprimento (float): O comprimento do paralelepípedo no eixo X.
        largura (float): A largura do paralelepípedo no eixo Y.
        altura (float): A altura do paralelepípedo no eixo Z.
        n_div (int): Número de divisões para a malha.
        cor (str): Cor das faces do paralelepípedo.
        salvar_como (str, optional): Nome do arquivo para salvar a imagem na pasta 'imagens/'.
                                     Se None, a imagem não é salva.
    """
    vertices = constroi_vertices_paralelepipedo(comprimento, largura, altura, n_div)
    faces = constroi_faces_paralelepipedo(n_div)
    fig = plt.figure(figsize=(10, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # Constrói uma lista de polígonos a partir das faces
    poly3d = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=cor, edgecolors="black", linewidths=0.6, alpha=1)
    ax.add_collection3d(collection)

    # Centraliza e ajusta limites do gráfico para melhor visualização
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Paralelepípedo")
    plt.tight_layout()

    # Salva a imagem na pasta 'imagens', se solicitado
    if salvar_como:
        caminho_pasta = os.path.join(os.getcwd(), "imagens")
        os.makedirs(caminho_pasta, exist_ok=True)
        caminho_arquivo = os.path.join(caminho_pasta, salvar_como)
        plt.savefig(caminho_arquivo, bbox_inches="tight")
    plt.show()

# Gera os vértices e arestas (wireframe) do paralelepípedo subdividido
def gerar_paralelepipedo(comprimento, largura, altura, n_div=1):
    """
    Retorna arrays de vértices (xyz) e arestas (índices) do paralelepípedo subdividido.

    Args:
        comprimento (float): O comprimento do paralelepípedo no eixo X.
        largura (float): A largura do paralelepípedo no eixo Y.
        altura (float): A altura do paralelepípedo no eixo Z.
        n_div (int): Número de divisões para a malha.

    Returns:
        tuple: Uma tupla contendo:
            - vertices (np.ndarray): Array de vértices (N, 3).
            - arestas (np.ndarray): Array de arestas (M, 2), representando conexões entre vértices.
    """
    vertices = constroi_vertices_paralelepipedo(comprimento, largura, altura, n_div)
    n = n_div + 1
    arestas = []

    # Ligações paralelas ao eixo x
    for i in range(n_div):
        for j in range(n):
            for k in range(n):
                v1 = i   * n * n + j * n + k
                v2 = (i+1) * n * n + j * n + k
                arestas.append([v1, v2])

    # Ligações paralelas ao eixo y
    for i in range(n):
        for j in range(n_div):
            for k in range(n):
                v1 = i * n * n + j   * n + k
                v2 = i * n * n + (j+1) * n + k
                arestas.append([v1, v2])

    # Ligações paralelas ao eixo z
    for i in range(n):
        for j in range(n):
            for k in range(n_div):
                v1 = i * n * n + j * n + k
                v2 = i * n * n + j * n + (k+1)
                arestas.append([v1, v2])

    return vertices, np.array(arestas)