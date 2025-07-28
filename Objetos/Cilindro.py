import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import os

# Gera todos os vértices de um cilindro 3D (base, topo e centros)
def constroi_vertices_cilindro(raio, altura, n_segments):
    """
    Gera os vértices de um cilindro com base no raio, altura e número de segmentos.

    Args:
        raio (float): O raio do cilindro.
        altura (float): A altura do cilindro.
        n_segments (int): O número de segmentos usados para aproximar o círculo da base/topo.

    Returns:
        np.ndarray: Array (N, 3) com todos os vértices do cilindro.
    """
    angulos = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    # Pontos da circunferência da base (z=0)
    base = np.array([[raio * np.cos(a), raio * np.sin(a), 0] for a in angulos])
    # Pontos da circunferência do topo (z=altura)
    topo = np.array([[raio * np.cos(a), raio * np.sin(a), altura] for a in angulos])
    # Centro da base e do topo
    centro_base = np.array([[0, 0, 0]])
    centro_topo = np.array([[0, 0, altura]])
    # Retorna todos empilhados: [base, topo, centro_base, centro_topo]
    return np.vstack([base, topo, centro_base, centro_topo])

# Gera todas as faces triangulares do cilindro (laterais, base e topo)
def constroi_faces_cilindro(n_segments):
    """
    Gera as faces triangulares para a malha de um cilindro.

    Args:
        n_segments (int): O número de segmentos usados para aproximar o círculo.

    Returns:
        np.ndarray: Array (M, 3) com índices de vértices para cada triângulo.
    """
    faces = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # Laterais (2 triângulos por segmento)
        faces.append([i, j, n_segments + i])
        faces.append([n_segments + i, j, n_segments + j])
        # Base (triângulo: centro, p1, p2) - índices ajustados para incluir os centros
        faces.append([2 * n_segments, j, i])
        # Topo (triângulo: centro, p1, p2) - índices ajustados para incluir os centros
        faces.append([2 * n_segments + 1, n_segments + i, n_segments + j])
    return np.array(faces)

# Gera todas as arestas (linhas) do cilindro (útil para wireframe)
def constroi_arestas_cilindro(n_segments):
    """
    Gera as arestas para a visualização wireframe de um cilindro.

    Args:
        n_segments (int): O número de segmentos usados para aproximar o círculo.

    Returns:
        np.ndarray: Array (K, 2) com índices de vértices para cada aresta.
    """
    edges = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # Areia da base
        edges.append([i, j])
        # Areia do topo
        edges.append([i + n_segments, j + n_segments])
        # Aresta vertical (liga base e topo)
        edges.append([i, i + n_segments])
    # Ligações com o centro das tampas
    centro_base = 2 * n_segments
    centro_topo = 2 * n_segments + 1
    for i in range(n_segments):
        edges.append([centro_base, i])
        edges.append([centro_topo, i + n_segments])
    return np.array(edges)

# Plota o cilindro em 3D, salva na pasta 'imagens' se solicitado
def plotar_cilindro(raio, altura, n_segments=30, cor="plum", salvar_como=None):
    """
    Plota o cilindro usando faces analíticas.
    Se salvar_como for passado, salva a imagem em /imagens.

    Args:
        raio (float): O raio do cilindro.
        altura (float): A altura do cilindro.
        n_segments (int): O número de segmentos.
        cor (str): A cor das faces.
        salvar_como (str, optional): Nome do arquivo para salvar a imagem.
    """
    vertices = constroi_vertices_cilindro(raio, altura, n_segments)
    faces = constroi_faces_cilindro(n_segments)

    fig = plt.figure(figsize=(10, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    poly3d = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=cor, edgecolors="black", linewidths=1, alpha=1)
    ax.add_collection3d(collection)

    # Centraliza e ajusta escala dos eixos para visualização
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cilindro")

    # Salva a imagem na pasta 'imagens', se solicitado
    if salvar_como:
        caminho_pasta = os.path.join(os.getcwd(), "imagens")
        os.makedirs(caminho_pasta, exist_ok=True)
        caminho_arquivo = os.path.join(caminho_pasta, salvar_como)
        plt.savefig(caminho_arquivo, bbox_inches="tight")
    plt.show()

# Retorna arrays de vértices (xyz) e arestas do cilindro, já posicionado na origem desejada
def gerar_cilindro(raio, altura, n_segments=30, origem=(0, 0, 0)):
    """
    Retorna arrays de vértices (xyz) e arestas (índices) do cilindro.

    Args:
        raio (float): O raio do cilindro.
        altura (float): A altura do cilindro.
        n_segments (int): O número de segmentos.
        origem (tuple): A origem (x, y, z) para posicionar o cilindro.

    Returns:
        tuple: Uma tupla contendo:
            - vertices (np.ndarray): Array de vértices (N, 3).
            - arestas (np.ndarray): Array de arestas (M, 2), representando conexões entre vértices.
    """
    vertices = constroi_vertices_cilindro(raio, altura, n_segments)
    vertices = vertices + np.array(origem)  # Aplica deslocamento de origem
    arestas = constroi_arestas_cilindro(n_segments)
    return vertices, arestas