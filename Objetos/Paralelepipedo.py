import numpy as np
from Utils.subdivisao_malha import subdivisao_malha
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

def constroi_vertices_paralelepipedo(comprimento, largura, altura):
    x, y, z = comprimento / 2, largura / 2, altura / 2
    return np.array([
        [-x, -y, -z, 1],
        [ x, -y, -z, 1],
        [ x,  y, -z, 1],
        [-x,  y, -z, 1],
        [-x, -y,  z, 1],
        [ x, -y,  z, 1],
        [ x,  y,  z, 1],
        [-x,  y,  z, 1],
    ])

def constroi_faces_paralelepipedo():
    return np.array([
        [0, 1, 2], [0, 2, 3],      # Base inferior
        [4, 7, 6], [4, 6, 5],      # Base superior
        [0, 4, 5], [0, 5, 1],      # Face frontal
        [1, 5, 6], [1, 6, 2],      # Lateral direita
        [2, 6, 7], [2, 7, 3],      # Face traseira
        [3, 7, 4], [3, 4, 0],      # Lateral esquerda
    ])

def constroi_arestas_paralelepipedo():
    return np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base inferior
        [4, 5], [5, 6], [6, 7], [7, 4],  # Base superior
        [0, 4], [1, 5], [2, 6], [3, 7]   # Verticais
    ])

def plotar_paralelepipedo(comprimento, largura, altura, n_subdiv=1, cor="lightblue"):
    """
    Plota o paralelepípedo usando faces analíticas (NÃO usa marching cubes).
    """
    vertices = constroi_vertices_paralelepipedo(comprimento, largura, altura)
    faces = constroi_faces_paralelepipedo()
    arestas = constroi_arestas_paralelepipedo()

    if n_subdiv > 1:
        vertices, arestas, faces = subdivisao_malha(vertices, faces, n_subdiv)

    fig = plt.figure(figsize=(10, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    verts = vertices[:, :3]
    poly3d = [[verts[i] for i in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=cor, edgecolors="black", linewidths=1, alpha=1)
    ax.add_collection3d(collection)

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max() + x.min())/2, (y.max() + y.min())/2, (z.max() + z.min())/2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Paralelepípedo")
    plt.show()

def constroi_paralelepipedo(comprimento, largura, altura, origem=np.array([0, 0, 0]), n_subdiv=1):
    """
    Retorna listas de vértices (xyz) e arestas (índices) do paralelepípedo.
    """
    vertices = constroi_vertices_paralelepipedo(comprimento, largura, altura)
    faces = constroi_faces_paralelepipedo()
    arestas = constroi_arestas_paralelepipedo()

    if n_subdiv > 1:
        vertices, arestas, faces = subdivisao_malha(vertices, faces, n_subdiv)

    vertices[:, :3] += origem
    return vertices[:, :3].tolist(), arestas.tolist()

def campo_paralelepipedo(X, Y, Z, comprimento, largura, altura):
    f = np.maximum(np.abs(X) - comprimento/2, np.maximum(np.abs(Y) - largura/2, np.abs(Z) - altura/2))
    return f

def plotar_paralelepipedo_marching_cubes(comprimento, largura, altura, grid_size=80, cor="deepskyblue"):
    # Domínio do grid, um pouco maior que o cubo para garantir fechamento
    pad_x = comprimento * 0.12
    pad_y = largura * 0.12
    pad_z = altura * 0.12
    x = np.linspace(-comprimento/2 - pad_x, comprimento/2 + pad_x, grid_size)
    y = np.linspace(-largura/2 - pad_y, largura/2 + pad_y, grid_size)
    z = np.linspace(-altura/2 - pad_z, altura/2 + pad_z, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    campo = campo_paralelepipedo(X, Y, Z, comprimento, largura, altura)
    verts, faces, _, _ = marching_cubes(campo, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], facecolor=cor, edgecolor='k', linewidths=0.3, alpha=1)
    ax.add_collection3d(mesh)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_box_aspect([comprimento, largura, altura])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Paralelepípedo (Marching Cubes)")
    plt.tight_layout()
    plt.show()
