import numpy as np
from Utils.subdivisao_malha import subdivisao_malha
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

def constroi_vertices_cano_reto(raio, comprimento, n_segments=30):
    angulos = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    base = np.array([[raio * np.cos(a), raio * np.sin(a), 0, 1] for a in angulos])
    topo = np.array([[raio * np.cos(a), raio * np.sin(a), comprimento, 1] for a in angulos])
    return np.vstack([base, topo])

def constroi_faces_cano_reto(n_segments):
    faces = []
    base = lambda i: i
    topo = lambda i: i + n_segments
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # Cada quad lateral é formado por dois triângulos
        faces.append([base(i), base(j), topo(i)])
        faces.append([topo(i), base(j), topo(j)])
    return np.array(faces)

def constroi_arestas_cano_reto(n_segments):
    edges = []
    # Laterais da base e do topo
    for offset in range(2):
        for i in range(n_segments):
            j = (i + 1) % n_segments
            edges.append([i + offset * n_segments, j + offset * n_segments])
    # Liga base e topo
    for i in range(n_segments):
        edges.append([i, i + n_segments])
    return np.array(edges)

def plotar_cano_reto(raio, comprimento, n_segments=30, n_subdiv=1, cor="skyblue"):
    """
    Plota o cano reto (cilindro aberto) utilizando apenas a malha analítica construída pelas funções de geração de faces e vértices.
    NÃO utiliza Marching Cubes.
    """
    vertices = constroi_vertices_cano_reto(raio, comprimento, n_segments)
    faces = constroi_faces_cano_reto(n_segments)
    arestas = constroi_arestas_cano_reto(n_segments)

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
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, z.max() + max_range / 2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cano Reto")
    plt.show()

def gerar_cano_reto(raio, comprimento, n_segments=30, origem=(0, 0, 0)):
    """
    Gera os vértices e arestas de um cano reto com base nas dimensões fornecidas e aplica uma translação para a origem dada.
    """
    vertices = constroi_vertices_cano_reto(raio, comprimento, n_segments)
    x0, y0, z0 = origem
    translacao = np.array([x0, y0, z0, 0])
    vertices += translacao
    arestas = constroi_arestas_cano_reto(n_segments)
    return vertices, arestas


def campo_cano_reto(X, Y, Z, raio, comprimento):
    R = np.sqrt(X**2 + Y**2)
    # O campo é zero na superfície lateral (R = raio) e não considera tampa
    # Fora do comprimento do cano (z < 0 ou z > comprimento) campo positivo (não tem tampa)
    f = np.maximum(R - raio, np.maximum(-Z, Z - comprimento))
    return f

def plotar_cano_marching_cubes(raio, comprimento, grid_size=64):
    x = np.linspace(-raio*1.2, raio*1.2, grid_size)
    y = np.linspace(-raio*1.2, raio*1.2, grid_size)
    z = np.linspace(0, comprimento, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    campo = campo_cano_reto(X, Y, Z, raio, comprimento)
    verts, faces, _, _ = marching_cubes(campo, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], facecolor='skyblue', edgecolor='k', linewidths=0.1, alpha=1.0)
    ax.add_collection3d(mesh)
    ax.set_xlim(verts[:,0].min(), verts[:,0].max())
    ax.set_ylim(verts[:,1].min(), verts[:,1].max())
    ax.set_zlim(verts[:,2].min(), verts[:,2].max())
    ax.set_box_aspect([1,1,comprimento/(raio*2)])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cano reto (Marching Cubes)')
    plt.tight_layout()
    plt.show()