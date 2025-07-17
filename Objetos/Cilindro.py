import numpy as np
from Utils.subdivisao_malha import subdivisao_malha
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

def constroi_vertices_cilindro(raio, altura, n_segments):
    angulos = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    base = np.array([[raio * np.cos(a), raio * np.sin(a), 0, 1] for a in angulos])
    topo = np.array([[raio * np.cos(a), raio * np.sin(a), altura, 1] for a in angulos])
    centro_base = np.array([[0, 0, 0, 1]])
    centro_topo = np.array([[0, 0, altura, 1]])
    return np.vstack([base, topo, centro_base, centro_topo])

def constroi_faces_cilindro(n_segments):
    faces = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # Laterais
        faces.append([i, j, n_segments + i])
        faces.append([n_segments + i, j, n_segments + j])
        # Base
        faces.append([2 * n_segments, j, i])
        # Topo
        faces.append([2 * n_segments + 1, n_segments + i, n_segments + j])
    return np.array(faces)

def constroi_arestas_cilindro(n_segments):
    edges = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        edges.append([i, j])  # base
        edges.append([i + n_segments, j + n_segments])  # topo
        edges.append([i, i + n_segments])  # vertical
    centro_base = 2 * n_segments
    centro_topo = 2 * n_segments + 1
    for i in range(n_segments):
        edges.append([centro_base, i])
        edges.append([centro_topo, i + n_segments])
    return np.array(edges)

def plotar_cilindro(raio, altura, n_segments=30, n_subdiv=1, cor="plum"):
    """
    Plota o cilindro usando faces analíticas (NÃO usa marching cubes).
    """
    vertices = constroi_vertices_cilindro(raio, altura, n_segments)
    faces = constroi_faces_cilindro(n_segments)
    arestas = constroi_arestas_cilindro(n_segments)

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
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cilindro")
    plt.show()

def constroi_cilindro(raio, altura, n_segments=30, origem=np.array([0, 0, 0]), n_subdiv=1):
    """
    Retorna listas de vértices (xyz) e arestas (índices) do cilindro.
    """
    vertices = constroi_vertices_cilindro(raio, altura, n_segments)
    faces = constroi_faces_cilindro(n_segments)
    arestas = constroi_arestas_cilindro(n_segments)

    if n_subdiv > 1:
        vertices, arestas, faces = subdivisao_malha(vertices, faces, n_subdiv)

    vertices[:, :3] += origem
    return vertices[:, :3].tolist(), arestas.tolist()

def campo_cilindro(X, Y, Z, raio, altura):
    R = np.sqrt(X**2 + Y**2)
    f = np.maximum(R - raio, np.maximum(-Z, Z - altura))
    return f

def plotar_cilindro_marching_cubes(raio, altura, grid_size=100, cor="lightcoral"):
    pad = raio * 0.15
    x = np.linspace(-raio - pad, raio + pad, grid_size)
    y = np.linspace(-raio - pad, raio + pad, grid_size)
    z = np.linspace(-pad, altura + pad, grid_size)   # Observe: agora cobre um pouco além do topo e base
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    campo = campo_cilindro(X, Y, Z, raio, altura)
    verts, faces, _, _ = marching_cubes(campo, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], facecolor=cor, edgecolor='k', linewidths=0.2, alpha=1)
    ax.add_collection3d(mesh)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cilindro (Marching Cubes)")
    plt.tight_layout()
    plt.show()
