import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

def curva_bezier(pontos_ctrl: np.ndarray, n_amostras: int):
    n = len(pontos_ctrl) - 1
    t = np.linspace(0, 1, n_amostras)
    curva = np.zeros((n_amostras, 3))
    for i in range(n + 1):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curva += np.outer(bernstein, pontos_ctrl[i])
    return curva

def constroi_vertices_cano_curvado(raio, pontos_ctrl, n_amostras=50, n_lados=20):
    caminho = curva_bezier(pontos_ctrl, n_amostras)
    tangentes = np.gradient(caminho, axis=0)
    tangentes = tangentes / np.linalg.norm(tangentes, axis=1)[:, np.newaxis]

    normals = []
    anterior = tangentes[0]
    for i in range(n_amostras):
        t = tangentes[i]
        if np.allclose(np.abs(np.dot(t, anterior)), 1.0):
            anterior = np.array([1, 0, 0]) if not np.allclose(t, [1,0,0]) else np.array([0,1,0])
        n = np.cross(anterior, t)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            n = np.cross([0,0,1], t)
            n = n / np.linalg.norm(n)
        else:
            n = n / n_norm
        b = np.cross(t, n)
        normals.append((n, b))
        anterior = t

    vertices = []
    for i in range(n_amostras):
        centro = caminho[i]
        n, b = normals[i]
        for a in np.linspace(0, 2 * np.pi, n_lados, endpoint=False):
            ponto = centro + raio * (np.cos(a) * n + np.sin(a) * b)
            vertices.append([*ponto, 1])
    return np.array(vertices)

def constroi_faces_sem_tampas(n_amostras, n_lados):
    faces = []
    for i in range(n_amostras - 1):
        for j in range(n_lados):
            prox = (j + 1) % n_lados
            idx0 = i * n_lados + j
            idx1 = i * n_lados + prox
            idx2 = (i + 1) * n_lados + j
            idx3 = (i + 1) * n_lados + prox

            faces.append([idx0, idx1, idx2])
            faces.append([idx2, idx1, idx3])
    return np.array(faces)

def constroi_arestas_cano_curvado(n_amostras, n_lados):
    edges = []
    for i in range(n_amostras):
        base_idx = i * n_lados
        for j in range(n_lados):
            edges.append([base_idx + j, base_idx + (j + 1) % n_lados])
            if i < n_amostras - 1:
                next_ring = base_idx + n_lados
                edges.append([base_idx + j, next_ring + j])
    return np.array(edges)

def gerar_cano_curvado(raio, pontos_ctrl, n_amostras=50, n_lados=20, origem=np.array([0, 0, 0]), n_subdiv=1):
    vertices = constroi_vertices_cano_curvado(raio, pontos_ctrl, n_amostras, n_lados)
    faces = constroi_faces_sem_tampas(n_amostras, n_lados)
    arestas = constroi_arestas_cano_curvado(n_amostras, n_lados)
    vertices[:, :3] += origem
    vertices_xyz = vertices[:, :3].tolist()
    arestas_list = arestas.tolist()
    return vertices_xyz, arestas_list

# CAMPO ESCALAR MODIFICADO: 
def campo_cano_curvado(X, Y, Z, caminho, raio):
    shape = X.shape
    pts_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    dists = np.linalg.norm(pts_grid[:, None, :] - caminho[None, :, :], axis=2)
    min_dist = np.min(dists, axis=1)
    idx_min = np.argmin(dists, axis=1)

    n_points = caminho.shape[0]
    # Deixe um "corte reto" mais evidente, excluindo 2 pontos em cada ponta (ajuste se quiser mais reto)
    margin = 2
    mask = (idx_min >= margin) & (idx_min <= n_points - 1 - margin)
    field = np.full_like(min_dist, 2*raio)  # fora, valor positivo

    field[mask] = min_dist[mask] - raio
    return field.reshape(shape)

def plotar_circulo_3d(ax, centro, normal, raio, cor='red', n_lados=50, **kwargs):
    theta = np.linspace(0, 2*np.pi, n_lados)
    if abs(normal[0]) < abs(normal[2]):
        v1 = np.cross(normal, [1,0,0])
    else:
        v1 = np.cross(normal, [0,0,1])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    pontos = [centro + raio*(np.cos(t)*v1 + np.sin(t)*v2) for t in theta]
    pontos = np.array(pontos)
    ax.plot(pontos[:,0], pontos[:,1], pontos[:,2], color=cor, **kwargs)

def plotar_cano_curvado(raio, pontos_ctrl, n_amostras=50, n_lados=20, cor="lightgreen"):
    vertices = constroi_vertices_cano_curvado(raio, pontos_ctrl, n_amostras, n_lados)
    faces = constroi_faces_sem_tampas(n_amostras, n_lados)
    arestas = constroi_arestas_cano_curvado(n_amostras, n_lados)

    fig = plt.figure(figsize=(10, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    verts = vertices[:, :3]
    poly3d = [[verts[i] for i in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=cor, edgecolors="black", linewidths=0.8, alpha=0.45)
    ax.add_collection3d(collection)

    for edge in arestas:
        ax.plot(*verts[edge].T, color="k", linewidth=0.5, alpha=0.6)

    caminho = curva_bezier(pontos_ctrl, n_amostras)
    base_centro = caminho[0]
    base_normal = np.gradient(caminho, axis=0)[0]
    base_normal = base_normal / np.linalg.norm(base_normal)
    plotar_circulo_3d(ax, base_centro, base_normal, raio, cor='red', n_lados=n_lados, lw=2, alpha=0.85)
    topo_centro = caminho[-1]
    topo_normal = np.gradient(caminho, axis=0)[-1]
    topo_normal = topo_normal / np.linalg.norm(topo_normal)
    plotar_circulo_3d(ax, topo_centro, topo_normal, raio, cor='blue', n_lados=n_lados, lw=2, alpha=0.85)

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cano Curvado")
    plt.tight_layout()
    plt.show()

def plotar_cano_curvado_marching_cubes(
    raio, pontos_ctrl, grid_size=120, n_amostras=200, cor="lime", mostrar_corte=False
):
    caminho = curva_bezier(pontos_ctrl, n_amostras)
    min_xyz = caminho.min(axis=0) - raio * 1.15
    max_xyz = caminho.max(axis=0) + raio * 1.15
    x = np.linspace(min_xyz[0], max_xyz[0], grid_size)
    y = np.linspace(min_xyz[1], max_xyz[1], grid_size)
    z = np.linspace(min_xyz[2], max_xyz[2], grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    campo = campo_cano_curvado(X, Y, Z, caminho, raio)
    verts, faces, _, _ = marching_cubes(
        campo, level=0.0, spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0])
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plotar a malha com bastante transparência
    mesh = Poly3DCollection(verts[faces], facecolor=cor, edgecolor='k', linewidths=0.10, alpha=0.18)
    ax.add_collection3d(mesh)

    # 2. Plotar círculos das extremidades para mostrar que está aberto
    base_centro = caminho[2]   # margin=2 usado no campo, igual aqui!
    base_normal = np.gradient(caminho, axis=0)[2]
    base_normal = base_normal / np.linalg.norm(base_normal)
    plotar_circulo_3d(ax, base_centro, base_normal, raio, cor='red', n_lados=60, lw=2, alpha=0.9)

    topo_centro = caminho[-3]  # margin=2 usado no campo, igual aqui!
    topo_normal = np.gradient(caminho, axis=0)[-3]
    topo_normal = topo_normal / np.linalg.norm(topo_normal)
    plotar_circulo_3d(ax, topo_centro, topo_normal, raio, cor='blue', n_lados=60, lw=2, alpha=0.9)

    # 3. Ajustar visualização
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    cx = max_xyz[0] - min_xyz[0]
    cy = max_xyz[1] - min_xyz[1]
    cz = max_xyz[2] - min_xyz[2]
    ax.set_box_aspect([cx, cy, cz])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cano Curvado (Marching Cubes)")
    plt.tight_layout()
    plt.show()
