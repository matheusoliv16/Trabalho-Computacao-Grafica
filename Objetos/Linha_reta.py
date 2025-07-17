import matplotlib.pyplot as plt
import numpy as np

def constroi_vertices_linha(comprimento=4.0, direcao=np.array([1, 0, 0]), origem=np.array([0, 0, 0])):
    direcao = direcao / np.linalg.norm(direcao)
    p0 = np.append(origem, 1)
    p1 = np.append(origem + direcao * comprimento, 1)
    vertices = np.array([p0, p1])
    arestas = np.array([[0, 1]])
    return vertices, arestas

def plotar_linha(comprimento=4.0, direcao=np.array([1, 0, 0]), origem=np.array([0, 0, 0]), cor="black"):
    vertices, arestas = constroi_vertices_linha(comprimento, direcao, origem)
    v = vertices[:, :3]

    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([v[0, 0], v[1, 0]], [v[0, 1], v[1, 1]], [v[0, 2], v[1, 2]], color=cor, linewidth=3)

    ax.set_xlim(v[:, 0].min() - 1, v[:, 0].max() + 1)
    ax.set_ylim(v[:, 1].min() - 1, v[:, 1].max() + 1)
    ax.set_zlim(v[:, 2].min() - 1, v[:, 2].max() + 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Linha Reta")
    plt.show()

def constroi_linha_reta(comprimento=4.0, direcao=np.array([1, 0, 0]), origem=np.array([0, 0, 0])):
    direcao = direcao / np.linalg.norm(direcao)
    p0 = origem
    p1 = origem + direcao * comprimento
    vertices = [p0.tolist(), p1.tolist()]
    arestas = [[0, 1]]
    return vertices, arestas
