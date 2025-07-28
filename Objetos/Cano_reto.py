import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# Constrói os vértices de um cano reto (sólido ou tubular)
def constroi_vertices_cano_reto(raio_externo, comprimento, n_segments=30, raio_interno=None):
    """
    Constrói os vértices de um cano reto em 3D.

    Args:
        raio_externo (float): Raio externo do cano.
        comprimento (float): Comprimento do cano no eixo Z.
        n_segments (int): Número de segmentos para aproximar o círculo da base/topo.
        raio_interno (float, optional): Raio interno do cano (para tubos). Se None ou <= 0, o cano é sólido.

    Returns:
        np.ndarray: Array (N, 3) dos vértices do cano.
    """
    if raio_interno is None or raio_interno <= 0:
        # Sem espessura: apenas cilindro sólido externo
        angulos = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
        base = np.array([[raio_externo * np.cos(a), raio_externo * np.sin(a), 0] for a in angulos])
        topo = np.array([[raio_externo * np.cos(a), raio_externo * np.sin(a), comprimento] for a in angulos])
        # Empilha vértices da base e topo externos
        return np.vstack([base, topo])
    else:
        # Com espessura: monta as bases e topos internos e externos
        angulos = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
        # Externo
        base_ext = np.array([[raio_externo * np.cos(a), raio_externo * np.sin(a), 0] for a in angulos])
        topo_ext = np.array([[raio_externo * np.cos(a), raio_externo * np.sin(a), comprimento] for a in angulos])
        # Interno
        base_int = np.array([[raio_interno * np.cos(a), raio_interno * np.sin(a), 0] for a in angulos])
        topo_int = np.array([[raio_interno * np.cos(a), raio_interno * np.sin(a), comprimento] for a in angulos])
        # Ordem: base_ext, topo_ext, base_int, topo_int
        return np.vstack([base_ext, topo_ext, base_int, topo_int])

# Constrói as faces triangulares do cano reto
def constroi_faces_cano_reto(n_segments, raio_interno=None):
    """
    Gera as faces triangulares para a malha de um cano reto.

    Args:
        n_segments (int): O número de segmentos usados para aproximar o círculo.
        raio_interno (float, optional): Raio interno do cano. Se None ou <= 0, o cano é sólido.

    Returns:
        np.ndarray: Array (M, 3) com índices de vértices para cada triângulo.
    """
    faces = []
    if raio_interno is None or raio_interno <= 0:
        # Sem espessura: apenas faces da casca lateral
        for i in range(n_segments):
            j = (i + 1) % n_segments
            # Dois triângulos por segmento lateral
            faces.append([i, j, i + n_segments])
            faces.append([i + n_segments, j, j + n_segments])
        return np.array(faces)
    else:
        # Com espessura: adiciona faces para lateral externa, interna, tampas superior/inferior
        # Vértices: 0..n_segments-1 (base_ext), n_segments..2n_segments-1 (topo_ext),
        # 2n_segments..3n_segments-1 (base_int), 3n_segments..4n_segments-1 (topo_int)
        offset_int_base = 2 * n_segments
        offset_int_topo = 3 * n_segments

        for i in range(n_segments):
            j = (i + 1) % n_segments
            # Lateral externa
            faces.append([i, j, i + n_segments])
            faces.append([i + n_segments, j, j + n_segments])
            # Lateral interna (invertido para normal correta)
            faces.append([offset_int_base + i, offset_int_topo + i, offset_int_base + j])
            faces.append([offset_int_base + j, offset_int_topo + i, offset_int_topo + j])
            # Tampa inferior (anel entre base externa e interna)
            faces.append([i, offset_int_base + i, j])
            faces.append([j, offset_int_base + i, offset_int_base + j])
            # Tampa superior (anel entre topo externo e interno)
            faces.append([i + n_segments, j + n_segments, offset_int_topo + i])
            faces.append([j + n_segments, offset_int_topo + j, offset_int_topo + i])
        return np.array(faces)


# Constrói as arestas do cano reto para visualização wireframe
def constroi_arestas_cano_reto(n_segments, raio_interno=None):
    """
    Gera as arestas para a visualização wireframe de um cano reto.

    Args:
        n_segments (int): O número de segmentos.
        raio_interno (float, optional): Raio interno do cano.

    Returns:
        np.ndarray: Array (K, 2) com índices de vértices para cada aresta.
    """
    edges = []
    if raio_interno is None or raio_interno <= 0:
        # Sem espessura: apenas arestas básicas
        for i in range(n_segments):
            # Lateral da base (circular)
            edges.append([i, (i + 1) % n_segments])
            # Lateral do topo (circular)
            edges.append([i + n_segments, (i + 1) % n_segments + n_segments])
            # Ligação base-topo (vertical)
            edges.append([i, i + n_segments])
        return np.array(edges)
    else:
        # Com espessura: arestas externas, internas e entre elas
        offset = 2 * n_segments  # início dos vértices internos na lista
        for i in range(n_segments):
            # Externo (base, topo, vertical)
            edges.append([i, (i + 1) % n_segments])
            edges.append([i + n_segments, (i + 1) % n_segments + n_segments])
            edges.append([i, i + n_segments])
            # Interno (base, topo, vertical)
            edges.append([offset + i, offset + (i + 1) % n_segments])
            edges.append([offset + i + n_segments, offset + (i + 1) % n_segments + n_segments])
            edges.append([offset + i, offset + i + n_segments])
            # Ligação externo-interno (base e topo)
            edges.append([i, offset + i])
            edges.append([i + n_segments, offset + i + n_segments])
        return np.array(edges)

# Plota o cano reto em 3D usando Matplotlib; salva se 'salvar_como' for especificado
def plotar_cano_reto(raio_externo, comprimento, n_segments=30, cor="yellow", raio_interno=None, salvar_como=None):
    """
    Plota um cano reto em 3D usando Matplotlib.

    Args:
        raio_externo (float): Raio externo do cano.
        comprimento (float): Comprimento do cano.
        n_segments (int): Número de segmentos para aproximar o círculo.
        cor (str): Cor das faces.
        raio_interno (float, optional): Raio interno do cano.
        salvar_como (str, optional): Nome do arquivo para salvar a imagem na pasta 'imagens/'.
    """
    vertices = constroi_vertices_cano_reto(raio_externo, comprimento, n_segments, raio_interno)
    faces = constroi_faces_cano_reto(n_segments, raio_interno)

    fig = plt.figure(figsize=(10, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    # Cria coleção de polígonos a partir dos vértices e faces
    poly3d = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=cor, edgecolors="black", linewidths=1, alpha=1)
    ax.add_collection3d(collection)

    # Centraliza visualização e ajusta limites
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, z.max() + max_range / 2) # Ajusta limite Z para começar em 0

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cano Reto")

    # Salva imagem na pasta, se solicitado
    if salvar_como:
        caminho_pasta = os.path.join(os.getcwd(), "imagens")
        os.makedirs(caminho_pasta, exist_ok=True)
        caminho_arquivo = os.path.join(caminho_pasta, salvar_como)
        plt.savefig(caminho_arquivo, bbox_inches="tight")
    plt.show()

# Gera os vértices e arestas de um cano reto já posicionado
def gerar_cano_reto(raio_externo, comprimento, n_segments=30, origem=(0, 0, 0), raio_interno=None):
    """
    Retorna arrays de vértices (xyz) e arestas (índices) do cano reto.

    Args:
        raio_externo (float): Raio externo do cano.
        comprimento (float): Comprimento do cano.
        n_segments (int): Número de segmentos.
        origem (tuple): A origem (x, y, z) para posicionar o cano.
        raio_interno (float, optional): Raio interno do cano.

    Returns:
        tuple: Uma tupla contendo:
            - vertices (np.ndarray): Array de vértices (N, 3).
            - arestas (np.ndarray): Array de arestas (M, 2), representando conexões entre vértices.
    """
    # Define o raio interno (considera None ou <= 0 como 0 para a geração)
    if raio_interno is not None and raio_interno > 0:
        raio_int = raio_interno
    else:
        raio_int = None # Passa None para constroi_vertices_cano_reto para gerar sólido

    # Constrói os vértices base (centrados na origem temporária)
    vertices = constroi_vertices_cano_reto(
        raio_externo=raio_externo,
        comprimento=comprimento,
        n_segments=n_segments,
        raio_interno=raio_int
    )
    # Desloca todos os vértices para a origem desejada
    vertices = vertices + np.array(origem)
    # Constrói as arestas (os índices não mudam com a translação)
    arestas = constroi_arestas_cano_reto(n_segments, raio_interno=raio_int)
    return vertices, arestas

# Plota o cano reto em 3D de forma interativa usando Plotly
def plotar_cano_reto_interativo(raio_externo, comprimento, n_segments=30, raio_interno=None, cor="yellow"):
    """
    Plota um cano reto em 3D de forma interativa usando Plotly.

    Args:
        raio_externo (float): Raio externo do cano.
        comprimento (float): Comprimento do cano.
        n_segments (int): Número de segmentos na seção circular.
        raio_interno (float, optional): Raio interno do cano. None para sólido.
        cor (str): Cor das faces.
    """
    # Define o raio interno (considera None ou <= 0 como 0.0 para a geração)
    if raio_interno is None or raio_interno <= 0:
        raio_interno = 0.0

    # Gera pontos nos círculos da base e topo, internos e externos
    theta = np.linspace(0, 2 * np.pi, n_segments + 1) # +1 para fechar o círculo
    x_ext = raio_externo * np.cos(theta)
    y_ext = raio_externo * np.sin(theta)
    x_int = raio_interno * np.cos(theta)
    y_int = raio_interno * np.sin(theta)
    z0 = np.zeros(n_segments + 1)
    z1 = np.full(n_segments + 1, comprimento)

    # Empilha os vértices na ordem esperada pelo constroi_faces_cano_reto para Plotly
    # [base_ext, topo_ext, base_int, topo_int]
    V = np.vstack([
        np.column_stack([x_ext, y_ext, z0]),   # base externa
        np.column_stack([x_ext, y_ext, z1]),   # topo externa
        np.column_stack([x_int, y_int, z0]),   # base interna
        np.column_stack([x_int, y_int, z1])    # topo interna
    ])

    # Gera as faces triangulares para Plotly (mesmo raciocínio do Matplotlib)
    # Adaptação para Plotly, que prefere índices sequenciais
    F = []
    # Offset para os vértices internos
    offset_int = (n_segments + 1)

    for i in range(n_segments):
        j = (i + 1) # % (n_segments + 1) - o loop vai até n_segments-1, j vai até n_segments
        # Lateral externa (usa i, i+1, i+offset_int, i+offset_int+1)
        F.append([i, j, i + offset_int])
        F.append([i + offset_int, j, j + offset_int])

        # Lateral interna (inverso para normal correta)
        if raio_interno > 0:
            # Vértices internos começam após os externos (base e topo)
            offset_internal_verts = 2 * offset_int
            F.append([offset_internal_verts + i, offset_internal_verts + j, offset_internal_verts + i + offset_int]) # Triângulo 1
            F.append([offset_internal_verts + i + offset_int, offset_internal_verts + j, offset_internal_verts + j + offset_int]) # Triângulo 2
            # Inverte a ordem dos vértices para corrigir a normal
            F[-2] = [F[-2][0], F[-2][2], F[-2][1]]
            F[-1] = [F[-1][0], F[-1][2], F[-1][1]]


        # Ligações entre externo/interno (paredes do topo e da base)
        if raio_interno > 0:
            # tampa base (anel entre base externa e base interna)
            F.append([i, i + offset_internal_verts, j])
            F.append([j, i + offset_internal_verts, j + offset_internal_verts])
            # tampa topo (anel entre topo externo e topo interno)
            F.append([i + offset_int, j + offset_int, i + offset_internal_verts + offset_int])
            F.append([j + offset_int, j + offset_internal_verts + offset_int, i + offset_internal_verts + offset_int])


    F = np.array(F)

    # Plota interativamente com Plotly
    fig = go.Figure(data=[
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color=cor, opacity=0.7
        )
    ])
    fig.update_layout(
        title="Cano reto",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()