import numpy as np
from scipy.special import comb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# Gera pontos de uma curva de Bézier 3D a partir dos pontos de controle
def curva_bezier(pontos_ctrl: np.ndarray, n_amostras: int):
    """
    Calcula pontos em uma curva de Bézier 3D.

    Args:
        pontos_ctrl (np.ndarray): Array (N, 3) dos pontos de controle.
        n_amostras (int): Número de pontos a serem gerados na curva.

    Returns:
        np.ndarray: Array (n_amostras, 3) dos pontos da curva.
    """
    n = len(pontos_ctrl) - 1  # Grau da curva
    t = np.linspace(0, 1, n_amostras)  # Parâmetro da curva (de 0 a 1)
    curva = np.zeros((n_amostras, 3)) # Inicializa array para armazenar os pontos da curva
    for i in range(n + 1):
        # Calcula o polinômio de Bernstein para cada ponto de controle
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        # Adiciona a contribuição do ponto de controle atual multiplicada pelo seu polinômio de Bernstein
        curva += np.outer(bernstein, pontos_ctrl[i])
    return curva

# Constrói os vértices do cano curvado, podendo ser sólido ou tubular
def constroi_vertices_cano_curvado(raio, pontos_ctrl, n_amostras=50, n_lados=20, espessura=None):
    """
    Constrói os vértices de um cano curvado 3D (sólido ou tubular).

    Se espessura=None: retorna apenas a malha do cilindro curvado sólido.
    Se espessura > 0: retorna a malha do tubo, ou seja, pontos externos seguidos dos internos.

    Args:
        raio (float): Raio externo do cano.
        pontos_ctrl (np.ndarray): Pontos de controle da curva de Bézier central.
        n_amostras (int): Número de segmentos ao longo da curva.
        n_lados (int): Número de segmentos na seção circular.
        espessura (float, optional): Espessura da parede do tubo. None para sólido.

    Returns:
        np.ndarray: Array (N, 3) dos vértices do cano.
    """
    caminho = curva_bezier(pontos_ctrl, n_amostras)  # Trajetória do centro do tubo
    # Calcula os vetores tangentes a cada ponto do caminho e os normaliza
    tangentes = np.gradient(caminho, axis=0)
    tangentes = tangentes / np.linalg.norm(tangentes, axis=1)[:, np.newaxis]

    # Define um vetor normal inicial ortogonal ao primeiro tangente
    t0 = tangentes[0]
    # Se o primeiro tangente for quase vertical (paralelo ao eixo Z), usa um vetor 'anterior' no eixo X
    if np.abs(t0[0]) < 1e-5 and np.abs(t0[1]) < 1e-5:
        anterior = np.array([1, 0, 0])
    else:
        # Caso contrário, usa um vetor 'anterior' no eixo Z
        anterior = np.array([0, 0, 1])

    normals = []
    # Calcula um referencial ortonormal (normal 'n', binormal 'b') para cada ponto da curva
    # Este é um método simples para gerar o sistema de coordenadas local ao longo da curva (Frenet frame adaptado)
    for i in range(n_amostras):
        t = tangentes[i]
        # Calcula o vetor normal 'n' (ortogonal ao 'anterior' e 't')
        n = np.cross(anterior, t)
        n_norm = np.linalg.norm(n)
        # Se o cross product for zero ou muito pequeno, o 'anterior' era paralelo ao 't'.
        # Usa um vetor alternativo ([1,0,0]) para calcular um novo 'n'.
        if n_norm < 1e-8:
            n = np.cross([1,0,0], t)
            n = n / np.linalg.norm(n)
        else:
            # Normaliza o vetor normal
            n = n / n_norm
        # Calcula o vetor binormal 'b' (ortogonal a 't' e 'n')
        b = np.cross(t, n)
        normals.append((n, b)) # Armazena o par (normal, binormal)
        anterior = t # O tangente atual se torna o 'anterior' para o próximo passo

    vertices = []
    # Suporte a espessura (tubo ou sólido)
    if espessura is None or espessura == 0:
        # Apenas parede externa (cilindro sólido)
        for i in range(n_amostras):
            centro = caminho[i] # Ponto central na trajetória
            n, b = normals[i]   # Base ortonormal local
            # Cria os pontos da circunferência externa para este segmento
            for a in np.linspace(0, 2 * np.pi, n_lados, endpoint=False):
                ponto = centro + raio * (np.cos(a) * n + np.sin(a) * b)
                vertices.append(ponto)
        return np.array(vertices)  # shape (n_amostras * n_lados, 3)
    else:
        # Com espessura: cria pontos para a superfície externa e interna
        raio_int = raio - espessura
        # Superfície externa
        for i in range(n_amostras):
            centro = caminho[i]
            n, b = normals[i]
            for a in np.linspace(0, 2 * np.pi, n_lados, endpoint=False):
                ponto = centro + raio * (np.cos(a) * n + np.sin(a) * b)
                vertices.append(ponto)
        # Superfície interna (menor raio)
        for i in range(n_amostras):
            centro = caminho[i]
            n, b = normals[i]
            for a in np.linspace(0, 2 * np.pi, n_lados, endpoint=False):
                ponto = centro + (raio_int) * (np.cos(a) * n + np.sin(a) * b)
                vertices.append(ponto)
        return np.array(vertices)  # shape (2 * n_amostras * n_lados, 3)

# Constrói as faces triangulares do cano curvado (sem tampas nas extremidades)
def constroi_faces_sem_tampas(n_amostras, n_lados, espessura=None):
    """
    Gera as faces triangulares para a malha lateral de um cano curvado (sem tampas nas pontas).

    Args:
        n_amostras (int): Número de segmentos ao longo da curva.
        n_lados (int): Número de segmentos na seção circular.
        espessura (float, optional): Espessura da parede do tubo. None para sólido.

    Returns:
        np.ndarray: Array (M, 3) com índices de vértices para cada triângulo.
    """
    faces = []
    # Caso não tenha espessura, malha simples de tubo sólido
    if espessura is None or espessura == 0:
        # Percorre os segmentos longitudinais (anéis)
        for i in range(1, n_amostras-1):  # Só entre 1 e n-2 (cria cano aberto nas pontas para simplificar)
            # Percorre os segmentos circulares em cada anel
            for j in range(n_lados):
                prox = (j + 1) % n_lados # Índice do próximo vértice no anel
                # Índices dos 4 vértices que formam um quadrilátero lateral
                idx0 = i * n_lados + j
                idx1 = i * n_lados + prox
                idx2 = (i + 1) * n_lados + j
                idx3 = (i + 1) * n_lados + prox
                # Dois triângulos para cada quadrilátero
                faces.append([idx0, idx1, idx2])
                faces.append([idx2, idx1, idx3])
        return np.array(faces)
    else:
        # Com espessura: adiciona faces para lateral externa, interna e paredes
        offset = n_amostras * n_lados # Deslocamento para os índices dos vértices internos

        # Superfície externa
        for i in range(1, n_amostras-1):
            for j in range(n_lados):
                prox = (j + 1) % n_lados
                idx0 = i * n_lados + j
                idx1 = i * n_lados + prox
                idx2 = (i + 1) * n_lados + j
                idx3 = (i + 1) * n_lados + prox
                faces.append([idx0, idx1, idx2])
                faces.append([idx2, idx1, idx3])

        # Superfície interna (inverter sentido da normal para plotagem correta)
        for i in range(1, n_amostras-1):
            for j in range(n_lados):
                prox = (j + 1) % n_lados
                idx0 = offset + i * n_lados + j
                idx1 = offset + i * n_lados + prox
                idx2 = offset + (i + 1) * n_lados + j
                idx3 = offset + (i + 1) * n_lados + prox
                # Atenção: ordem invertida para normal apontar para dentro
                faces.append([idx2, idx1, idx0])
                faces.append([idx3, idx1, idx2])

        # Faces laterais (ligando externo e interno - paredes do anel)
        # Percorre os segmentos longitudinais (anéis) EXCETO os primeiros e últimos anéis
        for i in range(1, n_amostras-1):
            # Percorre os segmentos circulares em cada anel
            for j in range(n_lados):
                jp = (j + 1) % n_lados # Índice do próximo vértice no anel
                # Índices dos 4 vértices que formam um quadrilátero na parede do anel
                vidx_ext1 = i * n_lados + j
                vidx_ext2 = i * n_lados + jp
                vidx_int1 = offset + i * n_lados + j
                vidx_int2 = offset + i * n_lados + jp
                # Dois triângulos para cada quadrilátero na parede
                faces.append([vidx_ext1, vidx_int1, vidx_ext2])
                faces.append([vidx_ext2, vidx_int1, vidx_int2])

        return np.array(faces)

# Constrói as arestas da malha do cano curvado, para visualização wireframe
def constroi_arestas_cano_curvado(n_amostras, n_lados, espessura=None):
    """
    Gera as arestas para a visualização wireframe de um cano curvado.

    Args:
        n_amostras (int): Número de segmentos ao longo da curva.
        n_lados (int): Número de segmentos na seção circular.
        espessura (float, optional): Espessura da parede do tubo. None para sólido.

    Returns:
        np.ndarray: Array (K, 2) com índices de vértices para cada aresta.
    """
    edges = []
    if espessura is None or espessura == 0:
        # Apenas arestas básicas (circulares e longitudinais) da superfície externa
        for i in range(n_amostras):
            base_idx = i * n_lados # Índice inicial do anel atual
            for j in range(n_lados):
                # Ligações circulares dentro do anel
                edges.append([base_idx + j, base_idx + (j + 1) % n_lados])
                # Ligações longitudinais (entre anéis consecutivos)
                if i < n_amostras - 1: # Evita ir além do último anel
                    next_ring = base_idx + n_lados # Índice inicial do próximo anel
                    edges.append([base_idx + j, next_ring + j])
        return np.array(edges)
    else:
        # Com espessura: adiciona arestas para superfícies externa, interna e ligações entre elas
        offset = n_amostras * n_lados # Deslocamento para os índices dos vértices internos
        for i in range(n_amostras):
            base_idx = i * n_lados
            base_idx_int = offset + i * n_lados # Índice inicial do anel interno atual
            for j in range(n_lados):
                # Arestas da superfície externa
                edges.append([base_idx + j, base_idx + (j + 1) % n_lados])
                if i < n_amostras - 1:
                    next_ring = base_idx + n_lados
                    edges.append([base_idx + j, next_ring + j])
                # Arestas da superfície interna
                edges.append([base_idx_int + j, base_idx_int + (j + 1) % n_lados])
                if i < n_amostras - 1:
                    next_ring_int = base_idx_int + n_lados
                    edges.append([base_idx_int + j, next_ring_int + j])
                # Arestas que ligam a superfície externa à interna (nas bases e topos dos anéis)
                edges.append([base_idx + j, base_idx_int + j])
        return np.array(edges)

# Retorna os vértices e arestas de um cano curvado já posicionado na origem desejada
def gerar_cano_curvado(raio, pontos_ctrl, n_amostras=50, n_lados=20, origem=(0, 0, 0), espessura=None):
    """
    Gera vértices e arestas para um cano curvado e o posiciona em uma origem.

    Args:
        raio (float): Raio externo do cano.
        pontos_ctrl (np.ndarray): Pontos de controle da curva de Bézier central.
        n_amostras (int): Número de segmentos ao longo da curva.
        n_lados (int): Número de segmentos na seção circular.
        origem (tuple): Ponto (x, y, z) para onde deslocar o cano.
        espessura (float, optional): Espessura da parede do tubo. None para sólido.

    Returns:
        tuple: Uma tupla contendo:
            - vertices (np.ndarray): Array (N, 3) de vértices.
            - arestas (np.ndarray): Array (M, 2) de índices de arestas.
    """
    # Constrói os vértices base (centrados na origem temporária)
    vertices = constroi_vertices_cano_curvado(raio, pontos_ctrl, n_amostras, n_lados, espessura=espessura)
    # Desloca todos os vértices para a origem desejada
    vertices = vertices + np.array(origem)
    # Constrói as arestas (os índices não mudam com a translação)
    arestas = constroi_arestas_cano_curvado(n_amostras, n_lados, espessura=espessura)
    return vertices, arestas

# Plota o cano curvado em 3D (matplotlib), salva a imagem se solicitado
def plotar_cano_curvado(
    raio, pontos_ctrl, n_amostras=50, n_lados=20, cor="lightgreen", espessura=None, salvar_como=None
):
    """
    Plota um cano curvado em 3D usando Matplotlib.

    Args:
        raio (float): Raio externo do cano.
        pontos_ctrl (np.ndarray): Pontos de controle da curva de Bézier central.
        n_amostras (int): Número de segmentos ao longo da curva.
        n_lados (int): Número de segmentos na seção circular.
        cor (str): Cor das faces do cano.
        espessura (float, optional): Espessura da parede do tubo. None para sólido.
        salvar_como (str, optional): Nome do arquivo para salvar a imagem na pasta 'imagens/'.
                                     Se None, a imagem não é salva.
    """
    # Constrói os vértices e faces
    vertices = constroi_vertices_cano_curvado(raio, pontos_ctrl, n_amostras, n_lados, espessura=espessura)
    faces = constroi_faces_sem_tampas(n_amostras, n_lados, espessura=espessura)

    # Cria a figura e o subplot 3D
    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # Monta coleção de polígonos para plotar as faces
    poly3d = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=cor, edgecolors="k", linewidths=0.4, alpha=1.0)
    # Adiciona a coleção ao subplot
    ax.add_collection3d(collection)

    # Ajusta limites dos eixos para centralizar o cano e garantir proporção
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Define rótulos dos eixos e título
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cano Curvado")
    # Ajusta o layout
    plt.tight_layout()

    # Salva a imagem na pasta especificada, se solicitado
    if salvar_como:
        caminho_pasta = os.path.join(os.getcwd(), "imagens")
        os.makedirs(caminho_pasta, exist_ok=True)
        caminho_arquivo = os.path.join(caminho_pasta, salvar_como)
        plt.savefig(caminho_arquivo, bbox_inches="tight")
    # Mostra a figura
    plt.show()

# Plota o cano curvado em 3D de forma interativa (Plotly)
def plotar_cano_curvado_interativo(raio, pontos_ctrl, n_amostras=50, n_lados=20, cor="lightgreen", espessura=None, origem=(0,0,0), titulo="Cano Curvado (Plotly Interativo)"):
    """
    Plota um cano curvado em 3D de forma interativa usando Plotly.

    Args:
        raio (float): Raio externo do cano.
        pontos_ctrl (np.ndarray): Pontos de controle da curva de Bézier central.
        n_amostras (int): Número de segmentos ao longo da curva.
        n_lados (int): Número de segmentos na seção circular.
        cor (str): Cor das faces do cano.
        espessura (float, optional): Espessura da parede do tubo. None para sólido.
        origem (tuple): Ponto (x, y, z) para onde deslocar o cano.
        titulo (str): Título do gráfico Plotly.
    """
    # Gera os vértices e faces para o mesh
    # Adiciona 1 a n_segments para fechar os anéis no Plotly
    vertices = constroi_vertices_cano_curvado(raio, pontos_ctrl, n_amostras + 1, n_lados + 1, espessura=espessura)
    vertices = vertices + np.array(origem)
    # Geração de faces para Plotly Mesh3d (adaptado do Matplotlib)
    F = []
    if espessura is None or espessura == 0:
        for i in range(n_amostras): # Itera sobre os anéis (menos o último)
            for j in range(n_lados): # Itera sobre os segmentos circulares
                prox = (j + 1) % (n_lados + 1) # Índice do próximo vértice no anel (inclui o último ponto)
                idx0 = i * (n_lados + 1) + j
                idx1 = i * (n_lados + 1) + prox
                idx2 = (i + 1) * (n_lados + 1) + j
                idx3 = (i + 1) * (n_lados + 1) + prox
                # Dois triângulos para cada quadrilátero na superfície
                F.append([idx0, idx1, idx3])
                F.append([idx0, idx3, idx2])
    else:
        # Com espessura: faces externas, internas e laterais
        offset = (n_amostras + 1) * (n_lados + 1) # Deslocamento para os índices internos

        # Superfície externa
        for i in range(n_amostras):
             for j in range(n_lados):
                prox = (j + 1) % (n_lados + 1)
                idx0 = i * (n_lados + 1) + j
                idx1 = i * (n_lados + 1) + prox
                idx2 = (i + 1) * (n_lados + 1) + j
                idx3 = (i + 1) * (n_lados + 1) + prox
                F.append([idx0, idx1, idx3])
                F.append([idx0, idx3, idx2])

        # Superfície interna (ordem invertida para normal)
        for i in range(n_amostras):
            for j in range(n_lados):
                prox = (j + 1) % (n_lados + 1)
                idx0 = offset + i * (n_lados + 1) + j
                idx1 = offset + i * (n_lados + 1) + prox
                idx2 = offset + (i + 1) * (n_lados + 1) + j
                idx3 = offset + (i + 1) * (n_lados + 1) + prox
                F.append([idx0, idx3, idx1]) # Ordem invertida
                F.append([idx2, idx3, idx0]) # Ordem invertida

        # Tampas (início e fim do tubo)
        # Tampa inicial (base)
        base_ext_indices = np.arange(n_lados + 1)
        base_int_indices = np.arange(offset, offset + n_lados + 1)
        for j in range(n_lados):
            jp = (j + 1) % (n_lados + 1)
            F.append([base_ext_indices[j], base_ext_indices[jp], base_int_indices[j]])
            F.append([base_ext_indices[jp], base_int_indices[jp], base_int_indices[j]])

        # Tampa final (topo)
        topo_ext_indices = np.arange((n_amostras) * (n_lados + 1), (n_amostras + 1) * (n_lados + 1))
        topo_int_indices = np.arange(offset + (n_amostras) * (n_lados + 1), offset + (n_amostras + 1) * (n_lados + 1))
        for j in range(n_lados):
            jp = (j + 1) % (n_lados + 1)
            F.append([topo_ext_indices[j], topo_int_indices[j], topo_ext_indices[jp]]) # Ordem invertida para normal
            F.append([topo_ext_indices[jp], topo_int_indices[j], topo_int_indices[jp]]) # Ordem invertida


    F = np.array(F)
    V = np.array(vertices)


    # Plota interativamente com Plotly
    fig = go.Figure(data=[
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color=cor, opacity=0.7
        )
    ])
    fig.update_layout(
        title=titulo,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()