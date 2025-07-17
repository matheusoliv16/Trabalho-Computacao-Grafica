import numpy as np

def constroi_arestas_e_vertices(vertices, faces):
    """
    Função genérica que recebe um array de vértices e um array de faces
    e retorna:
      - Lista de vértices [[x, y, z], ...]
      - Lista de arestas únicas [[i, j], ...]
    Suporta faces com 3 (triângulos) ou mais vértices (polígonos).
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    # Remove coordenada homogênea se presente
    if vertices.shape[1] == 4:
        vertices = vertices[:, :3]
    vertices_xyz = vertices.tolist()

    # Calcula arestas únicas a partir das faces
    arestas_set = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            aresta = tuple(sorted((a, b)))  # sempre na ordem menor->maior
            arestas_set.add(aresta)
    arestas_list = [list(edge) for edge in sorted(arestas_set)]

    return vertices_xyz, arestas_list
