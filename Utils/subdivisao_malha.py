import numpy as np

def subdivisao_malha(vértices, faces, nivel=1):
    """
    Subdivide as faces de uma malha aplicando interpolação bilinear (quadriláteros) 
    e barycêntrica (triângulos).

    Parâmetros:
      - vértices: array numpy (N, 3) ou (N, 4)
      - faces: lista de faces (cada face é uma lista com 3 ou 4 índices)
      - nivel: quantidade de subdivisões

    Retorna:
      - novos_vértices: array numpy de vértices (em coordenadas homogêneas)
      - novas_arestas: array numpy de arestas (pares de índices)
      - novas_faces: array numpy de novas faces (listas de índices)
    """
    # Garante coordenadas homogêneas
    if vértices.shape[1] == 3:
        vértices = np.concatenate([vértices, np.ones((vértices.shape[0], 1))], axis=1)

    novos_vértices = []
    novas_faces = []
    arestas_set = set()
    mapa_índices = dict()
    prox_indice = 0

    for ind_face, face in enumerate(faces):
        if len(face) == 4:
            # Quadrilátero: interpolação bilinear
            pts = [vértices[i][:3] for i in face]
            for lin in range(nivel + 1):
                s = lin / nivel
                for col in range(nivel + 1):
                    t = col / nivel
                    ponto = (
                        (1 - s) * (1 - t) * pts[0]
                        + s * (1 - t) * pts[1]
                        + s * t * pts[2]
                        + (1 - s) * t * pts[3]
                    )
                    novos_vértices.append([*ponto, 1])
                    mapa_índices[(ind_face, lin, col)] = prox_indice
                    prox_indice += 1

            for lin in range(nivel):
                for col in range(nivel):
                    a = mapa_índices[(ind_face, lin, col)]
                    b = mapa_índices[(ind_face, lin + 1, col)]
                    c = mapa_índices[(ind_face, lin + 1, col + 1)]
                    d = mapa_índices[(ind_face, lin, col + 1)]
                    novas_faces.append([a, b, c])
                    novas_faces.append([a, c, d])
                    arestas_set.update({
                        tuple(sorted((a, b))),
                        tuple(sorted((b, c))),
                        tuple(sorted((c, d))),
                        tuple(sorted((d, a))),
                        tuple(sorted((a, c)))
                    })
        elif len(face) == 3:
            # Triângulo: interpolação barycêntrica
            pa, pb, pc = [vértices[i][:3] for i in face]
            for i in range(nivel + 1):
                for j in range(nivel + 1 - i):
                    u = i / nivel
                    v = j / nivel
                    w = 1 - u - v
                    p = w * pa + u * pb + v * pc
                    novos_vértices.append([*p, 1])
                    mapa_índices[(ind_face, i, j)] = prox_indice
                    prox_indice += 1
            for i in range(nivel):
                for j in range(nivel - i):
                    v1 = mapa_índices[(ind_face, i, j)]
                    v2 = mapa_índices[(ind_face, i + 1, j)]
                    v3 = mapa_índices[(ind_face, i, j + 1)]
                    novas_faces.append([v1, v2, v3])
                    arestas_set.update({
                        tuple(sorted((v1, v2))),
                        tuple(sorted((v2, v3))),
                        tuple(sorted((v3, v1)))
                    })
                    if j < nivel - i - 1:
                        v4 = mapa_índices[(ind_face, i + 1, j + 1)]
                        novas_faces.append([v2, v4, v3])
                        arestas_set.update({
                            tuple(sorted((v2, v4))),
                            tuple(sorted((v4, v3))),
                            tuple(sorted((v3, v2)))  # redundância para clareza
                        })
        else:
            raise ValueError("A face deve possuir 3 ou 4 vértices.")

    novos_vértices = np.array(novos_vértices)
    novas_arestas = np.array(list(arestas_set))
    novas_faces = np.array(novas_faces)
    return novos_vértices, novas_arestas, novas_faces
