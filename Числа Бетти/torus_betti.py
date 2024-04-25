import math
import numpy as np
import gudhi
from random import choice
import matplotlib.pyplot as plt


def get_torus_points_evenly(r=1, R=2, n=100, offset=(0, 0, 0)):
    points = []
    for _ in range(n):
        U = choice(np.linspace(0, 1, 100))
        V = choice(np.linspace(0, 1, 100))
        phi = 2 * math.pi * U
        psi = 2 * math.pi * V
        x = ((R + r * math.cos(phi)) * math.cos(psi)) + offset[0]
        y = ((R + r * math.cos(phi)) * math.sin(psi)) + offset[1]
        z = (r * math.sin(phi)) + offset[2]
        points.append((x, y, z))
    return points


def torus_with_ears_evenly(n=1000, r1=1, R1=3, r2=1.2, R2=2, r3=0.8, R3=3, offset2=(-5, 0, 2), offset3=(-9, 0, -5),
                           alpha2=2 / 3, alpha3=1 / 4):
    torus_1 = get_torus_points_evenly(r1, R1, n)
    torus_2 = get_torus_points_evenly(r2, R2, n)
    alpha2 = 2 * math.pi * alpha2
    for i in range(n):
        x, y, z = torus_2[i]
        x_new = x * math.sin(alpha2) + z * math.cos(alpha2)
        z_new = z * math.sin(alpha2) - x * math.cos(alpha2)
        torus_2[i] = (x_new + offset2[0], y + offset2[1], z_new + offset2[2])
    torus_3 = get_torus_points_evenly(r3, R3, n)
    alpha3 = - 2 * math.pi * alpha3
    for i in range(n):
        x, y, z = torus_3[i]
        x_new = x * math.sin(alpha3) + z * math.cos(alpha3)
        z_new = z * math.sin(alpha3) - x * math.cos(alpha3)
        torus_3[i] = (x_new + offset3[0], y + offset3[1], z_new + offset3[2])
    return torus_1 + torus_2 + torus_3


points = torus_with_ears_evenly(n=800)
for eps in np.linspace(0.5, 4, 36):
    try:
        rips_complex = gudhi.RipsComplex(points=points, max_edge_length=eps)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)

        diag = simplex_tree.persistence()
        betti = simplex_tree.betti_numbers()
        file = open('res_eared_torus.txt', 'a')
        file.write(f'eps={eps}, betti={betti}\n')
        file.close()
    except Exception as ex:
        pass

