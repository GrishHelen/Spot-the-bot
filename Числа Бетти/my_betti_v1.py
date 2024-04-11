#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.spatial import distance
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

from generate_topology_objects import *


class ComputingBetti:
    def __init__(self, vector_points, eps: float = 0):
        self.max_dim = len(vector_points[0])
        self.points = vector_points
        self.eps = eps
        self.labels = list(range(len(self.points)))
        self._construct_all_subsets()

    def _construct_all_subsets(self):
        n = self.points.shape[0]
        self.all_subsets = [[] for _ in range(self.max_dim + 1)]  # self.all_subsets[dim][subset_ind] = (subset, diam)
        self.all_subsets[0] = [frozenset([i]) for i in range(n)]
        for cur_dim in range(1, self.max_dim + 1):  # O(max_dim) - up to 5
            for subset_ind in range(len(self.all_subsets[cur_dim - 1])):  # O(cnt of p-dim subsets) = O(n^dim)
                for j in range(max(self.all_subsets[cur_dim - 1][subset_ind]) + 1, n):  # O(n)
                    diam = 0
                    for p_ind in self.all_subsets[cur_dim - 1][subset_ind]:  # O(cur_dim), cur_dim<5 хран
                        dist = distance.minkowski(self.points[p_ind], self.points[j])
                        if dist >= eps:
                            break
                        diam = max(diam, dist)
                    else:
                        self.all_subsets[cur_dim].append(
                            (self.all_subsets[cur_dim - 1][subset_ind].union(frozenset([j])), diam))

    # def rank_Z2(self, matrix: np.matrix) -> int:
    #     n, m = matrix.shape
    #     rk = 0
    #     for i in range(m):
    #         if rk == min(n, m):
    #             break
    #         k = matrix[rk:, i].argmax() + rk
    #         if matrix[k, i] == 0:
    #             continue
    #         if k != rk:
    #             for col in range(i, m):
    #                 matrix[rk, col], matrix[k, col] = matrix[k, col], matrix[rk, col]
    #         for row in range(rk + 1, n):
    #             if matrix[row, i]:
    #                 for col in range(i, m):
    #                     matrix[row, col] = matrix[row, col] ^ matrix[rk, col]
    #         rk += 1
    #     return rk
    # это вариант без njit. Но с njit вроде бы быстрее
    import numpy as np
    import time

    def rank_z2(self, rows):
        n = len(rows)
        m = len(rows[0])
        rows = np.asarray(rows)
        rank = 0
        for pivot_ind in range(n):
            if rows[pivot_ind].any():
                low = 0
                for i in range(m):
                    if rows[pivot_ind][i]:
                        low = i
                        break
                rank += 1
                for index in range(pivot_ind + 1, n):
                    if rows[index][low]:
                        rows[index] = rows[index] ^ rows[pivot_ind]
        return rank

    def count_betti_numbers(self) -> int:
        n = 1
        betti_numbers = [0] * (self.max_dim + 1)
        cnt_nm1 = len(self.all_subsets[n - 1])
        cnt_n = len(self.all_subsets[n])
        if cnt_nm1 == 0:
            Z_p = cnt_n
        elif cnt_n == 0:
            Z_p = 0
        else:
            boundary_matrix = np.matrix(np.ndarray(shape=(cnt_nm1, cnt_n)), dtype=bool)
            for i in range(cnt_nm1):
                for j in range(cnt_n):
                    if self.all_subsets[n][j][0].issuperset(self.all_subsets[n - 1][i][0]):
                        boundary_matrix[i, j] = 1
            Z_p = cnt_n - self.rank_z2(boundary_matrix)

        for n in range(1, self.max_dim):
            # needs subsets of dims n-1,n,n+1
            cnt_np1 = len(self.all_subsets[n + 1])
            cnt_n = len(self.all_subsets[n])

            if cnt_n == 0:
                B_p = cnt_np1
            elif cnt_np1 == 0:
                B_p = 0
            else:
                boundary_matrix = np.matrix(np.ndarray(shape=(cnt_n, cnt_np1)), dtype=bool)
                for i in range(cnt_n):
                    for j in range(cnt_np1):
                        if self.all_subsets[n + 1][j][0].issuperset(self.all_subsets[n][i][0]):
                            boundary_matrix[i, j] = 1
                B_p = self.rank_z2(boundary_matrix)
            betti_numbers[n] = Z_p - B_p

            Z_p = cnt_np1 - B_p


def my_3d_draw(coordinates):
    fig = plt.figure(figsize=(6, 6), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    lim = max([max(i) for i in coordinates])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    if type(coordinates) != np.matrix:
        coordinates = np.matrix(coordinates)

    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], alpha=0.8, cmap=cm.Wistia)
    fig.savefig("torus.png", dpi=130, bbox_inches='tight', transparent=True)
    plt.show()


filename = '../optimizations of lang/rus_100to5pca_sparse_18059points.npy'
filename = '../SVD_embeddings/rus_100to5_pca.npy'
lang_coords = np.load(filename, allow_pickle=True)[()]

if __name__ == "__main__":
    # torus_points = get_torus_points(2, 5, 10)
    # my_3d_draw(torus_points)
    # annulus_points = annulus(200)
    eps = 5
    print(lang_coords.shape)

    # vector_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
    # eps = 2.24
    computing_betti = ComputingBetti(lang_coords, eps)

    betti_nums = computing_betti.count_betti_numbers()

    # plt.scatter([i[0] for i in torus_points], [i[1] for i in torus_points])
    # plt.show()

    # computing_betti.simplicial_complex.draw()
