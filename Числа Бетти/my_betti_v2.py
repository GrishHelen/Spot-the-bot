#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

max_eps = 0.001
max_dim = 5
boundary_ranks = [0] * (max_dim + 2)
dist_matrix = np.load('../optimizations of lang/all_dists_rus_pca5.npy', allow_pickle=True)[()]
cnt_points = dist_matrix.shape[0]
dists_ones = [set() for _ in range(cnt_points)]
for i in range(cnt_points):
    dists_ones[i] = set(np.where(dist_matrix[i] <= max_eps)[0])

boundary_ranks[1] = cnt_points

simplices = [[{i} for i in range(cnt_points)], []]
simpices_ind_old = 0
simpices_ind_new = 1

for cur_dim in range(2, max_dim):  # ~5
    cnt_old_simplices = len(simplices[simpices_ind_old])
    for old_simplex_ind in range(cnt_old_simplices):  # O(cnt_old_simplices) (<= O(cnt_points^dim)) (depends on eps)
        print(old_simplex_ind, boundary_ranks[cur_dim])

        # may be change dists_ones to matrix of 0/1 -> np, bool, bit opers

        probable_new_points = (dists_ones[min(simplices[simpices_ind_old][old_simplex_ind])]).difference(
            simplices[simpices_ind_old][old_simplex_ind])
        for p_ind in simplices[simpices_ind_old][old_simplex_ind]:  # O(cur_dim-1)
            probable_new_points.intersection_update(dists_ones[p_ind])  # O(cnt_points)
            if len(probable_new_points) == 0:
                break
        # probable_new_points - set of points, which may be added to this simplex to get the new one

        for new_point in probable_new_points:  # < O(cnt_points) (depends on eps)
            new_simplex = simplices[simpices_ind_old][old_simplex_ind].union({new_point})  # O(dim+1)
            # check new_simplex whether it is linearly independent
            # with ready new simplices (simplices[simpices_ind_new])
            for ind in range(boundary_ranks[cur_dim]):  # O(rk * 2*cnt_points)
                if new_simplex & simplices[simpices_ind_new][ind]:
                    new_simplex ^= simplices[simpices_ind_new][ind]
            if len(new_simplex) == cur_dim:
                new_simplex = simplices[simpices_ind_old][old_simplex_ind].union({new_point})  # O(dim+1)
                simplices[simpices_ind_new].append(new_simplex.copy())
                boundary_ranks[cur_dim] += 1

    print(*boundary_ranks)
    simpices_ind_old, simpices_ind_new = simpices_ind_new, simpices_ind_old
    simplices[simpices_ind_new] = []
