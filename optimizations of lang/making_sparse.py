import numpy as np
from scipy.spatial import distance, cKDTree
from sys import getsizeof
import pyflann
from pyflann.index import FLANN
import matplotlib.pyplot as plt
import random

filename = '../SVD_embeddings/rus_100to5_pca.npy'
filename = '../SVD_embeddings/rus_100to2_tsne.npy'
lang_words = np.load('../SVD_embeddings/russian_nofraglit_SVD_100_dict.npy', allow_pickle=True)[()]
lang_words = list(lang_words.keys())
print(len(lang_words))
output_file = 'rus_100to2tsne_sparse'
lang_coords = np.load(filename, allow_pickle=True)[()]
n, dim = lang_coords.shape
distance_upper_bound = 10

# limits = [[0, 0] for _ in range(dim)]
#
# for point in lang_coords:
#     for i in range(dim):
#         limits[i][0] = min(limits[i][0], point[i])
#         limits[i][1] = max(limits[i][1], point[i])
# for i in limits:
#     print(int(i[1] - i[0] + 1), '|', *i)
# d = 1
# for i in limits:
#     d *= (i[1] - i[0])
# print(d)

# [[-1, 51], [-7, 11], [-4, 6], [-2, 6], [-1, 5]]
# 52, 18, 10, 8, 6
# for i in range(2, lang_coords.shape[0]):
#     if lang_coords.shape[0] % i == 0:
#         print(i, lang_coords.shape[0] // i)


# # flann
# flann_kdtree = FLANN()
# pyflann.set_distance_type('euclidean')
# flann_kdtree.build_index(lang_coords, algorithm='kdtree')
# pyflann.set_distance_type('euclidean')
# flann_res = flann_kdtree.nn_index(qpts=lang_coords, num_neighbors=100)
# # neighbours are sorted by dist
# print('FLANN ready')

# ckdtree
ckdtree = cKDTree(lang_coords)
ckdtree_res = ckdtree.query(lang_coords, k=700, eps=0, p=2, workers=-1, distance_upper_bound=distance_upper_bound)

new_points = []
max_use = 300
cnt_used = [0] * n
words = []
for i in range(n):
    if cnt_used[i] > max_use:
        continue
    a = list(ckdtree_res[1][i])
    if a[-1] == n:
        t = a.index(n)
        # print(i, t)
        cur_indxs = list(ckdtree_res[1][i])[:t]
    else:
        cur_indxs = list(ckdtree_res[1][i])
    cur_points = []
    for ind in cur_indxs:
        cur_points.append(lang_coords[ind])
        cnt_used[ind] += 1

    cur_points = np.asarray(cur_points)  # neighbours
    new_point = np.mean(cur_points, axis=0)
    help_kdtree = cKDTree(cur_points)
    new_point_ind = help_kdtree.query(new_point, k=1, eps=0, p=2, workers=-1)[1]
    new_points.append(cur_points[new_point_ind])
    words.append(lang_words[cur_indxs[new_point_ind]])
    # print(cur_points.shape, new_point.shape)
new_points = np.asarray(new_points)
print(new_points.shape)

# with open(f"{output_file}_{new_points.shape[0]}points.npy", 'wb') as f:
#     np.save(f, new_points)
# with open(f'{output_file}_{new_points.shape[0]}points_words.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(words))

print(f"{len(new_points) / len(lang_words) * 100}%")

fig, ax = plt.subplots()
ax.scatter(new_points[:, 1], new_points[:, 0], s=0.7, alpha=0.7, edgecolors="k")
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
fig.show()
