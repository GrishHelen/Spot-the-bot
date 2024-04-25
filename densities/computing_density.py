import numpy as np
from math import dist
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Process, Lock, freeze_support
from multiprocessing.shared_memory import SharedMemory
from sys import getsizeof
import seaborn
import pandas as pd


def process(lock, r, span_le, span_ri, points_le, points_ri, density_shape, points_span_inds_shape, points_span_shape,
            points_shape):
    densities_shared = SharedMemory('densities')
    densities = np.ndarray(shape=density_shape, buffer=densities_shared.buf)
    points_span_inds_shared = SharedMemory('points_span_inds')
    points_span_inds = np.ndarray(shape=points_span_inds_shape, buffer=points_span_inds_shared.buf, dtype=np.int64)
    points_span_shared = SharedMemory('points_span')
    points_span = np.ndarray(shape=points_span_shape, buffer=points_span_shared.buf)
    points_shared = SharedMemory('points')
    points = np.ndarray(shape=points_shape, buffer=points_shared.buf)

    for span_ind in range(span_le, span_ri):
        span_point = points_span[span_ind]
        cnt = sum([1 for point_ind in range(points_le, points_ri) if dist(span_point, points[point_ind]) <= r])
        lock.acquire()
        a = list(points_span_inds[span_ind])
        densities[*a] = cnt
        lock.release()


lang = 'rus'
filename = f"../SVD_embeddings/{lang}_100to2_tsne.npy"

if __name__ == '__main__':
    freeze_support()
    lock = Lock()

    r = 6
    bins = 52

    help_ndarray = np.load(filename, allow_pickle=True)[()]
    # print(1)
    # help_ndarray = pd.DataFrame(help_ndarray, columns=['x', 'y'])
    # print(2)
    # seaborn.kdeplot(help_ndarray, x='x', y='y', fill=True, thresh=0, levels=11)
    # print(3)
    # plt.show()
    points_shared = SharedMemory(create=True, size=getsizeof(help_ndarray[0]) * len(help_ndarray), name='points')
    points = np.ndarray(shape=help_ndarray.shape, buffer=points_shared.buf)
    points[:] = np.copy(help_ndarray)

    dim = len(points[0])
    bin_edge = np.histogram_bin_edges(points, bins=bins)

    densities = np.ndarray(shape=[bins + 1] * dim)
    densities_shared = SharedMemory(create=True, size=getsizeof(densities), name='densities')
    densities = np.ndarray(shape=[bins + 1] * dim,
                           buffer=densities_shared.buf)  # amount of points in the spheres (vicinities)

    help_ndarray = np.asarray(list(product(bin_edge, repeat=dim)))
    points_span_shared = SharedMemory(create=True, size=getsizeof(help_ndarray), name='points_span')
    points_span = np.ndarray(shape=help_ndarray.shape, buffer=points_span_shared.buf)
    points_span[:] = np.copy(help_ndarray)

    help_ndarray = np.asarray(list(product([i for i in range(bins + 1)], repeat=dim)), dtype=np.int64)
    points_span_inds_shared = SharedMemory(create=True, size=getsizeof(help_ndarray[0]) * len(help_ndarray),
                                           name='points_span_inds')
    points_span_inds = np.ndarray(shape=help_ndarray.shape, buffer=points_span_inds_shared.buf, dtype=np.int64)
    points_span_inds[:] = np.copy(help_ndarray)

    separators_inds_span = [i * ((bins + 1) ** (dim - 1)) for i in range(bins + 2)]
    separators_inds_points = [[0, 0] for _ in range(bins + 2)]
    for i in range(bins + 1):
        if i != 0:
            separators_inds_points[i] = separators_inds_points[i - 1][::]

        while separators_inds_points[i][0] < len(points) and bin_edge[i] - r > points[separators_inds_points[i][0]][0]:
            separators_inds_points[i][0] = separators_inds_points[i][0] + 1

        separators_inds_points[i][1] = max(separators_inds_points[i][0], separators_inds_points[i][1])
        while (separators_inds_points[i][1] < len(points) and
               bin_edge[i + 1] + r > points[separators_inds_points[i][0]][0]):
            separators_inds_points[i][1] = separators_inds_points[i][1] + 1

    processes = []
    for i in range(bins + 1):
        value_process = Process(target=process,
                                args=(lock, r,
                                      separators_inds_span[i], separators_inds_span[i + 1],
                                      separators_inds_points[i][0], separators_inds_points[i][1],
                                      densities.shape, points_span_inds.shape, points_span.shape, points.shape))
        processes.append(value_process)
        value_process.start()
    for value_process in processes:
        value_process.join()
    print(f"densities_{lang}_r{r}_dim{dim}_bins{bins}.npy")
    # with open(f"densities_{lang}_r{r}_dim{dim}_bins{bins}.npy", 'wb') as f:
    #     np.save(f, densities)

    cs = plt.contour(bin_edge, bin_edge, densities)
    # cs.clabel(colors='black', inline=False)
    cs2 = plt.contourf(bin_edge, bin_edge, densities, color='k')
    plt.xlim(-200, 200)
    plt.ylim(-150, 150)
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
             for pc in cs2.collections]
    plt.legend(proxy, [str(i) for i in list(cs.levels)])
    plt.show()
    # plt.legend()

    # fig, ax = plt.subplots()
    # ax.scatter(points_span[:, 0], points_span[:, 1], s=2, alpha=0.7, edgecolors="k")
    # # ax.set_xlim(-0.1, 0.1)
    # # ax.set_ylim(-0.8, 0)
    # fig.show()

    # fig, ax = plt.subplots()
    # ax.scatter(points[:, 1], points[:, 0], s=0.2, alpha=0.7, edgecolors="k")
    # ax.set_xlim(-200, 200)
    # ax.set_ylim(-150, 150)
    # fig.show()
