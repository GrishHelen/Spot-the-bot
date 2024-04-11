import math
import numpy as np
import gudhi

# annulus_points = get_annulus_points(n=500)
# torus_points = get_torus_points(2, 3, num=20)
# eared_torus = torus_with_ears(n=20)

filename = '../optimizations of lang/rus_100to5pca_sparse.npy'
lang_coords = np.load(filename, allow_pickle=True)[()]
max_dist = 50
print(lang_coords.shape)
max_dim = lang_coords.shape[0]
band = 0
for eps in np.linspace(0.01, 5, 10):
    try:
        print(f'eps={eps}')
        rips_complex = gudhi.RipsComplex(points=lang_coords, max_edge_length=eps)
        print('here-1')
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)

        message = "Number of simplices=" + repr(simplex_tree.num_simplices())
        print(message)

        diag = simplex_tree.persistence()

        betti = simplex_tree.betti_numbers()
        print(f"betti_numbers()={betti}")
    except Exception as ex:
        print('err', ex)
    print('_' * 10)

    # gudhi.plot_persistence_diagram(diag, band=band)
    # plt.show()
