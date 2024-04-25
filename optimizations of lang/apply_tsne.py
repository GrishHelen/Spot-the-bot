#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

new_dim = 2

dictionary_path = f"../SVD_embeddings/ru_word_cbow_space.npy"
coordinates = np.load(dictionary_path, allow_pickle=True, encoding='bytes')[()]


def apply_tsne(output_file, to_dim):
    X_embedded = TSNE(n_components=to_dim, learning_rate='auto', init='pca', perplexity=500, n_jobs=-1).fit_transform(
        np.array(coordinates))

    # сохраняем новые 2d координаты
    with open(f"{output_file}.npy", 'wb') as f:
        np.save(f, X_embedded)


def apply_pca(output_file):
    X_embedded = PCA(n_components=2).fit_transform(np.array(coordinates))

    with open(f"{output_file}.npy", 'wb') as f:
        np.save(f, X_embedded)


apply_tsne(f'rus_cbow_100to2_tsne', new_dim)
