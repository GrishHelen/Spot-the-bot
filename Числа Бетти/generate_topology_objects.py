import copy
import numpy as np
import math
from random import random, choice
import matplotlib.pyplot as plt
from matplotlib import cm


def my_3d_draw(coordinates,n):
    fig = plt.figure(figsize=(6, 6), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    # lim = max([max(np.abs(i)) for i in coordinates])
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    # ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    if type(coordinates) != np.matrix:
        coordinates = np.asarray(coordinates)

    axes = list(map(int, list('102')))
    colors = ['#1f77b4', '#17becf', '#2ca02c']
    for i in range(3):
        ax.scatter(coordinates[i*n:(i+1)*n, axes[0]], coordinates[i*n:(i+1)*n, axes[1]], coordinates[i*n:(i+1)*n, axes[2]], s=2, alpha=0.3,
                   color=colors[i])
    fig.savefig("torus.png", dpi=130, bbox_inches='tight', transparent=True)
    plt.show()


def get_circle_points(r=10, num=10):
    u = np.linspace(0, 2 * np.pi * (num - 1) / num, num)
    x = np.sin(u) * r
    y = np.cos(u) * r
    return np.vstack((x, y))


def get_sphere_points(samples=100, r=1):
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x * r, y * r, z * r))

    return points


def get_torus_points(r1=1, r2=2, num=6):
    u = np.linspace(0, 2 * np.pi * (num - 1) / num, num)
    v = np.linspace(0, 2 * np.pi * (num - 1) / num, num)
    u, v = np.meshgrid(u, v)
    u = u.reshape((num ** 2,))
    v = v.reshape((num ** 2,))
    X = (r2 + r1 * np.cos(u)) * np.cos(v)
    Y = (r2 + r1 * np.cos(u)) * np.sin(v)
    Z = r1 * np.sin(u)

    torus = np.vstack((X, Y, Z)).transpose()
    return torus


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


def get_annulus_points(n, r1=1, r2=2, offset=(0, 0)):
    result = []
    for i in range(n):
        x = 0
        y = 0
        while not (r1 <= math.hypot(x, y) <= r2):
            x = 2 * r2 * random() - r2
            y = 2 * r2 * random() - r2
        result.append((offset[0] + x, offset[1] + y))
    return result


def double_annulus(n, r1=1, R1=2, r2=1.5, R2=4, offset=(5, 0)):
    X = []
    Y = []
    for i in range(n):
        x = 0
        y = 0
        while not (r1 <= math.hypot(x, y) <= R1):
            x = 2 * R1 * random() - R1
            y = 2 * R1 * random() - R1
        X.append(x)
        Y.append(y)
    for i in range(int((R2 - r2) * n / (R1 - r1))):
        x = 0
        y = 0
        while not (r2 <= math.hypot(x, y) <= R2):
            x = 2 * R2 * random() - R2
            y = 2 * R2 * random() - R2
        X.append(x + offset[0])
        Y.append(y + offset[1])
    return X, Y


def torus_with_ears(n=10, r1=1, R1=3, r2=1.2, R2=2, r3=0.8, R3=3.5, offset2=(7, 0, 0), offset3=(0, 0, 2), to_draw=False,
                    alpha=0.8):
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    Theta, Phi = np.meshgrid(theta, phi)
    x1 = (R1 + r1 * np.cos(Phi)) * np.cos(Theta)
    y1 = (R1 + r1 * np.cos(Phi)) * np.sin(Theta)
    z1 = r1 * np.sin(Phi)

    x2 = (R2 + r2 * np.cos(Phi)) * np.cos(Theta) + offset2[0]
    y2 = (R2 + r2 * np.cos(Phi)) * np.sin(Theta) + offset2[1]
    z2 = r2 * np.sin(Phi) + offset2[2]

    x3 = (R3 + r3 * np.cos(Phi)) * np.cos(Theta) + offset3[0]
    y3 = (R3 + r3 * np.cos(Phi)) * np.sin(Theta) + offset3[1]
    z3 = r3 * np.sin(Phi) + offset3[2]

    if to_draw:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, y1, z1, s=1, color='r', alpha=alpha)
        ax.scatter(x2, y2, z2, s=1, color='g', alpha=alpha)
        ax.scatter(x3, y3, z3, s=1, color='b', alpha=alpha)
        plt.show()

    x1 = np.reshape(x1, (-1, 1))
    y1 = np.reshape(y1, (-1, 1))
    z1 = np.reshape(z1, (-1, 1))

    x2 = np.reshape(x2, (-1, 1))
    y2 = np.reshape(y2, (-1, 1))
    z2 = np.reshape(z2, (-1, 1))

    x3 = np.reshape(x3, (-1, 1))
    y3 = np.reshape(y3, (-1, 1))
    z3 = np.reshape(z3, (-1, 1))

    tor1 = np.hstack([x1, y1, z1])
    tor2 = np.hstack([x2, y2, z2])
    tor3 = np.hstack([x3, y3, z3])

    return np.vstack([tor1, tor2, tor3])


def torus_with_ears_evenly(n=1000, r1=1, R1=3, r2=1.2, R2=2, r3=0.8, R3=3, offset2=(-4.5, 0, 1.5), offset3=(-5.5, 0, -1.5),
                           alpha2=2 / 3, alpha3=(1 / 3)):
    torus_1 = get_torus_points_evenly(r1, R1, n)
    torus_2 = get_torus_points_evenly(r2, R2, n)
    alpha2 = 2 * math.pi * alpha2
    for i in range(n):
        x, y, z = torus_2[i]
        x_new = x * math.sin(alpha2) + z * math.cos(alpha2)
        z_new = z * math.sin(alpha2) - x * math.cos(alpha2)
        torus_2[i] = (x_new + offset2[0], y + offset2[1], z_new + offset2[2])
    torus_3 = get_torus_points_evenly(r3, R3, n)
    alpha3 = 2 * math.pi * alpha3
    for i in range(n):
        x, y, z = torus_3[i]
        x_new = x * math.sin(alpha3) + z * math.cos(alpha3)
        z_new = z * math.sin(alpha3) - x * math.cos(alpha3)
        torus_3[i] = (x_new + offset3[0], y + offset3[1], z_new + offset3[2])
    return torus_1 + torus_2 + torus_3

n=5000
torus = torus_with_ears_evenly(n=n)
my_3d_draw(torus,n)
