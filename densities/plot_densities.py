import numpy as np
import matplotlib.pyplot as plt

# a = np.load("densities_rus_r7_dim2_bins50.npy", allow_pickle=True)[()]
x = np.load("x_coord_densities.npy", allow_pickle=True)[()]
y = np.load("y_coord_densities.npy", allow_pickle=True)[()]
densities = np.load("densities.npy", allow_pickle=True)[()]

print()
# x = np.linspace(-100, 100, 51)
# y = np.linspace(-100, 100, 51)
# cs = plt.contourf(x, y, a)
cs = plt.contour(x, y, densities)
cs.clabel(colors='black', inline=False)
plt.show()
