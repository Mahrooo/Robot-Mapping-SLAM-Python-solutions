import scipy
import numpy as np
import unscented_transformation
from plot import draw_probe_ellipse
import matplotlib.pyplot as plt

# This is the main script for computing a transformed distribution according
# to the unscented transform. This script calls all the required
# functions in the correct order.
# If you are unsure about the input and return values of functions you
# should read their documentation which tells you the expected dimensions.

# Turn off pagination and open a new figure for plotting
plt.clf()
plt.grid('on')

# Initial distribution
sigma = 0.1*np.eye(2)
mu = np.array([1., 2.])
n = len(mu)

# Compute lambda
alpha = 0.9
beta = 2.
kappa = 1.
lamb = pow(alpha, 2)*(n + kappa) - n

# Compute the sigma points corresponding to mu and sigma
sigma_points, w_m, w_c = unscented_transformation.compute_sigma_points(mu, sigma, lamb, alpha, beta)

# Plot original distribution with sampled sigma points
plt.plot(mu[0], mu[1], 'ro', markersize=12., linewidth=3., fillstyle='none', label='original dis')
draw_probe_ellipse(mu, sigma, 0.9, 'r')
plt.plot(sigma_points[0], sigma_points[1], 'kx',
         markersize=10., linewidth=3.)

# Transform sigma points
sigma_points_trans = unscented_transformation.transform(sigma_points)

# Recover mu and sigma of the transformed distribution
mu_trans, sigma_trans = unscented_transformation.recover_gaussian(sigma_points_trans, w_m, w_c)

# Plot transformed sigma points with corresponding mu and sigma
plt.plot(mu_trans[0], mu_trans[1], 'bo', markersize=12., linewidth=3., fillstyle='none', label='transformed dis')
plt.legend()
draw_probe_ellipse(mu_trans, sigma_trans, 0.9, color='b')
plt.plot(sigma_points_trans[0], sigma_points_trans[1], 'kx',
         markersize=10., linewidth=3.)

# Figure axes setup
plt.title('Unscented Transformation', fontsize=20)
x_min = min([mu[0], mu_trans[0]])
x_max = max([mu[0], mu_trans[0]])
y_min = min([mu[1], mu_trans[1]])
y_max = max([mu[1], mu_trans[1]])
plt.axis([x_min-3, x_max+3, y_min-3, y_max+3])
plt.axis('equal')
plt.show()

# Print and save plot
plt.savefig('unscented.png')