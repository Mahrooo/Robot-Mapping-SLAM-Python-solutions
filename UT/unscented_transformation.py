import numpy as np


def compute_sigma_points(mu, sigma, lam, a, b):
    """This function samples 2n+1 sigma points from the distribution given by
    mu and sigma according to the unscented transform, where n is the dimensionality of mu.
    Each column of sigma_points should represent one sigma point i.e. sigma_points has a
    dimensionality of nx2n+1. The corresponding weights w_m and w_c of the points are
    computed using lambda, alpha, and beta:
    w_m = [w_m_0, ..., w_m_2n], w_c = [w_c_0, ..., w_c_2n] (i.e. each of size 1x2n+1)
    They are later used to recover the mean and covariance respectively.
    """
    # dimentionality:
    n = len(mu)

    # TODO: compute all sigma  points
    sigma_points = np.tile(mu, (2*n+1, 1)).T
    sigma_points[:, 1:n+1] += np.sqrt((n+lam)*sigma)
    sigma_points[:, n+1:] -= np.sqrt((n+lam)*sigma)

    # TODO: compute weight vectors w_m and w_c
    w_m = np.ones(2*n+1,)/(2*(n+lam))
    w_c = w_m.copy()
    w_m[0] = 2*lam*w_m[0]
    w_c[0] = w_m[0]+(1-a**2+b)

    return sigma_points, w_m, w_c


def recover_gaussian(sigma_points, w_m, w_c):
    """This function computes the recovered Gaussian distribution (mu and sigma)
    given the sigma points (size: nx2n+1) and their weights w_m and w_c:
    w_m = [w_m_0, ..., w_m_2n], w_c = [w_c_0, ..., w_c_2n].
    The weight vectors are each 1x2n+1 in size,
    where n is the dimensionality of the distribution."""

    # Try to vectorize your operations as much as possible

    # TODO: compute mu
    mu = np.dot(sigma_points, w_m)

    # TODO: compute sigma
    t = sigma_points - np.tile(mu, (len(w_m), 1)).T
    sigma = np.dot(np.multiply(w_c, t), t.T)
    return mu, sigma


def transform(points):
    """This function applies a transformation to a set of points.
       Each column in points is one point,
       i.e. points = [[x1, y1], [x2, y2], ...]
       Select which function you want to use by uncommenting it
       (deleting the corresponding %{...%})
       while keeping all other functions commented."""

    # Function 1 (linear)
    # Applies a translation to [x; y]
    points[0, :] = points[0, :] + 1
    points[1, :] = points[1, :] + 2


    """
    # Function 2 (nonlinear)
    # Computes the polar coordinates corresponding to [x; y]
    x = points[0, :]
    y = points[1, :]
    r = np.sqrt(pow(x, 2) + pow(y, 2))
    theta = np.arctan2(y, x)
    points = np.vstack([r, theta])

    # Function 3 (nonlinear)
    points = points*np.cos(points)*np.sin(points)"""


    return points


