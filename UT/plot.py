import scipy
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.patches import Arrow


def draw_probe_ellipse(xy, covar, alpha, color=None, **kwargs):
    """Generates an ellipse object based of a point and related
    covariance assuming 2 dimensions

    Args:
        xy (2x1 array): (x,y) of the ellipse position
        covar (2x2 array): covariance matrix of landmark point
        alpha (float):

    Kwargs:
        color (string): matplotlib color convention
    Returns:
         (matplotlib Ellipse Object): Ellipse object for drawing

    """

    b24ac = scipy.sqrt(pow(covar[0, 0] - covar[1, 1], 2) + 4 * pow(covar[0, 1], 2))
    c2inv = chi2.ppf(alpha, 2.)

    a = scipy.real(scipy.sqrt(c2inv * .5 * (covar[0, 0] + covar[1, 1] + b24ac)))
    b = scipy.real(scipy.sqrt(c2inv * .5 * (covar[0, 0] + covar[1, 1] - b24ac)))

    if covar[0, 0] != covar[1, 1]:
        theta = .5 * scipy.arctan(2 * covar[0, 1] / (covar[0, 0] - covar[1, 1]))
    else:
        theta = scipy.sign(covar[0, 1]) * scipy.pi / 4

    if covar[1, 1] > covar[0, 0]:
        swap = a
        a = b
        b = swap

    ellipse = Ellipse(xy, 2 * a, 2 * b, angle=theta * 180. / scipy.pi, edgecolor=color, fill=False, **kwargs)
    plt.gca().add_patch(ellipse)
    return ellipse


def _rot(theta, vec):
    """ there are a number of vector rotations in draw robot that are
    not necessary to individually program.
    """

    rmat = scipy.array([[scipy.cos(theta), -1 * scipy.sin(theta)],
                        [scipy.sin(theta), scipy.cos(theta)]])
    return scipy.dot(rmat, vec)