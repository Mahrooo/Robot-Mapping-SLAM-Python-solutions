import numpy as np
import scipy
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.patches import Arrow

def read_world(filename):
    # Reads the world definition and returns a dictionary of landmarks.
    #
    # filename: path of the file to load
    # landmarks: dictionary containing the parsed information
    #
    # Each landmark contains the following information:
    # - id : id of the landmark
    # - x  : x-coordinate
    # - y  : y-coordinate
    #
    # Examples:
    # - Obtain x-coordinate of the 5-th landmark
    #   output["x"][4]

    data = np.loadtxt(filename).T
    output = {'id': data[0,:], 'x': data[1,:], 'y': data[2,:]}
    
    return output


def read_data(filename):
    
    # Reads the odometry and sensor readings from a file.
    #
    # filename: path to the file to parse
    # data: structure containing the parsed information
    #
    # The data is returned in a structure where the u_t and z_t are stored
    # within a single entry. A z_t can contain observations of multiple
    # landmarks.
    #
    # Usage:
    # - access the readings for timestep i:
    #   data.timestep(i)
    #   this returns a structure containing the odometry reading and all
    #   landmark obsevations, which can be accessed as follows
    # - odometry reading at timestep i:
    #   data.timestep(i).odometry
    # - senor reading at timestep i:
    #   data.timestep(i).sensor
    #
    # Odometry readings have the following fields:
    # - r1 : rotation 1
    # - t  : translation
    # - r2 : rotation 2
    # which correspond to the identically labeled variables in the motion
    # mode.
    #
    # Sensor readings can again be indexed and each of the entris has the
    # following fields:
    # - id      : id of the observed landmark
    # - range   : measured range to the landmark
    # - bearing : measured angle to the landmark (you can ignore this)
    #
    # Examples:
    # - Translational component of the odometry reading at timestep 10
    #   data.timestep(10).odometry.t
    # - Measured range to the second landmark observed at timestep 4
    #   data.timestep(4).sensor(2).range
    output = {'sensor':[],'odometry':[]}

    data = np.genfromtxt(filename, dtype='object')
    idx = np.squeeze(data[:,0] == b'ODOMETRY')
    for inp in data[idx,1:].astype(float):
        output['odometry'] += [{'r1':inp[0],
                                    't':inp[1],
                                    'r2':inp[2]}]

    idxarray =np.where(idx)
    idxarray = np.append(idxarray,[len(idx)])
    for i in range(len(idxarray) - 1):
        temp = []
        
        for j in np.arange(idxarray[i] + 1, idxarray[i + 1]):
            temp += [{'id':int(data[j,1]) - 1,
                      'range':float(data[j,2]),
                      'bearing':float(data[j,3])}]
                
        output['sensor'] += [temp]
    return output


def normalize_angle(phi):
    # Normalize phi to be between -pi and pi
    return (phi + np.pi) % (2*np.pi) - np.pi
    

def normalize_all_bearings(z):
    # Go over the observations vector and normalize the bearings
    # The expected format of z is [range; bearing; range; bearing; ...]
    z[1::2] = normalize_angle(z[1::2])

    return z


def plot_state(mu, sigma, landmarks, timestep, observedLandmarks, z, window,type_robot):
    # Visualizes the state of the EKF SLAM algorithm.
    #
    # The resulting plot displays the following information:
    # - map ground truth (black +'s)
    # - current robot pose estimate (red)
    # - current landmark pose estimates (blue)
    # - visualization of the observations made at this time step (line between robot and landmark)

    plt.clf()
    plt.grid('on')
    draw_probe_ellipse(mu[:3], sigma[:3,:3], 0.6, 'r')
    plt.plot(landmarks['x'], landmarks['y'], 'k+', markersize=10, linewidth=5)

    for i in range(len(observedLandmarks)):
        if observedLandmarks[i]:
            plt.plot(mu[2*i + 3],mu[2*i + 4], 'bo', fillstyle='none', markersize=10, linewidth=5)
            draw_probe_ellipse(mu[2*i + 3:2*i+ 5], sigma[2*i + 3:2*i+ 5,2*i + 3:2*i + 5], 0.6, 'b')
        
    for i in range(len(z)):
        mX = mu[2*z[i]['id'] + 3]
        mY = mu[2*z[i]['id'] + 4]
        plt.plot([mu[0], mX], [mu[1], mY], color='k', linewidth=1)

    drawrobot(mu[:3], 'g', type_robot, 0.3, 0.3)
    plt.xlim([-2., 12.])
    plt.ylim([-2., 12.])
    plt.title('EKF-SLAM, t=%i' %timestep)

    if window:
        plt.draw()
        plt.pause(0.1)
    else:
        filename = '../ekf_%03d.png'.format(timestep)
        plt.savefig(filename)


def drawrobot(xvec, color, type=2, W=.2, L=.6):
    """Draws a robot at a set pose using matplotlib in current plot
   
    Args:
        xvec (3x1 array): robot position and direction
        color (string or rbg or rgba array): color following matplotlib specs      positions
    
    Kwargs:
        type (int [0:5]): dictates robot to be drawn with follow selections:
            - 0 : draws only a cross with orientation theta
            - 1 : draws a differential drive robot without contour
            - 2 : draws a differential drive robot with round shape
            - 3 : draws a round shaped robot with a line at theta
            - 4 : draws a differential drive robot with rectangular shape
            - 5 : draws a rectangular shaped robot with a line at theta
        W (float): robot width [m]    
        L (float): robot length [m]
    
    Returns:
        h (list): matplotlib object list added to current axes
    """
    
    theta = xvec[2]
    t = scipy.array([xvec[0], xvec[1]])
    r = []
    h = []
    
    if type ==0:
        cs = .1
        h += [plt.plot([cs,-cs,None,0.,0.]+t[0],
                       [0.,0.,None,cs,-cs]+t[1],
                       color,
                       lw=2.)]
    elif type == 1:
        xy = W*scipy.array((scipy.cos(theta + scipy.pi/2),
                            scipy.sin(theta + scipy.pi/2)))
        
        temp = Rectangle(t + xy, .03, .02, color=color, angle=theta)
        h += [plt.gca().add_artist(temp)]
        temp = Rectangle(t - xy, .03, .02, color=color, angle=theta)
        h += [plt.gca().add_artist(temp)]
        rin = _rot(theta,scipy.array([0, W + .03]))
        
        temp = Arrow(xvec[0] - rin[0],
                     xvec[1] - rin[1],
                     rin[0],
                     rin[1],
                     color=color)
        h += [temp]
        plt.gca().add_artist(temp)
        
    elif type == 2:
        xy = W*np.array((np.cos(theta + np.pi/2),
                            np.sin(theta + np.pi/2)))
        
        temp = Rectangle(t + xy, .03, .02, color=color, angle=theta)
        plt.gca().add_artist(temp)
        temp = Rectangle(t - xy, .03, .02, color=color, angle=theta)
        plt.gca().add_artist(temp)
        
        #axis between wheels here (requires a rotation)
        
        # The lines from the matlab come with no explanation, but do matrix
        #math to yield a rotated arrow
        rin = _rot(theta,np.array([0,W + .015]))
        
        temp = Arrow(xvec[0] - rin[0],
                     xvec[1] - rin[1],
                     rin[0],
                     rin[1],
                     color=color)
        plt.gca().add_artist(temp)
        
    elif type == 3:
        temp = Ellipse(xvec[:2],
                       W + .015,
                       W + .015,
                       angle=theta,
                       edgecolor=color,
                       fill=False)
        plt.gca().add_artist(temp)
        
        rin = _rot(theta,np.array([W + .015,0]))
        plt.plot(xvec[0]+np.array([-rin[0],rin[0]]),
                 xvec[1]+np.array([-rin[1],rin[1]]),
                 color=color,
                 lw=2.)
        
    elif type == 4:
        xy = W*np.array((np.cos(theta + np.pi/2),
                            np.sin(theta + np.pi/2)))
        
        temp = Rectangle(t + xy, .03, .02, color=color, angle=theta)
        plt.gca().add_artist(temp)
        h += [temp]
        
        temp = Rectangle(t - xy, .03, .02, color=color, angle=theta)
        plt.gca().add_artist(temp)
        h +=[temp]
                
        rin = _rot(theta,np.array([W + .015,0]))
        h += [plt.plot(xvec[0]+np.array([-rin[0],rin[0]]),
                       xvec[1]+np.array([-rin[1],rin[1]]),
                       color=color,
                       lw=2.)] 
        
        temp = Arrow(xvec[0] - rin[0],
                     xvec[1] - rin[1],
                     rin[0],
                     rin[1],
                     color=color)
        h += [temp]
                       
        temp = Rectangle(t, L, W, color=color, angle=theta)
        plt.gca().add_artist(temp)
        h +=[temp] 
        
        
    elif type == 5:
        rin = _rot(theta,np.array([W + .015,0]))
        h += [plt.plot(xvec[0]+np.array([-rin[0],rin[0]]),
                       xvec[1]+np.array([-rin[1],rin[1]]),
                       color=color,
                       lw=2.)] 
        
        temp = Arrow(xvec[0] - rin[0],
                     xvec[1] - rin[1],
                     rin[0],
                     rin[1],
                     color=color)
        h += [temp]
                       
        temp = Rectangle(t, L, W, color=color, angle=theta)
        plt.gca().add_artist(temp)
        h +=[temp] 
        
    else:
        raise ValueError('type out of bounds')
    
def draw_probe_ellipse(xy, covar, alpha, color=None, **kwargs):
    """Generates an ellipse object based of a point and related
    covariance assuming 2 dimensions
   
    Args:
        xy (2x1 array): (x,y) of the ellipse position
        covar (2x2 array): covariance matrix of landmark point
        alpha (float):
   
    Kwargs:   
        color (string): matplotlib color convention for ellipse edge
    Returns:
         (matplotlib Ellipse Object): Ellipse object for drawing
 
    """
    
    b24ac = np.sqrt(pow(covar[0,0] - covar[1,1],2) + 4*pow(covar[0,1],2))
    c2inv = chi2.ppf(alpha, 2.)
    
    a = np.real(np.sqrt(c2inv*.5*(covar[0,0] + covar[1,1] + b24ac)))
    b = np.real(np.sqrt(c2inv*.5*(covar[0,0] + covar[1,1] - b24ac)))

    if covar[0,0] != covar[1,1]:
        theta = .5*np.arctan(2*covar[0,1]/(covar[0,0] - covar[1,1]))
        print(theta)
    else:
        theta = np.sign(covar[0,1])*np.pi/4
        
    if covar[1,1] > covar[0,0]:
        swap = a
        a = b
        b = swap

    ellipse = Ellipse(xy, 2*a, 2*b, angle=theta*180./scipy.pi, edgecolor=color, fill=False, **kwargs)
    plt.gca().add_patch(ellipse)
    return ellipse

    
def _rot(theta, vec):
    """ there are a number of vector rotations in draw robot that are 
    not necessary to individually program.
    """

    rmat = np.array([[np.cos(theta), -1*np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]]) 
    return np.dot(rmat,vec)

    




