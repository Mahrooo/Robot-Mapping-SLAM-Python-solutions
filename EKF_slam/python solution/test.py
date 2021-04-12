from tools import *
import EKF_SLAM
import prediction
import correction

# This is the main extended Kalman filter SLAM loop. This script calls all the required
# functions in the correct order.
#
# You can disable the plotting or change the number of steps the filter
# runs for to ease the debugging. You should however not change the order
# or calls of any of the other lines, as it might break the framework.
#
# If you are unsure about the input and return values of functions you
# should read their documentation which tells you the expected dimensions.

# Read world data, i.e. landmarks. The true landmark positions are not given to the robot
landmarks =read_world('world.dat') # dict with 'id', 'x', and 'y' keys.  

# load landmarks;
# Read sensor readings, i.e. odometry and range-bearing sensor
data =read_data('sensor_data.dat')
# data['odometry']: motion model
# data['sensor']: range-bearing observation model

infty = 1000.
# Get the number of landmarks in the map
n = len(landmarks['id'])

# observedLandmarks is a vector that keeps track of which landmarks have been observed so far.
# observedLandmarks(i) will be true if the landmark with id = i has been observed at some point by the robot
observedLandmarks = np.zeros((n,)).astype(bool)#repmat(false,1,N);

# Initialize belief:
# mu: 2N+3x1 vector representing the mean of the normal distribution
# The first 3 components of mu correspond to the pose of the robot,
# and the landmark poses (xi, yi) are stacked in ascending id order.
# sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution
mu = np.zeros((2*n + 3,))
robSigma =  np.zeros((3, 3))
robMapSigma = np.zeros((3, 2*n))
mapSigma = infty*np.eye(2*n)
sigma =  np.zeros((2*n + 3, 2*n + 3))
sigma[:3,:3] = robSigma
sigma[:3,3:] = robMapSigma
sigma[3:,:3] = robMapSigma.T
sigma[3:,3:] = mapSigma

# toogle the visualization type
showGui = True # show a window while the algorithm runs
#showGui = False # plot to files instead

# Perform filter update for each odometry-observation pair read from the
# data file.
x = []
y = []
#frames = [] # for storing the generated images
for t in range(len(data['sensor'])):
#for t in range(20): #for debugging 
   
   # Perform the prediction step of the EKF
   mu, sigma = prediction.prediction(mu, sigma, data['odometry'][t])

   # Perform the correction step of the EKF
   mu, sigma, observedLandmarks = correction.correction(mu, sigma, data['sensor'][t], observedLandmarks)

   #Generate visualization plots of the current state of the filter
   fig = plot_state(mu, sigma, landmarks, t, observedLandmarks, data['sensor'][t], showGui,3)

   print(r'Current state vector:')
   print('mu = '),
   print(mu[0])
   x.append([mu[0]])
   y.append([mu[1]])
   

print("Final system covariance matrix: %f".format(sigma))
# Display the final state estimate
print("Final robot pose:")
print("mu_robot = ",  mu[:3],"sigma_robot = ", sigma[:3,:3])
#path = np.vstack(path)
plt.plot(x,y,'c-', fillstyle='none', linewidth=2)
plt.show()


