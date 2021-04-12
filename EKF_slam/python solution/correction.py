from tools import *
def correction(mu, sigma, z, observedLandmarks):
    # Updates the belief, i. e., mu and sigma after observing landmarks, according to the sensor model
    # The employed sensor model measures the range and bearing of a landmark
    # mu: 2N+3 x 1 vector representing the state mean.
    # The first 3 components of mu correspond to the current estimate of the robot pose [x; y; theta]
    # The current pose estimate of the landmark with id = j is: [mu(2*j+2); mu(2*j+3)]
    # sigma: 2N+3 x 2N+3 is the covariance matrix
    # z: struct array containing the landmark observations.
    # Each observation z(i) has an id z(i).id, a range z(i).range, and a bearing z(i).bearing
    # The vector observedLandmarks indicates which landmarks have been observed
    # at some point by the robot.
    # observedLandmarks(j) is false if the landmark with id = j has never been observed before.

    # Number of measurements in this time step
    m = len(z)

    # Z: vectorized form of all measurements made in this time step: [range_1; bearing_1; range_2; bearing_2; ...; range_m; bearing_m]
    # ExpectedZ: vectorized form of all expected measurements in the same form.
    # They are initialized here and should be filled out in the for loop below
    Z = np.zeros((m*2,))
    expectedZ = np.zeros((m*2,))

    # Iterate over the measurements and compute the H matrix
    # (stacked Jacobian blocks of the measurement function)
    # H will be 2m x 2N+3
    H = []
    for i in range(m):
        #Get the id of the landmark corresponding to the i-th observation
        landmarkId = z[i]['id']
        # If the landmark is obeserved for the first time:
        if (observedLandmarks[landmarkId] == False):
            # TODO: Initialize its pose in mu based on the measurement and the current robot pose:
            mu[2*landmarkId + 3] = mu[0] + z[i]['range']*np.cos(z[i]['bearing'] + mu[2])#x position of landmark
            mu[2*landmarkId + 4] = mu[1] + z[i]['range']*np.sin(z[i]['bearing'] + mu[2])#y position of landmark

            # Indicate in the observedLandmarks vector that this landmark has been observed
            observedLandmarks[landmarkId] = True
    

        # TODO: Add the landmark measurement to the Z vector
        Z[2*i] = z[i]['range']
        Z[2*i+1] = normalize_angle(z[i]['bearing'])
    
        # TODO: Use the current estimate of the landmark pose
        # to compute the corresponding expected measurement in expectedZ:
        ux = mu[2*landmarkId + 3] - mu[0]
        uy = mu[2*landmarkId + 4] - mu[1]
        r = np.sqrt(pow(ux,2) + pow(uy,2)) 
            
        expectedZ[2*i] = r #again, this could have been vectorized with a different approach to the landmarks
        expectedZ[2*i + 1] = normalize_angle(np.arctan2(uy,ux) - mu[2])
    
        Hi = scipy.zeros((2, len(mu)))

        Hi[:,0] = scipy.array([-ux*r, uy])
        Hi[:,1] = scipy.array([-uy*r, -ux])
        Hi[:,2] = scipy.array([0., -pow(r,2)])
        Hi[:,2*landmarkId + 3] = scipy.array([ux*r, -uy])
        Hi[:,2*landmarkId + 4] = scipy.array([uy*r, ux])
        # Augment H with the new Hi
        H += [Hi/pow(r,2)]	

        
    H = scipy.vstack(H)
    # TODO: Construct the sensor noise matrix Q
    sensor_noise = 0.01
    Q = sensor_noise*np.eye(2*m)

    #TODO: Compute the Kalman gain
    invr = np.linalg.inv(np.dot(H,np.dot(sigma,H.T))+Q)
    Kt = np.dot(sigma,np.dot(H.T,invr))


    #TODO: Compute the difference between the expected and recorded measurements.
    # Remember to normalize the bearings after subtracting!
    # (hint: use the normalize_all_bearings function available in tools)
    delta = normalize_all_bearings(Z - expectedZ)

    # TODO: Finish the correction step by computing the new mu and sigma.
    # Normalize theta in the robot pose.
    #mu = np.expand_dims(mu, axis=1)
    mu += np.dot(Kt, delta)
    sigma = np.dot(np.eye(len(sigma)) - np.dot(Kt, H),sigma)
    #mu = np.squeeze(mu, axis=1)


    return mu, sigma, observedLandmarks