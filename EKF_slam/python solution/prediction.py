from tools import *

def prediction(mu, sigma, u):
    #Updates the belief concerning the robot pose according to 
    #the motion model (From the original MATLAB code: Use u.r1,
    #u.t, and u.r2 to access the rotation and translation values)
    #In this case u['r1'], u['t'] and u['r2'] to access values

    # mu: 2N+3 x 1 vector representing the state mean
    # sigma: 2N+3 x 2N+3 covariance matrix
    # u: odometry reading dictionary ('r1', 't', 'r2')

    # TODO: Compute the new mu based on the noise-free (odometry-based) motion model
    # Remember to normalize theta after the update (hint: use the function normalize_angle available in tools)
    tran = u['t']
    rot1 = u['r1']
    rot2 = u['r2']

    # TODO: Compute the 3x3 Jacobian Gx of the motion model
    
    Gtx = np.zeros((3,3))
    Gtx[0,2] = -tran*np.sin(mu[2]+rot1)
    Gtx[1,2] = tran*np.cos(mu[2]+rot1)

    mu[:3]+= np.array([tran*np.cos(mu[2]+rot1),
                       tran*np.sin(mu[2]+rot1),
                       rot2+rot1])

    mu[2] = normalize_angle(mu[2])
    

    # TODO: Construct the full Jacobian G
    nLandmarks = (len(mu)-3)//2
    Fx = np.concatenate((np.eye(3),np.zeros((3,2*nLandmarks))),axis=1)
    Gt = np.eye(len(mu))+np.dot(Fx.T,np.dot(Gtx,Fx))

    # Motion noise
    motionNoise = 0.1
    R3 = np.array([[motionNoise, 0., 0.], 
                      [0., motionNoise, 0.], 
                      [0., 0., motionNoise/10.]])
    
    R = np.zeros((sigma.shape[0],sigma.shape[0]))
    R[:3,:3] = R3

    # TODO: Compute the predicted sigma after incorporating the motion
    sigma = np.dot(Gt,np.dot(sigma,Gt.T))+R

    return mu, sigma

