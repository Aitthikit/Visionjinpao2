import numpy as np
class KF:
    def __init__(self, Y):
        # Initial error covariance matrix
        self.P = np.eye(2)

        # Measurement matrix (assuming we can directly measure the state)
        self.H = np.eye(2)

        self.Q = np.eye(2) 

        # Measurement noise covariance matrix 0.001
        self.R = np.eye(2) * 0.001

        self.X = np.array([Y[0], 0.785])
    def update(self,Y,dt):
        self.A = np.array([[1, dt],
                 [0, 1]])
    
    
        # update
        self.X = self.A@self.X

        # print(X)
        self.P = self.A @ self.P @ self.A.T + self.Q

        # # Measure
        self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        # input
        

        #state update

        self.X = self.X + self.K @ (Y - self.H @ self.X)
        self.P = (np.eye(2) - self.K @ self.H) @ self.P