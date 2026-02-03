# dc_motor.py

import numpy as np
from scipy.linalg import expm
    
class DCMotor_State_Space:
    def __init__(self, R=1.0, L=0.5, K=0.01, J=0.01, b=0.1):
        self.A = np.array([[-R/L, -K/L],
                           [ K/J, -b/J]])

        self.B = np.array([[1/L],
                           [0.0]])   # COLUMN vector

        self.C = np.array([[0, 1]])

        self.D = np.array([[0]])

        self.x = np.zeros((2,1))

    def set_state(self, x0):
        self.x = np.asarray(x0, dtype=float).reshape(2,1)

    def step(self, u, dt):
        # dx = self.A @ self.x + self.B * float(u)
        # self.x += dx * dt
        # return self.x[1,0]    # return omega scalar

        # Exact discretization
        M = np.block([
            [self.A, self.B],
            [np.zeros((1, 3))]
        ])
        Md = expm(M * dt)

        Ad = Md[:2, :2]
        Bd = Md[:2, 2:]

        self.x = Ad @ self.x + Bd * u
        return self.x[1,0]

class DCMotor:
    def __init__(self, R=0.5, L=0.5, K=0.01, J=0.01, b=0.01):
        self.R = R
        self.L = L
        self.K = K
        self.J = J
        self.b = b
        self.x = np.zeros(2)

    def set_state(self, x0):
        x0 = np.asarray(x0, dtype=float)

        if x0.shape != (2,):
            raise ValueError("State must be length 2: [i, omega]")

        self.x = x0.copy()     # store an independent copy

    def step(self, u, dt):
        i, omega = self.x
        di_dt = (u - self.R*i - self.K*omega)/self.L
        domega_dt = (self.K*i - self.b*omega)/self.J
        self.x[0] += di_dt * dt
        self.x[1] += domega_dt * dt
        return self.x[1]