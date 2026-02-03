import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete, lti

class Inv_Pendulum:
    def __init__(self, M=0.5, m=0.2, b=0.1, I = 0.006, g=9.81, l=0.3):
        self.M = M      # mass of the cart
        self.m = m      # mass of the pendulum
        self.b = b      # coefficient of friction for cart
        self.I = I      # mass moment of inertia of the pendulum
        self.g = g      # acceleration due to gravity
        self.l = l      # length to pendulum center of mass

        # State-space representation
        p = self.I*(self.M + self.m) + self.M*self.m*self.l**2  # denominator for the A and B matrices

        self.A = np.array([[0, 1, 0, 0],
                           [0, -(self.I + self.m*self.l**2)*self.b/p, (self.m**2 * self.g * self.l**2)/p, 0],
                           [0, 0, 0, 1],
                           [0, -(self.m*self.l*self.b)/p, self.m*self.g*self.l*(self.M + self.m)/p, 0]])

        self.B = np.array([[0],
                           [(self.I + self.m*self.l**2)/p],
                           [0],
                           [self.m*self.l/p]])

        self.C = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        self.D = np.array([[0],
                           [0]])

        self.x = np.zeros((4,1))   # initial state vector

        dt = 0.01
        # Precompute discretization once
        M_aug = np.block([
            [self.A, self.B],
            [np.zeros((1, 5))]
        ])
        Md = expm(M_aug * dt)   # or dt

        self.Ad = Md[:4, :4]
        self.Bd = Md[:4, 4:]
        
        #self.Ad, self.Bd, self.Cd, self.Dd, dt = cont2discrete((self.A, self.B, self.C, self.D), dt, method='zoh')
        eigs = np.linalg.eigvals(self.Ad)
        print("Discrete poles:", eigs, "max magnitude:", max(abs(eigs)))

    def set_state(self, x0):
        self.x = np.asarray(x0, dtype=float).reshape(4,1)

    def step(self, u):
        self.x = self.Ad @ self.x + self.Bd * u
        return self.x.flatten()