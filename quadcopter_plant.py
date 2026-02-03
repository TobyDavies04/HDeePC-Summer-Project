import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Quadcopter 6DOF Mathematical Model
# Class for Quadcopter State and Dynamical Functions for Simulation
# 

class Quadcopter:
    def __init__(self, mass, arm_length, lift_constant, drag_constant, rotor_inertia_moment, inertia,  A_drag):
        #Inertia and A_drag are a list of the three relative parameters
        self.g = 9.80665
        self.mass = mass
        self.L = arm_length
        self.k = lift_constant
        self.b = drag_constant
        self.Ir = rotor_inertia_moment
        self.I = inertia
        self.A_drag = A_drag
        self.state = np.zeros(12)  # [x, y, z, phi, theta, psi, u, v, w, p, q, r]
        self.inputs = np.zeros(4)  # [w1, w2, w3, w4]
        self.build_mixer()

        A = np.zeros((12,12))
        A[0,6] = 1
        A[1,7] = 1
        A[2,8] = 1
        A[3, 9] = 1
        A[4, 10] = 1
        A[5, 11] = 1
        A[6, 4] = -self.g
        A[7, 3] = self.g

        B = np.zeros((12,4))
        B[8, 0] = 1 / self.mass
        B[9, 1] = 1 / self.I[0]
        B[10, 2] = 1 / self.I[1]
        B[11, 3] = 1 / self.I[2]

        C = np.zeros((6,12))
        C[0,0] = 1
        C[1,1] = 1
        C[2,2] = 1
        C[3,3] = 1
        C[4,4] = 1
        C[5,5] = 1
        self.A = A
        self.B = B
        self.C = C
        self.D = np.zeros((6,4))
 
    def body_inertial_frame(self, phi, theta, psi):
        R = np.array([
            [np.cos(theta) * np.cos(psi), np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi), 
            np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],

            [np.sin(psi) * np.cos(theta), np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi), 
            np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)],

            [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]])
        
        return R
    
    def euler_to_angular(self, phi, theta):
        Wn = np.array([
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.cos(theta) * np.sin(phi)], 
            [0, -np.sin(phi), np.cos(theta) * np.cos(phi)]])
        return Wn
    
    def build_mixer(self):
        k, L, b = self.k, self.L, self.b

        self.M = np.array([
            [ k,    k,    k,    k   ],
            [ 0,   k*L,   0,  -k*L  ],
            [-k*L,  0,   k*L,  0    ],
            [-b,    b,   -b,   b    ]
        ])

        self.M_inv = np.linalg.inv(self.M)

    def rotor_to_forces(self, omega):
        omega_sq = omega**2
        return self.M @ omega_sq  # returns [T, tau_phi, tau_theta, tau_psi]

    
    def dynamics(self, state, control_inputs):

        # Unpack state
        x, y, z, phi, theta, psi, u, v, w, p, q, r = state
        w1, w2, w3, w4 = control_inputs
        omega_g = w1 - w2 + w3 - w4
        R = self.body_inertial_frame(state[3], state[4], state[5])
        Wn = self.euler_to_angular(phi, theta)
        F = self.rotor_to_forces(control_inputs)
        Ix, Iy, Iz = self.I

        xi_dot = R @ np.array([u, v, w])     # inertial position rates
        x_dot, y_dot, z_dot = xi_dot
        
        # Translational accelerations
        ax = (F[0] / self.mass) * R[0,2] - (self.A_drag[0] * (1/self.mass) * x_dot)
        ay = (F[0] / self.mass) * R[1,2] - (self.A_drag[1] * (1/self.mass) * y_dot)
        az = (F[0] / self.mass) * R[2,2] - (self.A_drag[2] * (1/self.mass) * z_dot) - self.g
        
        # Rotational accelerations
        p_dot = ((Iy - Iz) * (q * r) / Ix) - self.Ir * omega_g * (q / Ix) + (F[1] / Ix)
        q_dot = ((Iz - Ix) * (p * r) / Iy) - self.Ir * omega_g * (-p / Iy) + (F[2] / Iy)
        r_dot = ((Ix - Iy) * (p * q) / Iz) + (F[3] / Iz)
        
        phi_dot, theta_dot, psi_dot = np.linalg.inv(Wn) @ np.array([p, q, r])

        return np.array([x_dot, y_dot, z_dot,
                         phi_dot, theta_dot, psi_dot,
                         ax, ay, az,
                         p_dot, q_dot, r_dot])

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(self, control_inputs, dt):
        x = self.state

        k1 = self.dynamics(x, control_inputs)
        k2 = self.dynamics(x + 0.5 * dt * k1, control_inputs)
        k3 = self.dynamics(x + 0.5 * dt * k2, control_inputs)
        k4 = self.dynamics(x + dt * k3, control_inputs)

        self.state = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self.state[5] = self.wrap_angle(self.state[5])   # psi

# ### Simulation Loop for Initial Testing
# quadcopter = Quadcopter(0.468, 0.225, 2.98e-6, 1.14e-7, 3.357e-5, [4.856e-3, 4.856e-3, 8.801e-3], [0.25, 0.25, 0.25]) 
#                         #mass, arm_length, lift_constant, drag_constant, rotor_inertia_moment, inertia,  A_drag
#                         #Note: Hover is around 625 rad/s
# dt = 0.0001
# sim_steps = 150000
# time = np.arange(sim_steps) * dt
# x_history = []
# y_history = []
# z_history = []
# phi_history = []
# theta_history = []
# psi_history = []
# omega_i = np.array([620.5, 620.5, 620.5, 620.5])

# w0 = 620.5          # hover-ish
# A  = 10.0           # amplitude in rad/s (start small: 2–10)
# T_pulse = 4.0

# for k in range(sim_steps):
#     t = k*dt

#     if 0 <= t <= T_pulse:
#         # smooth envelope: starts/ends at 0
#         env = 0.5*(1 - np.cos(2*np.pi*t/T_pulse))
#         d = A * env * np.sin(2*np.pi*(t/T_pulse))  # one oscillation inside envelope
#     else:
#         d = 0.0

#     omega_i = np.array([w0, w0 + d, w0, w0 - d])
#     quadcopter.step(omega_i, dt)

#     x_history.append(quadcopter.state[0])
#     y_history.append(quadcopter.state[1])
#     z_history.append(quadcopter.state[2])
#     phi_history.append(quadcopter.state[3])
#     theta_history.append(quadcopter.state[4])
#     psi_history.append(quadcopter.state[5])

# plt.figure(figsize=(8, 5))
# plt.plot(time, x_history, label='x (m)')
# plt.plot(time, y_history, label='y (m)')
# plt.plot(time, z_history, label='z (m)')
# plt.xlabel('Time (s)')
# plt.ylabel('Position (m)')
# plt.title('Quadcopter Position vs Time (Open-loop, Equal Rotor Speeds)')
# plt.legend()
# plt.grid(True)

# plt.figure(figsize=(8, 5))
# rad2deg = 180 / np.pi
# plt.plot(time, np.array(phi_history) * rad2deg, label='Roll φ (deg)')
# plt.plot(time, np.array(theta_history) * rad2deg, label='Pitch θ (deg)')
# plt.plot(time, np.array(psi_history) * rad2deg, label='Yaw ψ (deg)')
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (rad)')
# plt.title('Quadcopter Euler Angles vs Time (Open-loop)')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()