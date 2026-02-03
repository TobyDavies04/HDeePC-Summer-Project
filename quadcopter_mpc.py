import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from MPC_Controller import MPC_Controller
from quadcopter_plant import Quadcopter

### Quadcopter Plant
quadcopter = Quadcopter(0.468, 0.225, 2.98e-6, 1.14e-7, 3.357e-5, [4.856e-3, 4.856e-3, 8.801e-3], [0.25, 0.25, 0.25]) 
                        #mass, arm_length, lift_constant, drag_constant, rotor_inertia_moment, inertia,  A_drag
                        #Note: Hover is around 625 rad/s
dt = 0.001
sim_steps = 20000
time = np.arange(sim_steps) * dt
x_history = []
y_history = []
z_history = []
phi_history = []
theta_history = []
psi_history = []
w_i_history = np.array([]).reshape(0,4)

### State Space Model of Quadcopter Plant 
g = 9.80665  # gravitational acceleration (m/s^2)
A = np.zeros((12,12))
A[0,6] = 1
A[1,7] = 1
A[2,8] = 1
A[3, 9] = 1
A[4, 10] = 1
A[5, 11] = 1
A[6, 4] = -g
A[7, 3] = g

B = np.zeros((12,4))
B[8, 0] = 1 / quadcopter.mass
B[9, 1] = 1 / quadcopter.I[0]
B[10, 2] = 1 / quadcopter.I[1]
B[11, 3] = 1 / quadcopter.I[2]

C = np.zeros((6,12))
C[0,0] = 1
C[1,1] = 1
C[2,2] = 1
C[3,3] = 1
C[4,4] = 1
C[5,5] = 1

D = np.zeros((6,4))

Q = np.diag([0, 0, 200, 600, 600, 600])
R = np.diag([0.1, 0.1, 0.1, 0.1])

# u_min = np.array([[0],[0],[0],[0]])
# u_max = np.array([[2000],[2000],[2000],[2000]])
u_min = np.array([0, -1, -1, -1])
u_max = np.array([10, 1, 1, 1])
# du_min = np.array([-1, -0.1, -0.1, -0.1])
# du_max = np.array([1, 0.1, 0.1, 0.1])
du_min = None
du_max = None

ref = np.array([0, 0, 0, 0, 0, 0])  # desired position and orientation, [x, y, z, phi, theta, psi]
radius = 0.5          # radius (m)
omega = 1.0    # rad per timestep
z_ref = 0.0005

mpc = MPC_Controller(A, B, C, D, Q, R, dt, ref, u_min, u_max, du_min, du_max, Np=50, Nc=30)

for k in range(sim_steps):
    # if k == 15000:
    #     ref = np.array([0, 0, 1, 0, 0, 0])  # new reference position
    # if k == 5000:
    #     ref = np.array([0, z_ref * k, 1, 0.0, 0.0, 0.0])

    t = k * dt
    theta_traj = omega * t

    # Desired position (for logging only)
    x_ref = radius * np.cos(theta_traj)
    y_ref = radius * np.sin(theta_traj)

    # Desired accelerations
    x_ddot_ref = -radius * omega**2 * np.cos(theta_traj)
    y_ddot_ref = -radius * omega**2 * np.sin(theta_traj)

    # Convert accel → attitude (small-angle hover)
    phi_ref   = y_ddot_ref / g
    theta_ref =  -x_ddot_ref / g

    # MPC reference (THIS is the key)
    ref = np.array([
        0.0,        # x ignored
        0.0,        # y ignored
        z_ref * k,      # altitude
        phi_ref,    # roll
        theta_ref,  # pitch
        0.0         # yaw
    ])

    mpc.x = quadcopter.state.reshape(-1,1).copy()
    mpc.ref = ref.reshape(-1,1)

    x_pred, y_pred, u_applied = mpc.step()

    hover_force = quadcopter.mass * g
    force_vect = u_applied.flatten()
    force_vect[0] += hover_force
    #print("Control inputs (rotor speeds):", force_vect)
    # 4) Apply MPC control input to the real inverted pendulum

    w_i = np.sqrt(np.clip(quadcopter.M_inv @ force_vect, 0, None))
    quadcopter.step(w_i, dt)

    x_history.append(quadcopter.state[0])
    y_history.append(quadcopter.state[1])
    z_history.append(quadcopter.state[2])
    phi_history.append(quadcopter.state[3])
    theta_history.append(quadcopter.state[4])
    psi_history.append(quadcopter.state[5])
    w_i_history = np.vstack([w_i_history, w_i.reshape(1,4)])
    print(f"Step {k+1}/{sim_steps} completed.", end='\r')

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(time, w_i_history[:, 0], label='w1 (rad/s)')
plt.plot(time, w_i_history[:, 1], label='w2 (rad/s)')
plt.plot(time, w_i_history[:, 2], label='w3 (rad/s)')
plt.plot(time, w_i_history[:, 3], label='w4 (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('Rotor Speeds (rad/s)')
plt.title('Quadcopter Propellor vs Time')
plt.legend()
plt.grid(True)


plt.figure(figsize=(8, 5))
plt.plot(time, x_history, label='x (m)')
plt.plot(time, y_history, label='y (m)')
plt.plot(time, z_history, label='z (m)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Quadcopter Position vs Time')
plt.legend()
plt.grid(True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_history, y_history, z_history, linewidth=2)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Quadrotor 3D Trajectory')

plt.figure(figsize=(8, 5))
rad2deg = 180 / np.pi
plt.plot(time, np.array(phi_history) * rad2deg, label='Roll φ (deg)')
plt.plot(time, np.array(theta_history) * rad2deg, label='Pitch θ (deg)')
plt.plot(time, np.array(psi_history) * rad2deg, label='Yaw ψ (deg)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Quadcopter Euler Angles vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()