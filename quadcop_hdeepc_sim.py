import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from HDeePC_Controller import HDeePC_Controller
from quadcopter_plant import Quadcopter

#### HDeePC Control of Quadcopter with Cascaded Position and Attitude Control Loops

### Quadcopter Plant
quadcopter = Quadcopter(0.468, 0.225, 2.98e-6, 1.14e-7, 3.357e-5, [4.856e-3, 4.856e-3, 8.801e-3], [0.25, 0.25, 0.25]) 
                        #mass, arm_length, lift_constant, drag_constant, rotor_inertia_moment, inertia,  A_drag
                        #Note: Hover is around 625 rad/s
dt = 0.001
sim_steps = 2000
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
A = quadcopter.A
B = quadcopter.B
C = quadcopter.C
D = quadcopter.D

print("Quadcopter State-Space Matrices:")
print("A =", A) 
print("B =", B)
print("C =", C)
print("D =", D)

# Outer-loop position gains
Kp_xy = 1.5
Kd_xy = 2.0

Kp_z = 3.0
Kd_z = 2.0

Q = [0, 0, 10000, 600, 600, 600]
R = [0.1, 0.1, 0.1, 0.1]

u_min = np.array([0, -1, -1, -1])
u_max = np.array([10, 1, 1, 1])
du_min = None
du_max = None

# DeePC parameters
# u_data = np.load("u_history_IP.npy")   #Need to get data for quadcopter
# x_data = np.load("x_history_IP.npy")

#Added_Code
T_data = 300  # data length
m = 4     # number of inputs
p_u = 6   # unknown outputs: angles + angular rates

u_d = 6 * (np.random.rand(m, T_data+1) - 0.5)
y_d = np.zeros((p_u, T_data))
x_d = np.zeros((12, T_data+1))


x_d[:, 0] = np.random.rand(12)
noiseM = 0.01

for k in range(T_data):
    # propagate full nonlinear quadcopter model
    quadcopter.state = x_d[:, k].copy()
    quadcopter.step(u_d[:, k], dt)
    x_d[:, k+1] = quadcopter.state.copy()

    # extract unknown outputs
    u, v, w = x_d[6:9, k]
    p, q, r = x_d[9:12, k]   # angular rates (or however stored)

    noise = noiseM * 2 * (np.random.rand(p_u) - 0.5)
    y_d[:, k] = np.array([u, v, w, p, q, r]) + noise

u_d = u_d.T  # shape (T_data, m)
u_d = u_d[:-1, :]  # remove last input to match y_d length
y_d = y_d.T  # shape (T_data, p_u)    
print("input data shape:", u_d.shape)
print("output data shape:", y_d.shape)
#Finished_added_code

T_ini = 6
x0 = quadcopter.state.copy()                    # [1, 0, 0, 0]
y0 = x0[6:]            # outputs: [position, angle]

#HDeePC specific system matrices
nu = 12  # number of unknown states
nk = 0 # number of known states
pu = 6   # number of unknown outputs
pk = 0   # number of known outputs
NP = [nu, nk, pu, pk]

#Chek these are correct
Ac = A[nu:, :nu]
Ak = A[nu:, nu:]
Bk = B[nu:, :]
Cc = C[pu:, :pu]
Ck = C[pu:, pu:]
Dk = D[pu:, :]

Cy = np.zeros((pk, pu))
Ay = Ac.copy()

# Past outputs: assume we sat at x0 for T_ini steps with zero input
# y_uini = np.tile(y0[:pu].reshape(1, pu), (T_ini, 1))
# u_ini = np.zeros((T_ini, 4))                          # shape (T_ini, 1)
u_ini = u_d[:T_ini, :]
y_uini = y_d[:T_ini, :]
print("y_uini shape:", y_uini.shape)
print("u_ini shape:", u_ini.shape)

N = 60

# Histories for simulation start
state_history = np.empty((0, nu+nk))
y_u_history = y_uini.copy()
u_history = u_ini.copy()

lambda_g = 3e1
lambda_y = 1e6
lambda_u = 1e6

ref = np.array([0, 0, 0, 0, 0, 0])  # desired position and orientation, [x, y, z, phi, theta, psi]
radius = 1          # radius (m)
omega = 2.5    # rad per timestep
#z_ref = 0.0005

hdeepc = HDeePC_Controller(Ac, Ak, Ay, Bk, Cc, Ck, Cy, Dk, u_d,
                            y_d, u_ini, y_uini, N, Q, R, ref, u_min, u_max, 
                            du_min, du_max, lambda_g, lambda_y, lambda_u, NP, calculate_Ay_Cy=False)

for k in range(sim_steps):
    t = k * dt
    if k < 10000:
        # Desired position
        x_d = radius * np.cos(omega * t)
        y_d = radius * np.sin(omega * t)

        z_d = 0.0001 * k

        # Desired velocity
        xd_d = -radius * omega * np.sin(omega * t)
        yd_d =  radius * omega * np.cos(omega * t)
        zd_d = 0.0

        # Desired acceleration (feedforward)
        xdd_d = -radius * omega**2 * np.cos(omega * t)
        ydd_d = -radius * omega**2 * np.sin(omega * t)
        zdd_d = 0.0
    elif k >= 10000:
        x_d = 3
        y_d = 3
        z_d = 3

        xd_d = 0.0
        yd_d = 0.0
        zd_d = 0.0

        xdd_d = 0.0
        ydd_d = 0.0
        zdd_d = 0.0


    x, y, z = quadcopter.state[0:3]
    phi, theta, psi = quadcopter.state[3:6]
    xd, yd, zd = quadcopter.state[6:9]

    # Position errors
    ex = x_d - x
    ey = y_d - y
    ez = z_d - z

    # Velocity errors
    evx = xd_d - xd
    evy = yd_d - yd
    evz = zd_d - zd

    # Commanded accelerations
    ax_cmd = Kp_xy * ex + Kd_xy * evx + xdd_d
    ay_cmd = Kp_xy * ey + Kd_xy * evy + ydd_d
    az_cmd = Kp_z  * ez + Kd_z  * evz + zdd_d

    theta_ref =  ax_cmd / g
    phi_ref   = -ay_cmd / g

    # Total thrust command
    T_ref = quadcopter.mass * (g + az_cmd)

    # MPC reference (THIS is the key)
    ref = np.array([
        0.0,        # x ignored
        0.0,        # y ignored
        z_d,      # altitude
        phi_ref,    # roll
        theta_ref,  # pitch
        0.0         # yaw
    ])

    #change this to using the update function
    u_ini_new = u_history[-T_ini:] #Get last T_ini inputs
    y_ini_new = y_u_history[-T_ini:, :pu]  
    x = quadcopter.state.copy()
    xk = x[nu:]    
    hdeepc.update(u_ini_new, y_ini_new, xk, ref)

    u_applied = hdeepc.opt_problem()
    #print("Optimized control input:", u_applied, u_applied.shape)

    hover_force = quadcopter.mass * g
    force_vect = u_applied.flatten()
    #force_vect[0] += hover_force
    #print("Control inputs (rotor speeds):", force_vect)
    # 4) Apply MPC control input to the real inverted pendulum

    w_i = np.sqrt(np.clip(quadcopter.M_inv @ force_vect, 0, None))
    quadcopter.step(w_i, dt)

    y_meas = x.copy()
    state_history = np.vstack([state_history, y_meas.reshape(1, -1)])
    y_u_history = np.vstack([y_u_history, y_meas[:pu].reshape(1, -1)])  # Ensure correct dimensions
    u_history = np.vstack([u_history, np.array(u_applied)])

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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_history, y_history, z_history, linewidth=2)
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# ax.set_title('Quadrotor 3D Trajectory')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot full trajectory
ax.plot(x_history, y_history, z_history, linewidth=2, label='Trajectory')

# Red dot (initial position)
dot, = ax.plot([x_history[0]],
               [y_history[0]],
               [z_history[0]],
               'ro', markersize=6, label='Current position')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Quadrotor 3D Trajectory')
ax.legend()

# Set axis limits for better visualization
ax.set_xlim(min(x_history), max(x_history))
ax.set_ylim(min(y_history), max(y_history))
ax.set_zlim(min(z_history), max(z_history))


def update(frame):
    dot.set_data([x_history[frame]],
                 [y_history[frame]])
    dot.set_3d_properties([z_history[frame]])
    return dot,


# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(x_history),
    interval=1,   # ms between frames
    blit=True
)

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