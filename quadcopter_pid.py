import numpy as np
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from quadcopter_plant import Quadcopter

### PID Controller for Quadcopter as a baseline comparison

### Quadcopter Plant
quadcopter = Quadcopter(0.468, 0.225, 2.98e-6, 1.14e-7, 3.357e-5, [4.856e-3, 4.856e-3, 8.801e-3], [0.25, 0.25, 0.25]) 
                        #mass, arm_length, lift_constant, drag_constant, rotor_inertia_moment, inertia,  A_drag
                        #Note: Hover is around 625 rad/s
dt = 0.01
sim_steps = 2000
time = np.arange(sim_steps) * dt
x_history = []
y_history = []
z_history = []
phi_history = []
theta_history = []
psi_history = []
w_i_history = np.array([]).reshape(0,4)

g = 9.80665  # gravitational acceleration (m/s^2)

# Outer-loop gains (position)
Kp_xy = 0.5
Kd_xy = 1.0

Kp_z  = 1.0
Kd_z  = 1.5


# Inner-loop gains (attitude + thrust)
Kp_att = np.array([10.0, 10.0, 5.0])
Kd_att = np.array([3.0,  3.0,  1.0])

Kp_T = 8.0
Kd_T = 5.0

ref = np.array([0, 0, 0, 0, 0, 0])  # desired position and orientation, [x, y, z, phi, theta, psi]
radius = 1          # radius (m)
omega = 2.5    # rad per timestep
#z_ref = 0.0005

for k in range(sim_steps):
    t = k * dt
    if k < 40000:
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
    elif k >= 40000:
        x_d = 0
        y_d = 0
        z_d = 0

        xd_d = 0.0
        yd_d = 0.0
        zd_d = 0.0

        xdd_d = 0.0
        ydd_d = 0.0
        zdd_d = 0.0

    x, y, z     = quadcopter.state[0:3]
    phi, theta, psi = quadcopter.state[3:6]
    vx, vy, vz  = quadcopter.state[6:9]
    p, q, r     = quadcopter.state[9:12]

    ex, ey, ez = x_d - x, y_d - y, z_d - z
    evx, evy, evz = xd_d - vx, yd_d - vy, zd_d - vz

    ax_cmd = Kp_xy * ex + Kd_xy * evx + xdd_d
    ay_cmd = Kp_xy * ey + Kd_xy * evy + ydd_d
    az_cmd = Kp_z  * ez + Kd_z  * evz + zdd_d

    theta_ref =  ax_cmd / g
    phi_ref   = -ay_cmd / g
    psi_ref   = 0.0

    T_ref = quadcopter.mass * (g + az_cmd)

    # =========================
    # INNER LOOP (Attitude + Thrust PID)
    # =========================
    # Attitude errors
    e_att = np.array([
        phi_ref   - phi,
        theta_ref - theta,
        psi_ref   - psi
    ])

    e_rate = np.array([-p, -q, -r])

    tau = Kp_att * e_att + Kd_att * e_rate

    # Thrust tracking (Z dynamics!)
    T = (
        T_ref
        + Kp_T * (z_d - z)
        + Kd_T * (zd_d - vz)
    )

    # =========================
    # Control vector → motors
    # =========================
    u = np.array([T, tau[0], tau[1], tau[2]])

    w_i = np.sqrt(
        np.clip(quadcopter.M_inv @ u, 0, None)
    )

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