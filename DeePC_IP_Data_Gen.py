import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Inv_Pendulum_Class import Inv_Pendulum
from MPC_Controller import MPC_Controller

inv_pendulum = Inv_Pendulum()
inv_pendulum.set_state([0, 0, 0.0, 0])   # start near upright (linear region)

#MPC parameters
A = inv_pendulum.A
B = inv_pendulum.B
C = inv_pendulum.C
D = inv_pendulum.D
Q = np.diag([1000, 1000])     # penalize (x - reference)^2
R = np.array([[1]])    # penalize u^2  (force/input)
T = 0.01                 # MPC sampling period

dt = T            # same as MPC sample time
sim_steps = 800   # run for _ m_seconds
t = np.arange(0, sim_steps * T, T)  # time vector

ref = np.array([0, 0]).reshape(2,1)  # upright position
u_min = -15.0    # min force
u_max = 15.0     # max force
du_min = -1
du_max = 1

mpc = MPC_Controller(A, B, C, D, Q, R, T, ref, u_min, u_max, du_min, du_max, Np=60, Nc=30)
mpc.x = inv_pendulum.x.copy()

PRBS_interval = 1        # Change PRBS every 20 steps
noise_amp = 5.0           # Small random noise (keep small)
angle_limit = 0.8         # Max safe angle rad (~17 degrees)

x_history = []
u_history = []

x_history.append(inv_pendulum.x.flatten())
u_history.append(0.0)

for k in range(sim_steps):

    disturbance = 0.0
    #impulse
    if k == 100:
        ref = np.array([1, 0]).reshape(2,1)  # new reference position
    if (k == 200):
        ref = np.array([2, 0]).reshape(2,1)  # new reference position
    #     disturbance = 30.0
    if (k == 300):
        ref = np.array([3, 0]).reshape(2,1)
        # disturbance = -30.0
    if (k == 400):
        ref = np.array([4, 0]).reshape(2,1)
    if (k == 500):
        ref = np.array([1, 0]).reshape(2,1)



    # === 1. Compute stabilising control from MPC ===
    mpc.x = inv_pendulum.x.copy()
    mpc.ref = ref   # (2x1)
    x_pred, y_pred, u_applied = mpc.step()       # MPC output (feedback)

    # === 2. Add PRBS excitation ===
    if k % PRBS_interval == 0:
        prbs = np.random.choice([-1, 1]) * 5.0   # choose amplitude (3 N)
    else:
        prbs = 0.0

    # === 3. Add small white noise excitation ===
    white_noise = noise_amp * (2*np.random.rand() - 1)

    # === 4. Combine everything ===
    u_fb = float(u_applied)
    u_total = u_fb + prbs + white_noise + disturbance

    # Clip for safety
    u_total = np.clip(u_total, -50, 50)
    # if k > 600:
    #     u_total = u_fb

    # === 5. Apply to system ===
    x = inv_pendulum.step(u_total)

    # === 6. Safety reset to stay in linear zone ===
    # if abs(x[2]) > angle_limit:
    #     print(f"RESET at step {k} (angle too large: {x[2]:.3f})")
    #     inv_pendulum.set_state([0,0,0,0])
    #     continue

    # === 7. Log data for DeePC ===
    u_history.append(u_total)
    x_history.append(x)

    print(f"k={k:3d}: u_fb={u_fb:.2f}, PRBS={prbs:.2f}, noise={white_noise:.2f}, u_total={u_total:.2f}, angle={x[2]:.3f}")

# Convert to arrays
u_history = np.array(u_history)
x_history = np.array(x_history)

np.save("u_history_IP.npy", np.array(u_history))
np.save("x_history_IP.npy", np.array(x_history))

plt.figure()
plt.plot(u_history)
plt.xlabel("Time step (k)")         # ← X-axis name
plt.ylabel("Control input u (F)")   # ← Y-axis name
plt.title("Control Input")

plt.figure()
plt.plot(x_history[:, 0])
#plt.axhline(ref[0], color='red', linestyle='--', label="Reference x")
plt.xlabel("Time step (k)")
plt.ylabel("Cart Position x (m)")
plt.title("Cart Displacement vs Time")
plt.legend()

plt.figure()
plt.plot(x_history[:, 2])
#plt.axhline(ref[1], color='red', linestyle='--', label="Reference angle")
plt.xlabel("Time step (k)")
plt.ylabel("Pendulum Angle θ (rad)")
plt.title("Pendulum Angle vs Time")
plt.legend()

#plt.show()

# Extract states
x_cart = x_history[:, 0]       # cart position
theta  = x_history[:, 2]       # pendulum angle deviation (rad)

dt = T                         # your sampling time
L = 1.2             # pendulum length

# Set up the figure
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 6)         # adjust as needed
ax.set_ylim(-0.5, 5)

# Cart (a rectangle) and pendulum (a line)
cart_width = 1
cart_height = 0.5

cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='k')
ax.add_patch(cart_patch)

(line,) = ax.plot([], [], lw=5)   # pendulum rod
(mass_point,) = ax.plot([], [], 'o', ms=10)  # pendulum bob

# Ground line
ax.axhline(0, color='gray', linewidth=1)

def init():
    cart_patch.set_xy((-cart_width/2, 0))  # initial cart position at 0
    line.set_data([], [])
    mass_point.set_data([], [])
    return cart_patch, line, mass_point

def update(frame):
    xc = x_cart[frame]      # cart position
    th = theta[frame]       # pendulum angle

    # Cart position
    cart_patch.set_xy((xc - cart_width/2, 0))

    # Pendulum end coordinates (pivot at top of cart)
    pivot_x = xc
    pivot_y = cart_height

    pend_x = pivot_x + L * np.sin(th)
    pend_y = pivot_y + L * np.cos(th)

    line.set_data([pivot_x, pend_x], [pivot_y, pend_y])
    mass_point.set_data([pend_x], [pend_y])

    return cart_patch, line, mass_point

ani = FuncAnimation(
    fig,
    update,
    frames=len(np.arange(0, sim_steps * T, T)),
    init_func=init,
    blit=True,
    interval=dt*1000  # in ms
)

plt.show()