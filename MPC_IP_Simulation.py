import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Inv_Pendulum_Class import Inv_Pendulum
from MPC_Controller import MPC_Controller

inv_pendulum = Inv_Pendulum()
inv_pendulum.set_state([1, 0, 0, 0])    # small angle from upright

#MPC parameters
A = inv_pendulum.A
B = inv_pendulum.B
C = inv_pendulum.C
D = inv_pendulum.D
Q = np.diag([1000, 1000])     # penalize (x - reference)^2
R = np.array([[1]])    # penalize u^2  (force/input)
T = 0.01                 # MPC sampling period

dt = T            # same as MPC sample time
sim_steps = 200   # run for _ m_seconds
t = np.arange(0, sim_steps * T, T)  # time vector

ref = np.array([1, 0]).reshape(2,1)  # upright position
u_min = np.array([-15.0])    # min force
u_max = np.array([15.0])     # max force
du_min = -1
du_max = 1
# du_min = None
# du_max  = None

mpc = MPC_Controller(A, B, C, D, Q, R, T, ref, u_min, u_max, du_min, du_max, Np=60, Nc=60, discretize=True)
mpc.x = inv_pendulum.x.copy()

x_history = []
u_history = []

x_history.append(inv_pendulum.x.flatten())
u_history.append(0.0)

for k in range(sim_steps):
    mpc.x = inv_pendulum.x.copy()
    # update reference
    mpc.ref = ref
    # 2) Run one MPC step
    x_pred, y_pred, u_applied = mpc.step()
    # 3) ADD DISTURBANCE
    disturbance = 0.0
    #impulse
    if k == 15:
        ref = np.array([3, 0]).reshape(2,1)  # new reference position
    if (k == 100):
        ref = np.array([0, 0]).reshape(2,1)  # new reference position
        disturbance = 20.0
    if (k == 200):
        ref = np.array([2, 0]).reshape(2,1)
        disturbance = -20.0
    if (k == 250):
        ref = np.array([1, 0]).reshape(2,1)
        disturbance = -20.0
    if (k == 250):
        ref = np.array([0, 0]).reshape(2,1)
        disturbance = -20.0
    total_force = float(u_applied) + disturbance
    # 4) Apply MPC control input to the real inverted pendulum
    x = inv_pendulum.step(total_force)
    print(f"Step {k}: Applied Force = {total_force:.3f}, Cart Pos = {x[0]:.3f}, Pendulum Angle = {x[2]:.3f}")
    # 5) Log data
    x_history.append(x)
    u_history.append(total_force)

#Saves data for DeePC use, uncomment to save
# np.save("u_history_IP_HDeePC.npy", np.array(u_history))
# np.save("x_history_IP_HDeePC.npy", np.array(x_history))

x_history = np.array(x_history)
u_history = np.array(u_history)

plt.figure()
plt.plot(u_history)
plt.xlabel("Time step (k)")         # ← X-axis name
plt.ylabel("Control input u (F)")   # ← Y-axis name
plt.title("Control Input")

plt.figure()
plt.plot(x_history[:, 0])
plt.axhline(ref[0], color='red', linestyle='--', label="Reference x")
plt.xlabel("Time step (k)")
plt.ylabel("Cart Position x (m)")
plt.title("Cart Displacement vs Time")
plt.legend()

plt.figure()
plt.plot(x_history[:, 2])
plt.axhline(ref[1], color='red', linestyle='--', label="Reference angle")
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
    frames=len(t),
    init_func=init,
    blit=True,
    interval=dt*1000  # in ms
)

plt.show()