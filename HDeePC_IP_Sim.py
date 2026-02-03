import numpy as np
import cvxpy as cp
import time
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Inv_Pendulum_Class import Inv_Pendulum
from HDeePC_Controller import HDeePC_Controller

### HDeePC for Inverted Pendulum
## Only the second two states are known, theta, theta_dot
inv_pendulum = Inv_Pendulum()
inv_pendulum.set_state([0, 0, 0, 0])    # small angle from upright

T = 0.01                 # sampling period
dt = T            # same as MPC sample time
sim_steps = 550   # run for _ m_seconds
t = np.arange(0, sim_steps * T, T)  # time vector
ref = np.array([0.5, 0, 0, 0]).reshape(4,1)  # upright position
u_min = -15    # min force
u_max = 15     # max force
du_min = -1
du_max = 1

#MPC parameters
A = inv_pendulum.Ad
B = inv_pendulum.Bd
C = np.eye(4)
D = np.zeros((4,1)).reshape(4,1)
# C = inv_pendulum.C
# D = inv_pendulum.D

# DeePC parameters
u_data = np.load("u_history_IP.npy")
x_data = np.load("x_history_IP.npy")
T_data = 200  # Length of data
u_d = np.array(u_data).reshape(-1, 1)   # (T,-1)
u_d = u_d[-T_data:]
yu_d = np.array(x_data[-T_data:, :2])  # rows of 4 element states (:2 for first two outputs)
Q = [1000, 0, 1000, 0]
R = [1]
T_ini = 6
x0 = inv_pendulum.x.copy()                    # [1, 0, 0, 0]
y0 = x0.copy()             # outputs: [position, angle]

#HDeePC specific system matrices
nu = 0  # number of unknown states
nk = 4  # number of known states
pu = 0   # number of unknown outputs
pk = 4   # number of known outputs
NP = [nu, nk, pu, pk]

Ac = A[nu:, :nu]
Ak = A[nu:, nu:]
Bk = B[nu:, :]
Cc = C[pu:, :pu]
Ck = C[pu:, pu:]
Dk = D[pu:, :]

Cy = np.zeros((pk, pu))
Ay = Ac.copy()

# Past outputs: assume we sat at x0 for T_ini steps with zero input
y_uini = np.tile(y0[:pu].reshape(1, pu), (T_ini, 1))
u_ini = np.zeros((T_ini, 1))                          # shape (T_ini, 1)
print("y_uini shape:", y_uini.shape)

N = 60

# Histories for simulation start
x_history = np.empty((0, 4))
print("x_history shape:", x_history.shape)
y_u_history = y_uini.copy()
u_history = u_ini.copy()

lambda_g = 3e1
lambda_y = 1e5
lambda_u = 1e5

hdeepc = HDeePC_Controller(Ac, Ak, Ay, Bk, Cc, Ck, Cy, Dk, u_d,
                            yu_d, u_ini, y_uini, N, Q, R, ref, u_min, u_max, 
                            du_min, du_max, lambda_g, lambda_y, lambda_u, NP, calculate_Ay_Cy=False)

plot_x = []
plot_u = []
solve_times = []
tracking_costs = []
input_costs = []
reg_costs = []

sim_start = time.time()
###Simulation with HDeePC
for k in range(sim_steps):
    if k == 250:
        #ref = np.array([1, 0, 0, 0]).reshape(4,1)  # new reference position
        x = inv_pendulum.step(30.0)  # apply disturbance
    u_ini_new = u_history[-T_ini:] #Get last T_ini inputs
    y_ini_new = y_u_history[-T_ini:, :pu]  ### Getting an error here as our y_ini_new is from x_history, but we actually only need 2 term
    x = inv_pendulum.x.copy()
    xk = x[nu:]    
    hdeepc.update(u_ini_new, y_ini_new, xk, ref)
    t0 = time.time()
    u_deepc = hdeepc.opt_problem()
    solve_times.append(time.time() - t0)
    x = inv_pendulum.step(float(u_deepc[0]))

    #Log data
    plot_x.append(x)
    plot_u.append(float(u_deepc[0]))
    y_meas = x.copy()
    x_history = np.vstack([x_history, y_meas.reshape(1, -1)])
    y_u_history = np.vstack([y_u_history, y_meas[:pu].reshape(1, -1)])  # Ensure correct dimensions
    u_history = np.vstack([u_history, np.array([[u_deepc[0]]])])
    print(f"Step {k}: Applied control u = {u_deepc[0]:.5f}, State x = {x}")
    tracking_costs.append(hdeepc.J_track)
    input_costs.append(hdeepc.J_input)
    reg_costs.append(hdeepc.J_reg)

total_sim_time = time.time() - sim_start
print(f"Total simulation time: {total_sim_time:.2f} s")
print(f"Average solve time: {np.mean(solve_times)*1000:.2f} ms")
print(f"Average tracking cost: {np.mean(tracking_costs):.2f}")
print(f"Average input cost: {np.mean(input_costs):.2f}")
print(f"Average regularization cost: {np.mean(reg_costs):.2f}")

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
