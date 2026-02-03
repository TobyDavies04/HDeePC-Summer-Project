import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Inv_Pendulum_Class import Inv_Pendulum
from MPC_Controller import MPC_Controller
from DeePC_Controller import DeePC_Controller, OptimizationProblem, OptimizationProblemVariables

#DeePC Controller Implementation
#Use working MPC Controller to generate data in real time
#Then implement DeePC Controller using the data

inv_pendulum = Inv_Pendulum()
inv_pendulum.set_state([0, 0, 0.3, 0])    # small angle from upright

#MPC parameters
A = inv_pendulum.A
B = inv_pendulum.B
C = inv_pendulum.C
D = inv_pendulum.D
Q = np.diag([500, 1000])     # penalize (x - reference)^2
R = np.array([[0.1]])    # penalize u^2  (force/input)
T = 0.01                 # MPC sampling period

dt = T            # same as MPC sample time
sim_steps = 3000   # run for _ m_seconds
t = np.arange(0, sim_steps * T, T)  # time vector

ref = np.array([5, 0]).reshape(2,1)  # upright position
u_min = -15.0    # min force
u_max = 15.0     # max force
du_min = -01.50
du_max = 01.50

mpc = MPC_Controller(A, B, C, D, Q, R, T, ref, u_min, u_max, du_min, du_max, Np=60, Nc=30)
mpc.x = inv_pendulum.x.copy()

#DeePC parameters

x_history = []
u_history = []

x_history.append(inv_pendulum.x.flatten())
u_history.append(0.0)

#Simulation loop to generate data using MPC first 1500 steps
#Then use DeePC for next 1500 steps for similar reference tracking
for k in range(sim_steps):
    if k < 1500:
        mpc.x = inv_pendulum.x.copy()
        # update reference
        mpc.ref = ref
        # 2) Run one MPC step
        x_pred, y_pred, u_applied = mpc.step()
        # 3) ADD DISTURBANCE
        disturbance = 0.0
        #impulse
        if (k == 1250 or k == 1000):
            disturbance = 100.0
        if k == 1000:
            ref = np.array([2, 0]).reshape(2,1)  # new reference position
        total_force = float(u_applied) + disturbance
        # 4) Apply MPC control input to the real inverted pendulum
        x = inv_pendulum.step(total_force)
        # 5) Log data
        x_history.append(x)
        u_history.append(total_force)

    elif k == 1500:
        #Initialize DeePC data

        N = 1000  # Length of data
        u_d = np.array(u_history).reshape(-1, 1)   # (T,1)
        y_d = np.array(x_history).reshape(-1,  x_history[0].size)  # (T,P)

        deepc = DeePC_Controller(u_d = u_d, y_d = y_d,
                                u_ini = None, y_ini = None,
                                slack_y = None,
                                T_ini = 20, N = 60,
                                Q = Q, R = R,
                                ref = ref,
                                lambda_g = 1.0, lambda_y = 1000.0,
                                u_min = u_min, u_max = u_max,
                                du_min = du_min, du_max = du_max)

        # deepc.u_d = np.array(u_history)
        # deepc.y_d = np.array(x_history)
        opt_problem = deepc.opt_setup()
    
    elif k > 1500:
        #Run DeePC controller
        u_ini_window = np.array(u_history[k - deepc.T_ini:k]).reshape(-1, 1)   # (T_ini, 1)
        y_ini_window = np.array(x_history[k - deepc.T_ini:k])                  # (T_ini, 2)

        deepc.u_ini = u_ini_window.flatten(order='C').reshape(-1, 1)  # (T_ini*1,1)
        deepc.y_ini = y_ini_window.flatten(order='C').reshape(-1, 1)  # (T_ini*2,1)

        deepc.ref = ref
        x_deepc, y_deepc, u_deepc = deepc.step(opt_problem)
        #Apply DeePC control input to the real inverted pendulum
        x = inv_pendulum.step(float(u_deepc))
        #Log data
        x_history.append(x)
        u_history.append(float(u_deepc))



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
ax.set_xlim(-1.5, 7)         # adjust as needed
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