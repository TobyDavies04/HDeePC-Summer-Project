import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Inv_Pendulum_Class import Inv_Pendulum
from DeePC_Controller import DeePC_Controller

#DeePC Controller Implementation
#Use data from offline MPC Controller 
#Then implement DeePC Controller using the data

inv_pendulum = Inv_Pendulum()
inv_pendulum.set_state([0, 0, 0, 0])    # small angle from upright
T = 0.01                 # sampling period
dt = T            # same as MPC sample time
sim_steps = 300   # run for _ m_seconds
t = np.arange(0, sim_steps * T, T)  # time vector
ref = np.array([0, 0]).reshape(2,1)  # upright position
u_min = -15.0    # min force
u_max = 15.0     # max force
du_min = -1
du_max = 1

# DeePC parameters
u_data = np.load("u_history_IP.npy")
x_data = np.load("x_history_IP.npy")

T_data = 800  # Length of data
u_d = np.array(u_data).reshape(-1, 1)   # (T,-1)
u_d = u_d[-T_data:]
y_d = np.array(x_data[-T_data:, [0, 2]])  # Extract only the first and third elements (T,2)
Q = np.diag([1000, 1000])
R = np.array([[1]])
T_ini = 1

# Get correct initial past window
u_ini = u_d[-T_ini:]     # shape (T_ini, 1)
y_ini = y_d[-T_ini:]     # shape (T_ini, 2)

# Histories for simulation start
x_history = y_ini.copy()   # these are outputs
u_history = u_ini.copy()


print("u_data shape:", u_data.shape)
print("x_data shape:", x_data.shape)
print("u_d shape:", u_d.shape)
print("y_d shape:", y_d.shape)
print("u_ini shape:", u_ini.shape)
print("y_ini shape:", y_ini.shape)

deepc = DeePC_Controller(u_d = u_d, y_d = y_d,
                        u_ini = u_ini, y_ini = y_ini,
                        T_ini = T_ini, N = 60,
                        Q = Q, R = R,
                        ref = ref,
                        lambda_g = 40, lambda_y = 100000.0,
                        u_min = u_min, u_max = u_max,
                        du_min = du_min, du_max = du_max)

# deepc.u_d = np.array(u_history)
# deepc.y_d = np.array(x_history)
opt_problem = deepc.opt_setup()

# x_history = np.zeros((0, 2))
# x_history = np.vstack([x_history, y_ini])
# u_history = np.zeros((0, 1))
# u_history = np.vstack([u_history, u_ini])
# print("Initial x_history shape:", x_history.shape)
# print("Initial u_history shape:", u_history.shape)

x_history = y_ini.copy()  # (T_ini, 2)
u_history = u_ini.copy()  # (T_ini, 1)
x_history = np.array(x_history)  # (T_ini, 2)
u_history = np.array(u_history)  # (T_ini, 1)

plot_x = []
plot_u = []

# x_history.append(inv_pendulum.x.flatten())
# u_history.append(0.0)

#Simulation loop
for k in range(sim_steps):
    if k == 100:
        ref = np.array([1, 0]).reshape(2,1)  # new reference position
    # Prepare initial trajectory
    u_ini_new = u_history[-deepc.T_ini:] #Get last T_ini inputs
    #print("u_ini_new shape:", u_ini_new.shape)
    y_ini_new = x_history[-deepc.T_ini:]
    #opt_problem = deepc.opt_setup()
    deepc.update(opt_problem, u_ini_new, y_ini_new, ref)
    u_deepc = deepc.solve_opt(opt_problem)
    #Apply DeePC control input to the real inverted pendulum
    x = inv_pendulum.step(float(u_deepc))
    #Log data
    plot_x.append(x)
    plot_u.append(float(u_deepc))

    # update histories with *current* measurement and input
    y_meas = np.array([x[0], x[2]])           # same outputs as in y_d
    #print("y_meas:", y_meas)
    x_history = np.vstack([x_history, y_meas.reshape(1, -1)])
    # print("u_history shape:", u_history.shape)
    # print("control input", u_deepc)
    u_history = np.vstack([u_history, [[u_deepc]]])
    print(f"Step {k}: Applied control u = {u_deepc:.5f}, State x = {x}")



# x_history = np.array(x_history)
# u_history = np.array(u_history)
plot_u = np.array(plot_u)
plot_x = np.array(plot_x)
print("plot_x shape:", plot_x.shape)
print("plot_u shape:", plot_u.shape)

plt.figure()
plt.plot(plot_u)
plt.xlabel("Time step (k)")         # ← X-axis name
plt.ylabel("Control input u (F)")   # ← Y-axis name
plt.title("Control Input")

plt.figure()
plt.plot(plot_x[:, 0])
plt.axhline(ref[0], color='red', linestyle='--', label="Reference x")
plt.xlabel("Time step (k)")
plt.ylabel("Cart Position x (m)")
plt.title("Cart Displacement vs Time")
plt.legend()

plt.figure()
plt.plot(plot_x[:, 1])
plt.axhline(ref[1], color='red', linestyle='--', label="Reference angle")
plt.xlabel("Time step (k)")
plt.ylabel("Pendulum Angle θ (rad)")
plt.title("Pendulum Angle vs Time")
plt.legend()

#plt.show()

# Extract states
x_cart = plot_x[:, 0]       # cart position
theta  = plot_x[:, 1]       # pendulum angle deviation (rad)

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