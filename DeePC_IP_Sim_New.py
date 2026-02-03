print("Entering the Shallows of the DeePC")
import numpy as np
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Inv_Pendulum_Class import Inv_Pendulum
from DeePC_Controller_New import DeePC_Controller

#DeePC Controller Implementation
#Use data from offline MPC Controller 
#Then implement DeePC Controller using the data

inv_pendulum = Inv_Pendulum()
inv_pendulum.set_state([1, 0, 0, 0])    # small angle from upright

T = 0.01                 # sampling period
dt = T            # same as MPC sample time
sim_steps = 600   # run for _ m_seconds
t = np.arange(0, sim_steps * T, T)  # time vector
ref = np.array([1, 0, 0, 0]).reshape(4,1)  # upright position
u_min = -15.0    # min force
u_max = 15.0     # max force
du_min = -1
du_max = 1

# DeePC parameters
u_data = np.load("u_history_IP.npy")
x_data = np.load("x_history_IP.npy")

T_data = 400  # Length of data
u_d = np.array(u_data).reshape(-1, 1)   # (T,-1)
u_d = u_d[-T_data:]
y_d = np.array(x_data[-T_data:, :])  # Extract only the first and third elements (T,2)
Q = np.diag([1000, 0, 1000, 0])
R = np.array([[1]])
T_ini = 60

# Initial past window should reflect the REAL starting state, not offline data
x0 = inv_pendulum.x.copy()                    # [1, 0, 0, 0]
#y0 = np.array([x0[0], x0[2]])    
y0 = x0.copy()             # outputs: [position, angle]

# Past outputs: assume we sat at x0 for T_ini steps with zero input
y_ini = np.tile(y0.reshape(1, -1), (T_ini, 1))    # shape (T_ini, 2)
u_ini = np.zeros((T_ini, 1))                      # shape (T_ini, 1)

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
                        lambda_g = 30, lambda_y = 1e5, lambda_u = 1e2,
                        u_min = u_min, u_max = u_max,
                        du_min = du_min, du_max = du_max)

plot_x = []
plot_u = []

# x_history.append(inv_pendulum.x.flatten())
# u_history.append(0.0)

#Simulation loop
for k in range(sim_steps):
    if k == 50:
        ref = np.array([3, 0, 0, 0]).reshape(4,1)  # new reference position
    if k == 150:
        ref = np.array([5, 0, 0, 0]).reshape(4,1)  # new reference position
    if k == 350:
        ref = np.array([2, 0, 0, 0]).reshape(4,1)  # new reference position
    # Prepare initial trajectory
    u_ini_new = u_history[-deepc.T_ini:] #Get last T_ini inputs
    #print("u_ini_new shape:", u_ini_new.shape)
    y_ini_new = x_history[-deepc.T_ini:]
    #opt_problem = deepc.opt_setup()
    deepc.update(u_ini_new, y_ini_new, ref)
    u_deepc = deepc.step_deepc()
    #Apply DeePC control input to the real inverted pendulum
    x = inv_pendulum.step(float(u_deepc))
    #Log data
    plot_x.append(x)
    plot_u.append(float(u_deepc))

    # update histories with *current* measurement and input
    #y_meas = np.array([x[0], x[2]])           # same outputs as in y_d
    y_meas = x.copy()
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
plt.plot(plot_x[:, 2])
plt.axhline(ref[2], color='red', linestyle='--', label="Reference angle")
plt.xlabel("Time step (k)")
plt.ylabel("Pendulum Angle θ (rad)")
plt.title("Pendulum Angle vs Time")
plt.legend()

#plt.show()

# Extract states
x_cart = plot_x[:, 0]       # cart position
theta  = plot_x[:, 2]       # pendulum angle deviation (rad)




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