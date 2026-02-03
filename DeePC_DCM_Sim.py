import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from DeePC_Controller import DeePC_Controller, OptimizationProblem, OptimizationProblemVariables
from DC_Motor_Class import DCMotor_State_Space

motor = DCMotor_State_Space()
motor.set_state([2, 1])    # zero current, zero speed

Q = np.array([[1000]])     # penalize (omega - reference)^2
R = np.array([[0.1]])    # penalize u^2  (torque/input)
T = 0.01                 # MPC sampling period

dt = T            # same as MPC sample time
sim_steps = 1000   # run for 3 seconds
t = np.arange(0, sim_steps * T, T)  # time vector
ref = 1 +  0.5 * np.sin(2 * np.pi * 0.5 * t)  # sine wave

u_min = -50.0    # min voltage
u_max = 50.0     # max voltage
du_min = -50.0
du_max = 50.0

# DeePC parameters
u_data = np.load("u_history_DCM.npy")
x_data = np.load("omega_history_DCM.npy")

T_data = 1000  # Length of data
u_d = np.array(u_data).reshape(-1, 1)   # (T,-1)
u_d = u_d[:T_data]
y_d = np.array(x_data).reshape(-1, 1)  # (T,1)
y_d = y_d[:T_data] # Extract only the first and third elements (T,2)
Q = np.diag([1000])
R = np.array([[0.1]])
T_ini = 20
u_ini = u_d[-T_ini:]
y_ini = y_d[-T_ini:]

deepc = DeePC_Controller(u_d = u_d, y_d = y_d,
                        u_ini = u_ini, y_ini = y_ini,
                        T_ini = T_ini, N = 15,
                        Q = Q, R = R,
                        ref = ref,
                        lambda_g = 30.0, lambda_y = 100000.0,
                        u_min = u_min, u_max = u_max,
                        du_min = du_min, du_max = du_max)

opt_problem = deepc.opt_setup()
x_history = y_ini.copy()  # (T_ini, 2)
u_history = u_ini.copy()  # (T_ini, 1)
x_history = np.array(x_history)  # (T_ini, 2)
u_history = np.array(u_history)  # (T_ini, 1)

plot_x = []
plot_u = []

for k in range(sim_steps):

    # reference at this time step
    if k < 500:
        ref_k = np.array([[ref[k]]])   # shape (1,1)
    else:
        ref_k = np.array([[1.5]])

    # prepare past window
    u_ini_new = u_history[-deepc.T_ini:]
    y_ini_new = x_history[-deepc.T_ini:]

    deepc.update(opt_problem, u_ini_new, y_ini_new, ref_k)

    # solve DeePC
    u_deepc = deepc.solve_opt(opt_problem)

    # apply input to motor
    x = motor.step(float(u_deepc), dt)

    # log
    plot_x.append(x)
    plot_u.append(float(u_deepc))
    print(f"Step {k}: u={u_deepc}, x={x}")

    # update histories
    x_history = np.vstack([x_history, [[x]]])
    u_history = np.vstack([u_history, [[u_deepc]]])



plt.figure()
plt.plot(plot_x)
plt.axhline(ref[0], color='red', linestyle='--', label="Reference")
plt.xlabel("Time step (k)")          # ← X-axis name
plt.ylabel("Motor speed ω (rad/s)")  # ← Y-axis name
plt.title("Motor Speed vs Time")
plt.legend()

plt.figure()
plt.plot(plot_u)
plt.xlabel("Time step (k)")         # ← X-axis name
plt.ylabel("Control input u (V)")   # ← Y-axis name
plt.title("Control Input")

plt.show()
