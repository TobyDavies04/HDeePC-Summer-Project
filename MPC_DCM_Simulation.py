import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm
import matplotlib.pyplot as plt
from DC_Motor_Class import DCMotor_State_Space
from MPC_Controller import MPC_Controller

motor = DCMotor_State_Space()
motor.set_state([2, 1])    # zero current, zero speed

# MPC parameters
A = motor.A
B = motor.B
C = motor.C
D = motor.D
Q = np.array([[1000]])     # penalize (omega - reference)^2
R = np.array([[0.1]])    # penalize u^2  (torque/input)
T = 0.01                 # MPC sampling period

dt = T            # same as MPC sample time
sim_steps = 1000   # run for 3 seconds
t = np.arange(0, sim_steps * T, T)  # time vector
ref = 1 +  0.5 * np.sin(2 * np.pi * 0.5 * t)  # sine wave

u_min = -24.0    # min voltage
u_max = 24.0     # max voltage
du_min = -24.0
du_max = 24.0

mpc = MPC_Controller(A, B, C, D, Q, R, T, ref, u_min, u_max, du_min, du_max, Np=15, Nc=5)
mpc.x = motor.x.copy()


omega_history = []
u_history = []

for k in range(sim_steps):

    mpc.x = motor.x.copy()
    # update reference
    #mpc.ref = np.array([ref[k]]).reshape(-1,1)
    if k < 500:
        mpc.ref = np.array([[ref[k]]])   # shape (1,1)
    else:
        mpc.ref = np.array([[1.5]])


    # 2) Run one MPC step
    x_pred, y_pred, u_applied = mpc.step()

    # 3) Apply the MPC's control input to the REAL motor
    omega = motor.step(float(u_applied), dt)

    # 4) Log data
    omega_history.append(omega)
    u_history.append(float(u_applied))

# np.save("u_history_DCM.npy", np.array(u_history))
# np.save("omega_history_DCM.npy", np.array(omega_history))

plt.figure()
plt.plot(omega_history)
plt.axhline(ref[0], color='red', linestyle='--', label="Reference")
plt.xlabel("Time step (k)")          # ← X-axis name
plt.ylabel("Motor speed ω (rad/s)")  # ← Y-axis name
plt.title("Motor Speed vs Time")
plt.legend()

plt.figure()
plt.plot(u_history)
plt.xlabel("Time step (k)")         # ← X-axis name
plt.ylabel("Control input u (V)")   # ← Y-axis name
plt.title("Control Input")


plt.show()