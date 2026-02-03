# simulate.py

import numpy as np
import matplotlib.pyplot as plt
from DC_Motor_Class import DCMotor

# Time
t = np.linspace(0, 10, 1001)
dt = t[1] - t[0]

# Input
V = 3.0 * np.ones_like(t)

# Create motor object
motor = DCMotor()
motor.set_state([0,0])

# Simulate
omega_hist = np.zeros_like(t)
for k in range(len(t)):
    omega_hist[k] = motor.step(V[k], dt)

# Plot
plt.plot(t, omega_hist)
plt.xlabel("Time (s)")
plt.ylabel("Omega (rad/s)")
plt.title("DC Motor Simulation")
plt.show()
