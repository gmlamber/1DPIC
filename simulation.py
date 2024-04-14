import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
L = 100  # Length of the domain
N = 1000  # Number of particles
dt = 0.1  # Time step
n_steps = 100  # Number of time steps

# Particle initialization
x = np.random.uniform(0, L, N)  # Initial positions
v1 = np.random.normal(1, 0.1, N//2)  # Velocity of first stream
v2 = np.random.normal(-1, 0.1, N//2)  # Velocity of second stream
v = np.concatenate((v1, v2))  # Combine velocities
charge = -1.0  # Charge of particles (electrons)

# Electric field and potential
E = np.zeros_like(x)  # Electric field
phi = np.zeros_like(x)  # Electrostatic potential

# Create figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, N/50)  # Adjust y-axis limit for better visualization
line, = ax.plot([], [], 'bo', ms=3)  # Particle visualization

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# Animation function: update the plot with particle positions at each time step
def animate(i):
    global x, v, E, phi

    # Particle push
    x += v * dt  # Update position

    # Periodic boundary conditions
    x = np.mod(x, L)

    # Charge deposition
    rho = np.bincount(x.astype(int), minlength=len(E)) / L  # Charge density

    # Electric field calculation
    E = np.gradient(phi)  # Electric field from potential

    # Particle acceleration
    v += charge * E * dt  # Acceleration from electric field

    # Potential update (Poisson equation)
    phi = np.fft.irfft(-rho)  # Fourier transform to get potential in real space

    # Update particle visualization
    line.set_data(x, np.zeros_like(x))
    return line,

# Create animation
ani = FuncAnimation(fig, animate, frames=n_steps, init_func=init, blit=True)

# Display animation
plt.xlabel('Position')
plt.ylabel('Density')
plt.title('Particle Density Evolution')
plt.show()