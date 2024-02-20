import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import queue

# Create a queue to hold the data
data_queue = queue.Queue()

def data_generator(q):
    """Simulate real-time data updates"""
    for _ in range(100):
        x = np.random.rand(10)
        y = np.random.rand(10)
        z = np.random.rand(10)
        q.put((x, y, z))

def plot_3d_realtime(q):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(num):
        if not q.empty():
            x, y, z = q.get()
            ax.clear()
            ax.scatter(x, y, z)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

    ani = FuncAnimation(fig, update, frames=100, interval=1000)
    plt.show()

# Simulate real-time data updates
data_generator(data_queue)

# Plot in real-time
plot_3d_realtime(data_queue)
