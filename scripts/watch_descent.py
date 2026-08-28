import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import json
import time

global stages
stages = int(sys.argv[1])
param_count = stages * 2 + stages * stages

v1 = 0
v2 = stages + 2
v3 = stages + 1

def read_params():
    data = ""
    with open(f"./cache/tableaux/checkpoints/s{stages}.json", "r") as f:
        data = f.read()

    j = json.loads(data)
    
    params = np.zeros((len(j), param_count))
    losses = np.zeros(len(j))
    ii = 0
    for t in j:
        _p = []
        for i in range(stages):
            _p.append(t["b"][i])
        for i in range(stages):
            _p.append(t["c"][i])
        for i in range(stages):
            for j in range(stages):
                _p.append(t["a"][i][j])

        params[ii] = np.array(_p)
        losses[ii] = t["loss"]
        
        ii += 1

    params = np.where(np.isnan(params), 0, params)
    losses = np.where(np.isnan(losses), 0, losses)

    return params, losses

def update_plot(frame):
    params, losses = read_params()
    p._offsets3d = (params[:, v1], params[:, v2], params[:, v3])
    p.set_array(losses)
    # p.set_clim(losses.min(), losses.max())
    # cbar.update_normal(p)

    return (p,)

params, losses = read_params()
fig = plt.figure() 
ax = fig.add_subplot(projection="3d")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
norm = colors.LogNorm(vmin=1e-6, vmax=1e3)
p = ax.scatter(params[:, v1], params[:, v2], params[:, v3], c=losses, norm=norm)
cbar = fig.colorbar(p, ax=ax)

ani = FuncAnimation(fig, update_plot, interval=1000)

plt.show()
