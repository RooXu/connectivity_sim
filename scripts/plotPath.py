from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

data = np.loadtxt("../output/agent_log_103.csv",delimiter=",",dtype=str)
print(data[1:,3:5])
print(type(data[1:,3:5]))
justCoords = data[1:,3:5].astype(np.float)

print(justCoords)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(data[1:,0].astype(np.float),justCoords[:,0],justCoords[:,1])
ax.set_xlabel("tick")
ax.set_ylabel("x")
ax.set_zlabel("y")
plt.show()