# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
x = np.linspace(-2*np.pi, 2*np.pi,1000)
y = np.sin(x)   
y2 = np.cos(x)
y3 = y**100+y2**100

plt.plot(x,y,label='x1')
plt.plot(x,y2,label='x2')
plt.plot(x,y3,label='x3')
plt.grid()
plt.xlabel('x',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.legend()
plt.show() 

# %%



