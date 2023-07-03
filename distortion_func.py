import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri
from scipy.stats import norm

#Define min max function and how to derive the bids from such function
def minmaxvar(l, u):
    result = 1 - (1 - u**(1/(l+1)))**(1+l)
    return result

#Create graph
x = np.arange(0, 1, 0.0001)
y0 = minmaxvar(0, x)
y0_1 = minmaxvar(0.1, x)
y0_25 = minmaxvar(0.25, x)
y0_5 = minmaxvar(0.5, x)
y1 = minmaxvar(1, x)
y1_5 = minmaxvar(1.5, x)
#Create figure and plot
plt.figure(figsize=(7, 5))
plt.plot(x, y0, label = '$\gamma$ = 0')
plt.plot(x, y0_1, label = '$\gamma$ = 0.1')
plt.plot(x, y0_25, label = '$\gamma$ = 0.25')
plt.plot(x, y0_5, label = '$\gamma$ = 0.5')
plt.plot(x, y1, label = '$\gamma$ = 1')
plt.plot(x, y1_5, label = '$\gamma$ = 1.5')
name_fig = 'Minmaxvar_function.eps'
plt.legend(loc = 4)
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.show()
plt.close()

def Wang(l, u):
    result = norm.cdf(ndtri(u)+l)
    return result

def comp_wang_minmaxvar(l, u):
    return Wang(l, minmaxvar(l, u))

#Function Defined as the composition of minmavar and minmaxvar distortion functions
def comp_minmaxvar_wang(l, u):
    return minmaxvar(l, Wang(l, u))


#Create graph
x = np.arange(0, 1, 0.0001)
y0 = comp_wang_minmaxvar(0, x)
y0_1 = comp_wang_minmaxvar(0.1, x)
y0_25 = comp_wang_minmaxvar(0.25, x)
y0_5 = comp_wang_minmaxvar(0.5, x)
y1 = comp_wang_minmaxvar(1, x)
print(y1)
y1_5 = comp_wang_minmaxvar(1.5, x)
#Create figure and plot
plt.figure(figsize=(7, 5))
plt.plot(x, y0, label = '$\gamma$ = 0')
plt.plot(x, y0_1, label = '$\gamma$ = 0.1')
plt.plot(x, y0_25, label = '$\gamma$ = 0.25')
plt.plot(x, y0_5, label = '$\gamma$ = 0.5')
plt.plot(x, y1, label = '$\gamma$ = 1')
plt.plot(x, y1_5, label = '$\gamma$ = 1.5')
name_fig = 'comp_wang_minmaxvar_function.eps'
plt.legend(loc = 4)
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.show()
plt.close()



#Create graph
x = np.arange(0, 1, 0.0001)
y0 = comp_minmaxvar_wang(0, x)
y0_1 = comp_minmaxvar_wang(0.1, x)
y0_25 = comp_minmaxvar_wang(0.25, x)
y0_5 = comp_minmaxvar_wang(0.5, x)
y1 = comp_minmaxvar_wang(1, x)
print(y1)
y1_5 = comp_minmaxvar_wang(1.5, x)
#Create figure and plot
plt.figure(figsize=(7, 5))
plt.plot(x, y0, label = '$\gamma$ = 0')
plt.plot(x, y0_1, label = '$\gamma$ = 0.1')
plt.plot(x, y0_25, label = '$\gamma$ = 0.25')
plt.plot(x, y0_5, label = '$\gamma$ = 0.5')
plt.plot(x, y1, label = '$\gamma$ = 1')
plt.plot(x, y1_5, label = '$\gamma$ = 1.5')
name_fig = 'comp_minmaxvar_wang_function.eps'
plt.legend(loc = 4)
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.show()
plt.close()