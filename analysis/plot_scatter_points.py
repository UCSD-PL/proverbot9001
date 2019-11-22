import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

from math import sqrt
SPINE_COLOR = 'gray'
def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax
def rand_jitter(arr):
    stdev = .001*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    ax = plt.plot(x_vals, y_vals, '--')

report_steps = np.array([int(line.split(',')[-2]) for line in open('search-report/proofs.csv') if "SUCCESS" in line])
lin_steps = np.array([int(line.split(',')[-1]) for line in open('search-report/proofs.csv') if "SUCCESS" in line])

x_axis = []
y_axis = []
below = 0
on = 0
above = 0
for i in range(len(report_steps)):
    if report_steps[i] == 0:
        pass
    else:
        x_axis.append(report_steps[i])
        if lin_steps[i] == 0:
            y_axis.append(1)
        else:
            y_axis.append(lin_steps[i])

x_axis = rand_jitter(x_axis)
y_axis = rand_jitter(y_axis)

for i in range(len(x_axis)):
    if x_axis[i] > y_axis[i]:
        below += 1
    elif x_axis[i] == y_axis[i]:
        on += 1
    else:
        above += 1
print(x_axis)
print(y_axis)
print(len(x_axis))  
print(below)  
print(on)  
print(above)  
# latexify()
plt.xlim(0, 1000)
plt.ylim(0, 1000)
# plt.set_yscale('log')
plt.xscale('symlog')
plt.yscale('symlog')
plt.xlabel('Length of proofs found by Proverbot9001')
plt.ylabel('Length of linearized proofs')
plt.scatter(x_axis, y_axis, alpha=0.25)
abline(1, (0,0))
f = plt.figure()
#plt.show()

f.savefig("lengths.pdf", bbox_inches='tight')
