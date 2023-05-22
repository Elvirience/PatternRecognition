import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import dot


def get_class(X, Z1, Z2, eps=0.5):
    d1 = dot(X, Z1) - norm(Z1)**2 / 2
    d2 = dot(X, Z2) - norm(Z2)**2 / 2
    if d1 > d2+eps:
        return 0
    elif d2 > d1+eps:
        return 1
    else:
        print('ОНР')


def solution_line(x1, Z1, Z2):
    if Z1[0] == Z2[0]:
        x2 = (norm(Z1)**2 - norm(Z2)**2) / (2*(Z1[1] - Z2[1]))
    elif Z1[1] == Z2[1]:
        x2 = (norm(Z1)**2 - norm(Z2)**2) / (2*(Z1[0] - Z2[0]))
    else:
        x2 = (x1*(Z2[0]-Z1[0]) + (norm(Z1)**2 - norm(Z2)**2)/2) / (Z1[1] - Z2[1])
    return x2


def hendleEvent(event):
    X = (event.xdata, event.ydata)
    if get_class(X, Z1, Z2) == 0:
        col = 'r'
    elif get_class(X, Z1, Z2) == 1:
        col = 'y'
    else:
        col = 'b'
    ax.scatter(X[0], X[1], c=col, label='One Point')
    fig.canvas.draw()


w1 = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
w2 = np.array([(5, 0), (5, 1), (6, 0), (6, 1), (6, 2), (6, 3)])
l1 = len(w1)
l2 = len(w2)
Z1 = np.sum(w1, axis=0) / l1
Z2 = np.sum(w2, axis=0) / l2

x1 = np.arange(-10, 10)
x2 = list(solution_line(i, Z1, Z2) for i in x1)

data = w1
data = np.r_[data, w2]
labels = np.zeros(l1)
labels = np.r_[labels, np.ones(l2)]

fig, ax = plt.subplots(figsize=(5, 5))

ax.grid(True)
ax.set_xlim([-2, 8])
ax.set_ylim([-2, 8])
ax.scatter(data[:, 0], data[:, 1], c=labels, s=100,
           cmap='autumn', edgecolors='black', linewidth=1.5, alpha=0.5)
ax.scatter(x=Z1[0], y=Z1[1], c='r')
ax.scatter(x=Z2[0], y=Z2[1], c='y')

if Z1[0] == Z2[0]:
    ax.plot(x1, x2)
elif Z1[1] == Z2[1]:
    ax.plot(x2, x1)
else:
    ax.plot(x1, x2)

fig.canvas.mpl_connect('button_press_event', hendleEvent)

plt.show()
