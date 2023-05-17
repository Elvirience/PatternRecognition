import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import dot
from random import randint
from math import factorial


def sub_points(point1, point2):
    return point1[0] - point2[0], point1[1] - point2[1]


def d(X, Z):
    return dot(X, Z) - norm(Z)**2 / 2


def get_class(X, Z, cls=0, eps=0.4, count=0):
    d0 = d(X, Z[0])
    for i in range(1, len(Z)):
        d_next = d(X, Z[i])
        if d_next > d0:
            d0 = d_next
            cls = i
    for j in range(0, len(Z)):
        if abs(d0 - d(X, Z[j])) <= eps:
            count += 1
    if count > 1:
        return 'ОНР'
    else:
        return cls


def classification(Z, M):
    cls_list = []
    for i in range(len(Z)):
        that_class = []
        for j in M:
            if get_class(j, Z) == i:
                that_class.append(j)
        if len(that_class) == 0:
            that_class.append(Z[i])
        that_class = np.array(that_class)
        cls_list.append(that_class)
    return cls_list


def get_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: np.random.randint(0, 255), range(3)))
    return color


def hendleEvent(event):
    X = (event.xdata, event.ydata)
    cls = get_class(X, Z)
    if cls == 'ОНР':
        col = 'royalblue'
        print(f'ОНР')
    else:
        col = cls_color[cls]
        print(f'Class {cls}')
    ax.scatter(X[0], X[1], c=col, label='One Point')
    fig.canvas.draw()


def solution_line(x, Z):
    line_list = []
    lZ = len(Z)
    for i in range(lZ - 1):
        y_list = []
        for j in range(i + 1, lZ):
            Y = []
            for k in x:
                if Z[i][0] == Z[j][0]:
                    y = (norm(Z[i]) ** 2 - norm(Z[j]) ** 2) / (2 * (Z[i][1] - Z[j][1]))
                    point = (k, y)
                elif Z[i][1] == Z[j][1]:
                    y = (norm(Z[i]) ** 2 - norm(Z[j]) ** 2) / (2 * (Z[i][0] - Z[j][0]))
                    point = (y, k)
                else:
                    y = (k * (Z[j][0] - Z[i][0]) + (norm(Z[i]) ** 2 - norm(Z[j]) ** 2) / 2) / (Z[i][1] - Z[j][1])
                    point = (k, y)
                Y.append(point)
            y_list.append(Y)
        line_list.append(y_list)
    return line_list


M = [(5.97, 5.90), (5.96, 5.85), (6.21, 6.62), (5.81, 5.47), (5.70, 5.77), (5.01, 5.88), (12.93, 6.61), (11.76, 6.29),
     (11.78, 5.69), (12.64, 6.79), (12.38, 5.79), (12.62, 6.04), (11.65, 5.31), (6.09, 11.61), (6.61, 11.82),
     (6.00, 12.23), (4.45, 12.15), (6.60, 13.04), (6.52, 10.82), (4.36, 12.65), (6.36, 11.29)]

# шаг 1
lM = len(M)
index = randint(0, lM - 1)
X1 = M[index]
Z1 = X1
w1 = np.array([[X1]])
K = 1
M.pop(index)
lM -= 1

# шаг 2
Dmax = norm(sub_points(Z1, M[0]))
for i in range(0, lM):
    D = norm(sub_points(Z1, M[i]))
    if D >= Dmax:
        Dmax = D
        Xl = M[i]
        index = i
Z2 = Xl
K += 1
w2 = np.array([[Xl]])
W = np.concatenate((w1, w2))
M.pop(index)
lM -= 1
Z = [Z1, Z2]
T = norm(sub_points(Z2, Z1))

# шаг 3
alpha = 0.3
lZ = len(Z)
while lM != 0:
    D = []
    for i in range(lM):
        Dmin = norm(sub_points(Z1, M[i]))
        for j in range(0, lZ):
            Dj = norm(sub_points(Z[j], M[i]))
            if Dj <= Dmin:
                Dmin = Dj
        D.append(Dmin)
    Dmax = D[0]
    for k in range(0, lM):
        if D[k] >= Dmax:
            Dmax = D[k]
            index = k
    if Dmax > alpha * T:
        Z.append(M[index])
        w = np.array([M[index]])
        W = np.concatenate((W, [w]))
        lZ += 1
        K += 1
        M.pop(index)
        lM -= 1
        Ck2 = factorial(K) / (2 * factorial(K - 2))
        sumDZ = 0
        for i in range(lZ - 1):
            for j in range(i + 1, lZ):
                sumDZ += norm(sub_points(Z[i], Z[j]))
        T = sumDZ / Ck2
    else:
        break

fig, ax = plt.subplots(figsize=(5, 5))

ax.grid(True, alpha=0.5)
ax.set_xlim([4, 14])
ax.set_ylim([4, 14])

# манипуляция для построения ax.scatter
cls_color = []
for i in range(lZ):
    color = get_color()
    cls_color.append(str(color))

cls_list = classification(Z, M)
print(cls_list)
data = cls_list[0]
labels1 = np.zeros(len(data))
for i in range(1, len(cls_list)):
    data = np.r_[data, cls_list[i]]
    my_array = np.empty(len(cls_list[i]))
    my_array.fill(i)
    labels1 = np.r_[labels1, my_array]

labels1 = list(labels1)
for i in range(len(labels1)):
    for j in range(len(cls_color)):
        if labels1[i] == j:
            labels1[i] = cls_color[j]
print(cls_color)
print(labels1)

# отображение центров кластеров
Z = np.array(Z)
labels2 = cls_color

ax.scatter(data[:, 0], data[:, 1], c=labels1, linewidth=1.5, alpha=0.9)
ax.scatter(Z[:, 0], Z[:, 1], c=labels2, edgecolors='black', linewidth=1.5)

x = np.arange(4, 14, 0.001)

line_list = solution_line(x, Z)


line = []
for i in range(lZ - 1):
    for j in line_list[i]:
        points = []
        for k in range(len(x)):
            # print(f'Точка {j[k]}, Класс {get_class(j[k], Z)}')
            if get_class(j[k], Z) == 'ОНР':
                points.append(j[k])
                try: # начало добработки
                    if get_class(j[k+1], Z) != 'ОНР':
                        line.append(points)
                        points = []
                except:
                    continue # конец
        line.append(points)

my_list = []
for i in line:
    v = np.array([])
    u = np.array([])
    for j in i:
        v = np.append(v, j[0])
        u = np.append(u, j[1])
    my_list.append([v, u])

for i in my_list:
    plt.plot(i[0], i[1], color='grey', linestyle='dashed', alpha=0.5)

"""
points = []
for i in range(lZ - 1):
    for j in line_list[i]:
        for k in range(len(x)):
            if get_class(j[k], Z) == 'ОНР':
                points.append(j[k])
print(f'points: {points}')

v = np.array([])
u = np.array([])
for i in points:
    v = np.append(v, i[0])
    u = np.append(u, i[1])
ax.scatter(v, u, s=4)
"""

fig.canvas.mpl_connect('button_press_event', hendleEvent)
plt.show()

"""
line = []
for i in range(lZ - 1):
    for j in line_list[i]:
        points = []
        for k in range(len(x)):
            X = (x[k], j[k])
            if get_class(X, Z) == 'ОНР':
                points.append(X)
        line.append(points)
print(f'line: {line}')

my_list = []
for i in line:
    v = np.array([])
    u = np.array([])
    for j in i:
        v = np.append(v, j[0])
        u = np.append(u, j[1])
    my_list.append([v, u])

for i in my_list:
    plt.plot(i[0], i[1], color='grey', linestyle='dashed', alpha=0.5)

fig.canvas.mpl_connect('button_press_event', hendleEvent)
plt.show()
"""