import numpy as np
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt


def another_w(i, W):
    w = []
    for j in range(len(W)):
        if j != i:
            w.append((j, W[j]))
    return w


def sub_points(point1, point2):
    return point1[0] - point2[0], point1[1] - point2[1]


def sum_points(point1, point2):
    return point1[0] + point2[0], point1[1] + point2[1]


def dev_point(point, N):
    return point[0] / N, point[1] / N


def d(X, w):
    return w[0] * X[0] + w[1] * X[1] + w[2]


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
    cls = get_class(X, w_solves)
    if cls == 'ОНР':
        col = 'royalblue'
        print(f'ОНР')
    else:
        col = cls_color[cls]
        print(f'Class {cls}')
    ax.scatter(X[0], X[1], c=col, label='One Point')
    fig.canvas.draw()


def solution_line(x, Z):
    points = []
    lZ = len(Z)
    for i in range(lZ - 1):
        for j in range(i + 1, lZ):
            for k in x:
                y = (k * (Z[j][0] - Z[i][0]) + Z[j][2] - Z[i][2]) / (Z[i][1] - Z[j][1])
                if get_class((k, y), Z) == 'ОНР':
                    points.append((k, y))
    return points


# Каждому свой цвет
def solution_line1(x, Z):
    lines = []
    lZ = len(Z)
    for i in range(lZ - 1):
        for j in range(i + 1, lZ):
            points = []
            for k in x:
                y = (k * (Z[j][0] - Z[i][0]) + Z[j][2] - Z[i][2]) / (Z[i][1] - Z[j][1])
                if get_class((k, y), Z) == 'ОНР':
                    points.append((k, y))
            lines.append(points)
    return lines


M = [(5.97, 5.90), (5.96, 5.85), (6.21, 6.62), (5.81, 5.47), (5.70, 5.77), (5.01, 5.88), (12.93, 6.61), (11.76, 6.29),
     (11.78, 5.69), (12.64, 6.79), (12.38, 5.79), (12.62, 6.04), (11.65, 5.31), (6.09, 11.61), (6.61, 11.82),
     (6.00, 12.23), (4.45, 12.15), (6.60, 13.04), (6.52, 10.82), (4.36, 12.65), (6.36, 11.29)]

lM = len(M)

# k-means

Z = [(5.81, 5.47), (12.38, 5.79), (6.00, 12.23)]
K = 3
Z = random.sample(M, K)
Z = [(5.81, 5.47), (12.38, 5.79), (6.00, 12.23)]
lZ = len(Z)
cls_index_list = list(range(lZ))

exit = False
iteration = 1
while exit == False:

    count = 0

    cls_list = []
    for i in M:
        D_min = norm(sub_points(i, Z[0]))
        index = 0
        for j in range(1, len(Z)):
            D_next = norm(sub_points(i, Z[j]))
            if D_next <= D_min:
                D_min = D_next
                index = j
        cls_list.append(index)
    cls_list_sorted = sorted(cls_list)

    N_list = []
    for i in cls_index_list:
        NN = 0
        for j in cls_list:
            if i == j:
                NN += 1
        N_list.append(NN)

    ZZ = []
    for i in cls_index_list:
        sum_X = (0, 0)
        for j in range(lM):
            if i == cls_list[j]:
                sum_X = sum_points(sum_X, M[j])
        ZZ.append(dev_point(sum_X, N_list[i]))

    for i in cls_index_list:
        if ZZ[i] != Z[i]:
            count += 1
    if count == 0:
        exit = True
    else:
        Z = ZZ
        iteration += 1

# Пополнение образов
W = []
for i in cls_index_list:
    w = []
    for j in range(lM):
        if i == cls_list[j]:
            a = list(M[j])
            a.append(1)
            w.append(np.array(a))
    W.append(w)

w_solves = []
lW = len(W)
for i in range(lW):
    w_solves.append([0, 0, 1])


c = 0.1
N = lM
m = 0
while m < N:
    for i in range(lW):
        w2 = another_w(i, w_solves)
        for j in W[i]:
            count = 0
            for k in w2:
                if d(j, w_solves[i]) <= d(j, k[1]):
                    w_solves[i] = w_solves[i] + c * j
                    w_solves[k[0]] = w_solves[k[0]] - c * j
                    count += 1
            if count == 0:
                m += 1
                if m == N:
                    break
            else:
                m = 0

fig, ax = plt.subplots(figsize=(5, 5))

ax.grid(True, alpha=0.5)
ax.set_xlim([4, 14])
ax.set_ylim([4, 14])

# манипуляция для построения ax.scatter
cls_color = []
for i in range(lZ):
    color = get_color()
    cls_color.append(str(color))

cls_list = classification(w_solves, M)
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

"""
# Без цвета
points = solution_line(x, w_solves)

v = np.array([])
u = np.array([])
for i in points:
    v = np.append(v, i[0])
    u = np.append(u, i[1])
ax.scatter(v, u, s=4)
"""

""" 
# Каждому свой цвет
lines = solution_line1(x, w_solves)

for i in range(len(cls_color)):
    v = np.array([])
    u = np.array([])
    for j in lines[i]:
        v = np.append(v, j[0])
        u = np.append(u, j[1])
    print(cls_color[i])
    ax.scatter(v, u, c=cls_color[i], s=4)
"""


# Логический цвет
lines = solution_line1(x, w_solves)

count = 0
for i in range(len(cls_color)):
    v = np.array([])
    u = np.array([])
    for j in lines[i]:
        v = np.append(v, j[0])
        u = np.append(u, j[1])
    ax.scatter(v, u, c=cls_color[count], s=4)
    count = i


fig.canvas.mpl_connect('button_press_event', hendleEvent)
plt.show()

