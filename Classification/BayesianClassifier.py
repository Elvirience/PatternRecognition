import numpy as np
import random
from random import randint
from math import log
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import norm
import matplotlib.pyplot as plt


def sub_points(point1, point2):
    return point1[0] - point2[0], point1[1] - point2[1]


def sum_points(point1, point2):
    return point1[0] + point2[0], point1[1] + point2[1]


def dev_point(point, N):
    return point[0] / N, point[1] / N


def coef_A(Cj_1, Cl_1):
    return Cl_1[1][1] - Cj_1[1][1]


def coef_B(x, m, j, l, Cj_1, Cl_1):
    return 2 * Cj_1[1][1] * m[j][1] - (Cj_1[1][0] + Cj_1[0][1]) * (x - m[j][0]) - 2 * Cl_1[1][1] * m[l][1] \
           + (Cl_1[1][0] + Cl_1[0][1]) * (x - m[l][0])


def coef_C(x, m, j, l, Cj, Cl, Cj_1, Cl_1, prob):
    return (Cj_1[1][0] + Cj_1[0][1]) * (x - m[j][0]) * m[j][1] - m[j][1] ** 2 * Cj_1[1][1] - Cj_1[0][0] \
           * (x - m[j][0]) ** 2 \
           - (Cl_1[1][0] + Cl_1[0][1]) * (x - m[l][0]) * m[l][1] + m[l][1] ** 2 * Cl_1[1][1] + Cl_1[0][0] \
           * (x - m[l][0]) ** 2 \
           - 2 * log(prob[l]) + log(det(Cl)) + 2 * log(prob[j]) - log(det(Cj))


def d(cls, X, m, prob, C, C_1):
    X = np.array(X)
    m = np.array(m[cls])
    A = np.dot((X - m), C_1)
    B = (X - m).T
    return log(prob[cls]) - log(abs(det(C))) / 2 - np.dot(A, B) / 2


def get_class(X, m, lZ, C, C_1, eps=0.4, cls=0, count=0):
    d0 = d(0, X, m, prob, C[0], C_1[0])
    for i in range(1, lZ):
        d_next = d(i, X, m, prob, C[i], C_1[i])
        if d_next > d0:
            d0 = d_next
            cls = i
    for j in range(0, lZ):
        if abs(d0 - d(j, X, m, prob, C[j], C_1[j])) <= eps:
            count += 1
    if count > 1:
        return 'ОНР'
    else:
        return cls


def classification(m, Z, lZ, M, C, C_1):
    clusters = []
    for i in range(lZ):
        that_class = []
        for j in M:
            if get_class(j, m, lZ, C, C_1) == i:
                that_class.append(j)
        if len(that_class) == 0:
            that_class.append(Z[i])
        that_class = np.array(that_class)
        clusters.append(that_class)
    return clusters


def get_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: np.random.randint(0, 255), range(3)))
    return color


def hendleEvent(event):
    X = (event.xdata, event.ydata)
    cls = get_class(X, m, lZ, cov_list, inv_list)
    if cls == 'ОНР':
        col = 'royalblue'
        print(f'ОНР')
    else:
        col = cls_color[cls]
        print(f'Class {cls}')
    ax.scatter(X[0], X[1], c=col, label='One Point')
    fig.canvas.draw()


def solution_line(x, m, prob, lZ, C, C_1):
    points = []
    for j in range(lZ - 1):
        for l in range(j + 1, lZ):
            A = coef_A(C_1[j], C_1[l])
            for k in x:
                B = coef_B(k, m, j, l, C_1[j], C_1[l])
                Cc = coef_C(k, m, j, l, C[j], C[l], C_1[j], C_1[l], prob)
                if (B ** 2) - 4 * A * Cc > 0:
                    y1 = (-B + ((B ** 2) - 4 * A * Cc) ** (1 / 2)) / (2 * A)
                    y2 = (-B - ((B ** 2) - 4 * A * Cc) ** (1 / 2)) / (2 * A)
                    if get_class((k, y1), m, lZ, cov_list, inv_list) == 'ОНР':
                        points.append((k, y1))
                    if get_class((k, y2), m, lZ, cov_list, inv_list) == 'ОНР':
                        points.append((k, y2))
    return points


M = [(5.97, 5.90), (5.96, 5.85), (6.21, 6.62), (5.81, 5.47), (5.70, 5.77), (5.01, 5.88), (12.93, 6.61), (11.76, 6.29),
     (11.78, 5.69), (12.64, 6.79), (12.38, 5.79), (12.62, 6.04), (11.65, 5.31), (6.09, 11.61), (6.61, 11.82),
     (6.00, 12.23), (4.45, 12.15), (6.60, 13.04), (6.52, 10.82), (4.36, 12.65), (6.36, 11.29)]

# шаг 1
lM = len(M)
# Z = [(5.81, 5.47), (12.38, 5.79), (6.00, 12.23)]
K = 3
Z = random.sample(M, K)
lZ = len(Z)
# упорядоченный список классов
cls_index_list = list(range(lZ))

exit = False
iteration = 1
while exit == False:

    print(iteration)
    count = 0

    # шаг 2
    # классификация образов из множества M
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

print(f'f N_list = {N_list}')
m = Z
prob = [i / lM for i in N_list]


cov_list = []
inv_list = []
for cls in range(lZ):
    sigma = np.array([[0, 0],
                      [0, 0]])
    for j in range(lM):
        if cls == cls_list[j]:
            sigma = sigma + (np.array([M[j]]).T * np.array([M[j]]))
    cov_matrix = sigma / N_list[cls] - np.array([m[cls]]).T * np.array([m[cls]])
    inv_matrix = inv(cov_matrix)
    cov_list.append(cov_matrix)
    inv_list.append(inv_matrix)


fig, ax = plt.subplots(figsize=(5, 5))

ax.grid(True, alpha=0.5)
ax.set_xlim([4, 14])
ax.set_ylim([4, 14])

# манипуляция для построения ax.scatter
cls_color = []
for i in range(lZ):
    color = get_color()
    cls_color.append(str(color))

clusters = classification(m, Z, lZ, M, cov_list, inv_list)
print(clusters)
data = clusters[0]
labels1 = np.zeros(len(data))
for i in range(1, len(clusters)):
    data = np.r_[data, clusters[i]]
    my_array = np.empty(len(clusters[i]))
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

points = solution_line(x, m, prob, lZ, cov_list, inv_list)
print(len(points))

v = np.array([])
u = np.array([])
for i in points:
    v = np.append(v, i[0])
    u = np.append(u, i[1])
ax.scatter(v, u, s=4)

print(len(v), len(u))

fig.canvas.mpl_connect('button_press_event', hendleEvent)
plt.show()
