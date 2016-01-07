from gurobipy import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
import math
import numpy as np

def delete_by_indices(lst, indices):
    indices_as_set = set(indices)
    return [ lst[i] for i in xrange(len(lst)) if i not in indices_as_set ]

def pcf(x, y, w1, w2, c1, c2, kisi, gama):
    return w1*(x - c1) + w2*( y - c2) + kisi * (abs(x - c1) + abs(y - c2)) - gama


A = [[-2.0, 0.5], [-2.0, -0.5], [-2.0, 2.0], [-2.0, -2.0], [-0.5, 2.0], [-0.5, -2.0], [0.5, 2.0], [0.5, -2.0], [2.0, 0.5], [2.0, -2.0], [2.0, -0.5], [2.0, 2.0], [12.0, 2.0], [12.0, -2.0], [12.0, 0.5], [12.0, -0.5], [13.5, 2.0], [13.5, -2.0], [14.5, 2.0], [14.5, -2.0], [16.0, -0.5], [16.0, 0.5], [16.0, 2.0], [16.0, -2.0]]

B = [[8.0, -6.0], [6.0, -1.0], [20.0, 6.0], [6.0, -6.0], [15.0, 6.0], [20.0, 1.0], [1.0, 6.0], [-1.0, 6.0], [-6.0, 1.0], [20.0, -6.0], [-6.0, -1.0], [-6.0, -6.0], [8.0, -1.0], [13.0, 6.0], [20.0, -1.0], [6.0, 1.0], [15.0, -6.0], [13.0, -6.0], [8.0, 1.0], [-1.0, -6.0], [-6.0, 6.0], [6.0, 6.0], [8.0, 6.0], [1.0, -6.0]]

n = 2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = y = np.arange(-15.0, 25.0, 0.25)
X, Y = np.meshgrid(x, y)

for nokta in range(len(A)):
    x1 = A[nokta][0]
    y1 = A[nokta][1]
    z1 = 0
    ax.scatter(x1, y1, z1, c='black', marker='o')

for point in range(len(B)):
    x2 = B[point][0]
    y2 = B[point][1]
    z2 = 0
    ax.scatter(x2, y2, z2, c='r', marker='^')

while len(A) != 0:
    # Create optimization model
    m = Model('PCF')

    # Create variables
    gamma = m.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')
    w = range(n)
    for i in range(n):
        w[i] = m.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i)

    ksi = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='ksi')

    c = random.randint(0, len(A)-1)

    m.update()
    hataA = {}

    for i in range(len(A)):
        hataA[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataA[%s]' % i)

        m.update()
        m.addConstr(quicksum((A[i][j] - A[c][j]) * w[j] for j in range(n)) + (ksi * quicksum(math.fabs(A[i][j] - A[c][j]) for j in range(n))) - gamma + 1.0 <= hataA[i])

    for z in range(len(B)):

        m.addConstr(quicksum((B[z][r] - A[c][r]) * -w[r] for r in range(n)) - (ksi * quicksum(math.fabs(B[z][r] - A[c][r]) for r in range(n))) + gamma + 1.0 <= 0)

    m.setObjective(quicksum(hataA[k] for k in hataA) / len(hataA), GRB.MINIMIZE)

    # Compute optimal solution
    m.optimize()

    print 'secilen rassal index:'
    print c
    print 'merkez nokta = ', A[c]
    print 'ksi \t= ', ksi.X
    for i in range(n):
        print 'w[',i,'] \t= ', w[i].X
    print 'gamma \t= ', gamma.X

    fDeger=0.0

    silinecekler = []
    for l in range(len(A)):
        fDeger = quicksum((A[l][j] - A[c][j]) * w[j].x for j in range(n)) + (ksi.x * quicksum(math.fabs(A[l][j] - A[c][j]) for j in range(n))) - gamma.x
        print 'hata A[',l,'] = ', fDeger.getValue()

        if fDeger.getValue() <= 0:
            print 'silinecek index: ', l
            silinecekler.append(l);
    Z = []
    zs = np.array([pcf(x, y, w[0].x, w[1].x, A[c][0], A[c][1], ksi.x, gamma.x) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)


    ax.plot_surface(X, Y, Z, rstride=10, cstride=10, alpha=0.9, linewidth=0.2,)


    print 'eski A :', A

    A = delete_by_indices(A, silinecekler)
    print 'yeni A :', A

plt.show()