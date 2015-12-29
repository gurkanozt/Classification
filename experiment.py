import csv
import RobustLinearProgramming
import numpy as np
import matplotlib.pyplot as plt
dosya = open("iris.csv","rb")
reader = csv.reader(dosya, quotechar=',')

"""data = []
X = []
Y = []
for row in reader:
    satir = []
    Y.append(float(row[-1]))
    for i in range(len(row)-1):
         satir.append(float(row[i]))
    X.append(satir)

print(X)
print(Y)

rlp = RobustLinearProgramming.RLP()

parameters = rlp.fit(X,Y,'10FCV')

print(parameters)
print(rlp.predict(X))"""

A = [[-2.0, 0.5], [-2.0, -0.5], [-2.0, 2.0], [-2.0, -2.0], [-0.5, 2.0], [-0.5, -2.0], [0.5, 2.0], [0.5, -2.0], [2.0, 0.5], [2.0, -2.0], [2.0, -0.5], [2.0, 2.0], [12.0, 2.0], [12.0, -2.0], [12.0, 0.5], [12.0, -0.5], [13.5, 2.0], [13.5, -2.0], [14.5, 2.0], [14.5, -2.0], [16.0, -0.5], [16.0, 0.5], [16.0, 2.0], [16.0, -2.0]]
aa = np.array(A)
B = [[8.0, -6.0], [6.0, -1.0], [20.0, 6.0], [6.0, -6.0], [15.0, 6.0], [20.0, 1.0], [1.0, 6.0], [-1.0, 6.0], [-6.0, 1.0], [20.0, -6.0], [-6.0, -1.0], [-6.0, -6.0], [8.0, -1.0], [13.0, 6.0], [20.0, -1.0], [6.0, 1.0], [15.0, -6.0], [13.0, -6.0], [8.0, 1.0], [-1.0, -6.0], [-6.0, 6.0], [6.0, 6.0], [8.0, 6.0], [1.0, -6.0]]
bb = np.array(B)

plt.scatter(aa[:, 0], aa[:, 1], c='b', marker='o')
plt.scatter(bb[:, 0], bb[:, 1], c='r', marker='^')
plt.show()