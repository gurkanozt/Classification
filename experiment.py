import csv
import  math
import RobustLinearProgramming
import PolyhedralConicFunctions
import numpy as np
import matplotlib.pyplot as plt
dosya = open("iris.csv","rb")
reader = csv.reader(dosya, quotechar=',')

data = []
A = []
B = []
for row in reader:
    satir = []
    for i in range(len(row) - 1):
        satir.append(float(row[i]))
    if float(row[-1]) == 1:
        A.append(satir)
    else:
        B.append(satir)
print(A)
print(B)


rlp = RobustLinearProgramming.RLP()

parameters = rlp.fit(A,B)

prediction = rlp.predict(A)


print(rlp.predict(B))


"""
A = [[-2.0, 0.5], [-2.0, -0.5], [-2.0, 2.0], [-2.0, -2.0], [-0.5, 2.0], [-0.5, -2.0], [0.5, 2.0], [0.5, -2.0], [2.0, 0.5], [2.0, -2.0], [2.0, -0.5], [2.0, 2.0], [12.0, 2.0], [12.0, -2.0], [12.0, 0.5], [12.0, -0.5], [13.5, 2.0], [13.5, -2.0], [14.5, 2.0], [14.5, -2.0], [16.0, -0.5], [16.0, 0.5], [16.0, 2.0], [16.0, -2.0]]
aa = np.array(A)
B = [[8.0, -6.0], [6.0, -1.0], [20.0, 6.0], [6.0, -6.0], [15.0, 6.0], [20.0, 1.0], [1.0, 6.0], [-1.0, 6.0], [-6.0, 1.0], [20.0, -6.0], [-6.0, -1.0], [-6.0, -6.0], [8.0, -1.0], [13.0, 6.0], [20.0, -1.0], [6.0, 1.0], [15.0, -6.0], [13.0, -6.0], [8.0, 1.0], [-1.0, -6.0], [-6.0, 6.0], [6.0, 6.0], [8.0, 6.0], [1.0, -6.0]]
bb = np.array(B)





deneme = PolyhedralConicFunctions.PCF_iterative()
pcfcs = deneme.fit(A,B)

for i in pcfcs:
    print 'PCF:' ,'gamma =',i.gamma,'ksi =', i.ksi ,'center =',i.center
    print '     normal =',i.w


predictions= deneme.predict(A)
print predictions


plt.scatter(aa[:, 0], aa[:, 1], c='b', marker='o')
plt.scatter(bb[:, 0], bb[:, 1], c='r', marker='^')
plt.show()"""


