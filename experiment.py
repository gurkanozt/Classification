import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import csv
import math
import RobustLinearProgramming
import PolyhedralConicFunctions
#import FSV
import PCF_MovingCenter

#RLP predictions are always -1 or 1 but some datasets have different labels. This function converts labels
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = 1
        elif labelData[i] == l2:
            lbls[i] = -1
    return lbls

""""
    Robust Linear Programming example
f = open('Heart.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(' ')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(len(line)-1)])
labels = np.array(labels)
X = np.array(X)
skf = StratifiedKFold(labels, 10)
acc = 0
for train, test in skf:
    pModel = RobustLinearProgramming.RLP()
    pModel.fit(X[train], labels[train], 1, 2)
    clabels = convertLabels(labels[test], 1, 2)
    acc += accuracy_score(clabels, pModel.predict(X[test]))
print "Ten fold average accuracy:", acc / 10

"""

""""
        Robust Linear Programming example
f = open('WBCD9Features.txt')
X = []
labels = []
for line in f:
    row = []
    line = line.split(',')
    labels.append(int(line[-1]))
    X.append([float(line[i]) for i in range(1, len(line)-1)])
labels = np.array(labels)
X = np.array(X)
skf = StratifiedKFold(labels, 10)
acc = 0
for train, test in skf:
    pModel = RobustLinearProgramming.RLP()
    pModel.fit(X[train], labels[train], 2, 4)
    clabels = convertLabels(labels[test],2,4)
    acc += accuracy_score(clabels, pModel.predict(X[test]))
print "Ten fold average accuracy:", acc/10
"""



"""
f = open('CCI2000.txt')
data = []
for line in f:
    row = []
    line = line.split(' ')
    if len(line) != 1:
        for i in line:
            row.append(float(i))
        data.append(row)

data = list(map(list, zip(*data)))

A = []
B = []
f = open('CCI2000Label.txt')
index = 0
for line in f:
    if float(line)<0:
        A.append(data[index])
    else:
        B.append(data[index])
    index += 1

fsv = FSV.FSV_iterate()

parameters = fsv.fit(A,B,rndm=True,a=0.5, l= 0.7)



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


