import numpy as np
import math
import random
from gurobipy import *

class PCF:
    def __init__(self):
        self.w = list()
        self.gamma = 0
        self.ksi = 0
        self.center = list()
#change
    def setParam(self, A, B, center):
        # center n boyutlu vektor olarak gelsin
        # set problem parameters dimension = number of features, m =  s(A), p = s(B)
        self.center = center
        dimension = len(A[0])
        #
        m = len(A)
        p = len(B)

        # initialize gurobi model
        model = Model()

        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')

        ksi = model.addVar(vtype=GRB.CONTINUOUS, lb =0, name='ksi')

        w = range(dimension)
        for i in range(dimension):
            w[i] = model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i)



        model.update()
        errorA = {}

        for i in range(m):
            errorA[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i)
            model.update()
            model.addConstr(quicksum((A[i][j] - center[j]) * w[j] for j in range(dimension)) + (ksi * quicksum(math.fabs(A[i][j] - center[j]) for j in range(dimension))) - gamma + 1.0 <= errorA[i]  )

        for i in range(p):
            model.addConstr(quicksum((B[i][j] - center[j]) * -w[j] for j in range(dimension)) - (ksi * quicksum(math.fabs(B[i][j] - center[j]) for j in range(dimension))) + gamma + 1.0 <= 0)

        model.setObjective(quicksum(errorA[i] for i in errorA) / len(errorA), GRB.MINIMIZE)
        model.optimize()
        self.gamma = gamma.X
        self.ksi = ksi.X
        for i in range(dimension):
            self.w.append(w[i].X)


class PCFC:
    def __init__(self):
        self.pcfs = list()
        self.dimension = 0


    def fit(self, A, B):
        self.dimension=len(A[0])

        while len(A) !=0 :
            center = A[random.randint(0, len(A)-1)]
            temp = PCF()
            temp.setParam(A,B,center)
            self.pcfs.append(temp)
            A = self.__updateSet(A,self.pcfs[-1], center)

        return self.pcfs

    def __delete(self,lst, indices):
        indices = set(indices)
        return [lst[i] for i in xrange(len(lst)) if i not in indices]

    def __updateSet(self, A, pc,center):

        deleted = []
        for i in range(len(A)):

            f = quicksum((A[i][j] - center[j]) * pc.w[j] for j in range(self.dimension)) + (pc.ksi * quicksum(math.fabs(A[i][j] - center[j]) for j in range(self.dimension))) - pc.gamma

            if f.getValue() <= 0.0:
                deleted.append(i)

        return self.__delete(A,deleted)







