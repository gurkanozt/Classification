import numpy as np
import math
from gurobipy import *


class RLP:
    def __init__(self):
        self.w = list()
        self.gamma=0

    def fit(self, A, B):

        dimension = len(A[0])

        model = Model()
        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='gamma')
        w = range(dimension)
        for i in range(dimension):
            w[i] = model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i)

        model.update()
        errorA = {}
        errorB = {}
        for i in range(len(A)):
             errorA[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i)
             model.update()
             model.addConstr(quicksum(A[i][j] * w[j] for j in range(dimension)) - gamma + 1.0 <= errorA[i])
        for i in range( len(B)):
             errorB[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorB[%s]' % i)
             model.update()
             model.addConstr(quicksum(-B[i][r] * w[r] for r in range(dimension)) + gamma + 1.0 <= errorB[i])

        model.setObjective(quicksum(errorA[i] for i in errorA) / len(errorA) +
                           quicksum(errorB[i] for i in errorB) / len(errorB), GRB.MINIMIZE)

        model.optimize()
        self.gamma = gamma.X
        for i in range(dimension):
            self.w.append(w[i].X)



        return self.gamma, self.w

    def predict(self, X):
        p = list()
        for i in range(len(X)):
             p.append(-1*math.copysign(1, (np.dot(self.w, X[i]) - self.gamma)))

        return p