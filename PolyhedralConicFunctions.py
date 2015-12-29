import numpy as np
import math
import random
from gurobipy import *


class PCF:
    def __init__(self):
        self.w = list()
        self.gamma = 0
        self.ksi = 0


class PCFC:
    def __init__(self):
        self.pcfs = list()

    def fit(self, X, Y):
        m = len(X)
        dimension = len(X[0])

        while m != 0:

            model = Model()
            gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')
            ksi = model.addVar(vtype=GRB.CONTINUOUS, lb =0, name='ksi')

            w = range(w)
            for i in range(dimension):
                w[i] = model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i)

            r = random.randint(0, m-1)

            model.update()
            errorA = {}

            for i in range(m):
                if Y[i] == 1.0:
                    errorA[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i)
                    model.update()
                    m.addConstr(quicksum((X[i][j] - X[r][j]) * w[j] for j in range(m)) + (ksi * quicksum(math.fabs(X[i][j] - X[r][j]) for j in range(m))) - gamma + 1.0 <= errorA[i])
                else:
                    m.addConstr(quicksum((X[i][j] - X[r][j]) * -w[j] for j in range(m)) - (ksi * quicksum(math.fabs(X[i][j] - X[r][j]) for j in range(m))) + gamma + 1.0 <= 0)

            model.setObjective(quicksum(errorA))