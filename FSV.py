import numpy as np
import math
from gurobipy import *


class FSV:
    def __init__(self):
        self.w = list()
        self.v=list()
        self.gamma= 0
        self.a = 5
        self.l = 0.8
        self.vo =list()
        self.dimension = 0
    def fit(self, X, Y ,a, l, v):
        self.a = a
        self.l = l
        self.vo =v
        m = len(X)
        self.dimension = len(X[0])

        model = Model()

        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='gamma')
        w = range(self.dimension)
        for i in range(self.dimension):
            w[i] = model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i)

        v = range(self.dimension)
        for i in range(self.dimension):
            v[i]=model.addVar(vtype=GRB.CONTINUOUS, name='v[%s]' % i)

        model.update()
        for i in range(self.dimension):
            model.addConstr(w[i]<=v[i])
            model.addConstr(w[i]>= -v[i])
        model.update()

        errorA = {}
        errorB = {}

        for i in range(m):
            if Y[i] == 1.0:
                errorA[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i)
                model.update()
                model.addConstr(quicksum(X[i][j] * w[j] for j in range(self.dimension)) - gamma + 1.0 <= errorA[i])

            else:
                errorB[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorB[%s]' % i)
                model.update()
                model.addConstr(quicksum(-X[i][r] * w[r] for r in range(self.dimension)) + gamma + 1.0 <= errorB[i])

        eps = self.epslon()
        model.setObjective((1-self.l)*(quicksum(errorA[i] for i in errorA) / len(errorA) +
                           quicksum(errorB[i] for i in errorB) / len(errorB)) + self.l*self.a*(np.dot(eps,np.subtract(self.vo - v))), GRB.MINIMIZE)

        model.optimize()
        self.gamma = gamma.X
        for i in range(self.dimension):
            self.w.append(w[i].X)
            self.v[v[i].X]

        return self.gamma, self.w, self.v


    def epslon(self):
         eps = list()
         tempvo   =  -1*self.a*self.vo
         for i in range(self.dimension):
           eps.append(math.exp(tempvo[i]))

         return eps