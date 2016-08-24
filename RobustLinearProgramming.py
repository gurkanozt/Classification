import numpy as np
import math
from gurobipy import *

"""
    Robust Linear Programming Discrimination of Two Linearly Inseperable Sets, Bennet and Mangasarian, 1992
    -To execute this algortihm Gurobi solver and gurobi.py are required
     http://www.gurobi.com/
     https://www.gurobi.com/documentation/6.5/quickstart_mac/the_gurobi_python_interfac.html

     A,B  the datasets belong different classes
"""

class RLP:
    def __init__(self):
        self.w = list()
        self.gamma = 0

    def fit(self, X, Y, lb1, lb2):
        # dimension = number of features
        dimension = len(X[0])

        #create a gurobi model
        model = Model()
        #add gamma variable to the model
        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='gamma')
        w = list()
        #add n dimensional w variables
        for i in range(dimension):
            w.append( model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i))

        model.update()
        errorA = list()
        errorB = list()
        #add error variables and constraints
        for i in range(len(X)):
            if Y[i] == lb1:
                 errorA.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i))
                 model.update()
                 model.addConstr(quicksum(X[i][j] * w[j] for j in range(dimension)) - gamma + 1.0 <= errorA[len(errorA)-1])
            elif Y[i] == lb2:
                 errorB.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorB[%s]' % i))
                 model.update()
                 model.addConstr(quicksum(-X[i][j] * w[j] for j in range(dimension)) + gamma + 1.0 <= errorB[len(errorB)-1])
        #set obective function
        model.setObjective(quicksum(i for i in errorA) / len(errorA) +
                           quicksum(i for i in errorB) / len(errorB), GRB.MINIMIZE)

        #get optimized gamma and w values
        model.optimize()
        self.gamma = gamma.X
        for i in range(dimension):
            self.w.append(w[i].X)
        return self.gamma, self.w

    #this fuction gives predictions for given dataset according to fitted model
    def predict(self, X):
        p = list()
        for i in range(len(X)):
             p.append(-1*math.copysign(1, (np.dot(self.w, X[i]) - self.gamma)))

        return p