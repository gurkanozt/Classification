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

        # set problem parameters dimension = number of features, m =  s(A), p = s(B)
        self.center = center
        dimension = len(A[0])
        #
        m = len(A)
        p = len(B)

        # initialize gurobi model
        model = Model()
        # define cone equation
        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')
        ksi = model.addVar(vtype=GRB.CONTINUOUS, lb =0, name='ksi')
        w = range(dimension)
        for i in range(dimension):
            w[i] = model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i)



        model.update()
        errorA = {}
        #set constraints
        for i in range(m):
            errorA[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i)
            model.update()
            model.addConstr(quicksum((A[i][j] - center[j]) * w[j] for j in range(dimension)) + (ksi * quicksum(math.fabs(A[i][j] - center[j]) for j in range(dimension))) - gamma + 1.0 <= errorA[i]  )

        for i in range(p):
            model.addConstr(quicksum((B[i][j] - center[j]) * -w[j] for j in range(dimension)) - (ksi * quicksum(math.fabs(B[i][j] - center[j]) for j in range(dimension))) + gamma + 1.0 <= 0)
        #set objective function
        model.setObjective(quicksum(errorA[i] for i in errorA) / len(errorA), GRB.MINIMIZE)
        #solve problem
        model.optimize()
        #get cone parameters from solution
        self.gamma = gamma.X
        self.ksi = ksi.X
        for i in range(dimension):
            self.w.append(w[i].X)


class PCF_iterative:
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

    def predict(self,X):

        predictions = list()
        for i in range(len(X)):
            for p in self.pcfs:
                #f = quicksum((X[i][j] - p.center[j]) * p.w[j] for j in range(self.dimension)) + (p.ksi * quicksum(math.fabs(X[i][j] - p.center[j]) for j in range(self.dimension))) - p.gamma
                f = np.dot(np.subtract(X[i], p.center), p.w) + (p.ksi * np.linalg.norm((np.subtract(X[i], p.center)), 1)) - p.gamma
                if f <= 0.0:
                    predictions.append(-1)
                    break
                else:
                    predictions.append(1)
        return predictions

    def __delete(self,lst, indices):
        indices = set(indices)
        return [lst[i] for i in xrange(len(lst)) if i not in indices]

    def __updateSet(self, A, pc,center):
        deleted = []
        for i in range(len(A)):

            #f = quicksum((A[i][j] - center[j]) * pc.w[j] for j in range(self.dimension)) + (pc.ksi * quicksum(math.fabs(A[i][j] - center[j]) for j in range(self.dimension))) - pc.gamma
            f = np.dot(np.subtract(A[i], center), pc.w) + (pc.ksi * np.linalg.norm((np.subtract(A[i], center)), 1)) - pc.gamma
            if f <= 0.0:
                deleted.append(i)

        return self.__delete(A,deleted)


class PCF_movingcenter:
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

    def predict(self,X):

        predictions = list()
        for i in range(len(X)):
            for p in self.pcfs:
                #f = quicksum((X[i][j] - p.center[j]) * p.w[j] for j in range(self.dimension)) + (p.ksi * quicksum(math.fabs(X[i][j] - p.center[j]) for j in range(self.dimension))) - p.gamma
                f = np.dot(np.subtract(X[i], p.center), p.w) + (p.ksi * np.linalg.norm((np.subtract(X[i], p.center)), 1)) - p.gamma
                if f <= 0.0:
                    predictions.append(-1)
                    break
                else:
                    predictions.append(1)
        return predictions

    def __delete(self,lst, indices):
        indices = set(indices)
        return [lst[i] for i in xrange(len(lst)) if i not in indices]

    def __updateSet(self, A, pc,center):
        deleted = []
        for i in range(len(A)):

            #f = quicksum((A[i][j] - center[j]) * pc.w[j] for j in range(self.dimension)) + (pc.ksi * quicksum(math.fabs(A[i][j] - center[j]) for j in range(self.dimension))) - pc.gamma
            f = np.dot(np.subtract(A[i], center), pc.w) + (pc.ksi * np.linalg.norm((np.subtract(A[i], center)), 1)) - pc.gamma
            if f <= 0.0:
                deleted.append(i)
        self.__updateSet(A,pc,np.mean(deleted,0))
        return self.__delete(A,deleted)



