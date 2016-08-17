import numpy as np
import math
from gurobipy import *
import  random

class FSV:
    def __init__(self):
        self.w = list()
        self.v=list()
        self.gamma= 0
        self.a = 5
        self.l = 0.8
        self.vo =list()
        self.dimension = 0
        self.y = list()
        self.z  = list()

    def solve(self, A,B,a,l,w,g,y,z,v):
        self.a = a
        self.l = l
        self.vo =v

        self.dimension = len(A[0])

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

        for i in range(len(A)):
            errorA[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i)
            model.update()
            model.addConstr(quicksum(A[i][j] * w[j] for j in range(self.dimension)) - gamma + 1.0 <= errorA[i])
        for i in range(len(B)):
          errorB[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorB[%s]' % i)
          model.update()
          model.addConstr(quicksum(-B[i][r] * w[r] for r in range(self.dimension)) + gamma + 1.0 <= errorB[i])

        eps = self.epslon()
        model.setObjective((1-self.l)*(quicksum(errorA[i] for i in errorA) / len(errorA) +
                           quicksum(errorB[i] for i in errorB) / len(errorB)) + self.l*self.a*(np.dot(eps,np.subtract(v - self.vo))), GRB.MINIMIZE)

        model.optimize()
        self.gamma = gamma.X
        for i in range(self.dimension):
            self.w.append(w[i].X)
            self.v[v[i].X]

        for i in range(len(A)):
            self.y.append(errorA[i].X)

        for i in range(len(B)):
            self.z.append(errorB[i].X)

        return self.w, self.gamma, self.y, self.z, self.v

    def epslon(self):
         eps = list()
         tempvo   =  -1*self.a*self.vo
         for i in range(self.dimension):
           eps.append(math.exp(tempvo[i]))

         return eps

class FSV_iterate:
    def __init__(self):
        self.dimension = 0
        self.w = list()
        self.gamma = 0
        self.a = 5
        self.l = 0.7
        self.y = list()
        self .z = list()
        self.v= list()

    def fit(self,A,B,a=5,l=0.7,rndm = True,*args):
        self.a = a
        self.l = l
        self.dimension =  len(A[0])
        tempFsv = FSV()
        if rndm:
            self.w = np.random.rand(self.dimension)
            self.gamma =  random.random()
            self.y = np.random.rand(len(A))
            self.z = np.random.rand(len(B))
            self.v = np.random.rand(self.dimension)
        else :
            self.w = args[0]
            self.gamma = args[1]
            self.y = args[2]
            self.z = args[3]
            self.v = args[4]

        parameters = tempFsv.solve(A, B, self.a, self.l, self.w, self.gamma, self.y, self.z, self.v)
        while self.checkZero(self.y, self.z, self.v, parameters[2], parameters[3], parameters[4] ) == False :
            self.w = parameters[0]
            self.gamma = parameters[1]
            self.y = parameters[2]
            self.z = parameters [3]
            self.v = parameters [4]
            parameters = tempFsv.solve(A, B, self.a, self.l, self.w, self.gamma, self.y, self.z, self.v)
        return parameters

    def checkZero(self, y, z, v, yi, zi, vi ):
         result = False
         t = (1-self.l)*(np.dot(np.ones(len(y)), np.subtract(yi,y))/len(y)+ np.dot(np.ones(len(z)),np.subtract(zi,z))/len(z))+self.l*self.a*np.dot(self.epslon(v),np.subtract(vi,v))
         if t == 0:
             result = True
         return result

    def epslon(self,vo):
        eps = list()
        tempvo = -1 * self.a * vo
        for i in range(self.dimension):
            eps.append(math.exp(tempvo[i]))
        return eps