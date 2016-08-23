import numpy as np
import math
from gurobipy import *
import random
"""
    -Feature Selection via Mathematical Programming, Bradley et al. , 1997
    -In this paper there are three algorithms named as FSS, FSV and FSB but only implemented algorithm is the FSV.
    -To execute this algortihm Gurobi solver and gurobi.py are required
     http://www.gurobi.com/
     https://www.gurobi.com/documentation/6.5/quickstart_mac/the_gurobi_python_interfac.html

    Implementation notes:
      a -> alfa
      l -> lambda
      a,l parameters are set 0.5 and 0.7 as default. To see a review of FSV algorithm please look "Use of Zero-Norm with Linear Models and Kernel Methods" "Weston et. al"
      A,B  the datasets belong different classes

"""
class FSV:

  def __init__(self):

      self.a = 0
      self.l = 0.7
      self.dimension = 0
      self.gamma = 0
      self.w = list()
      self.y = list()
      self.z = list()
      self.v = list()

  def fit(self, A, B, a=0.5, l=0.7):
    # dimension = number of features
      self.dimension = A[0]
    #set random starting parameters
      self.w = np.random.rand(self.dimension)
      self.gamma= np.random.rand(1)
      self.y = np.random.rand(self.dimension)
      self.z = np.random.rand(self.dimension)
      self.v = np.random.rand(self.dimension)

    #solve the problem for given parameters
      parameters = self.__solveModel(A, B)
    #untill the equaiton 17 holds (equation 17 from the  paper)
      while self.__check(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]) != True:
        self.w = parameters[0]
        self.gamma = parameters[1]
        self.y = parameters[2]
        self.z = parameters[3]
        self.v = parameters[4]
        parameters = self.solveModel(A,B)

      return self.w, self.gamma, self.y, self.z, self.v


  def __solveModel(self, A, B):
      #create a gurobi model
      model = Model()
      #add the gamma variable to the model
      gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='gamma')
      w = list()
      v = list()
      #add n dimensional w and v variables
      for i in range(self.dimension):
          w.append(model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i))
          v.append(model.addVar(vtype=GRB.CONTINUOUS, name='v[%s]' % i))
      #update the model to use them in constraints
      model.update()
      #add v constraints
      model.addConstr(-v[i] <= w[i]for i in range(self.dimension))
      model.addConstr(w[i] <= v[i] for i in range(self.dimension))
      model.update()
      errorA = list()
      errorB = list()
      #add error variables and constraints
      for i in range(len(A)):
          errorA.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorA[%s]' % i))
          model.update()
          model.addConstr(quicksum(A[i][j] * w[j] for j in range(self.dimension)) - self.gamma + 1.0 <= errorA[i])
      for i in range(len(B)):
          errorB.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errorB[%s]' % i))
          model.update()
          model.addConstr(quicksum(-B[i][r] * w[r] for r in range(self.dimension)) + self.gamma + 1.0 <= errorB[i])
      #t1 and t2 are  parts of objective function
      t1 = (1-self.l)*(np.dot(np.ones(len(self.y)),self.y)/len(self.y)+ np.dot(np.ones(len(self.z),self.z))/len(self.z) )
      t2 = self.l*self.a*(np.dot(self.__exp(), np.subtract(v,self.v)))
      #set the objective function
      model.setObjective(t1+t2, GRB.MINIMIZE)

      model.optimize()
      #return all wariables in same oder the paper
      return [w[i].X for i in range(self.dimension)], gamma.X ,[errorA[i].X for i in range(len(errorA))], [errorB[i].X for i in range(len(errorB))], [v[i].X for i in range(self.dimension)]
  #this function checks if it is hold or not equation 17 from the paper
  def __check(self,wi,gi,yi,zi,vi):
      temp1 = (1-self.l)*(np.dot(np.ones(len(yi)),np.subtract(yi-self.y))/len(self.y) +np.dot(np.ones(len(zi),np.subtract(zi,self.z)))/len(self.z))
      temp2 = self.l*self.a*(np.dot(self.__exp(), np.subtract(vi,self.v)))
      return temp1==temp2
  #this function retuns exponantial values of v
  def __exp(self):

       return [math.expm1(-1*i) for i in self.v]