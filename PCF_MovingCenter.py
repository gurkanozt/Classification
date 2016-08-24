import PolyhedralConicFunctions
import numpy as np
import  random
import gurobipy

class PCF_MovingCenter:
    def __init__(self):
        self.pcfs = list()
        self.dimension = 0

    def fit(self, A, B):
        self.dimension = A[0]
        while len(A) != 0:
            center = A[random.randint(0, len(A) - 1)]
            temp = PolyhedralConicFunctions.PCF()
            temp.setParam(A, B, center)
            while not np.isclose(center, temp.center, rtol=1, atol=1, equal_nan=False):
                temp.setParam(A, B, temp.center)
            self.pcfs.append(temp)
            A = self.__updateSet(A, self.pcfs[-1], center)
        return self.pcfs

    def predict(self, X):
        predictions = list()
        for i in range(len(X)):
            f = 0
            for p in self.pcfs:
                f = np.dot(np.subtract(X[i], p.center), p.w) + (
                p.ksi * np.linalg.norm((np.subtract(X[i], p.center)), 1)) - p.gamma
                if f <= 0.0:
                    f = -1
                    break
                else:
                    f = 1

            predictions.append(f)
        return predictions

    def __delete(self, lst, indices):
        indices = set(indices)
        return [lst[i] for i in xrange(len(lst)) if i not in indices]

    def __updateSet(self, A, pc, center):
        deleted = []

        for i in range(len(A)):

            f = np.dot(np.subtract(A[i], center), pc.w) + (
            pc.ksi * np.linalg.norm((np.subtract(A[i], center)), 1)) - pc.gamma
            if f <= 0.0:
                deleted.append(i)

        return self.__delete(A, deleted)