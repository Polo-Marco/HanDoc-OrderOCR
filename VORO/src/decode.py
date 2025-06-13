import numpy as np
from itertools import permutations
import math



class CountDecoder():
    """
    A simple decoder, it just count the number of 0's in the simetrical matrix
    """
    def __init__(self, P):
        self._P = P
        self.id = 'count'
    def run(self):
        A = self._P + np.finfo(float).eps
        #A = self._P
        A = (A + (1-A).T)/2
        for i in range(A.shape[0]):
            A[i,i] = np.finfo(float).eps
        #print(A)
        #print(A>0.5)
        T = (A>0.5).sum(axis=1)
        self.best_path = T.argsort()[::-1]


def test():
    P = np.array([[0.0, 0.7, 0.81, 0.6, 0.72],
                  [0.15, 0.0, 0.34, 0.6, 0.13],
                  [0.12, 0.8, 0.0, 0.61,0.89],
                  [0.4, 0.51, 0.55 , 0.0, 0.025],
                  [0.52, 0.8, 0.38,0.62,0.0]])
    print(P)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    decoder = CountDecoder(P)
    decoder.run()
    print(decoder.best_path)
if __name__=="__main__":
    test()
