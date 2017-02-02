import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
                
y = np.array([[0],
			  [1],
			  [1],
			  [0]])

