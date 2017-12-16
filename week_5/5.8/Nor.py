import numpy as np
import math

def sigmoid(x):
     return 1 / (1 + math.e**(-x))

def predict(x, thetaList):
    x = np.array(x)
    input = True
    a = 0
    thetaCount = 0

    for theta in thetaList:
        theta = np.array(theta)
        #print("processsing theta ", thetaCount )
        thetaCount += 1
        if input == True:
            act = sigmoid(np.dot(theta, x))
            input = False
            if len(thetaList) == 1:
                return act
        else:
            act_prime = np.append(1, act)
            act = sigmoid(np.dot(theta, act_prime))
    return act



#x = np.array([10, 1 ,0, 0])
theta = np.array([1, -15, -15, -15])
#theta1 = np.array([1, -15])
thetaList = [theta] #, theta1]



for q in range(0, 2):
    for r in range(0, 2):
        for s in range(0, 2):
            x = [10, q ,r, s]
            print(x[1:],  " --> ", predict(x,thetaList))


