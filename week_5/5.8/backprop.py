import numpy as np
from math import e
import random

def sigmoid(x):
    """Standard sigmoid; since it relies on ** to do computation, it broadcasts on vectors and matrices"""
    return 1 / (1 - (e**(-x)))

def derivative_sigmoid(x):
    """Expects input x to be already sigmoid-ed"""
    return x * (1 - x)

def tanh(x):
    """Standard tanh; since it relies on ** and * to do computation, it broadcasts on vectors and matrices"""
    return (1 - e ** (-2*x))/ (1 + e ** (-2*x))

def derived_tanh(x):
    """Expects input x to already be tanh-ed."""
    return 1 - tanh(x)

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


def forward(inputs,weights,function=sigmoid,step=-1):
    """Function needed to calculate activation on a particular layer.
    step=-1 calculates all layers, thus provides the output of the network
    step=0 returns the inputs
    any step in between, returns the output vector of that particular (hidden) layer"""
    #print("weights")
    #print(weights)
    if step == -1:
        for w in range(len(weights)):
            if w == 0:
                #np.append(1, inputs)
                act = function(np.dot(weights[w], inputs))
            else:
                act_prime = np.append(1, act)
                act = function(np.dot(weights[w], act_prime))
        return act

    elif step == 0:
        return inputs

    else:
        for w in range(step):
            if w == 0:
                #np.append(1, inputs)
                #print(type(weights[w]))
                act = function(np.dot(weights[w],inputs))
            else:
                act_prime = np.append(1, act)
                act = function(np.dot(weights[w], act_prime))
        return act



def backprop(inputs, outputs, weights, function=sigmoid, derivative=derivative_sigmoid, eta=1):
    """
    Function to calculate deltas matrix based on gradient descent / backpropagation algorithm.
    Deltas matrix represents the changes that are needed to be performed to the weights (per layer) to
    improve the performance of the neural net.
    :param inputs: (numpy) array representing the input vector.
    :param outputs:  (numpy) array representing the output vector.
    :param weights:  list of numpy arrays (matrices) that represent the weights per layer.
    :param function: activation function to be used, e.g. sigmoid or tanh
    :param derivative: derivative of activation function to be used.
    :param learnrate: rate of learning.
    :return: list of numpy arrays representing the delta per weight per layer.
    """
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    deltas = []
    layers = len(weights) # set current layer to output layer
    a_now = forward(inputs, weights, function, layers) # activation on current layer
    for i in range(0, layers):
        a_prev = forward(inputs, weights, function, layers-i-1) # calculate activation of previous layer
        if i == 0:
            error = np.array(derivative(a_now) * (outputs - a_now)).T  # calculate error on output
        else:
            error = np.expand_dims(derivative(a_now), axis=1) * weights[-i].T.dot(error)[1:] # calculate error on current layer
        delta = eta * np.expand_dims(a_prev, axis=1) * error.T # calculate adjustments to weights
        #print("calculated delta ", delta)
        #print("aprev again ", a_prev)
        deltas.insert(0, delta.T) # store adjustments
        a_now = a_prev # move one layer backwards

    return deltas



trainingSet = [ (np.array([1,1,0,0,0]), 0) ,(np.array([1,0,1,0,0]), 0) ,(np.array([1,0,0,1,0]), 0) ,(np.array([1,0,0,0,0]), 1)]
#print(trainingSet[0][0])

d = np.array([1, 0 ,0, 1])
theta = np.array([[1, -15, -15, -15], [1, -15, -15, -15], [1, -15, -15, -15]])
theta1 = np.array([[1, -15, -15, -15], [1 ,-15, -15, -15]])
theta2 = np.array([1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)])
thetaList = [theta2]
#input must be np array
#x = trainingSet[0][0]
#print(forward(d, thetaList, sigmoid, -1))

#deltas = backprop(d , np.array([0]), thetaList)
#print(deltas)
#x = 0
#deltas = backprop(trainingSet[x][0], [trainingSet[x][1]], thetaList)
#print(deltas)
for l in range(5000):
    for x in range(len(trainingSet)):
        print("epoch " , x)
        #print(np.array(trainingSet[x][0]), trainingSet[x][1])
        #deltas = backprop(d, [trainingSet[x][1]], thetaList)

        deltas = backprop(trainingSet[x][0], [trainingSet[x][1]], thetaList)
        #print(deltas)
        for index in range(len(thetaList)):

            print("theta before",thetaList[index])
            thetaList[index] = thetaList[index] + deltas[index]
            print("theta after", thetaList[index])


d = np.array([1, 0 ,0 ,0, 1])
print(forward(d, thetaList, sigmoid, -1))
d = np.array([1, 1 ,1, 1,1])
print(forward(d, thetaList, sigmoid, -1))
d = np.array([1, 0 ,0, 0 ,0])
print(forward(d, thetaList, sigmoid, -1))
# updating weights:
# given an array w of numpy arrays per layer and the deltas calculated by backprop, do
# for index in range(len(w)):
#     w[index] = w[index] + deltas[index]

