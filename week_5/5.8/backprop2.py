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


def forward(inputs,weights,function=sigmoid,step=-1):
    """Function needed to calculate activation on a particular layer.
    step=-1 calculates all layers, thus provides the output of the network
    step=0 returns the inputs
    any step in between, returns the output vector of that particular (hidden) layer"""

    if step == -1:
        for w in range(len(weights)):
            if w == 0:
                inputs = np.append(1, inputs)
                act = function(np.dot(inputs, weights[w]))
            else:
                act_prime = np.append(1, act)
                act = function(np.dot(act_prime, weights[w]))
        return act

    elif step == 0:
        return inputs

    else:
        for w in range(step):
            if w == 0:
                inputs = np.append(1, inputs)
                act = function(np.dot(inputs, weights[w]))
            else:
                act_prime = np.append(1, act)
                act = function(np.dot(act_prime, weights[w]))
        return act

def backprop(inputs, outputs, weights, function=sigmoid, derivative=derivative_sigmoid, eta=0.01):
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
            error = np.array(derivative(a_now) * (outputs - a_now))  # calculate error on output

        else:

            error = derivative(a_now) * (weights[-i]).dot(error) # calculate error on current layer

        #print(a_prev)
        # print("shape of a a_prev ", a_prev.shape)
        # print("shape of a a_prev appended ", np.expand_dims(np.append(1, a_prev), axis=1).shape)
        # print("shape of error ", error.shape)
        delta = eta * np.expand_dims(np.append(1, a_prev), axis=1) * error # calculate adjustments to weights
        #delta = eta * a_prev * error
        # print("delta ")
        # print(delta)
        # print("delta end")
        deltas.insert(0, delta.T) # store adjustments
        a_now = a_prev # move one layer backwards

    return deltas

## append weg halen arrays shapen naar 3,1 1,3
## bias weghalen bij input en alleen appenden bij forward
##
##
##


#############
# theta = np.array([[
#     [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)],
#     [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)],
#     [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]  ]])



# theta1 = np.array([[
#     [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)],
#     [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)],
#     [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]  ]])


# theta2 = np.array([[[1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]]])
# thetaList = [theta, theta1, theta2]
#thetaList = [theta1, theta2]


############### forward test #####################
# d = np.array([1, 0, 0, 1])
# print("forward ",  forward(d, thetaList, sigmoid, -1))
# print("inputs",    forward(d, thetaList, sigmoid, 0))
# print("1st layer", forward(d, thetaList, sigmoid, 1))
# print("2nd layer", forward(d, thetaList, sigmoid, 2))
# print("3nd layer", forward(d, thetaList, sigmoid, 3))
# print()
# print()
# d = np.array([1, 1, 1, 1])
# print(forward(d, thetaList, sigmoid, -1))
# d = np.array([1, 0, 0, 0])
# print(forward(d, thetaList, sigmoid, -1))
###################################################
# print(np.array([[[1]]]).shape )
# theta = np.array([[1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)],
#                   [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]] ).T
#
# print("shape of theta")
# print(theta.shape)
# print(theta)

theta1 = np.array([[1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)],
                   [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]] ).T
# print("shape of theta")
# print(theta1.shape)
# print(theta1)

theta2 = np.array([[1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]]).T
# print("shape of theta")
# print(theta2.shape)
# print(theta2)

# theta3 = np.array([[ [1, random.uniform(0.2, 0.5), random.uniform(0.2, 0.5)]]])
# print("shape theta3 ", theta3.shape)
thetaList = [ theta1, theta2 ]
trainingSet = [(np.array([[ 1, 1]]), np.array([[0]]).T),
               (np.array([[ 0, 1]]), np.array([[1]]).T),
               (np.array([[ 1, 0]]), np.array([[1]]).T),
               (np.array([[ 0, 0]]), np.array([[0]]).T)]
# print(trainingSet[0][0])

# print("enter functtion print thetaList")
# print(thetaList)
#print("shape eaxample ", trainingSet[0][0].shape)
# x =2
# print(forward(trainingSet[x][0], thetaList, sigmoid, 1))
# deltas = backprop(trainingSet[x][0], trainingSet[x][1], thetaList)
# print(deltas)

for l in range(5000):
    for x in range(len(trainingSet)):
        print("epoch ", x)
        # print(np.array(trainingSet[x][0]), trainingSet[x][1])
        # deltas = backprop(d, [trainingSet[x][1]], thetaList)

        deltas = backprop(trainingSet[x][0], trainingSet[x][1], thetaList)
        #print(deltas)
        for index in range(len(thetaList)):
            print("theta before", thetaList[index])
            thetaList[index] = thetaList[index] + deltas[index]
            print("theta after", thetaList[index])



# input must be np array
# x = trainingSet[0][0]
# print(forward(d, thetaList, sigmoid, -1))

# deltas = backprop(d , np.array([0]), thetaList)
# print(deltas)
# x = 0
# deltas = backprop(trainingSet[x][0], [trainingSet[x][1]], thetaList)
# print(deltas)




# updating weights:
# given an array w of numpy arrays per layer and the deltas calculated by backprop, do
# for index in range(len(w)):
#     w[index] = w[index] + deltas[index]