from numpy import *
import random

class Neuron:

    def __init__(self, weights, treshold):
        self.treshold = treshold
        self.weights = weights

    def getWeights(self):
        return self.weights

    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result

    def update(self, data ,inputSum, actual, desired):
        print("updating neuron")
        print("inputSum = ", inputSum)
        print("actual = ", actual)
        print("desired = ", desired)
        learnRate = 0.10

        for x in range(0, len(self.weights)):
            print("delta weight = ", (learnRate * data[x] * (1 / self.sigmoid(inputSum)) * (desired - actual)))
            self.weights[x] += float(learnRate * data[x] * (1 / self.sigmoid(inputSum)) * (desired - actual))

        print("new weights = " ,self.weights)
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def act(self, input):
        print("sigmoid function test = ",self.sigmoid(0))
        print("act() invoked")
        print("act input = ", input)
        print("res of sumweights = ", sum(self.weightMultiplier(input)))
        res = self.sigmoid(sum(self.weightMultiplier(input)))
        print("result of think = " , res)
        return  res

class Network:

    def __init__(self):
        self.norGate = Neuron([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)], 0)
        self.trainingSet = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        self.trainingAnswer = [1, 0, 0, 0, 0, 0, 0, 0]

    def calculateDistance(self, a, b):
        # calcululate difference between two classes represented by two lists
        result = 0
        for ele in range(0, len(a)):
            delta = (b[ele] - a[ele])
            result += delta * delta
        return np.sqrt(result)

    def showWeights(self):
        print(self.norGate.getWeights())

    def train(self):
        print("training")

        for x in range(0,10):
            i = 0
            for data in self.trainingSet:
                print("answer = ", self.trainingAnswer[i])
                result = float(self.think(data))
                if( self.trainingAnswer[i]) - result < 0.10:
                    print("accurate")
                    #return

                self.norGate.update( data , sum(data) , result , self.trainingAnswer[i] )
                i += 1




        print("done traing")

    def think(self,input):
        print("thinking")
        return self.norGate.act(input)




a = Network()
a.showWeights()
a.train()
print(a.think([0,0,0]))
print(a.think([0,0,1]))
print(a.think([0,0,0]))
print(a.think([0,0,0]))