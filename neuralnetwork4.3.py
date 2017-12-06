from numpy import *
import random

class Neuron:

    def __init__(self, weights, bias):
        self.bias = bias
        self.weights = weights
        self.oldWeights = [0 for x in range(0, len(self.weights))]


    def getWeights(self):
        return self.weights

    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result

    def subtractList(self, a , b):
        result = []
        i = 0
        for x in range(0, len(a)):
            #print(a[i])
            #print(b[i])
            res =  (float(a[i]) - float(b[i]))
            #print("result of subtract list = ", res)
            result.append(res)
            i += 1
        return result

    def update(self, data ,inputSum, actual, desired):
        print("updating neuron")

        self.oldWeights = self.weights[:]
        learnRate = 0.01

        for x in range(0, len(self.weights)):

            errorSig = ( desired - actual ) *  actual * ( 1 - actual )
            #print("errorSig = ", errorSig)
            tmp = float(learnRate * errorSig *   data[x])
            #print("tmp = ", tmp)

            self.weights[x] += tmp


        # a = self.subtractList(self.oldWeights, self.weights)
        # # print(a)
        # if sum(a) <= 0.01:
        #     print("converged")
        #     # return True

        print("new weights = " ,self.weights)
        print("-" * 50)
        print()

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def act(self, input):
        #bias = 3
        res = self.sigmoid(sum(self.weightMultiplier(input)) + self.bias)
        return  res

class Network:

    def __init__(self, it):
        #self.norGate = Neuron([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 0)
        self.iterations = it
        # self.trainingSet = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],[1, 0, 1], [1, 1, 0] ,[1,1, 1]]
        # self.trainingAnswer = [1, 0, 0, 0, 0, 0, 0, 0]

        self.trainingSet = [[0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0], [0,0, 1, 0], [1, 0, 0, 0]]
        self.trainingAnswer = [0, 1, 0, 0, 0]
        #self.trainingAnswer = [1, 0, 1, 1,1]


        self.norGate = Neuron([random.uniform(0, 1) for x in range(0, len(self.trainingSet[0]))], 3)

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
        print()

        for x in range(0, self.iterations):
            print("-"*25, x , "-"*25)
            i = 0
            cumError = 0

            for data in self.trainingSet:
                print("desired result = ", self.trainingAnswer[i])

                result = float(self.think(data))
                print("result = ", result)

                cumError += math.pow((self.trainingAnswer[i] - result), 2)
                print("cumError() = ", cumError)

                if self.norGate.update( data , sum(data), result , self.trainingAnswer[i] ) == True:
                    print("done traing")
                    return
                i += 1

            if cumError < 0.01:
                print("cum error is low")
                return True

            print("cumError(total) = ", cumError)



        print("done traing")

    def think(self,input):
        print("thinking about ", input)
        return self.norGate.act(input)




a = Network(1000000)
a.showWeights()
a.train()
print(a.think([0,0,0,0]))#
print()
print(a.think([0,0,1,1]))
print()
print(a.think([0,1,1,0]))
print()
print(a.think([1,0,0,1]))
print()
print(a.think([1,1,1,1]))#
print()
print(a.think([0,1,1,0]))
print()
print(a.think([1,1,0,1]))
print()
print(a.think([1,0,1,1]))
a.showWeights()