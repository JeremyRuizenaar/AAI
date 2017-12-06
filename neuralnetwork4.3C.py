from numpy import *
import random

class Neuron:

    def __init__(self, weights, bias):
        self.bias = bias
        self.weights = weights
        self.activation = 0

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

        learnRate = 0.01
        for x in range(0, len(self.weights)):

            errorSig = ( desired - actual ) *  actual * ( 1 - actual )
            tmp = float(learnRate * errorSig *   data[x])
            self.weights[x] += tmp

        print("new weights = " ,self.weights)
        print("-" * 50)
        print()

    def updateWeights(self, weights):
        self.weights = weights[:]

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def act(self, input):
        res = self.sigmoid(sum(self.weightMultiplier(input)) + self.bias)
        self.activation = res#bias might not work
        return  res

    def getAct(self):
        return self.activation

class Network:

    def __init__(self, it):
        self.iterations = it

        # and gates
        self.AndGate1 = Neuron([random.uniform(0, 1), random.uniform(0, 1)], 2)
        self.AndGate2 = Neuron([random.uniform(0, 1), random.uniform(0, 1)], 2)
        # or gatess
        self.OrGate1 = Neuron([random.uniform(0, 1), random.uniform(0, 1)], 2)
        self.OrGate2 = Neuron([random.uniform(0, 1), random.uniform(0, 1)], 2)
        # inverters
        self.inverter1 = Neuron([random.uniform(0, 1)], 1)
        self.inverter2 = Neuron([random.uniform(0, 1)], 1)

        # exor, and
        self.neuronList = [[self.AndGate1, "and1 "], [self.AndGate2, "and2 "], [self.OrGate1, "or1 "], [self.OrGate2, "or2 "], [self.inverter1, "inv1 "], [self.inverter2, "inv2"] ]

        self.trainingSet = [[0, 0], [0, 1], [1, 0,], [1, 1]]
        self.trainingAnswer = [[0,0 ],[1,0],[1,0],[0,1]]

    def calculateDistance(self, a, b):
        # calcululate difference between two classes represented by two lists
        result = 0
        for ele in range(0, len(a)):
            delta = (b[ele] - a[ele])
            result += delta * delta
        return sqrt(result)

    def dist(self, a):
        tmp = 0
        for i in a:
            tmp += i*i
        res = sqrt(tmp )
        return res

    def showWeights(self):
        for neuron in self.neuronList:
            print(neuron[1] ,neuron[0].getWeights())

    def train(self):
        print("training")
        print()

        for x in range(0, self.iterations):
            print("-"*25, x , "-"*25)
            i = 0
            cumError = 0

            for data in self.trainingSet:
                print()
                #self.showWeights()
                print()
                print("desired result = ", self.trainingAnswer[i])

                result = self.think(data)
                print("result = ", result)
                #print("result of distance = ", (self.calculateDistance( self.trainingAnswer[i] , result) ) )
                cumError += (self.calculateDistance(self.trainingAnswer[i], result))
                #cumError += math.pow((self.calculateDistance( self.trainingAnswer[i] , result) ), 2)
                print("cumError() = ", cumError)

                activationA = data[0]
                activationB = data[1]
                activationC = self.inverter1.getAct()
                activationD = self.inverter2.getAct()
                activationE = self.OrGate1.getAct()
                activationF = self.OrGate2.getAct()
                activationG = self.AndGate1.getAct()
                activationH = self.AndGate2.getAct()

                errorH = (1 - activationH) * (self.trainingAnswer[i][1] - self.AndGate2.getAct())
                errorG = (1 - activationG) * (self.trainingAnswer[i][0] - self.AndGate1.getAct())
                errorF = (1 - activationF) *  self.AndGate1.getWeights()[1] * errorG
                errorE = (1 - activationE) *  self.AndGate1.getWeights()[0] * errorG
                errorD = (1 - activationD) *  self.OrGate2.getWeights()[1]  * errorF
                errorC = (1 - activationC) *  self.OrGate2.getWeights()[0]  * errorF

                lR = 0.1
                w1 = self.AndGate1.getWeights()[0] + lR * activationE * errorG #eg
                w2 = self.AndGate1.getWeights()[1] + lR  * activationF *  errorG #fg
                w3 = self.OrGate1.getWeights()[0] + lR * activationA * errorE #ae
                w4 = self.OrGate1.getWeights()[1] + lR  * activationB * errorE#be
                w5 = self.OrGate2.getWeights()[0] + lR  * activationC * errorF #cf
                w6 = self.OrGate2.getWeights()[1] + lR  * activationD * errorF #df
                w7 = self.inverter1.getWeights()[0] + lR * activationA * errorC #ac
                w8 = self.inverter2.getWeights()[0] + lR  * activationB * errorD #bd
                w9 = self.AndGate2.getWeights()[0] + lR * activationA * errorH #ah
                w10 = self.AndGate2.getWeights()[1] + lR * activationB * errorH #bh

                self.AndGate1.updateWeights([w1, w2])
                self.OrGate1.updateWeights([w3, w4])
                self.OrGate2.updateWeights([w5, w6])
                self.inverter1.updateWeights([w7])
                self.inverter2.updateWeights([w8])
                self.AndGate2.updateWeights([w9, w10])

                i += 1

            if cumError < 0.1:
                print("cum error is low")
                return True

            print("cumError(total) = ", cumError)

        print("done traing")

    def think(self,input):
        print("thinking about ", input)
        result = [ self.AndGate1.act([self.OrGate1.act([input[0], input[1]]), self.OrGate2.act([self.inverter1.act([input[0]]), self.inverter2.act([input[1]])])]) , self.AndGate2.act([input[0], input[1]]) ]

        return result
#
#  ------------
#            ---(or)e
#  ------------
#  ---(inv)c         ------ (and)g
#          ---- (or)f
#  ---(inv)d
#-----------------------
#                       ----(and)h
#-----------------------
a = Network(100000000)

a.showWeights()
a.train()
print(a.think([0,0]))#
print()
print(a.think([1,1]))
print()
print(a.think([0,1]))
print()
print(a.think([1,0]))
print()

