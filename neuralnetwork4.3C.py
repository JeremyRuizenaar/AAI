from numpy import *
import random

class Neuron:

    def __init__(self, weights, bias):
        self.bias = bias
        self.weights = weights
        self.activation = 0
        self.summedInput = 0

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
        res = self.sigmoid(sum(self.weightMultiplier(input)))
        self.summedInput = sum(self.weightMultiplier(input)) + self.bias
        #print(type(self.summedInput))
        self.activation = res#bias might not work
        return  res

    def getAct(self):
        return self.activation

    def getSum(self):
        return self.summedInput

class Network:

    def __init__(self, it):
        self.iterations = it
        self.nOUT1 = Neuron([random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 1)], 1)
        self.nHID1 = Neuron([random.uniform(0.1, 1), random.uniform(0.1, 1)], 1)
        self.nHID2 = Neuron([random.uniform(0.1, 1), random.uniform(0.1, 1)], 1)
        self.nOUT2 = Neuron([random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 1)], 1)
        self.nHID3 = Neuron([random.uniform(0.1, 1), random.uniform(0.1, 1)], 1)
        # exor, and
        self.neuronList = [[self.nOUT1 , "n1 "] , [self.nHID1, "n2 "], [self.nHID2 , "n3 "], [self.nOUT2 , "n4 "], [self.nHID3 , "n5 "]]
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
                #self.showWeights()
                #print()
                #print("desired result = ", self.trainingAnswer[i])

                result = self.think(data)
                #print("result = ", result)
                distError = math.pow(self.dist(self.trainingAnswer[i]) - self.dist(result), 2)
                cumError += distError
                #print("cumError() = ", cumError)

                activationA = data[0]
                activationB = data[1]

                lR = 0.1

                errorOUT =    (self.trainingAnswer[i][0] - self.nOUT1.getAct() ) * ( 1 - self.nOUT1.getAct() ) * self.nOUT1.getAct()
                errorOUT2 =   (self.trainingAnswer[i][1] - self.nOUT2.getAct() ) * (1 - self.nOUT2.getAct()  ) * self.nOUT2.getAct()

                summError1 =  (errorOUT * self.nOUT1.getWeights()[0] *  (1 - self.nHID1.getAct()) * self.nHID1.getAct() ) + (errorOUT2 * self.nOUT1.getWeights()[0] *  (1 - self.nHID1.getAct()) * self.nHID1.getAct() )
                summError2 =  (errorOUT * self.nOUT1.getWeights()[1] *  (1 - self.nHID2.getAct()) * self.nHID2.getAct() ) + (errorOUT2 * self.nOUT1.getWeights()[1] *  (1 - self.nHID2.getAct()) * self.nHID2.getAct() )
                summError3 =  (errorOUT * self.nOUT1.getWeights()[2] *  (1 - self.nHID3.getAct()) * self.nHID3.getAct()) + (errorOUT2 * self.nOUT1.getWeights()[2] *  (1 - self.nHID3.getAct())  * self.nHID3.getAct())


                w1 = self.nOUT1.getWeights()[0] + lR * (self.nHID1.getAct() * errorOUT)
                w2 = self.nOUT1.getWeights()[1] + lR * (self.nHID2.getAct() * errorOUT)
                w3 = self.nOUT1.getWeights()[2] + lR * (self.nHID3.getAct() * errorOUT)

                w4 = self.nHID1.getWeights()[0] + lR * (activationA * summError1)
                w5 = self.nHID1.getWeights()[1] + lR * (activationB * summError1)

                w6 = self.nHID2.getWeights()[0] + lR * (activationA * summError2)
                w7 = self.nHID2.getWeights()[1] + lR * (activationB * summError2)

                w11= self.nHID3.getWeights()[0] + lR * (activationA * summError3)
                w12= self.nHID3.getWeights()[1] + lR * (activationA * summError3)

                w8 = self.nOUT2.getWeights()[0] + lR * (self.nHID1.getAct() * errorOUT2)
                w9 = self.nOUT2.getWeights()[1] + lR * (self.nHID2.getAct() * errorOUT2)
                w10= self.nOUT2.getWeights()[2] + lR * (self.nHID3.getAct() * errorOUT2)


                self.nOUT1.updateWeights([w1, w2, w3])
                self.nOUT2.updateWeights([w8, w9, w10])
                self.nHID1.updateWeights([w4, w5])
                self.nHID2.updateWeights([w6, w7])
                #self.nHID2.updateWeights([w9, w10])
                self.nHID3.updateWeights([w11, w12])

                i += 1


                        #0.03 for lr 0.01
            if cumError < 0.01:
                print("cum error is low")
                return True

            print("cumError(total) = ", cumError)

        print("done traing")

    def think(self,input):

        result = [self.nOUT1.act([self.nHID1.act([input[0],input[1]]), self.nHID2.act([input[0],input[1]]), self.nHID3.act([input[0],input[1]])  ] ) , self.nOUT2.act([self.nHID1.act([input[0],input[1]]), self.nHID2.act([input[0],input[1]]), self.nHID3.act([input[0],input[1]])  ])]

        return result



a = Network(3000000)

a.showWeights()
a.train()
a.showWeights()
print(a.think([0,0]))#
print()
print(a.think([1,1]))
print()
print(a.think([0,1]))
print()
print(a.think([1,0]))
print()

