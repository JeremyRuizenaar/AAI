from numpy import *
import random
import numpy as np


data   = np.genfromtxt('irisDataset.csv' , delimiter= ',', usecols=[0,1,2,3])
dataLabels = []
for x in range(0,45):
    dataLabels.append([1, 0, 0,])
for x in range(45,90):
    dataLabels.append([0, 1, 0,])
for x in range(90,135):
    dataLabels.append([0, 0, 1,])

test   = np.genfromtxt('validationIris.csv' , delimiter= ',', usecols=[0,1,2,3])
testLabels = []
for x in range(0,5):
    testLabels.append([1, 0, 0,])
for x in range(5,10):
    testLabels.append([0, 1, 0,])
for x in range(10,15):
    testLabels.append([0, 0, 1,])


# for x in range(0, 15):
#     #tmp = []
#     get = random.randint(20, 80)
#     testSet.append([data[get],dataLabels[get] ])
#     data = np.delete(data, get)
#     print("len of data ", len(data) , "after ", x)
#     del dataLabels[get]
#     #dataLabels = np.delete(dataLabels, get)
#
#


# print("data = ", data)
# print("datalabels = ", dataLabels)
# print("test = ", test)
# print("testset = ", testLabels)
# print("len of data and labels and test and testset = ", len(data), len(dataLabels), len(test), len(testLabels))

class Neuron:

    def __init__(self, weights, id):
        self.id = id
        self.weights = [random.uniform(0.2, 0.7) for x in range(0 , weights)]
        self.activation = 0
        self.summedInput = 0
        self.error = 0

    def getWeights(self):
        return self.weights

    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result

    def updateWeights(self, weights):
        self.weights = weights[:]

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def act(self, input):
        res = self.sigmoid(sum(self.weightMultiplier(input)))
        self.summedInput = sum(self.weightMultiplier(input)) #+ self.bias
        self.activation = res
        return  res

    def getAct(self):
        return self.activation

    def getSum(self):
        return self.summedInput

    def sDerivative(self, x):
        return x * (1 - x)

    def getId(self):
        return self.id

    def getError(self):
        return self.error

    def calcErrorOutput(self, res, answer):
        self.error = (answer[self.id] - self.activation ) * self.sDerivative(self.activation)

    def calcErrorHidden(self, nodes):
        self.error = 0
        for node in nodes:
            self.error += (node.getError() * node.getWeights()[self.id] * self.sDerivative(self.activation) )

    def setWeightsOuterAndHiddenLayer(self, rate, nodes):
        for node in nodes:
            self.weights[node.getId()] += rate * (node.getAct() * self.error)

    def setWeightsInputLayer(self, rate, input):
        i = 0
        for val in input:
            self.weights[i] += rate * (val * self.error)
            i+= 1

class Network:

    def __init__(self, it):
        self.iterations = it
        self.learnRate = 0.5
        self.errorLimit = 0.1

        self.nOUT1 = Neuron(6, 0)
        self.nOUT2 = Neuron(6, 1)
        self.nOUT3 = Neuron(6, 2)

        self.nHID1 = Neuron(6, 0)
        self.nHID2 = Neuron(6, 1)
        self.nHID3 = Neuron(6, 2)
        self.nHID4 = Neuron(6, 3)
        self.nHID5 = Neuron(6, 4)
        self.nHID6 = Neuron(6, 5)

        self.nHID7 = Neuron(4, 0)
        self.nHID8 = Neuron(4, 1)
        self.nHID9 = Neuron(4, 2)
        self.nHID10 = Neuron(4, 3)
        self.nHID11 = Neuron(4, 4)
        self.nHID12 = Neuron(4, 5)

        self.nHID13 = Neuron(6, 0)
        self.nHID14 = Neuron(6, 1)
        self.nHID15 = Neuron(6, 2)
        self.nHID16 = Neuron(6, 3)
        self.nHID17 = Neuron(6, 4)
        self.nHID18 = Neuron(6, 5)




        self.hiddenLayers = [[self.nHID7, self.nHID8, self.nHID9, self.nHID10, self.nHID11, self.nHID12],
                             [self.nHID1, self.nHID2, self.nHID3, self.nHID4, self.nHID5, self.nHID6],
                             #[self.nHID13, self.nHID14, self.nHID15, self.nHID16, self.nHID17, self.nHID18],
                             [self.nOUT1, self.nOUT2, self.nOUT3]
                             ]


        self.neuronList = [[self.nOUT1 , "out1 "] , [self.nOUT2 , "out2 "], [self.nOUT3 , "out3 "],
                           [self.nHID1, "hid1 "], [self.nHID2 , "hid2 "],  [self.nHID3 , "hid3 "],  [self.nHID4 , "hid4 "]]

        self.trainingSet = data
        self.trainingAnswer = dataLabels

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

    def sDerivative(self, x):
        return x * (1 - x)

    def forwardPropagateFirstLayer(self,  input):
        for node in self.hiddenLayers[0]:
            node.act([ele for ele in input])

    def forwardPropagateLayer(self, layer, prevLayer):
        for node in layer:
            node.act( [prev.activation for prev in prevLayer  ]  )

    def forwardPropagateLayers(self):
        for i in range(0, len(self.hiddenLayers)-1):
            self.forwardPropagateLayer(self.hiddenLayers[i+1], self.hiddenLayers[i])

    def forwardPropagateNetwork(self, input):

        self.forwardPropagateFirstLayer(input)
        self.forwardPropagateLayers()

        return [n.activation for n in self.hiddenLayers[-1]]

    def backPropagateOutputLayer(self,  res, ans):
        for node in self.hiddenLayers[-1]:
            node.calcErrorOutput(res, ans)

    def backPropagateLayer(self, layer , prevLayer):
        for node in layer:
            node.calcErrorHidden(prevLayer)

    def backPropagateLayers(self):
        for i in range(len(self.hiddenLayers) -1 , 0 , -1):
            self.backPropagateLayer(self.hiddenLayers[i-1], self.hiddenLayers[i])

    def backPropagateNetwork(self, res, answer):

        self.backPropagateOutputLayer(res, answer)
        self.backPropagateLayers()

    def updateWeights(self, rate, input):
        for node in self.hiddenLayers[0]:
            node.setWeightsInputLayer(rate, input)

        for x in range(1, len(self.hiddenLayers)):
            for node in self.hiddenLayers[x]:
                node.setWeightsOuterAndHiddenLayer(rate, self.hiddenLayers[x-1] )



    def train(self):
        print("training")
        print()

        for x in range(0, self.iterations):
            print("-"*25, " epoch ", x , "-"*25)
            # self.showWeights()
            i = 0
            cumError = 0

            for data in self.trainingSet:
                result = self.forwardPropagateNetwork(data)
                self.backPropagateNetwork(result, self.trainingAnswer[i])
                self.updateWeights(self.learnRate , data)
                distError = math.pow(self.dist(self.trainingAnswer[i]) - self.dist(result), 2)
                cumError += distError

                i += 1

            if cumError < self.errorLimit:
                return True

            print("cumulativeError(total) = ", cumError)
        print("done traing")

    def think(self,input):
        return self.forwardPropagateNetwork(input)




a = Network(3000000)
#
a.showWeights()
a.train()
a.showWeights()

for x in range(0, 15):
    result = a.think(test[x])
    print("aproximated result of ", test[x], " = " ,result ,)

    for x in range(0, len(result)):
        if result[x] <= 0.5:
            result[x] = 0
        else:
            result[x] = 1

    print(" rounded result = ", result)
    print( " correct result = ", testLabels[x] ) #


# cumError(total) =  0.0800013515723169
# -------------------------  epoch  2129 -------------------------
# cum error is low
# out1  [-5.9494578834065903, 2.5900144362017232, -10.633899828977329, 2.6917882166645417]
# out2  [-8.5164167028022213, -2.628542077364842, 10.398866070779281, -2.4986480384170182]
# out3  [8.8396962166106849, -4.0580600692192199, 2.1542034314462564, -3.6880838519548611]
# hid1  [-15.581554155020266, -29.145270378434375, 28.637032748129819, 25.924373049427547]
# hid2  [0.92500137339579569, 0.70238817591961389, 0.51036135983887088, 0.62313231385027867]
# hid3  [-0.94118821339060954, -2.4268049763502342, 4.0329173160717282, 1.9675707281328834]
# hid4  [0.75590714969988215, 0.85304571209768965, 0.69651138507512156, 0.51019782273087733]
# result of  [ 4.4  2.9  1.4  0.2]  =  [0.99459264307128448, 0.0062962067297559583, 0.00044105950632721599]  correct result =  [1, 0, 0]
# result of  [ 5.1  3.3  1.7  0.5]  =  [0.99455110907319844, 0.0063436500591252934, 0.00043966176131318062]  correct result =  [1, 0, 0]
# result of  [ 5.1  3.4  1.5  0.2]  =  [0.99486030247520463, 0.0059916983160238453, 0.00043466018433652875]  correct result =  [1, 0, 0]
# result of  [ 5.4  3.4  1.5  0.4]  =  [0.99485460138117687, 0.0059982108103686441, 0.00043444573695830221]  correct result =  [1, 0, 0]
# result of  [ 4.8  3.   1.4  0.3]  =  [0.99471617670715962, 0.0061557099314638259, 0.00043789795846137204]  correct result =  [1, 0, 0]
# result of  [ 4.9  2.4  3.3  1. ]  =  [0.0051187815641468301, 0.99446492711109602, 0.0036574380132527767]  correct result =  [0, 1, 0]
# result of  [ 6.2  2.2  4.5  1.5]  =  [1.2423462127866593e-05, 0.037776426563064548, 0.96231336954249624]  correct result =  [0, 1, 0]
# result of  [ 6.4  2.9  4.3  1.3]  =  [0.004769185935911799, 0.99483476939125981, 0.0037059093010943459]  correct result =  [0, 1, 0]
# result of  [ 5.8  2.7  3.9  1.2]  =  [0.0048302521352365946, 0.99477012390566277, 0.0036970424010837515]  correct result =  [0, 1, 0]
# result of  [ 6.1  3.   4.6  1.4]  =  [0.0047280538227298481, 0.99487828241234511, 0.0037124153246295062]  correct result =  [0, 1, 0]
# result of  [ 7.6  3.   6.6  2.1]  =  [1.2353496286522679e-05, 0.037516473050684958, 0.96259263754567559]  correct result =  [0, 0, 1]
# result of  [ 6.3  2.7  4.9  1.8]  =  [1.2358351184385596e-05, 0.037523234156061465, 0.96258135335374351]  correct result =  [0, 0, 1]
# result of  [ 7.9  3.8  6.4  2. ]  =  [4.5095953031438754e-05, 0.19921390969769423, 0.78982151739005302]  correct result =  [0, 0, 1]
# result of  [ 6.7  3.1  5.6  2.4]  =  [1.2353546550158371e-05, 0.037516327417165235, 0.96259306302312286]  correct result =  [0, 0, 1]
# result of  [ 6.7  3.3  5.7  2.5]  =  [1.2353570284622396e-05, 0.037516260530399963, 0.96259288811686372]  correct result =  [0, 0, 1]
#
# Process finished with exit code 0
#
# out1  [6.6407932100174598, -0.076393652551962746, -4.5056081687017571, -24.20468993273883, 0.89742687842254376, 7.1448100897852482]
# out2  [-5.9237534081575065, -1.2444903745406326, -5.0694520631699538, 5.3779341899774398, -1.7241085719062663, -2.9449498099382438]
# out3  [-4.1635748066176506, -1.6002838479678791, 5.3064485951939222, -5.284902877399948, -2.1346194867519523, -4.8033588114313899]
# hid1  [-1.3122217499000626, 1.1546289820865825, 1.1273612962353692, 0.69574120415462481, -8.1005927292790609, 1.2034326917967537]
# hid2  [-0.82285736399783338, -0.66747225393025766, -0.88078230583810335, -0.75252506738589275, -2.4930448363416122, -0.6296902651529932]
# hid3  [-7.2846105568857551, -0.43313742070789613, 2.2870252241164248, 7.3448041618547233, -2.1055907867279755, -0.51524333164584313]
# hid4  [2.9503085494734993, 0.14199738296937253, -1.8627556721738339, -8.5076895027202628, 8.9737021890562865, 0.25967269510360208]
# result of  [ 4.4  2.9  1.4  0.2]  =  [0.99525956511227198, 4.8515175093467261e-06, 0.0017879737745614256]  correct result =  [1, 0, 0]
# result of  [ 5.1  3.3  1.7  0.5]  =  [0.99518694640599081, 4.8505802128608823e-06, 0.0018041470849734572]  correct result =  [1, 0, 0]
# result of  [ 5.1  3.4  1.5  0.2]  =  [0.99550413638820767, 4.7532805270072286e-06, 0.0018023195124791324]  correct result =  [1, 0, 0]
# result of  [ 5.4  3.4  1.5  0.4]  =  [0.99552592639806403, 4.7357391674623916e-06, 0.0018101478652540958]  correct result =  [1, 0, 0]
# result of  [ 4.8  3.   1.4  0.3]  =  [0.99536328422553666, 4.8085745676538626e-06, 0.0017951471476942698]  correct result =  [1, 0, 0]
# result of  [ 4.9  2.4  3.3  1. ]  =  [8.5889111534246047e-12, 0.97462694850299891, 0.023578552046782827]  correct result =  [0, 1, 0]
# result of  [ 6.2  2.2  4.5  1.5]  =  [1.183954591006558e-05, 0.025100910653954134, 0.97325595688578581]  correct result =  [0, 1, 0]
# result of  [ 6.4  2.9  4.3  1.3]  =  [8.5989907304179841e-12, 0.97465253671791519, 0.023549202414631789]  correct result =  [0, 1, 0]
# result of  [ 5.8  2.7  3.9  1.2]  =  [8.5970783146369704e-12, 0.9746482612771582, 0.023554608987549849]  correct result =  [0, 1, 0]
# result of  [ 6.1  3.   4.6  1.4]  =  [8.5991474230784961e-12, 0.97465289668137445, 0.023548761007165894]  correct result =  [0, 1, 0]
# result of  [ 7.6  3.   6.6  2.1]  =  [2.0205261985569571e-05, 0.022318791613241597, 0.97612497036518497]  correct result =  [0, 0, 1]
# result of  [ 6.3  2.7  4.9  1.8]  =  [1.9992824948142296e-05, 0.022371218096519364, 0.97607219448320437]  correct result =  [0, 0, 1]
# result of  [ 7.9  3.8  6.4  2. ]  =  [1.2479222373681517e-11, 0.37735224045585347, 0.62784394396433396]  correct result =  [0, 0, 1]
# result of  [ 6.7  3.1  5.6  2.4]  =  [2.0205970845129529e-05, 0.02231872845155692, 0.97612537187536152]  correct result =  [0, 0, 1]
# result of  [ 6.7  3.3  5.7  2.5]  =  [2.0205742213212145e-05, 0.022318749432722128, 0.97612524363991149]  correct result =  [0, 0, 1]