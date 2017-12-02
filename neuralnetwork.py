class NeuralNetwork:

    def __init__(self):
        self.neurons = []


    def activate(self, input):
        pass

class Neuron:

    def __init__(self, weights):
        self.weights = weights


    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result



    def activate(self, input):
        return sum(self.weightMultiplier(input))
        #return sum(self.weightMultiplier(input)) - self.treshold >= 0



class BNeuron:

    def __init__(self, weights, treshold):
        self.treshold = treshold
        self.weights = weights


    def weightMultiplier(self, input):
        ele = 0
        result = []
        for each in self.weights:
            result.append(input[ele] * each)
            ele += 1

        return result



    def act(self, input):
        if sum(self.weightMultiplier(input)) - self.treshold >= 0:
            return 1
        else:
            return 0





#adder

# and gates
AndGate1 = BNeuron([1,1] , 2)
AndGate2 = BNeuron([1,1] , 2)
# or gatess
OrGate1 = BNeuron([1,1], 1)
OrGate2 = BNeuron([1,1], 1)
# inverters

inverter1 = BNeuron([-2] , -1)
inverter2 = BNeuron([-2] , -1)

# exor, and
a = 1
b = 1
print( AndGate1.act([OrGate1.act([a, b]) , OrGate2.act([inverter1.act([a])  , inverter2.act([b])])]), AndGate2.act([a, b]) )




# norgate out of perceptron
norGate = BNeuron([-1,-1,-1] , 0)
#print(norGate.activate([0,0,0]))

