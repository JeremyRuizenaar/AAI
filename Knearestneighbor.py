import numpy as np
import random

data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7])
for ele in data:
    if ele[6] == -1:
        ele[6] = 0
    if ele[4] == -1:
        ele[0] = 0
dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])

labels = []
for label in dates:
    if label < 20000301:
        labels.append('winter')
    elif 20000301 <= label < 20000601:
        labels.append('lente')
    elif 20000601 <= label < 20000901:
        labels.append('zomer')
    elif 20000901 <= label < 20001201:
        labels.append('herfst')
    else: # from 01−12 to end of year
        labels.append('winter')




validationdata = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7])
for ele in validationdata:
    if ele[6] == -1:
        ele[6] = 0
    if ele[4] == -1:
        ele[0] = 0
validationdates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
validationlabels = []

for label in validationdates:
    if label < 20010301:
        validationlabels.append('winter')
    elif 20010301 <= label < 20010601:
        validationlabels.append('lente')
    elif 20010601 <= label < 20010901:
        validationlabels.append('zomer')
    elif 20010901 <= label < 20011201:
        validationlabels.append('herfst')
    else: # from 01−12 to end of year
        validationlabels.append('winter')



def getDistancesToTrainingData(data,  trainingdata):
    if len(data) == 0:
        return None
    if len(trainingdata) == 0:
        return None
    distances = []
    # for each data element in the trainingdata add the distance and the label to the distanceslist
    for index in range(len(trainingdata[0])):
        distances.append((calculateDistance(data, trainingdata[0][index]),  trainingdata[1][index]))
    return distances

def calculateDistance(data, training):
    # calcululate difference between two classes represented by two lists
    result = 0
    for ele in range(0, len(data)):
        delta = (training[ele] - data[ele])
        result += delta*delta
    return np.sqrt(result)


def getNearestNeighbor(labeldDistances, k):
    if labeldDistances == None:
        return None
    # sort the list of distances
    labeldDistances.sort()
    occurencesDict = {}
    # find the labels of the k nearest neigbours with their occurences
    lblElement = 1
    for neighbour in range(0,k):
        if labeldDistances[neighbour][lblElement] in occurencesDict.keys():
            occurencesDict[labeldDistances[neighbour][lblElement]] +=1

        else:
            occurencesDict[labeldDistances[neighbour][lblElement]] = 1

    return occurencesDict



def extractMostRepresented(occurrencesDict):
    if occurrencesDict == None:
        return None
    # Sort the list from high to low too find the highest value in the occurrencesDict
    highestValue = sorted(occurrencesDict.values(), reverse=True)[0]

    # put keys found with the highest value in a list
    occurrenceList = []
    for i in occurrencesDict.keys():
        if occurrencesDict[i] == highestValue:
            occurrenceList.append(i)

    # If there is more than 1 occurrence, return a random element of the occurrenceList
    if len(occurrenceList) > 1:
        rng = random.randint(0, len(occurrenceList)-1)
        return occurrenceList[rng]

    else:
        return occurrenceList[0]







def getCorrectMatches(k):
    # Calculate the number of correct results returned for each entry in the validation data,
    # and match these against the validationlabels
    if k == 0:
        return None
    matches = 0
    for index in range(len(validationdata)):
        result = extractMostRepresented(getNearestNeighbor(getDistancesToTrainingData(validationdata[index], (data, labels)), k))
        if result == None:
            return None
        elif result == validationlabels[index]:
            matches +=1
        elif result != validationlabels[index]:
            pass
        else:
            pass
    return matches

def calculatePercentage(a, total):
    return (a / total) * 100

def findBestK():
    # Get the correct % of each K value
    resultList = []
    for k in range(57,60):
        print("validating k = ", k)
        resultList.append((k, calculatePercentage(getCorrectMatches(k), len(validationlabels))))

    # Find the highest correct % in the resultList
    bestScore = 0
    for tuple in resultList:
        if tuple[1] >= bestScore:
            bestScore = tuple[1]

    # Search for the tuple in the resultList with the highest correct % and return it
    for tuple in resultList:
        if tuple[1] == bestScore:
            return tuple
    # best k is 58 with 66% most

for x in range(0,50):
    print(findBestK())


