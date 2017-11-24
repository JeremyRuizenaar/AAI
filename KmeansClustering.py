import numpy as np
import random
import matplotlib.pyplot as plt


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




# validationdata = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7])
# for ele in validationdata:
#     if ele[6] == -1:
#         ele[6] = 0
#     if ele[4] == -1:
#         ele[0] = 0
# validationdates = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0])
# validationlabels = []

# for label in validationdates:
#     if label < 20010301:
#         validationlabels.append('winter')
#     elif 20010301 <= label < 20010601:
#         validationlabels.append('lente')
#     elif 20010601 <= label < 20010901:
#         validationlabels.append('zomer')
#     elif 20010901 <= label < 20011201:
#         validationlabels.append('herfst')
#     else: # from 01−12 to end of year
#         validationlabels.append('winter')

def getRandomCentroids(trainingData, Kcentroids = 1, first= " "   ):

    if first == "empty":
        # print("create empty list")
        emptyTupleList = []
        emptyList = [0 for x in range(0, len(data[0]))]

        for x in range(0, Kcentroids):
            emptyTupleList.append( ( x, emptyList ) )
        return emptyTupleList
    else:

        centroids = []
        for i in range(0, Kcentroids):
            rng = random.randint(0, len(trainingData)-1)
            centroids.append( ( i ,  trainingData[rng] ))
        return centroids


def intraClusterDistance(centroids, clusteredExamples):
    resultList = []
    totalIntraDist = 0
    for centroid in centroids:
        result = 0
        intraCentroidDist = 0

        for example in clusteredExamples:

            if example[1] == centroid[0]:
                result = calculateDistance(example[0], centroid[1])
                result = result * result
                intraCentroidDist += result

        totalIntraDist += intraCentroidDist
        # print("total intra distance for cluster ", centroid[0], " = ", intraCentroidDist)
    return totalIntraDist




def getClusterRepresentation(centroids, clusteredExamples):
    print("cluster representation")
    # print(centroids)
    # print(clusteredExamples)
     #[ 0 for x in range(0, len(centroids))]

    for centroid in centroids:
        representation = {}
        centroidCounter = 0
        # print(" centroid ", centroid[0], "  with data ", centroid[1])
        for example in clusteredExamples:

            if example[1] == centroid[0]:
                centroidCounter +=1
                # print("  and example  ", example)
                if example[2] in representation.keys():
                    representation[example[2]] += 1

                else:
                    representation[example[2]] = 1

        #print("cluster ", centroid[0] ," has vote count ", representation)
        print("cluster ", centroid[0])
        keys = representation.keys()
        list = []
        for each in keys:
            list.append(each)
        for key in list:
            val = representation[key]
            print(key , " has % ", (( val / centroidCounter) * 100 )  )





def cluster(data, centroids, oldCentroids, assignedList):
    #print("entering cluster")
    # print(data)
    # print("enter with new ", centroids)
    # print("enter with old  ", oldCentroids)
    totalField = len(centroids) * len(data[0][0])
    stableCounter = 0
    for z in range(0, len(centroids)):
        for i in range(0, len(data[0][0])):
            a = centroids[z][1][i]
            b = oldCentroids[z][1][i]
            if a == b:
                stableCounter += 1

    if stableCounter == totalField:
        print("all centroid means are  stable exiting recursion ")
        getClusterRepresentation(centroids, assignedList)
        tmp = intraClusterDistance(centroids, assignedList)
        #print("intra cluster (cluster) ",tmp)
        Gholder.append( (  len(centroids) ,tmp) )
        return 0

    else:





        assignedExampleList = []
        counter = 0
        for example in data[0]:
            nCentroid = sorted(getDistancesToCentroids(example , centroids ))[0][1]
            assignedExampleList.append((example, nCentroid, data[1][counter]))
            counter += 1


        newCentroids = []

        for centroid in centroids:
            means = [0 for i in range(0, len(centroid[1]))]
            regionTotal = 0

            for example in assignedExampleList:

                if  example[1] == centroid[0]:
                    counter = 0
                    regionTotal += 1
                    for ele in example[0]:
                        means[counter] += ele
                        counter+= 1


            if regionTotal == 0:
                #print("no data bount to region " , centroid[0])
                regionTotal = 1

            counter = 0
            for ele in means:
                means[counter] = means[counter] / regionTotal
                counter += 1

            newCentroids.append(( centroid[0] , means ))
        # print("new centroids ", newCentroids)
        cluster(data, newCentroids, centroids, assignedExampleList)





def getBestKvalue():
    pass




def getDistancesToCentroids(example, centroids):
    if len(example) == 0:
        return None
    if len(centroids) == 0:
        return None
    distances = []
    # for each data element in the trainingdata add the distance and the label to the distanceslist
    for index in range(len(centroids)):
        distances.append( ( calculateDistance(example, centroids[index][1]), centroids[index][0]  ) )

    return distances

def calculateDistance(example, centroid):
    # calcululate difference between two classes represented by two lists
    result = 0
    for ele in range(0, len(example)):
        delta = (centroid[ele] - example[ele])
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




scree = []
for x in range(1, 10):
    Gholder = []
    cluster( (data, labels), getRandomCentroids(data, Kcentroids=x) , getRandomCentroids(data, Kcentroids= x, first="empty"), [] )
    scree.append(Gholder[0])
print(scree)

plotlist = [ scree[x][1] for x in range(0 , len(scree)) ]
plt.plot(plotlist)
plt.ylabel('intra squaredcentroid distance')
plt.xlabel('k value')
plt.show()

