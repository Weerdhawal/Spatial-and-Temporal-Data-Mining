import math
import random
import time
import operator
from math import *
from Tkinter import *
import timeit

def loadCSV(fileName, noOfFeatures):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0]  # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance[:noOfFeatures])
    return dataset

def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")

    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple



def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])


# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
    if len(s) == 0:
        return False
    if len(s) > 1 and s[0] == "-":
        s = s[1:]
    for c in s:
        if c not in "0123456789.":
            return False
    return True
######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################

def distance(instance1, instance2,n):
    if n==1:
        result = euclidean_default(instance1, instance2)
    elif n==2:
        result = 1-cosine_similarity(instance1, instance2)
    elif n == 3:
        result = 1-jaccard_similarity(instance1, instance2)
    elif n==4:
        result = euclidean_similarity(instance1, instance2)
    elif n==5:
        result = cosine_similarity(instance1, instance2)
    elif n==6:
        result = jaccard_similarity(instance1, instance2)

    return result


def euclidean_default(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquares = 0
    # print ("Instance 1 length " + str(len(instance1)))
    for i in range(1, len(instance1)):
        # print instance1[i]
        sumOfSquares += (instance1[i] - instance2[i]) ** 2
    # print "Sum of Squares "
    # print sumOfSquares
    return sumOfSquares


def euclidean_similarity(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquares = 0
    for i in range(1, len(instance1)):
        # print instance1[i]
        sumOfSquares += (instance1[i] - instance2[i]) ** 2
    # convert to similarity
    result = round((1 / (1 + sumOfSquares)), 3)
    # print ("Euclidean similarity " + str(result))
    return result


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(article1, article2):
    if article1 == None or article2 == None:
        return float("inf")

    numerator = 0
    denominator1 = 0
    denominator2 = 0

    for i in range(1, len(article1)):
        numerator += article1[i] * article2[i]
        denominator1 += article1[i] * article1[i]
        denominator2 += article2[i] * article2[i]

    denominator = sqrt(denominator1) * sqrt(denominator2)
    # numerator = sum(a*b for a,b in zip(article1[:-1],article2[:-1]))
    # denominator = square_rooted(article1[:-1])*square_rooted(article2[:-1])
    if denominator == 0:
        result = float('inf')
    else:
        result = round(numerator / float(denominator), 3)

    # print ("Cosine similarity " + str(result))
    return (result)


def jaccard_similarity(article1, article2):
    if article1 == None or article2 == None:
        return float("inf")
    #print "This is jaccard"
    numerator = 0
    denominator = 0
    for index in range(1, len(article1)):
        # for b in article2:
        if article2[index] >= article1[index]:
            numerator += article1[index]
            denominator += article2[index]
        else:
            numerator += article2[index]
            denominator += article1[index]

    if denominator == 0:
        result = float('inf')
    else:
        result = round(numerator / float(denominator), 3)

    # print ("Jaccard similarity " + str(result))
    return (result)


def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    # print ("Number of instances: " + str(numInstances))
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    # print ("Number of attributes: " + str(instanceList[0][0]))
    means = [name] + [0] * (numAttributes - 1)
    # print means
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    # print ("Means: " + str(means))
    return tuple(means)


def assign(instance, centroids,n):
    minDistance = distance(instance, centroids[0],n)
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i],n)
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex


def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList


def assignAll(instances, centroids,n):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids,n)
        clusters[clusterIndex].append(instance)
    return clusters


def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        # print (" Mean Centroid : " + str(i) +" " + str(centroid))
        centroids.append(centroid)
    return centroids


def kmeans(instances, k,n, animation=False, initCentroids=None):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids

    if animation:
        delay = 1.0  # seconds
        canvas = prepareWindow(instances)
        clusters = createEmptyListOfLists(k)
        clusters[0] = instances
        paintClusters2D(canvas, clusters, centroids, "Initial centroids")
        time.sleep(delay)
    #print("Initial Centroids ")
    #for i in range(len(centroids)):
        #print("Initial Centroid " + str(i) + "   " + str(centroids[i]))

    prevCentroids = []
    prevWithiniss = float('inf')
    iteration = 0
    while (centroids != prevCentroids):
        iteration += 1

        #print iteration
        clusters = assignAll(instances, centroids,n)
        if animation:
            paintClusters2D(canvas, clusters, centroids, "Assign %d" % iteration)
            time.sleep(delay)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)

        #print iteration

        # for i in range(len(centroids)):
        #     print ("Intermediate: " + str(i)+" " + str(centroids[i]))

        # print ("Centroids " +  str(len(centroids[0])) +" " + str(centroids[0]))
        withinss = computeWithinss(clusters, centroids,n)
        if animation:
            paintClusters2D(canvas, clusters, centroids,
                            "Update %d, withinss %.1f" % (iteration, withinss))
            time.sleep(delay)
        if prevWithiniss <= withinss:
            break
        else:
            prevWithiniss=withinss

        if iteration==100:
            break
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iteration"]=iteration
    return result


def computeWithinss(clusters, centroids,n):
    result = 0

    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        # print centroid
        # print ("Centroid length: " + str(i)+ " " + str(len(centroids[i])))
        # print centroids[i]
        for instance in cluster:
            result += distance(centroid, instance,n)
    return result


# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n + 1):
        print
        "k-means trial %d," % i,
        trialClustering = kmeans(instances, k)
        print
        "withinss: %.1f" % trialClustering["withinss"]
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print
    "Trial with minimum withinss:", minWithinssTrial
    return bestClustering


######################################################################
# This section contains functions for visualizing datasets and
# clustered datasets.
######################################################################

def printTable(instances):
    for instance in instances:
        if instance != None:
            line = str(len(instance)) + " " + instance[0] + "\t"
            for i in range(1, len(instance)):
                line += "%.2f " % instance[i]
            print
            line


def extractAttribute(instances, index):
    result = []
    for instance in instances:
        result.append(instance[index])
    return result

def paintCircle(canvas, xc, yc, r, color):
    canvas.create_oval(xc-r, yc-r, xc+r, yc+r, outline=color)

def paintSquare(canvas, xc, yc, r, color):
    canvas.create_rectangle(xc-r, yc-r, xc+r, yc+r, fill=color)

def drawPoints(canvas, instances, color, shape):
    random.seed(0)
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / (maxX - minX)
    scaleY = float(height - 2*margin) / (maxY - minY)
    for instance in instances:
        x = 5*(random.random()-0.5)+margin+(instance[1]-minX)*scaleX
        y = 5*(random.random()-0.5)+height-margin-(instance[2]-minY)*scaleY
        if (shape == "square"):
            paintSquare(canvas, x, y, 5, color)
        else:
            paintCircle(canvas, x, y, 5, color)
    canvas.update()

def connectPoints(canvas, instances1, instances2, color):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    scaleX = float(width - 2*margin) / (maxX - minX)
    scaleY = float(height - 2*margin) / (maxY - minY)
    for p1 in instances1:
        for p2 in instances2:
            x1 = margin + (p1[1]-minX)*scaleX
            y1 = height - margin - (p1[2]-minY)*scaleY
            x2 = margin + (p2[1]-minX)*scaleX
            y2 = height - margin - (p2[2]-minY)*scaleY
            canvas.create_line(x1, y1, x2, y2, fill=color)
    canvas.update()

def mergeClusters(clusters):
    result = []
    for cluster in clusters:
        result.extend(cluster)
    return result


def saveClustersAndCentroid(clusterNum, centroid, currCluster):
    # with open("EuclideanDefaultGeneral/euclideanDefaultcluster_%s.txt" % str(clusterNum) , 'w') as f:
    # with open("EuclideanSimilarityGeneral/euclideanSimcluster_%s.txt" % str(clusterNum) , 'w') as f:
    # with open("CosineSimilarityGeneral/cosineSimcluster_%s.txt" % str(clusterNum) , 'w') as f:
    with open("Jaccard/jaccardSimcluster_%s.txt" % str(clusterNum), 'w') as f:
        if (centroid != None):
            for i in range(len(centroid)):
                if i > 0:
                    f.write(str(round(centroid[i], 5)) + ",")
                else:
                    f.write(str(centroid[i]) + ",")
            f.write("\n")
        for clusterMember in currCluster:
            for i in range(len(clusterMember)):
                f.write(str(round(clusterMember[i], 5)) + ",")
            f.write("\n")
    f.close()


def prepareWindow(instances):
    width = 500
    height = 500
    margin = 50
    root = Tk()
    canvas = Canvas(root, width=width, height=height, background="white")
    canvas.pack()
    canvas.data = {}
    canvas.data["margin"] = margin
    setBounds2D(canvas, instances)
    paintAxes(canvas)
    canvas.update()
    return canvas

def setBounds2D(canvas, instances):
    attributeX = extractAttribute(instances, 1)
    attributeY = extractAttribute(instances, 2)
    canvas.data["minX"] = min(attributeX)
    canvas.data["minY"] = min(attributeY)
    canvas.data["maxX"] = max(attributeX)
    canvas.data["maxY"] = max(attributeY)

def paintAxes(canvas):
    width = canvas.winfo_reqwidth()
    height = canvas.winfo_reqheight()
    margin = canvas.data["margin"]
    minX = canvas.data["minX"]
    minY = canvas.data["minY"]
    maxX = canvas.data["maxX"]
    maxY = canvas.data["maxY"]
    canvas.create_line(margin/2, height-margin/2, width-5, height-margin/2,
                       width=2, arrow=LAST)
    canvas.create_text(margin, height-margin/4,
                       text=str(minX), font="Sans 11")
    canvas.create_text(width-margin, height-margin/4,
                       text=str(maxX), font="Sans 11")
    canvas.create_line(margin/2, height-margin/2, margin/2, 5,
                       width=2, arrow=LAST)
    canvas.create_text(margin/4, height-margin,
                       text=str(minY), font="Sans 11", anchor=W)
    canvas.create_text(margin/4, margin,
                       text=str(maxY), font="Sans 11", anchor=W)
    canvas.update()


def showDataset2D(instances):
    canvas = prepareWindow(instances)
    paintDataset2D(canvas, instances)

def paintDataset2D(canvas, instances):
    canvas.delete(ALL)
    paintAxes(canvas)
    drawPoints(canvas, instances, "blue", "circle")
    canvas.update()

def saveCentroidsToAFile(centroid):
    # with open("EuclideanSimilarityGeneral/euclideanSimCentroids.csv", 'a') as f:
    # with open("CosineSimilarityGeneral/cosineSimCentroids.csv", 'a') as f:
    with open("Jaccard/jaccardSimCentroids.csv", 'a') as f:
        # with open("EuclideanDefaultGeneral/euclideanDefaultCentroids.csv", 'a') as f:
        if (centroid != None):
            for i in range(len(centroid)):
                if i > 0:
                    f.write(str(round(centroid[i], 5)) + ",")
                else:
                    f.write(str(centroid[i]) + ",")
            f.write("\n")

def showClusters2D(clusteringDictionary):
    clusters = clusteringDictionary["clusters"]
    centroids = clusteringDictionary["centroids"]
    withinss = clusteringDictionary["withinss"]
    canvas = prepareWindow(mergeClusters(clusters))
    paintClusters2D(canvas, clusters, centroids,
                    "Withinss: %.1f" % withinss)

def paintClusters2D(canvas, clusters, centroids, title=""):
    canvas.delete(ALL)
    paintAxes(canvas)
    colors = ["blue", "red", "green", "brown", "purple", "orange","black","magenta","cyan"]
    for clusterIndex in range(len(clusters)):
        color = colors[clusterIndex % len(colors)]
        instances = clusters[clusterIndex]
        centroid = centroids[clusterIndex]
        drawPoints(canvas, instances, color, "circle")
        if (centroid != None):
            drawPoints(canvas, [centroid], color, "square")
        connectPoints(canvas, [centroid], instances, color)
    width = canvas.winfo_reqwidth()
    canvas.create_text(width / 2, 20, text=title, font="Sans 14")
    canvas.update()


######################################################################
# Test code
######################################################################

# dataset = loadCSV("output/")
dataset = loadCSV("checkfinal_nc.csv", 100)
#dataset = loadCSV("data.csv",270)
# print dataset[:3]

# showDataset2D(dataset)
#dataset = random.sample(dataset,1000)
#showDataset2D(dataset)
noOfClusters = 20
#n=1
#if n==1:
for n in range(1,7):
    if n == 1:
        start_time = timeit.default_timer()
        euc_clustering = kmeans(dataset, noOfClusters,n)
        elapsed = timeit.default_timer() - start_time
        print "Time : %f " %elapsed
        print "Euc Withinss %f" %euc_clustering["withinss"]
        print "iteration %d" %euc_clustering["iteration"]
        print "Cluster distribution List"
        for i in range(0,len(euc_clustering["clusters"])):
            print "%d \t" %len(euc_clustering["clusters"][i])
    elif n == 2:
        start_time = timeit.default_timer()
        cos_clustering = kmeans(dataset, noOfClusters,n)
        elapsed = timeit.default_timer() - start_time
        print "Time : %f " %elapsed
        print "Cos Withinss %f" %cos_clustering["withinss"]
        print "iteration %d" % cos_clustering["iteration"]
        print "Cluster distribution List"
        for i in range(0, len(cos_clustering["clusters"])):
            print len(cos_clustering["clusters"][i])
    elif n == 3:
        start_time = timeit.default_timer()
        jac_clustering = kmeans(dataset, noOfClusters,n)
        elapsed = timeit.default_timer() - start_time
        print "Time : %f " %elapsed
        print "Jac Withinss %f" %jac_clustering["withinss"]
        print "iteration %d" %jac_clustering["iteration"]
        print "Cluster distribution List"
        for i in range(0,len(jac_clustering["clusters"])):
            print len(jac_clustering["clusters"][i])

    elif n == 4:
        start_time = timeit.default_timer()
        gen_euc_clustering = kmeans(dataset, noOfClusters,n)
        elapsed = timeit.default_timer() - start_time
        print "Time : %f " %elapsed
        print "Gen Euclidean Withinss %f" %gen_euc_clustering["withinss"]
        print "iteration %d" % gen_euc_clustering["iteration"]
        print   "Cluster distribution List"
        for i in range(0, len(gen_euc_clustering["clusters"])):
            print len(gen_euc_clustering["clusters"][i])
    elif n == 5:
        start_time = timeit.default_timer()
        gen_cos_clustering = kmeans(dataset, noOfClusters, n)
        elapsed = timeit.default_timer() - start_time
        print "Time : %f " %elapsed
        print "Gen Cosine Withinss %f" % gen_cos_clustering["withinss"]
        print "iteration %d" % gen_cos_clustering["iteration"]
        print "Cluster distribution List"
        for i in range(0, len(gen_cos_clustering["clusters"])):
            print len(gen_cos_clustering["clusters"][i])
    elif n == 6:
        start_time = timeit.default_timer()
        gen_jac_clustering = kmeans(dataset, noOfClusters, n)
        elapsed = timeit.default_timer() - start_time
        print "Time : %f " %elapsed
        print "Gen Jaccard Withinss %f" % gen_jac_clustering["withinss"]
        print "iteration %d" % gen_jac_clustering["iteration"]
        print "Cluster distribution List"
        for i in range(0, len(gen_jac_clustering["clusters"])):
            print len(gen_jac_clustering["clusters"][i])
#for i in range(0, noOfClusters):
#    print ("Size of Cluster " + str(i) +" " + str(len(clustering["clusters"][i])))
#    saveClustersAndCentroid(i, clustering["centroids"][i], clustering["clusters"][i])
#    saveCentroidsToAFile(clustering["centroids"][i])

#printTable(clustering["centroids"])
#print "Euc Withinss %f" %euc_clustering["withinss"]


#print clustering["withinss"]
#print clustering["iteration"]
