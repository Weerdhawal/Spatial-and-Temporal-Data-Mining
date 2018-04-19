
import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
import IPython
import matplotlib.pyplot as plt
import csv
from sklearn import tree
from io import StringIO
from IPython.display import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyPDF2 import PdfFileMerger
import os
import subprocess
"""
Data Engineering and Analysis
"""
#Load the dataset
AH_data = pd.read_csv("data1.csv", encoding = "ISO-8859-1")
data_clean = AH_data.dropna()

"""
Modeling and Prediction
"""
#Split Predictors and targets

predictors = data_clean[['also', 'another', 'anyone', 'can', 'christian', 'even', 'example', 'first', 'get', 'god', 'however', 'know', 'like', 'make', 'many', 'may', 'moral', 'morality', 'must', 'one', 'people', 'rather', 'read', 'reason', 'religion', 'right', 'see', 'seems', 'subject', 'sure', 'take', 'think', 'true', 'way', 'well', 'will', 'wrote', 'anything', 'article', 'atheists', 'believe', 'cant', 'doesnt', 'dont', 'fact', 'good', 'just', 'little', 'made', 'mean', 'might', 'much', 'nothing', 'now', 'point', 'question', 'really', 'said', 'say', 'since', 'someone', 'something', 'still', 'thats', 'thing', 'things', 'time', 'two', 'want', 'years', 'writes', 'keith', 'never', 'yet', 'email', 'need', 'please', 'software', 'thanks', 'files', 'find', 'using', 'looking', 'file', 'image', 'program', 'graphics', 'use', 'help', 'computer', 'windows', 'win', 'problem', 'ive', 'microsoft', 'dos', 'version', 'run', 'system', 'work', 'bus', 'card', 'new', 'controller', 'drive', 'ide', 'apple', 'mac', 'code', 'display', 'running', 'set', 'window', 'application', 'motif', 'server', 'asking', 'offer', 'sale', 'used', 'interested', 'call', 'sell', 'condition', 'shipping', 'price', 'car', 'cars', 'engine', 'dod', 'back', 'bike', 'bikes', 'bmw', 'got', 'motorcycle', 'ride', 'riding', 'baseball', 'game', 'season', 'team', 'better', 'games', 'hit', 'last', 'league', 'play', 'players', 'year', 'best', 'hes', 'fans', 'going', 'hockey', 'next', 'nhl', 'player', 'playoff', 'playoffs', 'teams', 'cup', 'played', 'algorithm', 'available', 'crypto', 'data', 'encryption', 'government', 'information', 'key', 'keys', 'pgp', 'public', 'secure', 'security', 'without', 'law', 'number', 'secret', 'nsa', 'phone', 'david', 'chip', 'clipper', 'escrow', 'power', 'medical', 'space', 'earth', 'moon', 'orbit', 'nasa', 'shuttle', 'gods', 'quite', 'faith', 'life', 'though', 'word', 'jesus', 'man', 'bible', 'christ', 'church', 'come', 'give', 'part', 'says', 'sin', 'whether', 'world', 'clh', 'christians', 'others', 'christianity', 'mayathosrutgersedu', 'control', 'gun', 'american', 'day', 'didnt', 'fbi', 'firearms', 'guns', 'weapons', 'state', 'atf', 'bill', 'compound', 'fire', 'waco', 'today', 'batf', 'israel', 'jews', 'argic', 'armenia', 'armenian', 'serdar', 'turks', 'armenians', 'children', 'closed', 'history', 'jewish', 'longer', 'muslim', 'muslims', 'policy', 'single', 'soviet', 'turkish', 'war', 'israeli', 'case', 'arab', 'arabs', 'peace', 'rights', 'killed', 'center', 'clayton', 'cramer', 'gay', 'look', 'opinions', 'clinton', 'study', 'different']]
# print(predictors)
targets = data_clean.mainCATEGORIES
# print(targets)

#Perform 5 fold cross validation
kf = KFold(n_splits=5)
# fold = 0
i=1
l=[] #list to store the accuracy values

test=[]
train=[]

# Running the 5 Fold Cross Validation!
def getValues(count):
    if count==1:
        for i in range(18828):
            if 0<=i<3765:
                test.append(i)
            else:
                train.append(i)
        print("phase 1")

    if count==2:
        for i in range(18828):
            if 3765<=i<7530:
                test.append(i)
            else:
                train.append(i)
        print("phase 2")

    if count==3:
        for i in range(18828):
            if 7530<=i<11295:
                test.append(i)
            else:
                train.append(i)
        print("phase 3")

    if count==4:
        for i in range(18828):
            if 11296<=i<15060:
                test.append(i)
            else:
                train.append(i)
        print("phase 4")

    if count==5:
        for i in range(18828):
            if 15060<=i<18828:
                test.append(i)
            else:
                train.append(i)
        print("phase 5")

    return test,train


# Loop runs 5 times to simulate the 5 Fold Cross Validation!
print('\n')
print("                       THE DECISION TREE ANALYSIS")
for x in range(5):
    test=[]
    train=[]
    # print(testing)
    test,train=getValues(x+1)
    pred_train = predictors.ix[train]
    tar_train = targets[train]
    pred_test = predictors.ix[test]
    tar_test = targets[test]

    #Build model on training data
    classifier=DecisionTreeClassifier()
    classifier=classifier.fit(pred_train,tar_train)
    predictions=classifier.predict(pred_test)
    #Displaying the decision tree
    out = StringIO()
    tree.export_graphviz(classifier, out_file=out)
    import pydotplus
    graph=pydotplus.graph_from_dot_data(out.getvalue())

    #Generate Graphs for the Decision Classifier
    millis = int(round(time.time() * 1000))  # Generate time system time in milliseconds
    Image(graph.write_pdf(str(i)+".pdf"))

    print("===========================================================")
    print('Iteration #'+str(x+1)+'\n')
    #Calculate accuracy
    l.append(sklearn.metrics.accuracy_score(tar_test, predictions)*100)
    print("Accuracy Score is "+str(l[i-1])+ '%')
    i+=1

    print(sklearn.metrics.confusion_matrix(tar_test,predictions))
    print(classification_report(tar_test,predictions))
#Generating a list for accuracy
l=[int(x) for x in l]
with open("DecisionTree_Accuracy.csv", "w") as fp_out:
    writer = csv.writer(fp_out, delimiter=",")
    writer.writerow(l)
a=0
for i in range(len(l)):
    a+=l[i]
# print(a)
print("===========================================================")
print("===========================================================")
print('The average accuracy after 5 Fold Cross Validation is: '+str(int(a/len(l)))+"%")
print("===========================================================")
print("===========================================================")


### VISUALIZING THE ACCURACY DATA ###
# data to plot
n_groups = 5

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects2 = plt.bar(index + bar_width, l, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Decision Tree')

plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Accuracy by Decision Tree')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5'))
plt.legend()

plt.tight_layout()
plt.show()

# Merging the Decision Tree graphs to clean the directory!
pdfs = ['1.pdf', '2.pdf', '3.pdf', '4.pdf','5.pdf']
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(open(pdf, 'rb'))
with open('DecisionTrees.pdf', 'wb') as fout:
    merger.write(fout)


sys.exit()
