
# coding: utf-8

# In[3]:

#Thesis Analysis Code
import csv
from collections import *
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import json
from pprint import pprint
import os
import pandas as pd
import string
import re
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib
import datetime
from nltk.corpus import stopwords
import codecs

#'/Users/Jeremy/Desktop/Thesis/hackathon_json_twitter/' + startday + '_tedcruz.txt'

filenames = []
bucksin = input("How many time buckets do you want to analyze?  ")
candidatesin = input("How many candidates do you want to analyze?:  ")
bucks = int(bucksin)
candidates = int(candidatesin)
numinput = candidates
polnames = []
filenamescut = []
numoffile = int(numinput)
for i in range(0,numoffile):
    print("The politicians available are: Rand Paul, Ted Cruz, Marco Rubio, Carly Fiorina, George Pataki, John Kasich, Donald Trump, Lindsey Graham, Rick Santorum, Jeb Bush, Mike Huckabee, Chris Christie, Ben Carson, Bobby Jindal, Rick Perry, Scott Walker")
    politician = input("Type the last name of the politician you want to analyze:   ")
    polnames.append(politician)
    path = "/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/"
    for file in os.listdir("/Users/Jeremy/Desktop/Thesis/Coding/TextFiles"):
        if file.endswith(politician + ".txt"):
            if os.path.getsize(path+file)>0:
                filenamescut.append(file)
                filenames.append('/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/'+file)
        
    
#1grams:
bi_vectorizer = CountVectorizer(input='filename', encoding=u'utf-8', strip_accents='unicode', lowercase=True, ngram_range = (1,2), analyzer =u'word', min_df = 1, vocabulary=None)
dtm = bi_vectorizer.fit_transform(filenames)  # a document term sparse matrix, each row corresponds to a file, each column is the frequency of a given bigram
vocab = bi_vectorizer.get_feature_names()  # a list of bigrams, with each bigram corresponding with the location in the DTM


dtm = dtm.toarray()  # convert to a regular array
candidatenumber = len(filenames)
bigramcombos =  len(vocab)
dtmnorm = np.zeros((candidatenumber, bigramcombos))

#Normalize
candidatenumber = len(filenames)
bigramcombos =  len(vocab)
def normalize(temp, rowsum):
    try: return (temp/rowsum)
    except: return 0
        
for i in range (0, candidatenumber):
    rowsum = np.sum(dtm[i])
    for j in range (0, bigramcombos):
        temp = dtm[i][j]
        dtmnorm[i][j] = normalize(temp,rowsum)

#vocab = np.array(vocab) #make vocab a numpy array

eucdist = euclidean_distances(dtmnorm) #calculate euclidean distances (each file is a vector)
cosdist = 1 - cosine_similarity(dtmnorm) #calculate cosine similarity, but 1- to make the larger angle recieve a larger value (exactly the same is 0, orthagonal is 1, diametrically opposed is -1)

#distance between all text files

#np.round(eucdist, 2)  
#np.round(cosdist, 2)

#distance between two text files
#distance between dist[1,2]





#Save to CSV
polnames = ''.join(polnames)
filenamesheader = ','.join(filenamescut)

a = np.asarray(eucdist)
b = np.asarray(cosdist)
with open('/Users/Jeremy/Desktop/Thesis/Coding/Csv/' + polnames+ '_Euclidean_Output.csv', 'wb') as f:
    np.savetxt(f,a,delimiter = ',',header = filenamesheader, comments = '')
with open('/Users/Jeremy/Desktop/Thesis/Coding/Csv/' + polnames+ '_Cosine_Output.csv', 'wb') as f:
    np.savetxt(f,b,delimiter = ',',header = filenamesheader, comments = '')


#Names of politicians
names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]
#Clear PLT
plt.clf()
#Scatter Code
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(cosdist)

xs, ys = pos[:, 0], pos[:, 1]
for x, y, name in zip(xs, ys, names):
    color = 'orange'
    plt.scatter(x, y, c=color)
    plt.text(x, y, name)
plt.savefig('/Users/Jeremy/Desktop/Thesis/Coding/Charts/'+ polnames + 'scatter.png')

#Dendro Code
plt.clf()
linkage_matrix = ward(cosdist)
# match dendrogram to that returned by R's hclust()
dendrogram(linkage_matrix, orientation="right", labels=names)
plt.tight_layout()
plt.savefig('/Users/Jeremy/Desktop/Thesis/Coding/Charts/'+ polnames + 'dendr.png')




#Request count for a specific word (can be done above)
#index = list(vocab).index('friend')
#print(dtm[0, index])



#TO DO:
#change in distances over time
#sentiment
#write a page explaining what I've done in the code 




#1. Examine distances between candidates
#2. Timeline...(includes poll numbers, significant events, debates, caucus )
#3. Distances between front runners (all others)/variance in front runners
#4. Word freq case... Counts of certain words: number of times trump tweeted about immigration
#5. Writing stuff down, here is what we are going to do and why its valuable: what is my learning objective, what do I want to get from the data
#6. Go back to the hypotheses
#7. add a baseline




#split candidates into weeks (look at one candidate over 5 weeks)
#think of a set of covariates that might be relevant...front runner status, geographic relevance around new hampshire hillary might be like bernie, differences in poll numbers, who won the primaries / exceeded expectations
#regression models 
#distance between bush and trump as a factor of front runner status/distance in the polls
#case study
#real clear politics








## 
# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



