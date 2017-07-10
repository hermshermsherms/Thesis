
# coding: utf-8

# In[ ]:

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
from nltk.stem import SnowballStemmer
ls = SnowballStemmer('english')
restricted = ['isis']
import string
import re
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib
import datetime
from nltk.corpus import stopwords
import codecs
import ipywidgets as widgets
from IPython.display import display
sorted(widgets.Widget.widget_types)
from pandas import HDFStore,DataFrame
#define word in and import stopwords
 

analyze = input("Do you want to reupload the data?  If yes, write Y.  If no, write N:  ")

if analyze == 'Y':
    
    def word_in (word, phrase):
        return (word in phrase.split())

    stopwords_file = '/Users/Jeremy/Desktop/Thesis/Coding/stopwords.txt'
    all_stopwords = set(codecs.open(stopwords_file, 'r', 'utf-8').read().splitlines())
    plt.ion()

    #open csv
    path_to_json = '/Users/Jeremy/Desktop/Thesis/Coding/JSONS'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    #panda dataframe
    jsons_data = pd.DataFrame(index=None, columns=['date', 'candidate', 'text','nostemtext'])

    indexcounter=0
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            d = json.load(json_file)

        #pull date candidate and text

        for tweets in d:
            date = tweets['date']
            #Put date in correct format
            newDate = re.sub('[()/]|[a-zA-Z]', '', date)
            timeSec = float(newDate) / 1000.0
            dateTime = datetime.datetime.fromtimestamp(timeSec).strftime('%Y-%m-%d %H:%M:%S.%f')
            candidate = tweets['source']['name']
            temp = tweets['text']
            temp = temp.lower()

            #remove stopwords
            for word in all_stopwords:
                if word_in(word, temp):
                    newtemp = re.sub(r'\b'+word+r'\b','', temp)
            #remove urls and special characters
            temp2 = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', temp)
            text = re.sub('\W+',' ', temp2 )
            text = text.encode('ascii', 'ignore')
            nostemtext = str(text)
            #stemming
            temp1 = text.split()
            concat = []
            for obj in temp1:
                obj = obj.decode("utf-8")
                ab = (ls.stem(obj))
                concat.append(ab)
            stemtext = " ".join(concat)   
            #store in dataframe
            jsons_data.loc[indexcounter] = [dateTime, candidate, stemtext, nostemtext]
            indexcounter=indexcounter+1

    #Print where candidate = sen rand paul, returns a dataframe     
    #df = jsons_data.query("candidate == 'Sen. Paul, Rand - (R – KY) Presidential Campaign'")

    #Print count of candidates
    #df7 = jsons_data['candidate'].value_counts()
    #print(df7)
    #jsons_data[jsons_data['candidate'] == 'Sen. Paul, Rand - (R – KY) Presidential Campaign'].head(5)

    #df1 returns all occurences of isis tweets, df2 counts by candidate
    #df1 = jsons_data[jsons_data["text"].str.contains("isis", na = False)]
    #df= df1.groupby('candidate').size()

    #Create time range
    #periodrange = pd.period_range('2-2-2016 00:00', '2-3-2016 00:00', freq='Min')

    jsons_data.to_pickle('Saved.pkl')  # save it

    
    
    
    
    
    
elif analyze == 'N':
    jsons_data = pd.read_pickle('Saved.pkl')  # load it


choice = input("Do you want to measure distance over time buckets (M) or analyze word frequencies (A)?")

if choice == 'M' or choice == 'm':
    #bucketsinput = input("How many time buckets do you want to analyze?:   ")
    #buckets = int(bucketsinput)
    
    datelist = pd.date_range('2015-01-01', freq='W', periods=80).tolist()
    for j in range(0,(len(datelist)-1)):

    
        df = jsons_data[jsons_data['candidate'] == 'Sen. Paul, Rand - (R – KY) 2016 Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df3 = df2['text']

        df = jsons_data[jsons_data['candidate'] == '(Fmr.) Sen. Cruz, Ted - (R – TX) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df4 = df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Sen. Rubio, Marco - (R – FL) 2016 Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df5 = df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Fiorina, Carly (R) Presidential Campaign ']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df6 = df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Trump, Donald J. (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df7 = df2['text']

        df = jsons_data[jsons_data['candidate'] == '(Fmr.) Sen. Graham, Lindsey - (R – SC) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df8 = df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Santorum, Rick (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df9 = df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Bush, Jeb (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df10=df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Huckabee, Mike (R) Presidential Campaign ']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df11=df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Christie, Chris (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df12 = df2['text']

        df = jsons_data[jsons_data['candidate'] == 'Carson, Ben (R) Presidential Campaign ']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df13 = df2['text']

        df = jsons_data[jsons_data['candidate'] == '(Fmr.) Jindal, Bobby (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df14 = df2['text']

        df = jsons_data[jsons_data['candidate'] == '(Fmr.) Perry, Rick (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df15 = df2['text']

        df = jsons_data[jsons_data['candidate'] == '(Fmr.) Walker, Scott (R) Presidential Campaign\t']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df16 = df2['text']

        df = jsons_data[jsons_data['candidate'] == '(Fmr.) Gov. Kasich, John (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df17 = df2['text']

        df = jsons_data[jsons_data['candidate'] == '(Fmr.) Pataki, George (R) Presidential Campaign']
        df2 = df[(df['date'] > str(datelist[j])) & (df['date'] <= str(datelist[j+1] ))]
        df18 = df2['text']
        
        dateuc = str(datelist[j])
        startday = dateuc[:11]
        
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_randpaul.txt', df3.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_tedcruz.txt', df4.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_marcorubio.txt', df5.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_carlyfiorina.txt', df6.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_donaldtrump.txt', df7.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_lindseygraham.txt', df8.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_ricksantorum.txt', df9.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_jebbush.txt', df10.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_mikehuckabee.txt', df11.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_chrischristie.txt', df12.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_bencarson.txt', df13.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_bobbyjindal.txt', df14.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_rickperry.txt', df15.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_scottwalker.txt', df16.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_johnkasich.txt', df17.values, fmt='%s')
        np.savetxt(r'/Users/Jeremy/Desktop/Thesis/Coding/TextFiles/' + startday + '_georgepataki.txt', df18.values, fmt='%s')

        
if choice == 'A' or choice == 'a':
    wordchoice = input('What word do you want to analyze?')
    indices = jsons_data['nostemtext'].str.contains(r'(?:\s|^)' + wordchoice + '(?:\s|$)', na = False)
    df1 = jsons_data[indices]
    df2 = df1.sort_values(['date'],ascending = [1])
    df2.to_csv('/Users/Jeremy/Desktop/Thesis/Coding/' + wordchoice + '_WordSearch.csv',sep=',')
    wordchoicestem = ls.stem(wordchoice)
    indices2 = jsons_data['text'].str.contains(r'(?:\s|^)' + wordchoicestem + '(?:\s|$)', na = False)
    df3 = jsons_data[indices2]
    df4 = df3.sort_values(['date'],ascending = [1])
    df4.to_csv('/Users/Jeremy/Desktop/Thesis/Coding/' + wordchoice + '_StemmedWordSearch.csv',sep=',')



# ## 

# In[2]:




# In[ ]:



