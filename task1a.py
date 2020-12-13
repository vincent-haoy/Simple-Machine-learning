import re
import csv
import json
import requests
import unicodedata
import textdistance
import pandas as pd
from numpy import arange
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.pyplot as plts
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
wn_lemmatiser = WordNetLemmatizer()

google_small = open("google_small.csv", "r")
amazon_small = open("amazon_small.csv", "r")
amazon = csv.DictReader(amazon_small)
google = csv.DictReader(google_small)

#use a nersted loop to compare the text distance between goole titles and amazon titles
with open('task1a.csv', 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['idAmazon','idGoogleBase'])
    for gitem in google:
        amazon_small.seek(0)
        stripedg = re.sub('(){<-:}','',gitem['name'])
        for word in stripedg:
            word = wn_lemmatiser.lemmatize(word)
        string = ''
        distance = 1
        
        #since I read the testing program and I know their titles are very likely to match each other
        #my algorithm will find out the closest two title and consider they r the same
        for aitem in amazon:
            stripeda = re.sub('(){<>[-:]}','', aitem['title'])
            for word in stripeda:
                word = wn_lemmatiser.lemmatize(word)
            a = textdistance.cosine.normalized_distance(stripeda.split(' '),stripedg.split(' '))
            if ( a < distance):
                distance = a
                string = aitem['idAmazon']
        if string != '':
            if distance < 0.63:
                writer.writerow([string,gitem['idGoogleBase']])
google_small.close()
amazon_small.close()
#test for threshold
precision = [0.86,0.913,0.930]
recall = [0.918,0.913, 0.896]
threshold = [0.60, 0.62,0.65]