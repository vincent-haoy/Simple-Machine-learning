import re
import csv
import math
import unicodedata
import textdistance
from numpy import arange
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
wn_lemmatiser = WordNetLemmatizer()
i = 0;

# this function round up/down to the nearest the multiple of the multiple given
# this function is only for report proper
def round_up(numb, multiple):
    try:
        numb = str(numb)
        numb = re.sub(r'[^0-9^\.]','', gitem['price'])
        numb = float(numb)
    except:
        return 0
    
    upper = int(math.ceil((numb / multiple)))* (multiple) 
    lower = int(math.ceil((numb / multiple))) * multiple - multiple
    if(upper - numb > numb - lower):
        if(lower < 0):
            return 0
        else:
            return lower
    else:
        return upper

#this function will return a block key
def block_assign(line):
    #normalization
    gg = line
    gg = re.sub(r'(\()|(\))|(-)|(&)|:|!|/','',gg)    
    #lemmatize
    for word in gg:
        word = wn_lemmatiser.lemmatize(word)
    gg = gg.split(' ')
    gg = [i for i in gg if (i != '' and len(i) > 2)]
    return gg

google_small = open("google.csv", "r")
amazon_small = open("amazon.csv", "r")

amazon = csv.DictReader(amazon_small)
google = csv.DictReader(google_small)

#I wanna tuse n grame algorithm to generate keys for the blocks,
#N is the length of sub-sentense u want to split
def N_gram(pharsed,gram):
    result = []
    for i in range(-gram+1,len(pharsed)- 1):
        temp = ''
        for g in range(i,i+gram):
            if g < 0 or g > len(pharsed)-1:
                temp = temp + "_ "
            else:
                temp = temp + pharsed[g]  + " "
        temp = temp[:-1]
        result.append(temp)
    return result

#appply nomalization, limmatization and N gram algorithm, set N as 2.
with open('amazon_blocks.csv', 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['block_key','product_id'])
    for aitem in amazon:
        gg = aitem['title']
        #the Momey value is irrelevant to code marking, but used in report
        Money = round_up(aitem['price'],600)
        array = block_assign(aitem['title'])
        result = N_gram(array,2)
        for answer in result:
            writer.writerow([answer,aitem['idAmazon']])
            
with open('google_blocks.csv', 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['block_key','product_id'])    
    for gitem in google:
        gg = gitem['name']
        #the Momey value is irrelevant to code marking, but used in report
        Money = round_up(gitem['price'],600)
        array = block_assign(gitem['name'])
        result = N_gram(array,2)
        for answer in result:
            writer.writerow([answer,gitem['id']])


                        

    
