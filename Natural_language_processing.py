#!/usr/bin/env python
# coding: utf-8

# **Warmup**

# In[1]:


import collections
import random

from collections import defaultdict
from collections import Counter


# In[2]:


file_1 = "dat410_europarl/europarl-v7.sv-en.lc.sv"
file_sv = open(file_1, "r")
read_sv = file_sv.read()
words_sv = read_sv.split(' ')


# In[3]:


file_2 = "dat410_europarl/europarl-v7.sv-en.lc.en"
file_sv_en = open(file_2, "r")
read_sv_en = file_sv_en.read()
words_sv_en = read_sv_en.split(' ')


# In[4]:


c_sv = collections.Counter(words_sv)
c_sv_en = collections.Counter(words_sv_en)
print("The 10 most common swedish words are",c_sv.most_common(10))
print("The 10 most common english words are",c_sv_en.most_common(10))
  


# In[5]:


freq_speaker = c_sv_en['speaker']
count_sv_en= len(words_sv_en)
prob_speaker = freq_speaker/count_sv_en
print("The probability that the ramdomnly chosen word is 'speaker' is", prob_speaker)


# In[7]:


freq_zebra = c_sv_en['zebra']
count_sv_en= len(words_sv_en)
prob_zebra = freq_zebra/count_sv_en
print("The probability that the ramdomnly chosen word is 'zebra' is", prob_zebra)


# Language Modelling

# In[8]:


word_lists = []
data_sets = []

for i in [words_sv, words_sv_en]:
    data_sets.append(i)
    c = Counter(i)
    print(c.most_common(10))
    word_lists.append(c)
    
all_words = word_lists[0]

# This piece of code is used when reading all of the different texts as it will merge all the counts of all the words in English
for i in word_lists[1:]:
    for key, item in i.items():
        all_words[key] += item

sum_words = sum(all_words.values())

def mle(sentence, datasets):
    sentence = sentence.split(" ")
    probability = 1
    
    for ind, word in enumerate(sentence[1:]):
        
        ind += 1
        count_prev_word = all_words[sentence[ind-1]]
        count_word_serie = 0

        for dset in datasets:
            for ind_dset, word_in_dset in enumerate(dset):
                if word_in_dset == word and sentence[ind-1] == dset[ind_dset-1]:
                    count_word_serie += 1
        
        if count_prev_word != 0:
          if count_word_serie/count_prev_word != 0:
            print(f"Probability suite {sentence[ind-1:ind+1]} {count_word_serie/count_prev_word}")  
          probability *= count_word_serie/count_prev_word
        
        
    return probability


# **Translation Modelling**

# In[9]:


file_sv = open(file_1, "r")
lines_sv = file_sv.readlines()
file_sv_en = open(file_2, "r")
lines_sv_en = file_sv_en.readlines()
pairs =list(zip(lines_sv, lines_sv_en))  #pairs of swedish-english lines


#Initialization
t = defaultdict(float)
for pair in pairs:
    sv_sent = pair[0]  #only swedish sentence
    en_sent = pair[1]  #only english sentence
    sv_words = sv_sent.split(' ')  #swedish words
    en_words = en_sent.split(' ')  #english words
    en_words.append(None)
    for sv_word in sv_words:
        for en_word in en_words:
            t[(sv_word,en_word)] = 1 #initial probability
            

for i in range(10):
    count_sv_en = defaultdict(float) #c(e,s)
    count_en =defaultdict(float) #c(e)
    for pair in pairs:
        sv_sent = pair[0]
        en_sent = pair[1]
        sv_words = sv_sent.split(' ')
        en_words = en_sent.split(' ')
        en_words.append(None)
        for sv_word in sv_words:
            deno = defaultdict(float)
            deno[sv_word]= 0.0
            for en_word in en_words:
                deno[sv_word]+=t[(sv_word,en_word)] # sum(t(s|e))--sum of probability of swedish word with each the english word in the sentence
            for en_word in en_words:
                nume= t[(sv_word,en_word)] # t(s|e) - probability of swedish word with that english word
                align=nume/deno[sv_word] # alignment probability
                count_sv_en[(en_word,sv_word)]+=align #updating pseudocount
                count_en[(en_word)]+= align #updating pseudocount
        for sv_word in sv_words:
            for en_word in en_words:
                t[(sv_word,en_word)] = count_sv_en[(en_word,sv_word)]/count_en[(en_word)]
    
tp={}
eng =[]
for key in t.keys():
    if key[1] == 'european':
        eng.append(key)        
bestprob = 0
bestmatch =None
for e in eng:
    tp[e] = t[e]

marklist = sorted(tp.items(), key=lambda x:x[1])
marklist.reverse() 
sortdict = dict(marklist[:10])
print(sortdict) #print the 10 most probable words that are aligned with 'european' in descending order                  


# **Decoding**

# In[17]:


# Based on the translation modelling we get the n most probable translations
def get_most_probable_translation(word, t, number):  
  tp={}
  eng =[]
  for key in t.keys():
      if key[0] == word:
          eng.append(key)        
  bestprob = 0
  bestmatch =None
  for e in eng:
      tp[e] = t[e]

  marklist = sorted(tp.items(), key=lambda x:x[1])
  marklist.reverse()
  #print(marklist)
  sortdict = dict(marklist[:number])
  return sortdict

# These are the test sentences
# sentence = "ni har begärt en debatt i ämnet under sammanträdesperiodens kommande dagar ."
# sentence = "jag ber er resa er för en tyst minut ."
sentence = "ni känner till från media att det skett en rad bombexplosioner och mord i sri lanka ."

sentence = sentence.split(" ")
translation = []

for ind, word in enumerate(sentence):
  print(translation)
  if ind == 0: # if we have the first word, we add its most probable translation to the string
    possible_words = get_most_probable_translation(word, t, 1)
    for k in possible_words.keys():
     translation.append(k[1])
  else:
    possible_words = get_most_probable_translation(word, t, 20) # We get the n most probable translations
    word = ""
    highest_score = 0
    for k in possible_words.keys(): # For each translation we get its probability, we keep the most probable word
      if k[1] != None:
        sentence_to_test = translation[-1]+" "+k[1]
        
        prob = mle(sentence_to_test, data_sets)
        if prob > highest_score:
          word = k[1]
          highest_score = prob
        if word == "": # If the word is empty we assign the first translation i.e. the "best" translation to it such that if none of the translations are valid, we simply translate it.
          word = list(possible_words.keys())[0][1]
          print(f"Word: {word}")

    #print(word)
    translation.append(word)

print(translation)

