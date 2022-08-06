import nltk
from sympy import hyper
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from random import randint
import spacy
pos = spacy.load("en_core_web_sm")
import requests
from bs4 import BeautifulSoup
import random

import pandas as pd 

lemmatizer = WordNetLemmatizer()

def find_related_word_online(word):
    r = requests.get("https://relatedwords.org/relatedto/" + word)
    soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
    sent = soup.prettify()[soup.prettify().find('"terms"'):]
    words = []
    count = 0
    while count != 3:
        ind1 = sent.find('"word":')+8
        ind2 = sent[ind1:].find('"')+ind1
        words.append(sent[ind1:ind2])
        sent = sent[ind2:]
        count+=1
        
    return words

def get_distractors(syn, word):
    distractors = []
    if word.isnumeric():
        if len(word) == 4:
            # if 4-digit number --> assume it's a year --> add/subtract random number btw 1-10
            for i in range(3):
                distractors.append(str(int(word) + randint(-10,10)))
            return distractors
        else:
            # else if other number --> add/subtract random number --> don't change +ve or -ve
            for i in range(3):
                distractors.append(str(int(word) + randint(-1000,1000)))
            return distractors
    
    elif word.isalnum():
        distractors = []
        word = word.lower()
        word_pos = pos(word)
        
        try:
            word_pos = word_pos.ents[0].label_
            return find_related_word_online(word)
        
        except IndexError:
            word_pos = word_pos[0].tag_
            print(word, "->", word_pos)

            count = 0
            for s in syn:
                hypernym = s.hypernyms()
                print("hype",hypernym)
                if len(hypernym) == 0:
                    continue
                for item in hypernym[0].hyponyms():
                    hyp = item.lemmas()[0].name()
                    if hyp == word:
                        # print(hyp, word)
                        continue
                    hyp = hyp.replace("_"," ")
                    hyp = " ".join(w.capitalize() for w in hyp.split())
                    hyp_pos = pos(hyp)
                    hyp_pos = hyp_pos[0].tag_
                    # print(word_pos, hyp_pos, hyp.lower())
                    if hyp_pos == word_pos and hyp not in distractors:
                        distractors.append(hyp)
                        count += 1
                    if count == 3: 
                        return distractors
        if len(distractors) == 0:
            return find_related_word_online(word)
        return distractors


# word = "neural network"
# stemmed_word = lemmatizer.lemmatize(word)
# synset_to_use = wn.synsets(stemmed_word)
# print(get_distractors(synset_to_use, word))

if __name__=="__main__":
    df = pd.read_csv("./data/qg_train.csv")
    all_answers = df['answer'][:10]
    all_distractors = []
    print(all_answers)

    for ans in all_answers:
        distractor = {}
        distr_phrases = []

        for word in ans.split(" "):
            # print(word)
            stemmed_word = lemmatizer.lemmatize(word)
            synset_to_use = wn.synsets(stemmed_word)
            d = get_distractors(synset_to_use, word)
            if d is None:
                continue
            distractor[word] = d
        
        while len(distr_phrases) < 3:
            distr = ""
            for k in distractor:
                rand_idx = int(random.random() * len(distractor[k]))
                distr += distractor[k][rand_idx] + " "
            if distr not in distr_phrases:
                distr_phrases.append(distr[:-1])
    
        all_distractors.append(distr_phrases)
    print(all_distractors)
                
        
