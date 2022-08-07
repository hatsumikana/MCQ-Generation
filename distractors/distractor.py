import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import random
import requests
from bs4 import BeautifulSoup

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

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return find_related_word_online(word)
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


if __name__=="__main__":
    # df = pd.read_csv("../data/hatsumi_new.csv")
    # ans = df['answer']
    ans = ["singing and dancing"]
    for a in ans:
        all_distractors = []
        dis = {}
        for word in a.split(" "):
            stemmed_word = lemmatizer.lemmatize(word)
            synset_to_use = wn.synsets(stemmed_word)
            if synset_to_use == []:
                continue
            distractors = get_distractors_wordnet(synset_to_use[0], word)
            dis[word] = distractors
        
        while len(all_distractors) < 3:
            distr = ""
            for word in dis:
                rand_idx = int(random.random() * len(dis[word]))
                distr += dis[word][rand_idx] + " "
            if not distr in all_distractors:
                all_distractors.append(distr[:-1])
        print(a, all_distractors)

            
        print()
        
            
        