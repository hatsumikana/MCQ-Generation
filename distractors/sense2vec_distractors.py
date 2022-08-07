
from collections import OrderedDict
from sense2vec import Sense2Vec
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import random
import pandas as pd 

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

def sense2vec_get_words(word,s2v):
    output = []
    word = word.lower()
    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)

    if sense == None:
        return find_related_word_online(word)

    most_similar = s2v.most_similar(sense, n=20)
 
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ").lower()
        if append_word.lower() != word:
            if sense.split("|")[1] == each_word[0].split("|")[1]:
                output.append(append_word.title().lower())

    out = list(OrderedDict.fromkeys(output))
    return out

def get_distractors(word):
    distractors = []
    if word.isnumeric():
        if len(word) == 4:
            # if 4-digit number --> assume it's a year --> add/subtract random number btw 1-10
            randomlist = random.sample(range(-10, 10), 3)
            for num in randomlist:
                distractors.append(str(int(word) + num))
            return distractors
        else:
            # else if other number --> add/subtract random number --> don't change +ve or -ve
            randomlist = random.sample(range(-1000, 1000), 3)
            for num in randomlist:
                distractors.append(str(int(word) + num))
            return distractors

    else:
        word = word.lower()
        s2v = Sense2Vec().from_disk('../s2v_old')
        distractors = sense2vec_get_words(word, s2v)
        return distractors


if __name__=="__main__":
    # df = pd.read_csv("../data/hatsumi_new.csv")
    # all_answers = df['answer']
    

    all_answers = ["singing and dancing"]
    for answer in all_answers:
        all_distractors = []
        dis = {}
        for word in answer.split(" "):
            distractor = get_distractors(word)
            dis[word] = distractor
        
        while len(all_distractors) < 3:
            distr = ""
            for word in dis:
                rand_idx = int(random.random() * len(dis[word]))
                distr += dis[word][rand_idx] + " "
            if not distr in all_distractors:
                all_distractors.append(distr[:-1])
        print(answer, all_distractors)

# if __name__=="__main__":
#     answer = "beautiful liar"
#     all_distractors = []
#     dis = {}
#     for word in answer.split(" "):
#         distractor = get_distractors(word)
#         dis[word] = distractor
    
#     while len(all_distractors) < 3:
#         distr = ""
#         for word in dis:
#             rand_idx = int(random.random() * len(dis[word]))
#             distr += dis[word][rand_idx] + " "
#         if distr not in all_distractors:
#             all_distractors.append(distr[:-1])
#     print(answer, all_distractors)