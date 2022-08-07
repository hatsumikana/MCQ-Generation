import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import random
import numpy as np
import pandas as pd 
from scipy import spatial

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

def glove_get_words(word):
    embeddings_dict = {}
    with open("../data/glove.6B.50d.txt", 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return find_closest_embeddings(word, embeddings_dict)

def find_closest_embeddings(word, embeddings_dict):
    embedding = embeddings_dict[word]
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

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
        distractors = glove_get_words(word)
        return distractors

# word = "knowles"
# s2v = Sense2Vec().from_disk('s2v_old')
# distractors = sense2vec_get_words(word,s2v)

# print ("Distractors for ",word, " : ")
# print (distractors)

if __name__=="__main__":
    df = pd.read_csv("../data/hatsumi_new.csv")
    all_answers = df['answer']
    

    # answer = "singing and dancing"
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
            if distr not in all_distractors:
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