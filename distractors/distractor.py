import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

import pandas as pd 

lemmatizer = WordNetLemmatizer()

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
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
    df = pd.read_csv("./data/qg_train.csv")
    ans = df['answer'][:10]
    
    for a in ans:
        print(a)
        print(nltk.pos_tag(a.split(" ")))
        for word in a.split(" "):
            stemmed_word = lemmatizer.lemmatize(word)
            synset_to_use = wn.synsets(stemmed_word)
            if synset_to_use == []:
                continue
            distractors = get_distractors_wordnet(synset_to_use[0], word)
            print(word, distractors)
            
        print()
        
        # if 4-digit number --> assume it's a year --> add/subtract random number btw 1-10
        # else if other number --> add/subtract random number --> don't change +ve or -ve
        # else if word --> check POS --> get distractors --> try to match POS 
        # if word is a NOUN --> entity recognizer (Check if word is a name) -->  
        