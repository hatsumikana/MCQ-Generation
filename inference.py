from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from nltk import sent_tokenize
import nltk
import re
import spacy
import pytextrank
from transformers import T5Tokenizer, T5ForConditionalGeneration
from QuestionGenerator import QuestionGenerator
import random
from collections import OrderedDict
from sense2vec import Sense2Vec
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import random
import pandas as pd 
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")
nltk.download('punkt')

def semanticsearch(para, topic, k=5):

    """
    Takes paragraph and it's topic as its input.
    Extracts top 5 best sentences best linked to the topic.

    Parameters
    ----------
    para : string
        Text for the pare
    topic : string
        Text for the topic
    k : int
        Number of sentences to be selected 
        (default value is 5)

    Returns
    -------
    data : list of strings
        List of k sentences best linked topic
          
    """

    # Separates the sentences in the given para
    passage = sent_tokenize(para)

    # Loads the Bi-Encoder Model 
    bi_encoder = SentenceTransformer('msmarco-distilbert-base-v4')
    bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens

    # Loads the Cross-Encoder Reranker
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # embedding the paragraph and topic
    corpus_embeddings = bi_encoder.encode(passage, convert_to_tensor=True, show_progress_bar=True)
    question_embedding = bi_encoder.encode(topic, convert_to_tensor=True)
    
    # enables gpu if available
    if torch.cuda.is_available():
        question_embedding = question_embedding.cuda()

    # Select 2 * k sentences from the para using the bi-encoder
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=k*2)
    hits = hits[0]  

    # Reranks the selected sentences and helps select k sentences
    cross_inp = [[topic, passage[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    results = []
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:min(k,len(passage))]:
        results.append(passage[hit['corpus_id']].lower())
    
    return results



def find_keyphrase(para):

    """
    Selects the best keyword from the text

    Parameters
    ----------
    sent : string
        Sentence

    Returns
    -------
    word : string
        the best keyword from the text
          
    """

    # To avoid alpha and numeric values as the keywords
    doc = nlp(para)
    i = 0
    while i < len(doc._.phrases):
        word = doc._.phrases[0].text
        word = re.sub(r'[^\w\s]', ' ', word)
        temp = re.sub(' ', '', word)
        if temp.isnumeric():
            return word
        elif temp.isalpha():
            return word
        i += 1
    
    return False

def sent_ans_extractor(para, topic , k=5):
    
    """
    Takes paragraph and it's topic as its input.
    Extracts top 5 best sentences best linked to the topic.
    Selects the best keyword from the text

    Parameters
    ----------
    para : string
        Text for the pare
    topic : string
        Text for the topic
    k : int
        Number of sentences to be selected 
        (default value is 5)

    Returns
    -------
    data : list of strings
        List of k sentences best linked topic
          
    """
    new_sents = []
    words = []

    sents = semanticsearch(para, topic, k)
    for i in sents:
        word = find_keyphrase(i)
        if word == False:
            continue
        new_sents.append(i)
        words.append(word)
    return new_sents, words

def question_generator(sentence, answer):
    
    """
    Takes the sentence and the answer as the input
    to generate a question

    Parameters
    ----------
    sentece : string
        Text for the sentence
    answer : string
        Text for the answer

    Returns
    -------
    question : string
        Text for the string
          
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    my_model_params = { "MODEL_DIR": "./outputs/final/", 
                    "MAX_SOURCE_TEXT_LENGTH": 75
                    } 

    # encode text
    tokenizer = T5Tokenizer.from_pretrained(my_model_params["MODEL_DIR"])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<answer>', '<context>']})

    # using T5 with language model layer
    model = T5ForConditionalGeneration.from_pretrained(my_model_params["MODEL_DIR"])
    model = model.to(device)  
    
    # prepare input
    qg_input = f"<answer> {answer} <context> {sentence}"

    # generate question
    qg =  QuestionGenerator(model, tokenizer, device, max_input_length=75, max_output_length=25)
    question = qg.generate(source_text=qg_input)
    return question


def find_related_word_online(word):

    """
    Takes word/phrase as an input and generates similar words/phrases 
    aka distractors using webscrapping from relatedwords.org website

    Parameters
    ----------
    word : string
        input words/phrases to generate distractors for

    Returns
    -------
    words : list of strings
        List of distractors for the given input
          
    """
    r = requests.get("https://relatedwords.org/relatedto/" + word)
    soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
    sent = soup.prettify()[soup.prettify().find('"terms"'):]
    words = []
    count = 0
    while count != 20:
        ind1 = sent.find('"word":')+8
        ind2 = sent[ind1:].find('"')+ind1
        words.append(sent[ind1:ind2])
        sent = sent[ind2:]
        count+=1
    return words

def sense2vec_get_words(word,s2v):
    
    """
    Takes word/phrase as an input and generates similar words/phrases 
    aka distractors using sense2vec

    Parameters
    ----------
    word : string
        input words/phrases to generate distractors for
    s2v : Module instance from Sense2Vec class
        Module instance from Sense2Vec class to generate distractor

    Returns
    -------
    distractors : list of strings
        List of distractors for the given input
          
    """
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
    
    """
    Takes word/phrase as an input and generates similar words/phrases aka distractors

    Parameters
    ----------
    word : string
        input words/phrases to generate distractors for

    Returns
    -------
    distractors : list of strings
        List of distractors for the given input
          
    """
    distractors = []
    if word.isnumeric():
        if len(word) == 4:
            # if 4-digit number --> assume it's a year --> add/subtract random number btw 1-10
            randomlist = random.sample(range(-10, 10), 20)
            for num in randomlist:
                distractors.append(str(int(word) + num))
            return distractors
        else:
            # else if other number --> add/subtract random number --> don't change +ve or -ve
            randomlist = random.sample(range(-1000, 1000), 20)
            for num in randomlist:
                distractors.append(str(int(word) + num))
            return distractors

    else:
        word = word.lower()
        s2v = Sense2Vec().from_disk('s2v_old')
        distractors = sense2vec_get_words(word, s2v)
        return distractors

def distractor_generator(answer):

    """
    Takes word.phrase as an input and generates similar words/phrases aka distractors

    Parameters
    ----------
    answer : string
        input words/phrases to generate distractors for

    Returns
    -------
    all_distractors : list of strings
        List of distractors for the given input
          
    """
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    all_distractors = []
    dis = {}
    for word in answer.split(" "):
        distractor = get_distractors(word)
        dis[word] = distractor
    
    while len(all_distractors) < 20:
        distr = ""
        for word in dis:
            rand_idx = int(random.random() * len(dis[word]))
            distr += dis[word][rand_idx] + " "
        if not distr in all_distractors:
            all_distractors.append(distr[:-1])

    cross_inp = [[answer, all_distractors[i]] for i in range(len(all_distractors))]
    cross_scores = cross_encoder.predict(cross_inp)

    results = []
    for i in sorted(range(len(cross_scores)), key=lambda i: cross_scores[i])[-3:]:
        results.append(all_distractors[i])

    return results

def MCQ_generator(para, topic , k=5):

    """
    Takes paragraph and it's topic as its input.
    Generates questions, correct and incorrect answers for MCQs

    Parameters
    ----------
    para : string
        Text for the pare
    topic : string
        Text for the topic
    k : int
        Number of sentences to be selected 
        (default value is 5)

    Returns
    -------
    data : list of strings
        List of k sentences best linked topic
          
    """
    sents, correct_ans = sent_ans_extractor(para, topic , k=5) # extracts the sentences and keywords
    questions = []
    all_ans = []
    for i,j in zip(sents, correct_ans):
        ques = question_generator(i, j) # generates the question
        questions.append(ques)

        temp = distractor_generator(j) # generates the distractors
        temp.append(j)

        random.shuffle(temp)
        all_ans.append(temp)
        
        # print('sentence :',i)
        # print('question :',ques)
        # print('all answer :',temp)
        # print('correct answer:',j)
        # print('\n')

    return questions, all_ans, correct_ans