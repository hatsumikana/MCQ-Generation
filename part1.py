from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from nltk import sent_tokenize
import nltk
import re
import spacy
import pytextrank
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

def sent_ans_extractor(para, title , k=5):
    
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

    sents = semanticsearch(para, title, k)
    for i in sents:
        word = find_keyphrase(i)
        if word == False:
            continue
        new_sents.append(i)
        words.append(word)
    return new_sents, words