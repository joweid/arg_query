import json
import os
import numpy as np

from itertools import islice, tee

from rank_bm25 import BM25Okapi

import BM25F.core
import BM25F.en
import BM25F.exp
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer, util

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def jaccard(query, text): 
    query_4_grams = set(list(zip(*(islice(seq, index, None) for index, seq in enumerate(tee(query.lower(), 4))))))
    text_4_grams = set(list(zip(*(islice(seq, index, None) for index, seq in enumerate(tee(text.lower(), 4))))))
    
    return float(len(query_4_grams.intersection(text_4_grams)) / len(query_4_grams.union(text_4_grams)))


def n_most_relevant(most_similar_indices, corpus, limit):
    n_most_relevant = []
    l = 0

    for i in most_similar_indices:
        n_most_relevant.append(corpus[i])

        l += 1
        if l == limit:
            break

    return n_most_relevant


def build_bm25f_structure(data, tokenizer):
    text_dict = {}
    bds = []
    bj = BM25F.exp.bag_jag()

    i = 0
    for line in data['lines']:
        for argument in line['arguments']:
            i += 1
            bd = BM25F.exp.bag_dict().read(tokenizer, {
                'id': str(i),
                'original_query': str(line['query']).lower(),
                'text': str(argument['text']).lower()
            })
            bds.append(bd)
            bj.append(bd)
            
            text_dict[i] = argument['text'].lower()
    
    return (bds, bj, text_dict)

def main():
    file = open('arguments.json')
    data = json.load(file)
    
    # Define queries
    queries = [
        'Are genetically modified foods healthy?',
        'Is veganism saving the planet?',
        'Do we need feminism?',
        'Should abortion be legal?',
        'Is communism good for society?',
        'Is climate change a real problem?',
        'Are there more than two genders?'
    ]

    # Initialize BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize BM25F
    stops = set(stopwords.words('english'))
    tokenizer = BM25F.en.Tokenizer(token_filter=stops)
    bds, bj, text_dict = build_bm25f_structure(data, tokenizer)


    # Query Matching
    json_output = {'output': []}
    for query in queries:
        #JACCARD
        j = {'query': query, 'methods': []}
        arguments_matching_to_query = []
    
        for line in data['lines']:
            for argument in line['arguments']:
                text = argument['text']
                jaccard_text = jaccard(query.lower(), text.lower())
                jaccard_original_query = jaccard(query.lower(), line['query'].lower())
                jaccard_mean = ((jaccard_text + jaccard_original_query) / 2)
                
                arguments_matching_to_query.append({'line': line['id'], 'original_query': line['query'],
                                                    'text': line['text'], 'clusters': str(line['clusters']),
                                                    'matched_argument': argument, 'score': jaccard_mean})
                
        top_10 = sorted(arguments_matching_to_query, key=lambda k: k['score'], reverse=True)[:10]
        j['methods'].append({'jaccard': top_10})


        #BM25F
        bm25f_query = BM25F.exp.bag_of_words().read(tokenizer, query.lower())

        boost = BM25F.core.param_dict(default=1.0)
        boost['original_query'] = 100
        boost['text'] = 0.1

        k1 = 2.0

        b = BM25F.core.param_dict(default=0.75)
        b['original_query'] = 0.50
        b['text'] = 1.00

        scorer = BM25F.core.batch('id', bm25f_query, bj, boost, k1, b)
        j['methods'].append({'bm25f': [text_dict[int(index)] for index in scorer.top(10, bds)]})


        #BERT
        query_embedding = model.encode(query.lower())
        arguments_matching_to_query = []

        for line in data['lines']:
            original_query_embedding = model.encode(line['query'])
            for argument in line['arguments']:
                text = argument['text']
                argument_embedding = model.encode(text.lower())

                cos_sim_text_output = str(util.cos_sim(query_embedding, argument_embedding)[0][0])
                cos_sim_original_query_output = str(util.cos_sim(query_embedding, original_query_embedding)[0][0])
                cosinus_similarity_text = float(cos_sim_text_output[cos_sim_text_output.rfind('(')+1:cos_sim_text_output.rfind(')')].strip())
                cosinus_similarity_original_query = float(cos_sim_original_query_output[cos_sim_text_output.rfind('(')+1:cos_sim_original_query_output.rfind(')')].strip())

                cosinus_similarity_mean = ((cosinus_similarity_text + cosinus_similarity_original_query) / 2)
                arguments_matching_to_query.append({'line': line['id'], 'original_query': line['query'],
                                                    'text': line['text'], 'clusters': str(line['clusters']), 
                                                    'matched_argument': argument, 'score': cosinus_similarity_mean})   

        top_10 = sorted(arguments_matching_to_query, key=lambda k: k['score'], reverse=True)[:10]   
        j['methods'].append({'bert': top_10})
        

        #TF-IDF
        corpus = []
        for line in data['lines']:
            for argument in line['arguments']:
                corpus.append(argument['text'].lower())
        
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        query_vector = tfidf_vectorizer.transform([query.lower()])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

        most_similar_indices = cosine_similarities.argsort()[0][::-1]
        
        # Define the n most relevant arguments e.g. 5 most relevant arguments
        relevant_arguments = n_most_relevant(most_similar_indices, corpus, 10)

        j['methods'].append({'tf-idf': relevant_arguments})
        

        json_output['output'].append(j)
    

    json_data = json.dumps(json_output, indent=2)
    with open('output.json', 'w') as f:
        f.write(json_data)
    
                
if __name__ == '__main__':
    main()

