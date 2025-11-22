from array import array
from collections import defaultdict
import collections
import math
import re
import string
import pandas as pd
import numpy as np
from numpy import linalg as la
from myapp.search.objects import Document
from nltk.stem import PorterStemmer

STOPWORDS = {
    "the", "and", "a", "an", "in", "on", "for", "to", "of", "with",
    "this", "that", "these", "those", "is", "are", "was", "were", "it",
    "be", "been", "at", "by", "from", "as", "but", "into", "about",
}

def token_cleaning_text(text):
    stemmer = PorterStemmer()
    #stop_words = set(stopwords.words("english"))
    #put everything in lowercase
    text=  text.lower()
    #get rid of punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))
    #remove special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    #tokenize the sentence
    text=  text.split()
    #filter words to delete stop words and stemming them
    text=[word for word in text if word not in STOPWORDS]
    text=[stemmer.stem(word) for word in text]
    return text


def create_index_tfidf(data):

    print("HOLA 1")
    print(data.columns)  
    
    print("HOLA 2")

    num_documents = len(data)
    index = defaultdict(list)
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)

    for _, row in data.iterrows():
        doc_id = row["pid"]
        terms = row["cleaned_title_description_extra_fields"]
        current_page_index = {}
        for position, term in enumerate(terms):
            if term is None:
                continue
            term = str(term)

            if term in current_page_index:
                current_page_index[term][1].append(position)
            else:
                current_page_index[term] = [doc_id, array('I', [position])]

        norm = 0
        for term, posting in current_page_index.items():

            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)
        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term
            tf[term].append(np.round(len(posting[1]) / norm, 4))
            #increment the doc frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)
    # Compute IDF
    for term in df:
        idf[term] = np.round(np.log(float(num_documents / df[term])), 4)
    return index, tf, df, idf


def rank_documents(terms, docs, index, idf, tf):
    result_docs = []
    doc_scores = []

    #for each term in the query (terms) obtain the set of documents that contain it
    docs_with_query_terms = []
    for term in terms:
        if term in index:
          #the term appears in at least one document because it's present in the index
          #get all documents where the term appears
          term_docs = {doc for doc, _ in index[term]}
          docs_with_query_terms.append(term_docs)

    # if no query terms are found in the index, there are no matching documents
    if not docs_with_query_terms:
        print("No query terms in the index")
        return result_docs, doc_scores

    # intersection, keep only documents that contain all query terms
    docs_with_all_query_terms = set.intersection(*docs_with_query_terms)
    if not docs_with_all_query_terms:
        print("No documents with all query terms")
        return result_docs, doc_scores

    #interested only on the element of the docVector corresponding to the query terms
    # remaining elements 0
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs_with_all_query_terms:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]  # check if multiply for idf

    # Calculate the score of each doc, cosine similarity between queyVector and each docVector

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    if len(result_docs) == 0:
        print("No results found.")

    return result_docs, doc_scores

def search_tf_idf(query, index, idf, tf):

    query = token_cleaning_text(query)
    docs = set()
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs=[posting[0] for posting in index[term]]

            # docs = docs Union term_docs
            docs |= set(term_docs)
        except:
            #term is not in index
            pass
    docs = list(docs)
    ranked_docs, doc_scores = rank_documents(query, docs, index, idf, tf)
    return ranked_docs, doc_scores



def search_in_corpus(query, corpus, corpus_dataframe):
    # 1. create create_tfidf_index
    index, tf, df, idf = create_index_tfidf(corpus_dataframe)
    print("\nCreated index, tf, df and idf...")

    # 2. apply ranking
    ranked_docs, doc_scores = search_tf_idf(query, index, idf, tf)
    
    print(f"Top 5 results:\n")
    for i, pid in enumerate(ranked_docs[:5]):
        title = corpus_dataframe[corpus_dataframe['pid'] == pid]['title'].iloc[0]
        print(f"{i+1}. document_id: {pid}\n   Title: {title}\n")

    ranked_documents = [corpus[pid] for pid in ranked_docs]

    return ranked_documents
