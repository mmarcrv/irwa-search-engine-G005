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


def get_top_n(bm25_model, corpus_dataframe, query_tokens, n=10):

    #we use the index to find the docs that contain each term
    docs_per_term = []
    for t in query_tokens:
        if t in index:
            term_docs = {posting[0] for posting in index[t]}
            docs_per_term.append(term_docs)
    
    docs_with_all_terms = set.intersection(*docs_per_term) #only keep the docs that contain all query terms
   
    if not docs_with_all_terms:
        print("No documents contain all query terms.")
        return []

    scores = np.array(bm25_model.get_scores(query_tokens))

    pid_to_df_index = {pid: i for i, pid in enumerate(corpus_dataframe["pid"])}

    filtered = []
    for pid in docs_with_all_terms:
        df_idx = pid_to_df_index.get(pid)
        if df_idx is not None:
            filtered.append((scores[df_idx], pid))

    filtered_scores_arr = np.array([score for score, _ in filtered])
    filtered_pids_arr   = np.array([pid    for _, pid    in filtered])

    top_n_local = np.argpartition(filtered_scores_arr, -n)[-n:]
    top_n_local = top_n_local[np.argsort(-filtered_scores_arr[top_n_local])]

    top_n_pids = filtered_pids_arr[top_n_local].tolist()

    return top_n_pids

def calculate_pepito_score(cosine_similarity, average_rating, discount, out_of_stock, cosine_factor=0.40, discount_factor=0.30, rating_factor=0.3, stock_penalty=0.9):
    # Normalize average_rating and dicount to have values between 0-1, we need matching scales to have a balanced score
    normalized_average_rating = average_rating / 5.0
    normalized_discount = discount / 100.0

    # Calculate pepito_score with the weights we have decided for each feature
    pepito_score = (
        cosine_factor * cosine_similarity +
        discount_factor * normalized_discount +
        rating_factor * normalized_average_rating
    )

    # we will apply the penalty if the item is not in stock
    if out_of_stock:
        pepito_score *= 1 - stock_penalty

    return pepito_score

def rank_documents_pepito(terms, docs, index, idf, tf, original_df, cosine_factor=0.40, discount_factor=0.30, rating_factor=0.3, stock_penalty=0.9):
    result_docs = []
    doc_scores = []

    #for each term in the query, obtain the set of documents that contain it
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

    # intersection: keep only documents that contain all query terms
    docs_with_all_query_terms = set.intersection(*docs_with_query_terms)
    if not docs_with_all_query_terms:
        print("No documents with all query terms")
        return result_docs, doc_scores

    #interested only on the element of the docVector corresponding to the query terms
    # remaining elements 0
    doc_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs_with_all_query_terms:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Calculate initial cosine similarity scores
    initial_cosine_scores = []
    for doc, curDocVec in doc_vectors.items():
        dot_product = np.dot(curDocVec, query_vector)
        doc_norm = la.norm(curDocVec)
        query_norm_val = la.norm(query_vector)
        if doc_norm > 0 and query_norm_val > 0:
            cosine_similarity = dot_product / (doc_norm * query_norm_val)
            initial_cosine_scores.append([cosine_similarity, doc])

    # Calculate Pepito Score for each document
    final_ranked_documents = []
    for cos_sim, doc_pid in initial_cosine_scores:
        doc_info = original_df[original_df['pid'] == doc_pid].iloc[0]
        average_rating = doc_info['average_rating']
        discount = doc_info['discount']
        out_of_stock = doc_info['out_of_stock']

        # this takes the default values but can be changed setting the factors to the desired ones
        pepito_score = calculate_pepito_score(cos_sim, average_rating, discount, out_of_stock, cosine_factor, discount_factor, rating_factor, stock_penalty)
        final_ranked_documents.append([pepito_score, doc_pid])

    # Rank documents by Pepito Score
    final_ranked_documents.sort(key=lambda x: x[0], reverse=True)

    result_docs = [x[1] for x in final_ranked_documents]
    doc_scores = final_ranked_documents

    return result_docs, doc_scores

def search_pepito(query_tokens, index, original_df, idf, tf, cosine_factor=0.40, discount_factor=0.30, rating_factor=0.3, stock_penalty=0.9):
    docs = set()

    for term in query_tokens:
        if term in index:
            term_docs = {posting[0] for posting in index[term]}
            docs.update(term_docs)

    docs_list = list(docs)
    ranked_docs, doc_scores = rank_documents_pepito(query_tokens, docs_list, index, idf, tf, original_df, cosine_factor, discount_factor, rating_factor, stock_penalty)
    return ranked_docs, doc_scores

def search_in_corpus(query, query_terms, corpus, corpus_dataframe, index, tf, idf, bm25, selected_engine):
    # 1. create create_tfidf_index
    # not in this function to avoid repeating this step for each query search

    # 2. apply ranking
    if selected_engine == "tfidf":
        ranked_docs, doc_scores = search_tf_idf(query_terms, index, idf, tf)
    elif selected_engine == "bm25":
        ranked_docs = get_top_n(bm25, corpus_dataframe, query_terms, n=10, index=index)
    else:
        # pepitoooo
        ranked_docs, doc_scores = search_pepito(query_terms, index, corpus_dataframe, idf, tf)

    print(f"Top 5 results:\n")
    for i, pid in enumerate(ranked_docs[:5]):
        title = corpus_dataframe[corpus_dataframe['pid'] == pid]['title'].iloc[0]
        print(f"{i+1}. document_id: {pid}\n   Title: {title}\n")

    ranked_documents = [corpus[pid] for pid in ranked_docs]

    return ranked_documents
