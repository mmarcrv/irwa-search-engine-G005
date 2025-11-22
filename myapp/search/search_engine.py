import random
import numpy as np

from myapp.search.objects import Document
from myapp.search.algorithms import search_in_corpus

def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res


class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query,  query_terms, corpus, corpus_dataframe):
        print("Search query:", search_query)

        results = []
        ### You should implement your search logic here:
        # results = dummy_search(corpus, search_id)  # replace with call to search algorithm

        results = search_in_corpus(search_query, query_terms, corpus, corpus_dataframe)
        
        return results
