import math
from itertools import chain
import time
import numpy as np

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
from inverted_index_colab import MultiFileReader


class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = sum(index.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, queries, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        # """
        s = set()
        s.update(queries)
        self.idf = self.calc_idf(list(s))

        dic = {}
        mylist = []
        terms = queries
        candidates = get_candidate_documents(terms, self.index)
        for doc_id in candidates:
            scoredoc = self._score(terms, doc_id)
            mylist.append((doc_id, scoredoc))
        mylist = sorted(mylist, key=lambda tup: tup[1], reverse=True)
        mylist = mylist[0:N]

        # YOUR CODE HERE
        return mylist

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.index.DL[doc_id]

        for term in query:
            if term in self.index.posting_locs.keys():
                 try:
                    term_frequencies = read_posting_list(self.index,term)
                    term_frequenciesdict = dict(term_frequencies)
                    if doc_id in term_frequenciesdict.keys():
                        freq = term_frequencies[doc_id]
                        numerator = self.idf[term] * freq * (self.k1 + 1)
                        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                        score += (numerator / denominator)
                 except:
                    pass
        return score


def get_candidate_documents(query_to_search,index):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    list of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = []
    for term in np.unique(query_to_search):
        try:
            current_list = read_posting_list(index,term)
            candidates += current_list
        except:
            pass
    candi = [tup[0] for tup in candidates]
    return np.unique(candi)

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
from contextlib import closing

def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list