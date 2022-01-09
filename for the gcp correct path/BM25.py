import math
from collections import defaultdict
from contextlib import closing

import math

from inverted_index_colab import MultiFileReader

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    try:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
          doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
          tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
          posting_list.append((doc_id, tf))
    except:
          pass
    return posting_list



class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.
        The frequnecy is normilzied by the max term frequency for each document.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    """

    def __init__(self, inverted, Q, k1=1.5, b=0.75):

        self.query = Q
        self.inverted = inverted
        self.b = b
        self.k1 = k1
        self.N_ = len(inverted.DL)
        self.Avg_doc_len = sum(inverted.DL) / len(inverted.DL)
        self.somthing = defaultdict(list)
        self.idf_dic = defaultdict(list)

    def calc_idf(self):
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
        # YOUR CODE HERE
        dic_idf = {}

        for i in range(0,len(self.query)):
            term = self.query[i]
            # if the term already added
            if term in dic_idf:
                continue

            else:
                # if the term exist in the corpus
                words_posting_lst = read_posting_list(self.inverted, term)
                if len(words_posting_lst) > 0:
                    dic_idf[term] = math.log1p(
                        ((self.N_ - len(words_posting_lst) + 0.5) / (len(words_posting_lst) + 0.5)) + 1)

        self.idf_dic = dic_idf

    def score_per_doc(self, id_doc):
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
        # YOUR CODE HERE
        score = 0

        for i in range(0, len(self.query)):
            term = self.query[i]
            curr_tf = ""
            if id_doc in self.tf_candi:
                for (word,tf) in self.tf_candi[id_doc]:
                    if word == term:
                        curr_tf = tf
                        continue
            if term in self.idf_dic:
                if curr_tf != "":
                    score += ((self.idf_dic[term]) * curr_tf * (self.k1 + 1)) / (
                            curr_tf + self.k1 * (
                            1 - self.b + ((self.b * self.inverted.DL[id_doc]) / self.Avg_doc_len)))
        return score

    def create_tf(self):
        c = {}
        for i in range(0, len(self.query)):
            term = self.query[i]
            words_posting_lst = read_posting_list(self.inverted, term)
            for couple in words_posting_lst:
                doc_id = couple[0]
                tf = couple[1]
                c[doc_id] = []
                c[doc_id].append((term, tf))
        self.tf_candi = c

    def score_calc_per_candidate(self):
        results = {}
        tf = 0
        df = 2
        if(len(results)==0):
            df = tf
        self.create_tf()
        self.calc_idf()
        for c in self.tf_candi:
            results[c] = self.score_per_doc(c)
        return results
