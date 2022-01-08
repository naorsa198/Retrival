import math
from collections import Counter
import pandas as pd
from flask import Flask, request, jsonify
from inverted_index_colab import InvertedIndex as bodyINV, MultiFileReader
from inverted_index_title_colab import InvertedIndex as inverted_index_title
import pickle
import re
import nltk
nltk.download('stopwords')

from nltk.stem.porter import *
from nltk.corpus import stopwords


inverted = bodyINV.read_index('/content/postings_gcp', 'index')
pv_clean = '/content/pageviewsC.pkl'
# inverted = bodyINV.read_index('.', 'index')
inverted_index_title = inverted_index_title.read_index('/content/postings_gcp_titles/postings_gcp','title_index')

# pv_clean = 'pageviewsC.pkl'
with open(pv_clean, 'rb') as f:
  wid2pv = pickle.loads(f.read())

print(len(wid2pv))
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # collecting docs that query's words appear in
    query_binary_similarity = {}
    for word in tokenized_query:
        try:
            posting_lst = read_posting_list(inverted_index_title, word)
            if len(posting_lst) > 0:
                for doc in posting_lst:
                    if doc[0] in query_binary_similarity:
                        query_binary_similarity[doc[0]] += 1
                    else:
                        query_binary_similarity[doc[0]] = 1
        except:
            pass
    sorted_query_similarity = {k: v for k, v in
                               sorted(query_binary_similarity.items(), key=lambda item: item[1], reverse=True)}
    #------------------------------------------------------------------------body--------------------------------------------------------
    body = []
    cosimiMone = {}
    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify([])
    word_weight_in_query = {}
    cosinesim = {}
    word_appearance_in_query = Counter(tokenized_query)
    for word in word_appearance_in_query:
        word_weight_in_query[word] = word_appearance_in_query[word] / len(tokenized_query)

    # create the mona of the cosinsimilarity for each doc that have word from the query
    query_Dom = 0
    for word, weigth in word_weight_in_query.items():
        query_Dom = query_Dom + math.pow(weigth, 2)
        try:
          postlist = read_posting_list(inverted, word)
          for doc_id, dfi in postlist:
              cosimiMone[doc_id] = cosimiMone.get(doc_id, 0) + ((dfi / inverted.DL[doc_id]) * weigth)
        except:
          pass
    # make all the cosinsimilarity for each doc
    query_Dom_sqr = math.sqrt(query_Dom)
    resSorted = []
    for doc_id, mona in cosimiMone.items():
        dominator = math.sqrt(inverted.tfidf_dom[doc_id]*query_Dom)
        resultSimiliarirt = cosimiMone[doc_id] / dominator
        var = (doc_id, resultSimiliarirt)
        resSorted.append(var)

    resSorted = sorted(resSorted, key=lambda tup: tup[1], reverse=True)
    resSorted = resSorted[0:100]
    # ------------------------------------------------------------------------------------put together title and body-----------------------

    joinDic = {}
    for (doc_id, sim) in resSorted:
        joinDic[doc_id] = sim*0.65+sorted_query_similarity.get(doc_id,0)*0.35
    joinList = list(joinDic.items())
    joinList = sorted(resSorted,key=lambda tup: tup[1], reverse=True)

    joinList= joinList[0:100]
    finalList = [ (doc_id,inverted.id_to_title[doc_id])for doc_id, cs in joinList]
    return jsonify(finalList)



    # END SOLUTION

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    cosimiMone= {}
    tokenized_query = token_query(query)
    if len(tokenized_query)==0:
        return jsonify(res)
    word_weight_in_query = {}
    cosinesim = {}
    word_appearance_in_query = Counter(tokenized_query)
    for word in word_appearance_in_query:
        word_weight_in_query[word] = word_appearance_in_query[word] / len(tokenized_query)

    # create the mona of the cosinsimilarity for each doc that have word from the query
    query_Dom = 0
    for word,weigth in word_weight_in_query.items():
        query_Dom =query_Dom +math.pow(weigth,2)
        postlist = read_posting_list(inverted, word)
        for doc_id,dfi in postlist:
            cosimiMone[doc_id] =cosimiMone.get(doc_id,0) + ((dfi / inverted.DL[doc_id])*weigth)

    # make all the cosinsimilarity for each doc
    query_Dom_sqr = math.sqrt(query_Dom)
    resSorted=[]
    for doc_id,mona in cosimiMone.items():
         dominator = math.sqrt(inverted.tfidf_dom[doc_id]*query_Dom)
         resultSimiliarirt = cosimiMone[doc_id]/dominator
         var = (doc_id,resultSimiliarirt)
         resSorted.append(var)

    resSorted = sorted(resSorted,key=lambda tup: tup[1], reverse=True)
    resSorted = resSorted[0:100]
    resSorted_titles = [ (doc_id,inverted.id_to_title[doc_id])for doc_id, cs in resSorted]
    # print(resSorted_titles)
    # print(len(resSorted))
    # print(inverted.id_to_title[39])
    return jsonify(resSorted_titles)


def token_query(query):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    all_stopwords = english_stopwords.union(corpus_stopwords)
    tokensQ = [token.group() for token in RE_WORD.finditer(query.lower())]
    filteredQ = [tok for tok in tokensQ if tok not in all_stopwords]

    return filteredQ

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




@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # collecting docs that query's words appear in
    query_binary_similarity = {}
    for word in tokenized_query:
        posting_lst = read_posting_list(inverted_index_title, word)
        if len(posting_lst) > 0:
            for doc in posting_lst:
                if doc[0] in query_binary_similarity:
                    query_binary_similarity[doc[0]] += 1
                else:
                    query_binary_similarity[doc[0]] = 1

    if len(query_binary_similarity) == 0:
        return jsonify(res)

    sorted_query_similarity = {k: v for k, v in
                               sorted(query_binary_similarity.items(), key=lambda item: item[1], reverse=True)}
    for key in sorted_query_similarity:
        res.append((key, inverted_index_title.docTitle[key]))

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = token_query(query)

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    # wiki_ids = request.form['json']
    # wiki_ids = wiki_ids.replace('[','')
    # wiki_ids = wiki_ids.replace(']','')
    # wiki_ids = wiki_ids.split(',')
    wiki_ids = request.get_json()
    print(wiki_ids)
    print(type(wiki_ids))
    print(wiki_ids)
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    df = pd.read_csv('pageRank.csv')
    df.columns = ['doc_id','rank']
    print(df.head())
    for doc_id in wiki_ids:
        x = df.loc[df['doc_id']==int(doc_id)]
        dff = pd.DataFrame(x)
        y = x['rank'].values
        if(len(y)!= 0):
            res.append(y[0])
        else:
            res.append("")

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        if doc_id in wid2pv:
            res.append(wid2pv[doc_id])
        else:
            res.append("")
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)