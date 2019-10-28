from flask import Flask
from flask import render_template
from flask import request
from search import Search
from collections import Counter
from math import log10, log
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from numpy import dot
from numpy.linalg import norm
import csv

import nltk
nltk.download('stopwords')

class Engine:
    def __init__(self):
        self.movie_data = self.load_data('data/movie.csv')
        self.stop_words = stopwords.words('english')
        self.unique_words = set()
        self.word_document_frequency = {}
        self.word_frequency_documents = {}
        self.document_lengths = {}

    # return the dataframe
    def load_data(self, path):
        data = []
        with open('static/movie.csv', 'r', encoding='utf-8') as csvfile:
            moviereader = csv.reader(csvfile)
            index = 0
            for movie_line in moviereader:
                if index == 500:
                    break
                try:
                    if (movie_line[20] == 'title'):
                        pass
                    else:
                        data.append({'overview': movie_line[9], 'id': movie_line[5], 'title': movie_line[20],
                                          'original_title': movie_line[8], 'poster_path': movie_line[11]})
                except:
                    pass
                index += 1
        return data

    def tokenize(self, document):
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

        terms = tokenizer.tokenize(document.lower())
        filtered = [word for word in terms if not word in self.stop_words]
        return filtered

    # return stemmed tokens of document
    def stem_tokenize(self, document):
        # terms = document.lower().split()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

        terms = tokenizer.tokenize(document.lower())
        filtered = [stemmer.stem(word) for word in terms if not word in self.stop_words]
        return filtered

    # return tokens of document
    def tokenize_title(self, document):
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

        terms = tokenizer.tokenize(document.lower())
        filtered = [word for word in terms]
        return filtered

    # return stemmed tokens of document
    def stem_tokenize_title(self, document):
        # terms = document.lower().split()
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

        terms = tokenizer.tokenize(document.lower())
        filtered = [stemmer.stem(word) for word in terms]
        return filtered

    def init(self):
        length = len(self.movie_data)

        for index in range(length):
            row = self.movie_data[index]

            document = row['overview']
            title = row['title']
            movie_id = row['id']

            # stemmed tokens from the document
            tokens = self.stem_tokenize(str(document))
            tokens = tokens + self.stem_tokenize_title(str(title))

            # add to unique tokens
            self.unique_words = self.unique_words.union(set(tokens))

            # add to postings
            for term in set(tokens):
                if term not in self.word_frequency_documents:
                    self.word_frequency_documents[term] = {}

                self.word_frequency_documents[term][movie_id] = tokens.count(term)

            # add to document frequency
            for term in self.word_frequency_documents:
                self.word_document_frequency[term] = len(self.word_frequency_documents[term])

            # add to doc lengths
            self.document_lengths[movie_id] = len(tokens)

    def calculate_tf(self, query_string):
        query_tokens = self.stem_tokenize(query_string)
        tf_scores = {}
        for token in query_tokens:

            for movie_id in self.word_frequency_documents[token]:

                if movie_id not in tf_scores:
                    tf_scores[movie_id] = {}

                tf_scores[movie_id][token] = self.word_frequency_documents[token][movie_id] / self.document_lengths[movie_id]
        return tf_scores

    def calculate_idf(self, query_string):
        query_tokens = self.stem_tokenize(query_string)
        idf_scores = {}
        for token in set(query_tokens):

            if token in self.word_document_frequency:
                idf_scores[token] = log10(len(self.document_lengths) / self.word_document_frequency[token])
            else:
                idf_scores[token] = 0
        return idf_scores

    def get_query_vector(self, query_string):
        query_tokens = self.stem_tokenize(query_string)
        query_length = len(query_tokens)

        vector = []
        for term in query_tokens:
            term_count = query_tokens.count(term)
            term_F = term_count / query_length
            query_idf = 0
            if term in query_tokens:
                query_idf = log10(query_length / term_count)

            tf_idf = term_F * query_idf
            vector.append(tf_idf)

        return vector

    def get_document_vector(self, query_string, movie_id):
        query_tokens = self.stem_tokenize(query_string)
        document_vector = []
        for query_term in query_tokens:
            if query_term in self.unique_words:
                if movie_id in self.word_frequency_documents[query_term]:
                    tf = self.word_frequency_documents[query_term][movie_id] / self.document_lengths[movie_id]
                    idf = 0
                    if query_term in self.word_document_frequency:
                        idf = log10(len(self.document_lengths) / self.word_document_frequency[query_term])
                    else:
                        idf = 0
                    tf_idf = tf * idf
                    document_vector.append(tf_idf)
                else:
                    document_vector.append(0)
            else:
                document_vector.append(0)

        return document_vector

    def generate_tf_idf_table(self, tf_scores, idf_scores, movie_id, query_tokens):
        table = '<table class="table table-striped"><tr>'
        table += '<th class="col_token">Token</th><th class="col_tf">TF</th><th class="col_idf">IDF</th><th class="col_tf_idf">TF * IDF</th>'
        table += '</tr>'
        final_score = 0.0
        for token in query_tokens:
            tf_score = tf_scores[movie_id][token] if token in tf_scores[movie_id] else 0.0
            idf_score = idf_scores[token] if token in idf_scores else 0.0
            final_score = final_score + (tf_score * idf_score)
            table += '<tr><td>' + token + '</td><td>' + str(tf_score) + '</td><td>' + str(
                idf_score) + '</td><td>' + str(tf_score * idf_score) + '</td></tr>'
        table += '</table>'
        return table

    def query_movie(self, query_string):
        tf_scores = self.calculate_tf(query_string)
        idf_scores = self.calculate_idf(query_string)

        query_tokens = self.stem_tokenize(query_string)

        query_vector = self.get_query_vector(query_string)

        document_similarity = Counter()

        document_vectors = {}
        for movie_id in tf_scores:
            document_vectors[movie_id] = self.get_document_vector(query_string, movie_id)
            document_similarity += {movie_id: dot(query_vector, document_vectors[movie_id]) / (
                    norm(query_vector) * norm(document_vectors[movie_id]))}

        results = []
        for item in document_similarity.most_common(5):
            row = [d for d in self.movie_data if (d['id'] == item[0])][0]

            print(row)
            result = {
                "id": row['id'],
                "title_eng": row['title'],
                "title_orig": row['original_title'],
                "overview": row['overview'],
                "poster": "https://image.tmdb.org/t/p/w300_and_h450_bestv2" + row['poster_path'],
                "cosine_similarity": item[1],
                "score_table": self.generate_tf_idf_table(tf_scores, idf_scores, item[0], query_tokens),
            }

            results.append(result)

        return results, query_tokens


engine = Engine()
engine.init()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def movie_search():
    query_string = request.form['query_string']
    results, query_tokens = engine.query_movie(query_string)
    return render_template('index.html', query_string=query_string, query_tokens=query_tokens, res=results)


if __name__ == '__main__':
    app.run()
