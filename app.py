from flask import Flask, request, render_template
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
tfidf = TfidfVectorizer(ngram_range=(3, 3))

def preprocess(user_text):
    out = []
    user_text = [user_text]
    for i in user_text:
        ind = ' '.join(re.findall(r'\w+|\d+', i))
        ind = ind.lower()
        ind = word_tokenize(ind)
        ind = [lemmatizer.lemmatize(word) for word in ind if word not in stopwords.words('english')]
        out.append(" ".join(ind))
    return out

def plagiarism_check(user_text, original_text):
    a = tfidf.fit_transform(preprocess(user_text))
    b = tfidf.transform(preprocess(original_text))
    return cosine_similarity(a, b)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/check', methods=['POST'])
def check_plagiarism():
    text1 = request.form['text1']
    text2 = request.form['text2']

    similarity_score = plagiarism_check(text1, text2)

    return render_template('result.html', similarity_score=similarity_score[0][0])


if __name__ == '__main__':
    app.run(debug=True)




