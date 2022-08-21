from array import array
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask,request,render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('web.html')
@app.route('/pre',methods=['POST'])
def pre():
    cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
   # val=list(request.form.values())
    ps = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    corpus=[]
    review = 'he is good'
 # print(review)
    review = review.lower()
    review = review.split()
  #print(review)
    review = [ps.lemmatize(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    X_fresh = cv.transform(corpus).toarray()
    print(X_fresh)
    classifier = pickle.load(open('class.pkl', "rb"))
    
    y_pred = classifier.predict(X_fresh)
    return render_template('web.html',prediction_text="predicted value is:{}".format(y_pred))
if __name__ == "__main__":
    app.run(debug=True)