"""
GCP sentiment analysis and CCC clustering.... this cleaned file for demo
"""
import os
import re
import pandas as pd
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer
from google.cloud import language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

#setting base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  BASE_DIR + '/Comment Analysis-c92a8b9aee2a.json'

#read dataframe
data=pd.read_csv("/home/pearlsoft/Downloads/comments.csv",nrows=200)

#1.Data preprocessing
def spanish_to_eng(data3):
    """
    #1.1 Sapanish To English Translator
    """
    tranlator=Translator()
    translated_words=[]
    for element in data3['Comments']:
        translated_words.append(tranlator.translate(element).text)
    data3['Translated_words']=translated_words 

spanish_to_eng(data)

#1.2 stop words removal

stop_words = set(stopwords.words('english'))
stop_words.add('subject')
stop_words.add('http')
def remove_stopwords(Comments):
    """
    stop words removal
    """
    return " ".join([word for word in str(Comments).split() if word not in stop_words])
data['Translated_words'] =data['Translated_words'].apply(lambda x: remove_stopwords(x))

#1.3 Lemmitization

lemmatizer = WordNetLemmatizer()
def lemmatize_words(Comments):
    """
    lemmitization
    """
    return " ".join([lemmatizer.lemmatize(word) for word in Comments.split()])

data["Translated_words"] = data["Translated_words"].apply(lambda x: lemmatize_words(x))

#1.4 punctuation removal
data['Translated_words']=data['Translated_words'].apply(lambda x:re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])"," ",x))
data['Translated_words']=data['Translated_words'].apply(lambda x: re.sub(' +', ' ', x).lower())  #remove extra white spaces

data2=data.drop(['Communication','Compassion','Competence','Comments','Unnamed: 4'],axis=1)
"""
2. Sentiment Analysis
"""
def language_analysis(text):
    """
    sentiment analysis
    """
    client = language.LanguageServiceClient()
    document = language.Document(content=text.encode('utf-8'), type=language.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    compoundscore = sentiment.score
    if compoundscore > 0:
        commenttype = 'Positive'
    elif compoundscore < 0:
        commenttype = 'Negative'
    else:
        commenttype = 'Neutral'
    return commenttype

data1=data.drop(['Communication','Compassion','Competence','Unnamed: 4'],axis=1)
data1['sentiment']=data['Translated_words'].apply(lambda x:language_analysis(x))

#3.Clustering

#3.1 Standardize features
documents = data1['Translated_words'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

#3.2 Create meanshift object
cluster = AgglomerativeClustering(n_clusters=3)

#3.3 Train model
f=features.toarray()
model = cluster.fit(f)

#4. View sentimet,cluster
data1['clusters'] =model.labels_
data1['clusters'] = data1['clusters'].map({0:'competence',1:'compassion',2:'communication'})
# data2=data1.drop(['Translated_words'],axis=1)
print(data2.head(5))
data2.to_csv("orginal_ccc")

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# result_data=pd.read_csv("/home/pearlsoft/projects_pearl/orginal1")
# gkk = result_data.groupby(['clusters','sentiment']).count()