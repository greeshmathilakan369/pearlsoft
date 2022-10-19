#final for gcp n clustering
from google.cloud import language
import os
import pandas as pd

#h
from posixpath import split
from googletrans import Translator,LANGUAGES
from pprint import pprint
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer



#setting base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("base:",BASE_DIR)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  BASE_DIR + '/Comment Analysis-c92a8b9aee2a.json'

#read dataframe
data=pd.read_csv("/home/pearlsoft/Downloads/comments.csv",nrows=200)

#data preprocessing
def spanishToeng(data):
    tranlator=Translator()
    translated_words=[]
    src_lang_key=[]
    full_source_lang=[]
    for element in data['Comments']:
        translated_words.append(tranlator.translate(element).text)
        src_lang_key.append(tranlator.translate(element).src)
        full_source_lang.append(LANGUAGES[tranlator.translate(element).src])
    data['Translated_words']=translated_words 
    data['src_language_key']=src_lang_key 
    data['full_source_language']=full_source_lang      
    # print(data['Translated_words'])

spanishToeng(data)

#1.stop words removal

stop_words = set(stopwords.words('english'))
stop_words.add('subject')
stop_words.add('http')
def remove_stopwords(Comments):
    return " ".join([word for word in str(Comments).split() if word not in stop_words])
data['Translated_words'] =data['Translated_words'].apply(lambda x: remove_stopwords(x))


#2. Lemmitization

lemmatizer = WordNetLemmatizer()
def lemmatize_words(Comments):
    return " ".join([lemmatizer.lemmatize(word) for word in Comments.split()])

data["Translated_words"] = data["Translated_words"].apply(lambda x: lemmatize_words(x))

#3. punctuation removal
data['Translated_words']=data['Translated_words'].apply(lambda x:re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])"," ",x))
data['Translated_words']=data['Translated_words'].apply(lambda x: re.sub(' +', ' ', x).lower())  #remove extra white spaces


data2=data.drop(['Communication','Compassion','Competence','Comments','Unnamed: 4','src_language_key','full_source_language'],axis=1)
# print(data2.head(5))




#fn for sentiment analysis
def language_analysis(text):

    client = language.LanguageServiceClient()
    # print("client::::::::::::::::::::",client)
    document = language.Document(content=text.encode('utf-8'), type=language.Document.Type.PLAIN_TEXT)
    
    # print("document:::::::::::::",document)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    # print("sentiment:::::::::::::::",sentiment)
    # print('Text: {}'.format(text))
    # print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))
    ent_analysis = client.analyze_entities(document=document)
    # print("entity::::::::::::::::::",ent_analysis)
    entities = ent_analysis.entities
    # print("entities:::::::::::::::::;",entities)
    compoundscore = sentiment.score
    # print(compoundscore)
    if sentiment.score > 0:
        commenttype = 'Positive'
    elif sentiment.score > -.2:
        commenttype = 'Negative'
    else:
        commenttype = 'Neutral'
    # print('commenttype--->',commenttype)
    return commenttype

# language_analysis(data['Translated_words'])

data1=data.drop(['Communication','Compassion','Competence','Unnamed: 4','src_language_key','full_source_language'],axis=1)
data1['sentiment']=data['Translated_words'].apply(lambda x:language_analysis(x))
#   print(data1.head(6))


#6.hierarchical algo
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

# Standardize features


documents = data1['Translated_words'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)


# Create meanshift object
cluster = AgglomerativeClustering(n_clusters=3)

# Train model
f=features.toarray()
model = cluster.fit(f)
# View predict class
# print("test:::::::::;;;",model.labels_)
data1['clusters'] =model.labels_
# data1=data.drop(['Communication','Compassion','Competence','Translated_words','Unnamed: 4','src_language_key','full_source_language'],axis=1)
data1['clusters'] = data1['clusters'].map({0:'competence',1:'compassion',2:'communication'})

print(data1.columns)
# data2=data1.drop(['Translated_words'],axis=1)
print(data1.head(5))
data1.to_csv("sentiments_ccc")