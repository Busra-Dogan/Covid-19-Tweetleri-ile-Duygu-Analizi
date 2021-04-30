import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS)
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


data=pd.read_csv('tertemiz.csv',delimiter=(';'))

data["duygu"].replace(1, value = "pozitif", inplace = True)
data["duygu"].replace(0, value = "negatif", inplace = True)


# veri onisleme
tweet = data.copy()
tweet['veri'] = tweet['veri'].str.lower()
for i in range(tweet.shape[0]) :
    tweet['veri'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([^\w\s])|(\d)", " ", tweet['veri'][i]).split())

#stopwords

stop_word_list = open('turkce-stop-words.txt','r').read().split()
docs = tweet['veri']
def token(values):
    filtered_words = [word for word in values.split() if word not in stop_word_list]
    not_stopword_doc = " ".join(filtered_words)
    return not_stopword_doc
docs = docs.map(lambda x: token(x))
tweet['stop_veri'] = docs

#işaretlenenler çekiliyor
tagged_tweet=tweet.loc[tweet.duygu.isin(['pozitif', 'negatif'])]
tweet_datas = tagged_tweet['stop_veri'].values.tolist()
tweet_sentiment= tagged_tweet['duygu'].values.tolist()

tweet_all = tweet['stop_veri'].iloc[730:].values.tolist()

#test ve train olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(tweet_datas, tweet_sentiment, test_size = 0.25, random_state = 42)
print('Eğitim seti uzunluğu: {}'.format(len(x_train)))
print('Test seti uzunluğu: {}'.format(len(x_test)))

#tfidf
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df = 5)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

tweet_all_tfidf = tfidf_vectorizer.transform(tweet_all)

#lOGISTIC REGRESYON
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train_tfidf, y_train)

y_pred = model.predict(x_test_tfidf)
print ("Accuracy={}".format(accuracy_score(y_test,  y_pred)))

tweet_all_pred = model.predict(tweet_all_tfidf)

#Confisuon Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt=".0f")
plt.xlabel('Öngörülen değerler')
plt.ylabel('Gerçek değerler')
plt.title('Başarı Skoru: {}'.format(accuracy_score(y_test,  y_pred)), size = 13)
plt.show()


#pasta dilimi
tweet['duygu'].iloc[730:]=tweet_all_pred
tweet['tagged_data']=tweet['duygu']

labels = Counter(tweet['tagged_data']).keys()
sum_ = Counter(tweet['tagged_data']).values()
df = pd.DataFrame(zip(labels,sum_), columns = ['duygu', 'Toplam'])
fig, ax = plt.subplots(figsize=(15, 10))
ax.pie(df.Toplam, labels =df.duygu, autopct = '%1.2f%%', startangle = 90 )
ax.axis('equal')

#günlük değer
import plotly.graph_objects as go
from plotly.offline import plot

degerler=pd.read_csv('degerler.csv',delimiter=(';'))

hasta = go.Scatter(x = degerler.Tarih,
                    y = degerler.Hasta,
                    mode = "lines+markers",
                    name = "Hasta / Cases",
                    marker = dict(color = 'rgb(30, 144, 255)'),
                    text= degerler.Hasta
                   )
olum = go.Scatter(x = degerler.Tarih,
                    y = degerler.Vefat,
                    mode = "lines+markers",
                    name = "Vefat / Death",
                    marker = dict(color = 'rgb(255, 69, 0)'),
                    text= degerler.Vefat
                   )
iyilesen = go.Scatter(x = degerler.Tarih,
                    y = degerler.Iyilesen,
                    mode = "lines+markers",
                    name = "İyileşen / Recovered",
                    marker = dict(color = 'rgb(50, 205, 50)'),
                    text= degerler.Iyilesen
                   )
degerler = [hasta, olum, iyilesen]
layout = dict(title = "Türkiye'deki Covid-19 Hasta, Vefat ve İyileşen Sayıları ", 
              xaxis= dict(title= 'Tarih'), yaxis= dict(title= 'Kişi Sayısı'), xaxis_tickangle=-45)
fig = dict(data = degerler, layout = layout)
plot(fig)



def show_wordcloud(data , title = None):
    wordcloud = WordCloud(background_color='black',stopwords=stopwords,max_words=200,max_font_size=40).generate(str(data))
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    plt.title(title, size = 25)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
    
show_wordcloud(tweet['stop_veri'])
pos = tweet['stop_veri'][tweet['duygu'] == 'pozitif']
show_wordcloud(pos , 'POZİTİF')
neg = tweet['stop_veri'][tweet['duygu'] == 'negatif']
show_wordcloud(neg , 'NEGATİF')



from collections import Counter
list = tweet['duygu'].ravel()
c= dict(Counter(list))
print(c)


from sklearn.metrics import classification_report
x=classification_report(y_test, y_pred)
print(x)
