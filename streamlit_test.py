#%%writefile app.py

# All imports
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import plotly.express as px
import re
import numpy as np
import datetime as dt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import praw
import requests
import os
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
import emoji
import statistics as stats
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

st.sidebar.image("AWlogo2.png")
# Streamlit title
st.title("Stock Market Analysis")

# Streamlit Sidebar

stock_search = st.sidebar.text_input('Stock Search',placeholder = "Input stock ticker here")

keywords_input = st.sidebar.text_input('NLP keywords',placeholder = "Keywords with space here")

# Checkboxes

nlp_model_checkbox = st.sidebar.checkbox("Press this if you want to enable NLP model")

lstm_model_checkbox = st.sidebar.checkbox("Press this if you want to enable LSTM model")

#arima_model_checkbox = st.sidebar.checkbox("Press this if you want to enable ARIMA model")

ready_to_run_code = st.checkbox("Click this when you are ready to run the code")

# Pick variables such as stock ticker and what we want to train on (Close, High, Open)

# Pick prediction date
prediction_date = st.sidebar.date_input('Prediction Date')

# Ticker string if multiple tickers
yf.pdr_override()

tickers = ""

# setting up reddit variables/dependencies
reddit_id = "wpotSHZd6nAbXpA-aqVTwQ"
reddit_secret = "clfa5nAXLOr3iHpbwm9RGrG7eFJZvg"
reddit_user = "HolyShoter"
reddit_pass = "GARgar109$"

reddit = praw.Reddit(
    client_id=reddit_id,
    client_secret=reddit_secret,
    password=reddit_pass,
    user_agent="USERAGENT",
    username=reddit_user,
    check_for_async = False
)

# dictionary to update VADER Lexicon
new_words = {
    'citron': -4.0,  
    'hindenburg': -4.0,        
    'moon': 4.0,
    'highs': 2.0,
    'mooning': 4.0,
    'long': 2.0,
    'short': -2.0,
    'call': 4.0,
    'calls': 4.0,    
    'put': -4.0,
    'puts': -4.0,    
    'break': 2.0,
    'tendie': 2.0,
     'tendies': 2.0,
     'town': 2.0,     
     'overvalued': -3.0,
     'undervalued': 3.0,
     'buy': 4.0,
     'sell': -4.0,
     'gone': -1.0,
     'gtfo': -1.7,
     'paper': -1.7,
     'bullish': 3.7,
     'bearish': -3.7,
     'bagholder': -1.7,
     'stonk': 1.9,
     'green': 1.9,
     'money': 1.2,
     'print': 2.2,
     'rocket': 2.2,
     'bull': 2.9,
     'bear': -2.9,
     'pumping': -1.0,
     'sus': -3.0,
     'offering': -2.3,
     'rip': -4.0,
     'downgrade': -3.0,
     'upgrade': 3.0,     
     'maintain': 1.0,          
     'pump': 1.9,
     'hot': 1.5,
     'drop': -2.5,
     'rebound': 1.5,  
     'crack': 2.5,
     "up":2.7,
     "down":-2.7,
     "comeback":3,
     "fall":-3,
     "asshole":-4,
     "suckers":-4,
     "fault":-3,
     "doesn't care":-4,
     }

# Setting up VADER

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(new_words)

tweet_num = 500
reddit_num = 70

if nlp_model_checkbox:
  tweet_num = st.sidebar.number_input('Tweet num',min_value = 400,max_value = 900, value = 500)
  reddit_num = st.sidebar.number_input('Reddit num',min_value = 50,max_value = 600,value = 70)
  
  st.write("Extended sidebar, please scroll down")

if lstm_model_checkbox:
  display_cols = st.sidebar.multiselect('Display Data', ["Close","Low","Open","High",],default = "High")

# Download data from yahoo finance
def download_data(ticker = tickers):
  data = yf.download(ticker)
  # st.write(ticker)
  cf = data

  cf["Date"] = cf.index

  df = cf

  dataframe = data[display_cols]
  dataframe.reset_index(level=0, inplace=True)

  # Plot stock data without ml or nlp
  plt.plot(dataframe["Date"],dataframe[display_cols])
  plt.ylabel("Dollars")
  plt.xlabel("Date")
  plt.legend()
  plt.draw()

  st.pyplot(plt)

# Machine Learning starts
  cf=dataframe
  pf = cf


  # if cf.shape[0] > 5000:
  #   st.write("Dataset too big, please use more recent stock")
  #   quit()

  # LSTM dataset code
  cf.index=cf['Date']

  df['Date'] = pd.to_datetime(df['Date'])
  df['Date'] = df['Date'].dt.strftime('%d.%m.%Y')
  df['year'] = pd.DatetimeIndex(df['Date']).year
  df['month'] = pd.DatetimeIndex(df['Date']).month
  df['day'] = pd.DatetimeIndex(df['Date']).day
  df['dayofyear'] = pd.DatetimeIndex(df['Date']).dayofyear
  df['weekofyear'] = pd.DatetimeIndex(df['Date']).weekofyear
  df['weekday'] = pd.DatetimeIndex(df['Date']).weekday
  df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
  df['is_month_start'] = pd.DatetimeIndex(df['Date']).is_month_start
  df['is_month_end'] = pd.DatetimeIndex(df['Date']).is_month_end

  df.loc[df['is_month_start']==False,'is_month_start']=0
  df.loc[df['is_month_start']==True,'is_month_start']=1
  df.loc[df['is_month_end']==False,'is_month_end']=0
  df.loc[df['is_month_end']==True,'is_month_end']=1

  df = df.drop(['Date','Open','Low','Close','Adj Close','Volume'], axis = 1) 


  # *********************** splitting the dataset

  X = df.drop('High', axis=1)
  y = df['High']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

  # ********************** Feature Scaling
  scaler=MinMaxScaler(feature_range=(0,1))
  X_train=scaler.fit_transform(X_train)
  X_test=scaler.fit_transform(X_test)

  # LSTM architecture

  def lstm_architecture():
    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(X_train,y_train,epochs=150,batch_size=25,verbose=2)

    # Testing
    predicted_closing_price=lstm_model.predict(X_test)

    valid_data=pd.DataFrame(index=range(0,len(y_test)),columns=['Date','y_test',"predicted"])

    valid_data['y_test'] = list(y_test)
    valid_data['predicted'] = predicted_closing_price
    valid_data['Date'] = y_test.index
    valid_data = valid_data.set_index('Date')
    valid_data = valid_data.sort_index()


  # st.write(valid_data.head())
  # Visuliazation LSTM
    plt.plot(valid_data)
    st.pyplot(plt)

  # Display LSTM prediction datatset
    # st.write(valid_data)

    #  ***************************************************** Prediction tools
    def lstm_prediction(prediction_date = prediction_date):
      data = [prediction_date]
      value_data=pd.DataFrame(data,index=range(0,1),columns=['Date'])

      value_data['year'] = pd.DatetimeIndex(value_data['Date']).year
      value_data['month'] = pd.DatetimeIndex(value_data['Date']).month
      value_data['day'] = pd.DatetimeIndex(value_data['Date']).day
      value_data['dayofyear'] = pd.DatetimeIndex(value_data['Date']).dayofyear
      value_data['weekofyear'] = pd.DatetimeIndex(value_data['Date']).weekofyear
      value_data['weekday'] = pd.DatetimeIndex(value_data['Date']).weekday
      value_data['quarter'] = pd.DatetimeIndex(value_data['Date']).quarter
      value_data['is_month_start'] = pd.DatetimeIndex(value_data['Date']).is_month_start
      value_data['is_month_end'] = pd.DatetimeIndex(value_data['Date']).is_month_end

      value_data.loc[value_data['is_month_start']==False,'is_month_start']=0
      value_data.loc[value_data['is_month_start']==True,'is_month_start']=1
      value_data.loc[value_data['is_month_end']==False,'is_month_end']=0
      value_data.loc[value_data['is_month_end']==True,'is_month_end']=1

      value_data = value_data.drop(['Date'], axis = 1) 

      value_data=scaler.transform(value_data)
      pr = lstm_model.predict(value_data)
      st.write("Predicted price: ")
      st.write(pr)

    lstm_prediction()
  
  if lstm_model_checkbox:
    lstm_architecture()


# NLP
def reddit_twitter_webscraping_sentiment(twitter_num_of_tweets = tweet_num,reddit_num_of_comment = reddit_num):
  bearer_token = "AAAAAAAAAAAAAAAAAAAAAGl9WAEAAAAAup10asDXc%2BPNXX7yTpSv%2B%2B0nJQ0%3DU2xJJZhYx8LHXFfEm4NqzTNHtrCRjRufnTjw9fZcdggkQGDl0O"
  
  use_keywords = keywords_input.split()

  def replace_RT(tweet, default_replace=""):
    tweet = re.sub('RT\s+', default_replace, tweet)
    return tweet

  def replace_user(tweet, default_replace=""):
    tweet = re.sub('\B@\w+', default_replace, tweet)
    return tweet

  def replace_url(tweet, default_replace=""):
    tweet = re.sub('(http|https):\/\/\S+', default_replace, tweet)
    return tweet

  def bearer_oauth(r):
      """
      Method required by bearer token authentication.
      """

      r.headers["Authorization"] = f"Bearer {bearer_token}"
      r.headers["User-Agent"] = "v2FilteredStreamPython"
      return r


  def get_rules():
      response = requests.get(
          "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
      )
      if response.status_code != 200:
          raise Exception(
              "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
          )
      print(json.dumps(response.json()))
      return response.json()


  def delete_all_rules(rules):
      if rules is None or "data" not in rules:
          return None

      ids = list(map(lambda rule: rule["id"], rules["data"]))
      payload = {"delete": {"ids": ids}}
      response = requests.post(
          "https://api.twitter.com/2/tweets/search/stream/rules",
          auth=bearer_oauth,
          json=payload
      )
      if response.status_code != 200:
          raise Exception(
              "Cannot delete rules (HTTP {}): {}".format(
                  response.status_code, response.text
              )
          )
      print(json.dumps(response.json()))


  def set_rules(delete):
      # You can adjust the rules if needed

      sample_rules = []
      for i in range(len(use_keywords)):
        sample_rules.append({"value":use_keywords[i]})

      payload = {"add": sample_rules}
      response = requests.post(
          "https://api.twitter.com/2/tweets/search/stream/rules",
          auth=bearer_oauth,
          json=payload,
      )
      if response.status_code != 201:
          raise Exception(
              "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
          )
      print(json.dumps(response.json()))


  def get_stream(set):
      response = requests.get(
          "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True,
      )
      print(response.status_code)
      if response.status_code != 200:
          raise Exception(
              "Cannot get stream (HTTP {}): {}".format(
                  response.status_code, response.text
              )
          )
      lst = []
      # KEYWORDS = keywords.split()
      def reddit_stream(amount_of_reddit = reddit_num_of_comment):
        for i in range(len(use_keywords)):
            try:
              for comment in reddit.subreddit(use_keywords[i]).stream.comments():
                sentence = comment.body
                display_sentence = comment.body
                sentence = replace_RT(sentence) # replace retweet
                sentence = replace_user(sentence) # replace user tag
                sentence = replace_url(sentence) # replace url
                sentence = sentence.lower()
                sentiment_reddit = analyzer.polarity_scores(sentence)
                if len(lst) < amount_of_reddit*(i+1):
                  print("NEW " + use_keywords[i] + ": " + display_sentence + " " + str(sentiment_reddit["compound"]))
                  lst.append(sentiment_reddit["compound"])
                else:
                  break
            except:
              pass
        # print(len(lst))
      reddit_stream()
  # and len(lst) <= len(KEYWORDS)*40+200
      lst_twitter = []
      def twitter_streams(amount_tweets = twitter_num_of_tweets):
        for response_line in response.iter_lines():
            if response_line:
              json_response = json.loads(response_line)
              tweet = json_response["data"]["text"]
              tweet = replace_RT(tweet) # replace retweet
              tweet = replace_user(tweet) # replace user tag
              tweet = replace_url(tweet) # replace url
              tweet = tweet.lower()
              sentiment = analyzer.polarity_scores(tweet)
        
              try:
                if detect(tweet) == "en" and len(lst_twitter) <= amount_tweets:
                  print("New Twitter: " + tweet + " " + str(sentiment["compound"]))
                  lst_twitter.append(sentiment["compound"])
                  # print(len(lst_twitter))
                elif len(lst_twitter) > amount_tweets:
                  break
                else:
                  pass
              except:
                pass
      twitter_streams()
        
      def web_scraping_sentiment():
        single_ticker = stock_search.split()[0].upper()
        finviz_url = "https://finviz.com/quote.ashx?t="
        apewisdom_url = "https://apewisdom.io/stocks/"
        finviz_url = finviz_url + single_ticker
        apewisdom_url = apewisdom_url + single_ticker + "/"

        finviz_req = Request(url=finviz_url, headers={'user-agent': 'my-app/0.0.1'})
        finviz_response = urlopen(finviz_req)
        apewisdom_req = Request(url=apewisdom_url, headers = {"user-agent": "my-app/0.0.1"})
        apewisdom_response = urlopen(apewisdom_req)
        df = pd.DataFrame(columns=['News_Title', 'Time'])
        sentiment_list = []
        news_table = {}


        soup = BeautifulSoup(apewisdom_response, features="lxml")
        html = BeautifulSoup(finviz_response, features="lxml")


        mentioning_users = soup.findAll("div",{"class":"details-small-tile"})[-2]
        upvotes = soup.findAll("div",{"class":"details-small-tile"})[-3]
        mentions = soup.findAll("div",{"class":"details-small-tile"})[-4]
        news_table = html.find(id='news-table')

        mentioning_users_percentage = mentioning_users.find("span").text
        upvotes_percentage = upvotes.find("span").text
        mentions_percentage = mentions.find("span").text
        sentiment = soup.findAll("div",{"class":"tile-value"})[-1].text


        dataRows = news_table.findAll('tr')


        for i, table_row in enumerate(dataRows):
            a_text = table_row.a.text
            td_text = table_row.td.text
            
            df = df.append({'News_Title': a_text, 'Time': td_text}, ignore_index=True)


        for i in range(50):
          word = df["News_Title"][i]
          news_title_sentiment = analyzer.polarity_scores(word)
          news_title_sentiment = news_title_sentiment["compound"]
          sentiment_list.append(news_title_sentiment)

        sentiment_list = stats.mean(sentiment_list)


        sentiment = sentiment[0:2]
        sentiment = int(sentiment) - 50
        if sentiment > 1:
          sentiment = sentiment*2/100
        elif sentiment < -1:
          sentiment = sentiment*2/100

        global final_sentiment
        final_sentiment = sentiment + sentiment_list
        # print(final_sentiment)
      web_scraping_sentiment()



      def find_mean_from_list():
        global final_list
        final_list = []
        final_list = lst + lst_twitter
        # print(len(final_list))
        final_list = stats.mean(final_list)
        # print("The mean is: " + str(final_list))
      find_mean_from_list()

      very_last_sentiment = (final_list + final_sentiment)/2
      st.write("The final sentiment is: " + str(very_last_sentiment))

  def main():
      rules = get_rules()
      delete = delete_all_rules(rules)
      set = set_rules(delete)
      get_stream(set)


  if __name__ == "__main__":
      main()

def main2():
  if lstm_model_checkbox:
    tickers = stock_search.replace(","," ")
    if len(tickers) > 0:
      download_data(tickers)
    else:
      st.write("Please type in a ticker")
    if nlp_model_checkbox:
      if len(keywords_input) > 0:
        reddit_twitter_webscraping_sentiment()
  elif nlp_model_checkbox:
    if len(keywords_input) > 0:
      reddit_twitter_webscraping_sentiment()
  else:
    st.write("Please check one checkbox")

if ready_to_run_code:
  main2()
