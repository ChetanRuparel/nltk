import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import json
from pprint import pprint
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize

class TwitterClient(object):

    def __init__(self):
        #Authentication keys got from twitter dev account
        consumer_key = 'TPc4fkzA3kzkUNRlWTeisVATk'
        consumer_secret = 'men0sNd6IbNJpXuvD11KJQVMQdLEUIy8kjz0uLSFq8N9SmO9ka'
        access_token = '706471922901479424-Z14KxXE1E9O6rSG1sKZhZ0Jh9RCgKuu'
        access_token_secret = 'DdSx2iqcU8PUxyA9Xht6fwBZenoOtNQ9Nn2yCgf7KYG7l'
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def get_tweets(self, query, count = 10):

        tweets = []

        try:
            #get the tweets from twitter
            fetched_tweets = self.api.search(q = query, count = count)

            n_instances = 100
            subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
            obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]


            train_subj_docs = subj_docs[:80]
            test_subj_docs = subj_docs[80:100]
            train_obj_docs = obj_docs[:80]
            test_obj_docs = obj_docs[80:100]
            training_docs = train_subj_docs+train_obj_docs
            testing_docs = test_subj_docs+test_obj_docs
            emotion_analyzer = SentimentAnalyzer()
            #get the negative words for feature extraction
            all_radical_slurs = emotion_analyzer.all_words([mark_negation(doc) for doc in training_docs])

            unigram_feats = emotion_analyzer.unigram_word_feats(all_radical_slurs, min_freq=4)

            emotion_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

            training_set = emotion_analyzer.apply_features(training_docs)
            test_set = emotion_analyzer.apply_features(testing_docs)

            trainer = NaiveBayesClassifier.train
            classifier = emotion_analyzer.train(trainer, training_set)

            #test sentences
            sentences = ["Ravi is the worst boy in class",
                "The story is full of mean bitchy characters",
                "I had a good day!",
                "The day was okay",
                "The day was very bad",
                "Harry potter is a good book",
                "New Tata electric car is a piece of shit",
                "It has been a long time since I had a good food",
                "Stop acting as a asshole"
            ]

            sid = SentimentIntensityAnalyzer()
            for sentence in sentences:
                print(sentence)
                ss = sid.polarity_scores(sentence)
                for k in sorted(ss):
                    print('{0}: {1}, '.format(k, ss[k]), end='')
                print()
            for tweet in fetched_tweets:
                print(tweet.text)
                ss = sid.polarity_scores(tweet.text)
                for k in sorted(ss):
                    print('{0}: {1}, '.format(k, ss[k]), end='')
                print()

            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))

def main():
    api = TwitterClient()
    tweets = api.get_tweets(query = 'books', count = 200)

if __name__ == "__main__":
    main()
