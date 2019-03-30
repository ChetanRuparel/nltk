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
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = 'TPc4fkzA3kzkUNRlWTeisVATk'
        consumer_secret = 'men0sNd6IbNJpXuvD11KJQVMQdLEUIy8kjz0uLSFq8N9SmO9ka'
        access_token = '706471922901479424-Z14KxXE1E9O6rSG1sKZhZ0Jh9RCgKuu'
        access_token_secret = 'DdSx2iqcU8PUxyA9Xht6fwBZenoOtNQ9Nn2yCgf7KYG7l'

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())




    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q = query, count = count)
            f = open("tweetsAI.json", "r")

            #json_file = open('tweetsAI.json')
            #data = json.load(json_file)
            #json_file.close()

            #print(data)



            #print(fetched_tweets)

            #with open('tweetsAI.json') as f:
                #data = json.load(f)

            #pprint(data)

            n_instances = 100
            subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
            obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
            print(len(subj_docs))
            print(len(obj_docs))

            print(subj_docs[0])

            train_subj_docs = subj_docs[:80]
            test_subj_docs = subj_docs[80:100]
            train_obj_docs = obj_docs[:80]
            test_obj_docs = obj_docs[80:100]
            training_docs = train_subj_docs+train_obj_docs
            testing_docs = test_subj_docs+test_obj_docs
            sentim_analyzer = SentimentAnalyzer()
            all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

            unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
            print(len(unigram_feats))
            sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

            training_set = sentim_analyzer.apply_features(training_docs)
            test_set = sentim_analyzer.apply_features(testing_docs)

            trainer = NaiveBayesClassifier.train
            classifier = sentim_analyzer.train(trainer, training_set)

            for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
                print('{0}: {1}'.format(key, value))

            sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
                "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
                "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
                "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
                "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
                "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
                "The book was good.",         # positive sentence
                "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
                "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
                "A really bad, horrible book.",       # negative sentence with booster words
                "At least it isn't a horrible book.", # negated negative sentence with contraction
                ":) and :D",     # emoticons handled
                "",              # an empty string is correctly handled
                "Today sux",     #  negative slang handled
                "Today sux!",    #  negative slang with punctuation emphasis handled
                "Today SUX!",    #  negative slang with capitalization emphasis
                "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
            ]
            paragraph = "It was one of the worst movies I've seen, despite good reviews. \
             Unbelievably bad acting!! Poor direction. VERY poor production. \
             The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

            lines_list = tokenize.sent_tokenize(paragraph)
            sentences.extend(lines_list)
            tricky_sentences = [
                "Most automated sentiment analysis tools are shit.",
                "VADER sentiment analysis is the shit.",
                "Sentiment analysis has never been good.",
                "Sentiment analysis with VADER has never been this good.",
                "Warren Beatty has never been so entertaining.",
                "I won't say that the movie is astounding and I wouldn't claim that \
                the movie is too banal either.",
                "I like to hate Michael Bay films, but I couldn't fault this one",
                "It's one thing to watch an Uwe Boll film, but another thing entirely \
                to pay for it",
                "The movie was too good",
                "This movie was actually neither that funny, nor super witty.",
                "This movie doesn't care about cleverness, wit or any other kind of \
                intelligent humor.",
                "Those who find ugly meanings in beautiful things are corrupt without \
                being charming.",
                "There are slow and repetitive parts, BUT it has just enough spice to \
                keep it interesting.",
                "The script is not fantastic, but the acting is decent and the cinematography \
                is EXCELLENT!",
                "Roger Dodger is one of the most compelling variations on this theme.",
                "Roger Dodger is one of the least compelling variations on this theme.",
                "Roger Dodger is at least compelling as a variation on the theme.",
                "they fall in love with the product",
                "but then it breaks",
                "usually around the time the 90 day warranty expires",
                "the twin towers collapsed today",
                "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
                under orders and in the ''least offensive way possible.''"
            ]
            sentences.extend(tricky_sentences)
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

            # parsing tweets one by one
            # for tweet in fetched_tweets:
            #     # empty dictionary to store required params of a tweet
            #     parsed_tweet = {}
            #
            #     # saving text of tweet
            #     parsed_tweet['text'] = tweet.text
            #     # saving sentiment of tweet
            #     parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
            #
            #     # appending parsed tweet to tweets list
            #     if tweet.retweet_count > 0:
            #         # if tweet has retweets, ensure that it is appended only once
            #         if parsed_tweet not in tweets:
            #             tweets.append(parsed_tweet)
            #     else:
            #         tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    tweets = api.get_tweets(query = 'books', count = 200)

    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    # percentage of neutral tweets
    #print("Neutral tweets percentage: {} %".format(100*len(tweets - ntweets - ptweets)/len(tweets)))

    # printing first 5 positive tweets
    print("\n\nPositive tweets:")
    for tweet in ptweets[:10]:
        print(tweet['text'])

    # printing first 5 negative tweets
    print("\n\nNegative tweets:")
    for tweet in ntweets[:10]:
        print(tweet['text'])

    print("\n \n \n")


if __name__ == "__main__":
    # calling main function
    main()
