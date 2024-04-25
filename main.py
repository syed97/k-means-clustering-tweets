'''
todo: 
- implement compute sse to rturn sse error values
- implement update centroids
- implement main loop of kmeans with convergence and max_iter check
'''
import pandas as pd
import string
import random
import numpy as np

# # Input and Pre-processing
# read textfile
with open("Health-Tweets\\nprhealth.txt", "r", encoding="utf8") as file:
    content = file.read()
# split text into lines
lines = content.split("\n")
# split line based on separator "|"
output = []
for line in lines:
    input = line.split("|")
    output.append(input)
# convert to dataframe
df = pd.DataFrame(output, columns=["tweetID","timestamp","tweet"])
# last row contains None values so dropping it
df.drop(4837, inplace=True)
# print(df)

# drop tweetID and timestamp columns
df.drop(columns=["tweetID","timestamp"], inplace=True, axis=1)

# function to remove words containing "@" character
def remove_words_with_at(tweet):
    words = tweet.split()  # Split the tweet into words
    filtered_words = [word for word in words if "@" not in word]  # Filter out words containing "@"
    return ' '.join(filtered_words)  # Join the remaining words back into a sentence

# function to remove "#" symbol from word
def remove_hashtag_symbol(tweet):
    words = tweet.split()  # Split the tweet into words
    filtered_words = [word[1:] if word.startswith("#") else word for word in words]  # Remove "#" symbol from words
    return ' '.join(filtered_words)  # Join the modified words back into a sentence

# remove urls from tweets
def remove_url(tweet):
    words = tweet.split()
    filtered_words = [word for word in words if not word.startswith("http")]
    return ' '.join(filtered_words) 

# remove colon at the end of certain words e.g. help:
def remove_colon(tweet):
    words = tweet.split()
    filtered_words = [word[:len(word)-1] if word[len(word)-1] == ":" else word for word in words]
    return ' '.join(filtered_words) 

# remove punctuation amrks like ?'! etc.
def remove_punctuation(tweet):
    words = tweet.split()
    filtered_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    return ' '.join(filtered_words) 

df['tweet'] = df['tweet'].apply(lambda x: remove_words_with_at(x))
df['tweet'] = df['tweet'].apply(lambda x: remove_hashtag_symbol(x))
df['tweet'] = df['tweet'].apply(lambda x: remove_url(x))
df['tweet'] = df['tweet'].apply(lambda x: remove_colon(x))
df['tweet'] = df['tweet'].apply(lambda x: x.lower()) # convert to lower case
df['tweet'] = df['tweet'].apply(lambda x: remove_punctuation(x))

# df.to_csv('data.csv')
# print(df)

# implement K-means clustering
'''
- init centroids
'''

class KMeans:

    centroids = {}

    def __init__(self, k=3, max_iters=300, dataframe=df):
        self.k = k
        self.max_iters = max_iters
        self.dataframe = df
        df['cluster'] = 0 # default

    def init_centroids(self):
        # assign k initial clusters
        for i in range(1,self.k+1):
            # get random tweet index from 0 to max no. of tweets(-1 b/c Python indexing)
            tweet_index = random.randint(0, self.dataframe.shape[0] - 1)
            # assign tweet index to cluster number
            self.centroids[i] = tweet_index
        print(self.centroids)

    def get_jaccard_distance(self, in1, in2):
        # get jaccard similarity between two word sets
        # A = {wA1, wA2, wA3, ...}, B = {wB1, wB2, wB3, ...}
        A = set(in1.split())
        B = set(in2.split())
        cardinality_union = len(A.union(B))
        cardinality_intersection = len(A.intersection(B))
        jaccard = (cardinality_union - cardinality_intersection)/cardinality_union
        return jaccard

    def assign_to_clusters(self):  
        # for idx, row in self.dataframe.iloc[:5].iterrows(): 
        for idx, row in self.dataframe.iterrows():
            # default case: initially, consider the tweet is completely unrelated 
            # to any other and assign to random cluster before applying jaccard check
            min_distance = 1
            cluster = random.randint(1,self.k)
            # assign to cluster with min jaccard distance to current tweet
            for i in range(1,self.k+1):
                jaccard = self.get_jaccard_distance(
                    row['tweet'],
                    df['tweet'].iloc[self.centroids[i]]
                )
                if jaccard < min_distance:
                    min_distance = jaccard
                    cluster = i
            # update cluster value for the row
            self.dataframe.at[idx, 'cluster'] = cluster
        # print(self.centroids) # sanity check
        # print(self.dataframe['cluster'].value_counts())
    
    def update_centroids(self):
        pass


model = KMeans(dataframe=df)
model.init_centroids()
model.assign_to_clusters()
