import tweepy

# Define the search query
query = "Python"
consumer_key = "NsTnUERSJ7ebLUAIx6C63J7yC"
consumer_secret = "ImPn8VJ2VllQ9gQUMxIae9SYL04nqn67DT3qgjzuUpyANTokkG"
access_token = "1643648404961017856-z18nRE7HD4MilUWOWCzflLutSVuYkS"
access_token_secret = "1OEVvRo7hKpbfWDOJIqOleR5K4WAe7oby9vVtcLk77v3w"


from kafka import KafkaProducer
import time

topic_name = "tweets"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

bootstrap_servers = ['localhost:9092']
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
# Create Kafka producer


# Define the search query
query = "film"

# Get the 100 most recent tweets with the search query
tweets = api.search(q=query,lang="en", count=100)

# Publish each tweet to Kafka
for tweet in tweets:
    future = producer.send(topic_name, tweet.text.encode('utf-8'))
    result = future.get(timeout=60)  # wait for acknowledgement
    print(result)
    time.sleep(3)

# Close the Kafka producer
producer.close()

