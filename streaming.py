from pyspark.ml import Pipeline ,PipelineModel

from pyspark.sql import SparkSession #Import the spark session
from pyspark import SparkContext #Create a spark context
from pyspark.sql import SQLContext #Create an SQL context

import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("Spark stream")\
    .master("local[*]")\
    .config("spark.executor.memory", "12g").config("spark.driver.memory", "12g")\
    .config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g")\
    .config('spark.executor.cores', '3').config('spark.cores.max', '3')\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.4").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


pipeline_model=PipelineModel.load("/pipeline")

def predict(line , pipeline_model , sprk): # function to make a predection on a tweet or line and outout happy or sad
#    pipeline_model=PipelineModel.load("/pipeline")
#    sprk = SparkSession.builder.appName("Prediction").getOrCreate()
    sample_df = sprk.createDataFrame([[str(line)]]).toDF('tweet')
    #-- preprocessing---
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', r'http\S+', ''))
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '@\w+', ''))
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '#', ''))
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', 'RT', ''))


    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '&amp;', ''))
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '&quot;', ''))
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '&gt;', ''))
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '&lt;', ''))


    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '-', ''))

    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '   ', ' '))
    sample_df = sample_df.withColumn('tweet', F.regexp_replace('tweet', '  ', ' '))


    #---

    result = pipeline_model.transform(sample_df)
    sentiment = result.select('prediction').first()[0]
    if(sentiment == 1):
        sentiment = "Happy"
    else:
        sentiment = "Sad"

    return sentiment

from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

batch_interval = 10
bootstrap_servers = ['localhost:9092']
topic_name = "tweets"


df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "localhost:9092") \
  .option("subscribe", "tweets") \
  .load()


from pyspark.sql.functions import col
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import re
from textblob import TextBlob
def cleanTweet(tweet: str) -> str:
    tweet = re.sub(r'http\S+', '', str(tweet))
    tweet = re.sub(r'bit.ly/\S+', '', str(tweet))
    tweet = tweet.strip('[link]')

    # remove users
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))

    # remove puntuation
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@â'
    tweet = re.sub('[' + my_punctuation + ']+', ' ', str(tweet))

    # remove number
    tweet = re.sub('([0-9]+)', '', str(tweet))

    # remove hashtag
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))

    return tweet


# Create a function to get the subjectifvity
def getSubjectivity(tweet: str) -> float:
    return TextBlob(tweet).sentiment.subjectivity


# Create a function to get the polarity
def getPolarity(tweet: str) -> float:
    return TextBlob(tweet).sentiment.polarity


def getSentiment(polarityValue: int) -> str:
    if polarityValue < 0:
        return 'Negative'
    elif polarityValue == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Define the sentiment analysis function
def predict_sentiment(message):
    tweet = cleanTweet(message)
    p = getPolarity(tweet)
    return getSentiment(p)

# Define a UDF that applies the predict_sentiment function on the value column
predict_sentiment_udf = udf(lambda message: predict_sentiment(message), StringType())


df = df.selectExpr("timestamp", "CAST(value AS STRING)" ) \
  .select("*") \
  .withColumn("sentiment", predict_sentiment_udf(col("value")))

 
query = df.writeStream \
  .outputMode("append") \
  .format("console") \
  .option("truncate", False) \
  .start()

# Start the streaming query
query.awaitTermination()



