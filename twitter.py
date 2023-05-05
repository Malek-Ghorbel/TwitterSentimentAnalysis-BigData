# -*- coding: utf-8 -*-


import pyspark
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
from pyspark.sql import SparkSession 

import pandas as pd

import re 
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, SQLTransformer,IndexToString,CountVectorizer 

from pyspark.ml.classification import LinearSVC
from pyspark.ml import Pipeline ,PipelineModel

from pyspark.ml.evaluation import MulticlassClassificationEvaluator 

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp import DocumentAssembler


import os
import gc

from pyspark.sql import SparkSession #Import the spark session
from pyspark import SparkContext #Create a spark context
from pyspark.sql import SQLContext #Create an SQL context

import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("Spark NLP")\
    .config("spark.executor.memory", "12g").config("spark.driver.memory", "12g")\
    .config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g")\
    .config('spark.executor.cores', '3').config('spark.cores.max', '3')\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "/spark-nlp_2.12:4.4.0.jar").getOrCreate()

"""**Charging the dataset (exisitng in drive)**

"""



training_data = spark.read.csv("/training.csv", inferSchema = True, header = False) #Read in the data
#training_data.show(10)

"""**Operations for Data preprocessing**"""

columns = ["target", "id", "date", "flag", "user", "tweet"]  

training_data = training_data.select(col("_c0").alias(columns[0]), col("_c1").alias(columns[1]), col("_c2").alias(columns[2]),
                      col("_c3").alias(columns[3]), col("_c4").alias(columns[4]), col("_c5").alias(columns[5]))
training_data.show(10)

"""remove unnecessary columns"""

training_data = training_data.select('target' ,'tweet')
training_data.show(10)

"""normalizing the sentiment column values"""

training_data = training_data.withColumn("target", when(training_data["target"] == 4, 1).otherwise(training_data["target"]))
training_data.groupBy("target").count().orderBy("count").show()

"""remove unneeded words from tweets (mentions, links ...)"""

training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', r'http\S+', '')) 
training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '@\w+', '')) 
training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '#', ''))
training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', 'RT', ''))


training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '&amp;', ''))
training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '&quot;', ''))
training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '&gt;', ''))
training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '&lt;', ''))


training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '-', ''))

training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '   ', ' '))
training_data = training_data.withColumn('tweet', F.regexp_replace('tweet', '  ', ' '))


training_data = training_data.filter((training_data.tweet!= ' ') &(training_data.tweet!= '')& (training_data.tweet!= '   '))

"""Splitting data into training and test"""

Train_Test_sets = training_data.randomSplit([0.75, 0.25])
train_set = Train_Test_sets[0] 
test_set = Train_Test_sets[1]

"""Defining the stages for the Natural Language Processing (NLP) pipeline that will be applied to data"""

#from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

#Turn tweets into documents
document_assembler = DocumentAssembler() \
    .setInputCol("tweet") \
    .setOutputCol("document")

#Turn these documents into tokens
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

#Normalizing the tokens (Remove punctautions ..)
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

#Remove stop words from tokens
stopwords_cleaner = StopWordsCleaner() \
    .setInputCols(["normalized"]) \
    .setOutputCol("cleanTokens") \
    .setCaseSensitive(False)

#turn the documented_tokens into array of tokens
finisher = Finisher() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCols(["tokens"]) \
    .setOutputAsArray(True)

#Hashing the tokens
hashingTF = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=1000)
idf = IDF(inputCol="tf", outputCol="features")

#Classification based on the hashed tokens using the ML model Support Vector Machine
svm = LinearSVC(featuresCol="features", labelCol="target")

#define the pipeline
nlp_pipeline = Pipeline(
    stages=[
        document_assembler,
        tokenizer, 
        normalizer, 
        stopwords_cleaner, 
        finisher, 
        hashingTF, 
        idf, 
        svm
    ]
)


# nlp_pipeline.setStages([
#     document_assembler,
#     tokenizer,
#     normalizer,
#     stopwords_cleaner,
#     finisher,
#     hashingTF,
#     idf,
#     svm
# ])

#get the model
p=nlp_pipeline.fit(train_set)

"""Evaluation"""

def evaluate(input_set):
    results=p.transform(input_set)
    evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(results)
    print("Accuracy = %g" % (accuracy))
    print("Error = %g " % (1.0 - accuracy))
    return accuracy

evaluate(test_set)

"""Saving the model"""

p.save("/pipeline")



