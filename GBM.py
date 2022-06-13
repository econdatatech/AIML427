import sys
#from operator import add
from pyspark.sql import SparkSession
import time
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.sql.functions import when, lit
#from pyspark.sql.types import StructType,StructField, StringType, IntegerType
#from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
#doesn't like to have pandas imported... even when I pass the package with spark-submit
#import pandas as pd


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: GBM <inputfile> <output directory> ", file=sys.stderr)
        sys.exit(-1)


    spark = SparkSession\
        .builder\
        .appName("GBM")\
        .getOrCreate()

    # Load the data stored in csv format as a DataFrame.

    data = spark.read.option("sep", ",").csv(sys.argv[1]+'/mfa_wrangled.bz2',  header ='true',inferSchema='true')
    new_cols=(column.replace('.', '_') for column in data.columns)
    data = data.toDF(*new_cols)

    #would have been more elegant had I managed to import pandas
    #n_skip_rows = 3
    #row_rdd = spark.sparkContext.textFile(sys.argv[1]+'/features.csv').zipWithIndex().filter(lambda row: row[1] >= n_skip_rows).map(lambda row: row[0])
    #cols = pd.read_csv(sys.argv[1]+'/features.csv',nrows=5,header=[0,1,2])
    #cols.columns=['_'.join(col).strip() for col in cols.columns.values]
    #cols.rename(columns={"feature_statistics_number": 'track_id'},inplace=True)
    #features = spark.read.csv(row_rdd, header ='true',inferSchema='true').toDF(*cols.columns)
    #trackspd = pd.read_csv(sys.argv[1]+'/tracks.csv',skiprows=[1,2],usecols=['Unnamed: 0','track.7']).rename(columns={'Unnamed: 0':'track_id',"track.7": "Genre"})
    #mySchema = StructType([StructField("track_id", IntegerType(), True),StructField("Genre", StringType(), True)])
    #tracks = spark.createDataFrame(trackspd,schema=mySchema)

    data.groupby(['Genre']).count().sort('count',ascending=False).show()
    data=data.filter(data['Genre'].isin(['Rock','Experimental']))
    data = data.withColumn('label', when(data.Genre=='Rock', lit('1')).otherwise('0'))
    data=data.drop('Genre')

    #would have been more elegant had I managed to import pandas
    #data = features.join(tracks,['track_id'],how='inner')
    #data.show()
    #data.groupby(['Genre']).count().sort('count',ascending=False).show()
    #data=data.filter(data['Genre'].isin(['Rock','Experimental']))
    #data = data.withColumn('label', when(data.Genre=='Rock', lit('1')).otherwise('0'))
    #data=data.drop('Genre').drop('track_id')

    assembler = VectorAssembler().setInputCols(data.columns[:-1]).setOutputCol('features')
    data=assembler.transform(data)
    labelIndexer = StringIndexer(inputCol='label', outputCol="indexedLabel").fit(data)
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    start = time.time()

    seed=23
    (trainingData, testData) = data.randomSplit([0.7, 0.3],seed)

    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions_train = model.transform(trainingData)

    predictions_test = model.transform(testData)

    # Select example rows to display.
    predictions_test.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

    accuracy_train = evaluator.evaluate(predictions_train)
    accuracy_test = evaluator.evaluate(predictions_test)

    end = time.time()

    columns = ['ValueType', 'Value']
    vals = [
         ('Train accuracy', (accuracy_train)),
         ('Test accuracy', (accuracy_test)),
        ('Run time', (end-start)/60)

    ]
    dfnor = spark.createDataFrame(vals, columns)

    dfnor.coalesce(1).write.csv(sys.argv[2]+'/normal.csv')

    ############

    #Standard Scale
    (trainingData, testData) = data.randomSplit([0.7, 0.3],seed)

    start = time.time()

    scaler = StandardScaler(inputCol="features", outputCol="normFeatures")
    scaly = scaler.fit(trainingData)

    data = scaly.transform(data)

    featureIndexer =VectorIndexer(inputCol="normFeatures", outputCol="indexedFeatures", maxCategories=4).fit(data)

    (trainingData, testData) = data.randomSplit([0.7, 0.3],seed)



    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions_train = model.transform(trainingData)

    predictions_test = model.transform(testData)

    # Select example rows to display.
    predictions_test.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

    accuracy_train = evaluator.evaluate(predictions_train)
    accuracy_test = evaluator.evaluate(predictions_test)

    end = time.time()

    columns = ['ValueType', 'Value']
    vals = [
         ('Train accuracy', (accuracy_train)),
         ('Test accuracy', (accuracy_test)),
        ('Run time', (end-start)/60)

    ]
    dfsc = spark.createDataFrame(vals, columns)


    dfsc.coalesce(1).write.csv(sys.argv[2]+'/scaled.csv')


    #########

    pca = PCA(k=518,inputCol="normFeatures", outputCol="pca")
    model = pca.fit(trainingData)

    data = model.transform(data)

    featureIndexer =VectorIndexer(inputCol="pca", outputCol="indexedFeatures", maxCategories=4).fit(data)

    start = time.time()

    (trainingData, testData) = data.randomSplit([0.7, 0.3],seed)

    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions_train = model.transform(trainingData)

    predictions_test = model.transform(testData)

    # Select example rows to display.
    predictions_test.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

    accuracy_train = evaluator.evaluate(predictions_train)
    accuracy_test = evaluator.evaluate(predictions_test)

    end = time.time()

    columns = ['ValueType', 'Value']
    vals = [
         ('Train accuracy', (accuracy_train)),
         ('Test accuracy', (accuracy_test)),
        ('Run time', (end-start)/60)

    ]
    dfpca = spark.createDataFrame(vals, columns)

    dfpca.coalesce(1).write.csv(sys.argv[2]+'/pca.csv')
