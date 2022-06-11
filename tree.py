import sys
from operator import add
from pyspark.sql import SparkSession
import time
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: tree <inputfile> <output directory> <seed>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("KDD99Tree")\
        .getOrCreate()

    start = time.time()
    #amended from https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier

    seed = sys.argv[3]
    # Load the data stored in csv format as a DataFrame.
    data = spark.read.load(sys.argv[1],
                         format="csv", sep=",", inferSchema="true", header="false").toDF('duration',
                          'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                          'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                          'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                          'is_host_login', 'is_guest_login','count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                          'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                          'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                          'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                          'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','label')

    assembler = VectorAssembler().setInputCols(['duration',
                          'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                          'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                          'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                          'is_host_login', 'is_guest_login','count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                          'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                          'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                          'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                          'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']).setOutputCol('features')

    data=assembler.transform(data)
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    #featureIndexer = [VectorIndexer(inputCol=column, outputCol=column+"_index", maxCategories=4).fit(data) for column in list(set(data.columns[:-1])) ]
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)


    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3],seed)

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions_train = model.transform(trainingData)

    predictions_test = model.transform(testData)

    # Select example rows to display.
    # predictions.select("prediction", "indexedLabel", "features").show(5)

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
    df = spark.createDataFrame(vals, columns)

    #print("Train accuracy = %g " % (accuracy_train))
    #
    #print("Test accuracy = %g " % (accuracy_test))

    #print("Execution time = %g " % ((endlg-startlg)/60))

    #treeModel = model.stages[2]
    # summary only
    #print(treeModel)


    #Report the training and test results including the max, min,
    #average accuracy and the standard deviation obtained from the 10 runs,

    df.coalesce(1).write.csv(sys.argv[2]+seed)

    spark.stop()
