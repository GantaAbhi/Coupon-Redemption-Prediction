# ----------------------------------------
# 1) START SPARK SESSION
# ----------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("Coupon Redemption Prediction").getOrCreate()

# ----------------------------------------
# 2) LOAD DATASET
# ----------------------------------------
data = spark.read.csv("marketing_campaign.csv", header=True, inferSchema=True)

data.printSchema()

# ----------------------------------------
# 3) FIX COLUMN NAMES (IMPORTANT STEP)
# ----------------------------------------
for c in data.columns:
    data = data.withColumnRenamed(c, c.strip().replace(" ", "_"))

# ----------------------------------------
# 4) HANDLE MISSING VALUES
# ----------------------------------------
clean_data = data.na.drop()

# ----------------------------------------
# 5) CREATE AGE COLUMN
# ----------------------------------------
clean_data = clean_data.withColumn("age", expr("2024 - Year_Birth"))

# ----------------------------------------
# 6) ENCODE CATEGORICAL VARIABLES
# ----------------------------------------
education_indexer = StringIndexer(inputCol="Education", outputCol="education_index")
marital_indexer = StringIndexer(inputCol="Marital_Status", outputCol="marital_index")

data_indexed = education_indexer.fit(clean_data).transform(clean_data)
data_indexed = marital_indexer.fit(data_indexed).transform(data_indexed)

# ----------------------------------------
# 7) FEATURES
# ----------------------------------------
feature_cols = [
    "education_index",
    "marital_index",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "age"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

final_df = assembler.transform(data_indexed).select(col("features"), col("Response").alias("label"))

# ----------------------------------------
# 8) TRAIN TEST SPLIT
# ----------------------------------------
train, test = final_df.randomSplit([0.7, 0.3], seed=42)

# ----------------------------------------
# 9) MODEL
# ----------------------------------------
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(train)

predictions = model.transform(test)

# ----------------------------------------
# 10) EVALUATION
# ----------------------------------------
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

print("Accuracy:", accuracy_eval.evaluate(predictions))
print("F1 Score:", f1_eval.evaluate(predictions))

# ----------------------------------------
# 11) VISUALIZATION DATA
# ----------------------------------------
clean_data.groupBy("Education").count().show()
clean_data.groupBy("Marital_Status").count().show()