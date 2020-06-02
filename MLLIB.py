# importing some libraries
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

# check if spark context is defined
print(sc.version)


# importing the MF libraries
from pyspark.mllib.recommendation import ALS, \ MatrixFactorizationModel, Rating

# reading the movielens data
df_rdd = sc.textFile('C:/Users/Vusal/Desktop/DDA new/ml-1m/ratings.dat')\.map(lambda x: x.split("::"))
            
ratings= df_rdd.map(lambda l:\Rating(int(l[0]),int(l[1]),float(l[2])))

# Splitting the data into train and test sets.
X_train, X_test= ratings.randomSplit([0.8, 0.2])

# Training the model
rank = 10
numIterations = 10
model = ALS.train(X_train, rank, numIterations)

# Evaluate the model on testdata
# dropping the ratings on the tests data
testdata = X_test.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
# joining the prediction with the original test dataset
ratesAndPreds = X_test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

# calculating error
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))
