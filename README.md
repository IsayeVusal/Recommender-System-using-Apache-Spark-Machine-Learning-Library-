# Recommender-System-using-Apache-Spark-Machine-Learning-Library

Matrix factorization will be used in Apache Spark MLLIB library build in function.

A recommendation system is an information filtering mechanism that attempts to predict the rating a user would give a particular product. There are some algorithms to create a Recommendation System.

Apache Spark ML implements alternating least squares (ALS) for collaborative filtering, a very popular algorithm for making recommendations.

ALS recommender is a matrix factorization algorithm that uses Alternating Least Squares with Weighted-Lamda-Regularization (ALS-WR). It factors the user to item matrix A into the user-to-feature matrix U and the item-to-feature matrix M:

It runs the ALS algorithm in a parallel fashion. The ALS algorithm should uncover the latent factors that explain the observed user to item ratings and tries to find optimal factor weights to minimize the least squares between predicted and actual ratings.


