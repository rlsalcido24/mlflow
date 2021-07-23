# Databricks notebook source
# MAGIC %md # Predicting wine quality

# COMMAND ----------

# MAGIC %md ### Import data and combine

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

display(data)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ##Data Visualization
# MAGIC 
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC 
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful in noticing correlations between features and a binary label.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC 
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocessing Data
# MAGIC Prior to training a model, check for missing values and split the data into training and test sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality

# COMMAND ----------

# MAGIC %md ## Building a Baseline Model
# MAGIC 
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC 
# MAGIC The following code builds a simple classifier using scikit-learn. It uses MLflow to keep track of the model accuracy, and to save the model for later use.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature

# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, signature=signature)

# COMMAND ----------

# batch inference 
import mlflow
logged_model = 'runs:/a3501d93a3624d869e047813d3d33d96/random_forest_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
predictiondf = pd.DataFrame(loaded_model.predict(pd.DataFrame(X_train)))
display(predictiondf)

# COMMAND ----------

# MAGIC %md Examine the learned feature importances output by the model as a sanity-check.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md As illustrated by the boxplots shown previously, both alcohol and density are important in predicting quality.

# COMMAND ----------

# MAGIC %md ##Experimenting with a new model
# MAGIC 
# MAGIC The random forest model performed well even without hyperparameter tuning.
# MAGIC 
# MAGIC The following code uses the xgboost library to train a more accurate model. It runs a parallel hyperparameter sweep to train multiple
# MAGIC models in parallel, using Hyperopt and SparkTrials. As before, the code tracks the performance of each parameter configuration with MLflow.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(test, "test")], early_stopping_rounds=50)
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=96,
    trials=spark_trials, 
    rstate=np.random.RandomState(123)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC #Model serving
# MAGIC To productionize the model for low latency predictions, use MLflow model serving (AWS|Azure) to deploy the model to an endpoint.
# MAGIC 
# MAGIC The following code illustrates how to issue requests using a REST API to get predictions from the deployed model.

# COMMAND ----------

# MAGIC %md
# MAGIC You need a Databricks token to issue requests to your model endpoint. You can generate a token from the User Settings page (under the profile icon on the upper right). Copy the token into the next cell.

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "<YOUR_TOKEN>"

# COMMAND ----------

# MAGIC %md
# MAGIC Click Models in the left sidebar and navigate to the registered wine model. Click the serving tab, and then click Enable Serving.
# MAGIC 
# MAGIC Then, under Call The Model, click the Python button to display a Python code snippet to issue requests. Copy the code into this notebook. It should look similar to the code in the next cell.
# MAGIC 
# MAGIC You can use the token to make these requests from outside Databricks notebooks as well.

# COMMAND ----------

# Replace with code snippet from the model serving page
import os
import requests
import pandas as pd
 
def score_model(dataset: pd.DataFrame):
  url = 'https://<DATABRICKS_URL>/model/wine_quality/Production/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split')
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC The model predictions from the endpoint should agree with the results from locally evaluating the model.

# COMMAND ----------

# Model serving is designed for low-latency predictions on smaller batches of data
num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])
# Compare the results from the deployed model and the trained model
pd.DataFrame({
  "Model Prediction": model_evaluations,
  "Served Model Prediction": served_predictions,
})
