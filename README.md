# Optimizing a Machine Learning Pipeline in Azure

## Overview
This project was completed as part of Udacity's Machine Learning with Microsoft Azure Nanodegree.
In this project, an Azure ML pipeline was built and optmised using the Python SDK and a base Scikit-learn model. The performance of the resulting model was compared to that of the best model produced by an Azure AutoML run.

## Summary

The dataset used in this project contains bank marketing data, and the objective was to predict if a client will subscribe to a term deposit using a base Scikit-learn model. The hyperdrive run of the Azure SDK was used for optimising the base logistic regression pipeline and for hyperparamters tuning. The accuracy metric was used to compare the best performing optimised hyperdrive model with the best performing model produced by an AutoML config run. For the hyperdrive experiment, the best performing model was a logistic regression with hyperparamters c = 0.9 and max_iteration = 75. The AutoML experiment produced a VotingEnsemble model as the best performing model with similar performance (accuracy difference of 0.002) as the model produced by the hyperdrive run. 


## Scikit-learn Pipeline

The pipeline architecture consists of a training script that specifies the training data, training model and model hyperparameters to be used, as well as a jupyter notebook containing information about the hyperparameter sampling method, early termination policy, type of estimator to be used, and hyperdrive configuration parameters. The training data is a bank marketing data stored in csv format in an Azure blobstorage. The training script accesses the data set using the TabularDatasetFactory method() in Azure and performs data cleaning and one-hot encoding of non-numeric features. It also specifies a logistic regression model with regularisation strength and max_iteration as hyperparamters. The notebook is used to specify a random sampling method, and a bandit policy for early termination. An SKlearn estimator was also specified and these parameters were used to set up the hyperdrive config. A computer cluster was created for remote training of the model using the hyperdrive configuration. The hyperdrive experiment was submitted and when the run was completed, the child run with the best metric was retrevied using the `get_best_run_by_primary_metric()` method. The best model was finally registered using the joblib library for future access and deployment.

Azure Machine Learning supports three sampling methods over the hyperparameter space namely: Random, Grid and Bayesian methods. The random sampling method randomly selects the hyperparameter values from a defined search space rather than an exhaustive search which takes longer to complete. The Baysian sampling policy does not support early termination while the grid sampling policy supports only discrete hyperparameters. In this project, the random sampling method was used to sample the regularization and max_iteration hyperparameters of the logistic regression model because it runs faster, supports both discrete and continous hyperparameters, as well as the early termination of low performing runs.

The bandit early stopping policy with a smaller allowable slack that provides more aggressive savings by terminating poorly performing runs to improve computational efficiency was chosen for this project. The bandit policy is based on a slack factor/amount and an evaluation interval. It terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best peforming run. The slack factor is specified as a ratio while the slack amount is specified as an absolute amount. The early policy termination was set to a slack factor of 0.1 and evaluation interval of 1 so that performance is evaluated at the end of each child run.

## AutoML

A classification task and the primary metric of 'accuracy' was specified in the `AutoMLConfig`. The AutoML generated multiple classification models and optimised the computation using the parameters specified in AutoMLConfig. The model with the highest accuracy value amongst all other models at the end of the AutoML run was found to be the VotingEnsemble model.

## Pipeline comparison

In terms of performance accuracy, the logistic regression model produced by the `HyperDrive` experiment (accuracy = 0.915) performed similar to the VotingEnsemble model (accuracy = 0.913) produced by the `AutoML` experiment with a slight difference of 0.002 in accuracy. This difference in accuracy can be explained by the fact that different models analyse data differently resulting in different accuracy numbers. In terms of architecture, unlike the hyperdrive experiment, the AutoML does not specify a model or list of model hyperparamters to optimise but rather, it explores different machine learing models for the specified task based on the specified primary metric. The best performing model can then be selected at the end of the experiment.


## Future work

If more resources are available, future experiments can be improved by using a grid sampling method for the hyperdrive experiment to explore all hyperparamter combinations while the `experiment_timeout period` parameter of AutoML config can be increased to allow for further exploration of available models in Azure Machine Learning. 



