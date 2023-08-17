# Dimensionality Reduction

## Introduction

We will apply dimensionality reduction as a preprocessing step in a machine learning workflow.

## Objectives 
- Practice performing PCA using the scikit-learn library
- Interpret the amount of variance captured by a given number of PCA components
- Evaluate model performance with and without dimensionality reduction
- Plot the decision boundary of classification experiments to visually inspect their performance 

## Our Task: Reduce the Dimensionality of the Iris Dataset as Part of a Machine Learning Workflow

![irises](iris.jpg)

<span>Photo by <a href="https://unsplash.com/@yoksel?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Yoksel 🌿 Zok</a> on <a href="https://unsplash.com/s/photos/iris?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>

### Dimensionality Reduction in ML

While it is possible to use dimensionality reduction as a standalone analysis technique, we will frequently see it used as a preprocessing step in a predictive machine learning workflow.

The two main reasons to use dimensionality reduction in machine learning are:

1. **Reducing computational complexity:** Often the internal logic of a machine learning algorithm means that the complexity increases by an order of magnitude with every additional dimension (feature). So maybe there are {n^2} operations for 2 features, {n^4} operations for 4 features, etc. If we can reduce the number of dimensions (features) prior to fitting/predicting with the model, the model will be faster and use fewer computational resources (memory, processing power, etc.)
2. **Improving model performance:** In some cases even if we had unlimited computational capacity, our models would still struggle to fit on data with too many dimensions, known as the *curse of dimensionality*. Generally this applies when there are hundreds of features (or more). We can also sometimes see reductions in overfitting with dimensionality reduction, depending on the data and the model.

There is no guarantee that dimensionality reduction will produce improved results — it all depends on how our features are related to each other, and the details of the machine learning algorithm we are using. In this lab we will walk through several different dimensionality reduction techniques and observe their impacts on the modeling process.

### The Iris Dataset

For this lab we will use the famous Iris Dataset that comes with scikit-learn. This is a classic "toy" dataset where we are trying to identify the species of iris based on the provided attributes.


Part of why we use this dataset for so many examples is that there is clear predictive power in each of the features (i.e. the distributions of feature values differ for each of the targets):

### Sequence of Operations:

#### 1. Perform a Train-Test Split

Because we are using dimensionality reduction within a predictive modeling context, we need to perform a train-test split prior to taking any other steps.

#### 2. Scale Data

Both the model we are using (logistic regression with regularization) and our dimensionality reduction techniques are distance-based, so we need to scale our data before performing any analysis.

#### 3. Evaluate Model Performance without PCA

Before performing PCA, we'll fit a vanilla logistic regression model on the provided features and evaluate its performance, including the time taken.

#### 4. Perform and Visualize PCA

Using the `PCA` transformer class from scikit-learn, fit and transform the training data so that the four dimensions of the original features have been projected down to two dimensions. We'll then identify how much of the variance is captured, and plot the data points using these two dimensions as the x-axis and y-axis.

#### 5. Evaluate Model Performance with PCA

We will fit and evaluate a new logistic regression model on the transformed data.

#### BONUS: Manifold Dimensionality Reduction

Another, more-advanced technique to consider for dimensionality reduction is *manifold learning*. Fortunately scikit-learn also provides an interface to this technique that works the same way as any other transformer.