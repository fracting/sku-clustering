# sku-clustering

# Table of contents

- [Problem description](#Problem-description)
- [Summary of report](#Summary-of-report)
- [Comparision of different solutions](#Comparision-of-different-solutions)
  - [Data cleaning and preprocessing](#Data-cleaning-and-preprocessing)
  - [Combination of configurations](#Combination-of-configurations)
- [Evaluation of performance](#Evaluation-of-performance)
- [Result](#Result)
- [Methodology](#Methodology)

# Problem description

Perform **unsupervised(?) machine learning(?)** over a dataset of 26k SKUs, suggest 5 or more categories, and present any other interesting insight.

# Summary of report

This report consists of 4 chapters.

 * Chapter 1: [Exploratory data analysis - Part 1 - Abnormal detection.](https://github.com/fracting/sku-clustering/blob/main/Chapter%201%20-%20Exploratory%20data%20analysis%20%281%29.ipynb)
 
 * Chapter 2: [Exploratory data analysis - Part 2 - Why TF-IDF is a good start?](https://github.com/fracting/sku-clustering/blob/main/Chapter%202%20-%20Exploratory%20data%20analysis%20%282%29.ipynb)

 * Chapter 3: Presents a machine learning solution using `word embedding + KMean++`.

 * Chapter 4: Presents a non machine learning approach and explains why it works better.

For machine learning approaches:
 * `TF-IDF + KMean++`, `word embedding + KMean++` and `TF-IDF + word embedding + KMean++` all works reluctantly.
 * Unfortunately, none of them are outstanding.

For **non machine learning approach**:
 * network analysis was used, which **outperforms all machine learning models** being tried.
 
**Note:** due to the unique approach taken in this project, Airflow and Spark was not used during experiment. To show respect to the interview, an implementation using Airflow + Spark will be provided separately with some delay.

# Comparision of different solutions

## Data cleaning and preprocessing
 * abnormal detection and removal
 * html tags removal
 * stopwords removal
 * digits and puntuctuations removal 

## Combination of configurations
 * different feature selection strategies (human)
 * different feature extraction models (machine)
 * different number of clusters (2~10) (for debugging)
 * different clustering algorithms
 * optional tricks

### feature selection (human)
 * long_description
 * product_name
 * category_token
 * long_description (+product_name) (+category_token) (+gender)
 * long_description + product_name + category_token (-brand) (-color)

### feature extraction (machine)
 * TF-IDF
 * Count vectorizer
 * Word embedding (pretrained / non-pretrained) 
 * Combination of TF-IDF and word embedding
 
### clustering algorithm
 * kmean++ (random seed or manually assigned centroids)
 * dbscan (Density-Based, hard to turn hyperparameters, gave up)
 * others (too slow, doesn't scale)
 * connected graph components (non machine learning)
 

**Experiments show that all components are important but some are more critical:**

 * Importance of clustering algorithm (and hyperparameter)
  
 * \> Importance of feature selection (manually)

 * \> Importance of feature extraction (machine learning)

## Evaluation of performance

As an open challenge without standard ground truth, commonsense knowledge is used intensively to judge the performance of different solutions.

By randomly sampling every class and manually inspecting the samples, we can estimate the quality of the clustering model given enough positive and negative samples.

At the same time, **diff set analysis** is also used when comparing different models.

Assume A1 and B1 are two equivalent classes predicted by two models, by inspecting `A1-B1` and `B1-A1`, we can tell which model generates better outcome.

## Result

| Feature extraction      | Clustering algorithm       | Columns                       | Number of clusters | Quality     | Note                     |
|-------------------------|----------------------------|-------------------------------|--------------------|-------------|--------------------------|
| TF-IDF                  | KMean++                    | category > name > description | 5                  | reluctantly |                          |
| Word Embedding          | KMean++                    | category > name > description | 5                  | reluctantly |                          |
| TF-IDF + Word Embedding | KMean++                    | category > name > description | 5                  | reluctantly |                          |
| Count vectorizer        | KMean++                    | category                      | 5                  | reluctantly |                          |
| -                       | Connected graph components | category                      | 5                  | outstanding | with human intervention  |

Despite **many machine learning approached being tried** with different hyperparameters, surprisingly **none of them works outstandingly**.

Typically, `TF-IDF` / `word embedding` / `TF-IDF + word embedding` works ok with most feature selection strategies.

`Count vectorizer` works reluctantly only with category_token feature (we will explain why)

Due to the characteristic of the given dataset, machine learning approaches works better with number of clusters between 2 to 4. However, the problem description explicitly asked for 5 or more clusters. As the number of clusters increases, the quality of clustering starts to decrease. A common pattern observed is, class `Facial` and class `Accessories Hair` are likely to mix with each other, class `Jumpsuit` and class `Polo` are likely to mix with each other. (we will explain why)

On the other side, **a none machine learning approach** using network analysis to extract connected graph components on category_token, **outperforms all machine learning approaches** being tried. (we will explain why)

Due to time limit we will not present all details but save some space for most interesting findings.

## Methodology

As described in the previous section, a combination of `feature selection strategies`, `feature extraction models` and `clustering algorithm` could easily expand to a huge solution space with over hundreds of potential configurations, it's a waste of time to exhaust all combinations, not to mention the final best solution is not even a machine learning algorithm.

Managing of huge searching space is always a great challenge in machine learning, here we introduce one of the most important ideas in machine learning: **dimension reduction**.

Dimension reduction is a fundamental concept appeared acrossed many different areas of machine learning.

An image classification model reduces an input space of million pixels and million samples to an output space of thousands of classes. Typically, a CNN layer reduce the number of neurons in a layer from millions to a few hundreds, comparing with a FC layer.

An word embedding model reduces an input space of several ten thousands tokens to an output space of a few-hundreds-dimensional real-valued-vector.

A recommendation model reduces an input space of large amount of users and large amount of items to a latent matrix with a shape of (few hundreds, few hundres).

Dimension reduction is not only a common pattern in model design, but also **a powerful technique for model debugging** when an algorithm fails.

Examples including:
 * **down sampling**: if a model doesn't work with a large dataset, start debugging with a smaller dataset
 * **reduce number of features**: if a model doesn't work with a large feature set, start debugging with less features
 * **reduce complexity of features**: if a model doesn't work with complex features, start debugging with simpler feature
 * **reduce number of clusters**: if a model doesn't work with a large number of clusters, start with a smaller number of clusters and then apply hierarchical clustering

**The dimension reduction strategy was used again and again during the project, as a result it naturally lead us to the simplest but best solution which doesn't use machine learning at all. We will show some concrete applications of this methodology in the following chapters.**

**Dimension reduction is the secret sauce in big data and machine learning engineering.**

Machine learning has been frequently accused as an alchemy. While this is not wrong, in this report we will see that, instead of blindly tuning hyperparameters, with an efficient dimension reduction strategy, every step makes sense even in alchemy.
