---
layout: post
title:  "Clustering Analysis - An introduction to unsupervised learning"
resume: "Briefly, clustering is the collection of procedures used to describe methods for grouping unlabeled data"
date:   2019-08-25 23:50:00
categories: [unsupervised, machine-learning]
tags: [unsupervised, machine-learning]
permalink: "/machine-learning/clustering-analysis-an-introduction-to-unsupervised-learning"
thumbnail: "/assets/img/clustering-analysis-an-introduction.jpg"
status: 1
---
{:.image}
![Clustring Analysis - An introduction to unsupervised learning](/assets/img/clustering-analysis-an-introduction.jpg)

{:.intro}
*Cluster analysis is the grouping of individuals in a population in order to discover structure
in the data. In some sense, we would like the individuals within a group to be close
or similar to one another, but dissimilar from individuals in other groups.*

In contrast to supervised learning (i.e classification), unsupervised learning fits a model to observations assuming there is no dependent random variable, output, or response. None of the observations is treated differently from the others. An informal way to say this is that there is no $Y$. For this reason, sometimes classification data that includes the $Y$ as the class is called labeled data but clustering data is not necessarily called unlabeled.

Clustering is fundamentally a collection of methods of data exploration. One often uses a method to see if natural groupings are present in the data. If groupings do emerge, these may be named and their properties summarized. The results of a cluster analysis may produce an identifiable structure that can be used to generate hypotheses (to be tested on a separate data set) to account for the observed data.


# Reasons for interest in unsupervised procedures

Collecting and labeling a large set of sample patterns can be surprisingly costly. Possible solutions could be:

* Train a classifier on a small set of labeled samples, and then "tuned up" by allowing it to run without supervision;
* Train a clustering algorithm with large amounts of unlabeled data, and only then use supervision to label the groupings found.

Besides, there are several other reasons for adopt clustering analysis. Indeed, clustering has been successfully used in different  fields, including bioinformatics, image processing, and information retrieval.

# Clustering methods

Many clustering methods were designed based on different approaches such as partitional, hierarchical, probabilistic, and density-based. This post is about partitional methods. 

Partitional clustering methods try to organize data into k clusters (where k is an
input parameter), by optimizing a certain objective function that captures a local and global structure of grouping. Most of the partitioning methods start with an initial assignment and then use an iterative relocation procedure which moves data points from one cluster to another to optimize the objective function.

On the other hand, partitional clustering methods often generate partitions (a division of the data set into mutually non-overlapping groups) where each data point belongs to one and only one cluster. These methods are the most commonly used because of their simplicity and their competitive computational complexity. One of the most popular and simple of these clustering algorithms is **k-means**

# K-Means: A Centroid-based clustering method

The k-means is the most fundamental partitional clustering method which is based on the idea that a center can represent a cluster.


A centroid is the most representative point within the cluster. Usually, this is the mean of the values of the points of data in the cluster. Given an initial clustering, centroid-based methods find the centroids of the clusters, reassign the data points to new clusters
defined by proximity to the centroids, and then repeat the procedure. The similarity between two clusters is the similarity between their respective centroids. 

> The most obvious measure of the similarity (or dissimilarity) between two samples is the distance between them. K-means is typically used with the Euclidean metric for computing the distance between points and cluster centers

You must keep in mind that most obvious measure of the similarity (or dissimilarity) between two samples is the distance between them.
Thus, the key choices to be made, aside from an initial clustering, are the distance metric to be used, the centroids to be used, and how many iterations of the procedure to use.

## Understanding K-means

Intuitively, the K-means clustering approach is the following. The analyst picks the number of clusters K and makes initial guesses about the cluster centers. The procedure starts at those K centers, and each center absorbs nearby points, based on distance (e.g Euclidean). Then, based on the absorbed cases, new cluster centers, usually the mean, are found. The procedure is then repeated: The new centers are allowed to absorb nearby points based on a norm, new centers are found, and so on. An example is exhibited in the video below:

<div class="video-container">
	<center><iframe width="500" height="315" src="https://www.youtube.com/embed/Q7FMcIYm4aI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>
</div>

Mathematically, we can represent K-means in this way:

Let $X = \{x_j\}, i = 1,...,n$ be the set of $n$ d-dimensional points to be clustered into a set of $K$ clusters, $C=\{c_k, k=1,...,K\}$. K-means algorithm finds a partition such that the squared error between the empirical mean of a cluster and the points in the cluster is minimized. Let $\mu_k$ be the mean of cluster $c_k$ (centroid). The squared error between $\mu_k$ and the points in cluster $c_k$ is defined as

$$
J(c_k) = \sum_{x_i \in c_k}||x_i-\mu_k||^2
$$

The goal of K-means is to minimize the sum of the squared error over all K clusters,

$$
J(c_k) = \sum_{k=1}^K \sum_{x_i \in c_k}||x_i-\mu_k||^2
$$

The K-means algorithm requires three user-specified parameters: 

* Number of clusters K (the most critical choice)
* Cluster initialization
* Distance metric. K-means is typically used with the Euclidean metric for computing the distance between points and cluster centers.

## Number of clusters of K-means

Automatically determining the number of clusters has been one of the most difficult problems in data clustering. Most methods for automatically determining the number of clusters cast it into the problem of model selection. Usually, clustering algorithms are run with different values of K; the best value of K is then chosen based on a predefined criterion.

One method to validate the number of clusters is the elbow method. The idea of the elbow method is to run k-means clustering on the dataset for a range of values of k and for each value of k calculate the sum of squared errors (SSE). Furthermore, we plot a line chart of the SSE for each value of k. If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best. On the figure below, the most probable value for the initial number of clusters is 5 as the "elbow" is in 4 and 5 intervals.

{:.image}
![Clustring Analysis - An Introduction](/assets/img/clustering-analysis-an-introduction-elbow.png)

# Pratical example in Python code

For a practical example in Python code, check the video below (skip for 17:20 min). The video is available in portuguese for now.

<div class="video-container">
	<center><iframe width="560" height="315" src="https://www.youtube.com/embed/xBIs5_ic5hU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>
</div>

## References

1. [Statistical Pattern Recognition - Andrew R. Webb](https://amzn.to/2Zo4t2k)

2. [Pattern Classification - Richard O. Duda](https://amzn.to/32dtcZ5)
