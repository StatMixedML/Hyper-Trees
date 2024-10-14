# Hyper-Trees
We introduce the concept of Hyper-Trees and offer a new direction in applying tree-based models to time series data. Unlike conventional applications of decision trees that forecast time series directly, Hyper-Trees are designed to learn the parameters of time series models. Our framework combines the effectiveness of gradient boosted trees on tabular data with the advantages of established time series models, thereby naturally inducing a time series inductive bias to tree models. By relating the parameters of a target time series model to features, Hyper-Trees also address the issue of parameter non-stationarity. To resolve the inherent scaling issue of boosted trees when estimating a large number of target model parameters, we combine decision trees and neural networks within a unified framework. In this novel approach, the trees first generate informative representations from the input features, which a shallow network then maps to the target model parameters. With our research, we aim to explore the effectiveness of Hyper-Trees across various forecasting scenarios and to extend the application of gradient boosted trees outside their conventional use in time series modeling. 

<center>
    <img height="350" src="figures/hypertree.png">
</center>

## `General Information`
This repo contains the official implementation of our paper [Forecasting with Hyper-Trees](https://arxiv.org/pdf/2405.07836). The source code of our Hyper-Tree architecture will be made available upon final publication of the paper.

## `News`
:boom: [2024-05-01] Create repository and initial commits.

## `Feedback`
We encourage you to provide feedback by opening a [new discussion](https://github.com/StatMixedML/Hyper-Trees/discussions).

## `Reference Paper`
[![Arxiv link](https://img.shields.io/badge/arXiv-Forecasting%20with%20Hyper--Trees-color=brightgreen)](https://arxiv.org/pdf/2405.07836) <br/>
