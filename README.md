# COVID Fake News Detection

## Overview
Fake news represents one of the most pressing issues faced by social media ecosystems today. While there have been many examples of fake news consumption leading to the adoption of inaccurate beliefs and harmful behaviors, fake news regarding the COVID-19 pandemic is arguably more dangerous since it may lead to worse public health outcomes. To combat this issue, social media companies have employed Machine Learning to augment their ability to distinguish between fake and real news.  

While ML has advanced our ability to identify and root out fake news, this approach is not without its limitations. Fake news, by its very nature, is constantly evolving. A ML algorithm capable of detecting fake news with high accuracy at one point in time may become significantly less accurate when applied to news sampled from a later time period. By examining the way in which fake news detection algorithms decay over time, we hope to better understand the limitations of applying ML to the issue of fake news.  

## Academic Question
Using COVID-19 related social media data, we ask the following question: **To what extent does the context-dependent and fast-moving nature of fake news represent a limitation for ML models?**  

Our project comprises two main components:
1. Create a fake news detection algorithm using existing COVID-related real and fake news datasets
1. Measure the decay of our fake news detection algorithm by applying it to recent COVID-related news

## Data 
* We use the [COVID-19 Fake News  dataset](https://paperswithcode.com/dataset/covid-19-fake-news-dataset) introduced by Patwa et al. in [Fighting an Infodemic: COVID-19 Fake News Dataset](https://paperswithcode.com/paper/fighting-an-infodemic-covid-19-fake-news). The dataset comprises 10,700 COVID-related social media posts from various platforms which have been hand-labelled real or fake. 
  * "Real" posts comprise tweets from credible and relevant sources such as the World Health Organization (WHO) and the Centers for Disease Control and Prevention (CDC) among others.
  * "Fake" posts include tweets, posts, and articles which make claims about COVID-19 that have been labeled false by credible fact-checking sites such as [PolitiFact.com](https://www.politifact.com/factchecks/list/).
* Additionally, we use methods consistent with those employed by Patwa et al. to **create our own dataset containing recent cases of fake and real news**.

## Evaluation
We'll use an F-1 score to evaluate our models' performance:  
<img src="https://render.githubusercontent.com/render/math?math=\text{F-1}=\frac{2*Precision*Recall}{Precision+Recall}=\frac{2*TP}{2*TP+FP+FN}">

## Dependencies
* Python ≥3.5
* numpy
* pandas
* matplotlib
* nltk
* sklearn ≥0.20
* ...

## Contributors 
* Hannah Schweren
* Marco Schildt
* Steve Kerr

## Workflow
<img src="https://github.com/smkerr/COVID-fake-news-detection/blob/main/img/workflow.png">

## Sources
* [A Heuristic-driven Uncertainty based Ensemble Framework for Fake News Detection in Tweets and News Articles](https://arxiv.org/abs/2104.01791) - basis for our paper
* [Fighting an Infodemic: COVID-19 Fake News Dataset](https://paperswithcode.com/paper/fighting-an-infodemic-covid-19-fake-news) - fake news dataset
* [Defending Against Neural Fake News](https://paperswithcode.com/paper/defending-against-neural-fake-news) - real news dataset 
* ...

## License
...
