# COVID Fake News Detection

## Overview
Fake news represents one of the most pressing issues faced by social media ecosystems today. While there have been many examples of fake news consumption leading to the adoption of inaccurate beliefs and harmful behaviors, fake news regarding the COVID-19 pandemic is arguably more dangerous since it may lead to worse public health outcomes. To combat this issue, social media companies have employed ML to augment their ability to distinguish between fake and real news.  

While ML has advanced our ability to identify and root out fake news, this approach is not without its limitations. Fake news, by its very nature, is constantly evolving. A ML algorithm capable of detecting fake news with high accuracy at one point in time may become significantly less accurate when tasked with identifying fake news sampled from a different time period. By examining the way in which fake news detection algorithms decay over time, we hope to better understand the limitations of applying ML to the issue of fake news.  

Using COVID-19 related social media data, we ask the following question: **To what extent does the contextual and fast-moving nature of fake news represent a limitation for ML models?**  

Our project comprises two main components:
1. Using existing COVID-related real and fake news datasets to build a fake news detection algorithm. (standard end-to-end ML project)
2. Using more recent COVID-related real and fake news dataset compiled by our team, we will apply the same fake news detection algorithm, to measure model decay. 

## Data 
* For **fake news**, we use the [COVID-19 Fake News dataset](https://paperswithcode.com/dataset/covid-19-fake-news-dataset) introduced by Patwa et al. in [Fighting an Infodemic: COVID-19 Fake News Dataset](https://paperswithcode.com/paper/fighting-an-infodemic-covid-19-fake-news). The dataset comprises 5,100 fake COVID-related English-language social media posts from various platforms.
* For **real news**, we use the [RealNews dataset](https://paperswithcode.com/dataset/realnews) introduced by Zellers et al. in [Defending Against Neural Fake News](https://paperswithcode.com/paper/defending-against-neural-fake-news). The dataset comprises 5,600 COVID-related tweets from credible and relevant Twitter accounts such as the World Health Organization (WHO) and the Centers for Disease Control and Prevention (CDC) among others.
* Lastly, we use methods consistent with those employed by Patwa et al. and Zellers et al to **create our own dataset containing recent cases of fake and real news**.

## Evaluation
* We'll use an F-1 score to evaluate our models' performance:
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
