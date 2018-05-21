---
layout: post
title:  "Contest about Tencent Advertisement"
date:   2018-05-20 08:00
categories: ML
permalink: /archivers/8
---

![](/image/8.1.png)

I did some attempt to solve the problem. Thought they seems not work well, but I really want to write some methods that may will be used one day.

<!--more-->

## 1. arrange dataset && extract feature

The given dataset is so large, about 4G. And at the same time, the number of negative samples are about 10 times to positive samples. So develop a method to arrange them is so important for use to save time and energy. We arrange them with:

1. transform .data file to .csv file.
2. assemble them to one big file.
3. extract feature from the big file and save our feature to files.

Note that we must use nearly all samples to train our model, because it will make us score higher. So we devide all our dataset into about 20 pacakges. Each one contain all positice samples and the same number of negative samples. 

We use `onehot`, `embedding` and `CountVectorizer` method to convert our origin file to our train matrix. We extract some cross-feature to improve the result. The feature matrix are too big, so we save them in .npz(csr_matrix) format. The example code here:

```Python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix

...
```

## 2. build model

We apply a two layer structure, but in fact we have no time to test the second step for time limiting.

1. we train our model with 20 train-prediction pair packages.
2. we use the result of all 20 prediction result to train the other model.

In detail, we train our model with 1 of 20 packages, and make prediction with the others 10 packages and the predtion feature matrix. When we apply the second method, we also train with our 20 packages, but the number of feature is 19 (20 - 1, because we do not make prediction itself in first step), and drop the correspond column in predtion feature matrix column.

I think you must know what I means, hah...
more information, visit code here: [github.com](https://github.com/zyoohv/zyoohv.github.io/blob/master/code_repository/tencent_ad_contest/model/)

Some usefull model:

deepFM: [https://github.com/ChenglongChen/tensorflow-DeepFM](https://github.com/ChenglongChen/tensorflow-DeepFM)

xlearn: [https://github.com/aksnzhy/xlearn](https://github.com/aksnzhy/xlearn)

## 3. turn parameters

The detail look up our source file please. We give some methodes to turn them:

1. Use small dataset, big learning rate(0.1), small iterator_num to speed up.
2. Too large iterator_num may cause over fitting, and too small iterator_num may can not score high. So we'd better set a valid dataset and earyl_stoping.
3. CV is not dependable.

## 4. note

1. The processing of the dataset waste too much time. I should use a more faster method, or just use the part dataset to test the result.
2. I should always store the middle result. It means that if my code report error in last line code, I do not need run them all again.
3. Extimate the runing time, if avaiable, run them in serveral computer.