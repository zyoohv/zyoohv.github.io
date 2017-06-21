---
layout: post
title:  "File Arranging in Machine-Learning Contest"
date:   2017-06-21 08:00:00
categories: ML
permalink: /archivers/ML
---

![](/image/06-21.jpg)

When we take part in machine learning contest, how to arrange ours files always be a difficult problem. For example, how to make it easy to append new model? how to make it easy to change the algorithms we used? how to make it more quick? also, how to make ours code brief?

<!--more-->

It's obvious that a good structure will reduce our trouble when we need changing our methods. And of course, it's inevitable for us to change our methods such as parameters, algorithms and so on. If you have a flexible strategy, you will find it's really easy to optimize them.

## basic principle

There are some basic principles we need to obey. When we look up some great project, we usually can find them in their projects. But today, we will get the explanation that why they arrange files in that style.

### 1. README.md

**README.md file should at the top level of the archive.** If someone runs your code or view the information of your project, the first place they go should be your REAME file. What's more, it always be the only place where they can learn from. A good README file should include follows:

1.  The hardware / OS platform you used  
2.  Any necessary 3rd-party software (+ installation steps)  
3.  How to train your model  
4.  How to make predictions on a new test set.  

If you add your mail address or something like it, it will gets better.

### 2. setting.json

**The setting.json file should at the top of the archive.** This file specifies the path to the train, test, model, and output directories. Further more, you'd better put all configure path into this file. Next time we want to specify ourselves file, we can just edit it. It should including:

1.  This is the **only place** that specifies the path to these directories.
2.  Any code that is doing I/O should use the appropriate base paths from setting.json

### 3. split training and prediction codes

We all know that reducing interconnection of the code is the core thinking of object oriented programing, and it especially suits this condition. Generally speaking, we usually do cross-validation or training operation many times, and predict the result or make decision only after we get a good enough model with good parameters. If we put these code together, we will do many redundant job and waste much time both of us and computers.

For example, if you are using python, there would be two entry points to your code:

*train.py*, which would  
  1.  Read training data from TRAIN_DATA_PATH (specified in setting.json)
  2.  Train your model
  3.  Save your model to MODEL_PATH (specified in setting.json)  

*predict.py*, which would  
  1.  Read test data from TEST_DATA_PATH (specified in setting.json)
  2.  Load your model from MODEL_PATH (specified in setting.json)
  3.  Use your model to make predictions on new samples
  4.  Save your predictions to SUBMISSION_PATH (specified in setting.json)  

### 4. build your own package

**build your own package is another way to reduce interconnection of your code.** For example, we can make file `abstract_feature` and make it be a new package made by us. And thus, when we need update our methods, we just edit it or establish new file in it. **An important idea of package is to decide a good interface**, just like this:

```python
'''abstract_feature package

support interface:

- train_split(days=7): dtrain, dlabel: type=dataframe
- load_full(): dtrain: type=dataframe
'''
import numpy as np

def train_split(days=7):
    # do some operation

def load_full():
    # do some operation
```

So next time when you need change algorithms or fix bug about this package, you don't need edit code's which calls this interface.

### 5. learning more, reading more

A good Design-Patterns can make your code more flexible and more safe, the meaning of safe may be less bugs. Thinking more and practice more, your code must be better and better.

At the same time, don't forget reading more code from **Github** or some awesome guys' blog, many skills which they show will benefit us too.

## powerful model

Generally speaking, we'd better use packages instead of our own methods, especially when they finish same tasks. Powerful packages can help us save both time and energy, Although sometimes they are really slow. :)

## reference

[https://www.kaggle.com/wiki/ModelSubmissionBestPractices](https://www.kaggle.com/wiki/ModelSubmissionBestPractices)
