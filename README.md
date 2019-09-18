# Fisheries-monitoring-to-stop-illegal-practices
This is a Image Classification problem.

Machine learning has the ability to transform what we know about our oceans and how we manage them.In the Western and Central Pacific, where 60% of the world’s tuna is caught, illegal, unreported, and unregulated fishing practices are threatening marine ecosystems, global seafood supplies and local livelihoods.Our goal is to predict the likelihood of fish species in each picture.

Dataset used:can be found in dataset file.

Model used : ResNet34 
architechture info: https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8

learn.recorder.plot():
It helps to visualize the loss change for range of learning rate.We then choose a lr that is just before the loss starts increasing.

Error metric used :
error_rate(source:https://docs.fast.ai/metrics.html#error_rate): It calculates mean difference between predicted and target image.


# Google Colab 
visit the below link to get started with google colab :

https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c

Then jump to dataset file and download the data

Modify the code given in fish_classifer.py and test your model

Here I have used resnet34.Choose the model based on the data, your system performance and outcome you wish.
