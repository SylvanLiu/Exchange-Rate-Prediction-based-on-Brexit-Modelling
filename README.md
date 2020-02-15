# ExchangeRatePrediction

This project is proposed for my interest in exploring the impact of Brexit event on the exchange rate between the RU and the UK and implemented by machine learning related approaches.

Firstly, it makes a rough prediction only according to the historical data of the exchange rate of GBP to EUR, by reproducing/using a specific Clockwork RNN(Jan Koutník, 2014).

Secondly, it is going to optimise/refine the rough tendency by learning the features of 'Brexit Events'(BEs) through a multilayer perceptron, with the n input(n is the number of BE features), and m output(m is the number of how many days after that BR). 

We concern the differences between m values of the true exchange rate and the m values which are predicted by cwrnn as the expected values of m output, so the errors are the differences between the output values and their expected value, then adjust the weights by backpropagation those errors. We train it with all happened BEs iteratively, and we assume the features that new BE has. Input there features into the trained model, finally attain the more precise future tendency by adding every output value on each date it corresponds to.

*The authorised data of exchange rates come from the online database https://fred.stlouisfed.org/categories/15 (The file 'GBP2EUR.csv' is merely an example that shows the shape of the final data we requested from the internet.)*

*The Brexit event data 'BREXIT.csv' comes from the website https://www.womblebonddickinson.com/uk/insights/timelines/brexit-timeline*

[1] Koutnik, J., Greff, K., Gomez, F. and Schmidhuber, J., 2014. A clockwork rnn. arXiv preprint arXiv:1402.3511. [online] Available at: https://arxiv.org/abs/1402.3511

[2] Federal Reserve Bank of San Francisco, 2017. Brexit: Whither the Pound? [online] Available at: https://www.frbsf.org/economic-research/publications/economic-letter/2017/april/brexit-whither-the-pound/

## Predicted Result (Roughly)
![alt text](https://github.com/SylvanLiu/ExchangeRatePrediction/blob/master/Prediction.png)

## Predicted Result (Roughly)
![alt text](https://github.com/SylvanLiu/ExchangeRatePrediction/blob/master/Prediction_.png
