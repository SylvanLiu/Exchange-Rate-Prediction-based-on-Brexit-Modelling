# ExchangeRatePrediction

This is proposed for my personal interest about exploring the impact of Brexit event on the exchange rate between the RU and the UK, and implemented by machine learning related approaches.

Firstly, it makes a rough prediction only according to the historical data of the exchange rate of GBP to EUR, by using(reproducing) the Clockwork RNN(Jan Koutník, 2014).

*The authorised data of exchange rates come from the online database https://fred.stlouisfed.org/categories/15 (The file 'GBP2EUR.csv' is merely an example that shows the shape of the final data we requested from the internet.)*

Secondly, it is going to optimise/refine the rough tendency by learning the features of 'Brexit Events'(BEs) through a multilayer perceptron, with the n input(n is the number of BE features), and m output(m is the number of how many days after that BR). We concern the differences between m values of the true exchange rate and the m values predicted by cwrnn as the expected values of m output, so the errors are the differences between the output values and their expected value, then adjust the weights by backpropagation those errors. We train it with all happened BEs iteratively, and we assume the features that new BE has, then input features into the trained model, and predict the more precise future tendency by adding output values on the rough tendency.

The Brexit event data 'BREXIT.csv' come from the website https://www.womblebonddickinson.com/uk/insights/timelines/brexit-timeline

[1] Koutnik, J., Greff, K., Gomez, F. and Schmidhuber, J., 2014. A clockwork rnn. arXiv preprint arXiv:1402.3511. [online] Available at: https://arxiv.org/abs/1402.3511

[2] Federal Reserve Bank of San Francisco, 2017. Brexit: Whither the Pound? [online] Available at: https://www.frbsf.org/economic-research/publications/economic-letter/2017/april/brexit-whither-the-pound/

## Predicted Result (Roughly)
![alt text](https://github.com/SylvanLiu/ExchangeRatePrediction/blob/master/Prediction.png)

#### 短期内走势类别定义:

##### 1.突变型: (正/负)
定义: 在事件出现/公布当天突然出现相对的断崖式下跌/上升.
权重计算方式: 短期内平均’势能’的前后落差.
##### 2.反转型: (正/负)
定义: 由相对平稳到开始下跌/上升, 或在短期/长期的持续趋势下突然出现逆转.
权重计算方式: 归一化系数(避免权重过大) * 反转前趋势持续时间 * 反转前趋势内落差 * 反转后持续的时间 * 指数系数^反转后, 在与反转前同等时间内的趋势内落差(采用指数是因为此处可正可负, 若负则说明事件影响力低, 从而此处转变为负指数幂, 减小权重)
##### 3.加速型: (正/负)
定义: 在事件出现/公布当天突然出现加速之前所持续的趋势(可认为前后一阶导数会出现明显变化)
权重计算方式: 短期内一阶导数的平均’势能’的前后落差.
##### 4.扰动型:
定义: 在事件出现/公布前汇率走势相对平缓稳定, 而事件出现/公布后汇率开始出现明显持续的相对波动.
权重计算方式: 归一化系数(避免权重过大) *事件前稳定持续时间*事件后波动持续时间*事件后波动落差*事件后波动频率
##### 5.稳定型:
定义: 在事件出现/公布前汇率存在明显持续波动, 而事件出现/公布后汇率的波动程度明显降低.
权重计算方式: 归一化系数(避免权重过大) *时间前波动持续时间*事件前波动落差*事件前波动频率*时间后稳定持续时间
