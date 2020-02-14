# ExchangeRatePrediction

This is proposed for my personal favour, and implemented by data analysis, mathematical modeling, also by reproducing a Clockwork RNN(Jan Koutník, 2014) .

Firstly, it makes a rough prediction according to the historical data of the exchange rate of GBP to EUR.
This has been implemented in tendency_prediction.py. The authorised data of exchange rates come from online database https://fred.stlouisfed.org/categories/15 (The file 'GBP2EUR.csv is merely an example that shows the shape of the final data we requested from the internet.')

Secondly, it is going to optimise/refine the rough tendency by involving the model of Brexit events, based on clustering and regression technologies. The Brexit event data 'BREXIT.csv' come from the website https://www.womblebonddickinson.com/uk/insights/timelines/brexit-timeline (This part is still in progress.)

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
