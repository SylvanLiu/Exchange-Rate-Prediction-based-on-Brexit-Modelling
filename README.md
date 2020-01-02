# ExchangeRatePrediction

Project aims to combine the chronological data of Brexit events and EUR as well as GBP exchange rates, build a model to analyse in what tendency will Brexit events impact on the exchange rates. And subsequently make a prediction to the exchange rates trend during the UK general election on December 12.

Firstly, it makes a rough prediction according to the historical exchange rate data of GBP to EUR.
This has been implemented in tendency_prediction.py, around a Clockwork RNN (Jan Koutník, 2014) https://arxiv.org/abs/1402.3511

Secondly, it is going to optimise the rough tendency by involving the model of Brexit events.
The Brexit event data 'BREXIT.csv' come from the website https://www.theweek.co.uk/100284/brexit-timeline-key-dates-in-the-uk-s-break-up-with-the-eu

