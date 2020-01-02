# ExchangeRatePrediction

Firstly, it makes a rough prediction according to the historical exchange rate data of GBP to EUR.
This has been implemented in tendency_prediction.py, around a Clockwork RNN (Jan Koutník, 2014) https://arxiv.org/abs/1402.3511

The authorised data of exchange rates come from online database https://fred.stlouisfed.org/series/DEXUSUK and https://fred.stlouisfed.org/series/DEXUSEU (The file 'GBP2EUR.csv is merely an example that shows the shape of the final data we requested from the internet.')

Secondly, it is going to optimise the rough tendency by involving the model of Brexit events.
The Brexit event data 'BREXIT.csv' come from the website https://www.theweek.co.uk/100284/brexit-timeline-key-dates-in-the-uk-s-break-up-with-the-eu

[1] Koutnik, J., Greff, K., Gomez, F. and Schmidhuber, J., 2014. A clockwork rnn. arXiv preprint arXiv:1402.3511.

[2] Federal Reserve Bank of San Francisco, 2017. Brexit: Whither the Pound? [online] Available at: https://www.frbsf.org/economic-research/publications/economic-letter/2017/april/brexit-whither-the-pound/ [Accessed 31 Oct. 2019].
