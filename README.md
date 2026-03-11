HMM (Hidden Markov Models) 
BTC-USD written by AI.
Started with GPT and then used Claude to get it fixed up and running.

https://fvfbnxarig5auzkds9alpr.streamlit.app/


ToDo;

i think candlestick data is pulled live (the candle_close data gets retrieved even for live candles that haven't closed yet.)

data broken or no longer providing desired results. 
discord bot not pushing msg's either.

since strategy is not returning good results anymore, try stop losses.

not sure why strat has changed maybe HMM model returns very different results if a few days pass, few new latest candles and the few from 2 years ago also move forward.

might be worth analyzing how model works further
display regime status, in dashboard or discord bot

what indicators flagged the buy/sell

[detect_regimes] Regime distribution:
regime
Neutral    12116
Bull        3304
Crash       1944
