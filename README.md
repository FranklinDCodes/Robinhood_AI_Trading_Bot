# Robinhood_AI_Trading_Bot
A bot that places orders on Robinhood based on feedback from a series of PyTorch neural networks.


------ File Descriptions ------

  main.py: 
    This code loops infinitly, looking to make profitable sales during market hours, and making purchases near market close based on neural network feedback.
  
  daily_market_report.py: 
    Contains 1 function that uses the neural networks in "nets" to rate over 700 stocks on the likelyhood of their price increasing.
  
  market_data_LSTM.py: 
    Produces live or historical stock market data as input for training or getting information from neural networks.

  
  nets/ 
  
      Contains 15 different neural networks that each were trained on a unique list of stocks.


  operating_data/ 
  
      unconfirmed.csv: 
        A list of placed but unexecuted buy and sell orders.
      
      owned.csv: 
        A list of currently owned stocks.
      
      last_run.txt: 
        A date string of the last time purchases were made. Just to make sure AI reccomendations are only purchased once a day.
        
      otp.txt: 
        The code required to generate a one time OTP to login to Robinhood (it's currently just 0s, not a legitimate code).
        
      pw.txt: 
        The ecrypted digits of the Robinhood password (it's just "password," not an actual password).

  
  logs/ 
  
      id.txt: 
        A perpetually counting value used by main.py to assign unique ids to market orders in logs.db and other files.
  
      logs.db: 
        A database containing multiple different logs to record any trades made by main.py


  nn_generator/

      network_trainer.py: 
        The code that generated and trained neural networks in the nets directory.

      market_data_LSTM.py:
        Same as in root directory, used for fetching historical market data from market_data_12_year.db to train the neural networks.

      market_data_12_year.db:
        NOT PRESENT IN REPOSITORY. Too large to upload, but it contains 12 years of historical data on each the stocks that the neural networks are trained on.
