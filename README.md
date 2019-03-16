# q-trader

# https://linuxacademy.com/blog/amazon-web-services-2/deploying-a-containerized-flask-application-with-aws-ecs-and-docker/
# Install AWS CLI for Mac
xcode-select --install 
brew install awscli

# Configure AWS CLI
https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
aws configure

# Create Docker Repo on ECR
aws ecr create-repository --repository-name mia88

Add below to your .bash_profile:
export AWS_ECR_URL=Your AWS ECR Repository URL
export AWS_REGION=You AWS  Region

# Build and deploy Docker image to AWS ECR
./deploy.sh

# Feature Selection Tool
https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0

From eToro: RSI, 9MA RSI, MA20 + Bollinger Bands, MA50, MACD
MAMA - MESA Adaptive Moving Average
KAMA Kaufman Adaptive Moving Average

# Convert NN to Regression
# Generic MA trading system with binary Features
# Test model on S&P500 / NASDAQ / Bitcoin
# Strategy: Buy when price is above weekly PSAR (0.008, 0.2). Sell when price is below
# Strategy: Buy when daily SMA 200 is going up and price closes higher 200 SMA. Sell otherwise
# 50 MA Trading Systems
https://www.amazon.com/dp/B07JYLCK68/?ref=idea_lv_dp_vv_d&tag=aiponsite-20&linkCode=iil&ascsubtag=amzn1.ideas.S5BXKSCC6S15
# Top 5 indicators: http://www.newtraderu.com/2019/01/04/the-top-5-effective-technical-indicators/

# Set SL and TP based on: 
+ stats (y_pred_id, minr, maxr)
+ ATR
# Option to set SL and TP on trade level

# Use Kraken price data / Use cryptowat.ch API
https://api.kraken.com/0/public/OHLC?pair=ETHUSD&interval=1440

# Bitfinex Integration

# Use volume bars
https://www.investopedia.com/articles/trading/10/data-based-intraday-chart-intervals.asp

# LSTM
https://towardsdatascience.com/predicting-ethereum-prices-with-long-short-term-memory-lstm-2a5465d3fd
https://www.datacamp.com/community/tutorials/lstm-python-stock-market
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

# Precision, Recall, ROC, AUC
https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c

# TODO: Practice with Hourly Data

# Google Colaboratory:
https://colab.research.google.com/notebooks/welcome.ipynb

# Most used indicators
https://www.quantinsti.com/blog/indicators-build-trend-following-strategy/

# Hyperparameter Tuning
https://en.wikipedia.org/wiki/Hyperparameter_optimization
https://medium.freecodecamp.org/an-introduction-to-high-dimensional-hyper-parameter-tuning-df5c0106e5a4

# The Ocean Algo Trading
https://medium.com/the-ocean-trade/the-ocean-x-algo-trading-lesson-1-time-series-analysis-fa3b76f1d4a3

# Moon Ingress
https://www.astro.com/swisseph/ing_mo.txt

# Financial Astrology and Neural Networks
https://www.scribd.com/document/187532408/Alphee-Lavoie-Neural-Networks-in-Financial-Astrology
The basic neural net in the program has the following parameters: for the 7-year market 
(i.e. 2000 examples for learning), it has 600-700 inputs. 
According to the theory of neural networks, the neural net with 600 inputs could be educated at the market 
with price history of more than 20 (sometimes 30) years. Otherwise the neural net will be 'over-educated' 
(it means that the neural net works perfectly within the optimizing interval and does not work at all 
within the testing interval). We created a specialized neural net, and 7 years of history is enough. 
In this case, the total process of learning takes only 3-4 minutes). 
The program allows for setting the amount of hidden neurons; in our opinion, less than 100 is enough

# A Machine Learning framework for Algorithmic trading on Energy markets
https://towardsdatascience.com/https-medium-com-skuttruf-machine-learning-in-finance-algorithmic-trading-on-energy-markets-cb68f7471475

# TODO: Trade multiple coins

# TODO: Use ephemeris for price prediction

# TODO: Predict price rise for 1 week / month. Use weekly / monthly market return

# Time Series Forecasting
https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/

# Kaggle Courses
https://www.kaggle.com/learn/overview

# Cross Validation
https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f
Keras: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9

# Turning Machine Learning Models into APIs in Python
https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

# Securing Docker secrets on AWS
https://aws.amazon.com/blogs/security/how-to-manage-secrets-for-amazon-ec2-container-service-based-applications-by-using-amazon-s3-and-docker/

# Tips to reduce Docker image size
https://hackernoon.com/tips-to-reduce-docker-image-sizes-876095da3b34

# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html
# https://dzone.com/articles/deploying-docker-containers-to-aws-ecs

# Use Training + Validation + Test Data to avoid overfitting
# Stop training when MSE on test reaches bottom while MSE on validation still goes down

# Deploy Docker to AWS ECS and Fargate using Terraform scripts
# https://thecode.pub/easy-deploy-your-docker-applications-to-aws-using-ecs-and-fargate-a988a1cc842f

# XGBOOST: https://www.kaggle.com/shreyams/stock-price-prediction-94-xgboost

# Cloud Based Trading
# https://www.quantinsti.com/blog/epat-project-automated-trading-maxime-fages-derek-wong/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com

# Read: Прогнозирование финансовых временных рядов с MLP в Keras
# https://habr.com/post/327022/

# Read: Deep Learning – Artificial Neural Network Using TensorFlow In Python 
# https://www.quantinsti.com/blog/deep-learning-artificial-neural-network-tensorflow-python/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com

#TODO: AutoKeras: Build optimal NN architecture: https://towardsdatascience.com/autokeras-the-killer-of-googles-automl-9e84c552a319

#TODO: Implement Random Forest
#TODO: https://medium.com/@huangkh19951228/predicting-cryptocurrency-price-with-tensorflow-and-keras-e1674b0dc58a

# TODO: Predict DR and 
# TODO: Adjust strategy to HOLD when DR is less that exchange fee

# See: https://www.vantagepointsoftware.com/mendelsohn/preprocessing-data-neural-networks/

# Exit strategy: Sell permanently when state 80 is changed to other state

# TODO:
# Separate train_model and run_model procedures

# Calculate R in USD

# Trade with daily averege price: split order in small chunks and execute during day

# Populate Trade Log for train/test mode

# Use Monte Carlo to find best parameters 

# Ensemble strategy: avg of best Q tables

# Add month/day to state

# Test price change scenario

# Sentiment analysis: https://github.com/Crypto-AI/Stocktalk

# Training: Load best Q and try to improve it. Save Q if improved

# Optimize loops. See https://www.datascience.com/blog/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/

# Store execution history in csv
# Load best Q based on execution history

# Solve Unknown State Problem: Find similar state

# Test model with train or test data?

# Implement Dyna Q

# Predict DR based on State (use R table)

# Implement Parameterised Feature List
# Use function list: https://realpython.com/blog/python/primer-on-python-decorators/
# Lambda, map, reduce: https://www.python-course.eu/lambda.php

# Automatic Data Reload (based on file date)

# Stop Iterating when Model Converges (define converge criteria)
# Converge Criteria: best result is not improved after n epochs (n is another parameter)

# ********************** Useful Links ************************************
Reinforcement learning tutorial using Python and Keras
https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/

AutoML
https://www.datacamp.com/community/tutorials/automated-machine-learning-auto-keras
https://www.pyimagesearch.com/2019/01/07/auto-keras-and-automl-a-getting-started-guide/

As a trader, what did you learn after blowing up your account that helped you become successful later?
https://www.quora.com/As-a-trader-what-did-you-learn-after-blowing-up-your-account-that-helped-you-become-successful-later

Reinforcement Learning for Optimized Trade Execution
http://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf

10 Ways To Limit Your Drawdowns In Trading Capital
http://www.newtraderu.com/2014/06/26/9645/

Multivariate Time Series Forecasting with LSTMs in Keras
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

Arbitrage: https://hackernoon.com/how-to-make-money-on-arbitrage-with-cryptocurrencies-6618bdad3ce1

Using the latest advancements in deep learning to predict stock price movements
https://towardsdatascience.com/aifortrading-2edd6fac689d
https://github.com/borisbanushev/stockpredictionai

*** From Evgeny Tartakovsy
Google Cloud
AutoKeras 
RL 
DEAP: https://deap.readthedocs.io/en/master/ 
skit-learn, 
TPOT: https://automl.info/tpot/
Убрать гетеростатистичность - бары не по времени а по объему / трейдам

Evolution Strategies
https://blog.openai.com/evolution-strategies/
https://arxiv.org/pdf/1703.03864.pdf

Predicting Ethereum prices with Long Short Term Memory (LSTM)
https://towardsdatascience.com/predicting-ethereum-prices-with-long-short-term-memory-lstm-2a5465d3fd

AI for algorithmic trading: 7 mistakes that could make me broke
https://medium.com/@alexrachnog/ai-for-algorithmic-trading-7-mistakes-that-could-make-me-broke-a41f94048b8c

Telegram Bot Development
https://groosha.gitbooks.io/telegram-bot-lessons/content/

High Frequency Trading Architecture
https://towardsdatascience.com/assembling-an-entry-level-high-frequency-trading-hft-system-e7538545b2a9

Deep Reinforcement Learning: https://habr.com/ru/post/437020/

LEAN Trading Engine: https://github.com/QuantConnect/Lean

Stop Loss / Take Profit / Position Sizing
http://www.newtraderu.com/2018/10/27/the-ultimate-guide-to-risk-managment/

Deploy Docker to AWS
https://dzone.com/articles/deploying-docker-containers-to-aws-ecs

Top 6 Risk Ratios You must Know!
https://www.wallstreetmojo.com/risk-adjusted-returns/

# ********************** Features ************************************
Fixed position sizing
Position sizing based on balance %


# ********************** Lessons Learned *************************************
AutoML is very slow and not efficient for time series

Shorting can be good on bear market, but not so good at bull market

Model is as good as train data. Model needs to be trained on variety of data

Predict DR for several days => not any better

Do not trust signals if you cannot test them on historical data

Do not put all your eggs in one basket. Diversify.

Trade as soon as signal is produced. Late entry is better than no entry

Short trade SR Calculation should be based on original price.
Daily open/close position is recommended for short trade. Then daily DR can be used for SR calculation

HH/LL Stop Loss is not better than % SL

Historical Results which were used for training cannot be trusted
Only Test Results can be trusted
