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


# Kaggle Courses
https://www.kaggle.com/learn/overview

# Cross Validation
https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f
Keras: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

# Hyperparameter Optimization
https://en.wikipedia.org/wiki/Hyperparameter_optimization

# Turning Machine Learning Models into APIs in Python
https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

# Securing Docker secrets on AWS
https://aws.amazon.com/blogs/security/how-to-manage-secrets-for-amazon-ec2-container-service-based-applications-by-using-amazon-s3-and-docker/

# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html
# https://dzone.com/articles/deploying-docker-containers-to-aws-ecs

# Use Training + Validation + Test Data to avoid overfitting
# Stop training when MSE on test reaches bottom while MSE on validation still goes down

# Deploy Docker to AWS ECS and Fargate using Terraform scripts
# https://thecode.pub/easy-deploy-your-docker-applications-to-aws-using-ecs-and-fargate-a988a1cc842f

# Deploy Docker to AWS
# https://dzone.com/articles/deploying-docker-containers-to-aws-ecs

# XGBOOST: https://www.kaggle.com/shreyams/stock-price-prediction-94-xgboost

# Cloud Based Trading
# https://www.quantinsti.com/blog/epat-project-automated-trading-maxime-fages-derek-wong/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com

# Read: Прогнозирование финансовых временных рядов с MLP в Keras
# https://habr.com/post/327022/

# Read: Deep Learning – Artificial Neural Network Using TensorFlow In Python 
# https://www.quantinsti.com/blog/deep-learning-artificial-neural-network-tensorflow-python/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com

#TODO: Calculate daily price with time shift using hourly data

#TODO: Calculate Expectancy Ratio: http://www.newtraderu.com/2017/11/27/formula-profitable-trading/

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

# Integrate with Telegram Bot

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
