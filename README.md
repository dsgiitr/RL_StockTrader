# **Project Description**

![](/dqn/images/1.jpeg)

**Deep Q network**

In this project, we attempted to build a stock trading bot based on reinforcement learning using deep Q networks.The idea was that our stock trading bot, which was the agent in this case, will interact with the environment (i.e., the stock market) and eventually figure out when to buy, sell or hold stock to maximize profits, which is the reward in this case.

![](/dqn/images/2.jpeg)

The Q values were updated using a neural network in accordance with the bellman equation. Also,the target network and experience replay have been used to stabilize the model so that the neural network can converge eventually. Epsilon greedy policy has been used to allow the agent to explore the environment properly.

![](/dqn/images/3.jpeg)

**Minute wise trading**

Distinguishably our trading bot was training to do minute wise trading. We trained our model on data accounting minute wise stock prices from 9:30 A.M. to 3:30 P.M. on the daily basis.

**Short selling**

In order to make bot advance we also added a trading strategy i.e. short selling which makes the agent more versatile. In short selling, a position is opened by borrowing stocks that the bot predicts will decrease in value by a set future time. The bot checks the stock prices till end of the day for the stock prices to fall and if the price falls the bot buys the borrowed stocks that it short sold previously on the same day. Before the borrowing stocks the bot is predicting that the price will continue to decline and it can purchase them at a lower cost to earn the profit. Interest on the closing value of stocks borrowed to short sell is also applied which is subtracted from the profits earned.

**State, Actions &amp; Rewards**

**State** - State of our agent consists of trace of how the stock prices fell and rose over a period of past 100 time instances.

**Actions** - We provided our agent with three different actions of buying, holding and selling (including traditional and short selling) of stocks.

**Rewards** - We rewarded the agent with a randomly factorized Gaussian function for buying the stocks. No reward is given for holding the stocks. Reward awarded for selling is the profit gain as positive reward and negative reward for the loss.

![](/dqn/images/4.PNG)

The plot shows the rewards awarded to the model while trading the stocks at the time of training. 

![](/dqn/images/5.PNG)

The plot shows the loss occurred to the model while trading the stocks at the time of training.

The agent finally gives the profit to be 0.000008% and leaving us with the main balance of 461127452.

# **Repository Structure**

- **main.py** - This is the main script which imports all the other files to run the model.
- **loader.py** - This is the dataloader file which loads minute wise data of various years
- **env.py** - It contains the environment created for the model
- **model.py** - It includes the neural network architecture of DQN
- **train.py** - It consists of the functions required to train the model
- **test.py** - It consists of the functions required to test the model
- **visualisation.py** - It consists of all the graphs plotted to visualize the results given by the model both during training and testing
- **utils.py** - It is save and load the training data

# **Requirements**

This project requires:

- Python (3 or higher version)
- Numpy
- Pandas
- Matplotlib
- Pytorch

# **Running**

For running the model, we need to run the main.py file, which imports all the other files in it and calls them including the train.py and test.py files for training followed by testing the minute wise data present in the folder named oneminutedata.


