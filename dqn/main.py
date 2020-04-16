import copy
import torch
import torch.nn as nn
from loader import load_data
from env import Environment
from model import Q_Network
from train_model import train_dqn
from test_model import test_by_q
from visualization import plot_train_test, plot_totalreward, plot_totalloss, plot_reward, plot_profits, plot_actions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = load_data(2013, 2016)
test = load_data(2016, 2017)

plot_train_test(train)
plot_train_test(test)

balance_train = 1000000000
balance_test = 1000000000
train_env = Environment(train, balance_train)
test_env = Environment(test, balance_test)

input_size = train_env.history_t + 1
output_size = 3
hidden_1 = 150
hidden_2 = 60
LR = 0.001

# Creating 2 models to avoid bootstrapping
Q = Q_Network(input_size, hidden_1, hidden_2, output_size)
Q_new = copy.deepcopy(Q)

# Utilising multiple GPU's if present
if torch.cuda.device_count() > 1:
    Q = nn.DataParallel(Q)
    Q_new = nn.DataParallel(Q_new)
    
Q = Q.to(device)
Q_new = Q_new.to(device)

Q, total_losses, total_rewards = train_dqn(train_env, Q, Q_new, LR, balance_train)

plot_totalreward(total_rewards)
plot_totalloss(total_losses)

test_rewards, test_profits, test_actions, index = test_by_q(test_env, Q, balance_test)

plot_reward(test_rewards)
plot_profits(test_profits)
plot_actions(test_actions, index)







