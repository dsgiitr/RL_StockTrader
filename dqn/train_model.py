import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from utils import save, load

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_dqn(env, Q, Q_new, LR, balance):

    # Parameters
    balance_start = balance 
    loss_function = nn.MSELoss() 
    optimizer = optim.Adam(list(Q.parameters()), lr=LR)    
    epoch_num = 200
    step_max = len(env.data)
    memory_size = 2500
    batch_size = 500
    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    epochs = []
    epsilon = 1.0
    epsilon_decrease = 5e-7
    epsilon_min = 0.01
    start_reduce_epsilon = 20000
    train_freq = 100
    update_q_freq = 10000
    gamma = 0.997
    epoch = 0
    
    try:
        epoch, Q, optimizer, epsilon, total_losses, total_rewards, epochs = load(Q, optimizer)
        print("file loaded")
        Q.train()
    except FileNotFoundError:
        print("file not found")
    
    while epoch < epoch_num:
        state = env.reset(balance_start)
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while (not done and step < step_max):
            
            if (step % 1 == 0):
                # Epsilon Greedy 
                if np.random.rand() > epsilon:
                    action = Q((torch.from_numpy(np.array(state, dtype=np.float32).reshape(1, -1))).to(device))
                    action = np.argmax(action.data.cpu().numpy())
                else:
                    action = random.randrange(3)

                state_n, reward, done, profits, balance = env.step(action)
      
                memory.append((state, action, reward, state_n, done))    # Update Memory

                if len(memory) > memory_size:
                    memory.pop(0)

                if len(memory) == memory_size:
                    if total_step % train_freq == 0:
                        # Experience Replay
                        shuffled_memory = np.random.permutation(memory) # Selecting random batches of data
                        memory_idx = range(len(shuffled_memory))
                        for i in memory_idx[::batch_size]:
                            batch = np.array(shuffled_memory[i:i + batch_size])
                            b_state = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                            b_action = np.array(batch[:, 1].tolist(), dtype=np.int32).reshape(batch_size, -1)
                            b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32).reshape(batch_size, -1)
                            b_state_n = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                            b_done = np.array(batch[:, 4].tolist(), dtype=np.bool).reshape(batch_size, -1)

                            q = Q((torch.from_numpy(b_state)).to(device))
                            q_ = Q_new((torch.from_numpy(b_state_n)).to(device))
                            maxQ = np.max(q_.data.cpu().numpy(), axis=1)
                            target = copy.deepcopy(q.data) 
                            for j in range(batch_size): # update target by using bellman equation
                                target[j, b_action[j]] = torch.from_numpy(b_reward[j] + gamma * maxQ[j] * (not b_done[j])).float().to(device)
                            Q.zero_grad()
                            loss = loss_function(q, target)
                            total_loss += loss.data.item()
                            loss.backward(retain_graph=True)
                            optimizer.step()

                if total_step % update_q_freq == 0:  # Update Q_new
                    Q_new = copy.deepcopy(Q)

                if epsilon > epsilon_min and (total_step > start_reduce_epsilon or epoch > 0): # Decay Epsilon
                    epsilon -= epsilon_decrease
                    epsilon = max(epsilon_min,epsilon)
        
                total_reward += reward
                state = state_n

                print("Epoch:", epoch, "Epsilon:", epsilon, "Step:", step, "Reward:", total_reward, "Total Balance:", balance)

            step += 1
            total_step += 1
        
        total_rewards.append(total_reward)
        total_losses.append(total_loss)
        epochs.append(epoch)
        epoch += 1

        if (epoch % 1 == 0 and epoch != 0):
            save(epoch, Q, optimizer, total_rewards, total_losses, epsilon, epochs)
            print("file saved")
           
    return Q, total_losses, total_rewards, epochs
