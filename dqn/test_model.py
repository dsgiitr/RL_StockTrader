import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Find test results   
def test_by_q(env, Q, balance):

    state = env.reset(balance)
    index = []
    test_acts = []
    test_rewards = []
    test_profits = []

    for l in range(len(env.data)-1):
        index.append(l)
        action = Q(torch.from_numpy(np.array(state, dtype=np.float32).reshape(1, -1)).to(device)).squeeze().data.cpu().numpy()
        action = np.argmax(action, axis=0)
        if (l % 50 == 0):
            test_acts.append(action.item())   
        state_n, reward, done, profits, balance = env.step(action)
        print("Rewards: ", reward, "Profits: ", profits, "Balance: ", balance)
        test_rewards.append(reward)
        if (l % 30 == 0):
            test_profits.append(profits)
        state = state_n
        
    return test_rewards, test_profits, test_acts, index

