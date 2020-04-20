import matplotlib.pyplot as plt

# Plotting the dataset
def plot_train_test(data):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Closing Value')
    fig.suptitle('Closing Value')
    fig.tight_layout()
    y = data.iloc[:, 0]
    plt.plot(y)
    plt.show()
    
# Plot train graphs            
def plot_totalreward(total_rewards):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Rewards')
    fig.suptitle('Rewards')
    fig.tight_layout()
    y = total_rewards
    plt.plot(y)
    plt.show()
    
def plot_totalloss(total_losses):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Total loss')
    fig.suptitle('Total Loss')
    fig.tight_layout()
    y = total_losses
    plt.plot(y)
    plt.show()
    
# Plot test graphs    
def plot_reward(rewards):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Rewards')
    fig.suptitle('Rewards')
    fig.tight_layout()
    y = rewards
    plt.plot(y)
    plt.show()
    
def plot_profits(profits):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Total Profits')
    fig.suptitle('Total Profits')
    fig.tight_layout()
    y = profits
    plt.plot(y)
    plt.show()
    
def plot_actions(actions, index):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    colors=['red' if i==1 else 'blue' if i==2 else None for i in actions]
    markers=['X' if i==1 else 'D' if i==2 else None for i in actions]
    ax = plt.gca()
    for x, y, c, m in zip(index, actions, colors, markers):
        ax.scatter(x, y, alpha=0.8, c=c,marker=m)   
    ax.set_xlabel('Steps')
    ax.set_ylabel('Actions')
    fig.suptitle('Actions')
    fig.tight_layout()
    plt.show()

