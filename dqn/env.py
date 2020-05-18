import math
import numpy as np

# Creating Environment for model
class Environment:

    def __init__(self, data, balance, history_t = 100):       
        self.data = data
        self.history_t = history_t
        self.reset(balance)
        

    def reset(self, balance):
        self.t = 0 # randomly initialise start
        self.balance = balance
        self.done = False   
        self.profits = 0
        self.positions = []
        self.number = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_t)] # keeps track of rise and fall in stocks
        return [self.position_value] + self.history

    def step(self, act):
        reward = 0
        stock_no = 1  
        # Taking closing stocks as reference
        stock_t = self.data.iloc[self.t, :]['C']
        if self.t == 0:
            stock_t_ = stock_t 
        else:
            stock_t_ = self.data.iloc[(self.t - 1), :]['C']
        interest_b = 0.1 # interest on borrow
        balance_lim = 200000000
        
        def findNSE(i, i_n): # finds when stock price falls within the given period
            k = i
            for j in range(i+1, i_n , 1): 
                if (self.data.iloc[j, :]['C'] < self.data.iloc[i, :]['C']).item():
                    k = j
                    break                    
            return k - i 
        
        def ub(): # finds the upperbound
            t_n = self.t    
            while self.data.iloc[t_n, 2] != 1530:
                t_n += 1
            return t_n
        
        def gaussian(x, mean, variance):
            sd = math.sqrt(variance)
            g = (1/(math.sqrt(2*math.pi)*sd))*(np.exp((-0.5)*(math.pow(((x-mean)/sd), 2))))
            return g
        
        # at the end sell all available stocks
        if self.data.iloc[self.t, 2] == 1530:
            profit = 0 
            for k in range(len(self.positions) - 1):
                profit += ((self.number[k] * stock_t) - self.positions[k])
            reward += np.float64(profit) 
            self.profits = profit
            self.balance += profit
            self.positions = []
            self.number = []
        
        # buying stocks   
        if act == 1:
            self.positions.append(stock_no * stock_t) # updates position 
            self.number.append(stock_no)  # number of stocks bought
            self.balance -= stock_no * stock_t
            reward += gaussian(act, 4.5, 1)*stock_no*(stock_t - stock_t_)
            
        if len(self.positions) >= 20: # preventing from keeping on buying stocks
            reward -= stock_t * stock_no

        # selling stocks
        elif act == 2:
            # short sell stocks           
            if len(self.positions) == 0:
                if findNSE(self.t, ub()) == 0: # if stock value keeps on rising
                    profit = self.data.iloc[ub() - 1, :]['C'] - self.data.iloc[self.t, :]['C']
                    reward -= ((stock_t * interest_b * (ub() - self.t) * 1.5) + profit)
                    self.balance -= ((stock_t * interest_b * (ub() - self.t)) + profit)
                elif findNSE(self.t, ub()) > 0: # if stock value falls
                    profit = self.data.iloc[self.t, :]['C'] - self.data.iloc[ub() - 1, :]['C']
                    reward += (profit - (stock_t * interest_b * findNSE(self.t, ub())))
                    self.balance += (profit - (stock_t * interest_b * findNSE(self.t, ub())))
                self.profits = profit
            # sell all stocks at once       
            else:  
                profit = 0                 
                for k in range(len(self.positions) - 1):
                    profit += ((self.number[k] * stock_t) - self.positions[k])
                reward += np.float64(profit)
                self.profits = profit
                self.balance += profit
                self.positions = []
                self.number = []
        
        # end the scenario once time runs out
        if self.t == len(self.data):
            self.done = True
        
        # punish if balance goes less than a limit and end scenario
        if self.balance <= balance_lim:
            reward -= stock_t * stock_no * 200
            self.done = True

        self.t += 1
        self.position_trace = []
        self.position_value = 0 # keeps trace of profits we get
        for k in range(len(self.positions) - 1):
            self.position_trace.append((self.number[k] * stock_t) - self.positions[k])
        for k in range(len(self.positions) - 1):
            self.position_value += ((2*((self.position_trace[k]-min(self.position_trace))/(max(self.position_trace)-min(self.position_trace)+1)))-1)/(len(self.positions))
        self.history.pop(self.history_t-1)
        self.history.insert(0, stock_t - stock_t_)
        
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        return [self.position_value] + self.history, reward, self.done, self.profits, self.balance
