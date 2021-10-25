# Import library
import numpy as np
import matplotlib.pyplot as plt

# Shapere function
def Shapere(x):
    z = np.sum(np.square(x))
    return z

# Parameter setting
d = 10
xMin, xMax = -100, 100
vMin, vMax = -0.2*(xMax - xMin), 0.2*(xMax - xMin)
MaxIter = 3000
ps = 10
c1 = 2
c2 = 2
w = 0.9 - ((0.9 - 0.4)/MaxIter)*np.linspace(0,MaxIter,MaxIter)

def limitV(V):
    for i in range(len(V)):
        if V[i] > vMax:
            V[i] = vMax
        if V[i] < vMin:
            V[i] = vMin
    return V

def limitX(X):
    for i in range(len(X)):
        if X[i] > xMax:
            X[i] = xMax
        if X[i] < xMin:
            X[i] = xMin
    return X

#%% Algorithm
def Optimization():
    class Particle():
        def __init__(self):
            self.position = np.random.uniform(xMin,50,[ps,d])
            self.velocity = np.random.uniform(vMin,vMax,[ps,d])
            self.cost = np.zeros(ps)
            self.cost[:] = Shapere(self.position[:])
            self.pbest = np.copy(self.position)
            self.pbest_cost = np.copy(self.cost)
            self.index = np.argmin(self.pbest_cost)
            self.gbest = self.pbest[self.index]
            self.gbest_cost = self.pbest_cost[self.index]
            self.BestCost = np.zeros(MaxIter)
            
        def evaluate(self):
            for it in range(MaxIter):
                for i in range(ps):
                    self.velocity[i] = (w[it]*self.velocity[i]
                                        +c1*np.random.randn(d)*(self.pbest[i] - self.position[i])
                                        +c2*np.random.randn(d)*(self.gbest - self.position[i]))
                    self.velocity[i] = limitV(self.velocity[i])
                    self.position[i] = self.position[i] + self.velocity[i]
                    self.position[i] = limitX(self.position[i])
                    self.cost[i] = Shapere(self.position[i])
                    if self.cost[i] < self.pbest_cost[i]:
                        self.pbest[i] = self.position[i]                        
                        self.pbest_cost[i] = self.cost[i]
                        if self.pbest_cost[i] < self.gbest_cost:
                            self.gbest = self.pbest[i]
                            self.gbest_cost = self.pbest_cost[i]
                self.BestCost[it] = self.gbest_cost
                
        def Plot(self):
            plt.semilogy(self.BestCost)
            plt.ylim([10e-120, 10e20])
            plt.xlim([0,3000])
            plt.ylabel("Best Function value")
            plt.xlabel("Number of Iteration")
            plt.title("Particle Swam Optimization of Sphere Function")
            
            print("Best fitness value=",self.gbest_cost)
    
    a = Particle()
    a.evaluate()
    a.Plot()
    
#%% Run
Optimization()