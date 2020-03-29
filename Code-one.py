
import numpy as np
import pandas as pd
import math
import random

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

   
#Read dataset
df = pd.read_csv('dataset.csv')
            
tot_features=len(df.columns)-1
total_features=tot_features

x=df[df.columns[:tot_features]]
y=df[df.columns[-1]]

#splitting dataset and applying KNN
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y)
_classifier = KNeighborsClassifier(n_neighbors=5)
_classifier.fit(x_train, y_train)
predictions = _classifier.predict(x_test)
total_acc = accuracy_score(y_true = y_test, y_pred = predictions)
total_error = 1 - total_acc

swarm_size = 30   #population size
max_iterations = 50
omega = 0.2  #used in the fitness function
delta=0.2   #to set an upper limit for including a slightly worse particle in LAHC


def mutate(agent):
            percent=0.2
            numChange=int(tot_features*percent)
            pos=np.random.randint(0,tot_features-1,numChange) #choose random positions to be mutated
            agent[pos]=1-agent[pos] #mutation
            return agent

def LAHC(particle):
            _lambda = 15 #upper limit on number of iterations in LAHC
            target_fitness = find_fitness(particle) #original fitness
            for i in range(_lambda):
                    new_particle = mutate(particle) #first mutation
                    temp = find_fitness(particle)
                    if temp < target_fitness:
                        particle = new_particle.copy() #updation
                        target_fitness = temp
                    elif (temp<=(1+delta)*target_fitness):
                        temp_particle = new_particle.copy()
                        for j in range(_lambda):
							temp_particle1 = mutate(temp_particle) #second mutation
							temp_fitness = find_fitness(temp_particle1)
							if temp_fitness < target_fitness:
								target_fitness=temp_fitness
								particle=temp_particle1.copy() #updation
								break
            return particle   


def find_fitness(particle):
            features = []
            for x in range(len(particle)):
					if particle[x]>=0.5: #convert it to zeros and ones
                        features.append(df.columns[x])
            if(len(features)==0):
                        return 10000
            new_x_train = x_train[features].copy()
                new_x_test = x_test[features].copy()

            _classifier = KNeighborsClassifier(n_neighbors=5)
            _classifier.fit(new_x_train, y_train)
            predictions = _classifier.predict(new_x_test)
            acc = accuracy_score(y_true = y_test, y_pred = predictions)
            fitness = acc
            err=1-acc
            num_features = len(features)
            fitness =  alpha*err + (1-alpha)*(num_features/total_features)

            return fitness

def transfer_func(velocity): #to convert into an array of zeros and ones
            t=[]
            for i in range(len(velocity)):
                    t.append(abs(velocity[i]/(math.sqrt(1+velocity[i]*velocity[i])))) #transfer function inside paranthesis
            return t

#initialize swarm position and swarm velocity of SSD
swarm_vel = np.random.uniform(low=0, high=1, size=(swarm_size,tot_features))

swarm_pos = np.random.uniform(size=(swarm_size,tot_features))
swarm_pos = np.where(swarm_pos>=0.5,1,0)

c = 100
alpha= 0.9

gbest_fitness=100000
pbest_fitness = np.zeros(swarm_size)
pbest_fitness.fill(np.inf)  #initialize with the worse possible values
pbest = np.empty((swarm_size,tot_features))
gbest = np.empty(tot_features)
pbest.fill(np.inf)
gbest.fill(np.inf)

for itr in range(max_iterations):

                for i in range(swarm_size):
                  
                    swarm_pos[i] = LAHC(swarm_pos[i])   
                    fitness = find_fitness(swarm_pos[i])

                    if fitness < gbest_fitness:

                        gbest=swarm_pos[i].copy() #updating global best
                        gbest_fitness=fitness



                    if fitness < pbest_fitness[i]:
                        pbest[i] = swarm_pos[i].copy() #updating personal best
                        pbest_fitness[i]=fitness

                    r1 = random.random()
                    r2 = random.random()

					#updating the swarm velocity
                    if r1 < 0.5:
                        swarm_vel[i] = c*math.sin(r2)*(pbest[i]-swarm_pos[i]) +math.sin(r2)* (gbest-swarm_pos[i])
                    else:
                        swarm_vel[i] = c*math.cos(r2)*(pbest[i]-swarm_pos[i]) + math.cos(r2)*(gbest-swarm_pos[i])
                    
					#decaying value of c
					c=aplha*c;
					
					#applying transfer function and then updating the swarm position
                    t = transfer_func(swarm_vel[i])
                    for j in range(len(swarm_pos[i])):
                        if(t[j] < 0.5):
                            swarm_pos[i][j] = swarm_pos[i][j]
                        else:
                            swarm_pos[i][j] = 1 - swarm_pos[i][j]



selected_features = gbest
print(gbest_fitness)
            
number_of_selected_features = np.sum(selected_features)
print("#",number_of_selected_features)

features=[]
for j in range(len(selected_features)):
                if selected_features[j]==1:
                        features.append(df.columns[j])
new_x_train = x_train[features]
new_x_test = x_test[features]

_classifier = KNeighborsClassifier(n_neighbors=5)
_classifier.fit(new_x_train, y_train)
predictions = _classifier.predict(new_x_test)
acc = accuracy_score(y_true = y_test, y_pred = predictions)
fitness = acc
print("Acc:",fitness)
print("\n\n")            
            
        

