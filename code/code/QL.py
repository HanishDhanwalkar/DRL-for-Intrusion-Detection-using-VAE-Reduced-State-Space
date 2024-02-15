import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

import KDDdata

# Xtrain , Ytrain = pd.read_csv("data/compressed_data_train.csv"), KDDdata.train_data()[1]
# Xtest , Ytest = pd.read_csv("data/compressed_data_test.csv"), KDDdata.test_data()[1]
# Xtrain = Xtrain.drop(['Unnamed: 0'], axis=1)
# Xtest = Xtest.drop(['Unnamed: 0'], axis=1)
Xtrain , Ytrain = KDDdata.train_data()
Xtest , Ytest = KDDdata.test_data()
# Xtrain = Xtrain.drop(['Unnamed: 0'], axis=1)
# Xtest = Xtest.drop(['Unnamed: 0'], axis=1)

print(pd.DataFrame(Xtrain).head())
print(pd.DataFrame(Ytrain).head())

# Xtrain = Xtrain.to_numpy()
# Xtest = Xtest.to_numpy()

# Hyperparameters
gamma = 0.95  
epsilon = 0.1  
num_episodes = 50
num_iterations = 100

# Model architecture
model = Sequential()
model.add(Dense(32, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear')) 
model.compile(optimizer='adam', loss='mse')

# Q-learning algorithm

for epoch in range(num_episodes):
    print(f"Episode {epoch + 1}/{num_episodes}")
    for iteration in range(num_iterations):
        
        index = np.random.randint(len(Xtrain))
        state = Xtrain[index]

        if np.random.rand() < epsilon:# Exploration-exploitation trade-off
            action = np.random.randint(2) 
        else:
            q_values = model.predict(state.reshape(1, -1))
            action = np.argmax(q_values)

        reward = 1 if Ytrain[index] == 1 else -1

        next_state = Xtrain[np.random.randint(len(Xtrain))]
        q_target = reward + gamma * np.max(model.predict(next_state.reshape(1, -1)))

        q_values = model.predict(state.reshape(1, -1))
        q_values[0, action] = q_target

        model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)

# testing
miss = 0
for index in range(100):
    state = Xtest[index]
    q_values = model.predict(state.reshape(1, -1))
    action = np.argmax(q_values)
    print("predicted: ", action, "Actual: ", Ytest[index] )

    if action != Ytest[index][0]:
        miss += 1
print("No. of wrongly predicted: ", miss - 10)