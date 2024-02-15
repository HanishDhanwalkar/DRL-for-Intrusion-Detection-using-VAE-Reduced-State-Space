# DRL-for-Intrusion-Detection-using-VAE-Reduced-State-Space

## Table of Contents
1. [Description](#desc)
2. [Reference Papers](#papers)
3. [Tasks-done](#tasks)
4. [Tasks-to-do](#tasks-to-do)

<a name="desc"></a>
## Description
 Implement a VAE to learn a reduced state space representation from the NSL-KDD dataset, capturing essential features of normal network traffic.  Develop an RL agent that operates within the reduced state space. The RL agent's primary objective is to make adaptive decisions for intrusion detection, minimizing false positives and false negatives.
</br>

<a name="papers"></a>
## Papers:
1. Application of deep reinforcement learning to intrusion detection for supervised problems by Manuel Lopez-Martin∗, Belen Carro, Antonio Sanchez-Esguevillas
2. Adversarial environment reinforcement learning algorithm for intrusion detection by Guillermo Caminero, Manuel Lopez-Martin, Belen Carro
 
</br>

<a name="tasks"></a>
## Tasks done:
1. Implement the VAE for feature extraction, training it on the dataset.
2. Create a reduced state space using the VAE's latent representations.
3. Prepare the data for RL training, including defining states, actions, and rewards.
4. Implement the RL agent with the reduced state space.
5. Design the RL agent's action space, rewards, and episode setup.
6. Train the RL agent using the NSL-KDD dataset.

<a name="tasks-to-do"></a>
## Tasks to be done:
1. Evaluate the RL agent's performance using various metrics (e.g., detection rate, false positive rate).
2. Fine-tune hyperparameters for both the VAE and RL components based on evaluation results.
3. Consider using cross-validation to assess the model's generalization.




