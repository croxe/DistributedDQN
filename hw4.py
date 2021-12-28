import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt
from memory_remote import ReplayBuffer_remote
from custom_cartpole import CartPoleEnv

FloatTensor = torch.FloatTensor
# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Set result saveing floder
result_floder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)

def plot_result(total_rewards ,learning_num, legend, num_proc):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
        
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig(ENV_NAME + "_DQN_"+ str(num_proc) +".png")
    plt.show()

ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

Memory_Server = ReplayBuffer_remote.remote(2000)
simulator = CartPoleEnv()

@ray.remote
class DQN_Model_Server():
    def __init__(self, env, hyper_params, memory=Memory_Server):
        state = env.reset()
        self.batch_size = hyper_params['batch_size']  
        self.update_steps = hyper_params['update_steps']
        self.learning_episodes = hyper_params['training_episodes'] 
        self.beta = hyper_params['beta']
        self.test_interval = hyper_params['test_interval']
        self.action_space = hyper_params['action_space']
        input_len = len(state)
        output_len = hyper_params['action_space']
        self.eval_model = DQNModel(input_len, output_len, learning_rate=hyper_params['learning_rate'])
        self.target_model = DQNModel(input_len, output_len)
        self.steps = 0
        self.memory = memory
        self.batch_num = hyper_params['training_episodes'] // hyper_params['test_interval']
        self.result = []
        self.old_q_networks = []
        self.new_q_network = False
        self.episodes = 0
        self.use_target_model = hyper_params['use_target_model']
       
    
    def update_batch(self):
        if self.episodes >= self.learning_episodes or self.batch_size > ray.get(self.memory.__len__.remote()) :  
            return
        batch = ray.get(self.memory.sample.remote(self.batch_size))
        (states, actions, reward, next_states, is_terminal) = batch
        terminal = FloatTensor([0 if t else 1 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size, dtype=torch.long)
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        if self.use_target_model:
            _, q_next = self.target_model.predict_batch(next_states)
        else:
            _, q_next = self.eval_model.predict_batch(next_states)
        q_target = torch.add(reward, terminal * (torch.max(q_next, dim = 1)[0] * self.beta))
        self.eval_model.fit(q_values, q_target)
        self.steps += self.update_steps
        if self.episodes // self.test_interval + 1 > len(self.old_q_networks):
            self.old_q_networks.append(ray.put(self.eval_model))
            self.new_q_network = True
    
    def get_steps(self):
        return self.steps
    
    def get_done_and_steps(self):
        return self.episodes >= self.learning_episodes, self.steps
    
    def set_avg_reward(self, avg_reward):
        self.result.append(avg_reward)
    
    def get_states(self):
        return self.episodes >= self.learning_episodes

    def get_len_q(self):
        return len(self.old_q_networks)
    
    def get_network(self):
        if self.new_q_network:
            self.new_q_network = False
            return self.old_q_networks[-1]
        else:
            return None
    
    def get_results(self):
        return self.result
    
    def get_model(self):
        return self.eval_model
    
    def replace_model(self):
        self.target_model.replace(self.eval_model)
        
    def update_episodes(self):
        self.episodes += self.test_interval
    
    
    
class DQN_agent():
    def __init__(self, env, hyper_params, cw_num, ew_num, memory=Memory_Server, action_space=2,
                 training_episodes=10000, test_interval=50):
        self.action_space = hyper_params['action_space']
        self.training_episodes = hyper_params['training_episodes']        
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.memory_size = hyper_params['memory_size']
        self.test_interval = hyper_params['test_interval']
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.cw_num = cw_num
        self.ew_num = ew_num      
        self.memory = Memory_Server
        self.model_server = DQN_Model_Server.remote(env, hyper_params, memory=Memory_Server)
        
    def learn_and_evaluate(self):
        workers_list = []
        for i in range(self.cw_num):
            cw_id = collecting_worker.remote(self.model_server, self.env, self.update_steps, self.max_episode_steps,
                                             self.training_episodes, self.test_interval, self.model_replace_freq, Memory_Server,
                                             self.action_space, self.final_epsilon)
            workers_list.append(cw_id)
        for i in range(self.ew_num):
            ew_id = evaluation_worker.remote(self.model_server, self.env, self.max_episode_steps, self.training_episodes, self.test_interval,
                                             self.ew_num)
            workers_list.append(ew_id)
        ray.wait(workers_list, len(workers_list))
        return ray.get(self.model_server.get_results.remote())


    
@ray.remote
def collecting_worker(DQN_server, env, update_steps, max_episode_steps, training_episodes, test_interval,
                      model_replace_freq, memory=Memory_Server, action_space=2, final_epsilon=0.1):
    def linear_decrease(initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(curr_state, curr_steps):
        epsilon = linear_decrease(1, final_epsilon, curr_steps, 100000)
        if uniform(0, 1) < epsilon:
            #return action
            return randint(0, action_space - 1)
        else:
            #return action
            return ray.get(DQN_server.get_model.remote()).predict(state)
        
    while True:
        # call predict(state)
        # call update()
        collect_done, curr_steps = ray.get(DQN_server.get_done_and_steps.remote())
        if collect_done:
            break
        for episode in tqdm(range(test_interval), desc="Training"):
            state = env.reset()
            done = False
            steps = 0
            while steps < max_episode_steps and not done:
                action = explore_or_exploit_policy(state, curr_steps)
                next_state, reward, done, _ = env.step(action)
                memory.add.remote(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                curr_steps += 1
                if (steps % update_steps) == 0:
                    DQN_server.update_batch.remote()
                if (curr_steps % model_replace_freq) == 0:
                    DQN_server.replace_model.remote()
        DQN_server.update_episodes.remote()
    
    
@ray.remote
def evaluation_worker(DQN_server, env, max_episode_steps, training_episodes, test_interval, eval_worker, trials=30):
    while True:
        if ray.get(DQN_server.get_states.remote()):
            break
        # call eval model
        eval_model = ray.get(DQN_server.get_network.remote())
        if eval_model == None:
            continue
        total_reward = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = env.reset()
            steps = 0
            done = False
            while steps < max_episode_steps and not done:
                action = ray.get(eval_model).predict(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
        avg_reward = total_reward / trials
        print(avg_reward)
        #f = open(result_file, "a+")
        #f.write(str(avg_reward) + "\n")
        #f.close()
        #if avg_reward >= self.best_reward:
        #    self.best_reward = avg_reward
        #    self.save_model()
        DQN_server.set_avg_reward.remote(avg_reward)
    #return avg_reward

    

hyperparams_CartPole = {
    'training_episodes' : 10000,
    'test_interval' : 50,
    'epsilon_decay_steps' : 100000,   
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True,
    'action_space' : 2,
}
cw_num = 12
ew_num = 12
start_time = time.time()
agent = DQN_agent(simulator, hyperparams_CartPole, cw_num=cw_num, ew_num=ew_num, memory=Memory_Server)
final_result = agent.learn_and_evaluate()
run_time = time.time() - start_time

print("Learning time:\n", run_time)
print(final_result)
plot_result(final_result, hyperparams_CartPole['test_interval'], 
            ["DQN " + str(cw_num) + "Processes; Time: " + str(run_time)], cw_num)