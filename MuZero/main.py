"""
Adapted MuZero (https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules)
to learn policy and value function for classical planning. Uses MCTS for single player.
"""
import argparse
import logging
import os
import re
import subprocess
import sys
import time
import pickle
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy


from pyperplan import grounding
from pyperplan.pddl.parser import Parser
import feature
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

from trainer import Trainer
from policy import Rlstm
from planningEnv import PlanningAgent
from mcts import execute_episode
from util.action_encoder import ActionEncoder
from config import StackedStateConfig
from config import AlphaZeroConfig as config
from util.alphazero_util import action_spaces_new
from util.alphazero_util import parse_global_list_training, HelperTrainingExample

def _parse(domain_file, problem_file):
    # Parsing
    parser = Parser(domain_file, problem_file)
    logging.info(f"Parsing Domain {domain_file}")
    domain = parser.parse_domain()
    logging.info(f"Parsing Problem {problem_file}")
    problem = parser.parse_problem(domain)
    logging.debug(domain)
    logging.info("{} Predicates parsed".format(len(domain.predicates)))
    logging.info("{} Actions parsed".format(len(domain.actions)))
    logging.info("{} Objects parsed".format(len(problem.objects)))
    logging.info("{} Constants parsed".format(len(domain.constants)))
    return problem


def log(test_env, iteration, step_idx, total_rew):
    """
    Logs one step in a testing episode.
    :param test_env: Test env given a task.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    time.sleep(0.3)
    print(f"Planning task: {test_env.task.name}")
    print(f"Training Episodes: {iteration}")
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")


if __name__ == '__main__':

    usage = 'python3 main.py <DOMAIN> <PROBLEM>'
    parser = argparse.ArgumentParser(usage=usage, description='command line usage.')

    parser.add_argument('domain',  type=str, help='path to PDDL domain file')
    parser.add_argument('problem', type=str, help='path to PDDL problem file')
    
    if len(sys.argv) < 2:
        print('See the usage below:')
        parser.print_help()
        sys.exit(2)
    else:
        namespace_arguments = parser.parse_args()
        domain = namespace_arguments.domain
        problem = namespace_arguments.problem

    pro = _parse(domain, problem)
    task = grounding.ground(pro)
    facts = sorted(task.facts)
    dict_facts = {}
    i=0
    for fact in facts:
        dict_facts[fact]= i
        i+=1
    all_action_spaces = action_spaces_new(task)
    all_action_spaces = sorted(all_action_spaces)
    n_actions = len(all_action_spaces)
    ae = ActionEncoder()
    ae.fit(list_all_action=all_action_spaces)

    #Read training data
    csvfile = open( "dataset.csv", "r" )

    # sniff into 10KB of the file to check its dialect
    dialect = csv.Sniffer().sniff( csvfile.read( 10*1024 ) )
    csvfile.seek(0)

    # read csv file according to dialect
    reader = csv.reader( csvfile, dialect )

    # read header
    next(reader)
    items = [row for row in reader]
    csvfile.close()
    
    trainer = Trainer(lambda: Rlstm(len(all_action_spaces), len(facts)))
    network = trainer.step_model
    network.model.save('initial.hdf5')
    best_trainer = Trainer(lambda: Rlstm(len(all_action_spaces), len(facts)))
    #Comment when resuming training
    network_best = best_trainer.step_model

    #Uncomment to load saved models when resuming training
    #network.model.save("best_model.hdf5")
    #network_best.model = load_model("best_model.hdf5")
    #network.model = load_model("best_model.hdf5")
    
    global_list_training = deque(maxlen=100000)
    #Uncomment to load a saved training data
    #global_list_training.extend(pickle.load(open("global_list_training.p", "rb")))
    
    #Comment the next three lines when resuming training
    maxi = np.ones([3], dtype=np.float32)
    maxi[2]=0
    np.savetxt('myfile.csv', maxi, delimiter=',')
    
    #Uncomment these when resuming training from a saved state
    #maxi = np.loadtxt('myfile.csv', delimiter=',')
    #print(maxi)
    
    maxi1 = np.zeros([3], dtype=np.float32)

    
    def test_agent(iteration, test_env):
        """
        Test function to check the performance of the trained model
        :param iteration: number of steps
        :test_env: a test agent with rules and trained NN
        """
        total_rew = 0
        stacked_state, goal, reward, done, _ = test_env.reset()
        step_idx = 0
        while not done:
            log(test_env, iteration, step_idx, total_rew)
            p, v, _ = network.step(test_env.get_obs_for_state(stacked_state))
            print(v[0])
            possible_action, possible_action_keys = test_env.get_possible_actions(stacked_state.head)
            possible_action_ohe = test_env.ae.transform(possible_action_keys).sum(axis=0)
            p= p[0]*possible_action_ohe
            sum_policy_state = np.sum(p)
            if sum_policy_state > 0:
                ## normalize to sum 1
                p /= sum_policy_state
            else:
                print("All valid moves were masked, do workaround.")
                p += possible_action_ohe
                p /= np.sum(p)
            action_key = test_env.ae.inverse_transform([np.argmax(p)])[0]
            for oper in possible_action:
                if oper.name==action_key:
                    action = oper
            stacked_state, done, _ = test_env.step(action)
            step_idx+=1
            total_rew +=v[0][0]
        log(test_env, iteration, step_idx, total_rew)

    #Uncomment these when resuming training
    #value_losses = pickle.load(open("value_losses.p", "rb"))
    #policy_losses = pickle.load(open("pi_losses.p", "rb"))

    #Comment the two lines when resuming training
    value_losses=[]
    policy_losses=[]

    #Maximum length to take
    max_steps = 50

    #initial index for dataset
    p=0

    
    for i in range(0, 200):
        path = "test_prob/"
        t=0
        #Generate dataset. 5000 is the number of training data generated between each training cycle 
        while t<5000:
            
            list_training = []
            idx = p
            #idx = random.randint(0, len(items)-1)
            item = items[idx][0]
            cost = int(items[idx][1])
            
            print(item, cost)
            problem = path + item                
            pro = _parse(domain, problem)
            task = grounding.ground(pro)
            if task.goal_reached(task.initial_state):
                continue

            j=0
            while True:

                obs, pis, h ,total_reward, done_state = execute_episode(network,
                                                                    50,
                                                                    PlanningAgent(n_actions, task, ae, dict_facts, maxi,  max_simulation= min(cost+5, max_steps)))
                
                list_training.extend(obs)
                t+=len(obs)
                planFound = task.goal_reached(done_state.state.head)
                print(planFound)

                if planFound and done_state.depth<cost:
                    items.remove(items[idx])
                    items.insert(idx, [item, done_state.depth])
                    cost = done_state.depth
                    with open('dataset.csv', "w") as f:
                        writer = csv.writer(f)
                        fieldnames=['Problem', 'Cost']
                        writer.writerow(fieldnames)
                        writer.writerows(items)

                if j>0:
                    break
                j+=1  

            global_list_training.extend(list_training)
            pickle.dump(global_list_training, open("global_list_training.p","wb"))

            p+=1
            if p>=len(items):
                p=0
        print(len(global_list_training))
                
        X, action_proba, y, _ = parse_global_list_training(global_list_training, dict_facts)
        
        #change input into numpy array
        X = np.array(X)
        y = np.array(y)
        
        #Normalize value 
        maxi1[0] = max(y)
        y = y/maxi1[0]
        
        #change input into numpy array
        action_proba = np.array(action_proba)
        list_training = []

        #Train the NN
        hist = trainer.train(X, action_proba, y)

        #Save value and policy losses
        value_losses.extend(hist.history['v_loss'])
        policy_losses.extend(hist.history['pi_loss'])


        #Test the tained model 
        cnt=0
        win= np.zeros((2))
        path = "test_prob/"
        #Read training data
        csvfile = open( "dataset.csv", "r" )

        # sniff into 10KB of the file to check its dialect
        dialect = csv.Sniffer().sniff( csvfile.read( 10*1024 ) )
        csvfile.seek(0)

        # read csv file according to dialect
        reader = csv.reader( csvfile, dialect )

        # read header
        next(reader)
        items1 = [row for row in reader]
        csvfile.close()
        #items = os.listdir(path)
        
        while cnt < 11:
            #item = random.choice(items)
            idx = random.randint(0, len(items)-1)
            item = items1[idx][0]
            problem = path + item
            cost = int(items1[idx][1])
            print(problem)
                
            problem = _parse(domain, problem)
            task = grounding.ground(problem)
            goal = task.goals
            if task.goal_reached(task.initial_state):
                #print("Current state is goal state")
                #items.remove(items[idx])
                continue

            test_envb = PlanningAgent(n_actions, task, ae, dict_facts, maxi=maxi, max_simulation=100)
            test_envc = PlanningAgent(n_actions, task, ae, dict_facts, maxi=maxi1, max_simulation=100)

            _, _, reward_best,total_reward, done_state_best = execute_episode(network_best, 10, test_envb, test=True)
            print("===================================")
            _, _, reward_current,total_reward, done_state_current = execute_episode(network, 10, test_envc, test=True)
            if task.goal_reached(done_state_best.state.head)==False and task.goal_reached(done_state_current.state.head)==False:
            
                if reward_best<reward_current:
                    win[0]+=1
                    print("Best model found less expensive path")
                elif reward_best>reward_current:
                    win[1]+=1
                    print("Current model found less expensive path")
                
            elif task.goal_reached(done_state_best.state.head)== True and task.goal_reached(done_state_current.state.head)==True:
                if done_state_best.depth < done_state_current.depth:
                    win[0]+=1
                    print("Best model found a better solution")
                elif done_state_best.depth > done_state_current.depth:
                    win[1]+=1
                    print("Current model found a better solution")
                else:
                    print("same")
            elif task.goal_reached(done_state_best.state.head)==True and task.goal_reached(done_state_current.state.head)==False:
                win[0]+=1
                print("Best model found a solution")
            else:
                win[1]+=1
                print("Current model found a solution")
            cnt+=1
        print("score, ", win)
        if win[1] <= 4 + win[0]:
            print("Best model wins")
            # Replace the model of the trained with the best model.
            network.model = load_model("best_model.hdf5")
        else:
            #Replace the best model with the current model
            #Update the min & max values of value   
            maxi=maxi1
            network.model.save("best_model.hdf5")
            name = "best_model-"+str(i)+".hdf5"
            network.model.save(name)
            network_best.model = load_model('best_model.hdf5')
            
        print(i, max_steps)
        np.savetxt('myfile.csv', maxi, delimiter=',')

        
        pickle.dump(value_losses, open("value_losses.p","wb"))
        pickle.dump(policy_losses, open("pi_losses.p","wb"))

        
