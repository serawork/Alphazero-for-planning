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
    
    #Create a trainer with with input and output dimensions defined by the largest problem
    trainer = Trainer(lambda: Rlstm(len(all_action_spaces), len(facts)))
    network = trainer.step_model
    network.model.save('initial.hdf5')
    best_trainer = Trainer(lambda: Rlstm(len(all_action_spaces), len(facts)))
    network_best = best_trainer.step_model
    #Comment when resumed from a saved state
    network.model.save("best_model.hdf5")

    #Uncomment to resume from a saved state
    #network_best.model = load_model("best_model.hdf5")
    #network.model = load_model("best_model.hdf5")


    #A collection to store generated data
    #Change the buffer size depending on the complexity of the model
    global_list_training = deque(maxlen=100000)

    #Uncomment when resuming from saved state
    #global_list_training.extend(pickle.load(open("global_list_training.p", "rb")))   

    
    def test_agent(iteration, test_env):
        total_rew = 0
        stacked_state, goal, reward, done, _ = test_env.reset()
        step_idx = 0
        search_start_time = time.process_time()
        while not done:
            log(test_env, iteration, step_idx, total_rew)
            p, v= network.step(test_env.get_obs_for_state(stacked_state))
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
            stacked_state, done, _, _ = test_env.step(action)
            step_idx+=1
            total_rew +=v[0][0]
        end = time.process_time()
        return step_idx, end-search_start_time
        #log(test_env, iteration, step_idx, total_rew)

    #Test
    """
    path = "test/"
    items = os.listdir(path)
    data = []
    for item in items:
        #item = random.choice(items)
        problem = path + item
        print(problem)
            
        problem = _parse(domain, problem)
        task = grounding.ground(problem)
        goal = task.goals
        if task.goal_reached(task.initial_state):
            print("Current state is goal state")
            items.remove(item)
            continue

        test_env = PlanningAgent(n_actions, task, ae, dict_facts, max_simulation=100)
        search_start_time = time.process_time()
        _, _, reward_best,N, total_reward, done_state = execute_episode(network_best, 20, test_env)
        end_time = time.process_time()
        data.append([item, done_state.depth, N, end_time - search_start_time])
  
        with open('gr-Planner(11-2-21).csv', "w") as f:
            writer = csv.writer(f)
            fieldnames=['Problem', 'Cost', 'Nodes', 'time']
            writer.writerow(fieldnames)
            writer.writerows(data)

    """
    #Stores the losses during training. Comment when resuming from saved state
    value_losses=[]
    policy_losses=[]
    #Uncomment when resuming from saved model
    #value_losses = pickle.load(open("value_losses.p", "rb"))
    #policy_losses = pickle.load(open("pi_losses.p", "rb"))
    
    #the maximum number of steps during self-play 
    t=50
    for i in range(0, 100):
        path = "test_prob/"
        p=0
        #randomly pick 30 problems from the dataset to generate training data. Adjust value depending on the frequency of training.
        while p < 30:
            
            list_training = []
            idx = random.randint(0, len(items)-1)
            item = items[idx][0]
            #item = item1[0]
            cost1 = int(items[idx][1])
            cost = int(items[idx][1])
            
            print(item, cost)
            problem = path + item                
            pro = _parse(domain, problem)
            task = grounding.ground(pro)
            if task.goal_reached(task.initial_state):
                continue

            j=0
            while True:
                obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                     60,
                                                                    PlanningAgent(n_actions, task, ae, dict_facts, max_simulation=min(cost, t)))

                
                #Reward is calculated as inverse of the heuristic value and is (0, 1].
                if cost>t:
                    v = t + h - cost
                    ret = 0.9**v

                    #update the cost if a value with a smaller one if found
                    if v < 0  and j==0:
                        done_state.depth=t+h
                        items.remove(items[idx])
                        items.insert(idx, [item, done_state.depth])
                        cost = done_state.depth
                        with open('dataset.csv', "w") as f:
                            writer = csv.writer(f)
                            fieldnames=['Problem', 'Cost']
                            writer.writerow(fieldnames)
                            writer.writerows(items)
                    for o in obs:
                        o.value = ret

                
                list_training.extend(obs)
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
                if j>2:
                    break
                j+=1
            p+=1

            #Update and save the training data
            global_list_training.extend(list_training)
            pickle.dump(global_list_training, open("global_list_training.p","wb"))
        print(len(global_list_training))
        X, action_proba, y, _ = parse_global_list_training(global_list_training, dict_facts)
        X = np.array(X)
        y = np.array(y)
        
        action_proba = np.array(action_proba)
        list_training = []

        hist = trainer.train(X, action_proba, y)
        value_losses.extend(hist.history['v_loss'])
        policy_losses.extend(hist.history['pi_loss'])
        
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

            test_env = PlanningAgent(n_actions, task, ae, dict_facts, max_simulation=config.MAX_LENGTH_PLANNER)

            _, _, reward_best, total_reward, done_state_best = execute_episode(network_best, 20, test_env)
            print("===================================")
            _, _, reward_current, total_reward, done_state_current = execute_episode(network, 20, test_env)

            #Compare the best model and the current trained model and update best model
            if task.goal_reached(done_state_best.state.head)==False and task.goal_reached(done_state_current.state.head)==False:
                """
                if reward_best>reward_current:
                    win[0]+=1
                    print("Best model found less expensive path")
                elif reward_best<reward_current:
                    win[1]+=1
                    print("Current model found less expensive path")
                """
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
            network.model = load_model("best_model.hdf5")
        else:
            network.model.save("best_model.hdf5")
            network_best.model = load_model('best_model.hdf5')
            name = "best_model-"+str(i)+".hdf5"
            network.model.save(name)
            network_best.model = load_model('best_model.hdf5')
        
        #Save the training losses
        pickle.dump(value_losses, open("value_losses.p","wb"))
        pickle.dump(policy_losses, open("pi_losses.p","wb"))
        
        

    
