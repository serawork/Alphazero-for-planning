
from collections import defaultdict
import itertools
import numpy as np
from copy import deepcopy

from pyperplan import grounding
from pyperplan.task import Operator, Task

def get_feature_instances(problem):
    domain = problem.domain
    actions = domain.actions.values()
    predicates = domain.predicates.values()
    objects = problem.objects
    statics = grounding._get_statics(predicates, actions)

    obj = {}
    feature = {}
    edges={}
    for pred in predicates:
        if len(pred.signature)==1 and pred.name not in statics:
            obj[pred.name] = 0
        elif len(pred.signature)==2 and pred.name not in statics:
            edges[pred.name] = 0

    A = {}
    for objc in objects:
        feature[objc]=deepcopy(obj)
        for sobj in objects:
            name = objc + " " + sobj
            A[name]=deepcopy(edges)
    return feature, A

def get_state_repr(state, feature, E):
    sta = []
    for i in state:
        st = i[i.find('(')+1:i.find(')')].split(" ")
        sta.append(st)
    for state in sta:
        if len(state)==2:
            feature[state[1]][state[0]]=1
        if len(state) > 2:
            E[state[1] + " " + state[2]][state[0]]=1

    feat = sorted(feature.items())
    features = []
    edges = []
    for f in feat:
        b = sorted(f[1].items())
        f1=[]
        for f in b:
            f1.append(f[1])
        features.append(f1)

    feat = sorted(E.items())
    for f in feat:
        b = sorted(f[1].items())
        f1=[]
        for f in b:
            f1.append(f[1])
        edges.append(f1)
    features = np.asarray(features)
    edges = np.asarray(edges)

    return features, edges

def get_bow(state, dict_facts):
    staterep =[]
    sta = [0]*len(dict_facts)
    for i in state:
        #staterep.append(dict_facts[i])
        sta[dict_facts[i]] = 1
    
    return sta
        
    
