#!/usr/bin/python
"""
Script Description:
-------------------
This script performs preprocessing operations for preparing data to be used with the GNN A3T (Graph Neural Network Attention with 
Aggregated Temporal Embedding) model. The preprocessing includes dataset reformation, unit conversion, handling missing values, and
 adjusting the frequency in the time series values.

Preprocessing Steps:
-------------------
1. Dataset Reformation:
   - Describe any steps taken to reformat the dataset for compatibility with the GNN A3T model.

2. Unit Conversion:
   - Convert units if necessary to ensure consistency and proper input for the model.

3. Handling Missing Values:
   - Implement strategies to handle missing values in the dataset, such as imputation or removal.

4. Frequency Adjustment:
   - Adjust the frequency of time series values to meet the desired temporal resolution of the GNN A3T model.

5. Additional Preprocessing Steps:
   - Mention any other specific preprocessing steps required for the GNN A3T model.

Note: Add detailed explanations and code comments throughout the script for better understanding of each preprocessing step.

Maintainer: Efterpi Paraskevoulakou <e.paraskevoulakou@unipi.gr>
Project: HE-CODECO
License: Apache 2.0
"""
##### import useful libraries #####
import pandas as pd
import numpy as np
import json
import os
import torch


### ENV_VARIABLES #####
network_topology_json = os.getenv("TOPOLOGY_PATH", "/Users/pepiparaskevoulakou/Desktop/Projects/HE-CODECO/dl_pipeline/dl_pipeline-1/data/topology.json")


def load_data(data):
    #data = pd.read_csv(data_path)
    data_selected = data[['time', 'node_name', 'cpu', 'memory']]
    data_selected = data_selected.iloc[:len(data_selected) - 4] #removing last 4 rows because 5th node's last row is missing


    selected_nodes = []
    node_names = data_selected['node_name'].unique()

    for i in range(len(node_names)):
        new_nodes = data_selected[data_selected['node_name'] == node_names[i]]
        new_nodes['memory'] = new_nodes['memory'].fillna(method='ffill') #forward fill null values
        new_nodes.reset_index(drop=True, inplace=True)
        selected_nodes.append(new_nodes)

    return selected_nodes, node_names

def minute_mean(metric1, metric2, nodes_list):
    nodes_per_min_mean = []
    for node in nodes_list:
        res1 = node[metric1].groupby(np.arange(len(node[metric1]))//20).mean() #there are 20 3-second intervals in a minute
        res2 = node[metric2].groupby(np.arange(len(node[metric2]))//20).mean()
        name = pd.Series(node['node_name'], index=range(len(res1)))
        node_per_min=pd.concat([res1,res2,name],axis=1)
        nodes_per_min_mean.append(node_per_min)
        
    return nodes_per_min_mean
    
def five_min_sample(nodes_list):
    nodes_five_min = []
    for node in nodes_list:
        res = node[::5]
        nodes_five_min.append(res)
        
    return nodes_five_min


def timesteps_config(nodes_list, sampled_data):
    time = pd.Series(np.zeros(sampled_data[0].shape[0])) #create pd.Series for timesteps (as many as the rows of each node)
    dataset_days = int(sampled_data[0].shape[0]/288)
    timesteps_per_day = 288 #288 five-minute intervals in each day
    for i in range(dataset_days):
        time[timesteps_per_day*i:timesteps_per_day*(i+1)] = time[timesteps_per_day*i:timesteps_per_day*(i+1)].index.to_series()/timesteps_per_day-i
        time[timesteps_per_day*dataset_days:] = time[timesteps_per_day*dataset_days:].index.to_series()/timesteps_per_day-dataset_days


    for i in range(len(nodes_list)):
        sampled_data[i].reset_index(drop=True, inplace=True)
        sampled_data[i]['time'] = time

    dataframes = sampled_data
    df2 = pd.concat(dataframes)
    df2['index'] = df2.index
    df2.sort_values(by=['index', 'node_name'], inplace=True)
    df2.reset_index(drop=True, inplace=True)
    data_selected = df2.drop(columns=['index'])

    sampled_selected_nodes = []
    node_names = data_selected['node_name'].unique()

    for i in range(len(node_names)):
        new_nodes = data_selected[data_selected['node_name'] == node_names[i]]
        sampled_selected_nodes.append(new_nodes)

    return sampled_selected_nodes

def metrics_arrays(sampled_selected_nodes, metric='memory'):

    nodes_time = sampled_selected_nodes[0]['time'].to_numpy()
    nodes_metrics = []
    for i in range(len(sampled_selected_nodes)):
        selected_node_metric = sampled_selected_nodes[i][metric].to_numpy()

        nodes_metrics.append(np.array([selected_node_metric, nodes_time]))

    X_nodes_input = np.array(nodes_metrics)
    return X_nodes_input

def read_topology(topology_path):

    # Opening JSON file
    f = open(topology_path)

    # returns JSON object as a dictionary
    data = json.load(f)
    
    # Iterating through the json list
    connections = []
    for i in data['connections']:
        connections.append(np.array(i))
    adjacency_matrix = np.array(connections)
    # Closing file
    f.close()

    return(adjacency_matrix)

def normalization(X_nodes_input, adjacency_matrix):
    n = X_nodes_input.shape[2]*0.7
    l = X_nodes_input[:,:,0:int(n)]

    means = np.mean(l, axis=2)
    std = np.std(l, axis=2)

    for i in range(X_nodes_input.shape[0]):
        X_nodes_input[i][0] = (X_nodes_input[i][0] - means[i][0]) / std[i][0]

    topology_matrix = torch.from_numpy(adjacency_matrix)
    nodes_metrics_input = torch.from_numpy(X_nodes_input)

    return nodes_metrics_input, topology_matrix, means, std


def GNN_operations_A3T(data):
    nodes_list, node_names = load_data(data)
    min_mean = minute_mean('cpu', 'memory', nodes_list)
    sampled_data = five_min_sample(min_mean)
    sampled_selected_nodes = timesteps_config(nodes_list, sampled_data)
    X_nodes_input=metrics_arrays(sampled_selected_nodes, "memory")
    nodes_metrics_input, topology_matrix, mean, std = normalization(X_nodes_input, read_topology(network_topology_json))

    return {"node_list":nodes_list, "node_names_": nodes_list, "min_mean": min_mean,"sampled_data": sampled_data,  
                            "sampled_selected_nodes": sampled_selected_nodes, "X_nodes_input": X_nodes_input, 
                            "nodes_metrics_input":nodes_metrics_input, "topology_matrix": topology_matrix, "mean":mean, "std": std}

