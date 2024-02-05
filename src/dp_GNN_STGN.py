
#!/usr/bin/python
"""
Script Description:
-------------------
This script performs preprocessing operations for preparing data to be used with the ICOM's GNN STGN (Spatial-Temporal Graph Neural Network) model. 
The preprocessing includes dataset reformation, unit conversion, handling missing values, and adjusting the frequency in the time series values.

Preprocessing Steps:
-------------------
1. Dataset Reformation:
   - Describe any steps taken to reformat the dataset for compatibility with the GNN STGN model.

2. Unit Conversion:
   - Convert units if necessary to ensure consistency and proper input for the model.

3. Handling Missing Values:
   - Implement strategies to handle missing values in the dataset, such as imputation or removal.

4. Frequency Adjustment:
   - Adjust the frequency of time series values to meet the desired temporal resolution of the GNN STGN model.

5. Additional Preprocessing Steps:
   - Mention any other specific preprocessing steps required for the GNN STGN model.

Note: Add detailed explanations and code comments throughout the script for better understanding of each preprocessing step.

Maintainer: Efterpi Paraskevoulakou <e.paraskevoulakou@unipi.gr>
Project: HE-CODECO
License: Apache 2.0
"""
##### import useful libraries #####
import pandas as pd
import numpy as np
import os, json, time



### ENV_VARIABLES #####
network_topology_json = os.getenv("TOPOLOGY_PATH", "/Users/pepiparaskevoulakou/Desktop/Projects/HE-CODECO/dl_pipeline/dl_pipeline-1/data/topology.json")

########## ICOM necessary functions ##########

def load_and_transform(data):
    node_data_selected = data[['time', 'node_name', 'cpu', 'memory']]
    node_data_selected = node_data_selected.iloc[:len(node_data_selected) - 4] #removing last 4 rows because 5th node's last row is missing
    data_selected = node_data_selected
    worker1 = node_data_selected[node_data_selected['node_name']=='k8s-worker-1']
    node_data_selected[node_data_selected['node_name']=='k8s-worker-1'] = worker1.fillna(method='ffill')
    selected_nodes_2 = []
    node_names = node_data_selected['node_name'].unique()

    for i in range(len(node_names)): #creating a list of dataframes, one dataframe for each node
        new_nodes = data_selected[data_selected['node_name'] == node_names[i]]
        new_nodes.reset_index(drop=True, inplace=True)
        selected_nodes_2.append(new_nodes)

    #Update timestmps in the dataframe (for stable frequency)
    start_timestamp = 0
    time_increment = 3
    for i in range(len(selected_nodes_2)):
        selected_nodes_2[i]['time'] = start_timestamp + selected_nodes_2[i].index.to_series() * time_increment

    return selected_nodes_2


#function to produce the mean value of the metrics per minute
def minute_mean(metric1, metric2, nodes_list):
    nodes_per_min_mean = []
    for node in nodes_list:
        res1 = node[metric1].groupby(np.arange(len(node[metric1]))//20).mean()
        res2 = node[metric2].groupby(np.arange(len(node[metric2]))//20).mean()
        name = pd.Series(node['node_name'], index=range(len(res1)))

        node_per_min=pd.concat([res1,res2,name],axis=1)
        nodes_per_min_mean.append(node_per_min)
        
    return nodes_per_min_mean

#function to produce sampled metrics per 5 minutes out of the averaged values per minute
def five_min_sample(nodes_list):
    nodes_five_min = []
    for node in nodes_list:
        res = node[::5]
        res['time'] = res.index.to_series() * 0.2
        res = res.astype({"time": int})
        nodes_five_min.append(res)
        
    return nodes_five_min

def create_metrics_arrays(data):
    nodes_list = load_and_transform(data).copy()
    min_mean = minute_mean('cpu', 'memory', nodes_list)
    sampled_data = five_min_sample(min_mean) #returns the sampled data per 5 minutes

    #combine all nodes' dataframes into one 
    dataframes = sampled_data
    df2 = pd.concat(dataframes)
    df2.sort_values(by=['time', 'node_name'], inplace=True)
    df2.reset_index(drop=True, inplace=True)

    #separate nodes' dataframes after they have been sampled
    data_selected = df2
    selected_nodes_2 = []
    node_names = df2['node_name'].unique()

    for i in range(len(node_names)):
        new_nodes = data_selected[data_selected['node_name'] == node_names[i]]
        new_nodes.reset_index(drop=True, inplace=True)
        selected_nodes_2.append(new_nodes)

    #create separate array for each metric (cpy and memory)
    cpu_array = []
    ram_array = []
    for i in range(len(selected_nodes_2)):
        cpu_array.append(selected_nodes_2[i]['cpu'].to_numpy())
        ram_array.append(selected_nodes_2[i]['memory'].to_numpy())

    #transform into desired format nd.array of shape (number_of_instances, number_of_nodes)
    cpu_array = np.array(cpu_array)
    transpose_sepeds = np.transpose(cpu_array)
    cpu_array = transpose_sepeds

    ram_array = np.array(ram_array)
    transpose_sepeds = np.transpose(ram_array)
    ram_array = transpose_sepeds

    return cpu_array, ram_array, node_names


def preprocess(data_array: np.ndarray, train_size: float , val_size: float):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    num_train = int(num_time_steps * train_size)
    
    train_array = data_array[:num_train]
    mean, std = data_array[:num_train+num_val].mean(axis=0), data_array[:num_train+num_val].std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array, mean, std


def preprocess(data_array: np.ndarray, train_size: float , val_size: float):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    num_train = int(num_time_steps * train_size)
    
    train_array = data_array[:num_train]
    mean, std = data_array[:num_train+num_val].mean(axis=0), data_array[:num_train+num_val].std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array, mean, std


def read_topology(topology_path):
    '''
    Function that reads the net_topology and returns the adjacency_matrix
    '''
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



def GNN_operations_STGN(data):
    '''
    Overall dp_operations
    '''
    cpu_array, ram_array, node_names=  create_metrics_arrays(data)
    cpu_train_array, cpu_val_array, cpu_test_array, cpu_mean, cpu_std = preprocess(ram_array, train_size=0.8, val_size=0.2)
    ram_train_array, ram_val_array, ram_test_array, ram_mean, ram_std = preprocess(cpu_array, train_size=0.8, val_size=0.2)
    net_topology = read_topology(topology_path=network_topology_json)

    return {"node_names": node_names, "cpu_train_array": cpu_train_array, "cpu_val_array":cpu_val_array, "cpu_test_array": cpu_test_array, 
                            "cpu_mean":cpu_mean, "cpu_std": cpu_std, "ram_train_array":ram_train_array, "ram_val_array": ram_val_array,
                            "ram_test_array": ram_test_array, "ram_mean": ram_mean, "ram_std": ram_std, "net_topology": net_topology}
    