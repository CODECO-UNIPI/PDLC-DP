from pandas.core.frame import DataFrame
from data_generation import create_data
import pandas as pd
import numpy as np
import os
import warnings
import threading
from GNN_STGN import GNN_operations_STGN
from GNN_A3T import GNN_operations_A3T
from dp_RL import RL_operations_, k8s_node_name



# Suppress all warnings
warnings.filterwarnings("ignore")

# ENV VARIABLES
filepath_to_csv = "/Users/pepiparaskevoulakou/Downloads/data_codeco.csv"

#useful variables that may change their values across the time
df = pd.DataFrame()


def generate_df(data_dict:dict):
    for key_feature, feature_values in data_dict.items():
        df[key_feature] = feature_values
    
    df.rename(columns={'timestamp': 'time'}, inplace=True)
    df.rename(columns={'mem': 'memory'}, inplace=True)

    return df


def read_csv(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=",")
        df.rename(columns={'timestamp': 'time'}, inplace=True)
        df.rename(columns={'mem': 'memory'}, inplace=True)
        return df
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    data = read_csv(filepath_to_csv)
    if isinstance(data, pd.core.frame.DataFrame):

        node_names, cpu_train_array, cpu_val_array, cpu_test_array, cpu_mean, cpu_std, ram_train_array, ram_val_array, ram_test_array, ram_mean, ram_std, net_topology= (GNN_operations_STGN(data))
        STGN_dict_values = {"node_names": node_names, "cpu_train_array": cpu_train_array, "cpu_val_array":cpu_val_array, "cpu_test_array": cpu_test_array, 
                            "cpu_mean":cpu_mean, "cpu_std": cpu_std, "ram_train_array":ram_train_array, "ram_val_array": ram_val_array,
                            "ram_test_array": ram_test_array, "ram_mean": ram_mean, "ram_std": ram_std, "net_topology": net_topology}

        nodes_list, node_names_, min_mean, sampled_data, sampled_selected_nodes, X_nodes_input, nodes_metrics_input, topology_matrix, mean, std = (GNN_operations_A3T(data))
        A3T_dict_values = {"node_list":nodes_list, "node_names_": node_names_, "min_mean": min_mean,"sampled_data": sampled_data,  
                            "sampled_selected_nodes": sampled_selected_nodes, "X_nodes_input": X_nodes_input, 
                            "nodes_metrics_input":nodes_metrics_input, "topology_matrix": topology_matrix, "mean":mean, "std": std}

        print(STGN_dict_values)

        print(RL_operations_(data))                    




    else:
        print("Cannot perform pp operations")
