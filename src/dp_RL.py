
#!/usr/bin/python
"""
Script Description:
-------------------
This script performs preprocessing operations for preparing data to be used with the RL (Reinforcement Learning) model. 
The steps are described below based on the reported partners' operations:

Preprocessing Steps:
-------------------
1. Dataset Reformation:
   - Describe any steps taken to reformat the dataset for compatibility with the I2CAT model.

2. Unit Conversion:
   - Convert the CPU and RAM unit to the correct units (in case of CPU to cores used and in case of memory to GB/MB).

3.  Creation of auxiliar data structures:
    - Finds the MAX_cpu and MAX_memory of the node (necessary  to be attached to  the k8s_node class)

4. Numerical pod ids:
    -Rename process to new pods to the next available int. This is done for existing and new pods. 
    A mapping between the original id and new numerical id.

Note: Add detailed explanations and code comments throughout the script for better understanding of each preprocessing step.

Maintainer: Efterpi Paraskevoulakou <e.paraskevoulakou@unipi.gr>
Project: HE-CODECO
License: Apache 2.0
"""

##### Import necessary libraries 
import os, json, time
from types import resolve_bases
import pandas as pd
import json
import os
import subprocess
import re
import requests
from cluster_config_rl import K8S_node_config


#USEFUL PATHS
filepath_to_csv = "/Users/pepiparaskevoulakou/Downloads/data_codeco.csv"
filepath_2 = "/Users/pepiparaskevoulakou/Downloads/data_i2cat.csv"

# Uuseful variables
k8s_node_name = os.getenv("NODE_NAME", "k8s-master")
#k8s_node_name = os.environ['NODE_NAME']


def read_csv(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=",")
        df.rename(columns={'timestamp': 'time'}, inplace=True)
        df.rename(columns={'mem': 'memory'}, inplace=True)
        df.rename(columns={'ram': 'memory'}, inplace=True)
        return df
    except Exception as e:
        print(e)
        return None


def convert_and_remove_suffix(value_with_suffix):
    '''
    function to detect the suffix n (used to cpu metric)
     '''
    suffix = value_with_suffix[-1]
    value = int(value_with_suffix[:-1])
    
    if suffix == 'n':
        return value / 1000000  # Convert nanocores to cores
    else:
        return value

def convert_and_remove_ki(value_with_ki):
    '''
    function to detect the suffix Ki (used to memory metric)
     '''
    return int(value_with_ki[:-2])


def unit_conversion(data):
    '''
    Unit conversion if the dataset is not in the correct format
    '''
    data['cpu'] = data['cpu'].apply(convert_and_remove_suffix)
    data['memory'] = data['memory'].apply(convert_and_remove_ki)
    return data


def get_node_capacity(node_name):
    '''
    This function that acquires the cpu and memory capacity for the specific node
    Args:
        Input:
            - node_name: (str) The name of the node that the script is executed
    Output:
        Returns the cpu and memory capacity 
    '''
    try:
        # Run kubectl describe node command and capture output
        result = subprocess.run(['kubectl', 'describe', 'node', node_name], capture_output=True, text=True, check=True)

        # Parse the output to extract capacity information
        output = result.stdout

        # Extract CPU and memory capacity using regular expressions
        cpu_capacity_match = re.search(r"cpu\s*:\s*([0-9]+)m", output)
        memory_capacity_match = re.search(r"memory\s*:\s*([0-9]+)Ki", output)

        if cpu_capacity_match and memory_capacity_match:
            cpu_capacity = int(cpu_capacity_match.group(1))
            memory_capacity = int(memory_capacity_match.group(1))
            return cpu_capacity, memory_capacity
        else:
            print("Unable to extract capacity information.")
            return None, None

    except subprocess.CalledProcessError as e:
        print(f"Error running kubectl describe node: {e}")
        return None, None



def get_running_pods_within_node(node_name):
    '''
    Function that counts the avalable pods within a node
    Note: Additional configs must be done to give access to pod in Node and pod info (RBAC privilleges)
    '''
    running_pods = []

    # Check if the NODE_NAME environment variable is set
    if 'NODE_NAME' in os.environ:
        print(f"Pod is running on node: {node_name}")

        # Use the Downward API to get the pod's namespace
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as namespace_file:
            namespace = namespace_file.read().strip()

        print(f"Pod is in namespace: {namespace}")

        # Query the Kubernetes API to get the list of pods running on the node
        api_url = f"https://kubernetes.default.svc/api/v1/namespaces/{namespace}/pods?fieldSelector=spec.nodeName={node_name}"
        
        try:
        
            response = requests.get(api_url, headers={"Authorization": "Bearer " + open("/var/run/secrets/kubernetes.io/serviceaccount/token").read()})

            if response.status_code == 200:
                pods = response.json().get('items', [])
                print("Running pods on the node:")
                for pod in pods:
                    pod_name = pod['metadata']['name']
                    print(pod_name)
                    running_pods.append(pod_name)

                return running_pods

            else:
                print(f"Failed to retrieve pod list. Status code: {response.status_code}")
                return None
        
        except Exception as e:
            print(e)

            return None

    else:
        print("NODE_NAME environment variable not found. Are you running this script within a Kubernetes pod?")
        return None
        


def map_pods_to_numbers(list_of_pods):
    '''
    Mapping pods based on index
    '''
    if list_of_pods == None:
        return []
    else:
        for running_pod in list_of_pods:
            if running_pod not in list_of_pods:
                K8S_node_config.list_with_pods.append(running_pod)
            else:
                pass
        
        pod_mapping_list = [index for index, _ in enumerate(K8S_node_config.list_with_pods)]
    
    return pod_mapping_list

def check_df_format(data, node_name):
    '''
    Additional functionality to check the format of the dataset, 
    We found that i2cat is working with kube-state-metrics dataset, thus we handle it 
    '''
    data_cpu_column_type, data_memory_column_type = data['cpu'].dtype, data['memory'].dtype
    if data_cpu_column_type in [str, object] and data_memory_column_type in [str, object]:
        data = data[data['type'] == 'node']
        data = data[data['name'] == f'{node_name}']
        data  =  unit_conversion(data)
        data = data[['time', 'cpu', 'memory']]

        return data

    else:
        data = data[['time', 'cpu', 'memory']]

        return data
    

def RL_operations_(data):
    '''
    Function that executes all the mandatory operations for the RL model
    '''
    node_name = k8s_node_name
    data = check_df_format(data, node_name)
    cpu_capac, memory_capac = get_node_capacity(k8s_node_name)
    list_of_pods = get_running_pods_within_node(k8s_node_name)
    if list_of_pods == None:
        #map_pods_to_numbers = map_pods_to_numbers(get_running_pods_within_node(k8s_node_name))
        return{"node_name": node_name, "df_coverted": data, "node_cpu_capacity": cpu_capac, "node_memory_capacity": memory_capac, "list_with_mapped_pods": None }
    else:
        indexed_pods = map_pods_to_numbers(list_of_pods)
        return{"node_name": node_name, "df_coverted": data, "node_cpu_capacity": cpu_capac, "node_memory_capacity": memory_capac, "list_with_mapped_pods": indexed_pods }







