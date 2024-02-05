from datetime import time
from pandas.core.frame import DataFrame
#from data_generation import create_data
import pandas as pd
import numpy as np
import os, time
import warnings
import threading
from dp_GNN_STGN import GNN_operations_STGN
from dp_GNN_A3T import GNN_operations_A3T
from dp_RL import RL_operations_, k8s_node_name
from dp_CA import CA_operations_



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

        print("\033[94m" + "[INFO]: GNN_operations_STGN are prepared" + "\033[0m")
        print(GNN_operations_STGN(data)) # Pre-processed data for ICOM's STGN model
        time.sleep(2)

        print("\033[94m" + "[INFO]: GNN_operations_A3T are prepared" + "\033[0m")
        print(GNN_operations_A3T(data)) # Pre-processed data for ICOM's A3T model
        time.sleep(2)

        print("\033[94m" + "[INFO]: RL_operations_ are prepared" + "\033[0m")
        print(RL_operations_(data))  # Pre-processed data for i2cat's RL model
        time.sleep(2)

        print("\033[94m" + "[INFO]: CA_operations_ are prepared" + "\033[0m")
        print(CA_operations_(data)) # Pre-processed data for FOR CA subcomponent
        time.sleep(2)
    else:
        pass



