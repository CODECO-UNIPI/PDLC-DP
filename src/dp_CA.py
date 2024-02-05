#!/usr/bin/python
"""
Script Description:
-------------------
This script performs preprocessing operations for preparing data to be used to the CA (Context awareness) part. 
The preprocessing includes dataset reformation, unit conversion, normalization, providing the max values (per node).

-------------------

Note: Add detailed explanations and code comments throughout the script for better understanding of each preprocessing step.

Maintainer: Efterpi Paraskevoulakou <e.paraskevoulakou@unipi.gr>
Project: HE-CODECO
License: Apache 2.0
"""


##### import useful libraries #####
import pandas as pd
import numpy as np
import json, warnings
import os
from sklearn.preprocessing import MinMaxScaler


# Suppress all warnings
warnings.filterwarnings("ignore")

# ENV VARIABLES
filepath_to_csv = "/Users/pepiparaskevoulakou/Downloads/data_codeco.csv"

# features that are handled currently
features_to_be_handled = ['node_name', 'cpu', 'memory', 'link_energy', 'ibw','ebw','node_energy'] 



def read_csv(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=",")
        df.rename(columns={'timestamp': 'time'}, inplace=True)
        df.rename(columns={'mem': 'memory'}, inplace=True)
        return df
    except Exception as e:
        print(e)
        return None

def unit_conversion(df):
    '''
    Description:
    -------------------------
   Function that converts all metrics if required to formats that can be compared;
   e.g., metrics used in the computation of resilience or greeness are first normalized
   Note: All parameters in use. Currently: CPU, Mem, energy_node, energy_link, bw
    -------------------------
    Args:
        - Input: (<class> pandas.dataframe) dataframe
        - Output: (<class> pandas.dataframe)  dataframe with potantial unit conversions
    '''
    # Identify columns with numeric values as strings or objects
    numeric_str_columns = df.select_dtypes(include=['object']).columns

    # Convert numeric values as strings or objects to numeric types
    df[numeric_str_columns] = df[numeric_str_columns].apply(pd.to_numeric, errors='coerce')

    return df

def normalization(df):
    '''
    Description:
    -------------------------
   Function that calculates the normalized values of a dataset; 
   e.g., metrics used in the computation of resilience or greeness are first normalized
    -------------------------
    Args:
        - Input: (<class> pandas.dataframe) dataframe
        - Output: (<class> pandas.dataframe) normalized dataframe
    '''
    # Identify numerical columns for normalization
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Create a copy of the original DataFrame to avoid modifying the input
    normalized_df = df.copy()

    # Apply Min-Max scaling to numerical columns
    scaler = MinMaxScaler()
    normalized_df[numerical_columns] = scaler.fit_transform(normalized_df[numerical_columns])

    return normalized_df


def maximization(df):
    '''
    Description:
    -------------------------
   Functions that finds the max value from each feature
    -------------------------
    Args:
        - Input: (<class> pandas.dataframe) dataframe
        - Output: (<class> dict) max values for each of the features' values 
        (currently the features that are provided above in the "features_to_be_handled list")
    '''
    # Identify numerical columns excluding object or string type
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate and return the max value for each numerical column
    max_values = df[numerical_columns].max()

    return {"node_cpu_max": max_values[0], "node_memory_max": max_values[1], "node_link_energy_max": max_values[2], 
                "node_ibw_max": max_values[3], "node_ebw_max": max_values[4], "node_energy_max": max_values[5]}


def CA_operations_(data):
    data = data[features_to_be_handled]
    data = unit_conversion(data)
    max_values = maximization(data)
    normalized_data = normalization(data)

    return {"max_features_values": max_values, "normalized_data": normalized_data}



        

