import cupy as cp
import numpy as np
import pandas as pd
import scipy.io as sio

import inspect
import concurrent.futures

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

###### utils ######
def add_key_value_to_mat(input_file, output_file, new_key, new_value):
    """
    Read a .mat file, add a new key-value pair, and save it as a new .mat file.

    Parameters:
    input_file (str): The path to the input .mat file.
    output_file (str): The path to the output .mat file.
    new_key (str): The name of the key to be added (string).
    new_value (any type): The value to be added, which can be a numpy array, list, or scalar.

    Returns:
    None
    """
    # 1. Read the existing .mat file
    mat_data = sio.loadmat(input_file)

    # 2. Remove automatically generated metadata keys from MATLAB (e.g., '__header__', '__version__', '__globals__')
    mat_data = {key: value for key, value in mat_data.items() if not key.startswith('__')}

    # 3. Add the new key-value pair
    mat_data[new_key] = new_value

    # 4. Save the modified dictionary to a new .mat file
    sio.savemat(output_file, mat_data)
    print(f"Successfully added key '{new_key}' to the .mat file and saved it as {output_file}")

def load_dataset(dataset_name, root_dir="data", embedding_type='llm'):
    dataset_path = Path(root_dir) / dataset_name
    dataset_file = dataset_path / f"{dataset_name}.mat"
    drug_embedding_file = dataset_path / f"{dataset_name}_drug_embedding_{embedding_type}.csv"
    disease_embedding_file = dataset_path / f"{dataset_name}_disease_embedding_{embedding_type}.csv"
    
    dataset = read_mat_file(dataset_file)
    drug_embeddings = pd.read_csv(drug_embedding_file).values
    disease_embeddings = pd.read_csv(disease_embedding_file).values

    # Extract the required columns as arrays
    didr = dataset['didr']  # Drug-disease association matrix
    drug_ChemS = dataset['drug_ChemS']  # Drug chemical similarity matrix
    drug_AtcS = dataset['drug_AtcS']  # Drug ATC (Anatomical Therapeutic Chemical) similarity matrix
    drug_SideS = dataset['drug_SideS']  # Drug side effect similarity matrix
    drug_DDIS = dataset['drug_DDIS']  # Drug-drug interaction similarity matrix
    drug_TargetS = dataset['drug_TargetS']  # Drug target similarity matrix
    drug_GepS = dataset['drug_GepS']  # Drug gene expression similarity matrix
    drug_KgS = dataset['drug_KgS']  # Drug gene expression similarity matrix
    drug_LlmS = dataset['drug_LlmS']  # Drug gene expression similarity matrix
    disease_PhS = dataset['disease_PhS']  # Disease phenotype similarity matrix
    disease_DoS = dataset['disease_DoS']  # Disease ontology similarity matrix
    disease_LlmS = dataset['disease_LlmS']  # Disease ontology similarity matrix

    # -five indicates that previous studies used five drug similarity matrices, excluding the drug perturbation gene expression similarity matrix
    # Wrr and Wdd matrices (in list form)
    Wrr_five = [drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS]  # Five drug similarity matrices used in previous studies
    Wrr_six_geps = Wrr_five + [drug_GepS]
    Wrr_six_kgs = Wrr_five + [drug_KgS]
    Wrr_six_llms = Wrr_five + [drug_LlmS]
    Wrr_seven_llms_kgs = Wrr_five + [drug_LlmS, drug_KgS]
    Wrr_seven_llms_geps = Wrr_five + [drug_LlmS, drug_GepS]
    Wrr_seven_kgs_geps = Wrr_five + [drug_KgS, drug_GepS]
    Wrr_eight = Wrr_five + [drug_GepS, drug_KgS, drug_LlmS]  # Drug similarity matrices
    Wdd_two = [disease_PhS, disease_DoS]
    Wdd_three = Wdd_two + [disease_LlmS]  # Disease similarity matrices

    # Combine drug and disease similarity matrices into 3D arrays (dimensions: n x n x 6)
    Trr_eight = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS, drug_GepS, drug_KgS, drug_LlmS), axis=2)  # Six drug similarity matrices (3D array)
    Trr_seven_llms_kgs = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS, drug_LlmS, drug_KgS), axis=2)
    Trr_seven_llms_geps = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS, drug_LlmS, drug_GepS), axis=2)
    Trr_seven_kgs_geps = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS, drug_KgS, drug_GepS), axis=2)
    Trr_six_geps = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS, drug_GepS), axis=2)
    Trr_six_kgs = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS, drug_KgS), axis=2)
    Trr_six_llms = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS, drug_LlmS), axis=2)
    Trr_five = np.stack((drug_ChemS, drug_AtcS, drug_SideS, drug_DDIS, drug_TargetS), axis=2)  # Five drug similarity matrices (3D array)
    Tdd_three = np.stack((disease_PhS, disease_DoS, disease_LlmS), axis=2)  # Two disease similarity matrices (3D array)
    Tdd_two = np.stack((disease_PhS, disease_DoS), axis=2)

    # Drug-disease association matrix
    Wrd = didr.T  # Transpose to align dimensions as necessary for matrix operations

    # Drug name and disease name
    drug_name = dataset['Wrname']
    disease_name = dataset['Wdname']

    return (drug_name, disease_name, 
            Wrd, 
            Wrr_eight, 
            Wrr_seven_llms_kgs, Wrr_seven_llms_geps, Wrr_seven_kgs_geps, 
            Wrr_six_geps, Wrr_six_kgs, Wrr_six_llms, 
            Wrr_five, 
            Wdd_three, 
            Wdd_two, 
            Trr_eight, 
            Trr_seven_llms_kgs, Trr_seven_llms_geps, Trr_seven_kgs_geps, 
            Trr_six_geps, Trr_six_kgs, Trr_six_llms, Trr_five, 
            Tdd_three, 
            Tdd_two, 
            drug_embeddings, disease_embeddings)

def wrap_for_algorithm_func(algorithm_func, Wrd, Wrr_list, Wdd_list, Trr, Tdd, **kwargs):
    """
    Wrapper function to call the provided algorithm function with the appropriate parameters.
    
    Args:
    - algorithm_func (function): The algorithm function to be called.
    - Wrd (np.ndarray): The known drug-disease association matrix.
    - Wrr_list (list of np.ndarray): List of drug similarity matrices.
    - Trr (np.ndarray): A single drug similarity matrix (if the algorithm requires it).
    - Wdd_list (list of np.ndarray): List of disease similarity matrices.
    - Tdd (np.ndarray): A single disease similarity matrix (if the algorithm requires it).
    - kwargs (dict): Additional keyword arguments for the algorithm function.

    Returns:
    - result (Any): The result of the algorithm function.
    - algorithm_name (str): The name of the algorithm function.
    """

    # Get the signature (parameters) of the algorithm_func
    func_signature = inspect.signature(algorithm_func)
    func_params = func_signature.parameters

    # Check for required parameters and add them to kwargs
    if 'Wrr_list' in func_params:
        kwargs['Wrr_list'] = Wrr_list
    elif 'Trr' in func_params:
        kwargs['Trr'] = Trr

    if 'Wdd_list' in func_params:
        kwargs['Wdd_list'] = Wdd_list
    elif 'Tdd' in func_params:
        kwargs['Tdd'] = Tdd

    # Add Wrd to kwargs
    kwargs['Wrd'] = Wrd

    # Call the algorithm function with the gathered parameters
    return algorithm_func(**kwargs)

def normalize_matrix(matrix):
    scaler = MinMaxScaler()
    return scaler.fit_transform(matrix)

def read_mat_file(file_path):
    """
    Read a MAT file and return all the variables.
    
    Parameters:
    - file_path: str, the path of the MAT file
    
    Returns:
    - data_dict: dict, a dictionary containing all the variables in the MAT file
    """
    # Load the MAT file using scipy.io
    mat_data = sio.loadmat(file_path)
    
    # Remove metadata from the MAT file (keys starting with '__')
    data_dict = {key: value for key, value in mat_data.items() if not key.startswith('__')}

    for key, value in data_dict.items():
        if not key in ['Wdname', 'Wrname']:
            data_dict[key] = normalize_matrix(value)
        else:
            # Convert arrays to scalars if applicable
            data_dict[key] = np.array([x[0][0] if isinstance(x, np.ndarray) and len(x) > 0 else x for x in data_dict[key]])
    
    return data_dict

def extract_and_merge_datasets(dataset_files):
    """
    Read all dataset files and merge the eight matrices from the four datasets.
    
    Parameters:
    - dataset_files: list, containing paths of four .mat files
    
    Returns:
    - merged_data: dict, containing the merged eight matrices
    """
    drug_sim_matrices = []  # List to store drug similarity matrices
    disease_sim_matrices = []  # List to store disease similarity matrices
    all_Wdname = []  # List to store disease IDs
    all_Wrname = []  # List to store drug IDs

    for file_path in dataset_files:
        # Read the .mat file (assume `read_mat_file` is defined elsewhere)
        data = read_mat_file(file_path)
        
        # Extract drug and disease similarity matrices
        drug_sim_matrices.append([
            data['drug_AtcS'], data['drug_ChemS'], data['drug_DDIS'], 
            data['drug_GepS'], data['drug_SideS'], data['drug_TargetS']
        ])
        disease_sim_matrices.append([data['disease_DoS'], data['disease_PhS']])
        
        # Extract drug and disease IDs
        all_Wdname.append(data['Wdname'].flatten())  # Disease IDs
        all_Wrname.append(data['Wrname'].flatten())  # Drug IDs

    # Combine all disease and drug IDs
    common_Wdname = np.unique(np.concatenate(all_Wdname))
    common_Wrname = np.unique(np.concatenate(all_Wrname))

    # Build a mapping table for disease and drug IDs
    disease_id_map = {id_: i for i, id_ in enumerate(common_Wdname)}
    drug_id_map = {id_: i for i, id_ in enumerate(common_Wrname)}

    # Initialize merged matrices
    merged_drug_sim_matrices = [np.zeros((len(common_Wrname), len(common_Wrname))) for _ in range(6)]
    merged_disease_sim_matrices = [np.zeros((len(common_Wdname), len(common_Wdname))) for _ in range(2)]

    # Initialize a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Merge each matrix after Min-Max normalization
    for i in range(len(dataset_files)):
        # Merge drug similarity matrices
        for j in range(6):
            drug_matrix = drug_sim_matrices[i][j]
            # Flatten and normalize the matrix using MinMaxScaler
            scaled_drug_matrix = scaler.fit_transform(drug_matrix)
            wr_indices = [drug_id_map[id_] for id_ in all_Wrname[i]]
            merged_drug_sim_matrices[j][np.ix_(wr_indices, wr_indices)] += scaled_drug_matrix

        # Merge disease similarity matrices
        for j in range(2):
            disease_matrix = disease_sim_matrices[i][j]
            # Flatten and normalize the matrix using MinMaxScaler
            scaled_disease_matrix = scaler.fit_transform(disease_matrix)
            wd_indices = [disease_id_map[id_] for id_ in all_Wdname[i]]
            merged_disease_sim_matrices[j][np.ix_(wd_indices, wd_indices)] += scaled_disease_matrix

    # Normalize merged drug similarity matrices
    for j in range(6):
        merged_drug_sim_matrices[j] = scaler.fit_transform(merged_drug_sim_matrices[j])

    # Normalize merged disease similarity matrices
    for j in range(2):
        merged_disease_sim_matrices[j] = scaler.fit_transform(merged_disease_sim_matrices[j])

    # Return the merged data
    merged_data = {
        'drug_AtcS': merged_drug_sim_matrices[0],
        'drug_ChemS': merged_drug_sim_matrices[1],
        'drug_DDIS': merged_drug_sim_matrices[2],
        'drug_GepS': merged_drug_sim_matrices[3],
        'drug_SideS': merged_drug_sim_matrices[4],
        'drug_TargetS': merged_drug_sim_matrices[5],
        'disease_DoS': merged_disease_sim_matrices[0],
        'disease_PhS': merged_disease_sim_matrices[1],
        'Wdname': common_Wdname,
        'Wrname': common_Wrname
    }
    
    return merged_data

def precompute_similarity_features(Trr, Tdd, num_drugs, num_diseases, use_gpu=True):
    print("Precomputing drug and disease similarity features...")

    # Decide whether to use cupy or numpy
    xp = cp if use_gpu else np

    # Convert to cupy arrays if input is numpy and use_gpu is True
    if use_gpu:
        if isinstance(Trr, np.ndarray):
            Trr = cp.asarray(Trr)
        if isinstance(Tdd, np.ndarray):
            Tdd = cp.asarray(Tdd)

    # Check dimensions of Trr and Tdd
    if len(Trr.shape) != 3 or len(Tdd.shape) != 3:
        raise ValueError("Trr and Tdd must be 3-dimensional arrays.")

    # Precompute drug similarity features for Trr
    num_drugs, _, num_features_drug = Trr.shape
    drug_sim_features = xp.zeros((num_drugs, 4 * num_features_drug))
    
    means = xp.mean(Trr, axis=1)
    maxs = xp.max(Trr, axis=1)
    mins = xp.min(Trr, axis=1)
    stds = xp.std(Trr, axis=1)
    
    # Concatenate all features along the last dimension
    drug_sim_features = xp.concatenate([means, maxs, mins, stds], axis=1)

    # Precompute disease similarity features for Tdd
    num_diseases, _, num_features_disease = Tdd.shape
    disease_sim_features = xp.zeros((num_diseases, 4 * num_features_disease))
    
    means = xp.mean(Tdd, axis=1)
    maxs = xp.max(Tdd, axis=1)
    mins = xp.min(Tdd, axis=1)
    stds = xp.std(Tdd, axis=1)
    
    # Concatenate all features along the last dimension
    disease_sim_features = xp.concatenate([means, maxs, mins, stds], axis=1)

    # Convert back to numpy before returning if use_gpu is True
    if use_gpu:
        return cp.asnumpy(drug_sim_features), cp.asnumpy(disease_sim_features)
    else:
        return drug_sim_features, disease_sim_features

def extract_embeddings(dataset_name, merged_data, drug_embeddings, disease_embeddings, type):
    """
    Extract a subset of drug and disease embeddings from a large embedding file based on a smaller dataset.

    Args:
        dataset_name (str): The name of the dataset to process.
        merged_data (dict): Dictionary containing the merged dataset with full drug and disease IDs.
        drug_embeddings (np.ndarray): Full matrix of drug embeddings.
        disease_embeddings (np.ndarray): Full matrix of disease embeddings.

    Returns:
        None. Saves the extracted drug and disease embeddings as CSV files.

    Example:
        extract_embeddings("example_dataset", merged_data, drug_embeddings, disease_embeddings)

    Files Saved:
        - data/example_dataset/example_dataset_drug_embedding.csv
        - data/example_dataset/example_dataset_disease_embedding.csv
    """
    # Read the target dataset from a .mat file
    dataset = read_mat_file(f'data/{dataset_name}/{dataset_name}.mat')

    # Retrieve the drug and disease ID lists from the merged data
    merged_drug_ids = merged_data['Wrname']
    merged_disease_ids = merged_data['Wdname']

    # Retrieve the drug and disease ID lists from the target dataset
    drug_ids = dataset['Wrname']
    disease_ids = dataset['Wdname']

    # Create ID-to-index mappings for drugs and diseases
    drug_id_to_index = {id_: idx for idx, id_ in enumerate(merged_drug_ids)}
    disease_id_to_index = {id_: idx for idx, id_ in enumerate(merged_disease_ids)}

    # Get the indices of the target dataset's drugs in the full embeddings
    drug_indices = [drug_id_to_index[id_] for id_ in drug_ids if id_ in drug_id_to_index]

    # Get the indices of the target dataset's diseases in the full embeddings
    disease_indices = [disease_id_to_index[id_] for id_ in disease_ids if id_ in disease_id_to_index]

    # Extract the embeddings for the selected drugs
    small_drug_embeddings = drug_embeddings[drug_indices, :]
    
    # Extract the embeddings for the selected diseases
    small_disease_embeddings = disease_embeddings[disease_indices, :]

    # Print the shape of the extracted embeddings
    print(f"Extracted drug embeddings shape: {small_drug_embeddings.shape}")
    print(f"Extracted disease embeddings shape: {small_disease_embeddings.shape}")

    # Save the extracted drug embeddings to a CSV file
    drug_file_path = f'data/{dataset_name}/{dataset_name}_drug_embedding_{type}.csv'
    pd.DataFrame(small_drug_embeddings).to_csv(drug_file_path, index=False)
    
    # Save the extracted disease embeddings to a CSV file
    disease_file_path = f'data/{dataset_name}/{dataset_name}_disease_embedding_{type}.csv'
    pd.DataFrame(small_disease_embeddings).to_csv(disease_file_path, index=False)

    # Final message indicating the file locations
    print(f"Files have been saved at: {drug_file_path} and {disease_file_path}")

def extract_features(idx, num_diseases, drug_sim_features, disease_sim_features, drug_embeddings=None, disease_embeddings=None):
    i, j = divmod(idx, num_diseases)
    features = []

    # Add precomputed drug similarity features (must exist)
    features.extend(drug_sim_features[i])

    # Add drug embeddings to features if not None
    if drug_embeddings is not None:
        features.extend(drug_embeddings[i])

    # Add precomputed disease similarity features (must exist)
    features.extend(disease_sim_features[j])

    # Add disease embeddings to features if not None
    if disease_embeddings is not None:
        features.extend(disease_embeddings[j])

    return np.array(features)

def optimized_feature_extraction(indices, num_diseases, drug_sim_features, disease_sim_features, drug_embeddings=None, disease_embeddings=None):
    # Parallel extraction of features for the training set
    with concurrent.futures.ThreadPoolExecutor() as executor:
        features = list(executor.map(
            lambda idx: extract_features(
                idx, num_diseases, drug_sim_features, disease_sim_features, drug_embeddings, disease_embeddings), 
            indices))
    
    return features
