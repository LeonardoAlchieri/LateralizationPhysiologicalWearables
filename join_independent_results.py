from pandas import HDFStore, DataFrame
from glob import glob
from tqdm.auto import tqdm

def main():
    
    current_folder_avgs = "./data.nosync/mwc2022/results/nested_avg_smaller/"
    current_folder_all = "./data.nosync/mwc2022/results/nested_all_smaller/"
    
    avgs_files = glob(current_folder_avgs + "*.h5")
    all_files = glob(current_folder_all + "*.h5")

    averaged_results_cv: dict[str, DataFrame] = dict()
    for file in tqdm(avgs_files, desc='Loading file progress'):
        with HDFStore(file, mode='r') as store:
            # Iterate through the keys (group names) in the HDF5 file
            for key in store.keys():
                # Load each DataFrame and store it in the dictionary
                averaged_results_cv[key] = store[key]
            
    all_results_cv: dict[str, DataFrame] = dict()
    for file in tqdm(all_files, desc='Loading file progress'):
        with HDFStore(file) as store:
            # Iterate through the keys (group names) in the HDF5 file
            for key in store.keys():
                # TODO: fix the fucking keys, cause you are overwriting them!
                # Load each DataFrame and store it in the dictionary
                all_results_cv[key] = store[key]
    
    
    path_to_save_data_avgs: str = "./data.nosync/mwc2022/results/nested_avg_small.h5"
    path_to_save_data_all: str = "./data.nosync/mwc2022/results/nested_all_small.h5"
    
    
    with HDFStore(path_to_save_data_avgs) as store:
            # Save each DataFrame in the dictionary to the HDF5 file
            for key, value in averaged_results_cv.items():
                store.put(key, value)

        # Create an HDF5 file
    with HDFStore(path_to_save_data_all) as store:
        # Save each DataFrame in the dictionary to the HDF5 file
        for key, value in all_results_cv.items():
            store.put(key, value)
                    
if __name__ == "__main__":
    main()