import re
import argparse
from api.dataset import DatasetManager

class DatasetParser:
    def parse_columns(columns_info):
        if columns_info == None:
            return None
        
        return [ i.split('-') for i in columns_info.split(';') ]
        
    def extract_info(info):
        key_m = re.match(r"^[^\[(]*", info)
        split_m = re.search(r"\[(.+?)\]", info)
        columns_m = re.search(r"\((.+?)\)", info)
        
        key = key_m.group(0) if key_m else None
        split = split_m.group(1) if split_m else None
        columns = columns_m.group(1) if columns_m else None
        
        return key, split, DatasetParser.parse_columns(columns)

def parse_csv_source(key_train, key_test, columns):
    params = {"key_train": key_train, "make_split": True}
    if key_test:
        params["key_test"] = key_test
    if columns:
        params["columns"] = columns
    
    return params
        
def parse_hugging_source(key, split, columns):
    params = {"key": key, "split": "train+test", "make_split": False}
    
    if split:
        params["split"] = split
        params["make_split"] = True
    if columns:
        params["columns"] = columns
    
    return params

def parse_skm_source(key, variant, _):
    params = {"key": key, "variant": "train+test", "make_split": False}
    
    if variant:
        params["variant"] = variant
        params["make_split"] = True
    
    return params

def main():
    parser = argparse.ArgumentParser(description="CLI tool for NLP data augmentation for Multi-Label Classification.")
    parser.add_argument("--datasets", nargs="+", required=True, 
                        help='List of datasets. HF: Hugging Face dataset module, DF: Dataset Folder (./data) .Example: "HF:dair-ai/emotion" "DF:archivo.csv"')
    
    parser.add_argument("--action", choices=["describe"], required=True, 
                        help="Analysis actions")

    args = parser.parse_args()
    datasets_manager = DatasetManager()
    
    source_parsers = { "DF": parse_csv_source, "TF": parse_csv_source, "HF": parse_hugging_source, "SKM": parse_skm_source }
    for dataset in args.datasets:
        source, info = dataset.split(":")
        params = source_parsers[source]( *DatasetParser.extract_info(info) )
        datasets_manager.load_dataset(source, params)
        datasets_manager.items[-1].get_data()
        print(
            datasets_manager.items[-1].X_train.shape,
            datasets_manager.items[-1].X_test.shape,
            datasets_manager.items[-1].y_train.shape,
            datasets_manager.items[-1].y_test.shape)
        
  
if __name__ == "__main__":
    main()

"""
cli.py --datasets DF:coding_train.csv[coding_test.csv] HF:emotions()

"""