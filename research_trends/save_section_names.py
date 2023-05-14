import os
import json
import pickle
import argparse
import string
from tqdm import tqdm
from datetime import date

SCRATCH_PATH = '/scratch/datasets/aw588/'
UNARXIVE_PATH = SCRATCH_PATH + "unarXive"
CACHE_PATH = '/scratch/datasets/mog29/unarXive'
VALID_DISCIPLINES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]


def get_args():
    parser = argparse.ArgumentParser(description="Getting trends in the data")
    parser.add_argument('--start_index', type=int,
                        help="If set, we will slice from the provided index (inclusive)")
    parser.add_argument('--end_index', type=int,
                        help="If set, we will end the slice at the given index (non-inclusive)")
    args = parser.parse_args()
    return args

def read_jsonl(filepath):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def get_json_section_metadata(p2s_metadata, json_data):
    for paper in json_data:
        paper_id = paper['paper_id']

        # Get category information
        if 'categories' not in paper['metadata']:
            paper_categories = []
        else:
            paper_categories = paper['metadata']['categories'].split(' ')

        # Get section metadata
        section_name_number_pairs = []
        for body in paper['body_text']:
            section_name = body['section']
            section_num = body['sec_number']
            if section_name in ['', " "] or section_name in ['', ' ']:
                continue
            section_name_number_pairs.append((section_name, section_num))
        section_name_number_pairs = set(section_name_number_pairs)

        p2s_metadata[paper_id] = {
            "categories" : paper_categories,
            "name_number_pairs" : section_name_number_pairs
        }

if __name__ == "__main__":
    args = get_args()
    years = [str(i)[2:] for i in range(1991, 2023)]
    paper_to_section_metadata = {}

    print(len(years))

    start_index = 0 if args.start_index is None else args.start_index
    end_index = len(years) if args.end_index is None else args.end_index
    years = years[start_index:end_index]

    # Iterate over each year
    for year in years:
        if ".tar.xz" in year:
            continue
        year_folder = os.path.join(UNARXIVE_PATH, year)
        year_jsons = sorted(os.listdir(year_folder))

        # Iterate over each json in said year
        print(year)
        for year_json in tqdm(year_jsons):
            # Read the json
            full_path = os.path.join(year_folder, year_json)
            json_data = read_jsonl(full_path)
        
            get_json_section_metadata(paper_to_section_metadata, json_data)

    cache_filename = os.path.join(CACHE_PATH, f'paper_to_section_metadata_{start_index}_{end_index}.pkl')
    with open(cache_filename, 'wb') as f:
        pickle.dump(paper_to_section_metadata, f)



