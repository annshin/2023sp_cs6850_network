import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import ast


data_path = "/scratch/datasets/aw588/unarXive/"


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


desired_categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]

# Function to check if there's an intersection between desired_categories and paper_subjects
def intersection_exists(row):
    paper_subjects = ast.literal_eval(row['paper_subjects'])
    return bool(set(paper_subjects) & set(desired_categories))


def filter_by_subjects(file_path, desired_categories):
    df = pd.read_csv(file_path, dtype={'year': 'str'})

    filtered_df = df[df.apply(lambda row: intersection_exists(row) and row['paper_discipline'] in ['Computer Science', 'Statistics'], axis=1)]
    return filtered_df


def calculate_interdisciplinarity(df):
    df["interdisciplinary"] = df["paper_discipline"] != df["cited_paper_discipline"]
    interdisciplinarity = df.groupby("paper_id")["interdisciplinary"].mean().reset_index()
    interdisciplinarity.columns = ["paper_id", "interdisciplinarity"]
    return interdisciplinarity


def analyze_discipline_combinations(df):
    df["discipline_combination"] = df["paper_discipline"] + " - " + df["cited_paper_discipline"]
    discipline_combination_counts = df.groupby(["year", "discipline_combination"]).size().reset_index(name="count")
    return discipline_combination_counts


def count_interdisciplinary_combinations(data_df, year):
    year_data = data_df[data_df['year'] == year]

    interdisciplinary_combinations = year_data.groupby(['paper_discipline', 'cited_paper_discipline']).size().reset_index(name='count')
    
    # Pnly interdisciplinary combinations (different disciplines)
    interdisciplinary_combinations = interdisciplinary_combinations[interdisciplinary_combinations['paper_discipline'] != interdisciplinary_combinations['cited_paper_discipline']]
    
    return interdisciplinary_combinations


if __name__ == "__main__":
    # Load the data
    # full_year_range = ['01']
    full_year_range = ['17', '18', '19', '20', 
                  '21', '22']
    # '91', '92', '93', '94', '95', '96', '97', '98', '99', '00', 
    #               '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
    #               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
    #               '21', '22']

    for year in tqdm(full_year_range):
        print(f"Year: {year}")

        file_path = f"/home/aw588/git_annshin/2023sp_cs6850_network/analysis/interdisciplinary/general/data_{year}_df.csv"
        df = filter_by_subjects(file_path, desired_categories)
        print(df)

        df.to_csv(f"ml_only_cs_stat/data_ml_{year}_df.csv", index=False)
        print("Saved data to csv.")

        # Read the CSV file into a pandas DataFrame
        # df = pd.read_csv(f"ml/data_ml_{year}_df.csv")

        print("-----------------------------------------------------------------")

        print("Analysis: interdisciplinary combinations")
        interdisciplinary_combinations = count_interdisciplinary_combinations(df, year)

        print("Interdisciplinary combinations")
        print(interdisciplinary_combinations.head())

        # Save the data
        interdisciplinary_combinations.to_csv(f"interdisciplinary_combinations_ml_only_cs_stat/interdisciplinary_combinations_ml_{year}.csv", index=False)

        print("==================================================================")

