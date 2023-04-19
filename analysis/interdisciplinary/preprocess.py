import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


data_path = "../../data/"


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def preprocess_data(year_range):
    data = []
    # Add tqdm to show progress when iterating over year
    for year in tqdm(year_range):
        folder_path = f'{data_path}/{year}'
        for file in os.listdir(folder_path):
            if file.endswith(".jsonl"):
                file_path = os.path.join(folder_path, file)
                papers = read_jsonl(file_path)
                for paper in papers:
                    for cited_paper_id, cited_paper_info in paper["bib_entries"].items():
                        # import pdb; pdb.set_trace()
                        # TODO: count
                        if 'discipline' not in cited_paper_info:
                            continue
                        if cited_paper_info['discipline'] == '':
                            continue
                        data.append({
                            "year": year, # TODO: str vs. int?
                            "paper_id": paper["paper_id"],
                            "paper_discipline": paper["discipline"],
                            "paper_subjects": paper["metadata"]["categories"].split(" "),
                            "cited_paper_id": cited_paper_id,
                            "cited_paper_discipline": cited_paper_info["discipline"]
                        })
    return pd.DataFrame(data)

def calculate_interdisciplinarity(df):
    df["interdisciplinary"] = df["paper_discipline"] != df["cited_paper_discipline"]
    interdisciplinarity = df.groupby("paper_id")["interdisciplinary"].mean().reset_index()
    interdisciplinarity.columns = ["paper_id", "interdisciplinarity"]
    return interdisciplinarity

# Common discipline combinations and analyze changes over time
def analyze_discipline_combinations(df):
    df["discipline_combination"] = df["paper_discipline"] + " - " + df["cited_paper_discipline"]
    discipline_combination_counts = df.groupby(["year", "discipline_combination"]).size().reset_index(name="count")
    return discipline_combination_counts


def count_interdisciplinary_combinations(data_df, year):
    year_data = data_df[data_df['year'] == year]

    # Count for each combination
    interdisciplinary_combinations = year_data.groupby(['paper_discipline', 'cited_paper_discipline']).size().reset_index(name='count')
    
    # Pnly interdisciplinary combinations (different disciplines)
    interdisciplinary_combinations = interdisciplinary_combinations[interdisciplinary_combinations['paper_discipline'] != interdisciplinary_combinations['cited_paper_discipline']]
    
    return interdisciplinary_combinations


if __name__ == "__main__":
    output_path = "../analysis/statistics"

    # Load the data
    year_range = ['93', '97', '98', '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    data_df = preprocess_data(year_range)
    data_df.to_csv("general/data_df.csv", index=False)

    # # Read from csv
    # data_df = pd.read_csv("data/data_df.csv")
    # print("Data loaded")
    # print(data_df.head())

    # # # # interdisciplinarity = calculate_interdisciplinarity(data_df)
    # # # # discipline_combination_counts = analyze_discipline_combinations(data_df)

    # # # print(discipline_combination_counts.head())

    # ############################
    # print("Analysis: interdisciplinary combinations")
    # # for year in year_range:
    # #     print(f"Year: {year}")
    # #     year = int(year)
    # #     interdisciplinary_combinations = count_interdisciplinary_combinations(data_df, year)

    # #     print(interdisciplinary_combinations)
    # #     # Save the data
    # #     interdisciplinary_combinations.to_csv(f"analysis/interdisciplinary/interdisciplinary_combinations_{year}.csv", index=False)

    