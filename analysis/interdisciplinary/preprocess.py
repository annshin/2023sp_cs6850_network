import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


data_path = "/scratch/datasets/aw588/unarXive/"


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def preprocess_data(year_range):
    data = []
    # Add tqdm to show progress when iterating over year

    # For each year, count the number of cited paper where there's no discipline info
    no_discipline_info = []

    for year in tqdm(year_range):
        folder_path = f'{data_path}/{year}'
        discipline_info_no_field = 0
        discipline_info_empty = 0
        total_papers = 0
        for file in os.listdir(folder_path):
            if file.endswith(".jsonl"):
                file_path = os.path.join(folder_path, file)
                print(f"Reading {file_path}")
                papers = read_jsonl(file_path)
                total_papers += len(papers)
                for paper in papers:
                    if "categories" not in paper["metadata"]:
                        paper_subjects = []
                    else:
                        paper_subjects = paper["metadata"]["categories"].split(" ")
                
                    for cited_paper_id, cited_paper_info in paper["bib_entries"].items():
                        # import pdb; pdb.set_trace()
                        # TODO: count
                        if 'discipline' not in cited_paper_info:
                            discipline_info_no_field += 1
                            continue
                        if cited_paper_info['discipline'] == '':
                            discipline_info_empty += 1
                            continue

                        data.append({
                            "year": str(year), # TODO: str vs. int?
                            "paper_id": paper["paper_id"],
                            "paper_discipline": paper["discipline"],
                            "paper_subjects": paper_subjects,
                            "cited_paper_id": cited_paper_id,
                            "cited_paper_discipline": cited_paper_info["discipline"]
                        })
        no_discipline_info.append({
            "year": year,
            "total": total_papers,
            "no_discipline_info": discipline_info_no_field,
            "no_discipline_info_ratio": round(discipline_info_no_field / total_papers * 100, 2),
            "empty_discipline_info": discipline_info_empty,
            "empty_discipline_info_ratio": round(discipline_info_empty / total_papers * 100, 2)
        })
    return pd.DataFrame(data), pd.DataFrame(no_discipline_info)


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
    # Load the data
    # full_year_range = ['00']
    full_year_range = ['91', '92', '93', '94', '95', '96', '97', '98', '99', '00', 
                  '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                  '21', '22']

    for year in tqdm(full_year_range):
        print(f"Year: {year}")

        year_range = [year]
        df, no_discipline_df = preprocess_data(year_range)
        print("Data type of column before conversion:", df['year'].dtypes)
        print("df", df.head())
        print("no_discipline_df", no_discipline_df)
        df.to_csv(f"general/data_{year}_df.csv", index=False)
        no_discipline_df.to_csv(f"general/no_discipline_info_{year}_df.csv", index=False)
        print("Saved data to csv.")

        print("-----------------------------------------------------------------")

        print("Analysis: interdisciplinary combinations")
        interdisciplinary_combinations = count_interdisciplinary_combinations(df, year)

        print("Interdisciplinary combinations")
        print(interdisciplinary_combinations.head())
        # Save the data
        interdisciplinary_combinations.to_csv(f"interdisciplinary_combinations/interdisciplinary_combinations_{year}.csv", index=False)

        print("==================================================================")








    # df['year'] = df['year'].apply(lambda x: "{:02d}".format(x)).astype(str)
    # df.to_csv(f"general/data_{year}_df.csv", index=False)
    # # df.to_csv("general/data_df.csv", index=False)

    # # # Read from csv
    # # data_df = pd.read_csv("data/data_22_df.csv")
    # # print("Data loaded")
    # # print(data_df.head())
