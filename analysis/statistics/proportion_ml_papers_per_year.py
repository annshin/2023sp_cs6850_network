import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


data_path = "/scratch/datasets/aw588/unarXive/"
desired_categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def preprocess_data(year_range):
    data = []
    # Add tqdm to show progress when iterating over year
    for year in tqdm(year_range):
        folder_path = f'{data_path}/{year}'
        total_number_papers = 0
        desired_papers_count = 0
        number_papers_without_categories = 0
        for file in os.listdir(folder_path):
            if file.endswith(".jsonl"):
                file_path = os.path.join(folder_path, file)
                print(f"Reading {file_path}")
                papers = read_jsonl(file_path)
                for paper in papers:
                    total_number_papers += 1
                    if "categories" not in paper["metadata"]:
                        number_papers_without_categories += 1
                        continue
                    if any(category in paper["metadata"]["categories"].split(" ") for category in desired_categories):
                        desired_papers_count += 1
        data.append((year, desired_papers_count / total_number_papers, desired_papers_count, number_papers_without_categories, total_number_papers))
    return pd.DataFrame(data, columns=["Year", "Proportion", "MLPapersCount", "NbPapersNoCategories", "TotalPapersCount"])



if __name__ == "__main__":

    # Load the data
    year_range = ['91', '92', '93', '94', '95', '96', '97', '98', '99', '00', 
                  '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                  '21', '22']
    # year = "22"
    # year_range = [year]
    # year_range = ["22", "93"]

    # df = preprocess_data(year_range)
    # # df.to_csv(f"proportion_ml_paper_per_year_{year}_df.csv", index=False)
    # df.to_csv(f"proportion_ml_paper_per_year_df.csv", index=False)

    # Read csv
    # df = pd.read_csv(f"proportion_ml_paper_per_year_{year}_df.csv")
    df = pd.read_csv(f"proportion_ml_paper_per_year_df.csv")
    # print(df)

    # Check the data type of column
    print("Data type of column before conversion:", df['Year'].dtypes)

    # Convert the data type of column 'A' to str
    df['Year'] = df['Year'].apply(lambda x: "{:02d}".format(x)).astype(str)

    # Check the data type of column after conversion
    print("Data type of column after conversion:", df['Year'].dtypes)

    # Print the updated DataFrame
    print(df)

    # Plot the proportions
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 16})
    plt.plot(df["Year"], df["Proportion"]*100, marker='o', linestyle='-', linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Proportion of ML Papers")

    x_labels = year_range
    plt.gca().tick_params(axis='x', labelsize=12)
    plt.xticks(df["Year"], x_labels)


    # plt.title("Evolution of the proportion of ML Papers per Year")
    # PLT save
    # plt.savefig(f"proportion_ml_paper_per_year_{year}.png")
    plt.savefig(f"proportion_ml_paper_per_year.png")