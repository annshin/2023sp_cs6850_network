import os
import json, jsonlines
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pickle


data_path = "/scratch/datasets/aw588/unarXive/"

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

if __name__ == "__main__":
    # # Load the data
    # years_of_interest = ['02', '07', '12', '17', '22']
    years_of_interest = ['91', '92', '93', '94', '95', '96', '97', '98', '99', '00', 
                  '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                  '21', '22']
    # years_of_interest = ['91']
    disciplines_of_interest = ["Mathematics", "Physics", "Computer Science", "Economics"]
    disciplines_of_interest.sort()
    print(disciplines_of_interest)

    # Add "Machine Learning" as a special discipline
    special_disciplines = {"Machine Learning": [("Computer Science", ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]),
                                                ("Statistics", ["stat.ML"])]}


    results = {}
    root = "/home/aw588/git_annshin/2023sp_cs6850_network/analysis/interdisciplinary/general"

    # # Read each CSV file and compute the count of unique disciplines cited
    # for year in years_of_interest:
    #     print(f"Year: {year}")
    #     df = pd.read_csv(f'{root}/data_{year}_df.csv', dtype={'year': 'str'})
    #     results[year] = {}
    #     for discipline in disciplines_of_interest:
    #         print(f"Discipline: {discipline}")
    #         # Select rows where paper_discipline matches the discipline of interest
    #         df_discipline = df[df['paper_discipline'] == discipline]

    #         unique_cited_disciplines = df_discipline['cited_paper_discipline'].nunique()
    #         results[year][discipline] = unique_cited_disciplines

    #     # Process special disciplines
    #     print("Processing special disciplines")
    #     for special_discipline, conditions in special_disciplines.items():
    #         print(f"Special discipline: {special_discipline}")
    #         df_special = pd.DataFrame()
    #         for condition in conditions:
    #             discipline, subjects = condition
    #             df_temp = df[(df['paper_discipline'] == discipline) & (df['paper_subjects'].apply(lambda x: any(i in ast.literal_eval(x) for i in subjects)))]
    #             df_special = pd.concat([df_special, df_temp])
    #         unique_cited_disciplines = df_special['cited_paper_discipline'].nunique()
    #         results[year][special_discipline] = unique_cited_disciplines

    # with open('unique_cited_disciplines_with_ml.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    with open('unique_cited_disciplines_with_ml.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    print(results )
    # ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Scatter plot and line plot
    all_disciplines = disciplines_of_interest + list(special_disciplines.keys())
    for discipline in all_disciplines:
        counts = [results[year][discipline] if discipline in results[year] else float('nan') for year in years_of_interest]
        # If the discipline is Economics, ignore the counts until 2010
        if discipline == "Economics":
            counts = [float('nan')]*20 + counts[20:]
        if discipline == "Machine Learning" or discipline == "Computer Science":
            counts = [float('nan')]*3 + counts[3:]
        years = [f"20{year}" if not year.startswith('9') else f"19{year}" for year in years_of_interest]  # years as x-axis
        ax.plot(years, counts, marker='o', label=discipline)  # Line plot with scatter


    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Number of different disciplines cited', fontsize=18)
    desired_xticks = ['1995', '2000', '2005', '2010', '2015', '2020']
    desired_xticklabels = ['95', '00', '05', '10', '15', '20']

    ax.set_xticks(desired_xticks)
    ax.set_xticklabels(desired_xticks, fontsize=13)

    ax.legend(loc='best', fontsize=12)
    plt.subplots_adjust(bottom=0.17)


    fig.savefig(f'unique_cited_disciplines_with_ml.png')
