import os
import json, jsonlines
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt
import numpy as np
import pickle


data_path = "/scratch/datasets/aw588/unarXive/"


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


if __name__ == "__main__":
    # Load the data
    years_of_interest = ['02', '07', '12', '17', '22']
    # disciplines_of_interest = ["Mathematics", "Physics", "Computer Science", "Quantitative Finance", "Quantitative Biology", \
    #                            "Statistics", "Electrical Engineering and Systems Science", "Economics"]
    # disciplines_of_interest = ["Statistics", "Electrical Engineering and Systems Science", "Economics"]
    disciplines_of_interest = ["Mathematics", "Physics", "Computer Science", "Economics"]
    disciplines_of_interest.sort()

    results = {}
    root = "/home/aw588/git_annshin/2023sp_cs6850_network/analysis/interdisciplinary/general"

    # # Read each CSV file and compute the percentages
    # for year in years_of_interest:
    #     df = pd.read_csv(f'{root}/data_{year}_df.csv', dtype={'year': 'str'})
    #     results[year] = []
    #     for discipline in disciplines_of_interest:
    #         total_citations = df[df['paper_discipline'] == discipline].shape[0]
    #         same_discipline_citations = df[(df['paper_discipline'] == discipline) & (df['cited_paper_discipline'] == discipline)].shape[0]
    #         percentage = (same_discipline_citations / total_citations) * 100 if total_citations > 0 else 0
    #         print(f"Year: {year}, Discipline: {discipline}, Percentage: {percentage}, total_citations: {total_citations}, same_discipline_citations: {same_discipline_citations}")
    #         results[year].append(percentage)
    
    # with open(f'same_discipline_percentage.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    with open(f'same_discipline_percentage.pkl', 'rb') as f:
        results = pickle.load(f)

    labels = disciplines_of_interest
    data = np.array([results[year] for year in years_of_interest])

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print(f"Mean: {mean}, Standard Deviation: {std}")

    x = np.arange(len(labels))  # the label locations
    width = 0.8 / len(years_of_interest)  # the width of the bars, adjusted to avoid overlap

    fig, ax = plt.subplots(figsize=(10, 6))
    rects = [ax.bar(x - width*(len(years_of_interest)/2-i), data[i], width, label=f"20{years_of_interest[i]}") for i in range(len(years_of_interest))]

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Discipline', fontsize=16)
    ax.set_ylabel('Percentage', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='right', fontsize=13)  # Rotate labels for readability

    ax.legend(loc='best', fontsize=12)

    fig.tight_layout()
    fig.savefig(f'same_discipline_percentage.png')
