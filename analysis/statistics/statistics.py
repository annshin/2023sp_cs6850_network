import json
import os
from collections import defaultdict
from tqdm import tqdm


year_range = ['93', '97', '98', '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']


############################################
# General
############################################

def count_papers(data_path):
    total_papers = 0
    papers_by_discipline = defaultdict(int)
        
    for year in tqdm(year_range):
        folder = os.path.join(data_path, year)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, "r") as f:
                for line in f:
                    paper = json.loads(line)
                    paper_discipline = paper["discipline"]
                    total_papers += 1
                    papers_by_discipline[paper_discipline] += 1

    return total_papers, dict(papers_by_discipline)

############################################
# CS-specific
############################################

# Check the number of subjects for CS
def count_cs_subjects(data_path):
    """
    Get all the subjects for CS papers
    """
    cs_categories = defaultdict(int)
    for year in tqdm(year_range):
        folder = os.path.join(data_path, year)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, "r") as f:
                for line in f:
                    paper = json.loads(line)
                    # Sanity check
                    if paper["discipline"] != "Computer Science":
                        continue

                    categories = paper["metadata"]["categories"].split(" ")
                    
                    for category in categories:
                        if category.startswith("cs."):
                            cs_categories[category] += 1
    return dict(cs_categories)


if __name__ == "__main__":
    data_path = "../data/"
    output_path = "../analysis/statistics"

    ##############################
    # General
    ##############################

    # total_papers, papers_by_discipline = count_papers(data_path)

    # print(f"Total citing papers: {total_papers}")
    # print("Citing papers by discipline:")
    # for discipline, count in papers_by_discipline.items():
    #     print(f"{discipline}: {count}")

    ##############################
    # CS-specific
    ##############################

    # Get all the categories for CS papers, in the format: {"cs.CL": 100, "cs.CV": 200, ...}
    # Result: Total number of CS subjects: 39

    # cs_categories = count_cs_subjects(data_path)
    # # Dump the results
    # with open(os.path.join(output_path, "cs_categories.json"), "w") as f:
    #     json.dump(cs_categories, f, indent=4)
    
    # with open(os.path.join(output_path, "cs_categories.json"), "r") as f:
    #     cs_categories = json.load(f)
    # print(f"Total number of CS subjects: {len(cs_categories)}")
