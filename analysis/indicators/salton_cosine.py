import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import ast
import json

special_disciplines = {"Computer Science": ["cs.AI", "cs.CL", "cs.CV", "cs.LG"],
                       "Statistics": ["stat.ML"]}

def compute_salton_measure(citations_a, citations_b):
    denominator = 1
    numerator = 0
    for cited_discipline in citations_a:
        if cited_discipline in citations_b:
            numerator += citations_a[cited_discipline] * citations_b[cited_discipline]
    
    denominator_a = np.sqrt(sum([count ** 2 for count in citations_a.values()]))
    denominator_b = np.sqrt(sum([count ** 2 for count in citations_b.values()]))
    denominator = denominator_a * denominator_b

    return numerator / denominator

def compute_citation_dict(years, root):
    """
    Output: 
    - citation_dict
    - per year citation dict
    """
    citation_dict = {}
    d = {k: {} for k in years}

    for year in years:
        print(f"Year: {year}")
        path_to_csv = f"/home/aw588/git_annshin/2023sp_cs6850_network/analysis/interdisciplinary/general/data_{year}_df.csv"
        df = pd.read_csv(path_to_csv, dtype={'year': 'str'})

        for _, row in df.iterrows():
            if row['paper_discipline'] not in citation_dict:
                citation_dict[row['paper_discipline']] = defaultdict(int)
            if row['paper_discipline'] not in d[year]:
                d[year][row['paper_discipline']] = defaultdict(int)
            citation_dict[row['paper_discipline']][row['cited_paper_discipline']] += 1
            d[year][row['paper_discipline']][row['cited_paper_discipline']] += 1

            if row['paper_discipline'] in special_disciplines.keys():
                paper_subjects = ast.literal_eval(row['paper_subjects'])
                if any(subject in paper_subjects for subject in special_disciplines[row['paper_discipline']]):
                    if 'Machine Learning' not in citation_dict:
                        citation_dict['Machine Learning'] = defaultdict(int)
                    if 'Machine Learning' not in d[year]:
                        d[year]['Machine Learning'] = defaultdict(int)
                    citation_dict['Machine Learning'][row['cited_paper_discipline']] += 1
                    d[year]['Machine Learning'][row['cited_paper_discipline']] += 1
    return citation_dict, d


def compute_similarity_matrix(citation_dict, disciplines):
    # Compute the similarity matrix
    similarity_matrix = pd.DataFrame(0, index=disciplines, columns=disciplines, dtype=float)
    for discipline_a in disciplines:
        for discipline_b in disciplines:
            # if discipline_a == discipline_b:
            #     continue
            if discipline_a not in citation_dict or discipline_b not in citation_dict:
                similarity_matrix.loc[discipline_a, discipline_b] = 0
                continue
            salton_measure = compute_salton_measure(citation_dict[discipline_a], citation_dict[discipline_b])
            similarity_matrix.loc[discipline_a, discipline_b] = salton_measure

    return similarity_matrix


def compute_integration_score(citation_dict_per_year, similarity_matrix, disciplines):
    results = {}
    for discipline_citing in disciplines:
        if discipline_citing not in ["Physics", "Mathematics", "Computer Science", "Economics", "Machine Learning"]:
            continue
        print("=====================================")
        print(f"Discipline citing: {discipline_citing}")
        integration_score = 0.0
        if discipline_citing not in citation_dict_per_year:
            continue

        denominator = 0
        for key in citation_dict_per_year[discipline_citing]:
            if key in disciplines:
                # We don't count twice ML and CS
                if key != "Machine Learning":
                    denominator += citation_dict_per_year[discipline_citing][key]
        
        for discipline_cited_a in disciplines:
            if discipline_citing == discipline_cited_a:
                continue
            if discipline_citing == "Machine Learning" and discipline_cited_a == "Computer Science":
                continue
            for discipline_cited_b in disciplines:
                if discipline_citing == discipline_cited_b or discipline_cited_a == discipline_cited_b:
                    continue
                if discipline_citing == "Machine Learning" and discipline_cited_b == "Computer Science":
                    continue
                integration_score += (citation_dict_per_year[discipline_citing][discipline_cited_a]/denominator) * (citation_dict_per_year[discipline_citing][discipline_cited_b]/denominator) * (1 - similarity_matrix[discipline_cited_a][discipline_cited_b])

        results[discipline_citing] = integration_score
        print(f"Integration score: {integration_score}")
    return results






# years_of_interest = ['02', '07', '12', '17', '22']
# years_of_interest = ['12', '02']
years_of_interest = ['91', '92', '93', '94', '95', '96', '97', '98', '99', '00', 
                  '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                  '21', '22']
# Specify the disciplines
# disciplines = ["Physics", "Mathematics", "Computer Science", "Economics", "Machine Learning"]
# disciplines_similarity = ["Physics", "Mathematics", "Computer Science", "Economics", "Machine Learning", \
#                           "Quantitative Biology", "Quantitative Finance", "Statistics"]

root = "/home/aw588/git_annshin/2023sp_cs6850_network/analysis/interdisciplinary/general"


# citation_dict, d = compute_citation_dict(years_of_interest, root)

# # Save
# with open('integration_score_citation_dict.pkl', 'wb') as f:
#     pickle.dump(citation_dict, f)

# Load
with open('integration_score_citation_dict.pkl', 'rb') as f:
    citation_dict = pickle.load(f)
# print(citation_dict)

disciplines_similarity = list(citation_dict.keys())
disciplines_similarity = [disc for disc in disciplines_similarity if type(disc) == str]
disciplines_similarity.extend(citation_dict["Physics"].keys())
disciplines_similarity.extend(citation_dict["Mathematics"].keys())
disciplines_similarity.extend(citation_dict["Computer Science"].keys())
disciplines_similarity = list(set(disciplines_similarity))

disciplines_similarity = ["Physics", "Mathematics", "Computer Science", "Economics", "Machine Learning"]

# # Save
# with open('integration_score_per_year_citation_dict.pkl', 'wb') as f:
#     pickle.dump(d, f)

# Load
with open('integration_score_per_year_citation_dict.pkl', 'rb') as f:
    d = pickle.load(f)
# print(d)

# similarity_matrix = compute_similarity_matrix(citation_dict, disciplines_similarity)
# print(similarity_matrix)

# with open('integration_score_similarity_matrix_all.pkl', 'wb') as f:
#     pickle.dump(similarity_matrix, f)

# similarity_matrix.to_csv("integration_score_similarity_matrix_all.csv")

with open('integration_score_similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

# print(similarity_matrix)
similarity_matrix_obj = similarity_matrix.to_dict()
print(similarity_matrix)


results_integration_score_per_year = {}

for year in years_of_interest:
    print(f"Year: {year}")
    integration_score_results = compute_integration_score(d[year], similarity_matrix, disciplines_similarity)
    results_integration_score_per_year[year] = integration_score_results
    for discipline, integration_score in integration_score_results.items():
        print(f"{discipline}: {integration_score}")

    print("\n\n")

with open('integration_score_results.json', 'w') as f:
    json.dump(results_integration_score_per_year, f, indent=4)

