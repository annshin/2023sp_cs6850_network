import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle


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

    total_nb_paper = 0
    count_doc = defaultdict(int) # 
    count_doc_subject = defaultdict(int) #

    count_doc_cited = defaultdict(int) #
    doc_cited = defaultdict(set)
    count_doc_cited_subject = defaultdict(int) #
    doc_cited_subject = defaultdict(set)

    count_nb_references = defaultdict(int) #
    count_nb_references_subject = defaultdict(int) #

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
                    total_nb_paper += 1
                    discipline = paper["discipline"]
                    count_doc[discipline] += 1

                    if "categories" not in paper["metadata"]:
                        paper_subjects = []
                    else:
                        paper_subjects = paper["metadata"]["categories"].split(" ")

                    for subject in paper_subjects:
                        count_doc_subject[subject] += 1
                
                    for cited_paper_id, cited_paper_info in paper["bib_entries"].items():
                        count_nb_references[discipline] += 1
                        if cited_paper_id in doc_cited[discipline]:
                            continue
                        else:
                            doc_cited[discipline].add(cited_paper_id)

                        count_doc_cited[discipline] += 1
                        for subject in paper_subjects:
                            count_nb_references_subject[subject] += 1
                            if cited_paper_id in doc_cited_subject[subject]:
                                continue
                            else:
                                count_doc_cited_subject[subject] += 1
                                doc_cited_subject[subject].add(cited_paper_id)
    
    return total_nb_paper, count_doc, count_doc_subject, count_doc_cited, count_doc_cited_subject, count_nb_references, count_nb_references_subject, \
        doc_cited, doc_cited_subject



if __name__ == "__main__":
    # Load the data
    # full_year_range = ['00']
    full_year_range = ['91', '92', '93', '94', '95', '96', '97', '98', '99', '00', 
                  '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                  '21', '22']
    # full_year_range = ['91']

    total_nb_paper, count_doc, count_doc_subject, count_doc_cited, count_doc_cited_subject, count_nb_references, count_nb_references_subject, \
        doc_cited, doc_cited_subject = preprocess_data(full_year_range)
    # Save total_nb_paper, count_doc, count_doc_subject, count_doc_cited, count_doc_cited_subject to pickle

    with open('stats_total_nb_paper.pickle', 'wb') as handle:
        pickle.dump(total_nb_paper, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_count_doc.pickle', 'wb') as handle:
        pickle.dump(count_doc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_count_doc_subject.pickle', 'wb') as handle:
        pickle.dump(count_doc_subject, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_count_doc_cited.pickle', 'wb') as handle:
        pickle.dump(count_doc_cited, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_count_doc_cited_subject.pickle', 'wb') as handle:
        pickle.dump(count_doc_cited_subject, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_count_nb_references.pickle', 'wb') as handle:
        pickle.dump(count_nb_references, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_count_nb_references_subject.pickle', 'wb') as handle:
        pickle.dump(count_nb_references_subject, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_doc_cited.pickle', 'wb') as handle:
        pickle.dump(doc_cited, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('stats_doc_cited_subject.pickle', 'wb') as handle:
        pickle.dump(doc_cited_subject, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Read

    # with open('stats_total_nb_paper.pickle', 'rb') as handle:
    #     total_nb_paper = pickle.load(handle)
    # with open('stats_count_doc.pickle', 'rb') as handle:
    #     count_doc = pickle.load(handle)
    # with open('stats_count_doc_subject.pickle', 'rb') as handle:
    #     count_doc_subject = pickle.load(handle)
    # with open('stats_count_doc_cited.pickle', 'rb') as handle:
    #     count_doc_cited = pickle.load(handle)
    # with open('stats_count_doc_cited_subject.pickle', 'rb') as handle:
    #     count_doc_cited_subject = pickle.load(handle)
    # with open('stats_count_nb_references.pickle', 'rb') as handle:
    #     count_nb_references = pickle.load(handle)
    # with open('stats_count_nb_references_subject.pickle', 'rb') as handle:
    #     count_nb_references_subject = pickle.load(handle)
    # with open('stats_doc_cited.pickle', 'rb') as handle:
    #     doc_cited = pickle.load(handle)
    # with open('stats_doc_cited_subject.pickle', 'rb') as handle:
    #     doc_cited_subject = pickle.load(handle)
    
    print("total_nb_paper", total_nb_paper)
    print("count_doc", count_doc)
    print("count_doc_subject", count_doc_subject)
    print("count_doc_cited", count_doc_cited)
    print("count_doc_cited_subject", count_doc_cited_subject)

    print("count_nb_references", count_nb_references)
    print("count_nb_references_subject", count_nb_references_subject)

