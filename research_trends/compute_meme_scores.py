import os
import json
import pickle
import argparse
import string
from tqdm import tqdm
from datetime import date
import numpy as np
from time import time

SCRATCH_PATH = '/scratch/datasets/aw588/'
UNARXIVE_PATH = SCRATCH_PATH + "unarXive"
CACHE_PATH = '/scratch/datasets/mog29/unarXive'
VALID_DISCIPLINES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]
INTRO_SUBSTRING = 'introduction'
RW_SUBSTRING = 'related'
METHOD_SUBSTRINGS = ['method', 'model', 'approach']
IDF_THRESHOLD = 1.6


def compute_overall_frequencies(year_to_frequencies, paper_to_metadata):
    for paper, metadata in tqdm(paper_to_metadata.items()):
        categories = metadata['categories']
        found_match = any([cat in VALID_DISCIPLINES for cat in categories])
        if not found_match:
            continue

        year = int(metadata['release_date'][-1])
        for curr_year in range(year, 2023): 
            year_to_frequencies[curr_year]['num_papers'] += 1
            weighted_freq = (year - 1991 + 1) / (curr_year - 1991 + 1)
            year_to_frequencies[curr_year]['weighted_num_papers'] += weighted_freq

def is_meme_in_paper(paper, meme, meme_to_articles):
    if meme not in meme_to_articles:
        return False
    return paper in meme_to_articles[meme]

def is_meme_in_citations(citations, meme, meme_to_articles):
    truth_vals = []
    for paper in citations:
        truth_vals.append(is_meme_in_paper(paper, meme, meme_to_articles))
    return any(truth_vals)

def compute_n_gram_meme_score_terms(meme_to_score_components, meme_to_articles, paper_to_metadata, meme_to_idf):
    for meme, papers_appearing_in in tqdm(meme_to_articles.items()):
        meme_idf = meme_to_idf[meme]
        if meme_idf < IDF_THRESHOLD:
            continue

        for paper in papers_appearing_in:
            # Choose whether to skip the current appearance of the meme
            metadata = paper_to_metadata[paper]
            categories = metadata['categories']
            found_match = any([cat in VALID_DISCIPLINES for cat in categories])
            if not found_match:
                continue
            if meme not in meme_to_score_components:
                meme_to_score_components[meme] = {}

            # Extract other paper info
            paper_year = int(metadata["release_date"][-1])
            citations = metadata['cited_papers']

            # Get info for meme score
            meme_in_citations = is_meme_in_citations(citations, meme, meme_to_articles)

            for curr_year in range(paper_year, 2023):
                if curr_year not in meme_to_score_components[meme]:
                    meme_to_score_components[meme][curr_year] = {
                        'frequency' : 0,
                        'weighted_frequency' : 0,
                        'in_paper_in_citations' : 0,
                        'in_citations' : 0,
                        'in_paper_not_in_citations' : 0,
                        'not_in_citations' : 0,
                        'weighted_in_paper_in_citations' : 0,
                        'weighted_in_citations' : 0,
                        'weighted_in_paper_not_in_citations' : 0,
                        'weighted_not_in_citations' : 0
                    }

                year_weight = (paper_year - 1991 + 1) / (curr_year - 1991 + 1)
                meme_to_score_components[meme][curr_year]['frequency'] += 1
                meme_to_score_components[meme][curr_year]['weighted_frequency'] += year_weight
                if meme_in_citations:
                    meme_to_score_components[meme][curr_year]['in_paper_in_citations'] += 1
                    meme_to_score_components[meme][curr_year]['weighted_in_paper_in_citations'] += year_weight
                    meme_to_score_components[meme][curr_year]['in_citations'] += 1
                    meme_to_score_components[meme][curr_year]['weighted_in_citations'] += year_weight
                else:
                    meme_to_score_components[meme][curr_year]['in_paper_not_in_citations'] += 1
                    meme_to_score_components[meme][curr_year]['weighted_in_paper_not_in_citations'] += year_weight
                    meme_to_score_components[meme][curr_year]['not_in_citations'] += 1
                    meme_to_score_components[meme][curr_year]['weighted_not_in_citations'] += year_weight
                            
def save_annual_meme_scores(year_to_frequencies, meme_to_score_components):
    meme_to_year_scores = {}
    for meme, meme_year_dict in tqdm(meme_to_score_components.items()):
        meme_to_year_scores[meme] = {}
        for curr_year, year_info in meme_year_dict.items():
            # Compute the frequency scores
            total_frequency = year_to_frequencies[curr_year]['num_papers']
            meme_frequency = year_info['frequency']
            frequency_score = meme_frequency / total_frequency
            weighted_total_frequency = year_to_frequencies[curr_year]['weighted_num_papers']
            weighted_meme_frequency = year_info['weighted_frequency']
            weighted_frequency_score = weighted_meme_frequency / weighted_total_frequency

            # Compute sticking scores
            in_paper_in_citations = year_info['in_paper_in_citations']
            in_citations = year_info['in_citations']
            sticking_score = in_paper_in_citations / (1 + in_citations)
            weighted_in_paper_in_citations = year_info['weighted_in_paper_in_citations']
            weighted_in_citations = year_info['weighted_in_citations']
            weighted_sticking_score = weighted_in_paper_in_citations / (1 + weighted_in_citations)

            # Compute sparking scores
            in_paper_not_in_citations = year_info['in_paper_not_in_citations']
            not_in_citations = year_info['not_in_citations']
            sparking_score = in_paper_not_in_citations / (1 + not_in_citations)
            weighted_in_paper_not_in_citations = year_info['weighted_in_paper_not_in_citations']
            weighted_not_in_citations = year_info['weighted_not_in_citations']
            weighted_sparking_score = weighted_in_paper_not_in_citations / (1 + weighted_not_in_citations)

            # Compute meme scores
            meme_score = frequency_score * sticking_score / sparking_score
            weighted_meme_score = weighted_frequency_score * weighted_sticking_score / weighted_sparking_score
            meme_to_year_scores[meme][curr_year] = {
                "meme_score" : meme_score,
                "weighted_meme_score" : weighted_meme_score
            }

    meme_score_path = os.path.join(CACHE_PATH, 'meme_scores.pkl')
    with open(meme_score_path, 'wb') as f:
        pickle.dump(meme_to_year_scores, f)

if __name__ == "__main__":
    years = [i for i in range(1991, 2023)]
    year_to_frequencies = {year : {'num_papers' : 0, 'weighted_num_papers' : 0} for year in years}
    meme_to_score_components = {}

    # Load meme to idf
    start_time = time()
    idf_path = os.path.join(CACHE_PATH, 'meme_idf.pkl')
    with open(idf_path, 'rb') as f:
        meme_to_idf = pickle.load(f)
    end_time = time()
    print(f'Loading meme to idf took {end_time - start_time} seconds')

    # Load meme to papers
    start_time = time()
    meme_to_articles_path = os.path.join(CACHE_PATH, 'n_gram_to_papers.pkl')
    with open(meme_to_articles_path, 'rb') as f:
        meme_to_articles = pickle.load(f)
    end_time = time()
    print(f'Loading meme to articles took {end_time - start_time} seconds')

    start_time = time()
    paper_to_metadata_path = os.path.join(CACHE_PATH, 'paper_to_metadata.pkl')
    with open(paper_to_metadata_path, 'rb') as f:
        paper_to_metadata = pickle.load(f)
    end_time = time()
    print(f'Loading paper to metadata took {end_time - start_time} seconds')

    # Compute the total number of publications per year
    compute_overall_frequencies(year_to_frequencies, paper_to_metadata)

    # Compute the n_gram specific components of the meme score
    compute_n_gram_meme_score_terms(meme_to_score_components, meme_to_articles, paper_to_metadata)

    # Compute meme scores
    save_annual_meme_scores(year_to_frequencies, meme_to_score_components)
