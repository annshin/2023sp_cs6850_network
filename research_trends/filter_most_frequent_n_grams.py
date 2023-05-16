import os
import json
import pickle
import argparse
import string
from tqdm import tqdm
from datetime import date
import numpy as np
from time import time
from collections import Counter

from combine_meme_files import get_combined_n_grams, get_combined_metadata

SCRATCH_PATH = '/scratch/datasets/aw588/'
UNARXIVE_PATH = SCRATCH_PATH + "unarXive"
CACHE_PATH = '/scratch/datasets/mog29/unarXive'
VALID_DISCIPLINES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]
INTRO_SUBSTRING = 'introduction'
RW_SUBSTRING = 'related'
METHOD_SUBSTRINGS = ['method', 'model', 'approach']
IDF_THRESHOLD = 1.6


def compute_overall_frequencies(year_to_frequencies, paper_to_metadata, disciplines):
    for paper, metadata in tqdm(paper_to_metadata.items()):
        categories = metadata['categories']
        found_match = any([cat in disciplines for cat in categories])
        if not found_match:
            continue

        year = int(metadata['release_date'][-1])
        for curr_year in range(year, 2023): 
            year_to_frequencies[curr_year]['num_papers'] += 1

def is_meme_in_paper(paper, meme, meme_to_articles):
    if meme not in meme_to_articles:
        return False
    return paper in meme_to_articles[meme]

def is_meme_in_citations(citations, meme, meme_to_articles):
    truth_vals = []
    for paper in citations:
        truth_vals.append(is_meme_in_paper(paper, meme, meme_to_articles))
    return any(truth_vals)

def compute_n_gram_meme_score_terms(meme_to_score_components, meme_to_articles,
                                    paper_to_metadata, meme_to_idf, disciplines):
    for meme, papers_appearing_in in tqdm(meme_to_articles.items()):
        meme_idf = meme_to_idf[meme]
        if meme_idf < IDF_THRESHOLD:
            continue

        if len(papers_appearing_in) < 500:
            continue

        for paper in papers_appearing_in:
            # Choose whether to skip the current appearance of the meme
            metadata = paper_to_metadata[paper]
            categories = metadata['categories']
            found_match = any([cat in disciplines for cat in categories])
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
                        'in_paper_in_citations' : 0,
                        'in_citations' : 0,
                        'in_paper_not_in_citations' : 0,
                        'not_in_citations' : 0
                    }

                year_weight = (paper_year - 1991 + 1) / (curr_year - 1991 + 1)
                meme_to_score_components[meme][curr_year]['frequency'] += 1
                if meme_in_citations:
                    meme_to_score_components[meme][curr_year]['in_paper_in_citations'] += 1
                    meme_to_score_components[meme][curr_year]['in_citations'] += 1
                else:
                    meme_to_score_components[meme][curr_year]['in_paper_not_in_citations'] += 1
                    meme_to_score_components[meme][curr_year]['not_in_citations'] += 1
                            
def save_annual_meme_scores(year_to_frequencies, meme_to_score_components, discipline_suffix):
    meme_to_year_scores = {}
    for meme, meme_year_dict in tqdm(meme_to_score_components.items()):
        meme_to_year_scores[meme] = {}
        for curr_year, year_info in meme_year_dict.items():
            # Compute the frequency scores                                                                                                                                                                                                                                                
            total_frequency = year_to_frequencies[curr_year]['num_papers']
            meme_frequency = year_info['frequency']
            frequency_score = meme_frequency / total_frequency

            # Compute sticking scores                                                                                                                                                                                                                                                     
            in_paper_in_citations = year_info['in_paper_in_citations']
            in_citations = year_info['in_citations']
            sticking_score = in_paper_in_citations / (3 + in_citations)

            # Compute sparking scores                                                                                                                                                                                                                                                     
            in_paper_not_in_citations = year_info['in_paper_not_in_citations']
            not_in_citations = year_info['not_in_citations']
            sparking_score = (3+in_paper_not_in_citations) / (3 + not_in_citations)

            # Compute meme scores                                                                                                                                                                                                                                                         
            meme_score = frequency_score * sticking_score / sparking_score
            meme_to_year_scores[meme][curr_year] = {
                "meme_score" : meme_score,
            }

    meme_score_path = os.path.join(CACHE_PATH, f'meme_scores_{discipline_suffix}.pkl')
    with open(meme_score_path, 'wb') as f:
        pickle.dump(meme_to_year_scores, f)        

def get_most_common_memes(year_memes, paper_to_metadata, meme_to_idf, disciplines):
    # Get meme counts
    meme_to_counts = Counter()
    for meme, papers_appearing_in in year_memes.items():
        meme_idf = meme_to_idf[meme]
        if meme_idf < IDF_THRESHOLD:
            continue

        for paper in papers_appearing_in:
            # Choose whether to skip the current appearance of the meme
            metadata = paper_to_metadata[paper]
            categories = metadata['categories']
            found_match = any([cat in disciplines for cat in categories])
            if not found_match:
                continue
            meme_to_counts[meme] += 1

    # Filter out the top 10000
    print(f"Meme count before filtering: {len(meme_to_counts)}")
    most_common = meme_to_counts.most_common(10000)
    most_common = [pair[0] for pair in most_common]
    return most_common
        

if __name__ == "__main__":
    years = [i for i in range(1991, 2023)]

    # Load meme to idf
    start_time = time()
    idf_path = os.path.join(CACHE_PATH, 'meme_idf.pkl')
    with open(idf_path, 'rb') as f:
        meme_to_idf = pickle.load(f)
    end_time = time()
    print(f'Loading meme to idf took {end_time - start_time} seconds')

    # Get paper metadata
    paper_to_metadata = get_combined_metadata()

    # Get the disciplines to consider
    all_disciplines = [
        VALID_DISCIPLINES,
        ['astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR'],
        ['cond-mat.dis-nn', 'cond-mat.mes-hall', 'cond-mat.mtrl-sci', 'cond-mat.other',
         'cond-mat.quant-gas', 'cond-mat.soft', 'cond-mat.stat-mech', 'cond-mat.str-el',
         'cond-mat.supr-con'],
        ['hep-ex', 'hep-lat', 'hep-ph', 'hep-th'],
        ['nlin.AO', 'nlin.CD', 'nlin.CG', 'nlin.PS', 'nlin.SI'],
        ['nucl-ex', 'nucl-th'],
        ['eess.AS', 'eess.IV', 'eess.SP']]
    discipline_suffixes = [
        'ai',
        'astrophysics',
        'condensed_matter',
        'high_energy',
        'nonlinear',
        'nuclear',
        'signal_processing']

    
    # Iterate over each year
    for year in tqdm(years):
        # Load the memes for a given year
        year_meme_path = os.path.join(CACHE_PATH, f'n_gram_to_papers_{str(year)[2:]}.pkl')
        with open(year_meme_path, 'rb') as f:
            year_memes = pickle.load(f)

        # Iterate over disciplines
        for i in range(len(all_disciplines)):
            discipline = all_disciplines[i]
            discipline_suffix = discipline_suffixes[i]

            # Get list of 10000 most common memes
            most_common_memes = get_most_common_memes(year_memes, paper_to_metadata, meme_to_idf, discipline)

            # Save for the discipline
            common_meme_savepath = os.path.join(CACHE_PATH, f'year_{year}_most_common_{discipline_suffix}.pkl')
            with open(common_meme_savepath, 'wb') as f:
                pickle.dump(most_common_memes, f)

