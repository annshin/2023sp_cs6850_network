import os
import json
import pickle
import argparse
import string
from tqdm import tqdm
from datetime import date
import numpy as np
from time import time

from combine_meme_files import get_combined_metadata

SCRATCH_PATH = '/scratch/datasets/aw588/'
UNARXIVE_PATH = SCRATCH_PATH + "unarXive"
CACHE_PATH = '/scratch/datasets/mog29/unarXive'
VALID_DISCIPLINES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]
INTRO_SUBSTRING = 'introduction'
RW_SUBSTRING = 'related'
METHOD_SUBSTRINGS = ['method', 'model', 'approach']
IDF_THRESHOLD = 1.6


def compute_overall_frequencies(year_to_frequencies, paper_to_metadata, disciplines):
    max_year = max(list(year_to_frequencies.keys()))

    for paper, metadata in tqdm(paper_to_metadata.items()):
        categories = metadata['categories']
        found_match = any([cat in disciplines for cat in categories])
        if not found_match:
            continue

        year = int(metadata['release_date'][-1])
        for curr_year in range(year, max_year + 1): 
            year_to_frequencies[curr_year]['num_papers'] += 1

def get_combined_n_grams(years):
    new_years = [str(year)[2:] for year in years]

    # Iterate over all years
    all_paper_ids = set()
    meme_to_articles = {}
    for year in tqdm(new_years):
        year_meme_path = os.path.join(CACHE_PATH, f'n_gram_to_papers_{year}.pkl')
        with open(year_meme_path, 'rb') as f:
            year_memes = pickle.load(f)

        for meme, containing_docs in year_memes.items():
            if meme not in meme_to_articles:
                meme_to_articles[meme] = set()
            for containing_doc in containing_docs:
                meme_to_articles[meme].add(containing_doc)
                all_paper_ids.add(containing_doc)

    return meme_to_articles


def is_meme_in_paper(paper, meme, meme_to_articles):
    if meme not in meme_to_articles:
        return False
    return paper in meme_to_articles[meme]

def is_meme_in_citations(citations, meme, meme_to_articles):
    truth_vals = []
    for paper in citations:
        truth_vals.append(is_meme_in_paper(paper, meme, meme_to_articles))
    return any(truth_vals)

def compute_n_gram_meme_score_terms(year, meme_to_articles, common_memes, meme_to_score_components,
                                    paper_to_metadata, disciplines):
    for paper, metadata in tqdm(paper_to_metadata.items()):
        # Skip paper if out of discipline
        categories = metadata['categories']
        found_match = any([cat in disciplines for cat in categories])
        if not found_match:
            continue

        # Skip paper if published after year
        paper_year = int(metadata["release_date"][-1])
        if paper_year > year:
            continue
        citations = metadata['cited_papers']

        # Iterate over each meme
        for meme in common_memes:
            meme_in_paper = is_meme_in_paper(paper, meme, meme_to_articles)
            meme_in_citations = is_meme_in_citations(citations, meme, meme_to_articles)            

            if meme_in_paper:
                meme_to_score_components[meme]['frequency'] += 1
                if meme_in_citations:
                    meme_to_score_components[meme]['in_paper_in_citations'] += 1
                    meme_to_score_components[meme]['in_citations'] += 1
                else:
                    meme_to_score_components[meme]['in_paper_not_in_citations'] += 1
                    meme_to_score_components[meme]['not_in_citations'] += 1
            else:
                if meme_in_citations:
                    meme_to_score_components[meme]['in_citations'] += 1
                else:
                    meme_to_score_components[meme]['not_in_citations'] += 1                    
                            
def save_year_discipline_meme_scores(year_to_frequencies, meme_to_score_components, year, discipline_suffix):
    meme_to_scores = {}
    for meme, score_components in meme_to_score_components.items():
        # Compute the frequency score
        total_frequency = year_to_frequencies[year]['num_papers']
        meme_frequency = score_components['frequency']
        frequency_score = meme_frequency / (total_frequency + 1e-8)

        # Compute sticking scores
        in_paper_in_citations = score_components['in_paper_in_citations']
        in_citations = score_components['in_citations']
        sticking_score = in_paper_in_citations / (3 + in_citations)
        
        # Compute sparking scores
        in_paper_not_in_citations = score_components['in_paper_not_in_citations']
        not_in_citations = score_components['not_in_citations']
        sparking_score = (3+in_paper_not_in_citations) / (3 + not_in_citations)

        # Compute meme scores
        meme_score = frequency_score * sticking_score / sparking_score
        meme_to_scores[meme] = {"meme_score" : meme_score}

    meme_score_path = os.path.join(CACHE_PATH, f'year_{year}_meme_scores_{discipline_suffix}.pkl')
    with open(meme_score_path, 'wb') as f:
        pickle.dump(meme_to_scores, f)        

if __name__ == "__main__":
    years = [i for i in range(1991, 2023)]

    paper_to_metadata = get_combined_metadata()
    meme_to_articles = get_combined_n_grams(years)

    # Compute the total number of publications per year
    all_disciplines = [
        VALID_DISCIPLINES,
        ['astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR'],
        ['cond-mat.dis-nn', 'cond-mat.mes-hall', 'cond-mat.mtrl-sci', 'cond-mat.other',
         'cond-mat.quant-gas', 'cond-mat.soft', 'cond-mat.stat-mech', 'cond-mat.str-el',
         'cond-mat.supr-con'],
        ['hep-ex', 'hep-lat', 'hep-ph', 'hep-th'],
        ['nlin.AO', 'nlin.CD', 'nlin.CG', 'nlin.PS', 'nlin.SI'],
        ['nucl-ex', 'nucl-th']]
    discipline_suffixes = [
        'ai',
        'astrophysics',
        'condensed_matter',
        'high_energy',
        'nonlinear',
        'nuclear']

    # Iterate over each year/discipline pair
    for i in range(len(all_disciplines)):
        discipline = all_disciplines[i]
        discipline_suffix = discipline_suffixes[i]
        print(discipline, discipline_suffix)

        year_to_frequencies = {year : {'num_papers' : 0} for year in years}        
        compute_overall_frequencies(year_to_frequencies, paper_to_metadata, discipline)

        for year in years:
            print(year)

            # Load most common memes for that pairing
            common_meme_savepath = os.path.join(CACHE_PATH, f'year_{year}_most_common_{discipline_suffix}.pkl')
            with open(common_meme_savepath, 'rb') as f:
                common_memes = pickle.load(f)[:2500]

            meme_to_score_components = {meme : {
                'frequency' : 0,
                'in_paper_in_citations' : 0,
                'in_citations' : 0,
                'in_paper_not_in_citations' : 0,
                'not_in_citations' : 0
            } for meme in common_memes}
            compute_n_gram_meme_score_terms(year, meme_to_articles, common_memes, meme_to_score_components,
                                            paper_to_metadata, discipline)
            save_year_discipline_meme_scores(year_to_frequencies, meme_to_score_components, year, discipline_suffix)
            

