import os
import json
import pickle
import argparse
import string
from tqdm import tqdm
from datetime import date
import numpy as np

SCRATCH_PATH = '/scratch/datasets/aw588/'
UNARXIVE_PATH = SCRATCH_PATH + "unarXive"
CACHE_PATH = '/scratch/datasets/mog29/unarXive'
VALID_DISCIPLINES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]
INTRO_SUBSTRING = 'introduction'
RW_SUBSTRING = 'related'
METHOD_SUBSTRINGS = ['method', 'model', 'approach']

MAX_NGRAM_SIZE = 5

if __name__ == "__main__":
    years = [str(i)[2:] for i in range(1991, 2023)]

    # Iterate over all years
    all_paper_ids = set()
    meme_to_articles = {}
    for year in tqdm(years):
        year_meme_path = os.path.join(CACHE_PATH, f'n_gram_to_papers_{year}.pkl')
        with open(year_meme_path, 'rb') as f:
            year_memes = pickle.load(f)

        for meme, containing_docs in year_memes.items():
            if meme not in meme_to_articles:
                meme_to_articles[meme] = set()
            for containing_doc in containing_docs:
                meme_to_articles[meme].add(containing_doc)
                all_paper_ids.add(containing_doc)

    # Compute idf
    num_articles = len(all_paper_ids)
    meme_to_idf = {}
    for meme, containing_docs in meme_to_articles.items():
        num_meme_articles = len(containing_docs) + 1
        meme_idf = np.log10(num_articles / num_meme_articles)
        meme_to_idf[meme] = meme_idf

    idf_path = os.path.join(CACHE_PATH, 'meme_idf.pkl')
    with open(idf_path, 'wb') as f:
        pickle.dump(meme_to_idf, f)
