import os
import json
import pickle
import argparse
import string
from tqdm import tqdm
from datetime import date

SCRATCH_PATH = '/scratch/datasets/aw588/'
UNARXIVE_PATH = SCRATCH_PATH + "unarXive"
CHECKPOINT_PATH = '/scratch/datasets/mog29/unarXive_ngrams'
VALID_DISCIPLINES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]
MAX_NGRAM_SIZE = 5


def get_args():
    parser = argparse.ArgumentParser(description="Getting trends in the data")
    parser.add_argument('--load_meme2pubs', action='store_true',
                        help="If set, will load meme_to_publications from a checkpoint")
    parser.add_argument('--load_meme2unseenpubs', action='store_true',
                        help="If set, will load meme_to_publications from a checkpoint")
    parser.add_argument('--split_num', type=int,
                        help="Which quartile of papers to compute scores for")
    args = parser.parse_args()
    return args

def read_jsonl(filepath):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def get_memes_in_json(data, meme_to_publications):
    # Iterate over each paper
    for paper in data:
        # Decide whether to skip paper first
        if 'categories' not in paper['metadata']:
            continue
        else:
            paper_cats = paper['metadata']['categories'].split(' ')
            found_match = any([cat in VALID_DISCIPLINES for cat in paper_cats])
            if not found_match:
                continue
        if 'title' not in paper['metadata'] or 'abstract' not in paper:
            continue

        # Extract the n_grams associated with the paper
        extract_paper_memes(paper, meme_to_publications)

def extract_paper_memes(paper, meme_to_publications):
    paper_id = paper['paper_id']

    # Extract title memes
    processed_title = process_text(paper['metadata']['title'])
    add_memes(paper_id, processed_title, meme_to_publications) 

    processed_abstract = process_text(paper['abstract']['text'])
    add_memes(paper_id, processed_abstract, meme_to_publications)

def process_text(text):
    # Set to lowercase; remove newlines and punctuation
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split(' ')

def add_memes(paper_id, all_tokens, meme_to_publications):
    meme_to_following_word = {}
    for n in range(MAX_NGRAM_SIZE, 0, -1):
        for i in range(len(all_tokens) - n + 1):
            # Construct meme for the span
            meme_span = ' '.join(all_tokens[i:i+n])
            
            # Get the word following the meme
            if meme_span not in meme_to_following_word:
                meme_to_following_word[meme_span] = set()
            if i == (len(all_tokens) - n):
                meme_to_following_word[meme_span].add("N/A")
            else:
                meme_to_following_word[meme_span].add(all_tokens[i+n])

    # Add the memes that appear in varying contexts
    for meme, contexts in meme_to_following_word.items():
        add_meme = len(contexts) > 1 or "N/A" in contexts
        if add_meme:
            if meme not in meme_to_publications:
                meme_to_publications[meme] = set()
            meme_to_publications[meme].add(paper_id)

def get_meme_to_publication(years):
    meme_to_publications = {}
    for year in tqdm(years):
        if ".tar.xz" in year:
            continue
        year_folder = os.path.join(UNARXIVE_PATH, year)
        year_jsons = sorted(os.listdir(year_folder))
        
        # Iterate over each json in said year in order
        for year_json in year_jsons:
            # Read the json
            full_path = os.path.join(year_folder, year_json)
            json_data = read_jsonl(full_path)

            # Iterate over each paper within the json
            get_memes_in_json(json_data, meme_to_publications)

    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'checkpoint.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(meme_to_publications, f)

    return meme_to_publications


def get_all_pubs(meme_to_publications):
    all_pubs = set()
    for meme, pubs in meme_to_publications.items():
        for pub in pubs:
            all_pubs.add(pub)
    return all_pubs

def get_all_unseen_pubs(years, set_of_pubs):
    unseen_pubs = set()
    paper_to_metadata = {}
    for year in tqdm(years):
        if ".tar.xz" in year:
            continue
        year_folder = os.path.join(UNARXIVE_PATH, year)
        year_jsons = sorted(os.listdir(year_folder))
        
        # Iterate over each json in said year in order
        for year_json in year_jsons:
            # Read the json
            full_path = os.path.join(year_folder, year_json)
            json_data = read_jsonl(full_path)

            # Iterate over each paper within the json
            get_unseen_pubs_in_json(json_data, unseen_pubs,
                                    set_of_pubs, paper_to_metadata)


    metadata_path = os.path.join(CHECKPOINT_PATH, 'paper_to_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(paper_to_metadata, f)

    return unseen_pubs, paper_to_metadata

def get_unseen_pubs_in_json(json_data, unseen_pubs,
                            set_of_pubs, paper_to_metadata):
    for paper in json_data:
        # Decide whether to skip paper first
        if 'categories' not in paper['metadata']:
            continue
        else:
            paper_cats = paper['metadata']['categories'].split(' ')
            found_match = any([cat in VALID_DISCIPLINES for cat in paper_cats])
            if not found_match:
                continue
        if 'title' not in paper['metadata'] or 'abstract' not in paper:
            continue

        # Get paper metadata
        paper_id = paper['paper_id']
        paper_to_metadata[paper_id] = {}

        # Record paper date
        version_dates = paper['metadata']['versions']
        v1 = [version for version in version_dates if version['version'] == 'v1'][0]['created']
        day, month, year = v1.split(' ')[1:4]
        paper_to_metadata[paper_id]["data"] = [day, month, year]

        # Get the list of cited paper ids
        cited_papers = []
        bib_entries = paper['bib_entries']
        for _, ref_data in bib_entries.items():
            if 'ids' not in ref_data:
                continue
            elif 'arxiv_id' not in ref_data['ids']:
                continue
            else:
                ref_paper_id = ref_data['ids']['arxiv_id']
                if ref_paper_id == '':
                    continue

                cited_papers.append(ref_paper_id)
                if ref_paper_id not in set_of_pubs:
                    unseen_pubs.add(ref_paper_id)

        paper_to_metadata[paper_id]["cited_ids"] = cited_papers
            
def get_meme_to_unseen_pubs(years, set_of_unseen_pubs):
    meme_to_unseen_pubs = {}
    for year in tqdm(years):
        if ".tar.xz" in year:
            continue
        year_folder = os.path.join(UNARXIVE_PATH, year)
        year_jsons = sorted(os.listdir(year_folder))
        
        # Iterate over each json in said year in order
        for year_json in year_jsons:
            # Read the json
            full_path = os.path.join(year_folder, year_json)
            json_data = read_jsonl(full_path)

            # Iterate over each paper within the json
            get_unseen_memes_in_json(json_data, meme_to_unseen_pubs,
                                     set_of_unseen_pubs)

    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'unseen_checkpoint.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(meme_to_unseen_pubs, f)

    return meme_to_unseen_pubs

def get_unseen_memes_in_json(json_data, meme_to_unseen_pubs,
                             set_of_unseen_pubs):
    # Iterate over each paper
    for paper in json_data:
        paper_id = paper['paper_id']
        if paper_id not in set_of_unseen_pubs:
            continue

        extract_paper_memes(paper, meme_to_unseen_pubs)

def compute_meme_scores(years, meme_to_publications, meme_to_unseen_pubs,
                        paper_to_metadata, split_num):
    set_of_pubs = get_all_pubs(meme_to_publications)
    #set_of_pubs = get_all_pubs(meme_to_publications)
    #num_pubs = len(set_of_pubs)
    #pub_interval = num_pubs // 4
    #set_of_pubs = set(set_of_pubs[split_num*pub_interval:(split_num+1)*pub_interval])
    
    year_to_meme_scores = {year : {'num_papers' : 0} for year in years}
    year_to_meme_scores["overall"] = {'num_papers' : 0}
    
    # Iterate over each meme and first get the values for the meme scores
    for meme, meme_publications in tqdm(meme_to_publications.items()):
        for pub in meme_publications:
            pub_year = paper_to_metadata[pub]["data"][-1][2:]
            citations = paper_to_metadata[pub]["cited_ids"]
            
            meme_in_paper = is_meme_in_paper(pub, meme, meme_to_publications)
            meme_in_citations = is_meme_in_citations(citations, meme,
                                                     meme_to_publications,
                                                     meme_to_unseen_pubs)
            
            for key in [pub_year, 'overall']:
                if meme not in year_to_meme_scores[key]:
                    year_to_meme_scores[key][meme] = {
                        'frequency' : 0,
                        'in_paper_in_citations' : 0,
                        'in_paper_not_in_citations' : 0,
                        'in_citations' : 0,
                        'not_in_citations' : 0,
                        'score' : 0
                    }

                if meme_in_paper:
                    year_to_meme_scores[key][meme]['frequency'] += 1
                    if meme_in_citations:
                        year_to_meme_scores[key][meme]['in_paper_in_citations'] += 1
                        year_to_meme_scores[key][meme]['in_citations'] += 1                        
                    else:
                        year_to_meme_scores[key][meme]['in_paper_not_in_citations'] += 1
                        year_to_meme_scores[key][meme]['not_in_citations'] += 1                        
                else:
                    if meme_in_citations:
                        year_to_meme_scores[key][meme]['in_citations'] += 1
                    else:
                        year_to_meme_scores[key][meme]['not_in_citations'] += 1

    # Iterate over publications to determine frequencies
    for pub in set_of_pubs:
        pub_year = paper_to_metadata[pub]["data"][-1][2:]
        for key in [pub_year, 'overall']:
            if key not in year_to_meme_scores:
                year_to_meme_scores[key]['num_papers'] = 0
            year_to_meme_scores[key]['num_papers'] += 1

    # Compute the meme score
    for year, year_dict in year_to_meme_scores.items():
        num_papers = year_dict['num_papers']
        for meme, meme_info in year_dict.items():
            if meme == "num_papers":
                continue

            frequency_score = meme_info['frequency'] / num_papers
            sticking_score = meme_info['in_paper_in_citations'] / (meme_info['in_citations'] + 3)
            sparking_score = (meme_info['in_paper_not_in_citations'] + 3) / (meme_info['not_in_citations'] + 3)
            meme_score = frequency_score * sticking_score / sparking_score
            meme_info["score"] = meme_score

    savepath = os.path.join(CHECKPOINT_PATH, f'meme_scores.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(year_to_meme_scores, f)

def is_meme_in_paper(pub, meme, meme_to_publications):
    if meme not in meme_to_publications:
        return False
    citing_papers = meme_to_publications[meme]
    return pub in citing_papers

def is_meme_in_citations(citations, meme, meme_to_publications, meme_to_unseen_pubs):
    truth_vals = []
    for pub in citations:
        truth_vals.append(is_meme_in_paper(pub, meme, meme_to_publications))
        truth_vals.append(is_meme_in_paper(pub, meme, meme_to_unseen_pubs))

    return any(truth_vals)


if __name__ == "__main__":
    args = get_args()
    years = [str(i)[2:] for i in range(1991, 2023)]

    # First get a mapping from a meme to a list of publications featuring it
    if not args.load_meme2pubs:
        meme_to_publications = get_meme_to_publication(years)
    else:
        meme2pubs_savepath = os.path.join(CHECKPOINT_PATH, 'checkpoint.pkl')
        with open(meme2pubs_savepath, 'rb') as f:
            meme_to_publications = pickle.load(f)

    # Next repeat the process for publications that were cited but do not belong
    if not args.load_meme2unseenpubs:
        set_of_pubs = get_all_pubs(meme_to_publications)
        set_of_unseen_pubs, paper_to_metadata = get_all_unseen_pubs(years, set_of_pubs)
        meme_to_unseen_pubs = get_meme_to_unseen_pubs(years, set_of_unseen_pubs)
    else:
        meme2pubs_savepath = os.path.join(CHECKPOINT_PATH, 'unseen_checkpoint.pkl')
        with open(meme2pubs_savepath, 'rb') as f:
            meme_to_unseen_pubs = pickle.load(f)

        paper2metadata_savepath = os.path.join(CHECKPOINT_PATH, 'paper_to_metadata.pkl')
        with open(paper2metadata_savepath, 'rb') as f:
            paper_to_metadata = pickle.load(f)

    # Next, iterate over memes to compute their meme scores
    compute_meme_scores(years, meme_to_publications, meme_to_unseen_pubs,
                        paper_to_metadata, args.split_num)


