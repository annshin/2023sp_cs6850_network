import os
import json
import pickle
import argparse
import string
from tqdm import tqdm
from datetime import date
import spacy

SCRATCH_PATH = '/scratch/datasets/aw588/'
UNARXIVE_PATH = SCRATCH_PATH + "unarXive"
CACHE_PATH = '/scratch/datasets/mog29/unarXive'
VALID_DISCIPLINES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]
INTRO_SUBSTRING = 'introduction'
RW_SUBSTRING = 'related'
METHOD_SUBSTRINGS = ['method', 'model', 'approach']

MAX_NGRAM_SIZE = 5

def get_args():
    parser = argparse.ArgumentParser(description="Getting trends in the data")
    parser.add_argument('--start_index', type=int,
                        help="If set, we will slice from the provided index (inclusive)")
    parser.add_argument('--end_index', type=int,
                        help="If set, we will end the slice at the given index (non-inclusive)")
    args = parser.parse_args()
    return args

def read_jsonl(filepath):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def get_cited_papers(paper):
    cited_papers = []
    bib_entries = paper['bib_entries']
    for _, ref_data in bib_entries.items():
        if 'ids' not in ref_data:
            continue
        elif 'arxiv_id' not in ref_data['ids']:
            continue
        else:
            ref_paper_id = ref_data['ids']['arxiv_id']
            if ref_paper_id == "":
                continue

            cited_papers.append(ref_paper_id)

    return cited_papers

def process_text_basic(input_text):
    lowercase_text = input_text.lower()
    replaced_text = lowercase_text.replace('\n', ' ')
    return replaced_text

def add_paper_metadata(paper, paper_id, paper_to_metadata):
    if 'categories' not in paper['metadata']:
        paper_categories = []
    else:
        paper_categories = paper['metadata']['categories'].split(' ')

    # Get paper date
    version_dates = paper['metadata']['versions']
    v1 = [version for version in version_dates if version['version'] == 'v1'][0]['created']
    day_month_year = v1.split(' ')[1:4]

    # Get the papers the paper cites
    cited_papers = get_cited_papers(paper)
    paper_to_metadata[paper_id] = {
        "categories" : paper_categories,
        "release_date" : day_month_year,
        "cited_papers" : cited_papers
    }

def get_lemmatized_text(text, tokenizer):
    # First process the text normally
    text = text.lower().replace('\n', ' ')
    text = remove_bracketed_sections(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split(' ')
    tokens = [token for token in tokens if token != '']

    return tokens

def remove_bracketed_sections(text):
    new_text = ""
    num_brackets = 0

    for character in text:
        add_character = True
        if character == "{":
            num_brackets += 1
            add_character = False
        elif character == "}":
            num_brackets -= 1
            add_character = False
        elif num_brackets > 0:
            add_character = False

        if add_character:
            new_text += character

    return new_text

            
def get_paper_n_grams(paper, paper_id, n_gram_to_papers, tokenizer):
    # Get the lemmatized tokens of each relevant section
    title = [get_lemmatized_text(paper['metadata']['title'], tokenizer)]
    abstract = [get_lemmatized_text(paper['abstract']['text'], tokenizer)]
    intro = get_lemmatized_intro_texts(paper['body_text'], tokenizer)
    method = get_lemmatized_method_texts(paper['body_text'], tokenizer)
    collected_tokens = title + abstract + intro + method

    # Get n_grams from within this collection
    add_memes(n_gram_to_papers, paper_id, collected_tokens)

def get_lemmatized_intro_texts(body_text, tokenizer):
    intro = []
    for body in body_text:
        section_name = body['section']
        if section_name is None or section_name in ["", " "]:
            continue

        if INTRO_SUBSTRING in process_text_basic(section_name):
            intro.append(get_lemmatized_text(body['text'], tokenizer))

    return intro

def get_section_number(body_text, section_substring):
    for body in body_text:
        section_name = body['section']
        section_num = body['sec_number']
        if section_name is None or section_name in ["", " "]:
            continue
        if section_num is None or section_num in ["", " "]:
            continue

        if section_substring in process_text_basic(section_name):
            return section_num.split(".")[:1][0]

    return None

def get_lemmatized_method_texts(body_text, tokenizer):
    method = []
    for body in body_text:
        section_name = body['section']
        if section_name is None or section_name in ["", " "]:
            continue

        for method_substring in METHOD_SUBSTRINGS:
            if method_substring in process_text_basic(section_name):
                method.append(get_lemmatized_text(body['text'], tokenizer))

    return method

def get_method_numbers(body_text):
    method_numbers = set()

    # Heuristic 1: The methods follow the intro and rw (if they are in sequence)
    intro_number = get_section_number(body_text, INTRO_SUBSTRING)
    rw_number = get_section_number(body_text, RW_SUBSTRING)
    try:
        intro_number = int(intro_number)
        if rw_number is None:
            method_numbers.add(intro_number + 1)
        else:
            rw_number = int(rw_number)
            if rw_number != intro_number + 1:
                method_numbers.add(intro_number + 1)
            else:
                method_numbers.add(rw_number + 1)
    except:
        pass

    # Heuristic 2: Add all sections containing preselected substrings
    for body in body_text:
        section_name = body['section']
        section_num = body['sec_number']
        if section_name is None or section_name in ["", " "]:
            continue
        if section_num is None or section_num in ["", " "]:
            continue

        for method_substring in METHOD_SUBSTRINGS:
            if method_substring in process_text_basic(section_name):
                method_numbers.add(section_num.split(".")[:1][0])

    return method_numbers

def add_memes(n_gram_to_papers, paper_id, collected_tokens):
    meme_to_following_word = {}
    for n in range(MAX_NGRAM_SIZE, 0, -1):
        for token_list in collected_tokens:
            for i in range(len(token_list) - n + 1):
                # Construct meme for the span
                meme_span = ' '.join(token_list[i:i+n])

                # Get the word following the meme
                if meme_span not in meme_to_following_word:
                    meme_to_following_word[meme_span] = set()
                if i == (len(token_list) - n):
                    meme_to_following_word[meme_span].add("N/A")
                else:
                    meme_to_following_word[meme_span].add(token_list[i+n])

    # Add the memes that appear in varying contexts
    for meme, contexts in meme_to_following_word.items():
        add_meme = len(contexts) > 1 or "N/A" in contexts
        if add_meme:
            if meme not in n_gram_to_papers:
                n_gram_to_papers[meme] = set()
            n_gram_to_papers[meme].add(paper_id)

def get_json_n_grams(json_data, paper_to_metadata, n_gram_to_papers,
                     tokenizer):
    for paper in json_data:
        if 'title' not in paper['metadata'] or 'abstract' not in paper:
            continue
        paper_id = paper['paper_id']
        add_paper_metadata(paper, paper_id, paper_to_metadata)
        get_paper_n_grams(paper, paper_id, n_gram_to_papers, tokenizer)

if __name__ == "__main__":
    args = get_args()
    start_index = 0 if args.start_index is None else args.start_index
    end_index = len(years) if args.end_index is None else args.end_index

    years = [str(i)[2:] for i in range(1991, 2023)]
    years = years[start_index:end_index]

    tokenizer = spacy.load("en_core_web_sm")

    # Iterate over each year
    for year in years:
        paper_to_metadata = {}
        n_gram_to_papers = {}

        if ".tar.xz" in year:
            continue
        year_folder = os.path.join(UNARXIVE_PATH, year)
        year_jsons = sorted(os.listdir(year_folder))

        # Iterate over each json in said year
        print(year)
        for year_json in tqdm(year_jsons):
            # Read the json
            full_path = os.path.join(year_folder, year_json)
            json_data = read_jsonl(full_path)
        
            get_json_n_grams(json_data, paper_to_metadata, n_gram_to_papers, tokenizer)

        n_gram_filename = os.path.join(CACHE_PATH, f'n_gram_to_papers_{year}.pkl')
        with open(n_gram_filename, 'wb') as f:
            pickle.dump(n_gram_to_papers, f)

        metadata_filename = os.path.join(CACHE_PATH, f'paper_to_metadata_{year}.pkl')
        with open(metadata_filename, 'wb') as f:
            pickle.dump(paper_to_metadata, f)



