# adapted fromhttps://github.com/synthbot-anon/sample-code/blob/main/src/cleanup_unigram_tokenizer.py 
import requests
from datasets import load_dataset
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import Unigram
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def get_token_counts(tokenizer, dataset):
    dataset_token_counts = {
        index: 0 for index in tokenizer.get_vocab().values() if index is not None
    }

    for text in dataset:
        tokens = tokenizer.encode(text).tokens
        encoding = [tokenizer.token_to_id(x) for x in tokens]
        for index in encoding:
            dataset_token_counts[index] += 1

    return dataset_token_counts


def find_rare_tokens(token_counts, threshold_fraction=None, threshold_count=None):
    if threshold_fraction is not None:
        assert (
            threshold_count is None
        ), "Either threshold_fraction or threshold_count must be provided"
        threshold_count = threshold_fraction * len(token_counts)
    else:
        assert (
            threshold_count is not None
        ), "Either threshold_fraction or threshold_count must be provided"

    useless_tokens = {
        token_id for token_id, count in token_counts.items() if count < threshold_count
    }

    return set(useless_tokens)


def find_unk_tokens(tokenizer, dataset):
    unk_tokens = set()
    for text in dataset:
        tokens = tokenizer.encode(text).tokens
        encoding = [tokenizer.token_to_id(x) for x in tokens]
        for token, token_index in zip(tokens, encoding):
            if token_index == None:
                unk_tokens.add(token)

    return unk_tokens


def create_unigram_subtokenizer(
    tokenizer, scores, remove_token_ids, ignore_tokens, unk_token_id
):
    remove_tokens = set()
    for token_id in remove_token_ids:
        if token_id in ignore_tokens:
            continue
        token = scores[token_id][0]
        remove_tokens.add(token)

    subscores = scores[:]
    for i in reversed(range(len(scores))):
        token, _ = scores[i]
        if len(token) == 1:
            continue
        if token not in remove_tokens:
            continue
        subscores[i][1] = -99

    subscores = [tuple(x) for x in subscores]

    subtokenizer_model = Unigram(vocab=subscores, unk_id=unk_token_id)
    subtokenizer = Tokenizer(subtokenizer_model)

    if tokenizer.pre_tokenizer is not None:
        subtokenizer.pre_tokenizer = tokenizer.pre_tokenizer
    if tokenizer.post_processor is not None:
        subtokenizer.post_processor = tokenizer.post_processor
    if tokenizer.normalizer is not None:
        subtokenizer.normalizer = tokenizer.normalizer
    if tokenizer.decoder is not None:
        subtokenizer.decoder = tokenizer.decoder

    return subtokenizer, subscores


def wrap_unigram_autotokenizer(tokenizer, scores):
    ignore_tokens = {
        tokenizer.unk_token_id: tokenizer.unk_token,
        tokenizer.pad_token_id: tokenizer.pad_token,
        tokenizer.eos_token_id: tokenizer.eos_token,
        tokenizer.bos_token_id: tokenizer.bos_token,
        tokenizer.mask_token_id: tokenizer.mask_token,
        tokenizer.cls_token_id: tokenizer.cls_token,
    }
    ignore_tokens = {v: k for k, v in ignore_tokens.items() if k is not None}

    unk_token_id = tokenizer.unk_token_id
    subtokenizer, scores = create_unigram_subtokenizer(
        tokenizer._tokenizer, scores, set(), ignore_tokens, unk_token_id
    )
    return subtokenizer, ignore_tokens, unk_token_id


def fix_unigram_tokenizer(tokenizer_repo, training_dataset, inference_dataset):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    tokenizer_data = requests.get(
        f"https://huggingface.co/{tokenizer_repo}/resolve/main/tokenizer.json?download=true"
    ).json()
    tokenizer_config = requests.get(
        f"https://huggingface.co/{tokenizer_repo}/resolve/main/tokenizer_config.json?download=true"
    ).json()
    scores = tokenizer_data["model"]["vocab"]

    tokenizer, ignore_tokens, unk_token_id = wrap_unigram_autotokenizer(
        tokenizer, scores
    )

    # validate the tokenizer against the datasets
    unk_tokens = find_unk_tokens(tokenizer, training_dataset)
    unk_tokens = unk_tokens.union(find_unk_tokens(tokenizer, inference_dataset))
    if unk_tokens:
        raise Exception(f"UNK token found in training dataset: {unk_tokens}")

    pony_token_counts = get_token_counts(tokenizer, training_dataset)
    generic_token_counts = get_token_counts(tokenizer, inference_dataset)
    underrepresented_tokens = find_rare_tokens(pony_token_counts, threshold_count=50)
    useless_tokens = find_rare_tokens(generic_token_counts, threshold_fraction=0.0001)

    bad_tokens = underrepresented_tokens.union(useless_tokens)
    bad_tokens = bad_tokens - set(ignore_tokens.values())

    #import pdb
    #pdb.set_trace()

    subtokenizer, subscores = create_unigram_subtokenizer(
        tokenizer, scores, bad_tokens, ignore_tokens, unk_token_id
    )

    vocab = "\n".join(list(subtokenizer.get_vocab()))
    with open("vocab.model", "w", encoding='utf-8') as f:
        f.write(vocab)

    tokenizer_config["added_tokens_decoder"] = {
        k: AddedToken(**v) for k, v in tokenizer_config["added_tokens_decoder"].items()
    }

    converted_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=subtokenizer,
        **tokenizer_config,
    )

    return converted_tokenizer

import random
from itertools import groupby
from g2p_en import G2p
import time
import re
import string

g2p = G2p()

class HorsePhonemizer:
    def __init__(self, horsewords_dictionary = 'new_horsewords.clean'):
        self.horsedict = {}
        with open(horsewords_dictionary, 'r') as f:
            while line := f.readline():
                baseword, transcription = line.split('  ')
                self.horsedict[baseword] = transcription

    def phonemize(self, text):
        """Uses g2p_en + a dictionary to convert a string into contiguous ARPAbet characters"""
        spl = text.split()
        l = ''
        for s in spl:
            s_up = s.strip().upper()
            if s_up in self.horsedict:
                arpabet = ''.join(self.horsedict[s_up].split())
                l += arpabet + ' '
            else:
                p = [arp for arp in g2p(s) if arp != ' ']
                arpabet_string = ''.join(p)
                l += arpabet_string + ' '
        return l.strip()

    def random_phonemize(self, text, prob=0.2, grow_prob=0.2, seed=0):
        """ Randomly phonemize spans of text.
        `prob` influences the base probability of an index being phonemized
        `grow_prob` adds a probability for the previous index being phonemized."""
        text = clean_spaces(text)
        # Split including words or isolated punctuation
        spl = re.findall(r'[\w\']+|[.,!?;:]', text)
        splbits = [0 for s in spl]
        idxs = list(t[0] for t in enumerate(spl))

        random.seed(seed)
        random.shuffle(idxs)

        for idx in idxs[:int(prob*len(spl))]:
            splbits[idx] = 1
            if random.random() < grow_prob:
                if idx > 0:
                    splbits[idx-1] = 1

        ret = ''

        for key, group in groupby(enumerate(splbits),
            key = lambda t: t[1] == 1):
            g = list(group)
            g = [spl[t[0]] for t in g]
            str_to_process = clean_spaces(' '.join(g))
            if key == 0:
                ret += str_to_process+' '
            else:
                ret += '{'+self.phonemize(str_to_process)+'} '

        return clean_spaces(ret)

hphzr = HorsePhonemizer()

if __name__ == "__main__":
    generics_kb = load_dataset(
        "community-datasets/generics_kb", name="generics_kb_best", split="train"
    )
    ponyspeech_dataset = load_dataset("synthbot/pony-speech", split="train")

    pony_graphemes = [x.replace("ñ", "n") for x in ponyspeech_dataset["transcription"]]

    n = 240000
    generic_graphemes = generics_kb.shuffle().select(range(n))["generic_sentence"]

    training_dataset = pony_graphemes
    inference_dataset = generic_graphemes + training_dataset

    training_dataset = [hphzr.phonemize(text) for text in training_dataset]
    inference_dataset = [hphzr.phonemize(text) for text in inference_dataset]

    clean_tokenizer = fix_unigram_tokenizer(
        "therealvul/tokenizer_g2pen_v3", training_dataset, inference_dataset
    )
    clean_tokenizer.save_pretrained("./fixed_tokenizer_g2pen")
