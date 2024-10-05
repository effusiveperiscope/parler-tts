
from transformers import AutoTokenizer 
from huggingface_hub import hf_hub_download
from g2p_en import G2p
from itertools import groupby
import numpy as np
import string
import re
import json
import random
import time

def clean_spaces(text):
    """Remove spaces before punctuation and on inside of opening brace."""
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'{\s+','{', text)
    return text

# Credit to Synthbot for pruned tokenizers
class HybridPhonemeTokenizer:
    def __init__(self,
        tokenizer_eng = 'synthbot/parlertts_tokenizer_clean',
        tokenizer_g2p = 'therealvul/g2pen_tokenizer_clean',
        eng_special = {
            'pad_token': "<pad>",
            'eos_token': "</s>",
            'unk_token': "<unk>",
        },
        g2p_special = {
            'unk_token': "[UNK]",
            'pad_token': "[PAD]",
            'cls_token': "[CLS]",
            'eos_token': "[SEP]",
            'mask_token':"[MASk]",
        },
         **kwargs):
        self.name_or_path = 'hybrid_phoneme_tokenizer'
        self.tokenizer_eng = AutoTokenizer.from_pretrained(
            tokenizer_eng, **eng_special)
        self.tokenizer_g2p = AutoTokenizer.from_pretrained(
            tokenizer_g2p, **g2p_special)
        tokenizer_eng_vocab_path = hf_hub_download(repo_id=
            tokenizer_eng, filename="tokenizer.json")
        with open(tokenizer_eng_vocab_path, encoding='utf-8') as f:
            tokenizer_eng_scores = json.load(f)["model"]["vocab"]

        # To avoid expanding vocab size and changing embedding length,
        # we re-map g2p IDs to disabled IDs in the english tokenizer

        # maps from g2p id to external id
        self.g2p_to_ext = list()
        # maps from external id to g2p id
        self.ext_to_g2p = dict()
        for i,t in enumerate(tokenizer_eng_scores):
            token, score = t
            if score == -99.0:
                self.g2p_to_ext.append(i)
                self.ext_to_g2p[i] = (len(
                    self.g2p_to_ext) - 1)

        # The vocab size of the g2p tokenizer must be smaller or equal to
        # the number of disabled tokens in the eng tokenizer
        assert len(self.tokenizer_g2p.get_vocab()) < len(self.g2p_to_ext)

        # Not sure if this is actually necessary - ByteLevel pretokenizer
        # removes possibility of <unk> tokens
        self.special_tokens = {
            self.tokenizer_g2p.pad_token_id: self.tokenizer_eng.pad_token_id,
            self.tokenizer_g2p.bos_token_id: self.tokenizer_eng.bos_token_id,
            self.tokenizer_g2p.cls_token_id: self.tokenizer_eng.cls_token_id,
            self.tokenizer_g2p.eos_token_id: self.tokenizer_eng.eos_token_id,
            self.tokenizer_g2p.unk_token_id: self.tokenizer_eng.unk_token_id,
            self.tokenizer_g2p.mask_token_id: self.tokenizer_eng.mask_token_id
        }
        self.pad_token_id = self.tokenizer_eng.pad_token_id
        self.bos_token_id = self.tokenizer_eng.bos_token_id
        self.eos_token_id = self.tokenizer_eng.eos_token_id

    def ext_is_g2p_id(self, id):
        return id in self.ext_to_g2p

    def ext_to_g2p_id(self, id):
        return self.ext_to_g2p[id]

    def g2p_to_ext_id(self, id):
        return self.g2p_to_ext[id]

    def preprocess(self, text):
        # Replace multiple spaces with one space
        # And replace ñ with n
        text = re.sub(r'\s+', ' ', text).replace('ñ', 'n')
        return text

    def max_vocab_length(self):
        return len(self.tokenizer_eng.get_vocab())

    def __call__(self, text):
        text = self.preprocess(text)
        parts = re.split(r'({.*?})', text)
        result = []
        for i, part in enumerate(parts):
            if not len(part):
                continue
            part = part.strip()
            if not (part.startswith('{') and part.endswith('}')):
                ids = self.tokenizer_eng(part, add_special_tokens=False)['input_ids']
                result += [i for i in ids]
            else:
                ids = self.tokenizer_g2p(part[1:-1])['input_ids']
                for i,id in enumerate(ids):
                    if id in self.special_tokens:
                        ids[i] = self.special_tokens[id]
                    else:
                        ids[i] = self.g2p_to_ext_id(id)
                result += [i for i in ids]
        return {'input_ids': result, 'attention_mask': list(np.ones_like(result))}

    # Returns string constructed from decoded tokens with space handling
    def _list_decode(self, input_ids, skip_special_tokens=False):
        decode_args = {
            'clean_up_tokenization_spaces': True,
            'skip_special_tokens': skip_special_tokens
        }
        output = ''
        for isg2p, group in groupby(input_ids,
            key=lambda x: self.ext_is_g2p_id(x)):
            g = list(group)
            if isg2p:
                if len(output) == 0 or output[-1] != ' ':
                    output += ' '
                output += '{'
                output += self.tokenizer_g2p.decode(
                    [self.ext_to_g2p_id(i) for i in g],
                     **decode_args)
                output += '}'
            else:
                decoded = self.tokenizer_eng.decode(
                    g, **decode_args)
                if len(output) and output[-1] == '}':
                    if len(decoded) and not decoded[0] in string.punctuation:
                        output += ' '
                output += decoded
        return clean_spaces(output.strip())

    # Returns list of string tokens with no space handling
    def _decode_tokens(self, input_ids, skip_special_tokens=False):
        toks = []
        for isg2p, group in groupby(input_ids,
            key=lambda x: self.ext_is_g2p_id(x)):
            g = list(group)
            if isg2p:
                toks.extend(
                    [self.tokenizer_g2p.decode(
                        self.ext_to_g2p_id(i)) for i in g])
            else:
                toks.extend([self.tokenizer_eng.decode(i) for i in g])
        return toks
    
    def batch_decode(self, input_ids, skip_special_tokens=False):
        if not isinstance(input_ids[0], list):
            return self._decode_tokens(input_ids)

        return [self._list_decode(l, skip_special_tokens) for l in input_ids]

    

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

    def random_phonemize(self, text, prob=0.2, grow_prob=0.2, seed=None):
        """ Randomly phonemize spans of text.
        `prob` influences the base probability of an index being phonemized
        `grow_prob` adds a probability for the previous index being phonemized."""
        text = clean_spaces(text)
        # Split including words or isolated punctuation
        spl = re.findall(r'[\w\']+|[.,!?;:]', text)
        splbits = [0 for s in spl]
        idxs = list(t[0] for t in enumerate(spl))

        if seed is not None:
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

horse_phonemizer = HorsePhonemizer()