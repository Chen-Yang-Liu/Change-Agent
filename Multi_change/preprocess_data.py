import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type = str, default = 'LEVIR_MCI', help= 'the name of the dataset')
parser.add_argument('--word_count_threshold', default=5, type=int)

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}
DATA_PATH_ROOT = 'xxxx'
def main(args):
    if args.dataset == 'LEVIR_MCI':
        input_captions_json = f'{DATA_PATH_ROOT}\Levir-MCI-dataset\LevirCCcaptions.json'
        input_image_dir = f'{DATA_PATH_ROOT}\Levir-MCI-dataset\images'
        input_vocab_json = ''
        output_vocab_json = 'vocab.json'
        save_dir = './data/LEVIR_MCI/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir + 'tokens/')):
        os.makedirs(os.path.join(save_dir + 'tokens/'))
    print('Loading captions')
    assert args.dataset in {'LEVIR_MCI'}

    if args.dataset == 'LEVIR_MCI':
        with open(input_captions_json, 'r') as f:
            data = json.load(f)
        # Read image paths and captions for each image
        max_length = -1
        all_cap_tokens = []
        for img in data['images']:
            captions = []    
            for c in img['sentences']:
                # Update word frequency
                assert len(c['raw']) > 0, 'error: some image has no caption'
                captions.append(c['raw'])
            tokens_list = []
            for cap in captions:
                cap_tokens = tokenize(cap,
                                    add_start_token=True,
                                    add_end_token=True,
                                    punct_to_keep=[';', ','],
                                    punct_to_remove=['?', '.'])
                tokens_list.append(cap_tokens)
                max_length = max(max_length, len(cap_tokens))
            all_cap_tokens.append((img['filename'], tokens_list))

        # Then save the tokenized captions in txt
        print('Saving captions')
        for img, tokens_list in all_cap_tokens:
            i = img.split('.')[0]
            token_len = len(tokens_list)
            tokens_list = json.dumps(tokens_list)
            f = open(os.path.join(save_dir + 'tokens/' + i + '.txt'), 'w')
            f.write(tokens_list)
            f.close()


        #Considering each image pair has 5 annotations, two strategies can be adopted to generate list for training:
        # a: creating training list with a self-defined token_id[0:4], each token list corresponds to specific captions;
        # or b: randomly select one of the five captions during training;
          
            if i.split('_')[0] == 'train':
               f = open(os.path.join(save_dir + 'train' + '.txt'), 'a')
               f.write(img + '\n')
               f.close

            # if i.split('_')[0] == 'train':
            #     f = open(os.path.join(save_dir + 'train' + '.txt'), 'a')
            #     for j in range(token_len):
            #         f.write(img + '-' + str(j) + '\n')
            #     f.close

            elif i.split('_')[0] == 'val':
                f = open(os.path.join(save_dir + 'val' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()
            elif i.split('_')[0] == 'test':
                f = open(os.path.join(save_dir + 'test' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()

    print('max_length of the dataset:', max_length)
    # Either create the vocab or load it from disk
    if input_vocab_json == '':
        print('Building vocab')
        word_freq = build_vocab(all_cap_tokens, args.word_count_threshold)
    else:
        print('Loading vocab')
        with open(input_vocab_json, 'r') as f:
            word_freq = json.load(f)
    if output_vocab_json != '':
        with open(os.path.join(save_dir + output_vocab_json), 'w') as f:
            json.dump(word_freq, f)


def tokenize(s, delim=' ',add_start_token=True, 
    add_end_token=True, punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    for q in tokens:
        if q == '':
            tokens.remove(q)
    if tokens[0] == '':
        tokens.remove(tokens[0])
    if tokens[-1] == '':
        tokens.remove(tokens[-1])
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, min_token_count=1):#Calculate the number of independent words and tokenize vocab
    token_to_count = {}
    for it in sequences:
        for seq in it[1]:
            for token in seq:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count > min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
