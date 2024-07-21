import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
#import cv2 as cv
from imageio import imread
from random import *

class LEVIRCCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split, token_folder = None, vocab_file = None, max_length = 41, allow_unk = 0, max_iters=None):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.mean = [0.39073*255,  0.38623*255, 0.32989*255]
        self.std = [0.15329*255,  0.14628*255, 0.13648*255]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
        self.files = []
        if split =='train':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name.split('-')[0])
                img_fileB = img_fileA.replace('A', 'B')

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('A', 'label'))

                if '-' in name:
                    token_id = name.split('-')[-1]
                else:
                    token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('-')[0]
                })
        elif split =='val':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('A', 'B')

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('A', 'label'))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
        elif split =='test':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
                img_fileB = img_fileA.replace('A', 'B')

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace('A', 'label'))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
                else:
                    token_file = None
                self.files.append({
                    "imgA": imgA,
                    "imgB": imgB,
                    "seg_label": seg_label,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = datafiles["imgA"]
        imgB = datafiles["imgB"]
        seg_label = datafiles["seg_label"]

        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        seg_label = seg_label.transpose(2, 0, 1)[0]
        seg_label[seg_label==255] = 2
        seg_label[seg_label==128] = 1


        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]      
        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)

            #token = np.zeros((1, self.max_length), dtype=int)
            #j = randint(0, len(caption_list) - 1)
            #tokens_encode = encode(caption_list[j], self.word_vocab,
            #            allow_unk=self.allow_unk == 1)
            #token[0, :len(tokens_encode)] = tokens_encode
            #token_len = len(tokens_encode)

            token_all = np.zeros((len(caption_list),self.max_length),dtype=int)
            token_all_len = np.zeros((len(caption_list),1),dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
                if self.split == 'train' and nochange_cap in caption_list:
                    tokens = nochange_cap
                tokens_encode = encode(tokens, self.word_vocab,
                                    allow_unk=self.allow_unk == 1)
                token_all[j,:len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        return imgA.copy(), imgB.copy(), seg_label.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name
