import sys

# 添加特定路径到 Python 解释器的搜索路径中
# sys.path.append('F:\LCY\Change_Agent\Change-Agent-git\Multi_change')
import os.path

import cv2
import torch.optim
import argparse
import json

from skimage import measure

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from imageio.v2 import imread


# compute_change_map(path_A, path_B)函数: 生成一个掩膜mask用来表示两个图像之间的变化区域
'''
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    change_map: 变化区域的掩膜
'''
# def compute_change_mask(path_A, path_B):
#     import cv2
#     import numpy as np
#     img_A = cv2.imread(path_A)
#     img_B = cv2.imread(path_B)
#     change_map = (img_B-img_A).astype(np.uint8)
#     # 阈值化
#     change_map = cv2.cvtColor(change_map, cv2.COLOR_BGR2GRAY)
#     change_map = cv2.threshold(change_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     cv2.imwrite('E:\change_map.png', change_map)
#     return 'I have save the changed mask in E:\change_map.png'

# compute_change_caption(path_A, path_B)函数：生成一个文本用于描述两个图像之间变化
'''
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    caption: 变化描述文本
'''
class Change_Perception(object):
    def define_args(self):


        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        print(script_dir)
        parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

        parser.add_argument('--data_folder', default='D:\Dataset\Caption\change_caption\Levir-MCI-dataset\images',
                            help='folder with data files')
        parser.add_argument('--list_path', default='F:\LCY\Change_Agent\Change-Agent-git\Multi_change\data\LEVIR_MCI/',
                            help='path of the data lists')
        parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
        parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')

        # inference
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
        parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth',help='path to checkpoint')
        parser.add_argument('--result_path', default="./predict_result/",
                            help='path to save the result of masks and captions')

        # backbone parameters
        parser.add_argument('--network', default='segformer-mit_b1',
                            help='define the backbone encoder to extract features')
        parser.add_argument('--encoder_dim', type=int, default=512,
                            help='the dimension of extracted features using backbone ')
        parser.add_argument('--feat_size', type=int, default=16,
                            help='define the output size of encoder to extract features')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        # Model parameters
        parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
        parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
        parser.add_argument('--decoder_n_layers', type=int, default=1)
        parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')

        args = parser.parse_args()

        return args

    def __init__(self,):
        """
        Training and validation.
        """
        args = self.define_args()
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)
        # Load checkpoint
        snapshot_full_path = args.checkpoint

        checkpoint = torch.load(snapshot_full_path)
        self.encoder = Encoder(args.network)
        self.encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
                                         feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                         heads=args.n_heads, dropout=args.dropout)
        self.decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                     vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                     word_vocab=self.word_vocab, n_head=args.n_heads,
                                     n_layers=args.decoder_n_layers, dropout=args.dropout)

        self.encoder.load_state_dict(checkpoint['encoder_dict'])
        self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
        self.decoder.load_state_dict(checkpoint['decoder_dict'])
        # Move to GPU, if available
        self.encoder.eval()
        self.encoder = self.encoder.cuda()
        self.encoder_trans.eval()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder.eval()
        self.decoder = self.decoder.cuda()


    def preprocess(self, path_A, path_B):

        imgA = imread(path_A)
        imgB = imread(path_B)
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        if imgA.shape[1] != 256 or imgA.shape[2] != 256:
            imgA = cv2.resize(imgA, (256, 256))
            imgB = cv2.resize(imgB, (256, 256))

        imgA = torch.FloatTensor(imgA)
        imgB = torch.FloatTensor(imgB)
        imgA = imgA.unsqueeze(0)  # (1, 3, 256, 256)
        imgB = imgB.unsqueeze(0)

        return imgA, imgB

    def generate_change_caption(self, path_A, path_B):
        print('model_infer_change_captioning: start')
        imgA, imgB = self.preprocess(path_A, path_B)
        # Move to GPU, if available
        imgA = imgA.cuda()
        imgB = imgB.cuda()
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        seq = self.decoder.sample(feat1, feat2, k=1)
        pred_seq = [w for w in seq if w not in {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}]
        pred_caption = ""
        for i in pred_seq:
            pred_caption += (list(self.word_vocab.keys())[i]) + " "

        caption ='there is road change'
        caption = pred_caption
        print('change captioning:', caption)
        return caption

    def change_detection(self, path_A, path_B, savepath_mask):
        print('model_infer_change_detection: start')
        imgA, imgB = self.preprocess(path_A, path_B)
        # Move to GPU, if available
        imgA = imgA.cuda()
        imgB = imgB.cuda()
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        # for segmentation
        pred_seg = seg_pre.data.cpu().numpy()
        pred_seg = np.argmax(pred_seg, axis=1)
        # 保存图片
        pred = pred_seg[0].astype(np.uint8)
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = [0, 255, 255]
        pred_rgb[pred == 2] = [0, 0, 255]

        cv2.imwrite(savepath_mask, pred_rgb)
        print('model_infer: mask saved in', savepath_mask)

        print('model_infer_change_detection: end')
        return pred # (256,256,3)
        # return 'change detection successfully. '

    def compute_object_num(self, changed_mask, object):
        print("compute num start")
        # compute the number of connected components
        mask = changed_mask
        mask_cp = 0 * mask.copy()
        if object == 'road':
            mask_cp[mask == 1] = 255
        elif object == 'building':
            mask_cp[mask == 2] = 255
        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)
        # get bboxes by a for loop
        bboxes = []
        for prop in props:
            # print('Found bbox', prop.bbox, 'area:', prop.area)
            if prop.area > 5:
                bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
        num = len(bboxes)
        # visual
        # mask_array_copy = mask.copy()*255
        # for bbox in bboxes:
        #     print('Found bbox', bbox)
        #     cv2.rectangle(mask_array_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), 2)
        # cv2.namedWindow('findCorners', 0)
        # cv2.resizeWindow('findCorners', 600, 600)
        # cv2.imshow('findCorners', mask_array_copy)
        # cv2.waitKey(0)
        print('Found', num, object)
        print('compute num end')
        # return
        num_str = 'Found ' + str(num) + ' changed ' + object
        return num_str

    # design more tool functions:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
    parser.add_argument('--imgA_path', default=r'F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_A.png')
    parser.add_argument('--imgB_path', default=r'F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_B.png')
    parser.add_argument('--mask_save_path', default=r'./CDmask.png')

    args = parser.parse_args()

    imgA_path = args.imgA_path
    imgB_path = args.imgB_path

    Change_Perception = Change_Perception()
    Change_Perception.generate_change_caption(imgA_path, imgB_path)
    Change_Perception.change_detection(imgA_path, imgB_path, args.mask_save_path)
