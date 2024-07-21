import torch
from torch import nn, einsum
import torchvision.models as models
from einops import rearrange

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg11': #512,1/32H,1/32W
            cnn = models.vgg11(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg16': #512,1/32H,1/32W
            cnn = models.vgg16(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet121': #no AdaptiveAvgPool2d #1024,1/32H,1/32W
            cnn = models.densenet121(pretrained=True) 
            modules = list(cnn.children())[:-1] 
        elif self.network=='densenet169': #1664,1/32H,1/32W
            cnn = models.densenet169(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet201': #1920,1/32H,1/32W
            cnn = models.densenet201(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='regnet_x_400mf': #400,1/32H,1/32W
            cnn = models.regnet_x_400mf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_8gf': #1920,1/32H,1/32W
            cnn = models.regnet_x_8gf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_16gf': #2048,1/32H,1/32W
            cnn = models.regnet_x_16gf(pretrained=True) 
            modules = list(cnn.children())[:-2]
        elif 'segformer' in self.network:
            from .segformer import Segformer_baseline
            self.cnn = Segformer_baseline(backbone=self.network.split('-')[-1])
        if 'segformer' not in self.network:
           self.cnn = nn.Sequential(*modules)
        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        if 'segformer' not in self.network:
            # feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            # feat2 = self.cnn(imageB)
            feat1 = imageA
            feat2 = imageB
            feat1_list = []
            feat2_list = []
            cnn_list = list(self.cnn.children())
            for module in cnn_list:
                feat1 = module(feat1)
                feat2 = module(feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
            feat1_list = feat1_list[-4:]
            feat2_list = feat2_list[-4:]
        else:
            # feat1_list, feat2_list = self.cnn(imageA, imageB)
            feat1_list, feat1 = self.cnn.segformer.stage_123(imageA)
            feat2_list, feat2 = self.cnn.segformer.stage_123(imageB)
            feat1_CD = self.cnn.segformer.stage_4(feat1)
            feat2_CD = self.cnn.segformer.stage_4(feat2)
            feat1_list.append(feat1_CD)
            feat2_list.append(feat2_CD)
            #
            feat1_CC = feat1_CD
            feat2_CC = feat2_CD
            feat1_list.append(feat1_CC)
            feat2_list.append(feat2_CC)

        return feat1_list, feat2_list

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        if 'segformer' in self.network:
            for p in self.cnn.parameters():
                p.requires_grad = fine_tune
            # for p in self.cnn2.parameters():
            #     p.requires_grad = fine_tune
        else:
            for p in self.cnn.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 3 through 4
            for c in list(self.cnn.children())[:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Dynamic_conv(nn.Module):
    def __init__(self, dim):
        super(Dynamic_conv, self).__init__()
        self.d_conv_3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim
        )
        self.d_conv_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim)
        self.d_conv_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim)
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(3*dim)
        self.conv_1 = nn.Conv2d(3*dim, dim, 1)
    def forward(self, x):
        x1 = self.d_conv_3x3(x)
        x2 = self.d_conv_1x5(x)
        x3 = self.d_conv_5x1(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.BN(x)
        x = self.activation(x)
        x = self.conv_1(x)
        return x


class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads = 8, dropout = 0.):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        dim_head = attention_dim // heads
        self.scale = (attention_dim // self.heads) ** -0.5

        self.to_q = nn.Linear(dim_q, attention_dim, bias=True)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias=True)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias=True)
        self.Q_LN = nn.LayerNorm(dim_q)
        self.K_LN = nn.LayerNorm(dim_kv)
        self.V_LN = nn.LayerNorm(dim_kv)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(attention_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        #
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(1 * dim_kv, dim_kv, 1),
            nn.BatchNorm2d(dim_kv),
            nn.ReLU(),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(2 * dim_kv, dim_kv, 1),
            nn.BatchNorm2d(dim_kv),
            nn.ReLU(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x1, x2, x3):
        batch, L, c = x1.shape
        cross = not x1.equal(x2)

        h = torch.sqrt(torch.tensor(L).float()).int()
        w = torch.sqrt(torch.tensor(L).float()).int()
        x1_feat = x1.transpose(-1, 1).view(batch, c, h, w)
        x2_feat = x2.transpose(-1, 1).view(batch, c, h, w)
        x3_feat = x3.transpose(-1, 1).view(batch, c, h, w)
        x1_feat_buff = x1_feat
        if cross:
            dif = x2_feat - x1_feat
            # x1_feat_dif = torch.cat([x1_feat, dif], dim=1)
            # x2_feat_dif = torch.cat([x2_feat, dif], dim=1)
            # x3_feat_dif = torch.cat([x2_feat, dif], dim=1)
            # x1_feat = self.fuse_conv(x1_feat_dif) + x1_feat
            x2_feat = self.fuse_conv(x1_feat*dif) #+ dif
            x3_feat = x2_feat#self.fuse_conv2(x3_feat_dif)# + x3_feat

        x1 = x1_feat.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
        x2 = x2_feat.view(batch, c, -1).transpose(-1, 1)
        x3 = x3_feat.view(batch, c, -1).transpose(-1, 1)
        x1_feat_buff = x1_feat_buff.view(batch, c, -1).transpose(-1, 1)
        # add LN
        # x1 = self.Q_LN(x1)
        # x2 = self.K_LN(x2)

        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        out = out # + x1_feat_buff
        return out  #(b,n,dim)

class Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout = 0., norm_first=False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)


        self.Q_d_conv = Dynamic_conv(dim_q)
        self.K_d_conv = Dynamic_conv(dim_kv)
        # self.V_d_conv = Dynamic_conv(dim_kv)

        group = dim_q
        self.PCM = nn.Sequential(
            nn.Conv2d(dim_q, dim_q, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=group),
            # the 1st convolution
            nn.BatchNorm2d(dim_q),
            nn.GELU(),
            nn.Conv2d(dim_q, dim_q, kernel_size=(1, 1), stride=1),
        )

    def forward(self, x1, x2, x3):
        batch, L, c = x1.shape
        h = torch.sqrt(torch.tensor(L).float()).int()
        w = torch.sqrt(torch.tensor(L).float()).int()
        x1_feat = x1.transpose(-1, 1).view(batch, c, h, w)
        x2_feat = x2.transpose(-1, 1).view(batch, c, h, w)
        x3_feat = x3.transpose(-1, 1).view(batch, c, h, w)
        if True:
            x1_feat = x1_feat + self.Q_d_conv(x1_feat)#.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
            x2_feat = x2_feat + self.K_d_conv(x2_feat)#.view(batch, c, -1).transpose(-1, 1)
            x3_feat = x3_feat + self.K_d_conv(x3_feat)#.view(batch, c, -1).transpose(-1, 1)
            x1 = x1_feat.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
            x2 = x2_feat.view(batch, c, -1).transpose(-1, 1)
            x3 = x3_feat.view(batch, c, -1).transpose(-1, 1)
            # res:
            res = x1_feat  # self.PCM(x1_feat)
            res = res.view(batch, c, -1).transpose(-1, 1)


        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + res  # batch, hw, c
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3) + res)  # batch, hw, c
            x = self.norm2(self.feedforward(x) + x)
        return x


class Q_Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout = 0., norm_first=False):
        super(Q_Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout = dropout)
        self.att2 = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim=4*dim_q, dropout = dropout)
        self.norm0 = nn.LayerNorm(dim_q)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x1 = self.att(self.norm0(x1), self.norm0(x1), self.norm0(x1)) + x1
            x = self.att2(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x1 = self.norm0(self.att(x1, x1, x1) + x1)
            x = self.norm1(self.att(x1, x2, x3) + x1)
            x = self.norm2(self.feedforward(x) + x)
        return x


class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, train_stage, n_layers, feature_size, heads, dropout=0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.train_stage = train_stage
        # change captioning branch
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        self.Dynamic_DIF_aware_TR = nn.ModuleList([])
        for i in range(n_layers):
            self.Dynamic_DIF_aware_TR.append(nn.ModuleList([
                Transformer(dim_q=channels, dim_kv=channels, heads=heads, attention_dim=channels, hidden_dim=4*channels, dropout=dropout, norm_first=False),
                Transformer(dim_q=channels, dim_kv=channels, heads=heads, attention_dim=channels,
                            hidden_dim=4 * channels, dropout=dropout, norm_first=False),
                nn.Linear(channels* 2, channels)
            ]))
        ## all modules related to captioning:
        self.cap_modules_list = [self.h_embedding, self.w_embedding,
                                 self.Dynamic_DIF_aware_TR]

        # change detection branch
        self.h_embedding_CD = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding_CD = nn.Embedding(w_feat, int(channels/2))
        dims = [64, 128, 320, 512]
        decoder_dim = 512
        self.Transformer_aug_CD = nn.ModuleList([nn.ModuleList([
            Transformer(dim_q=dims[-1], dim_kv=dims[-1], heads=heads, attention_dim=dims[-1], hidden_dim=4 * dims[-1],
                        dropout=dropout, norm_first=False),
            Transformer(dim_q=dims[-1], dim_kv=dims[-1], heads=heads, attention_dim=dims[-1], hidden_dim=4 * dims[-1],
                        dropout=dropout, norm_first=False)
        ])
            for layer_num in range(3)
        ])

        self.conv_dif = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim)
        ) for i, dim in enumerate(dims)])
        self.conv_fuse = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3*dim, 2*dim, 3, stride=1, padding=1), nn.BatchNorm2d(2 * dim), nn.ReLU(),
            nn.Conv2d(2 * dim, 2 * dim, 1)
        ) for i, dim in enumerate(dims)])

        self.cos = torch.nn.CosineSimilarity(dim=1)

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2 * dim, 2*dim, 1), nn.BatchNorm2d(2*dim), nn.ReLU(),
            nn.ConvTranspose2d(dim*2, 2*dims[max(i-1,0)], 4, stride=2, padding=1),
        ) for i, dim in enumerate(dims)])

        num_classes = 3 # background, road, building
        self.to_seg = nn.Sequential(
            nn.ConvTranspose2d(dims[0] * 2, dims[0], 4, stride=2, padding=1),
            nn.Conv2d(int(dims[0]), num_classes, 1),
        )

        # all modules related to change detection:
        self.CD_modules_list = [self.Transformer_aug_CD, self.conv_dif, self.conv_fuse, self.cos,
                                self.to_fused, self.to_seg,self.h_embedding_CD, self.w_embedding_CD
                                ]

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def add_pos_embedding(self, x):
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        return x

    def add_pos_embedding_CD(self, x):
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding_CD(pos_h)
        embed_w = self.h_embedding_CD(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        return x
    def prepare_caption(self, img1, img2, CD_feat_list=None):
        batch, c, h, w = img1.shape
        img1 = img1.view(batch, c, -1).transpose(-1, 1)  # batch, hw, c
        img2 = img2.view(batch, c, -1).transpose(-1, 1)
        img_sa1, img_sa2 = img1, img2

        for (l, m, linear) in self.Dynamic_DIF_aware_TR:
            img_sa1_tr1 = l(img_sa1, img_sa2, img_sa2) #+ img_sa1
            img_sa2_tr1 = m(img_sa2, img_sa1, img_sa1) #+ img_sa2
            img_sa1 = img_sa1_tr1 #+ img_sa1
            img_sa2 = img_sa2_tr1 #+ img_sa2

        img1 = img_sa1_tr1.transpose(-1, 1).view(batch, c, h, w)
        img2 = img_sa2_tr1.transpose(-1, 1).view(batch, c, h, w)
        # prepare A B feat for detection branch
        feat_list = []
        feat_list.append(img1)
        feat_list.append(img2)
        return img1, img2, feat_list


    def change_detection(self, img1_list, img2_list, CC_feat_list = None):
        feat_num = len(img1_list)
        img_fus_list = []
        # fisrtly aug the single-temporal last features by semantic Transformer neck
        feat_1_last = img1_list[-1]
        feat_2_last = img2_list[-1]
        b, n, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, n, -1).transpose(-1, 1)
        feat_2_last = feat_2_last.view(b, n, -1).transpose(-1, 1)

        # feat_1_last = self.Transformer_aug(feat_1_last)
        # feat_2_last = self.Transformer_aug(feat_2_last)#.transpose(-1, 1).view(b, n, h, w)

        feat_1_last = feat_1_last.transpose(-1, 1).view(b, n, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, n, h, w)
        img1_list[-1] = feat_1_last
        img2_list[-1] = feat_2_last
        # secondly fuse bi-temporal features in every level
        for k in range(feat_num):
            # method 1
            dif = self.conv_dif[k](img2_list[k] - img1_list[k]) + self.cos(img1_list[k], img2_list[k]).unsqueeze(1)
            fus = torch.cat([img1_list[k], dif, img2_list[k]], dim=1)
            fus = self.conv_fuse[k](fus)
            img_fus_list.append(fus)
        up = self.to_fused[-1](img_fus_list[-1])
        for i in range(feat_num-1, 0, -1): # 1,2,3
            i = i-1
            img_fus = img_fus_list[i] + up #img1_list[i] - img2_list[i]#
            img_fus = self.to_fused[i](img_fus)
            up = img_fus
            # img_fus_list.append(img_fus)
        # fused = torch.cat(img_fus_list, dim=1) # 换成Unet那种？
        seg = self.to_seg(img_fus)
        return seg

    def CC_neck_s0(self, feat_1_last, feat_2_last):
        feat_1_last = self.add_pos_embedding(feat_1_last)  # feat_1_last + pos_embedding
        feat_2_last = self.add_pos_embedding(feat_2_last)  # feat_2_last + pos_embedding
        b, c, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, c, -1).transpose(-1, 1)  # (b,hw,c)
        feat_2_last = feat_2_last.view(b, c, -1).transpose(-1, 1)  # (b,hw,c)

        feat_1_last = feat_1_last.transpose(-1, 1).view(b, c, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, c, h, w)

        # prepare feat for detection branch
        feat = []
        feat.append(feat_1_last)
        feat.append(feat_2_last)
        return feat_1_last, feat_2_last, feat

    def CD_neck_s0(self, img1_list, img2_list):
        feat_1_last = self.add_pos_embedding_CD(img1_list[-1])  # img1_list[-1] + pos_embedding
        feat_2_last = self.add_pos_embedding_CD(img2_list[-1])  # img2_list[-1] + pos_embedding
        b, c, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, c, -1).transpose(-1, 1)  # (b,hw,c)
        feat_2_last = feat_2_last.view(b, c, -1).transpose(-1, 1)
        for layerA, layerB in self.Transformer_aug_CD:
            feat_1_last = layerA(feat_1_last, feat_2_last, feat_2_last) +feat_1_last
            feat_2_last = layerB(feat_2_last, feat_1_last, feat_1_last) +feat_2_last
        feat_1_last = feat_1_last.transpose(-1, 1).view(b, c, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, c, h, w)
        img1_list[-1] = feat_1_last
        img2_list[-1] = feat_2_last

        # prepare CD branch feat for caption branch
        feat = []
        feat.append(img1_list[-1])
        feat.append(img2_list[-1])

        return img1_list, img2_list, feat

    def forward(self, img1_list, img2_list):
        # 1. get A B feature from backbone
        CD_img1_list, CD_img2_list = img1_list[:4], img2_list[:4]
        CC_img1_feat, CC_img2_feat = img1_list[-1], img2_list[-1]
        # 2.1 neck-stage 0 for detection:
        CD_img1_list, CD_img2_list, _ = self.CD_neck_s0(CD_img1_list, CD_img2_list)
        # 2.2 neck-stage 0 for captioning
        CC_img1_feat, CC_img2_feat, _ = self.CC_neck_s0(CC_img1_feat, CC_img2_feat)

        # captioning
        img1_cap, img2_cap, CC_feat_list = self.prepare_caption(CC_img1_feat, CC_img2_feat, None)
        # detection
        seg = self.change_detection(CD_img1_list, CD_img2_list, None)

        return img1_cap, img2_cap, seg

    def fine_tune(self, goal, fine_tune=True):
        """
        :param fine_tune: Allow?
        0: no fine-tune
        1: fine-tune the captioning related modules
        """
        assert goal in [0, 1]
        for p in self.parameters():
            p.requires_grad = False
        if goal == 1: # fine-tune the captioning module
            for m in self.cap_modules_list:
                m.train()
                for p in m.parameters():
                    p.requires_grad = True
            # # Set CD related modules to eval()
            for m in self.CD_modules_list:
                m.eval()
        elif goal ==0: # fine-tune the detection module
            for m in self.CD_modules_list:
                m.train()
                for p in m.parameters():
                    p.requires_grad = True
            # Set captioning related modules to eval()
            for m in self.cap_modules_list:
                m.eval()