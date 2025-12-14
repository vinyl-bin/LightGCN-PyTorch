"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    """
    LightGCN 모델 구현
    """
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        # config 설정값 불러오기
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        # nn.Embedding: 학습 가능한 임베딩 테이블 생성
        # (num_users x latent_dim) 크기의 룩업 테이블
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            # LightGCN은 비선형 활성화 함수를 쓰지 않으므로 정규 분포 초기화가 더 나을 수 있음
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # 그래프 구조체 (Sparse Graph)
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        # 희소 행렬(Sparse Matrix)에 드롭아웃 적용하는 함수
        size = x.size()
        # indices().t(): 0이 아닌 값들의 인덱스 가져오기 (전치하여 (N, 2) 형태)
        index = x.indices().t()
        values = x.values()
        # 랜덤 마스킹 생성: keep_prob 확률로 유지
        random_index = torch.rand(len(values)) + keep_prob
        # 1 이상이면 True, 아니면 False
        random_index = random_index.int().bool()
        # 마스킹된 인덱스와 값만 유지
        index = index[random_index]
        # 값 스케일링 (Inverted Dropout): 드롭아웃 후 기댓값 유지를 위해 keep_prob로 나눔
        values = values[random_index]/keep_prob
        # 새로운 희소 텐서 생성
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    # LightGCN 핵심 연산 함수!!
    def computer(self):
        """
        propagate methods for lightGCN
        LightGCN의 전파(Propagation) 메서드
        """       
        # users_emb: 유저 임베딩 가중치 가져오기
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # 모든 임베딩 연결 (User + Item) -> (N+M, Dim)
        # users_emb과 items_emb을 연결하여 모든 임베딩 벡터를 생성
        all_emb = torch.cat([users_emb, items_emb])
        # embs 리스트에 초기(0층) 임베딩 저장
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        # 레이어 수 만큼 Graph Convolution 수행
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    # Sparse Matrix Multiplication (희소 행렬 곱): Graph 구조 x 현재 임베딩
                    # 이웃 노드의 정보를 집계(Aggregate)하는 과정
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # Sparse Matrix Multiplication (희소 행렬 곱): Graph 구조 x 현재 임베딩
                # 이웃 노드의 정보를 집계(Aggregate)하는 과정
                all_emb = torch.sparse.mm(g_droped, all_emb)
            # 현재 층의 임베딩 저장
            embs.append(all_emb)
        # torch.stack: 텐서들을 새로운 차원으로 쌓음. (N+M, Dim) -> (N+M, Layer+1, Dim)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        # 각 층의 임베딩 평균 계산 (Readout): LightGCN의 핵심 아이디어 (가중 합 대신 단순 평균)
        light_out = torch.mean(embs, dim=1)
        # 최종 임베딩을 다시 유저와 아이템으로 분리
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        # 모든 유저와 아이템의 최종 임베딩 계산
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        # 학습을 위한 임베딩 가져오기 (GCN 적용 후 임베딩 & 초기 임베딩)
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        # 관련 임베딩 모두 가져오기 (GCN 후 임베딩, 초기 임베딩)
        # userEmb0 등이 초기 임베딩 (L2 정규화 용도)
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        
        # L2 정규화 손실 계산
        # embedding.norm(2).pow(2): L2 노름(유클리드 거리)의 제곱 -> 가중치 크기 제어
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        # Positive 아이템 점수 계산 (내적)
        # torch.mul: 원소별 곱셈 (Element-wise product)
        # torch.sum(..., dim=1): 차원 1 방향으로 합 -> 내적 값 완성
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        
        # Negative 아이템 점수 계산
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        # BPR Loss 계산: Softplus(neg - pos)
        # softplus(x) = log(1 + exp(x)), ReLU의 부드러운 버전
        # (neg - pos)가 클수록(오답일수록) Loss가 커짐
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
