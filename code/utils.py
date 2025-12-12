'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

이 파일은 LightGCN 추천 시스템의 핵심 유틸리티 함수들을 포함합니다:
- BPR 손실 함수 클래스
- 네거티브 샘플링 함수
- 평가 메트릭 (Recall, Precision, NDCG, MRR, AUC)
- 유틸리티 함수 (시드 설정, 배치 처리, 타이머 등)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
# C++ 확장 모듈 로드 시도, cppimport를 사용해서 .cpp 파일을 즉석에서 컴파일
# 수백만 개의 negative samples를 생성해야 함
# Python의 for 루프는 매우 느림
# C++는 10~100배 빠름
# 학습 시간을 크게 단축
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)  # C++ 샘플링 함수 로드
    sampling.seed(world.seed)  # 랜덤 시드 설정
    sample_ext = True  # C++ 확장 사용 가능
except:
    world.cprint("Cpp extension not loaded")  # C++ 확장 로드 실패
    sample_ext = False  # Python 구현 사용

# BPR (Bayesian Personalized Ranking) 손실 함수 클래스
class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']  # 가중치 감쇠 (L2 정규화)
        self.lr = config['lr']  # 학습률
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)  # Adam optimizer

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)  # BPR 손실 계산
        reg_loss = reg_loss*self.weight_decay  # 정규화 항에 가중치 적용
        loss = loss + reg_loss  # 총 손실 = BPR 손실 + 정규화 손실

        self.opt.zero_grad()  # 그래디언트 초기화
        loss.backward()  # 역전파
        self.opt.step()  # 파라미터 업데이트

        return loss.cpu().item()  # 손실 값 반환


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
         # C++ 확장 가능해서 C++로 샘플링
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        # C++ 확장 불가능해서 Python으로 샘플링
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    # 랜덤 유저 샘플링(선택)
    # 매개변수 : low, high, size -> low ~ high-1 사이의 정수를 size만큼 랜덤으로 선택
    users = np.random.randint(0, dataset.n_users, user_num)
    # 모든 유저의 positive item 리스트
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    # enumerate : 인덱스와 값 동시 반환
    for i, user in enumerate(users):
        start = time()
        # 해당 유저가 좋아하는 item 리스트
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        # 0부터 len(posForUser)-1 사이의 정수를 랜덤으로 선택 
        posindex = np.random.randint(0, len(posForUser))
        # 유저가 좋아한 item 중 하나 랜덤 선택
        positem = posForUser[posindex]
        while True:
            # 전체 데이터 중 랜덤으로 하나 선택해서 negative_item이 아니면 다시 선택, negative_item이면 break
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        # 유저, positive_item, negative_item 순서로 리스트에 추가
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

# 랜덤 시드 고정 함수 -> 같은 seed이면 항상 같은 결과를 얻음 -> 실험 재현성 확보
def set_seed(seed):
    # numpy의 random seed 고정
    np.random.seed(seed)

    # GPU 사용가능하면
    if torch.cuda.is_available():
        # 현재 GPU의 random seed 고정
        torch.cuda.manual_seed(seed)
        # 모든 GPU의 random seed 고정
        torch.cuda.manual_seed_all(seed)
    # CPU의 random seed 고정
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        # f-string 문자열 포맷팅
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    # os에 맞게 경로 결합
    return os.path.join(world.FILE_PATH,file)

# *args 일반 가변인자 -> 여러 값이 튜플 형식으로 전달 ex) var_args_ex('hello','world',2023)
# **args 키워드 가변인자 -> 여러 키워드 인자가 딕셔너리 형식으로 전달 ex) var_args_ex(name='world',age=2023)
def minibatch(*tensors, **kwargs):

    # dict.get(key, default): 키가 없으면 기본값 반환 
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            # yield:  return과 달리 값을 여러 번 반환할 수 있음
            # 슬라이싱하여 batch_size만큼의 데이터 반환
            yield tensor[i:i + batch_size]
    else:
        # batch_size만큼의 반복하여 i 인덱스 i 지정
        for i in range(0, len(tensors[0]), batch_size):
            # 모든 tensors의 i ~ i+batch_size 인덱스 슬라이싱하여 tuple로 반환
            yield tuple(x[i:i + batch_size] for x in tensors)

# 여러 배열을 동일한 순서로 섞는 함수
def shuffle(*arrays, **kwargs):

    # dict.get(key, default): 키가 없으면 기본값 반환, 기본값이 False 
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
