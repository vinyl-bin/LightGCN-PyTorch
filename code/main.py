import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
# 랜덤 시드 고정 (실험 재현성 확보)
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

# 모델 생성 및 GPU/CPU로 전송
# world.model_name에 따라 'mf' 또는 'lgn' 모델 선택
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)  # GPU 사용 가능하면 GPU로, 아니면 CPU로

# BPR 손실 함수 객체 생성
bpr = utils.BPRLoss(Recmodel, world.config)

# 모델 가중치 저장/로드 파일명
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

# 기존 모델 가중치 로드 (world.LOAD가 True인 경우)
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# Negative 샘플 비율
Neg_k = 1

# TensorBoard 초기화 (학습 과정 시각화)
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

# 학습 루프
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        # 10 epoch마다 테스트 수행
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        # 학습 수행
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        # 매 epoch마다 모델 가중치 저장
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    # TensorBoard 종료
    if world.tensorboard:
        w.close()