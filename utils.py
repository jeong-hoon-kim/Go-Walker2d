import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """
    시드 고정
    """
    # 기본 시드 고정
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # PyTorch 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    
    # CuDNN 관련 설정: 재현성을 위한 결정론적 연산 사용
    # 이 설정은 연산 속도를 약간 저하시킬 수 있습니다.
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    print(f"seed set: {seed}")