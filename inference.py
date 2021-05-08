import torch
from network import FINED
import os
import numpy as np
from datasets import BSDS_Dataset
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from os.path import join
import cv2
import time
#from ptflops import get_model_complexity_info

os.environ["CUDA_VISIBLE_DEVICES"]="0"
weight_file = 'weights/final-model.pth'

model = FINED(pretrain=weight_file,isTrain=False)
#model = FINED(pretrain=weight_file,isTrain=True)
model.cuda()
model.eval()
#checkpoint = torch.load(weight_file)
#model.load_state_dict(checkpoint)

test_dataset = BSDS_Dataset(split="test",scale=[0.5, 1, 1.5])
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=1, drop_last=True, shuffle=False)

save_dir = 'results'
#save_dir = 'results_full'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    if not os.path.exists(join(save_dir,"mat")):
        os.mkdir(join(save_dir,"mat"))
    if not os.path.exists(join(save_dir,"png")):
        os.mkdir(join(save_dir,"png"))

idx = 0
start_time = time.time()

with torch.no_grad():
    for i, (image, ori, img_files) in enumerate(test_loader):
        h = ori.size()[1]
        w = ori.size()[2]
        ms_fuse = np.zeros((h,w))

        for img in image:
            img = img.cuda()
            out = model(img)
            fuse = out[-1].squeeze().detach().cpu().numpy()
            fuse = cv2.resize(fuse, (w, h), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse
        ms_fuse /= len(image)

        filename = img_files[0][0][5:-4]
        result = Image.fromarray(255 - (ms_fuse * 255).astype(np.uint8))
        result.save(join(save_dir,"png", "%s.png" % filename))
        io.savemat(os.path.join(save_dir,"mat", '{}.mat'.format(filename)), {'result': ms_fuse})
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)), end="\r")
        idx = idx + 1
    print('finished.')
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('total params in all are %d ' % pytorch_total_params)