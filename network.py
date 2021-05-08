import torch
import torch.nn as nn
from mylayers import RCUBlock, CRPBlock, MSBlock

class FINED(nn.Module):
    def __init__(self,pretrain=False,isTrain=True):
        super(FINED, self).__init__()
        self.isTrain = isTrain
        ############## STAGE 1##################
        self.conv1_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)

        if self.isTrain:
            self.msblock1_1 = MSBlock(16, 4)
            self.msblock1_2 = MSBlock(16, 4)

            self.RCU1_1 = RCUBlock(32, 32, 2, 2)
            self.RCU1_2 = RCUBlock(32, 32, 2, 2)

            self.conv1_1_down = nn.Conv2d(32, 8, 1, padding=0)
            self.conv1_2_down = nn.Conv2d(32, 8, 1, padding=0)

            self.crp1 = CRPBlock(8, 8, 4)

            self.score_stage1 = nn.Conv2d(8, 1, 1)
            self.st1_BN = nn.BatchNorm2d(1, affine=True, eps=1e-5, momentum=0.1)

        ############## STAGE 2 ##################
        self.conv2_1 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        if self.isTrain:
            self.msblock2_1 = MSBlock(64, 4)
            self.msblock2_2 = MSBlock(64, 4)

            self.RCU2_1 = RCUBlock(32, 32, 2, 2)
            self.RCU2_2 = RCUBlock(32, 32, 2, 2)

            self.conv2_1_down = nn.Conv2d(32, 8, 1, padding=0)
            self.conv2_2_down = nn.Conv2d(32, 8, 1, padding=0)

            self.crp2 = CRPBlock(8, 8, 4)

            self.score_stage2 = nn.Conv2d(8, 1, 1)
            self.st2_BN = nn.BatchNorm2d(1, affine=True, eps=1e-5, momentum=0.1)

        ############## STAGE 3 ##################
        self.conv3_1 = nn.Conv2d(64, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)

        self.msblock3_1 = MSBlock(256, 4)
        self.msblock3_2 = MSBlock(256, 4)

        self.RCU3_1 = RCUBlock(32, 32, 2, 2)
        self.RCU3_2 = RCUBlock(32, 32, 2, 2)

        # CONV DOWN
        self.conv3_1_down = nn.Conv2d(32, 8, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(32, 8, 1, padding=0)

        self.crp3 = CRPBlock(8,8,4)

        # SCORE
        self.score_stage3 = nn.Conv2d(8, 1, 1)
        self.st3_BN = nn.BatchNorm2d(1, affine=True, eps=1e-5, momentum=0.1)

        # POOL
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True) # pooling biasa

        # RELU
        self.relu = nn.ReLU()
        if self.isTrain:
            self.score_final = nn.Conv2d(3, 1, 1)
            self.final_BN = nn.BatchNorm2d(1, affine=True, eps=1e-5, momentum=0.1)

        if pretrain:
            state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    #print('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    #print('init the weights of %s from mean 0, std 0.01 gaussian distribution' % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        if 'BN' in name:
                            param.zero_()
                        else:
                            param.normal_(0, 0.01)
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]
        ############## STAGE 1 ##################
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        if self.isTrain:
            er1_1 = self.msblock1_1(conv1_1)
            er1_2 = self.msblock1_2(conv1_2)

            rcu1_1 = self.relu(self.RCU1_1(er1_1))
            rcu1_2 = self.relu(self.RCU1_2(er1_2))

            conv1_1_down = self.conv1_1_down(rcu1_1)
            conv1_2_down = self.conv1_2_down(rcu1_2)

            crp1 = self.crp1(conv1_1_down + conv1_2_down)
            o1_out = self.score_stage1(crp1)
            so1 = crop(o1_out, img_H, img_W)
            so1 = self.st1_BN(so1)
        ############## END STAGE 1 ##################
        pool1 = self.maxpool(conv1_2)
        ############## STAGE 2 ##################
        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        if self.isTrain:
            er2_1 = self.msblock2_1(conv2_1)
            er2_2 = self.msblock2_2(conv2_2)

            rcu2_1 = self.relu(self.RCU2_1(er2_1))
            rcu2_2 = self.relu(self.RCU2_2(er2_2))

            conv2_1_down = self.conv2_1_down(rcu2_1)
            conv2_2_down = self.conv2_2_down(rcu2_2)

            crp2 = self.crp2(conv2_1_down + conv2_2_down)
            o2_out = self.score_stage2(crp2)
            upsample2 = nn.UpsamplingBilinear2d(size=(img_H, img_W))(o2_out)
            so2 = crop(upsample2, img_H, img_W)
            so2 = self.st2_BN(so2)
        ############## END STAGE 2 ##################
        pool2 = self.maxpool(conv2_2)
        ############## STAGE 3 ##################
        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))

        er3_1 = self.msblock3_1(conv3_1)
        er3_2 = self.msblock3_2(conv3_2)

        rcu3_1 = self.relu(self.RCU3_1(er3_1))
        rcu3_2 = self.relu(self.RCU3_2(er3_2))

        conv3_1_down = self.conv3_1_down(rcu3_1)
        conv3_2_down = self.conv3_2_down(rcu3_2)

        crp3 = self.crp3(conv3_1_down + conv3_2_down)
        o3_out = self.score_stage3(crp3)
        upsample3 = nn.UpsamplingBilinear2d(size=(img_H, img_W))(o3_out)
        so3 = crop(upsample3, img_H, img_W)
        so3 = self.st3_BN(so3)
        ############## END STAGE 3 ##################

        ############## FUSION ##################
        if self.isTrain:
            fusecat = torch.cat((so1, so2, so3), dim=1)
            fuse = self.score_final(fusecat)
            fuse = self.final_BN(fuse)
            results = [so1, so2, so3, fuse]
        else:
            results = [so3]
        results = [torch.sigmoid(r) for r in results]

        return results

def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]