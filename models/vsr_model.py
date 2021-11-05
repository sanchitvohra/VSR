import torch
import torch.nn as nn

from models.frnet import FRNet
from models.stnet import STNet
from models.vgg_nets import VGGFeatureExtractor

from utils.warp import backward_warp
from utils.cosine_annealing_restarts import CosineAnnealingLR_Restart
from utils.losses import CosineSimilarityLoss, CharbonnierLoss, LSGANLoss, VanillaGANLoss

class VSRModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        generator_config = opt['model']['generator']
        if generator_config['name'] == 'FRNet':
            self.net_G = FRNet(int(generator_config['in_nc']), int(generator_config['out_nc']), 
            int(generator_config['nf']), int(generator_config['nb'])).to(self.device)
            # model loading
        
        discriminator_config = opt['model']['discriminator']
        if discriminator_config['name'] == 'STNet':
            self.net_D = STNet(int(discriminator_config['in_nc']), 128, 128, 
            int(discriminator_config['tempo_range'])).to(self.device)
            # model loading

    def config_training(self):
        self.set_criterion()

        generator_config = self.opt['train']['generator']
        g_lr = float(generator_config['lr'])
        g_betas = (
        float(generator_config['beta1']),
        float(generator_config['beta2']))
        self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=g_lr, weight_decay=0, betas=g_betas)

        discriminator_config = self.opt['train']['discriminator']
        d_lr = float(discriminator_config['lr'])
        d_betas = (
        float(discriminator_config['beta1']),
        float(discriminator_config['beta2']))
        self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=d_lr, weight_decay=0, betas=d_betas)

        # set lr schedules for G
        lr_schedule = self.opt['train']['generator'].get('lr_schedule')
        self.sched_G = self.define_lr_schedule(lr_schedule, self.optim_G)

        # set lr schedules for D
        lr_schedule = self.opt['train']['discriminator'].get('lr_schedule')
        self.sched_D = self.define_lr_schedule(lr_schedule, self.optim_D)

    def define_lr_schedule(self, schedule_opt, optimizer):
        if schedule_opt is None:
            return None

        # parse
        if schedule_opt['type'] == 'FixedLR':
            schedule = None

        elif schedule_opt['type'] == 'MultiStepLR':
            schedule = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=schedule_opt['milestones'],
                gamma=schedule_opt['gamma']
            )

        elif schedule_opt['type'] == 'CosineAnnealingLR_Restart':
            schedule = CosineAnnealingLR_Restart(
                optimizer, schedule_opt['periods'],
                eta_min=schedule_opt['eta_min'],
                restarts=schedule_opt['restarts'],
                weights=schedule_opt['restart_weights']
            )

        else:
            raise ValueError('Unrecognized lr schedule: {}'.format(
                schedule_opt['type']))

        return schedule

    def define_criterion(self, criterion_opt):
        if criterion_opt is None:
            return None

        # parse
        if criterion_opt['type'] == 'MSE':
            criterion = nn.MSELoss(reduction=criterion_opt['reduction'])

        elif criterion_opt['type'] == 'L1':
            criterion = nn.L1Loss(reduction=criterion_opt['reduction'])

        elif criterion_opt['type'] == 'CB':
            criterion = CharbonnierLoss(reduction=criterion_opt['reduction'])

        elif criterion_opt['type'] == 'CosineSimilarity':
            criterion = CosineSimilarityLoss()

        elif criterion_opt['type'] == 'GAN':
            criterion = VanillaGANLoss(reduction=criterion_opt['reduction'])

        elif criterion_opt['type'] == 'LSGAN':
            criterion = LSGANLoss(reduction=criterion_opt['reduction'])

        else:
            raise ValueError('Unrecognized criterion: {}'.format(
                criterion_opt['type']))

        return criterion

    def set_criterion(self):
        # pixel criterion
        self.pix_crit = self.define_criterion(
            self.opt['train'].get('pixel_crit'))

        # warping criterion
        self.warp_crit = self.define_criterion(
            self.opt['train'].get('warping_crit'))

        # feature criterion
        self.feat_crit = self.define_criterion(
            self.opt['train'].get('feature_crit'))
        if self.feat_crit is not None:  # load feature extractor
            feature_layers = self.opt['train']['feature_crit'].get(
                'feature_layers', [8, 17, 26, 35])
            self.net_F = VGGFeatureExtractor(feature_layers).to(self.device)

        # flow & mask criterion
        self.flow_crit = self.define_criterion(
            self.opt['train'].get('flow_crit'))

        # ping-pong criterion
        self.pp_crit = self.define_criterion(
            self.opt['train'].get('pingpong_crit'))

        # feature matching criterion
        self.fm_crit = self.define_criterion(
            self.opt['train'].get('feature_matching_crit'))

        # gan criterion
        self.gan_crit = self.define_criterion(
            self.opt['train'].get('gan_crit'))

    def train_batch(self, data):
        """ Function for mini-batch training

            Parameters:
                :param data: a batch of training tensor with shape NTCHW
        """

        # ------------ prepare data ------------ #
        lr_data, gt_data = data['lr'], data['gt']

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        # generate upsampled data
        bi_data = self.net_G.upsample_func(
            lr_data.view(n * t, c, lr_h, lr_w)).view(n, t, c, gt_h, gt_w)

        # augment data for pingpong criterion
        if self.pp_crit is not None:
            # i.e., (0,1,2,...,t-2,t-1) -> (0,1,2,...,t-2,t-1,t-2,...,2,1,0)
            lr_rev = lr_data.flip(1)[:, 1:, ...]
            gt_rev = gt_data.flip(1)[:, 1:, ...]
            bi_rev = bi_data.flip(1)[:, 1:, ...]

            lr_data = torch.cat([lr_data, lr_rev], dim=1)
            gt_data = torch.cat([gt_data, gt_rev], dim=1)
            bi_data = torch.cat([bi_data, bi_rev], dim=1)


        # ------------ clear optimizers ------------ #
        self.net_G.train()
        self.net_D.train()
        self.optim_G.zero_grad()
        self.optim_D.zero_grad()


        # ------------ forward G ------------ #
        net_G_output_dict = self.net_G.forward_sequence(lr_data)
        hr_data = net_G_output_dict['hr_data']


        # ------------ forward D ------------ #
        for param in self.net_D.parameters():
            param.requires_grad = True

        # feed additional data
        net_D_input_dict = {
            'net_G': self.net_G,
            'lr_data': lr_data,
            'bi_data': bi_data,
            'use_pp_crit': True,
            'crop_border_ratio': self.opt['train']['discriminator'].get(
                'crop_border_ratio', 1.0)
        }
        net_D_input_dict.update(net_G_output_dict)

        # forward real sequence (gt)
        real_pred, net_D_oputput_dict = self.net_D.forward_sequence(
            gt_data, net_D_input_dict)

        # reuse internal data (e.g., lr optical flow) to reduce computations
        net_D_input_dict.update(net_D_oputput_dict)

        # forward fake sequence (hr)
        fake_pred, _ = self.net_D.forward_sequence(
            hr_data.detach(), net_D_input_dict)


        # ------------ optimize D ------------ #
        real_pred_D, fake_pred_D = real_pred[0], fake_pred[0]

        # select D update policy
        update_policy = self.opt['train']['discriminator']['update_policy']
        if update_policy == 'adaptive':
            # update D adaptively
            logged_real_pred_D = torch.log(torch.sigmoid(real_pred_D) + 1e-8)
            logged_fake_pred_D = torch.log(torch.sigmoid(fake_pred_D) + 1e-8)

            distance = logged_real_pred_D.mean() - logged_fake_pred_D.mean()

            threshold = self.opt['train']['discriminator']['update_threshold']
            upd_D = distance.item() < threshold
        else:
            upd_D = True

        if upd_D:
            self.cnt_upd_D += 1
            real_loss_D = self.gan_crit(real_pred_D, 1)
            fake_loss_D = self.gan_crit(fake_pred_D, 0)
            loss_D = real_loss_D + fake_loss_D

            # update D
            loss_D.backward()
            self.optim_D.step()
        else:
            loss_D = torch.zeros(1)

        # ------------ optimize G ------------ #
        for param in self.net_D.parameters():
            param.requires_grad = False

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
            loss_pix_G = pix_w * self.pix_crit(hr_data, gt_data)
            loss_G += loss_pix_G

        # warping (warp) loss
        if self.warp_crit is not None:
            lr_curr = net_G_output_dict['lr_curr']
            lr_prev = net_G_output_dict['lr_prev']
            lr_flow = net_G_output_dict['lr_flow']
            lr_warp = backward_warp(lr_prev, lr_flow)

            warp_w = self.opt['train']['warping_crit'].get('weight', 1)
            loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr)
            loss_G += loss_warp_G

        # feature (feat) loss
        if self.feat_crit is not None:
            hr_merge = hr_data.view(-1, c, gt_h, gt_w)
            gt_merge = gt_data.view(-1, c, gt_h, gt_w)

            hr_feat_lst = self.net_F(hr_merge)
            gt_feat_lst = self.net_F(gt_merge)
            loss_feat_G = 0
            for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
                loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())

            feat_w = self.opt['train']['feature_crit'].get('weight', 1)
            loss_feat_G = feat_w * loss_feat_G
            loss_G += loss_feat_G

        # ping-pong (pp) loss
        if self.pp_crit is not None:
            tempo_extent = self.opt['train']['tempo_extent']
            hr_data_fw = hr_data[:, :tempo_extent - 1, ...]     # -------->|
            hr_data_bw = hr_data[:, tempo_extent:, ...].flip(1) # <--------|

            pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
            loss_pp_G = pp_w * self.pp_crit(hr_data_fw, hr_data_bw)
            loss_G += loss_pp_G

        # feature matching (fm) loss
        if self.fm_crit is not None:
            fake_pred, _ = self.net_D.forward_sequence(hr_data, net_D_input_dict)
            fake_feat_lst, real_feat_lst = fake_pred[-1], real_pred[-1]

            layer_norm = self.opt['train']['feature_matching_crit'].get(
                'layer_norm', [12.0, 14.0, 24.0, 100.0])

            loss_fm_G = 0
            for i in range(len(real_feat_lst)):
                fake_feat, real_feat = fake_feat_lst[i], real_feat_lst[i]
                loss_fm_G += self.fm_crit(
                    fake_feat, real_feat.detach()) / layer_norm[i]

            fm_w = self.opt['train']['feature_matching_crit'].get('weight', 1)
            loss_fm_G = fm_w * loss_fm_G
            loss_G += loss_fm_G

        # gan loss
        if self.fm_crit is None:
            fake_pred, _ = self.net_D.forward_sequence(hr_data, net_D_input_dict)
        fake_pred_G = fake_pred[0]

        gan_w = self.opt['train']['gan_crit'].get('weight', 1)
        loss_gan_G = gan_w * self.gan_crit(fake_pred_G, True)
        loss_G += loss_gan_G

        # update G
        loss_G.backward()
        self.optim_G.step()
 