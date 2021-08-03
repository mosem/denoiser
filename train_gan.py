# taken from HIFI-GAN: https://github.com/jik876/hifi-gan

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from dataset import Gan_dataset, get_dataset_filelist
from gan_models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from utils import get_torch_device

torch.backends.cudnn.benchmark = True


def collate_fn_padd(batch):
    """
    add zero padding to support variable length
    """

    # get sequence lengths
    lengths_nb = torch.tensor([t[0].shape for t in batch]).to(get_torch_device())
    lengths_wb = torch.tensor([t[1].shape for t in batch]).to(get_torch_device())
    # pad
    batch_nb = [torch.Tensor(t[0]).to(get_torch_device()) for t in batch]
    batch_nb = torch.unsqueeze(torch.nn.utils.rnn.pad_sequence(batch_nb).T, 1)
    batch_wb = [torch.Tensor(t[1]).to(get_torch_device()) for t in batch]
    batch_wb = torch.unsqueeze(torch.nn.utils.rnn.pad_sequence(batch_wb).T, 1)
    # compute mask
    mask_nb = (batch_nb != 0).to(get_torch_device())
    mask_wb = (batch_wb != 0).to(get_torch_device())
    return (batch_nb, batch_wb), (lengths_nb, lengths_wb), (mask_nb, mask_wb)


def train(rank, args, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    # device = torch.device('cuda:{:d}'.format(rank))
    device = get_torch_device()

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(args.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", args.checkpoint_path)

    if os.path.isdir(args.checkpoint_path):
        cp_g = scan_checkpoint(args.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(args.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']


    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h.base_dir_NB)

    trainset = Gan_dataset(training_filelist,
                           h.sampling_rate,
                           h.base_dir_NB,
                           h.base_dir_HB)


    train_loader = DataLoader(trainset, shuffle=False,
                              batch_size=h.batch_size,
                              collate_fn=collate_fn_padd)

    if rank == 0:
        validset = Gan_dataset(validation_filelist,
                               h.sampling_rate,
                               h.base_dir_NB,
                               h.base_dir_HB)
        validation_loader = DataLoader(validset, shuffle=False,
                                       sampler=None,
                                       collate_fn=collate_fn_padd)

        sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), args.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))


        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y = batch[0]
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))

            y_g_hat = generator(x)
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Loss
            loss_audio = F.l1_loss(y, y_g_hat) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_audio

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % args.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y, y_g_hat).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % args.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(args.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(args.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                             else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                             else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % args.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/audio_error", mel_error, steps)

                # Validation
                if steps % args.validation_interval == 0:  # and steps != 0:
                    torch.cuda.empty_cache()
                    generator.eval()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y = batch[0]
                            y_g_hat = generator(x.to(device))
                            val_err_tot += F.l1_loss(y, y_g_hat).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def train_gan():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='./gan_config.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'gan_config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    train(0, a, h)


if __name__ == '__main__':
    train_gan()