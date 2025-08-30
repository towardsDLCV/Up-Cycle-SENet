import argparse
import datetime
import itertools
import random
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from data.dataset import SpeechDataset
from models.discriminator import Discriminator
from models.generator import *
from models.generator_C2N import *
from utils.metric import pesq_score
from utils.scheduler import get_scheduler
from utils.utils import *

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=50, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=1)
parser.add_argument("--n-epochs-decay", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')

parser.add_argument('--lr-policy', type=str, default='cosine',
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr', default=2e-4)

parser.add_argument('--sample-rate', type=int, default=16000, help="STFT hyperparam")
parser.add_argument('--max-len', type=int, default=32000)
parser.add_argument('--loss_type', type=str, default='l1gan', choices=['lsgan', 'ralsgan', 'l1gan',
                                                                       'wgan-gp', 'hingegan'])

parser.add_argument('--clean-train-dir', type=str, default="/data1/DEMAND/train/clean_trainset")
parser.add_argument('--noisy-train-dir', type=str, default="/data1/DEMAND/train/noisy_trainset")
parser.add_argument('--clean-test-dir', type=str, default="/data1/DEMAND/test/clean_testset")
parser.add_argument('--noisy-test-dir', type=str, default="/data1/DEMAND/test/noisy_testset")

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    summary = SummaryWriter()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # A: Noisy Sample / B: Clean_sample
    generator_A2B = Generator(args=args)
    generator_B2A = Generator_C2N(args=args)
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("using Distributed train")
            torch.cuda.set_device(args.gpu)
            generator_A2B.cuda(args.gpu)
            generator_B2A.cuda(args.gpu)
            discriminator_A.cuda(args.gpu)
            discriminator_B.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            generator_A2B = torch.nn.parallel.DistributedDataParallel(generator_A2B, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
            generator_B2A = torch.nn.parallel.DistributedDataParallel(generator_B2A, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
            discriminator_A = torch.nn.parallel.DistributedDataParallel(discriminator_A, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
            discriminator_B = torch.nn.parallel.DistributedDataParallel(discriminator_B, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)

        else:
            generator_A2B.cuda()
            generator_B2A.cuda()
            discriminator_A.cuda()
            discriminator_B.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            generator_A2B = torch.nn.parallel.DistributedDataParallel(generator_A2B)
            generator_B2A = torch.nn.parallel.DistributedDataParallel(generator_B2A)
            discriminator_A = torch.nn.parallel.DistributedDataParallel(discriminator_A)
            discriminator_B = torch.nn.parallel.DistributedDataParallel(discriminator_B)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator_A2B = generator_A2B.cuda(args.gpu)
        generator_B2A = generator_B2A.cuda(args.gpu)
        discriminator_A = discriminator_A.cuda(args.gpu)
        discriminator_B = discriminator_B.cuda(args.gpu)

    else:
        generator_A2B = torch.nn.DataParallel(generator_A2B).cuda()
        generator_B2A = torch.nn.DataParallel(generator_B2A).cuda()
        discriminator_A = torch.nn.DataParallel(discriminator_A).cuda()
        discriminator_B = torch.nn.DataParallel(discriminator_B).cuda()

    # Optimizer / criterion / Scheduler
    if args.loss_type == 'lsgan' or 'ralsgan':
        criterion_GAN = nn.MSELoss().cuda(args.gpu)
        print("Loss_type: LSGAN or RaLSGAN", criterion_GAN, args.loss_type)
    elif args.loss_type == 'l1gan':
        criterion_GAN = nn.L1Loss().cuda(args.gpu)
        print("Loss_type: L1GAN", criterion_GAN, args.loss_type)
    elif args.loss_type == 'wgan-gp' or 'hingegan':
        criterion_GAN = None
        print("Loss_type: WGAN_GP or HingeGAN", criterion_GAN, args.loss_type)
    else:
        raise Exception("Loss Error")

    criterion_cycle = nn.L1Loss().cuda(args.gpu)
    criterion_idt = nn.L1Loss().cuda(args.gpu)

    generator_optimizer = torch.optim.Adam(itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()),
                                           lr=args.lr,
                                           betas=(0.5, 0.999))
    discriminator_A_optimizer = torch.optim.Adam(discriminator_A.parameters(),
                                                 lr=args.lr,
                                                 betas=(0.5, 0.999))
    discriminator_B_optimizer = torch.optim.Adam(discriminator_B.parameters(),
                                                 lr=args.lr,
                                                 betas=(0.5, 0.999))

    generator_scheduler = get_scheduler(generator_optimizer, args)
    discriminator_A_scheduler = get_scheduler(discriminator_A_optimizer, args)
    discriminator_B_scheduler = get_scheduler(discriminator_B_optimizer, args)

    best_PESQ = -1e10
    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map models to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_PESQ = checkpoint['PESQ']

            generator_A2B.load_state_dict(checkpoint['G_A2B'])
            generator_B2A.load_state_dict(checkpoint['G_B2A'])
            discriminator_A.load_state_dict(checkpoint['D_A'])
            discriminator_B.load_state_dict(checkpoint['D_B'])

            generator_optimizer.load_state_dict(checkpoint['G_optimizer'])
            discriminator_A_optimizer.load_state_dict(checkpoint['D_A_optimizer'])
            discriminator_B_optimizer.load_state_dict(checkpoint['D_B_optimizer'])
            print("[PESQ]: ", best_PESQ, args.start_epoch)

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Dataset / Loader
    mixed_train_files = sorted(list(Path(args.noisy_train_dir).rglob('*.wav')))  # noisy
    clean_train_files = sorted(list(Path(args.clean_train_dir).rglob('*.wav')))  # clean
    mixed_test_files = sorted(list(Path(args.noisy_test_dir).rglob('*.wav')))
    clean_test_files = sorted(list(Path(args.clean_test_dir).rglob('*.wav')))

    train_dataset = SpeechDataset(args, mixed_train_files, clean_train_files, unpaired=True)
    test_dataset = SpeechDataset(args, mixed_test_files, clean_test_files, unpaired=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    # Train
    G_A2B_p = sum(p.numel() for p in generator_A2B.parameters() if p.requires_grad)
    G_B2A_p = sum(p.numel() for p in generator_B2A.parameters() if p.requires_grad)
    D_A_p = sum(p.numel() for p in discriminator_A.parameters() if p.requires_grad)
    D_B_p = sum(p.numel() for p in discriminator_B.parameters() if p.requires_grad)

    param = (G_B2A_p + G_A2B_p + D_B_p + D_A_p)
    print("Total Param: ", param)
    print("G_A2B: ", G_A2B_p)
    print("G_B2A: ", G_B2A_p)
    print("D_A: ", D_A_p)
    print("D_B: ", D_B_p)

    # Evaluate
    if args.evaluate:
        epoch = None
        PESQ = validate(test_loader, generator_A2B, generator_B2A, discriminator_A, discriminator_B,
                        criterion_GAN, criterion_cycle, criterion_idt,
                        args, epoch, summary)
        print(f"| PESQ: {PESQ:.4f} |".format(PESQ=PESQ))
        return

    for epoch in range(args.start_epoch, args.epochs + args.n_epochs_decay + 1):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        generator_scheduler.step()
        discriminator_A_scheduler.step()
        discriminator_B_scheduler.step()

        train(train_loader, epoch, generator_A2B, generator_B2A, discriminator_A, discriminator_B,
              criterion_GAN, criterion_cycle, criterion_idt,
              generator_optimizer, discriminator_A_optimizer, discriminator_B_optimizer,
              args, summary)

        print("--validate--")
        PESQ = validate(test_loader, generator_A2B, generator_B2A, discriminator_A, discriminator_B,
                        criterion_GAN, criterion_cycle, criterion_idt,
                        args, epoch, summary)

        print(f"| PESQ: {PESQ:.4f} |".format(PESQ=PESQ))

        if best_PESQ < PESQ:
            print("Found better validated models")
            best_PESQ = PESQ

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                torch.save({
                    'epoch': epoch + 1,
                    'PESQ': best_PESQ,
                    'G_A2B': generator_A2B.module.state_dict(),
                    'G_B2A': generator_B2A.module.state_dict(),
                    'D_A': discriminator_A.module.state_dict(),
                    'D_B': discriminator_B.module.state_dict(),
                    'G_optimizer': generator_optimizer.state_dict(),
                    'D_A_optimizer': discriminator_A_optimizer.state_dict(),
                    'D_B_optimizer': discriminator_B_optimizer.state_dict()
                }, "saved_models/checkpoint_%d.pth" % (epoch + 1))


def train(train_loader, epoch, generator_A2B, generator_B2A, discriminator_A, discriminator_B,
          criterion_GAN, criterion_cycle, criterion_idt,
          generator_optimizer, discriminator_A_optimizer, discriminator_B_optimizer,
          args, summary):

    generator_A2B.train()
    generator_B2A.train()
    discriminator_A.train()
    discriminator_B.train()

    if args.loss_type == 'wgan-gp':
        lambda_gp = 10

    end = time.time()
    for i, (mixed, target) in enumerate(train_loader):
        mixed = mixed.cuda(args.gpu, non_blocking=True)  # A: noisy
        target = target.cuda(args.gpu, non_blocking=True)  # B: Clean
        valid = Variable(Tensor(np.ones((mixed.size(0), *(1, 125)))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((mixed.size(0), *(1, 125)))), requires_grad=False)

        #############
        # Generator
        #############
        # Identity Loss
        loss_id_A = criterion_idt(generator_B2A(mixed), mixed)  # || F(A)-A ||
        loss_id_B = criterion_idt(generator_A2B(target), target)  # || G(B)-B ||
        loss_id = (loss_id_A + loss_id_B) / 2

        # Adversarial Loss
        fake_B = generator_A2B(mixed)
        fake_A = generator_B2A(target)

        if args.loss_type == 'l1gan' or 'lsgan':
            loss_GAN_A2B = criterion_GAN(discriminator_B(fake_B), valid)  # |D(G(A))-1|
            loss_GAN_B2A = criterion_GAN(discriminator_A(fake_A), valid)  # |D(F(B))-1|
        elif args.loss_type == 'wgan-gp':
            loss_GAN_A2B = -torch.mean(discriminator_B(fake_B))
            loss_GAN_B2A = -torch.mean(discriminator_A(fake_A))
        elif args.loss_type == 'ralsgan':
            mixed_pred = discriminator_A(mixed)
            fake_A_pred = discriminator_A(fake_A)
            target_pred = discriminator_B(target)
            fake_B_pred = discriminator_B(fake_B)
            loss_GAN_A2B = (criterion_GAN(target_pred - fake_B_pred.mean(0, keepdim=True), -valid) +
                            criterion_GAN(fake_B_pred - target_pred.mean(0, keepdim=True), valid)) / 2
            loss_GAN_B2A = (criterion_GAN(mixed_pred - fake_A_pred.mean(0, keepdim=True), -valid) +
                            criterion_GAN(fake_A_pred - mixed_pred.mean(0, keepdim=True), valid)) / 2
        elif args.loss_type == 'hingegan':
            fake_B_pred = discriminator_B(fake_B)
            fake_A_pred = discriminator_A(fake_A)
            loss_GAN_A2B = - fake_B_pred.mean()
            loss_GAN_B2A = - fake_A_pred.mean()
        else:
            raise Exception("Generator Loss Error")

        loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2
        # Cycle Consistency
        cycle_B = generator_A2B(fake_A)
        cycle_A = generator_B2A(fake_B)

        loss_cycle_A = criterion_cycle(cycle_A, mixed)  # || F(G(A))-A ||
        loss_cycle_B = criterion_cycle(cycle_B, target)  # || G(F(B))-B ||

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        generator_loss = loss_GAN + loss_cycle * 20 + loss_id * 10
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        #####################
        # Discriminator_A
        #####################
        if args.loss_type == 'l1gan' or 'lsgan':
            loss_real_A = criterion_GAN(discriminator_A(mixed), valid)  # |D_A(x)-1|
            loss_fake_A = criterion_GAN(discriminator_A(fake_A.detach()), fake)  # | D_A(F(y))-0|
            loss_discriminator_A = (loss_real_A + loss_fake_A) / 2
        elif args.loss_type == 'wgan-gp':
            real_A_valid = discriminator_A(mixed)
            fake_A_valid = discriminator_A(fake_A)
            gradient_A_penalty = compute_gradient_penalty(discriminator_A,
                                                          mixed.data,
                                                          fake_A.data)
            loss_discriminator_A = -torch.mean(real_A_valid) + torch.mean(fake_A_valid) + \
                                   lambda_gp * gradient_A_penalty
        elif args.loss_type == 'ralsgan':
            mixed_pred = discriminator_A(mixed)
            fake_A_pred = discriminator_A(fake_A.detach())
            loss_discriminator_A = (criterion_GAN(mixed_pred - torch.mean(fake_A_pred), valid) +
                                    criterion_GAN(fake_A_pred - torch.mean(mixed_pred), -valid)) / 2
        elif args.loss_type == 'hingegan':
            loss_real_A = torch.nn.ReLU()(1.0 - discriminator_A(mixed)).mean()
            loss_fake_A = torch.nn.ReLU()(1.0 + discriminator_A(fake_A.detach())).mean()
            loss_discriminator_A = (loss_real_A + loss_fake_A) / 2
        else:
            raise Exception("Discriminator_A Loss Error")

        discriminator_A_optimizer.zero_grad()
        loss_discriminator_A.backward()
        discriminator_A_optimizer.step()

        ######################
        # Discriminator_B
        ######################
        if args.loss_type == 'l1gan' or 'lsgan':
            loss_real_B = criterion_GAN(discriminator_B(target), valid)  # |D_B(y)-1|
            loss_fake_B = criterion_GAN(discriminator_B(fake_B.detach()), fake)  # |D_B(G(x))-0|
            loss_discriminator_B = (loss_real_B + loss_fake_B) / 2

        elif args.loss_type == 'wgan-gp':
            real_B_valid = discriminator_B(target)
            fake_B_valid = discriminator_B(fake_B)
            gradient_B_penalty = compute_gradient_penalty(discriminator_B,
                                                          target.data,
                                                          fake_B.data)
            loss_discriminator_B = -torch.mean(real_B_valid) + torch.mean(fake_B_valid) + \
                                   lambda_gp * gradient_B_penalty
        elif args.loss_type == 'ralsgan':
            target_pred = discriminator_B(target)
            fake_B_pred = discriminator_B(fake_B.detach())
            loss_discriminator_B = (criterion_GAN(target_pred - torch.mean(fake_B_pred), valid) +
                                    criterion_GAN(fake_B_pred - torch.mean(target_pred), -valid)) / 2
        elif args.loss_type == 'hingegan':
            loss_real_B = torch.nn.ReLU()(1.0 - discriminator_B(target)).mean()
            loss_fake_B = torch.nn.ReLU()(1.0 + discriminator_B(fake_B.detach())).mean()
            loss_discriminator_B = (loss_real_B + loss_fake_B) / 2
        else:
            raise Exception("Discriminator_B Loss Error")

        discriminator_B_optimizer.zero_grad()
        loss_discriminator_B.backward()
        discriminator_B_optimizer.step()

        discriminator_loss = (loss_discriminator_B + loss_discriminator_A) / 2

        niter = (epoch - 1) * len(train_loader) + i
        if args.gpu == 0:
            summary.add_scalar('Train/G_loss', generator_loss.item(), niter)
            summary.add_scalar('Train/D_loss', discriminator_loss.item(), niter)
            summary.add_scalar('Train/cyc', (loss_cycle * 20), niter)
            summary.add_scalar('Train/gan', loss_GAN.item(), niter)
            summary.add_scalar('Train/GAN_A2B', loss_GAN_A2B.item(), niter)
            summary.add_scalar('Train/GAN_B2A', loss_GAN_B2A.item(), niter)
            summary.add_scalar('Train/Cyc_A', (loss_cycle_A * 10), niter)
            summary.add_scalar('Train/Cyc_B', (loss_cycle_B * 10), niter)

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | D_loss: %f | G_loss_: %f | cyc: %f | ganA: %f | ganB: %f | ca %f | cb %f"
                  % (epoch, i, len(train_loader), discriminator_loss, generator_loss, loss_cycle * 20,
                     loss_GAN_A2B, loss_GAN_B2A, loss_cycle_A * 10, loss_cycle_B * 10))

    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


def validate(dataloader, generator_A2B, generator_B2A, discriminator_A, discriminator_B,
             criterion_GAN, criterion_cycle, criterion_idt,
             args, epoch, summary):

    generator_A2B.eval()
    generator_B2A.eval()
    discriminator_A.eval()
    discriminator_B.eval()

    score = pesq_score(dataloader, generator_A2B, generator_B2A, discriminator_A, discriminator_B,
                       criterion_GAN, criterion_cycle, criterion_idt,
                       args, epoch, summary)
    return score


if __name__ == "__main__":
    main()
