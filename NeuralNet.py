import os
import numpy as np
import torch
import random
import time
import timeit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from tabulate import tabulate
from torch import nn, optim
import Config as cfg
from Model_StatsLogger import Model_StatsLogger


class NeuralNet:
    def __init__(self, arch, dataset, epochs, threads, muxing, seed, cuda_conv,
                      LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, device, verbose, gpus, distributed, save_all_states, model_path):

        for gpu_num in range(gpus):
            cfg.LOG.write('NeuralNet __init__: arch={}, dataset={} threads={}, muxing={} epochs={}, cuda_conv={} '
                          'LR={} LRD={} WD={} MOMENTUM={} GAMMA={} MILESTONES={} '
                          'device={} verbose={} gpus={} distributed={} model_path={}'
                          .format(arch, dataset, threads, muxing, epochs, cuda_conv,
                                  LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, device, verbose, gpus, distributed, model_path), terminal=(gpu_num == 0), gpu_num=gpu_num)
            cfg.LOG.write('Seed = {}'.format(seed), terminal=(gpu_num == 0), gpu_num=gpu_num)

        if device =='cpu':
            self.device = torch.device('cpu')
        elif device =='cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            cfg.LOG.write('WARNING: Found no valid GPU device - Running on CPU')

        self.threads = threads
        self.muxing = muxing
        self.epochs = epochs
        self.cuda_conv = cuda_conv
        self.device = device
        self.verbose = verbose
        self.compute_1_thread = 0
        self.convs_number = 0
        self.LR = LR
        self.LRD = LRD
        self.WD = WD
        self.MOMENTUM = MOMENTUM
        self.GAMMA = GAMMA
        self.MILESTONES = MILESTONES
        self.gpus = gpus
        self.distributed = distributed
        self.model_path = model_path
        self.save_all_states = save_all_states

        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.enabled = False

        self.arch = '{}_{}'.format(arch, dataset)
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.cuda() if device == 'cuda' else self.criterion

        if self.cuda_conv:
            # threads 0 mse_on False
            assert threads is None, 'Error - cannot run two different models at the same time'
            self.cuda_model = cfg.MODELS[self.arch](threads=0, device=device, verbose=verbose)
            self.cuda_model = self.cuda_model.cuda() if device=='cuda' else self.cuda_model
            self.cuda_conv_stats = Model_StatsLogger(0, seed, verbose)
            self.cuda_optmizer = optim.SGD(self.cuda_model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM)
            self.train_scheduler_cuda = optim.lr_scheduler.MultiStepLR(self.cuda_optmizer, milestones=MILESTONES, gamma=GAMMA)  # learning rate decay

            for gpu_num in range(gpus):
                graphs_path = os.path.join(cfg.LOG.graph_path[gpu_num], 'Cuda_Conv')
                os.mkdir('{}'.format(graphs_path))

        else:
            assert threads, 'Error - Need to provide number of threads for non baseline simulation'
            self.smt_model = cfg.MODELS[self.arch](threads=threads, device=device, verbose=verbose, muxing=muxing)
            self.smt_model = self.smt_model.cuda() if device == 'cuda' else self.smt_model
            self.smt_stats = Model_StatsLogger(threads, seed, verbose, muxing=muxing)
            self.smt_optimizer = optim.SGD(self.smt_model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM)
            self.smt_train_scheduler = optim.lr_scheduler.MultiStepLR(self.smt_optimizer, milestones=MILESTONES, gamma=GAMMA)

            for gpu_num in range(gpus):
                graphs_path = os.path.join(cfg.LOG.graph_path[gpu_num], '{}_threads_Conv_m{}'.format(threads, muxing))
                os.mkdir('{}'.format(graphs_path))

        self.load_models()


    def load_models(self, gpu=0, disributed = 0):
        if self.model_path is not None:
            if os.path.isfile(self.model_path):
                chkp = torch.load(self.model_path)
            else:
                assert 0, 'Error: Cannot find model path {}'.format(self.model_path)

            assert (self.arch == chkp['arch'])
            try:
                if self.cuda_conv:
                    # threads 0 mse_on False
                    if disributed == 0:
                        self.cuda_model.load_state_dict(chkp['state_dict'], strict=True)
                    else:
                        self.cuda_model.module.load_state_dict(chkp['state_dict'], strict=True)
                    self.cuda_model = self.cuda_model.cuda() if self.device == 'cuda' else self.cuda_model
                    self.cuda_optmizer.load_state_dict(chkp['optimizer'])
                    self.train_scheduler_cuda.load_state_dict(chkp['scheduler'])
                    cfg.LOG.write('Loaded model successfully to CUDA model{}'.format('' if disributed == 0 else ' in distributed mode'), terminal=(gpu == 0), gpu_num=gpu)
                else:
                    if disributed == 0:
                        self.smt_model.load_state_dict(chkp['state_dict'], strict=True)
                    else:
                        self.smt_model.module.load_state_dict(chkp['state_dict'], strict=True)
                    self.smt_model = self.smt_model.cuda() if self.device == 'cuda' else self.smt_model
                    self.smt_optimizer.load_state_dict(chkp['optimizer'])
                    self.smt_train_scheduler.load_state_dict(chkp['scheduler'])
                    cfg.LOG.write('Loaded model successfully to {} SMT mode with M{}{}'.format(
                        self.smt_model.threads if disributed == 0 else self.smt_model.module.threads,
                        self.smt_model.muxing if disributed == 0 else self.smt_model.module.muxing,
                        '' if disributed == 0 else ' in distributed mode'), terminal=(gpu == 0), gpu_num=gpu)

                cfg.LOG.write('Loaded models successfully{}'.format('' if disributed == 0 else ' in distributed mode'), terminal=(gpu == 0), gpu_num=gpu)
            except RuntimeError as e:
                cfg.LOG.write('Loading model state warning, please review', terminal=(gpu == 0), gpu_num=gpu)
                cfg.LOG.write('{}'.format(e), terminal=(gpu == 0), gpu_num=gpu)

    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)


    def update_batch_size(self, train_set_size, test_set_size):
        if self.cuda_conv:
            self.cuda_conv_stats.progress['train'].update_batch_num(train_set_size)
            self.cuda_conv_stats.progress['test'].update_batch_num(test_set_size)
        else:
            self.smt_stats.progress['train'].update_batch_num(train_set_size)
            self.smt_stats.progress['test'].update_batch_num(test_set_size)

    def reset_accuracy_logger(self, mode):
        if self.cuda_conv:
            self.cuda_conv_stats.losses[mode].reset()
            self.cuda_conv_stats.top1[mode].reset()
            self.cuda_conv_stats.top5[mode].reset()
        else:
            self.smt_stats.losses[mode].reset()
            self.smt_stats.top1[mode].reset()
            self.smt_stats.top5[mode].reset()

    def switch_to_train_mode(self):
        # switch to train mode
        if self.cuda_conv:
            self.cuda_model.train()
        else:
            self.smt_model.train()

    def switch_to_test_mode(self):
        if self.cuda_conv:
            self.cuda_model.eval()
        else:
            self.smt_model.eval()


    def log_data_time(self, end, mode):
        # switch to train mode
        if self.cuda_conv:
            self.cuda_conv_stats.data_time[mode].update(time.time() - end)
        else:
            self.smt_stats.data_time[mode].update(time.time() - end)

    def log_batch_time(self, end, mode):
        # switch to train mode
        if self.cuda_conv:
            self.cuda_conv_stats.batch_time[mode].update(time.time() - end)
        else:
            self.smt_stats.batch_time[mode].update(time.time() - end)


    def compute_forward(self, images):
        cuda_out = None
        smt_out = None
        if self.cuda_conv:
            cuda_out = self.cuda_model(images)
        else:
            smt_out = self.smt_model(images)
        return cuda_out, smt_out

    def compute_loss(self,cuda_out, smt_out, target):
        cuda_loss = None
        smt_loss = None
        if self.cuda_conv:
            cuda_loss = self.criterion(cuda_out, target)
        else:
            smt_loss = self.criterion(smt_out, target)
        return cuda_loss, smt_loss

    def measure_accuracy_log(self, cuda_out, smt_out, cuda_loss, smt_loss, target, images_size, topk, mode):
        if self.cuda_conv:
            cuda_acc1, cuda_acc5 = self.cuda_conv_stats.accuracy(cuda_out, target, topk)
            self.cuda_conv_stats.losses[mode].update(cuda_loss.item(), images_size)
            self.cuda_conv_stats.top1[mode].update(cuda_acc1[0], images_size)
            self.cuda_conv_stats.top5[mode].update(cuda_acc5[0], images_size)
        else:
            acc1, acc5 = self.smt_stats.accuracy(smt_out, target, topk)
            self.smt_stats.losses[mode].update(smt_loss.item(), images_size)
            self.smt_stats.top1[mode].update(acc1[0], images_size)
            self.smt_stats.top5[mode].update(acc5[0], images_size)


    def zero_gradients(self):
        if self.cuda_conv:
            self.cuda_optmizer.zero_grad()
        else:
            self.smt_optimizer.zero_grad()

    def backward_compute(self, cuda_loss, smt_loss):
        if self.cuda_conv:
            cuda_loss.backward()
        else:
            smt_loss.backward()

    def compute_step(self):
        if self.cuda_conv:
            self.cuda_optmizer.step()
        else:
            self.smt_optimizer.step()


    def print_progress(self, epoch, batch, mode, gpu_num):
        if self.cuda_conv:
            self.cuda_conv_stats.progress[mode].print(' 0 cuda conv    ', epoch, batch, gpu_num)
        else:
            self.smt_stats.progress[mode].print(' {} thread conv M{}'.format(self.smt_stats.threads, self.smt_stats.muxing), epoch, batch, gpu_num)


    def log_history(self, epoch, mode='train'):
        if self.cuda_conv:
            self.cuda_conv_stats.log_history(epoch, mode)
        else:
            self.smt_stats.log_history(epoch, mode)


    def set_learning_rate(self):
        if self.LRD == 1:
            if self.cuda_conv:
                self.train_scheduler_cuda.step()
            else:
                self.smt_train_scheduler.step()


    def print_epoch_stats(self, epoch, mode='train', gpu_num=0):
        if mode == 'train':
            cfg.LOG.write_title("Training Epoch {} Stats".format(epoch), terminal=(gpu_num == 0), gpu_num=gpu_num)
        elif mode == 'test':
            cfg.LOG.write_title("Testing Epoch {} Stats".format(epoch), terminal=(gpu_num == 0), gpu_num=gpu_num)
        else:
            raise NotImplementedError

        stats_headers = ["Conv", "Avg. Loss", "Avg. Acc1", "Avg. Acc5"]
        stats = []
        if self.cuda_conv:
            stats.append(("{} Thread conv".format(self.cuda_conv_stats.threads), self.cuda_conv_stats.losses[mode].getAverage(),
                          self.cuda_conv_stats.top1[mode].getAverage(),
                          self.cuda_conv_stats.top5[mode].getAverage()))
        else:
            stats.append(("{} Thread conv M{}".format(self.smt_stats.threads, self.smt_stats.muxing), self.smt_stats.losses[mode].getAverage(),
                          self.smt_stats.top1[mode].getAverage(),
                          self.smt_stats.top5[mode].getAverage()))
        cfg.LOG.write(tabulate(stats, headers=stats_headers, tablefmt="grid"), date=False, terminal=(gpu_num == 0), gpu_num=gpu_num)


    def update_best_acc(self, epoch):
        if epoch >= 80 or self.save_all_states == 1:
            if self.cuda_conv:
                top1_acc = self.cuda_conv_stats.top1['test'].avg
                if top1_acc > self.cuda_conv_stats.best_top1_acc:
                    self.cuda_conv_stats.best_top1_acc = top1_acc
                    self.cuda_conv_stats.best_top1_epoch = epoch
                    self._save_state(epoch=epoch, best_top1_acc=top1_acc.item(), model=self.cuda_model, optimizer=self.cuda_optmizer, scheduler=self.train_scheduler_cuda, desc='Cuda_Conv')
            else:
                top1_acc = self.smt_stats.top1['test'].avg
                if top1_acc > self.smt_stats.best_top1_acc:
                    self.smt_stats.best_top1_acc = top1_acc
                    self.smt_stats.best_top1_epoch = epoch
                    self._save_state(epoch=epoch, best_top1_acc=top1_acc.item(), model=self.smt_model, optimizer=self.smt_optimizer, scheduler=self.smt_train_scheduler,desc='{}_threads_Conv_M{}'.format(self.smt_stats.threads, self.smt_stats.muxing))


    def export_stats(self, gpu = 0):
        #export stats results
        if self.cuda_conv:
            self.cuda_conv_stats.export_stats(gpu=gpu)
        else:
            self.smt_stats.export_stats(gpu=gpu)


    def plot_results(self, gpu = 0):
        #plot results for each convolution
        if self.cuda_conv:
            self.cuda_conv_stats.plot_results(gpu=gpu)
        else:
            self.smt_stats.plot_results(gpu=gpu)


    def distribute_model(self, gpu):
        self.criterion = self.criterion.cuda(gpu)
        if self.cuda_conv:
            self.cuda_model.cuda(gpu)
            self.cuda_model = nn.parallel.DistributedDataParallel(self.cuda_model, device_ids=[gpu])
            self.cuda_optmizer = optim.SGD(self.cuda_model.parameters(), lr=self.LR, weight_decay=self.WD, momentum=self.MOMENTUM)
            self.train_scheduler_cuda = optim.lr_scheduler.MultiStepLR(self.cuda_optmizer, milestones=self.MILESTONES, gamma=self.GAMMA)  # learning rate decay
        else:
            self.smt_model.cuda(gpu)
            self.smt_model = nn.parallel.DistributedDataParallel(self.smt_model, device_ids=[gpu])
            self.smt_optimizer = optim.SGD(self.smt_model.parameters(), lr=self.LR, weight_decay=self.WD, momentum=self.MOMENTUM)
            self.smt_train_scheduler = optim.lr_scheduler.MultiStepLR(self.smt_optimizer, milestones=self.MILESTONES, gamma=self.GAMMA) # learning rate decay

        self.load_models(gpu=gpu, disributed=1)


    def train(self, epoch, train_gen, gpu = 0):
        cfg.LOG.write_title('Training Epoch {}'.format(epoch), terminal=(gpu==0), gpu_num=gpu)

        if gpu == 0:
            self.print_verbose('NeuralNet train() epoch={}'.format(epoch), 2)
        self.reset_accuracy_logger('train')
        self.switch_to_train_mode()

        end = time.time()
        torch.cuda.synchronize()
        start = timeit.default_timer()

        for i, (images, target) in enumerate(train_gen):
            # measure data loading time

            self.log_data_time(end, 'train')

            if self.device == 'cuda':
                images = images.cuda(non_blocking=True, device=gpu)
                target = target.cuda(non_blocking=True, device=gpu)

            cuda_out, smt_out = self.compute_forward(images)

            cuda_loss, smt_loss = self.compute_loss(cuda_out, smt_out, target)

            # measure accuracy and record logs
            self.measure_accuracy_log(cuda_out, smt_out, cuda_loss, smt_loss, target, images.size(0), topk=(1, 5), mode='train')


            # compute gradient and do SGD step
            self.zero_gradients()

            self.backward_compute(cuda_loss, smt_loss)

            self.compute_step()

            # measure elapsed time
            self.log_batch_time(end, mode='train')

            end = time.time()

            if i % cfg.BATCH_SIZE == 0:
                self.print_progress(epoch, i, mode='train', gpu_num=gpu)

        self.set_learning_rate()
        torch.cuda.synchronize()
        stop = timeit.default_timer()
        self.log_history(epoch, mode='train')
        self.print_epoch_stats(epoch=epoch, mode='train', gpu_num=gpu)
        cfg.LOG.write('Total Epoch {} Time: {:6.2f} seconds'.format(epoch,stop - start), terminal=(gpu == 0), gpu_num=gpu)

        return


    def test_set(self, epoch, test_gen, gpu = 0):
        cfg.LOG.write_title('Testing Epoch {}'.format(epoch), terminal=(gpu==0), gpu_num=gpu)

        if gpu == 0:
            self.print_verbose('NeuralNet test_set() epoch={}'.format(epoch), 2)
        self.reset_accuracy_logger('test')
        self.switch_to_test_mode()

        with torch.no_grad():
            end = time.time()
            torch.cuda.synchronize()
            start = timeit.default_timer()

            for i, (images, target) in enumerate(test_gen):

                self.log_data_time(end, 'test')

                if self.device == 'cuda':
                    images = images.cuda(non_blocking=True, device=gpu)
                    target = target.cuda(non_blocking=True, device=gpu)

                cuda_out, smt_out = self.compute_forward(images)

                cuda_loss, smt_loss = self.compute_loss(cuda_out, smt_out, target)

                # measure accuracy and record logs
                self.measure_accuracy_log(cuda_out, smt_out, cuda_loss, smt_loss, target, images.size(0), topk=(1, 5), mode='test')

                # measure elapsed time
                self.log_batch_time(end, mode='test')

                end = time.time()

                if i % cfg.BATCH_SIZE == 0:
                    self.print_progress(epoch, i, mode='test', gpu_num=gpu)

            torch.cuda.synchronize()
            stop = timeit.default_timer()
            self.log_history(epoch, mode='test')

            self.print_epoch_stats(epoch=epoch, mode='test', gpu_num=gpu)
            cfg.LOG.write('Total Test Time: {:6.2f} seconds'.format(epoch, stop - start), terminal=(gpu == 0), gpu_num=gpu)


            if gpu == 0:
                self.update_best_acc(epoch)
        return