#
import sys
import torch
import os
from datetime import datetime
import argparse
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, ExtendDetBenchTrain
from effdet.efficientdet import HeadNet
import warnings
torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
LOCAL_RANK ='0'
SEED = 42
TRAIN_ROOT_PATH = '/home/tuanho/Workspace/datasets/dataset/train'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

class TrainGlobalConfig:
    num_workers = 2
    batch_size = 2
    n_epochs = 70 # n_epochs = 40
    lr = 0.0001

    folder = 'weights/effdet7-cutmix-augmix_fold1'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    # SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    # scheduler_params = dict(
    #     max_lr=0.001,
    #     epochs=n_epochs,
    #     steps_per_epoch=int(len(train_dataset) / batch_size),
    #     pct_start=0.1,
    #     anneal_strategy='cos', 
    #     final_div_factor=10**5
    # )
    
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    # --------------------

class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        if self.test or random.random() > 0.5:
            image, boxes = self.load_image_and_boxes(index)
        else:
            image, boxes = self.load_cutmix_image_and_boxes(index)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, train_loader, validation_loader, config, optimizer, scheduler, log, epoch, base_dir, log_path, device):
    best_summary_loss = 10**5
    for e in range(config.n_epochs):
        if config.verbose:
            lr = optimizer.param_groups[0]['lr']
            timestamp = datetime.utcnow().isoformat()
            log(f'\n{timestamp}\nLR: {lr}', config, log_path)

        t = time.time()
        summary_loss = train_one_epoch(model, train_loader, config, optimizer, device)
        log(f'[RESULT]: Train. Epoch: {epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}', config, log_path)
        save(f'{base_dir}/last-checkpoint.pt', model, optimizer, scheduler, best_summary_loss, epoch)

        t = time.time()
        summary_loss = validation(model, validation_loader, config, device)

        log(f'[RESULT]: Val. Epoch: {epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}', config, log_path)
        if summary_loss.avg < best_summary_loss:
            best_summary_loss = summary_loss.avg
            model.eval()
            save(f'{base_dir}/best-checkpoint-{str(epoch).zfill(3)}epoch.pt', model, optimizer, scheduler, best_summary_loss, epoch)
            for path in sorted(glob(f'{base_dir}/best-checkpoint-*epoch.pt'))[:-3]:
                os.remove(path)

        if config.validation_scheduler:
            scheduler.step(metrics=summary_loss.avg)

        epoch += 1

def validation(model, val_loader, config, device):
    model.eval()
    summary_loss = AverageMeter()
    t = time.time()
    for step, (images, targets, image_ids) in enumerate(val_loader):
        if config.verbose:
            if step % config.verbose_step == 0:
                print(
                    f'Val Step {step}/{len(val_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                )
        with torch.no_grad():
            images = torch.stack(images)
            batch_size = images.shape[0]
            images = images.to(device).float()
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]

            # loss, _, _ = self.model(images, boxes, labels)
            loss = model(images, boxes, labels)
            summary_loss.update(loss['loss'].detach().item(), batch_size)

    return summary_loss

def train_one_epoch(model, train_loader, config, optimizer, device):
    model.train()
    summary_loss = AverageMeter()
    t = time.time()
    for step, (images, targets, image_ids) in enumerate(train_loader):
        if config.verbose:
            if step % config.verbose_step == 0:
                print(
                    f'Train Step {step}/{len(train_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                )
        
        images = torch.stack(images)
        images = images.to(device).float()
        batch_size = images.shape[0]
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]

        optimizer.zero_grad()
        
        # loss, _, _ = self.model(images, boxes, labels)
        loss = model(images, boxes, labels)
        loss['loss'].backward()

        summary_loss.update(loss['loss'].detach().item(), batch_size)

        optimizer.step()

        if config.step_scheduler:
            scheduler.step()

    return summary_loss

def save(path, model, optimizer, scheduler, best_summary_loss, epoch):
    model.eval()
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_summary_loss': best_summary_loss,
        'epoch': epoch,
    }, path)

def load(path ,model, optimizer, scheduler, best_summary_loss, epoch):
    checkpoint = torch.load(path)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_summary_loss = checkpoint['best_summary_loss']
    epoch = checkpoint['epoch'] + 1
    
def log(message, config, log_path):
    if config.verbose:
        print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    args.distributed = False
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    if args.distributed:
        print(f'Training in distributed mode with multiple processes, 1 GPU per process. Process {args.rank}, total {args.world_size}.')
    else:
        print('Training with a single process on 1 GPU.')

    torch.manual_seed(SEED + args.rank)   

    seed_everything(SEED)
    marking = pd.read_csv('/home/tuanho/Workspace/datasets/dataset/train.csv')

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]
    marking.drop(columns=['bbox'], inplace=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    config = get_efficientdet_config('tf_efficientdet_d7')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('weights/tf_efficientdet_d7_53-6d1d7a95.pth')
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    model = ExtendDetBenchTrain(net, config)
    # model = DetBenchTrain(net, config)
    # device = torch.device('cuda:0')
    model.cuda()
    fold_number = 1
    train_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        marking=marking,
        transforms=get_train_transforms(),
        test=False,
    )
    validation_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        marking=marking,
        transforms=get_valid_transforms(),
        test=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )
    config = TrainGlobalConfig
    epoch = 0
    base_dir = f'./{config.folder}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    log_path = f'{base_dir}/log.txt'
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ] 

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    scheduler = config.SchedulerClass(optimizer, **config.scheduler_params)
    log(f'Fitter prepared. Device is {args.device}', config, log_path)

    train(model, train_loader, val_loader, config, optimizer, scheduler,log, epoch, base_dir, log_path, args.device)

if __name__ == '__main__':
    main()
