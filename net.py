import io
import itertools
import math
import os
import logging

import numpy as np
import torch
from torch import tensor
import torch.nn.functional as F
import torchvision
from apex import amp
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path

import cv2
logger = logging.getLogger(__name__)

scale_factor = 2
img_width = int(480 / scale_factor)
img_height = int(680 / scale_factor)
channels = 3

img_art_x = int((26 / 480) * img_width)
img_art_y = int((54 / 680) * img_height)
img_art_w = int((426 / 480) * img_width)
img_art_h = int((326 / 680) * img_height)


def cnn_to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(-1, channels, img_height, img_width)
    return x


def scale(x):
    x = torch.mul(x, 255.0)
    return x


def to_bytes(x):
    return x.to(torch.uint8)


class CNNAE(nn.Module):
    def __init__(self, using_amp=True):
        super(CNNAE, self).__init__()
        dtype_to_use = torch.half
        if not using_amp:
            dtype_to_use = torch.float

        self.mean = torch.tensor([0.5, 0.5, 0.5], dtype=dtype_to_use).cuda()
        self.std = torch.tensor([0.5, 0.5, 0.5], dtype=dtype_to_use).cuda()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (4, 3), stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, 5, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 8, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x_p = x.clone()
        x_p = x_p.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])

        x_p = self.encoder(x_p)
        x_p = self.decoder(x_p)

        x_p = 0.5 * (x_p + 1.0)
        x_p = x_p.clamp(0, 1)
        x_p = torch.mul(x_p, 255.0)
        x_p = x_p.view(-1, channels, img_height, img_width)
        return x_p

    def to_img(self, x):
        return cnn_to_img(x)

    def get_criterion_optimizer(self):
        criterion = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return criterion, optimizer

    def get_transforms(self, using_amp=True):
        base_transforms = [
            transforms.Resize((img_height, img_width), interpolation=3),
            transforms.ToTensor(),
            transforms.Lambda(scale)
        ]

        if using_amp:
            base_transforms.append(transforms.Lambda(to_bytes))

        return base_transforms


class MyDatasetCacheLoader(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            transform,
            use_file,
            cache_post_name,
            top=None):
        self.path = path
        self.cache_file_post_name = f'.cached.{cache_post_name}'
        self.cach_path = os.path.join(self.path, '_cache_tensors_')
        if not os.path.exists(self.cach_path):
            os.mkdir(self.cach_path)
        self.transform = transform
        self._cached_files = []

        for idx, file_path in enumerate(sorted((p for p in os.listdir(self.path) if p.endswith('.jpg')))):
            if top is not None and (idx % top) == 0 and idx > 0:
                break
            full_path = os.path.join(self.path, file_path)
            self._cached_files.append((file_path, None))
            if not self.cached_tensor_exists(idx):
                if use_file:
                    file_bytes = full_path
                else:
                    with open(full_path, 'rb') as fp:
                        file_bytes = io.BytesIO(bytes(fp.read()))
                self._cached_files[idx] = (file_path, file_bytes)

        self._num_of_file = len(self._cached_files)

    def cached_tensor_exists(self, index):
        file_name, _ = self._cached_files[index]
        return os.path.exists(os.path.join(self.cach_path, file_name+self.cache_file_post_name))

    def open_cached_tensor(self, index):
        file_name, _ = self._cached_files[index]
        cached_file = os.path.join(
            self.cach_path, file_name+self.cache_file_post_name)
        with open(cached_file, 'rb') as fp:
            return torch.load(fp)

    def save_cached_tensor(self, index):
        file_name, buffered_file = self._cached_files[index]
        cached_file = os.path.join(
            self.cach_path, file_name+self.cache_file_post_name)

        if isinstance(buffered_file, str):
            with open(buffered_file, 'rb') as bf:
                img = Image.open(bf).convert('RGB')
        else:
            img = Image.open(buffered_file).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        with open(cached_file, 'wb') as fp:
            torch.save(img, fp)

        return img

    def __getitem__(self, index):

        if not self.cached_tensor_exists(index):
            return self.save_cached_tensor(index)

        return self.open_cached_tensor(index)

    def __len__(self):
        return self._num_of_file

    def __del__(self):
        for (_file_name, file_ptr) in self._cached_files:
            if file_ptr is not None:
                if not isinstance(file_ptr, str):
                    file_ptr.close()


class BatchCacheIter:
    def __init__(self, cache_loader):
        self.cache_loader = cache_loader
        self.cur_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx < len(self):
            ret_val = self.cache_loader.get_batch(self.cur_idx)
            self.cur_idx += 1
            return ret_val
        else:
            raise StopIteration

    def __len__(self):
        return len(self.cache_loader)


class BatchCacheLoader:
    def __init__(self, dataloader, cache_path, max_cache, using_amp=True):
        _cache_paths = []
        self._cache_data = []
        self._cached_tensors = {}
        self._total_len = 0
        self._max_cache = max_cache
        self._using_amp = using_amp

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        batch_ext_name = "batch.cached"
        if not self._using_amp:
            batch_ext_name = "batch-full.cached"

        paths_exist = (os.path.exists(os.path.join(
            cache_path, f'{idx}.{batch_ext_name}')) for idx in range(len(dataloader)))
        cached_files_exist = all(paths_exist)

        if not cached_files_exist:
            for idx, data in enumerate(dataloader):
                batch_path = os.path.join(
                    cache_path, f'{idx}.{batch_ext_name}')
                if not os.path.exists(batch_path):
                    logger.info('Writing Batch Cache to %s', batch_path)
                    with open(batch_path, 'wb') as cp:
                        torch.save(data, cp)
                self._total_len += 1
        else:
            self._total_len = len(dataloader)
            logger.info('All %d batches cached', self._total_len)

        del dataloader

        for idx in range(self._total_len):
            batch_path = os.path.join(cache_path, f'{idx}.{batch_ext_name}')
            if self._max_cache is not None and idx >= self._max_cache:
                logger.info('Loading pointer cache from %s', batch_path)
                file_bytes = batch_path
                can_use_cached_tensor = False
            else:
                logger.info('Loading raw cache from %s', batch_path)
                file_bytes = open(batch_path, 'rb')
                can_use_cached_tensor = True
            self._cache_data.append((file_bytes, can_use_cached_tensor))

    def get_batch(self, idx):
        (op, can_use_cached_tensor) = self._cache_data[idx]
        if not can_use_cached_tensor:
            with open(op, 'rb') as cp:
                return torch.load(cp)
        else:
            if op is not None:
                op.seek(0, 0)
                tensor_data = torch.load(op)
                op.close()
                self._cache_data[idx] = (None, can_use_cached_tensor)
                if self._using_amp:
                    self._cached_tensors[idx] = tensor_data.half()
                else:
                    self._cached_tensors[idx] = tensor_data.float()

            self._cached_tensors[idx].pin_memory()
            return self._cached_tensors[idx]

    def __len__(self):
        return self._total_len

    def __iter__(self):
        return BatchCacheIter(self)


def train_batch(model, batch_idx, img, criterion, optimizer, dataset_len, gt_written, using_amp=True):
    img = img.cuda(non_blocking=True).requires_grad_()
    output = model(img)
    loss = criterion(output, img)
    logger.info('Train Step [{}/{}] ({:.0f}%)'.format(batch_idx,
                                                      dataset_len, float(batch_idx * 100.0 / dataset_len)))
    optimizer.zero_grad()
    if using_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    optimizer.step()

    if batch_idx == (dataset_len - 1):
        return loss.data, img, output.data.cpu()
    return loss.data, None, None


def start_training(cache_location: Path, results_dir: Path, batch_size: int):
    use_amp = False
    model = CNNAE(use_amp)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion, optimizer = model.get_criterion_optimizer()

    model_name = type(model).__name__
    cache_post_name = 'byte'

    if torch.cuda.is_available() and use_amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O1', loss_scale=16.0)
    else:
        model_name = f'{model_name}_vae_noamp'
        cache_post_name = 'float'
        batch_size = int(max((batch_size / 2, 1)))

    data_transforms = transforms.Compose(model.get_transforms(use_amp))

    cards_dataset = MyDatasetCacheLoader(
        path=str(cache_location),
        transform=data_transforms,
        use_file=False,
        cache_post_name=cache_post_name)

    dataset_loader = torch.utils.data.DataLoader(
        cards_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    dataset_loader = BatchCacheLoader(
        dataset_loader,
        cache_location / 'card_store' / 'cards' /
        'download' / f'_batch_cache_{batch_size}_',
        None,
        use_amp)

    model_path = results_dir / f'{model_name}.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()

    img_name_template = 'img_{}.png'
    gt_img_path = results_dir / img_name_template.format('gt')
    gt_written = gt_img_path.exists()

    if isinstance(model, CNNAE):
        epoch = 0
        dataset_size = int(len(cards_dataset) / batch_size)
        while epoch < 10000:
            model.train()

            train_batches_itr = (train_batch(model,
                                             batch_idx,
                                             img,
                                             criterion,
                                             optimizer,
                                             dataset_size,
                                             gt_written,
                                             use_amp) for (
                batch_idx, img) in enumerate(dataset_loader))

            train_results = list(train_batches_itr)
            cuda_img, output = next(((input_img, output_img) for (
                _, input_img, output_img) in train_results if output_img is not None))
            loss_results = torch.stack(
                tuple((loss_result for (loss_result, _, _) in train_results)), 0)
            average_loss = loss_results.mean().item()
            max_loss = loss_results.max().item()
            min_loss = loss_results.min().item()

            logger.info('epoch [{}/{}]\tMin-Loss:{:.3f}\tAvg-Loss:{:.3f}\tMax-Loss{:.3f}'
                        .format(epoch+1, 100, min_loss, average_loss, max_loss))

            if epoch % 1 == 0:
                pic = output.cpu().data[:10]
                pic = torch.div(pic.float(), 255.0)
                if not gt_written:
                    if not gt_img_path.exists():
                        gt_pic = cuda_img.cpu().data[:10]
                        gt_pic = torch.div(gt_pic.float(), 255.0)
                        torchvision.utils.save_image(gt_pic, str(gt_img_path))
                        gt_written = True
                    else:
                        gt_written = True
                img_path = img_name_template.format(epoch)
                torchvision.utils.save_image(pic, img_path)
                torch.save(model.state_dict(), model_path)
            epoch += 1
