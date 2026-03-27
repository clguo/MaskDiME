import os
import math
import h5py
import torch
import json
import random
import numpy as np
import pandas as pd
import blobfile as bf

from os import path as osp
from PIL import Image
from mpi4py import MPI
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


# ============================================================================
# ImageFolder dataloader
# ============================================================================


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        # print("class_names----------------", class_names)
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# ============================================================================
# CelebA dataloader
# ============================================================================


def load_data_celeba(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = CelebADataset(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
        )
    while True:
        yield from loader


def load_data_aisc(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = AISCDataset(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
        )
    while True:
        yield from loader
class AISCDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'split_isic.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_isic.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        # self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            # transforms.CenterCrop(image_size),
            # transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])
        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        # print("----------------",sample)
        # print("---------------------------")
        # print("0",sample[0])
        # print("1",sample[1])
        # print("2",sample[2])
        # print("2:",sample[2:])
        # print("---------------------------")
        labels = sample[2:].to_numpy()
        # print(self.query)
        if self.query != -1:
            labels = int(labels[self.query])
            # print("labels:",labels)
        else:
            labels = torch.from_numpy(labels.astype('float32'))

        img_file = sample['image_id']

        with open(osp.join(self.data_dir, 'images', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}
def load_data_funds(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = FundsDataset(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
        )
    while True:
        yield from loader
class FundsDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'split_funds.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_funds.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['split'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            # transforms.CenterCrop(image_size),
            # transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])
        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        # print("----------------",sample)
        # print("---------------------------")
        # print("0",sample[0])
        # print("1",sample[1])
        # print("2",sample[2])
        # print("2:",sample[2:])
        # print("---------------------------")
        labels = sample[2:].to_numpy()
        # print(self.query)
        if self.query != -1:
            labels = int(labels[self.query])
            # print("labels:",labels)
        else:
            labels = torch.from_numpy(labels.astype('float32'))

        img_file = sample['filename']

        with open(osp.join(self.data_dir, 'images', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}



def load_data_kvasir(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = KvasirDataset(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
        )
    while True:
        yield from loader
class KvasirDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'split_kvasir.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_kvasir.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['split'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            # transforms.CenterCrop(image_size),
            # transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])
        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))

        img_file = sample['filename']

        with open(osp.join(self.data_dir, 'images', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}

class BDD10KCE():
    def __init__(
        self,
        data_dir,
        image_size,
        partition,
        normalize=True
    ):
        self.cropSize = (image_size * 2, image_size)
        self.normalize = normalize
        self.root = osp.join(data_dir, 'images', '100k', partition)
        self.images = os.listdir(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        with open(osp.join(self.root, name), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transforms(img), 1

    def transforms(self, raw_image):
        new_width, new_height = (self.cropSize[1], self.cropSize[0])
        image = TF.resize(raw_image, (new_width, new_height), Image.BICUBIC)
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if self.normalize else image
        return image

class CelebADataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_celeba.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[3:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']

        with open(osp.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}
class CelebAMVDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_celeba.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']

        with open(osp.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}

class BDDOIADataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_size,
        partition='train',
        augment=False,
        query_label=-1,
        normalize=True,
    ):

        super(BDDOIADataset, self).__init__()

        self.imageRoot = osp.join(data_dir, 'data')
        self.gtRoot = osp.join(data_dir, f'{partition}_25k_images_actions.json')
        self.reasonRoot = osp.join(data_dir, f'{partition}_25k_images_reasons.json')
        self.cropSize = (image_size * 2, image_size)
        self.normalize = normalize
        self.query = query_label

        with open(self.gtRoot) as json_file:
            data = json.load(json_file)
        with open(self.reasonRoot) as json_file:
            reason = json.load(json_file)

        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = osp.join(self.imageRoot, img['file_name'])
                if osp.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.LongTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[ind]

        raw_image = Image.open(imgName).convert('RGB')
        target = np.array(self.targets[ind], dtype=np.int64)
        reason = np.array(self.reasons[ind], dtype=np.int64)

        image, target, reason = self.transforms(raw_image, target, reason)
        if self.query != -1:
            target = target[self.query]

        return image, target

    def transforms(self, raw_image, target, reason):

        new_width, new_height = (self.cropSize[1], self.cropSize[0])

        image = TF.resize(raw_image, (new_width, new_height), Image.BICUBIC)
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if self.normalize else image

        target = torch.FloatTensor(target)[0:4]
        reason = torch.FloatTensor(reason)

        return image, target, reason
class BDD10KCE():
    def __init__(
        self,
        data_dir,
        image_size,
        partition,
        normalize=True
    ):
        self.cropSize = (image_size * 2, image_size)
        self.normalize = normalize
        self.root = osp.join(data_dir, 'images', '100k', partition)
        self.images = os.listdir(self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        with open(osp.join(self.root, name), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transforms(img), 1

    def transforms(self, raw_image):
        new_width, new_height = (self.cropSize[1], self.cropSize[0])
        image = TF.resize(raw_image, (new_width, new_height), Image.BICUBIC)
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if self.normalize else image
        return image


class CelebAHQDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        from io import StringIO
        # read annotation files
        with open(osp.join(data_dir, 'CelebAMask-HQ-attribute-anno.txt'), 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' +  datastr.replace('  ', ' ')

        with open(osp.join(data_dir, 'CelebA-HQ-to-CelebA-mapping.txt'), 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']

        mapstr = ' '.join(mapstr)

        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        mapping_df.rename(columns={'orig_file': 'image_id'}, inplace=True)
        partition_df = pd.merge(mapping_df, partition_df, on='image_id')

        self.data_dir = data_dir

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        # print(labels)
        # print(self.query)
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['idx']

        with open(osp.join(self.data_dir, 'CelebA-HQ-img', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}

class CelebAMiniVal(CelebAMVDataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition=None,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        self.data = pd.read_csv('/zhome/53/8/212865/projects/counterfactual/DiME/utils/minival.csv').iloc[:, 1:]
        self.data = self.data[shard::num_shards]
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x,
        ])
        self.data_dir = data_dir
        self.class_cond = class_cond
        self.query = query_label




class CelebATDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_celeba.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']

        with open(osp.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}
class CelebAMiniTra(CelebATDataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition=None,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        self.data = pd.read_csv('/zhome/53/8/212865/projects/counterfactual/DiME/utils/train_minival.csv').iloc[:, 1:]
        self.data = self.data[shard::num_shards]
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x,
        ])
        self.data_dir = data_dir
        self.class_cond = class_cond
        self.query = query_label


