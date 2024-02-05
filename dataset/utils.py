import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import numpy as np
import random
from PIL import Image

from globals import CONFIG


class BaseDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.T = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        path, label = self.examples[index]
        img = Image.open(path).convert("RGB")
        img = self.T(img).to(CONFIG.dtype)
        label = torch.tensor(label).long()
        return img, label

class DomainAdaptationDataset(Dataset):
    def __init__(self, source_examples, target_examples, transform):
        self.source_examples = source_examples
        self.target_examples = target_examples
        self.T = transform

    def __len__(self):
        return len(self.source_examples)

    def __getitem__(self, index):
        src_path, src_label = self.source_examples[index]
        
        # targ_path = None
        # for t, l in self.target_examples:
        #     if l == src_label:
        #         targ_path = t
        #         break
        # if targ_path is None:
        #     print(f"Could not find target example for label {src_label}")
        targ_path, _ = random.choice(self.target_examples)
        src_img = Image.open(src_path).convert("RGB")
        targ_img = Image.open(targ_path).convert("RGB")

        src_img = self.T(src_img).to(CONFIG.dtype)
        targ_img = self.T(targ_img).to(CONFIG.dtype)

        return src_img, src_label, targ_img

######################################################
# TODO: modify 'BaseDataset' for the Domain Adaptation setting.
# Hint: randomly sample 'target_examples' to obtain targ_x
# class DomainAdaptationDataset(Dataset):
#    def __init__(self, source_examples, target_examples, transform):
#        self.source_examples = source_examples
#        self.target_examples = target_examples
#        self.T = transform
#
#    def __len__(self):
#        return len(self.source_examples)
#
#    def __getitem__(self, index):
#        src_x, src_y = ...
#        targ_x = ...
#
#        src_x = self.T(src_x)
#        targ_x = self.T(targ_x)
#        return src_x, src_y, targ_x

# [OPTIONAL] TODO: modify 'BaseDataset' for the Domain Generalization setting.
# Hint: combine the examples from the 3 source domains into a single 'examples' list
# class DomainGeneralizationDataset(Dataset):
#    def __init__(self, examples, transform):
#        self.examples = examples
#        self.T = transform
#
#    def __len__(self):
#        return len(self.examples)
#
#    def __getitem__(self, index):
#        x1, x2, x3 = self.examples[index]
#        x1, x2, x3 = self.T(x1), self.T(x2), self.T(x3)
#        targ_x = self.T(targ_x)
#        return x1, x2, x3


######################################################
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SeededDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device=""
    ):
        if not CONFIG.use_nondeterministic:
            generator = torch.Generator()
            generator.manual_seed(CONFIG.seed)

            worker_init_fn = seed_worker

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
