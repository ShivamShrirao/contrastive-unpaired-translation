import torch.utils.data as data


def CreateDataLoader(dataset, batch_size, workers=8, shuffle=True, distributed=True):
    if distributed:
        train_sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    else:
        train_sampler = None

    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(train_sampler is None) and shuffle,
        num_workers=workers, pin_memory=True, drop_last=False, sampler=train_sampler
    )
    return dataloader
