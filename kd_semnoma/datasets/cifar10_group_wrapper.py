import math
import torch
import argparse
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

# Compution device
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

# Download the datasets
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Wrapper that groups samples together
class GroupingWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, csi, group_size, num_samples=-1, seed=1):
        super().__init__()
        self.dataset = dataset
        self.group_size = group_size

        self.csi = csi
        # Decide how many groups of samples are created
        if num_samples == -1:  
            self.indices = torch.combinations(
                torch.arange(0, len(dataset)), r=self.group_size, with_replacement=False
            )
            self.num_samples = self.indices.size(0)
        elif num_samples == 0:
            self.num_samples = math.ceil(float(len(self.dataset)) / self.group_size) 
            num_new_elements = len(self.dataset)
            if len(self.dataset) % self.group_size != 0:                             
                num_new_elements += self.group_size - len(self.dataset) % self.group_size

            self.indices = torch.arange(0, num_new_elements) % len(self.dataset)
            self.indices = self.indices.reshape(-1, self.group_size) 
        else:
            self.num_samples = num_samples
            # Generate random combinations of indices
            r = torch.randint(0, len(dataset), (3 * self.num_samples, self.group_size)) 
            self.indices = r.unique(sorted=False, dim=0)[: self.num_samples, :].long()

            if self.indices.size(0) < self.num_samples:
                self.num_samples = self.indices.size(0)

        shuffled_indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )

        self.indices = shuffled_indices[self.indices] 

    def __getitem__(self, idx):
        """if self.with_replacement:"""

        indices = self.indices[idx]
        X = []
        y_single = torch.zeros(
            (self.group_size,),
            dtype=torch.long,
        )
        for i in range(self.group_size):
            x, cur_y = self.dataset.__getitem__(indices[i])
            X.append(x)
            if cur_y == 1:      
                y_single[i] = 1

        return (torch.stack(X, dim=0), self.csi[idx])

    def __len__(self):

        return self.num_samples  


def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--group_size", type=int, default=16)
    parse.add_argument("--train_val_split", type=tuple, default = (45000,5000))
    parse.add_argument("--num_workers", type = int, default = 0)   
    parse.add_argument("--min_snr", type=float, default=0.0)
    parse.add_argument("--max_snr", type=float, default=20.0)
    parse.add_argument("--num_train_samples", type=int, default=200000) 
    parse.add_argument("--num_val_samples", type=int, default=0)
    parse.add_argument("--num_test_samples", type=int, default=0)
    parse.add_argument("--batch_size", type=int, default=32)        
    #parse.add_argument("--counter", type=int, default=1)
    parse.add_argument("--regroup_every_training_epoch", type=int, default=0)
    args = parse.parse_args(args=[])
    return args

#SNR generation
def generate_noise_vec(data_length, args, min_snr=None, max_snr=None, seed=1):
    min_snr = min_snr if min_snr is not None else args.min_snr  
    max_snr = max_snr if max_snr is not None else args.max_snr
    return torch.empty(data_length, 1).uniform_(
            min_snr,
            max_snr,
            generator=torch.Generator().manual_seed(seed),
        )

'''
def getDataset(args, data_train_single ,data_val, test_data, csi_train, csi_val, csi_test):
    
    train_dataset = GroupingWrapper(dataset=data_train_single, csi=csi_train, group_size=args.group_size,num_samples=args.num_samples, seed=1)
    valid_dataset = GroupingWrapper(dataset=data_val, csi=csi_val, group_size=args.group_size, num_samples=args.num_samples, seed=1)
    test_dataset = GroupingWrapper(dataset=test_data, csi=csi_test, group_size=args.group_size, num_samples=args.num_samples, seed=1)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
'''


def train_dataloader(args, data_train_single, csi_train_0):
    counter = 1
    if args.regroup_every_training_epoch:
        csi_train = generate_noise_vec(len(data_train_single), args, seed=counter)
        train_dataset = GroupingWrapper(
            dataset=data_train_single,
            csi=csi_train,
            group_size=args.group_size,
            num_samples= args.num_train_samples,
            seed=counter,
        )
        counter +=1
    else: 
        train_dataset = GroupingWrapper(
            dataset=data_train_single,
            csi=csi_train_0,
            group_size=args.group_size,
            num_samples= args.num_train_samples,
            seed=1,
        )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    return train_loader


def val_dataloader(args, data_val, csi_val_0):
    valid_dataset = GroupingWrapper(
        dataset=data_val, 
        csi=csi_val_0, 
        group_size=args.group_size, 
        num_samples=args.num_val_samples, 
        seed=1
        )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    return valid_loader

# [0,20] dataset list
def test_dataloader(args, test_data):
    test_dataset = [
        GroupingWrapper(
            test_data,
            generate_noise_vec(len(test_data), args, min_snr=snr, max_snr=snr, seed=i),
            args.group_size,
            args.num_test_samples,
            seed=1,
        )
        for i, snr in enumerate(
            torch.arange(args.min_snr, args.max_snr + 1.0, 1.0)
        )
    ]
    return [
        DataLoader(
            dataset= dt,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        for dt in test_dataset
    ]
args = getArgs()


data_train_single, data_val = random_split(
    dataset=train_data,
    lengths=args.train_val_split,
    generator=torch.Generator().manual_seed(1),
)
len_data_val = len(data_val)


# Data Augmentation
'''
da_train = utils.DataAugmentation(
                img_size=64,
                with_random_hflip=True,
                with_random_crop=True)
da_val = utils.DataAugmentation(
                img_size=64,
                with_random_hflip=True,
                with_random_crop=True)
data_train_single = da_train.transform(data_train_single)
data_val = da_val.transform(data_val)
'''

num_train_samples = args.num_train_samples     
csi_train = generate_noise_vec(num_train_samples, args)
csi_val = generate_noise_vec(len_data_val, args)
#csi_test = generate_noise_vec(len(test_data), args)
train_loader = train_dataloader(args, data_train_single, csi_train)
valid_loader = val_dataloader(args, data_val, csi_val)
test_loader = test_dataloader(args, test_data)

dataloaders = {'train': train_loader, 'val': valid_loader}