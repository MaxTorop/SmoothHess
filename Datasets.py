from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision 
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


def CIFAR10(**kwargs):
    batch_size    = kwargs['batch_size']
    valstart      = kwargs['valstart']

    transform_test  = plain_transform()
    transform_val   = plain_transform()  
    transform_train = cifar10_augs() 

    train_data = torchvision.datasets.CIFAR10('./', train=True, transform = transform_train, download=True)
    val_data = torchvision.datasets.CIFAR10('./', train=True, transform = transform_val, download=True)
    test_data = torchvision.datasets.CIFAR10('./', train=False, transform = transform_test, download=True) 
    train_no_shuffle_no_aug_data = torchvision.datasets.CIFAR10('./', train=True, transform = transform_test, download=True) 

    num_train = len(train_data)
    indices = list(range(num_train))
    train_idx, valid_idx     = indices[:valstart], indices[valstart:]
    train_sampler            = SubsetRandomSampler(train_idx)
    valid_sampler            = SequentialSampler(valid_idx)
    train_sampler_no_shuffle = SequentialSampler(train_idx) 

    train_loader                   = DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, num_workers=1, pin_memory=True, drop_last=False)
    val_loader                     = DataLoader(val_data, batch_size=batch_size, sampler = valid_sampler, num_workers=1, pin_memory=True, drop_last = False) 
    test_loader                    = DataLoader(test_data, batch_size=batch_size, shuffle = False, num_workers=1, pin_memory=True, drop_last = False) 
    train_no_shuffle_no_aug_loader = DataLoader(train_no_shuffle_no_aug_data, batch_size=batch_size, sampler = train_sampler_no_shuffle, num_workers=1, pin_memory=True, drop_last = False) 
    return train_loader, train_data, val_loader, val_data, test_loader, test_data, train_no_shuffle_no_aug_data, train_no_shuffle_no_aug_loader


def FMNIST(**kwargs):
    batch_size    = kwargs['batch_size']
    valstart      = kwargs['valstart']

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_data = torchvision.datasets.FashionMNIST('./', train=True, transform = transform, download=True)
    val_data = torchvision.datasets.FashionMNIST('./', train=True, transform = transform, download=True)
    test_data = torchvision.datasets.FashionMNIST('./', train=False, transform = transform, download=True) 
    train_no_shuffle_no_aug_data = torchvision.datasets.FashionMNIST('./', train=True, transform = transform, download=True) 

    num_train = len(train_data)
    indices = list(range(num_train))
    train_idx, valid_idx     = indices[:valstart], indices[valstart:]
    train_sampler            = SubsetRandomSampler(train_idx)
    valid_sampler            = SequentialSampler(valid_idx)
    train_sampler_no_shuffle = SequentialSampler(train_idx) 

    train_loader                   = DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, pin_memory=True, drop_last=False)
    val_loader                     = DataLoader(val_data, batch_size=batch_size, sampler = valid_sampler, pin_memory=True, drop_last = False) 
    test_loader                    = DataLoader(test_data, batch_size=batch_size, shuffle = False, pin_memory=True, drop_last = False) 
    train_no_shuffle_no_aug_loader = DataLoader(train_no_shuffle_no_aug_data, batch_size=batch_size, sampler = train_sampler_no_shuffle, pin_memory=True, drop_last = False) 
    return train_loader, train_data, val_loader, val_data, test_loader, test_data, train_no_shuffle_no_aug_data, train_no_shuffle_no_aug_loader


def MNIST(**kwargs):
    batch_size    = kwargs['batch_size']
    valstart      = kwargs['valstart']

    transform=transforms.Compose([
    transforms.ToTensor(),
    ])
    
    train_data = torchvision.datasets.MNIST('./', train=True, transform = transform, download=True)
    val_data = torchvision.datasets.MNIST('./', train=True, transform = transform, download=True)
    test_data = torchvision.datasets.MNIST('./', train=False, transform = transform, download=True) 
    train_no_shuffle_no_aug_data = torchvision.datasets.MNIST('./', train=True, transform = transform, download=True) 

    num_train = len(train_data)
    indices = list(range(num_train))
    train_idx, valid_idx     = indices[:valstart], indices[valstart:]
    train_sampler            = SubsetRandomSampler(train_idx)
    valid_sampler            = SequentialSampler(valid_idx)
    train_sampler_no_shuffle = SequentialSampler(train_idx) 

    train_loader                   = DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, num_workers=1, pin_memory=True, drop_last=False)
    val_loader                     = DataLoader(val_data, batch_size=batch_size, sampler = valid_sampler, num_workers=1, pin_memory=True, drop_last = False) 
    test_loader                    = DataLoader(test_data, batch_size=batch_size, shuffle = False, num_workers=1, pin_memory=True, drop_last = False) 
    train_no_shuffle_no_aug_loader = DataLoader(train_no_shuffle_no_aug_data, batch_size=batch_size, sampler = train_sampler_no_shuffle, num_workers=1, pin_memory=True, drop_last = False) 
    return train_loader, train_data, val_loader, val_data, test_loader, test_data, train_no_shuffle_no_aug_data, train_no_shuffle_no_aug_loader



########## Transforms ##########


def plain_transform():
    transform = transforms.Compose([transforms.ToTensor()])
    return transform 


def cifar10_augs():
    transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),]
    )
    return transform 