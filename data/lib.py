
import matplotlib
matplotlib.use('Agg')
from data.cifar import CIFAR10, CIFAR100
from data.kvasir import KVASIR
import torchvision.transforms as transforms

def choose_dataset(dataset,seed,noise_type,noise_rate):
   
   if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.4566, 0.4066],
                                 std=[0.229, 0.224, 0.225])
        num_classes = 10
        all_train_data1 = CIFAR10(
            seed=seed,
            root='../data/',
            train=True,
            valid=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True,
            noise_type=noise_type,
            noise_rate=noise_rate)
        
    
        all_train_data2 = CIFAR10(
            seed=seed,
            root='../data/',
            train=True,
            valid=True,
            transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True)
    
        test_data = CIFAR10(
            seed=seed,
            root='../data/',
            train=False,
            valid=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    

   elif dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.2667, 0.2566, 0.2766])
        num_classes = 100
        all_train_data1 = CIFAR100(
            seed=seed,
            root='../data/',
            train=True,
            valid=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True,
            noise_type=noise_type,
            noise_rate=noise_rate)
    
        all_train_data2 = CIFAR100(
            seed=seed,
            root='../data/',
            train=True,
            valid=True,
            transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True)
    
        test_data = CIFAR100(
            seed=seed,
            root='../data/',
            train=False,
            valid=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
   if dataset == 'kvasir':
        print('start load')
        num_classes = 6
        all_train_data1 = KVASIR(
            seed=seed,
            image_path='../data/',
            label_path='../data/',
            transform=transforms.Compose([
            
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor()
            
                ]),
                noise_type=noise_type,
                noise_rate=noise_rate)
    
    
        all_train_data2 = KVASIR(
            seed=seed,
            image_path='../data/',
            label_path='../data/',
            transform=transforms.Compose([
           
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
                transforms.ToTensor()
            
            ]),
            )
    
        test_data = KVASIR(
            seed=seed,
            image_path='../data/',
            label_path='../data/',
            transform=transforms.Compose([
           
                transforms.ToTensor()
            
            ]))
        
   return all_train_data1, all_train_data2,test_data,num_classes