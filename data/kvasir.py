import os
import numpy as np
from PIL import Image
import random
from PIL import Image
from .utils import download_url, check_integrity, noisify
from torch.utils.data import Dataset
def pre_dataset(path):
# 配置参数
   
    output_dir = '../data/kvasir/'          # 输出目录
    image_size = (128, 128)                # 图像尺寸
    seed = 42                              # 随机种子
    class_names = sorted(os.listdir(path))  # 获取类别名称
    print(class_names)
    
# 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

# 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

# 初始化存储容器
    train_data = {'images': [], 'labels': []}
    val_data = {'images': [], 'labels': []}
    test_data = {'images': [], 'labels': []}

# 遍历每个类别
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(path, class_name)
        print(class_path)
        all_images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpg')]
        
        print(f"label:{label_idx}")
        
    # 随机打乱并确保可复现
        random.shuffle(all_images)
    
    # 计算划分点
        n_total = len(all_images)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
    
    # 划分数据集
        train_files = all_images[:n_train]
        val_files = all_images[n_train:n_train+n_val]
        test_files = all_images[n_train+n_val:]
    
    # 处理训练集
        for path_train in train_files:
            img = Image.open(path_train).convert('RGB').resize(image_size)
            train_data['images'].append(np.array(img))
            train_data['labels'].append(label_idx)
            print(f"train_sum:{len(train_data['images'])}")
    # 处理验证集
        for path_val in val_files:
            img = Image.open(path_val).convert('RGB').resize(image_size)
            val_data['images'].append(np.array(img))
            val_data['labels'].append(label_idx)
            print(f"val_sum:{len(val_data['images'])}")
    # 处理测试集
        for path_test in test_files:
            img = Image.open(path_test).convert('RGB').resize(image_size)
            test_data['images'].append(np.array(img))
            test_data['labels'].append(label_idx)
            print(f"test_sum:{len(test_data['images'])}")
# 转换为numpy数组并打乱顺序
    def shuffle_and_save(data, split_name):
        images = np.array(data['images'])
        labels = np.array(data['labels'])
    
    # 打乱顺序
        indices = np.random.permutation(len(images))
        print(f"indice:{indices}")
        images = images[indices]
        labels = labels[indices]
    
    # 保存文件
        np.save(os.path.join(output_dir, split_name, 'images_128.npy'), images)
        np.save(os.path.join(output_dir, split_name, 'labels_128.npy'), labels)

# 处理所有数据集
    shuffle_and_save(train_data, 'train')
    shuffle_and_save(val_data, 'valid')
    shuffle_and_save(test_data, 'test')

    print("finish!")
    print(f"train set: {len(train_data['images'])} ")
    print(f"valid set: {len(val_data['images'])} ")
    print(f"test set: {len(test_data['images'])} ")
class KVASIR(Dataset):
    def __init__(self,seed,image_path,label_path,transform=None, noise_type=None, noise_rate=None, random_state=0):
        self.seed=seed
        self.noise_type=noise_type
        self.nb_classes=6
        self.noise_rate=noise_rate
        self.dataset='kvasir'
        self.images = np.load(image_path)
        self.labels= np.load(label_path)
        self.transform = transform
        if noise_type is not None and noise_rate !=0:
            self.labels = self.labels.reshape(-1, 1)
            self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
            
            self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
            _train_labels=[i[0] for i in self.labels]
            self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)#设置噪声flag
            print('load data finish!')
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.noise_type is not None and self.noise_rate!=0:
            image,target=self.images[index],self.train_noisy_labels[index]
            noise_flag=self.noise_or_not[index]
        else:
            image,target = self.images[index],self.labels[index]
            noise_flag=0
        img = Image.fromarray(image)
        if self.transform is not None:
            img = self.transform(img)

        return img,target,noise_flag,index
if __name__ == '__main__':
    pre_dataset(path=r"../kvasir/kvasir-dataset-v2/")