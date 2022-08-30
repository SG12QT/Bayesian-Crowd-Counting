from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np


#res_h res_w是原图像大出切片大小长度
#在0-res_h,0-res_w随机选一个点作为新图左上角顶点即(i,j)
def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area



class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        #sorted 不改变原有元素
        #glob 匹配原有路径中所有以.jpg结尾的文件
        #im_list 返回按照文件名排好顺序的一个列表
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        #crop_size 图像切片大小 default 512
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            #对训练图像进行切割，网络接收的图片要求是大小相同的
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        #随机切片函数
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        #img为裁剪后的图片
        #test 之后发现 裁剪出来的图片很有可能并不包含任何人
        img = F.crop(img, i, j, h, w)
        #在Keypoints第三列 即最近点距离进行截取 截取4-128数据
        #nearest_dis 为一个列表 只包含距离数据
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        #keypoints[:, :2] 取前两列 即点的x,y
        #nearest_dis[:, None] 扩维度
        #为什么能保证 Keypoints 和 截取后的 nearest_dis维度一定相同?? 已解决
        #np.clip不改变长度 只是将数列数据规范到min-max范围
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        # bbox 维度为 (n,4) axis=1在行维度拼接
        # bbox 得到一个 框
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        #inner_area是求截取的图像中 包含的点
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        #哈达姆积
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        #采样率高于0.3
        mask = (ratio >= 0.3)

        #target 指所求点在图像中的概率?
        target = ratio[mask]
        keypoints = keypoints[mask]
        #改变坐标 图象被截取
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                #hflip水平翻转图像
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size
