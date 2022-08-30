from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        #找更小
        if im_h < min_size:
            #小的比min还小
            #ratio > 1
            #相当于等比放大
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            #小的比max还大
            #ratio < 1
            #等比缩小
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    #点距离(0,0)的距离 x^2+y^2作和
    square = np.sum(point*points, axis=1)
    #square[:,None]为维度扩展 例如[1,2,3]->[[1],[2],[3]]维度从(3)到(3,1)
    #square[:,None] + square[None,:]将会生成一个n维方阵
    #[[a],[b],[c]] + [[a,b,c]]
    #->
    #    [[a+a,a+b,a+c],   
    #    [b+a,b+b,b+c],
    #    [c+a,c+b,c+c]]
    #维度为(3,3)

    #point为一个n*2阶矩阵(n为点标记数量)
    #point.T为2*n阶矩阵
    #point @ point.T 为n为方阵

    #square为n*1阶行列式其中每个值为points中x^2+y^2
    #square为1*n阶行列式即转置

    #设点标记为(x1,y1)->(xn,yn)
    #square[:, None]+square[None, :]将得到如下方阵
    #Mij = xi^xi+yi^yi+xj^xj+yj^yj
    #2*np.matmul(point, point.T)将得到如下方阵
    #Nij = 2*(xi*xj+yi*yj)
    #可以发现 Mij - Nij = xi^2+yi^2+xj^2+yj^2-2(xi*xj+yi*yj)
    #即所求方阵可表示为 Kij = (xi-xj)^2 + (yi-yj)^2
    ##即两两标记点的相对距离，为对称矩阵
    #np.maximum将小于0的替换为0，此处作用
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    #dis是一个n维矩阵，其值为两两点之间的欧氏距离
    #第一行从1-n为点1距离点1到点n的欧氏距离，以此类推
    #np.partition矩阵快排
    #第二个参数指定第几列在其正确的位置上，该元素之前为所有小于该元素的元素，且不改变其原有位置(不一定按大小)，其后以此类推
    #第三个参数指定在哪个维度上进行排序
    #np.mean求均值axis=1指定求行均值
    #dis求完结果为距离点n第2，3，4近的点的距离的平均值
    #为什么不用1，因为矩阵中包含点到自身的距离，总为0
    #每个点最近的点的距离
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size    #获取图像长宽
    mat_path = im_path.replace('.jpg', '_ann.mat')  #获取对应点标注文件名称
    #points中annPoints记录的是点标记 对应xy根据大小可以判断
    points = loadmat(mat_path)['annPoints'].astype(np.float32)
    #idx_mask返回一个True False数组 用于确认点标记的正确性
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    #将不符合图像长宽的点标记排除
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        #ratio不为1说明图像大小需要调整
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='/home/teddy/UCF-QNRF_ECCV18',
                        help='original data directory')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048

    for phase in ['Train', 'Test']:
        sub_dir = os.path.join(args.origin_dir, phase)
        if phase == 'Train':    #Train集数据
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(save_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                with open('{}.txt'.format(sub_phase)) as f:
                    for i in f:
                        im_path = os.path.join(sub_dir, i.strip())
                        name = os.path.basename(im_path)    #返回文件名称
                        print(name)
                        im, points = generate_data(im_path) #重点在数据生成部分，此部分将图片大小进行调整
                        if sub_phase == 'train':
                            #此处处理完之后得到n*3阶矩阵前两阶为点x,y第三阶为该点离最近的3点的距离的平均值
                            dis = find_dis(points)
                            points = np.concatenate((points, dis), axis=1)
                        im_save_path = os.path.join(sub_save_dir, name)
                        im.save(im_save_path)
                        gd_save_path = im_save_path.replace('jpg', 'npy')
                        np.save(gd_save_path, points)
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for im_path in im_list:
                name = os.path.basename(im_path)
                print(name)
                im, points = generate_data(im_path)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)
