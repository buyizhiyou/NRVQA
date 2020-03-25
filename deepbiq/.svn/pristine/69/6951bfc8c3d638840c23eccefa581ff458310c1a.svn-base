import os
import shutil
import scipy.io as sio
import random


class DataSetSplit(object):
    def __init__(self, dataset_dir, out_dir):
        self.dataset_dir = dataset_dir
        self.img_dir = os.path.join(self.dataset_dir, 'Images')
        self.out_dir = out_dir

        matfn = os.path.join(self.dataset_dir, 'Data', 'AllMOS_release.mat')
        data = sio.loadmat(matfn)
        self.img_mos = data['AllMOS_release']
        img_name_matfn = os.path.join(self.dataset_dir, 'Data', 'AllImages_release.mat')
        img_name_data = sio.loadmat(img_name_matfn)
        self.img_name = img_name_data['AllImages_release']

        src_dir = os.path.join(self.img_dir, 'trainingImages')
        dst_dir = self.img_dir
        for f in os.listdir(src_dir):
            src_file = os.path.join(src_dir, f)
            dst_file = os.path.join(dst_dir, f)
            if os.path.isfile(src_file):
                shutil.copyfile(src_file, dst_file)

    def clean_dir(self):
        class_dir = ['bad', 'poor', 'fair', 'good', 'excellent']
        train_dir = os.path.join(self.out_dir, 'train')
        val_dir = os.path.join(self.out_dir, 'val')
        for i in class_dir:
            train_child_dir = os.path.join(train_dir, i)
            val_child_dir = os.path.join(val_dir, i)
            if os.path.exists(train_child_dir):
                shutil.rmtree(train_child_dir)
            if os.path.exists(val_child_dir):
                shutil.rmtree(val_child_dir)
            os.mkdir(train_child_dir)
            os.mkdir(val_child_dir)

    def split_train_val(self, data_list, class_dir):
        max_train_idx = len(data_list) * 4 / 5
        dst_train_path = os.path.join(self.out_dir, 'train', class_dir)
        dst_val_path = os.path.join(self.out_dir, 'val', class_dir)
        for idx, file in enumerate(data_list):
            src_path = os.path.join(self.img_dir, self.img_name[data_list[idx]][0][0])
            dst_path = None
            if idx < max_train_idx:
                dst_path = os.path.join(dst_train_path, self.img_name[data_list[idx]][0][0])
            else:
                dst_path = os.path.join(dst_val_path, self.img_name[data_list[idx]][0][0])
            shutil.copyfile(src_path, dst_path)

    def random_split(self):
        idx_bad = []
        idx_poor = []
        idx_fair = []
        idx_good = []
        idx_excellent = []

        for idx, str_name in enumerate(self.img_name):
            mos_score = self.img_mos[0, idx]
            if mos_score <= 20:
                idx_bad.append(idx)
            elif mos_score <= 40:
                idx_poor.append(idx)
            elif mos_score <= 60:
                idx_fair.append(idx)
            elif mos_score <= 80:
                idx_good.append(idx)
            else:
                idx_excellent.append(idx)

        random.shuffle(idx_bad)
        random.shuffle(idx_poor)
        random.shuffle(idx_fair)
        random.shuffle(idx_good)
        random.shuffle(idx_excellent)

        self.clean_dir()
        self.split_train_val(idx_bad, 'bad')
        self.split_train_val(idx_poor, 'poor')
        self.split_train_val(idx_fair, 'fair')
        self.split_train_val(idx_good, 'good')
        self.split_train_val(idx_excellent, 'excellent')


if __name__ == '__main__':
    dataset_split = DataSetSplit('./ChallengeDB_release', './dataset')
    dataset_split.random_split()