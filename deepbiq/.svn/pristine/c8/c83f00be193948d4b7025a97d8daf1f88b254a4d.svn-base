import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.optim
import torch.utils.data
from PIL import Image
import scipy.io as sio
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from comm_model import *


svr_save_path = './svr_mode.pkl'
svr_process_path = './svr_process.pkl'
feature_mode_path = '../trained_models/model_best.pth.tar'
image_dir = './ChallengeDB_release/Images'
matfn = './ChallengeDB_release/Data/AllMOS_release.mat'
mat_img_name = './ChallengeDB_release/Data/AllImages_release.mat'


def cal_lcc(x, y):
    n = x.shape[0]
    s1 = n * ((x * y).sum())
    s2 = x.sum() * y.sum()
    s3 = np.sqrt(n * ((x * x).sum()) - np.square(x.sum()))
    s4 = np.sqrt(n * ((y * y).sum()) - np.square(y.sum()))
    lcc = (s1 - s2) / (s3 * s4)
    return lcc


def img_dataset(feature_model):
    mos_data = sio.loadmat(matfn)
    mos = mos_data['AllMOS_release']
    img_name_data = sio.loadmat(mat_img_name)
    img_name = img_name_data['AllImages_release']

    normalize = get_imagenet_normalize()
    img_transform = transforms.Compose([transforms.ToTensor(), normalize])
    #img_num = 1169
    img_num = mos.shape[1]
    idx_arry = np.arange(0, img_num)
    np.random.shuffle(idx_arry)

    X = np.array([])
    Y = np.array([])

    crop_w = 224
    crop_h = 224
    img_w = 0
    img_h = 0
    crop_num_w = 5
    crop_num_h = 5

    for i, idx in enumerate(idx_arry):
        img_file_path = os.path.join(image_dir, img_name[idx][0][0])
        img_mos_score = mos[0, idx]
        print(i, ' process: ', img_file_path)
        crop_imgs = np.array([])
        crop_out = None
        img = Image.open(img_file_path)
        img_w, img_h = img.size
        crop_box = get_crop_box(img_w, img_h, crop_w, crop_h, crop_num_w, crop_num_h)
        for box in crop_box:
            crop_imgs = np.append(crop_imgs, img_transform(img.crop(box)).numpy())
        crop_imgs = crop_imgs.reshape(crop_num_w * crop_num_h, 3, 224, 224)
        crop_imgs = torch.from_numpy(crop_imgs).float()
        crop_out = feature_model.extract_feature(crop_imgs)
        crop_out = np.average(crop_out, axis=0)

        X = np.append(X, crop_out)
        Y = np.append(Y, img_mos_score)
    X = X.reshape(-1, 4096)

    print(X.shape)
    print(Y.shape)

    return X, Y


def main():
    feature_model = FeatureMode(feature_mode_path)
    data_x, data_y = img_dataset(feature_model)

    scaler_x = preprocessing.StandardScaler().fit(data_x)
    joblib.dump(scaler_x, svr_process_path)
    train_x = scaler_x.transform(data_x)

    X_train, X_test, y_train, y_test = train_test_split(train_x, data_y, test_size=0.2, random_state=0)
    print('------------')
    print('training svr model ......')

    parameters = {"C": [1e1, 1e2, 1e3], "gamma": [0.00025, 0.00020, 0.00015, 0.00010],
                  "epsilon": [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]}
    clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1, epsilon=0.01), cv=5, param_grid=parameters, n_jobs=10)
    clf.fit(X_train, y_train)

    #best score
    print("Best score: %0.3f" % clf.best_score_)
    print("Best parameters set:")
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    clf = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], epsilon=best_parameters['epsilon'])
    clf.fit(X_train, y_train)
    joblib.dump(clf, svr_save_path)
    pred_y_test = clf.predict(X_test)
    print('lcc:', cal_lcc(pred_y_test, y_test))


if __name__ == '__main__':
    main()




