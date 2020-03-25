#!/bin/sh

rm -rf dataset
mkdir -p dataset/train
mkdir -p dataset/val

#fine-tuned alexnet
rm checkpoint.pth.tar -rf
rm model_best.pth.tar -rf
rm ../trained_models/checkpoint.pth.tar -rf
rm ../trained_models/model_best.pth.tar -rf

python main.py --lr 0.001 --weight-decay 1e-5 -b 64 --epochs 50 --print-freq 1 --pretrained dataset
python main.py --lr 0.001 --weight-decay 1e-5 -b 64 --epochs 100 --print-freq 1 --pretrained --resume ./checkpoint.pth.tar dataset
python main.py --lr 0.001 --weight-decay 1e-5 -b 64 --epochs 150 --print-freq 1 --pretrained --resume ./checkpoint.pth.tar dataset
python main.py --lr 0.001 --weight-decay 1e-5 -b 64 --epochs 200 --print-freq 1 --pretrained --resume ./checkpoint.pth.tar dataset
python main.py --lr 0.001 --weight-decay 1e-5 -b 64 --epochs 250 --print-freq 1 --pretrained --resume ./checkpoint.pth.tar dataset
python main.py --lr 0.001 --weight-decay 1e-5 -b 64 --epochs 300 --print-freq 1 --pretrained --resume ./checkpoint.pth.tar dataset

mv checkpoint.pth.tar ../trained_models/
mv model_best.pth.tar ../trained_models/


#train svr model
rm svr_mode.pkl -rf
rm svr_process.pkl -rf
rm ../trained_models/svr_mode.pkl
rm ../trained_models/svr_process.pkl

python svr_train.py

mv svr_mode.pkl ../trained_models/
mv svr_process.pkl ../trained_models/
