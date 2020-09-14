# 无参考图片/视频质量评价(No-Reference Blind Video Quality Assessment)

## 深度学习

- [deepbiq](https://github.com/zhl2007/pytorch-image-quality-param-ctrl)
- [DIQA](https://towardsdatascience.com/deep-image-quality-assessment-with-tensorflow-2-0-69ed8c32f195)
- [VSFA](https://github.com/lidq92/VSFA)

## 传统方法

- BRISQUE(**extract 36 dimesion brisque features,you can train svr model in labeled datasets like TID2013/LIVE/CSIQ**)
- NIQE
- PIQE


## test

### brisque
*high score has high quality*
```
python test.py --mode brisque --path=imgs/origin.jpeg
python test.py --mode brisque --path=imgs/compression.jpeg
```
### niqe
*high score has low quality*
```
python test.py --mode niqe --path=imgs/origin.jpeg
python test.py --mode niqe --path=imgs/compression.jpeg
```
### piqe
*high score has low quality*
```
python test.py --mode piqe --path=imgs/origin.jpeg
python test.py --mode piqe --path=imgs/compression.jpeg
```
  




## 相关论文[reference papers](https://github.com/buyizhiyou/papers/tree/master/VQA_IQA)

## Thanks for your star!
