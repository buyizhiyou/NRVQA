# train
Download dataset LIVE and put into `LIVE` folder,run command below to train your model.  
```
python DIQA.py 
```
# test 
In the `models` folder, there is a model I trained and you can run command below to test your images or videos.
```
python predict_img.py --imgpath=../imgs/origin.jpg
python predict_video.py --videopath=./your/own/videopath
```
# reference  
[reference](https://towardsdatascience.com/deep-image-quality-assessment-with-tensorflow-2-0-69ed8c32f195)
