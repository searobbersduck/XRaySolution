# 

## kaggle比赛
[Kaggle SIIM-ACR Pneumothorax Segmentation Challenge](https://kaggle.com/c/siim-acr-pneumothorax-segmentation)

## 数据集介绍
1. 训练集数目为10675张，第一个阶段测试集数据有1372张
2. 数据集以dicom的格式给出，该格式包含了病人的姓名、id、年龄、性别、Modality、BodyPartExamined、ViewPosition以及X光数据信息
3. 掩模数据以CSV格式的文件给出，每行数据以ImageId,EncodedPixels给出，我们需要使用代码将EncodedPixels转化为图片形式，其中像素点为0的地方表示没有气胸，像素点为255的地方表示有气胸，若一张图片像素点全部为0，则表示该图片没有气胸。
4. 值得注意的是，并不是所有的图片都对应一个掩模，因为有些病人是没有气胸的，所以相应的也就没有掩模了。因此，这次比赛要判断是否含有气胸，如果含有的话分割出气胸。

## 数据集路径
`ln -s /data/zhangwd/data/external/xray/siim/siim ./data`
```
tree -L 1

.
├── dicom-images-test
├── dicom-images-train
└── train-rle.csv

```

## kernel


## 气胸相关知识
1. [手把手教你CT看气胸，一目了然！](https://www.dxy.cn/bbs/newweb/pc/post/41048098?source=rss)
2. [kaggle气胸疾病图像分割top5解决方案](http://www.360doc.com/content/19/1020/07/57922944_867913189.shtml)
3. [Kaggle SIIM-ACR Pneumothorax Segmentation Challenge 总结](https://aiotwe.com/MachineLearning/%E5%AE%9E%E6%88%98/Kaggle%20SIIM-ACR%20Pneumothorax%20Segmentation%20Challenge%20%E6%80%BB%E7%BB%93/)
    * 这篇blog讲解的比较详细，虽然效果不一定最佳，但是可以帮助我们更好的了解气胸

## reference
1. [sneddy/pneumothorax-segmentation](https://github.com/sneddy/pneumothorax-segmentation)


