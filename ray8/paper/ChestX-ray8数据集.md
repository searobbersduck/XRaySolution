# ChestX-ray8数据集

## 数据集描述
|数据集|数量|体位|病人数|收集年份|收集医院|tag数|tag获取方式
|-|-|-|-|-|-|-|-|
|ChestX-ray8|108,948|front-view|32717|1992-1995||8个常见标签|通过nlp技术从报告获取

信息来源：[ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases]([https://arxiv.org/pdf/1705.02315.pdf](https://arxiv.org/pdf/1705.02315.pdf))
> To tackle these issues, we propose a new chest X-ray database, namely “ChestX-ray8”, which comprises 108,948 frontal-view X-ray images of 32,717 (collected from the year of 1992 to 2015) unique patients with the text-mined eight common disease labels, mined from the text radiological reports via NLP techniques. In particular, we demonstrate that these commonly occurred thoracic diseases can be detected and even spatially-located via a unified weakly-supervised multi-label image classification and disease localization formulation. Our initial quantitative results are promising. However developing fully-automated deep learning based “reading chest X-rays” systems is still an arduous journey to be exploited. Details of accessing the ChestX-ray8 dataset can be found via the website

## 标签对应
|labels|Atelectasis|Cardiomegaly|Effusion|Infiltration|Mass|Nodule|Pneumonia|Pneumathorax|
|-|-|-|-|-|-|-|-|-|
|标签|肺不张|心包肿大|积液|浸润|肿块|结节|肺炎|气胸|


## 数据筛选方式
### 数据筛选方式：
* 不确定的不处理（尽量避免假阳性标签）
* 明确不包含任何疾病的，标记为正常
* 通过DNorm和MetaMap，利用专业词汇和专业词汇对应的语义特征进行抽取
* 文本处理方面有很多trick，这个需要细看

>A variety of Natural Language Processing (NLP) techniques are adopted for detecting the pathology keywords and removal of negation and uncertainty. Each radiological report will be either linked with one or more keywords or marked with ’Normal’ as the background category. As a result, the ChestX-ray8 database is composed of 108,948 frontal-view X-ray images (from 32,717 patients) and each image is labeled with one or multiple pathology keywords or “Normal” otherwise.

>The main body of each chest X-ray report is generally structured as “Comparison”, “Indication”, “Findings”, and “Impression” sections. Here, we focus on detecting disease concepts in the Findings and Impression sections. If a report contains neither of these two sections, the full-length report will then be considered. In the second pass, we code the reports as “Normal” if they do not contain any diseases (not limited to 8 predefined pathologies).

### 不同疾病之间是有关联性的
>It reveals some connections between different pathologies, which agree with radiologists’ domain knowledge, e.g., Infiltration is often associated with Atelectasis and Effusion. To some extend, this is similar with understanding the interactions and relationships among objects or concepts in natural images

### 对提出的报告处理方式的验证
将该方法再openi数据集上进行验证，该数据集有报告和医生确认过的标签

### bounding box 标注

1. 8种疾病，每种标注200例，每种疾病标注1个bounding box，但一张图像可能包括多种疾病
2. 不同疾病分开标注，单独存文件
3. 共983张图像，1600个病例

>In our labeling process, we first select 200 instances for each pathology (1,600 instances total), consisting of 983 images. Given an image and a disease keyword, a boardcertified radiologist identified only the corresponding disease instance in the image and labeled it with a B-Box. The B-Box is then outputted as an XML file. If one image contains multiple disease instances, each disease instance is labeled separately and stored into individual XML files. As an application of the proposed ChestX-ray8 database and benchmarking, we will demonstrate the detection and localization of thoracic diseases in the following.

### 模型
1. 分类网络+多标签分类
2. 最后的pooling，用的是Log-Sum-Exp (LSE) pooling
3. 画出每一类最后一层feature map的热度图，（SxSXD）,D为类别数
4. 每层归一化到255后，阈值{60，180}，画bboxing


## 数据下载链接

1. [Data split files could be downloaded via](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## Reference
1. [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases]([https://arxiv.org/pdf/1705.02315.pdf#page=3&zoom=100,0,409](https://arxiv.org/pdf/1705.02315.pdf#page=3&zoom=100,0,409))

