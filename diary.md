## 2020.1.13
- [x] CT部位识别新加接口调试（通过）
- [x] 解决有些DR数据没有ww/wc的问题，采用bitstored标签做归一化
- [x] 训练dr部位识别模型，数据预处理时，加入随机jit，反色，随机crop，随机翻转

## 2020.1.14
- [x] 挂载新训练的DR部位识别模型，跑正位预测
  - [x] dr部位识别trt模型挂载
  - [x] 跟进c++的正位预测
- [x] 自测新训练的DR部位识别模型
  - [x]  自测效果（和上一版DR部位识别模型测试集和预处理一样，测试集未做过增强）：Accuracy:0.997, 具体见: [DR部位识别模型自测](http://git.do.proxima-ai.com/cn.aitrox.ai/xrayproduct/blob/master/dr_pos_recog/traintest_model-20200114.ipyn)
- [ ] 测试跑通ray8的开源代码
  - [x] 对比移植code, 能够训练和绘制heatmap图
  - [x] 同步修改readme.md
  - [ ] 跑通测试代码
    - [ ] 显存溢出
  - [x] 跑通生成cam的代码
  - [x] 跑通训练代码
- [x] 弄懂cam的原理，对于不同类别cam的效果
- [x] 跟进肺结核检出的上线，测试不同的结核类型
  - [x] 模型已上传svn
- [x] 跟进病灶检出和部位识别代码中，关于dcm没有窗宽窗位的处理
- [x] 跟进CT部位识别
  - [x] .5服务器上调试数仓拉代码的code
  - [x] 要数据列表
  - [x] 在.5服务器上调试部位识别的code
- [x] 预处理时加入随机jit，crop，反色，训练肺结核检出模型
  - [x] 训练模型
  - [x] 测试集上测试AUC:0.8799, 具体见[肺结核模型自测](http://git.do.proxima-ai.com/cn.aitrox.ai/xrayproduct/blob/master/drCls/train/predict_dr_tuberculosis.ipynb)


## 2020.1.15
- [x] 跟进病灶检出和部位识别代码中，关于dcm没有窗宽窗位的处理
  - [x] 关于窗宽窗位的代码有问题，需要继续跟进 
- [x] 跟进肺结核检出的上线，测试不同的结核类型
  - [x] 关于窗宽窗位的代码有问题，需要继续跟进
接上一日安排
- [ ] 测试跑通ray8的开源代码
  - [x] 对比移植code, 能够训练和绘制heatmap图
  - [x] 同步修改readme.md
  - [ ] 跑通测试代码
    - [ ] 显存溢出
  - [x] 跑通生成cam的代码
  - [x] 跑通训练代码

## 2020.1.16
- [x] 跑通pneumothorax分割的baseline
  - [x] 跑通训练代码，5-folder 
- [ ] 查找ray8数据集，bboxing相关的开源code
- [x] 跟进病灶检出和部位识别代码中，关于dcm没有窗宽窗位的处理
  - [x] 编译测试通过 
- [x] ct 16w数据下载
  - [x] 已启动脚本进行下载  

## 2020.1.17
- [ ] 胸片自动报告
  - [ ] 利用ray8做出的dsn121模型，再openi数据集上，保存最后一层feature_map
  - [ ] 提取正位片
  - [ ] 提取带报告的文本
  - [ ] 生成字典
  - [ ] 跑通生成代码
