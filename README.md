# SAM_Paddle
[Only a Matter of Style: Age Transformation Using a Style-Based Regression Model](https://paperswithcode.com/paper/only-a-matter-of-style-age-transformation) 论文复现

官方源码：[https://github.com/yuval-alaluf/SAM](https://github.com/yuval-alaluf/SAM)

复现地址：[https://github.com/771979972/paddle-SAM](https://github.com/771979972/paddle-SAM)

# Dataset
训练集下载：[FFHQ](https://github.com/NVlabs/ffhq-dataset)。
测试集下载：[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。

数据集下载后分别将图片数据保存在```FFHQ\``` 和```CelebA_test\```。

在```CelebA_test\```文件夹下已提供5张测试图片。

# Training
首次训练，编译ranger算子.
```
python scripts/compile_ranger.py
```
然后再训练
```
python scripts/train.py \
--dataset_type=ffhq_aging \
--exp_dir=exp/test \
--workers=0 \
--batch_size=6 \
--test_batch_size=6 \
--test_workers=0 \
--val_interval=2500 \
--save_interval=5000  \
--start_from_encoded_w_plus \
--id_lambda=0.1 \
--lpips_lambda=0.1 \
--lpips_lambda_aging=0.1 \
--lpips_lambda_crop=0.6 \
--l2_lambda=0.25 \
--l2_lambda_aging=0.25 \
--l2_lambda_crop=1 \
--w_norm_lambda=0.005 \
--aging_lambda=5 \
--cycle_lambda=1 \
--input_nc=4 \
--target_age=uniform_random \
--use_weighted_id_loss
```
训练所需的预训练模型：
下载后将模型的参数保存在```pretrained_models\```中
| 模型(文件名) | Description
| :--- | :----------
|pSp Encoder(psp_ffhq_encode) | pSp 来自 [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel).
|FFHQ StyleGAN(stylegan2-ffhq-config-f) | StyleGAN 在FFHQ上训练，来自 [rosinality](https://github.com/rosinality/stylegan2-pytorch) ，输出1024x1024大小的图片
|IR-SE50 Model(model_ir_se50) | IR SE 模型，来自 [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 用于训练中计算ID loss。
|CurricularFace Backbone(CurricularFace_Backbone)  | 预训练的 CurricularFace model，来自 [HuangYG123](https://github.com/HuangYG123/CurricularFace) 用于Similarity的评估。
|VGG Age Classifier(dex_agg_classifier) | VGG age classifier，来自DEX，在FFHQ-Aging数据集上微调。
|AlexNet(alexnet和lin_alex)  | 用于lpips loss计算。

链接（百度网盘）：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg )
提取码：m3nb

link (Google Drive)：https://drive.google.com/drive/folders/1Bs4MQQ9Pw5E2W1-Y9JA2AkGSWwtDzzis?usp=sharing

# Test
### Inference
```
python scripts/inference_side_by_side.py 
--exp_dir=test_inference 
--checkpoint_path=pretrained_models/sam_ffhq_aging.pdparams 
--data_path=images 
--test_batch_size=4 
--test_workers=0 
--target_age=0,10,20,30,40,50,60,70,80,90,100
```
SAM模型下载地址：(文件名：sam_ffhq_aging.pdparams)

链接：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg )
提取码：m3nb

下载后将模型的参数保存在```pretrained_models\```中

# Log
训练日志为 log/vdlrecords.1629623399.log，训练过程中的测试结果在 log/timestamp.txt中

在当前目录执行

```
from visualdl.server import app
app.run(logdir="log")
```

# 结果
目前呈现的结果为运行24000步保存的模型的结果，据作者称论文的结果为运行了60000步.

图片从左到右分别是：输入图片，模型生成的0岁，10岁，20岁，30岁，40岁，50岁，60岁，70岁，80岁，90岁，100岁图片

### Pytorch与Paddle效果对比
| 模型 | 图片 |
| ------ | ------ |
| Pytorch | <img src="examples/866_torch.jpg" height = "104" width="1248"/><br/> |
| Paddle | <img src="examples/866_paddle.jpg" height = "104" width="1248"/><br/> |
| Pytorch | <img src="examples/1287_torch.jpg" height = "104" width="1248"/><br/> |
| Paddle | <img src="examples/1287_paddle.jpg" height = "104" width="1248"/><br/> |
| Pytorch | <img src="examples/2468_torch.jpg" height = "104" width="1248"/><br/> |
| Paddle | <img src="examples/2468_paddle.jpg" height = "104" width="1248"/><br/> |

### 以下是使用Paddle复现的其他结果
<img src="examples/000431_paddle.jpg" height = "104" width="1248"/><br/>
<img src="examples/001092.jpg" height = "104" width="1248"/><br/>
<img src="examples/198650.jpg" height = "104" width="1248"/><br/>

更多结果见example文件夹中.

