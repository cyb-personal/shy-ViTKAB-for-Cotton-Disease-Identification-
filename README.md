# TSSC-for-Pea-Disease-Identification


<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
> 官方TensorFlow 和 Keras实现 | 论文处于投刊阶段，标题：《TSSC: A New Deep Learning Model for Accurate Pea Leaf Disease Identification》  


> 提出时空尺度通道网络（TSSC）模型，基于TensorFlow框架实现五类豌豆常见病害与健康状态的高精度识别，助力农业病害智能化诊断。  


## 1. 研究背景与模型定位  

豌豆作为重要豆科作物，其叶片病害（如白粉病、潜叶虫、根腐病病等）易导致产量下降，传统人工检测存在效率低、依赖经验的问题。  

本文提出**时空尺度通道网络（TSSC）**，通过创新的多维度特征融合与跨尺度建模机制，解决豌豆病害“多类别区分难、复杂背景干扰大”的问题。模型基于TensorFlow框架实现，包含4个核心卷积层与3种注意力机制，在五类豌豆病害数据集上实现优异的分类性能，为农业病害自动化诊断提供高效解决方案。  


## 2. TSSC核心创新点  

1. **三层注意力机制协同**：  
   - **互补注意力（ complemented Squeeze-and-Excitation）**：从原始特征中提取主要的显著局部特征，从抑制的剩余通道信息中提取次级显著特征； 
   - **K通道注意力**：考虑每个通道与左右邻域通道的关联性，动态优化特征表达；  
   - **分散注意力（Split Attention）**：将通道分组并独立学习注意力，增强局部特征交互能力。  

2. **高效特征提取结构**：  
   采用4个精心设计的卷积层（核大小4×4、2×2、5×5、3×3），配合池化操作逐步压缩空间维度，在保留病害细节特征的同时提升计算效率。  

3. **多类别适配优化**：  
   针对五类病害的细粒度差异（如白粉病的白色霉层与霜霉病的黄色斑点），设计类别感知损失函数，降低相似病害的混淆度。  


## 3. 实验数据集：五类豌豆病害数据集  
=======
> 官方TensorFlow实现 | 论文处于投刊阶段，标题：《TSSC: A New Deep Learning Model for Accurate Pea Leaf Disease Identification》  
=======
> 官方TensorFlow实现 | 论文处于投刊阶段，标题：《TSSC: A New Deep Learning Model for Accurate Pea Leaf Disease Identification》  


> 提出时空尺度通道网络（TSSC）模型，基于TensorFlow框架实现五类豌豆常见病害与健康状态的高精度识别，助力农业病害智能化诊断。  


=======
> 官方TensorFlow实现 | 论文处于投刊阶段，标题：《TSSC: A New Deep Learning Model for Accurate Pea Leaf Disease Identification》  


> 提出时空尺度通道网络（TSSC）模型，基于TensorFlow框架实现五类豌豆常见病害与健康状态的高精度识别，助力农业病害智能化诊断。  


>>>>>>> Stashed changes
## 1. 研究背景与模型定位  

豌豆作为重要豆科作物，其叶片病害（如白粉病、锈病、霜霉病等）易导致产量下降，传统人工检测存在效率低、依赖经验的问题。  
>>>>>>> Stashed changes

本文提出**时空尺度通道网络（TSSC）**，通过创新的多维度特征融合与跨尺度建模机制，解决豌豆病害“多类别区分难、复杂背景干扰大”的问题。模型基于TensorFlow框架实现，包含4个核心卷积层与3种注意力机制，在五类豌豆病害数据集上实现优异的分类性能，为农业病害自动化诊断提供高效解决方案。  

<<<<<<< Updated upstream
> 提出时空尺度通道网络（TSSC）模型，基于TensorFlow框架实现五类豌豆常见病害与健康状态的高精度识别，助力农业病害智能化诊断。  

本文提出**时空尺度通道网络（TSSC）**，通过创新的多维度特征融合与跨尺度建模机制，解决豌豆病害“多类别区分难、复杂背景干扰大”的问题。模型基于TensorFlow框架实现，包含4个核心卷积层与3种注意力机制，在五类豌豆病害数据集上实现优异的分类性能，为农业病害自动化诊断提供高效解决方案。  

## 1. 研究背景与模型定位  
>>>>>>> Stashed changes

<<<<<<< Updated upstream
### 3.1 数据集概况  

<<<<<<< Updated upstream
=======

=======
>>>>>>> Stashed changes
## 2. TSSC核心创新点  

1. **三层注意力机制协同**：  
   - **SE注意力（Squeeze-and-Excitation）**：通过全局信息压缩与通道权重学习，强化关键病害特征通道；  
   - **三邻域通道注意力**：考虑每个通道与左右邻域通道的关联性，动态优化特征表达；  
   - **分裂注意力（Split Attention）**：将通道分组并独立学习注意力，增强局部特征交互能力。  

2. **高效特征提取结构**：  
   采用4个精心设计的卷积层（核大小4×4、2×2、5×5、3×3），配合池化操作逐步压缩空间维度，在保留病害细节特征的同时提升计算效率。  

3. **多类别适配优化**：  
   针对五类病害的细粒度差异（如白粉病的白色霉层与霜霉病的黄色斑点），设计类别感知损失函数，降低相似病害的混淆度。  


## 3. 实验数据集：五类豌豆病害数据集  

### 3.1 数据集概况  

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
本研究基于**五类豌豆病害识别数据集**，包含五种常见豌豆叶片状态，数据集存储于百度网盘，需自行下载后使用：  

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |
|------------|-------------------------|----------|------------|-----------------------|
<<<<<<< Updated upstream
<<<<<<< Updated upstream
| 五类豌豆数据集 | 白粉病（Powdery mildew）、根腐病（Root rot）、潜叶虫（Leaf miner）、褐斑病（Brown spot） + 健康叶片（Healthy） | 7,000+ | 统一resize至400×400（适配模型输入） | 3:1:1 |  

=======
本文提出**时空尺度通道网络（TSSC）**，通过创新的多维度特征融合与跨尺度建模机制，解决豌豆病害“多类别区分难、复杂背景干扰大”的问题。模型基于TensorFlow框架实现，包含4个核心卷积层与3种注意力机制，在五类豌豆病害数据集上实现优异的分类性能，为农业病害自动化诊断提供高效解决方案。  


## 2. TSSC核心创新点  

1. **三层注意力机制协同**：  
   - **SE注意力（Squeeze-and-Excitation）**：通过全局信息压缩与通道权重学习，强化关键病害特征通道；  
   - **三邻域通道注意力**：考虑每个通道与左右邻域通道的关联性，动态优化特征表达；  
   - **分裂注意力（Split Attention）**：将通道分组并独立学习注意力，增强局部特征交互能力。  

2. **高效特征提取结构**：  
   采用4个精心设计的卷积层（核大小4×4、2×2、5×5、3×3），配合池化操作逐步压缩空间维度，在保留病害细节特征的同时提升计算效率。  

3. **多类别适配优化**：  
   针对五类病害的细粒度差异（如白粉病的白色霉层与霜霉病的黄色斑点），设计类别感知损失函数，降低相似病害的混淆度。  
=======
| 五类豌豆数据集 | 白粉病（Powdery mildew）、根腐病（Root rot）、潜叶虫（Leaf miner）、褐斑病（Brown spot） + 健康叶片（Healthy） | 12,000+ | 统一resize至400×400（适配模型输入） | 7:1:2（通过代码自动划分） |  
=======
| 五类豌豆数据集 | 白粉病（Powdery mildew）、根腐病（Root rot）、潜叶虫（Leaf miner）、褐斑病（Brown spot） + 健康叶片（Healthy） | 12,000+ | 统一resize至400×400（适配模型输入） | 7:1:2（通过代码自动划分） |  


### 3.2 数据集获取与结构  
>>>>>>> Stashed changes

1. **下载链接**：  
   百度网盘链接：https://pan.baidu.com/s/1pAOw9JjZ3qtCVLDNJupn9Q 
   提取码: x3ab 

<<<<<<< Updated upstream
### 3.2 数据集获取与结构  
>>>>>>> Stashed changes

1. **下载链接**：  
   百度网盘链接：https://pan.baidu.com/s/1pAOw9JjZ3qtCVLDNJupn9Q 
   提取码: x3ab 

<<<<<<< Updated upstream
## 3. 实验数据集：五类豌豆病害数据集  

### 3.1 数据集概况  

本研究基于**五类豌豆病害识别数据集**，包含五种常见豌豆叶片状态，数据集存储于百度网盘，需自行下载后使用：  

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |
|------------|-------------------------|----------|------------|-----------------------|
| 五类豌豆数据集 | 白粉病（Powdery mildew）、根腐病（Root rot）、潜叶虫（Leaf miner）、褐斑病（Brown spot） + 健康叶片（Healthy） | 12,000+ | 统一resize至400×400（适配模型输入） | 7:1:2（通过代码自动划分） |  
>>>>>>> Stashed changes

### 3.2 数据集获取与结构  

<<<<<<< Updated upstream
1. **下载链接**：  
   百度网盘链接：https://pan.baidu.com/s/18FxZMhVcK-5hRwAhoJS8mQ  
   提取码: bq9g  

2. **文件夹组织**（下载后解压至项目根目录，结构如下）：  
```  
pea_disease_dataset/  
├── powdery_mildew/       # 豌豆白粉病叶片图像  
├── root_rot/             # 豌豆根腐病叶片图像  
├── leaf_miner/           # 豌豆潜叶虫危害叶片图像  
├── brown_spot/           # 豌豆褐斑病叶片图像  
└── healthy/              # 健康豌豆叶片图像  
```  

=======
### 3.2 数据集获取与结构  

1. **下载链接**：  
   百度网盘链接：https://pan.baidu.com/s/1pAOw9JjZ3qtCVLDNJupn9Q 
   提取码: x3ab 

=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
2. **文件夹组织**（下载后解压至项目根目录，结构如下）：  
```  
pea_disease_dataset/
├── 白粉病/       # 豌豆白粉病叶片图像
├── 潜叶虫/       # 豌豆潜叶虫病叶片图像
├── 褐斑病/       # 豌豆褐斑病叶片图像
├── 根腐病/       # 豌豆根腐病叶片图像
└── 健康叶片/     # 健康豌豆叶片图像
```  

<<<<<<< Updated upstream
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

## 4. 实验环境配置  

### 4.1 依赖安装  

推荐使用Anaconda创建虚拟环境，确保依赖版本匹配（TensorFlow框架核心依赖）：  

```bash  
# 1. 创建并激活虚拟环境  
conda create -n tssc-tf python=3.10  
conda activate tssc-tf  

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
# 2. 安装TensorFlow（支持GPU/CPU，示例为GPU版本）  
pip install Tensorflow-gpu1.14.0
pip install Keras2.2.4. 
=======
# 2. 安装TensorFlow（支持GPU/CPU，示例为CPU版本）  
pip install tensorflow==2.15.0  
>>>>>>> Stashed changes
=======
# 2. 安装TensorFlow（支持GPU/CPU，示例为CPU版本）  
pip install tensorflow==2.15.0  
>>>>>>> Stashed changes
=======
# 2. 安装TensorFlow（支持GPU/CPU，示例为CPU版本）  
pip install tensorflow==2.15.0  
>>>>>>> Stashed changes

# 3. 安装其他依赖库  
pip install numpy~=2.0.2 matplotlib~=3.9.4 opencv-python~=4.12.0.88  
pip install pandas~=2.3.1 pillow~=11.2.1 scikit-learn~=1.5.1  
pip install tqdm~=4.67.1 tensorboard~=2.15.1  
```  

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
## 5 代码使用说明  
=======
## 5. 代码使用说明  
>>>>>>> Stashed changes
=======
## 5. 代码使用说明  
>>>>>>> Stashed changes
=======
## 5. 代码使用说明  
>>>>>>> Stashed changes

### 5.1 模型训练  

运行`train.py`脚本启动训练，支持通过参数调整训练配置，示例命令：  

```bash  
python train.py \  
  --data_dir ./pea_disease_dataset \  # 数据集根目录（解压后的路径）  
  --epochs 60 \  
  --batch_size 16 \  
  --lr 1e-4 \  
  --weight_decay 1e-5 \  
  --save_dir ./weights \  
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
  --log_interval 10  # 每10个batch打印一次训练日志  
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
```  


#### 关键参数说明：  

| 参数名 | 含义 | 默认值 |
|-----------------|---------------------------------------|-----------------|
| `--data_dir` | 数据集根目录路径 | `./pea_disease_dataset` |
| `--epochs` | 训练轮数 | 60 |
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
| `--batch_size` | 批次大小（根据显存调整，8/16/32） | 16 |
=======
| `--batch_size` | 批次大小（根据GPU显存调整，8/16/32） | 16 |
>>>>>>> Stashed changes
=======
| `--batch_size` | 批次大小（根据GPU显存调整，8/16/32） | 16 |
>>>>>>> Stashed changes
| `--lr` | 初始学习率 | 1e-4 |
| `--save_dir` | 训练权重保存目录（.h5格式） | `./weights` |
| `--device` | 训练设备（`GPU`或`CPU`） | `GPU` |  


#### 训练输出：  

- 训练过程中，模型会自动保存**验证集宏平均F1值最高**的权重至`--save_dir`目录，文件名为`best_tssc.h5`；  
- 训练日志（损失值、准确率、F1值）通过TensorBoard可视化，可通过`tensorboard --logdir=./logs`查看。  
<<<<<<< Updated upstream

### 5.2 模型预测  

使用训练好的权重进行单张豌豆叶片图像预测，运行`predict.py`脚本，示例命令：  

```bash  
python predict.py \  
  --image_path ./examples/pea_powdery_mildew.jpg \  # 输入图像路径  
  --weight_path ./weights/best_tssc.h5 \  # 预训练权重路径（TensorFlow .h5格式）  
  --device GPU  
```  

=======
| `--batch_size` | 批次大小（根据GPU显存调整，8/16/32） | 16 |
| `--lr` | 初始学习率 | 1e-4 |
| `--save_dir` | 训练权重保存目录（.h5格式） | `./weights` |
| `--device` | 训练设备（`GPU`或`CPU`） | `GPU` |  
>>>>>>> Stashed changes

#### 预测输出示例：  

<<<<<<< Updated upstream
```  
输入图像路径：./examples/pea_powdery_mildew.jpg  
预测类别：白粉病（Powdery mildew）  
置信度：0.9976  
```  

### 5.2 模型预测  

<<<<<<< Updated upstream
## 6. 项目文件结构  

```  
tssc-for-pea-disease-identification/  
├── pea_disease_dataset/  # 五类豌豆病害数据集（需从百度网盘下载）  
├── examples/             # 预测示例图像（如powdery_mildew_example.jpg）  
├── models/                 #整体模型实现
│   ├── se_module.py          # SE注意力机制实现  
│   ├── split_attention.py    # 分裂注意力机制实现  
│   ├── three_neighbor_attention.py  # 三邻域通道注意力机制实现  
│   ├──  TSSC.py               # TSSC主模型（整合上述注意力模块）  
├── dataset/              # 数据处理文件夹  
│   ├── data_loader.py    # 数据集加载与预处理（自动划分训练/验证/测试集）  
├── train.py              # 模型训练脚本（TensorFlow版）  
├── predict.py            # 模型预测脚本（TensorFlow版）    
├── weights/              # 模型权重保存目录（自动生成）  
└── README.md             # 项目说明文档（本文档）  
```  


## 7. 已知问题与注意事项  

1. **框架适配**：本项目仅支持TensorFlow 2.10+版本，不兼容PyTorch环境；  
2. **输入尺寸**：模型固定输入为400×400×3，预测时会自动resize输入图像，建议原始图像分辨率≥400×400以保留细节；  
3. **数据集扩展**：如需新增病害类别，需补充对应图像数据并修改`TSSC.py`中输出层的`num_classes`参数。  

=======
#### 训练输出：  

- 训练过程中，模型会自动保存**验证集宏平均F1值最高**的权重至`--save_dir`目录，文件名为`best_tssc.h5`；  
- 训练日志（损失值、准确率、F1值）通过TensorBoard可视化，可通过`tensorboard --logdir=./logs`查看。  


### 5.2 模型预测  

使用训练好的权重进行单张豌豆叶片图像预测，运行`predict.py`脚本，示例命令：  

```bash  
python predict.py \  
  --image_path ./examples/pea_powdery_mildew.jpg \  # 输入图像路径  
  --weight_path ./weights/best_tssc.h5 \  # 预训练权重路径（TensorFlow .h5格式）  
```  


#### 预测输出示例：  

```  
输入图像路径：./examples/pea_powdery_mildew.jpg  
预测类别：白粉病（Powdery mildew）  
置信度：0.9976  
```  
>>>>>>> Stashed changes

## 8. 引用与联系方式  

<<<<<<< Updated upstream
### 8.1 引用方式  
=======
=======
使用训练好的权重进行单张豌豆叶片图像预测，运行`predict.py`脚本，示例命令：  

```bash  
python predict.py \  
  --image_path ./examples/pea_powdery_mildew.jpg \  # 输入图像路径  
  --weight_path ./weights/best_tssc.h5 \  # 预训练权重路径（TensorFlow .h5格式）  
```  


#### 预测输出示例：  

```  
输入图像路径：./examples/pea_powdery_mildew.jpg  
预测类别：白粉病（Powdery mildew）  
置信度：0.9976  
```  


>>>>>>> Stashed changes
=======


### 5.2 模型预测  

使用训练好的权重进行单张豌豆叶片图像预测，运行`predict.py`脚本，示例命令：  

```bash  
python predict.py \  
  --image_path ./examples/pea_powdery_mildew.jpg \  # 输入图像路径  
  --weight_path ./weights/best_tssc.h5 \  # 预训练权重路径（TensorFlow .h5格式）  
```  


#### 预测输出示例：  

```  
输入图像路径：./examples/pea_powdery_mildew.jpg  
预测类别：白粉病（Powdery mildew）  
置信度：0.9976  
```  


>>>>>>> Stashed changes
## 6. 项目文件结构  

```
tssc-for-pea-disease-identification/
├── pea_disease_dataset/  # 五类豌豆病害数据集（需从百度网盘下载）
├── examples/             # 预测示例图像（如rust_example.png）
├── models/               # 模型定义文件夹
│   ├── Tssc.py           # TSSC核心代码（含特征交互、尺度建模模块）
│   ├── se_module.py      # 实现 Squeeze-and-Excitation 注意力机制
│   ├── split_attention.py    # 实现split注意力机制
│   ├── three_neighbor_attention.py   # 实现三邻域通道注意力机制
├── dataset/              # 数据处理文件夹
│   ├── data_loader.py    # 数据集加载与预处理（自动划分训练/验证/测试集）
├── train.py              # 模型训练脚本
├── predict.py            # 模型预测脚本
└── README.md             # 项目说明文档（本文档）
```  


## 8. 已知问题与注意事项  

1. **框架适配**：本项目仅支持TensorFlow 2.10+版本，不兼容PyTorch环境；  
2. **输入尺寸**：模型固定输入为400×400×3，预测时会自动resize输入图像，建议原始图像分辨率≥400×400以保留细节；   
3. **数据集扩展**：如需新增病害类别，需补充对应图像数据并修改`TSSC.py`中输出层的`num_classes`参数。  


## 9. 引用与联系方式  

### 9.1 引用方式  
<<<<<<< Updated upstream
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：  

```bibtex  
@article{tssc_pea_disease,  
  title={TSSC: A New Deep Learning Model for Accurate Pea Leaf Disease Identification},  
  author={[作者姓名，待发表时补充]},  
  journal={[期刊名称，待录用后补充]},  
  year={2024},  
  note={Manuscript submitted for publication}  
}  
```  


<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
### 8.2 联系方式  

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：changyibu@huuc.edu.cn  
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
=======
### 9.2 联系方式  

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：changyibu@huuc.edu.cn  
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
>>>>>>> Stashed changes
=======
### 9.2 联系方式  

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：changyibu@huuc.edu.cn  
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
>>>>>>> Stashed changes
=======
### 9.2 联系方式  

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：changyibu@huuc.edu.cn  
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
>>>>>>> Stashed changes
