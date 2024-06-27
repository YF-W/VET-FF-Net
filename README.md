# Value Vector Enhanced Transformer for Feature Fusion Segmentation

paper address: 

***Semantic segmentation is of great significance for the development of smart healthcare, as it enables the precise labeling and separation of different anatomical structures or lesion areas. Currently, Transformer, with it exceptional ability to capture broad contextual information, have led to the widespread application of Vision Transformers (ViT) in semantic segmentation network architectures. However, in their underlying mechanism, the calculation of attention scores only involves the local target matching relevance between regional targets (Q) and global positions (K). Both theoretical research and preliminary experiments have shown that this approach is insufficient to enhance the global attention to image pixel information.***

# Paper:VET-FF Net (Value Vector Enhanced Transformer for Feature Fusion Segmentation Network)

**Authors: Yuefei Wang, Yuanhong Wei, Xi Yub, Jin Wang, Yutong Zhang, Li Zhang, Zhixuan Chen, Yuxuan Wan, Liangyan Zhao, Qinyu Zhao, Binxiong Li, Yunuo Zhou, Briant Loïc, Yixi Yang**

## 1. Architecture Overview

![image-20240627135653401](https://github.com/YF-W/VET-FF-Net/blob/78bbdef5fb4c9c40094ca8dd78c08030d61622d5/VET-FF%20Net.png)

***VET-FF Net is a multi-branch asymmetric codec network. Where the encoder uses ResNet 34 and ViT attention mechanism. The TGAD-FM module is introduced at the codec intersection to replace the simple processing method in the traditional U-shaped network, fusing multi-source coded features to enrich the key features such as texture and location of lesions. The decoder integrates multi-source information, including the semantic parsing results of the front layer, the ViT generalized context information, the CNN local feature information, and the key semantics of the TGAD-FM module by means of channel connectivity, so as to ensure the accuracy of the image parsing restoration.***

## 2. Our network baseline

![image-20240627135653402](https://github.com/YF-W/VET-FF-Net/blob/3a4dd2f911976b902ee21af857443dd0619b8c96/VET-FF%20Net%20baseline.png)

***We propose a "dual encoder - single decoder" network paradigm, which, unlike the traditional single coding and decoding structure, is able to capture a richer channel tensor in terms of semantic richness, thus capturing the different semantic contents of the foci in a more comprehensive way; in terms of semantic categorization, the multi-branching structure can be used to deploy different key tasks for different branches. Based on this idea, this study constructs a dual encoder branch, which adopts ResNet and ViT respectively. the ResNet branch is used for local feature extraction, and the VE-ViT branch is used for global feature extraction.***

## 3. Module 1: VE-ViT

![image-20240627135653404](https://github.com/YF-W/VET-FF-Net/blob/d49737e83fde75bef0d4907e16bf43d095f88890/VET-FF%20Net%20module1.png)

***We propose a self-attention mechanism that consolidates multi-dimensional comprehensive information, introducing Global-Local Enhancement and Depthwise Separable Patch Embedding. This mechanism enhances the value from four aspects: global max, sequential max, global average, and sequential average, thereby enriching the value. It integrates four types of local and global feature information and reallocates and recombines them through the attention mechanism, enabling the model to better handle image content. See article for detailed structure and function.***

| **Layer  Name**  | **Module  Structure**                               |
| ---------------- | --------------------------------------------------- |
| **Input  Size**  | Input1  = [1, 196, 768] Input2 = [1, 3, 224, 224]   |
| **Layer 1**      | V(x1)  = Input1 + Position Embedding                |
| **Output  Size** | [1,  196, 768]                                      |
| **Layer 2**      | V(x2)  = V(x1) + VE-MHSA[Layer  Norm(V(x1)),Input2] |
| **Output  Size** | [1,  196, 768]                                      |
| **Layer 3**      | Output  = V(x2) + MLP[Layer  Norm(V(x2))]           |
| **Output  Size** | [1,  196, 768]                                      |

## 4. Module 2: TGAD-FM

![image-20240627135525533](https://github.com/YF-W/VET-FF-Net/blob/44a4e52e4089f5b5e02969615e04668266dafd42/VET-FF%20Net%20module2.png)

***We construct a fusion module, TGAD-FM, between the encoder and decoder for the intercoder-decoder transition, which contains three branches. (1) Channel Concatenation Attention Branch (CCab). This branch is used to capture the key region information of feature channels at different levels and scales. (2) Spital Addition Attention Branch (SAab). This branch is used to capture the spatial information of feature semantics at different levels and scales. (3) Information Reinforcement and Inhibition Branch (IRIB). This branch is used to strengthen key features and weaken secondary features. The fusion module not only extracts complementary semantic information, but also ensures the diversity of features and strengthens the key semantics.***

| **Layer  Name**  | **Module  Structure**                |                       |                       |
| ---------------- | ------------------------------------ | --------------------- | --------------------- |
| **Input  Size**  | Input1=Input2=[1,64,112,112]         |                       |                       |
| **Layer  1**     | C(x1) = Concat(Input1,Input2)        | S(x1) = Input1+Input2 | I(x1) = Input1·Input2 |
| **Output  Size** | [1,128,112,112]                      | [1,64,112,112]        | [1,64,112,112]        |
| **Layer  2**     | C(x2)=CCab[C(x1)]                    | S(x2)=SAab[S(x1)]     | I(x2)=IRIB[I(x1)]     |
| **Output  Size** | [1,128,1,1]                          | [1,1,112,112]         | [1,64,112,112]        |
| **Layer  3**     | C(x3)=Conv[[1*1,64]][][ C(x1)·C(x2)] | S(x3)=S(x1)·S(x2)     | I(x3)=I(x1)·I(x2)     |
| **Output  Size** | [1,64,112,112]                       | [1,64,112,112]        | [1,64,112,112]        |
| **Layer 4**      | Output=C(x3)+S(x3)+I(x3)             |                       |                       |
| **Output  Size** | [1,64,112,112]                       |                       |                       |

## Datasets:

1. The LUNG dataset: https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data
2. The MICCAI2015-CVC ClinicDB dataset: https://polyp.grand-challenge.org/CVCClinicDB/
3. The NEVUS dataset: https://challenge.isic-archive.com/data/#2017
4. The MICCAI-Tooth-Segmentation dataset: https://tianchi.aliyun.com/dataset/156596
5. The Brain MRI segmentation dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
6. The 2018 Data Science Bowl dataset: https://www.kaggle.com/competitions/data-science-bowl-2018/data
7. The MICCAI 2020 TN-SCUI dataset: https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st
