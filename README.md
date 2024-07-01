# ENDEF-framework
machine learning homework

**1、introduction and run**

Generalizing to the Future: Mitigating Entity Bias in Fake News Detection (ENDEF)

The wide dissemination of fake news is increasingly threatening both individuals and society. Fake news detection aims to train a model on the past news and detect fake news of the future. Though great efforts have been made, existing fake news detection methods overlooked the unintended entity bias in the real-world data, which seriously influences models' generalization ability to future data. For example, 97% of news pieces in 2010-2017 containing the entity 'Donald Trump' are real in our data, but the percentage falls down to merely 33% in 2018. This would lead the model trained on the former set to hardly generalize to the latter, as it tends to predict news pieces about 'Donald Trump' as real for lower training loss. In this paper, we propose an entity debiasing framework (ENDEF) which generalizes fake news detection models to the future data by mitigating entity bias from a cause-effect perspective. Based on the causal graph among entities, news contents, and news veracity, we separately model the contribution of each cause (entities and contents) during training. In the inference stage, we remove the direct effect of the entities to mitigate entity bias. Extensive offline experiments on the English and Chinese datasets demonstrate that the proposed framework can largely improve the performance of base fake news detectors, and online tests verify its superiority in practice. To the best of our knowledge, this is the first work to explicitly improve the generalization ability of fake news detection models to the future data.

The proposed ENDEF is model-agnostic, and it can be implemented with diverse base models. This repository provides the implementations of ENDEF and five base models (BiGRU, EANN, BERT, MDFEND, BERT-Emo):

BiGRU：On the Properties of Neural Machine Translation: Encoder-Decoder Approaches
EANN: EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection (KDD 2018)
BERT: Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019)
MDFEND: MDFEND: Multi-domain Fake News Detection (CIKM 2021)
BERT-Emo: Mining Dual Emotion for Fake News Detection (WWW 2021)

Requirements

Python 3.6

PyTorch > 1.0

Pandas

Numpy

Tqdm

**run**

You can run this code through:

python main.py --gpu 1 --lr 0.0001 --model_name bigru



**2、new dataset**

In the first place in the new data set https://github.com/junyachen/Data-examples?tab=readme-ov-file download links data set


**3、the result**

3.1 gossipcop_v3-1_style_based_fake data set input into the model results

![image](https://github.com/ml-master/ENDEF-framework/assets/139029753/289c0c54-b721-4316-8809-cb0bbc6b3cf0)

3.2 gossipcop_v3-5_style_based_legitimate data set input into the model results

![image](https://github.com/ml-master/ENDEF-framework/assets/139029753/2e4d37b3-ff49-41f2-baf7-66c654cee892)


4、Ablation experiment

4.1 The results of the gossipcop dataset were not used under the proposed framework

![image](https://github.com/ml-master/ENDEF-framework/assets/139029753/cbb6f33e-ee6d-4843-bb00-620559c046f1)

4.2 The results of gossipcop data set use under the proposed framework

![image](https://github.com/ml-master/ENDEF-framework/assets/139029753/23ec553c-feea-4dde-b126-2d2a468c2573)


