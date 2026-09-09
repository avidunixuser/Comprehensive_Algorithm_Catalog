# Complete Table of Contents

## Comprehensive Algorithm Catalog

A guide to how computers learn, grouped by the feedback each method uses.
It also includes a closer look at Mixture of Experts, comparison tables,
and a glossary. Each numbered entry explains a method or model family.
Different model versions are kept separate where their details differ.

Start with each entry's **In plain English** introduction. Then read how it
works and its worked example. The math is optional on a first read.

[Repository overview](README.md) | [Reading guide](book/00-reading-guide.md)

## Preface: How to Read an Algorithm Catalog

[Read this chapter](book/00-reading-guide.md)

- [P.1 The book's organization](book/00-reading-guide.md#p1-the-books-organization)
- [P.2 What determines a learning category?](book/00-reading-guide.md#p2-what-determines-a-learning-category)
- [P.3 Scope and the meaning of comprehensive](book/00-reading-guide.md#p3-scope-and-the-meaning-of-comprehensive)
- [P.4 Anatomy of an entry](book/00-reading-guide.md#p4-anatomy-of-an-entry)
- [P.5 Evidence policy for worked examples](book/00-reading-guide.md#p5-evidence-policy-for-worked-examples)
- [P.6 Reading numbers without being misled](book/00-reading-guide.md#p6-reading-numbers-without-being-misled)
- [P.7 Mathematical and systems notation](book/00-reading-guide.md#p7-mathematical-and-systems-notation)
- [P.8 From a catalog to a deployed system](book/00-reading-guide.md#p8-from-a-catalog-to-a-deployed-system)

[Coverage and continuation manifest](book/00-reading-guide.md#coverage-and-continuation-manifest)

## 1. Supervised Learning Algorithms: Classical Models

[Read this chapter](book/01-supervised-classical.md)


**[1.1 Linear, generalized, and discriminant models](book/01-supervised-classical.md#11-linear-generalized-and-discriminant-models)**

- [1.1.1 Ordinary least squares](book/01-supervised-classical.md#111-ordinary-least-squares)
- [1.1.2 Ridge regression](book/01-supervised-classical.md#112-ridge-regression)
- [1.1.3 Lasso](book/01-supervised-classical.md#113-lasso)
- [1.1.4 Elastic net](book/01-supervised-classical.md#114-elastic-net)
- [1.1.5 Logistic regression: binary and multinomial](book/01-supervised-classical.md#115-logistic-regression-binary-and-multinomial)
- [1.1.6 Generalized additive models](book/01-supervised-classical.md#116-generalized-additive-models)
- [1.1.7 Linear discriminant analysis](book/01-supervised-classical.md#117-linear-discriminant-analysis)
- [1.1.8 Quadratic discriminant analysis](book/01-supervised-classical.md#118-quadratic-discriminant-analysis)

**[1.2 Trees and ensembles](book/01-supervised-classical.md#12-trees-and-ensembles)**

- [1.2.1 CART](book/01-supervised-classical.md#121-cart)
- [1.2.2 C4.5](book/01-supervised-classical.md#122-c45)
- [1.2.3 Bagging](book/01-supervised-classical.md#123-bagging)
- [1.2.4 Random forest](book/01-supervised-classical.md#124-random-forest)
- [1.2.5 Extremely randomized trees](book/01-supervised-classical.md#125-extremely-randomized-trees)
- [1.2.6 AdaBoost](book/01-supervised-classical.md#126-adaboost)
- [1.2.7 Gradient-boosted decision trees](book/01-supervised-classical.md#127-gradient-boosted-decision-trees)
- [1.2.8 XGBoost](book/01-supervised-classical.md#128-xgboost)
- [1.2.9 LightGBM](book/01-supervised-classical.md#129-lightgbm)
- [1.2.10 CatBoost](book/01-supervised-classical.md#1210-catboost)

**[1.3 Neighbors and kernels](book/01-supervised-classical.md#13-neighbors-and-kernels)**

- [1.3.1 k-nearest neighbors](book/01-supervised-classical.md#131-k-nearest-neighbors)
- [1.3.2 Support-vector classification](book/01-supervised-classical.md#132-support-vector-classification)
- [1.3.3 Support-vector regression](book/01-supervised-classical.md#133-support-vector-regression)
- [1.3.4 Kernel ridge regression](book/01-supervised-classical.md#134-kernel-ridge-regression)
- [1.3.5 Gaussian processes: regression and classification](book/01-supervised-classical.md#135-gaussian-processes-regression-and-classification)

**[1.4 Probabilistic, structured, and survival models](book/01-supervised-classical.md#14-probabilistic-structured-and-survival-models)**

- [1.4.1 Naive Bayes: Gaussian, multinomial, and Bernoulli](book/01-supervised-classical.md#141-naive-bayes-gaussian-multinomial-and-bernoulli)
- [1.4.2 Bayesian networks](book/01-supervised-classical.md#142-bayesian-networks)
- [1.4.3 Supervised hidden Markov models](book/01-supervised-classical.md#143-supervised-hidden-markov-models)
- [1.4.4 Conditional random fields](book/01-supervised-classical.md#144-conditional-random-fields)
- [1.4.5 Cox proportional-hazards regression](book/01-supervised-classical.md#145-cox-proportional-hazards-regression)

**[Public bibliography and evidence notes](book/01-supervised-classical.md#public-bibliography-and-evidence-notes)**


[Coverage and continuation manifest](book/01-supervised-classical.md#coverage-and-continuation-manifest)

## 1. Supervised Learning Algorithms: Neural Architectures

[Read this chapter](book/02-supervised-neural.md)


**[1.5 Neural foundations](book/02-supervised-neural.md#15-neural-foundations)**

- [1.5.1 Perceptron](book/02-supervised-neural.md#151-perceptron)
- [1.5.2 Multilayer perceptron (MLP)](book/02-supervised-neural.md#152-multilayer-perceptron-mlp)

**[1.6 Convolutional families](book/02-supervised-neural.md#16-convolutional-families)**

- [1.6.1 Generic convolutional neural network (CNN)](book/02-supervised-neural.md#161-generic-convolutional-neural-network-cnn)
- [1.6.2 LeNet](book/02-supervised-neural.md#162-lenet)
- [1.6.3 AlexNet](book/02-supervised-neural.md#163-alexnet)
- [1.6.4 VGG](book/02-supervised-neural.md#164-vgg)
- [1.6.5 ResNet](book/02-supervised-neural.md#165-resnet)
- [1.6.6 Inception](book/02-supervised-neural.md#166-inception)
- [1.6.7 EfficientNet](book/02-supervised-neural.md#167-efficientnet)
- [1.6.8 ConvNeXt](book/02-supervised-neural.md#168-convnext)
- [1.6.9 DenseNet](book/02-supervised-neural.md#169-densenet)
- [1.6.10 MobileNet](book/02-supervised-neural.md#1610-mobilenet)

**[1.7 Recurrent and sequence architectures](book/02-supervised-neural.md#17-recurrent-and-sequence-architectures)**

- [1.7.1 Vanilla recurrent neural network (RNN)](book/02-supervised-neural.md#171-vanilla-recurrent-neural-network-rnn)
- [1.7.2 Long short-term memory (LSTM)](book/02-supervised-neural.md#172-long-short-term-memory-lstm)
- [1.7.3 Gated recurrent unit (GRU)](book/02-supervised-neural.md#173-gated-recurrent-unit-gru)
- [1.7.4 Sequence-to-sequence with Bahdanau attention](book/02-supervised-neural.md#174-sequence-to-sequence-with-bahdanau-attention)
- [1.7.5 Original encoder-decoder Transformer](book/02-supervised-neural.md#175-original-encoder-decoder-transformer)

**[1.8 Vision transformers and prediction architectures](book/02-supervised-neural.md#18-vision-transformers-and-prediction-architectures)**

- [1.8.1 Vision Transformer (ViT)](book/02-supervised-neural.md#181-vision-transformer-vit)
- [1.8.2 Swin Transformer](book/02-supervised-neural.md#182-swin-transformer)
- [1.8.3 U-Net](book/02-supervised-neural.md#183-u-net)
- [1.8.4 Faster R-CNN](book/02-supervised-neural.md#184-faster-r-cnn)
- [1.8.5 YOLO, original version](book/02-supervised-neural.md#185-yolo-original-version)
- [1.8.6 DETR](book/02-supervised-neural.md#186-detr)

**[1.9 Graph networks](book/02-supervised-neural.md#19-graph-networks)**

- [1.9.1 Graph convolutional network (GCN)](book/02-supervised-neural.md#191-graph-convolutional-network-gcn)
- [1.9.2 Graph attention network (GAT)](book/02-supervised-neural.md#192-graph-attention-network-gat)
- [1.9.3 GraphSAGE](book/02-supervised-neural.md#193-graphsage)

**[1.10 Metric and event-based networks](book/02-supervised-neural.md#110-metric-and-event-based-networks)**

- [1.10.1 Supervised Siamese networks](book/02-supervised-neural.md#1101-supervised-siamese-networks)
- [1.10.2 Spiking neural networks (SNNs)](book/02-supervised-neural.md#1102-spiking-neural-networks-snns)

**[1.11 Supervised mixtures](book/02-supervised-neural.md#111-supervised-mixtures)**

- [1.11.1 Adaptive mixture of local experts](book/02-supervised-neural.md#1111-adaptive-mixture-of-local-experts)
- [1.11.2 GShard multilingual translation](book/02-supervised-neural.md#1112-gshard-multilingual-translation)

[Coverage and continuation manifest](book/02-supervised-neural.md#coverage-and-continuation-manifest)

## 2. Semi-Supervised Learning Algorithms

[Read this chapter](book/03-semi-supervised.md)

- [Reading the objectives and costs](book/03-semi-supervised.md#reading-the-objectives-and-costs)
- [Assumptions, failure modes, and evaluation](book/03-semi-supervised.md#assumptions-failure-modes-and-evaluation)

**[2.1 Self-labeling and multiple-view approaches](book/03-semi-supervised.md#21-self-labeling-and-multiple-view-approaches)**

- [2.1.1 Self-training and Pseudo-Label](book/03-semi-supervised.md#211-self-training-and-pseudo-label)
- [2.1.2 Co-training](book/03-semi-supervised.md#212-co-training)
- [2.1.3 Tri-training](book/03-semi-supervised.md#213-tri-training)
- [2.1.4 Noisy Student](book/03-semi-supervised.md#214-noisy-student)

**[2.2 Low-density boundaries and manifold regularization](book/03-semi-supervised.md#22-low-density-boundaries-and-manifold-regularization)**

- [2.2.1 Transductive and semi-supervised SVM](book/03-semi-supervised.md#221-transductive-and-semi-supervised-svm)
- [2.2.2 Laplacian SVM and manifold regularization](book/03-semi-supervised.md#222-laplacian-svm-and-manifold-regularization)

**[2.3 Graph label inference](book/03-semi-supervised.md#23-graph-label-inference)**

- [2.3.1 Label propagation through Gaussian fields and harmonic functions](book/03-semi-supervised.md#231-label-propagation-through-gaussian-fields-and-harmonic-functions)
- [2.3.2 Label spreading and local-and-global consistency](book/03-semi-supervised.md#232-label-spreading-and-local-and-global-consistency)

**[2.4 Entropy and consistency](book/03-semi-supervised.md#24-entropy-and-consistency)**

- [2.4.1 Entropy minimization](book/03-semi-supervised.md#241-entropy-minimization)
- [2.4.2 Pi Model](book/03-semi-supervised.md#242-pi-model)
- [2.4.3 Temporal ensembling](book/03-semi-supervised.md#243-temporal-ensembling)
- [2.4.4 Mean Teacher](book/03-semi-supervised.md#244-mean-teacher)
- [2.4.5 Virtual adversarial training](book/03-semi-supervised.md#245-virtual-adversarial-training)
- [2.4.6 Unsupervised Data Augmentation](book/03-semi-supervised.md#246-unsupervised-data-augmentation)

**[2.5 Combined modern recipes](book/03-semi-supervised.md#25-combined-modern-recipes)**

- [2.5.1 MixMatch](book/03-semi-supervised.md#251-mixmatch)
- [2.5.2 ReMixMatch](book/03-semi-supervised.md#252-remixmatch)
- [2.5.3 FixMatch](book/03-semi-supervised.md#253-fixmatch)
- [2.5.4 FlexMatch](book/03-semi-supervised.md#254-flexmatch)
- [2.5.5 FreeMatch](book/03-semi-supervised.md#255-freematch)

**[2.6 Generative and reconstruction approaches](book/03-semi-supervised.md#26-generative-and-reconstruction-approaches)**

- [2.6.1 Semi-supervised variational autoencoders: M1 and M2](book/03-semi-supervised.md#261-semi-supervised-variational-autoencoders-m1-and-m2)
- [2.6.2 Semi-supervised GAN with a K+1 classifier](book/03-semi-supervised.md#262-semi-supervised-gan-with-a-k1-classifier)
- [2.6.3 Ladder Networks](book/03-semi-supervised.md#263-ladder-networks)

**[Evidence and reproduction boundaries](book/03-semi-supervised.md#evidence-and-reproduction-boundaries)**


[Coverage and continuation manifest](book/03-semi-supervised.md#coverage-and-continuation-manifest)

## 3. Unsupervised Learning Algorithms: Classical Methods

[Read this chapter](book/04-unsupervised-classical.md)


**[3.1 Centroid, prototype, mixture, and hierarchical clustering](book/04-unsupervised-classical.md#31-centroid-prototype-mixture-and-hierarchical-clustering)**

- [3.1.1 k-means](book/04-unsupervised-classical.md#311-k-means)
- [3.1.2 Mini-batch k-means](book/04-unsupervised-classical.md#312-mini-batch-k-means)
- [3.1.3 k-medoids and PAM](book/04-unsupervised-classical.md#313-k-medoids-and-pam)
- [3.1.4 Gaussian mixtures with expectation-maximization](book/04-unsupervised-classical.md#314-gaussian-mixtures-with-expectation-maximization)
- [3.1.5 Agglomerative hierarchical clustering](book/04-unsupervised-classical.md#315-agglomerative-hierarchical-clustering)
- [3.1.6 BIRCH](book/04-unsupervised-classical.md#316-birch)

**[3.2 Density and connectivity clustering](book/04-unsupervised-classical.md#32-density-and-connectivity-clustering)**

- [3.2.1 DBSCAN](book/04-unsupervised-classical.md#321-dbscan)
- [3.2.2 OPTICS](book/04-unsupervised-classical.md#322-optics)
- [3.2.3 HDBSCAN](book/04-unsupervised-classical.md#323-hdbscan)
- [3.2.4 Spectral clustering](book/04-unsupervised-classical.md#324-spectral-clustering)
- [3.2.5 Mean shift](book/04-unsupervised-classical.md#325-mean-shift)
- [3.2.6 Affinity propagation](book/04-unsupervised-classical.md#326-affinity-propagation)

**[3.3 Representation and dimensionality reduction](book/04-unsupervised-classical.md#33-representation-and-dimensionality-reduction)**

- [3.3.1 Principal component analysis](book/04-unsupervised-classical.md#331-principal-component-analysis)
- [3.3.2 Kernel PCA](book/04-unsupervised-classical.md#332-kernel-pca)
- [3.3.3 Independent component analysis](book/04-unsupervised-classical.md#333-independent-component-analysis)
- [3.3.4 Non-negative matrix factorization](book/04-unsupervised-classical.md#334-non-negative-matrix-factorization)
- [3.3.5 t-SNE](book/04-unsupervised-classical.md#335-t-sne)
- [3.3.6 UMAP](book/04-unsupervised-classical.md#336-umap)
- [3.3.7 Isomap](book/04-unsupervised-classical.md#337-isomap)
- [3.3.8 Locally linear embedding](book/04-unsupervised-classical.md#338-locally-linear-embedding)
- [3.3.9 Self-organizing maps](book/04-unsupervised-classical.md#339-self-organizing-maps)

**[3.4 Pattern and topic discovery](book/04-unsupervised-classical.md#34-pattern-and-topic-discovery)**

- [3.4.1 Apriori](book/04-unsupervised-classical.md#341-apriori)
- [3.4.2 FP-growth](book/04-unsupervised-classical.md#342-fp-growth)
- [3.4.3 Eclat](book/04-unsupervised-classical.md#343-eclat)
- [3.4.4 Latent Dirichlet allocation](book/04-unsupervised-classical.md#344-latent-dirichlet-allocation)

**[3.5 Density and anomaly modeling](book/04-unsupervised-classical.md#35-density-and-anomaly-modeling)**

- [3.5.1 Kernel density estimation](book/04-unsupervised-classical.md#351-kernel-density-estimation)
- [3.5.2 Isolation Forest](book/04-unsupervised-classical.md#352-isolation-forest)
- [3.5.3 Local outlier factor](book/04-unsupervised-classical.md#353-local-outlier-factor)
- [3.5.4 One-class SVM](book/04-unsupervised-classical.md#354-one-class-svm)

[Coverage and continuation manifest](book/04-unsupervised-classical.md#coverage-and-continuation-manifest)

## 3. Unsupervised Learning Algorithms: Neural Generation and Representation Pretraining

[Read this chapter](book/05-unsupervised-neural.md)


**[3.6 Reconstruction and latent-variable models](book/05-unsupervised-neural.md#36-reconstruction-and-latent-variable-models)**

- [3.6.1 Autoencoders](book/05-unsupervised-neural.md#361-autoencoders)
- [3.6.2 Denoising autoencoders](book/05-unsupervised-neural.md#362-denoising-autoencoders)
- [3.6.3 Variational autoencoders](book/05-unsupervised-neural.md#363-variational-autoencoders)
- [3.6.4 Beta-VAE](book/05-unsupervised-neural.md#364-beta-vae)
- [3.6.5 VQ-VAE](book/05-unsupervised-neural.md#365-vq-vae)

**[3.7 Adversarial generation](book/05-unsupervised-neural.md#37-adversarial-generation)**

- [3.7.1 Generative adversarial networks](book/05-unsupervised-neural.md#371-generative-adversarial-networks)
- [3.7.2 DCGAN](book/05-unsupervised-neural.md#372-dcgan)
- [3.7.3 StyleGAN family](book/05-unsupervised-neural.md#373-stylegan-family)
- [3.7.4 CycleGAN](book/05-unsupervised-neural.md#374-cyclegan)

**[3.8 Diffusion and image-generation families](book/05-unsupervised-neural.md#38-diffusion-and-image-generation-families)**

- [3.8.1 Denoising diffusion probabilistic models](book/05-unsupervised-neural.md#381-denoising-diffusion-probabilistic-models)
- [3.8.2 Score-based SDE models](book/05-unsupervised-neural.md#382-score-based-sde-models)
- [3.8.3 Latent diffusion and Stable Diffusion](book/05-unsupervised-neural.md#383-latent-diffusion-and-stable-diffusion)
- [3.8.4 Diffusion Transformer](book/05-unsupervised-neural.md#384-diffusion-transformer)
- [3.8.5 DALL-E 1](book/05-unsupervised-neural.md#385-dall-e-1)
- [3.8.6 DALL-E 2](book/05-unsupervised-neural.md#386-dall-e-2)
- [3.8.7 DALL-E 3](book/05-unsupervised-neural.md#387-dall-e-3)

**[3.9 Flows and autoregression](book/05-unsupervised-neural.md#39-flows-and-autoregression)**

- [3.9.1 RealNVP](book/05-unsupervised-neural.md#391-realnvp)
- [3.9.2 Glow](book/05-unsupervised-neural.md#392-glow)
- [3.9.3 PixelCNN](book/05-unsupervised-neural.md#393-pixelcnn)
- [3.9.4 WaveNet](book/05-unsupervised-neural.md#394-wavenet)

**[3.10 Representation pretraining](book/05-unsupervised-neural.md#310-representation-pretraining)**

- [3.10.1 SimCLR](book/05-unsupervised-neural.md#3101-simclr)
- [3.10.2 MoCo](book/05-unsupervised-neural.md#3102-moco)
- [3.10.3 BYOL](book/05-unsupervised-neural.md#3103-byol)
- [3.10.4 DINO](book/05-unsupervised-neural.md#3104-dino)
- [3.10.5 Masked autoencoders](book/05-unsupervised-neural.md#3105-masked-autoencoders)
- [3.10.6 CLIP](book/05-unsupervised-neural.md#3106-clip)
- [3.10.7 Word2Vec](book/05-unsupervised-neural.md#3107-word2vec)
- [3.10.8 GloVe](book/05-unsupervised-neural.md#3108-glove)

**[Evidence reading notes](book/05-unsupervised-neural.md#evidence-reading-notes)**


[Coverage and continuation manifest](book/05-unsupervised-neural.md#coverage-and-continuation-manifest)

## 3. Unsupervised Learning Algorithms: Language and Multimodal Foundation Models

[Read this chapter](book/06-foundation-models.md)


**[3.11 Encoder and Encoder-Decoder Pretraining](book/06-foundation-models.md#311-encoder-and-encoder-decoder-pretraining)**

- [3.11.1 BERT](book/06-foundation-models.md#3111-bert)
- [3.11.2 RoBERTa](book/06-foundation-models.md#3112-roberta)
- [3.11.3 T5](book/06-foundation-models.md#3113-t5)
- [3.11.4 BART](book/06-foundation-models.md#3114-bart)

**[3.12 Public Decoder Research and Base-Model Families](book/06-foundation-models.md#312-public-decoder-research-and-base-model-families)**

- [3.12.1 GPT Family](book/06-foundation-models.md#3121-gpt-family)
- [3.12.2 LLaMA / Llama](book/06-foundation-models.md#3122-llama--llama)
- [3.12.3 Mistral Dense Models](book/06-foundation-models.md#3123-mistral-dense-models)
- [3.12.4 Qwen Dense Models](book/06-foundation-models.md#3124-qwen-dense-models)
- [3.12.5 DeepSeek LLM Dense Base](book/06-foundation-models.md#3125-deepseek-llm-dense-base)
- [3.12.6 BLOOM](book/06-foundation-models.md#3126-bloom)

**[3.13 Vendor Multimodal and Assistant Families](book/06-foundation-models.md#313-vendor-multimodal-and-assistant-families)**

- [3.13.1 Anthropic Claude](book/06-foundation-models.md#3131-anthropic-claude)
- [3.13.2 Google / DeepMind Gemini](book/06-foundation-models.md#3132-google--deepmind-gemini)
- [3.13.3 Cohere Command R](book/06-foundation-models.md#3133-cohere-command-r)
- [3.13.4 Baidu ERNIE](book/06-foundation-models.md#3134-baidu-ernie)
- [3.13.5 Amazon Titan](book/06-foundation-models.md#3135-amazon-titan)
- [3.13.6 Amazon Nova](book/06-foundation-models.md#3136-amazon-nova)

**[3.14 Open Code, Efficient, and Enterprise Families](book/06-foundation-models.md#314-open-code-efficient-and-enterprise-families)**

- [3.14.1 Microsoft Phi](book/06-foundation-models.md#3141-microsoft-phi)
- [3.14.2 NVIDIA Nemotron](book/06-foundation-models.md#3142-nvidia-nemotron)
- [3.14.3 IBM Granite](book/06-foundation-models.md#3143-ibm-granite)
- [3.14.4 StarCoder](book/06-foundation-models.md#3144-starcoder)
- [3.14.5 S4](book/06-foundation-models.md#3145-s4)
- [3.14.6 Mamba](book/06-foundation-models.md#3146-mamba)

[Coverage and continuation manifest](book/06-foundation-models.md#coverage-and-continuation-manifest)

## 3. Unsupervised Learning Algorithms: Sparse MoE Model Catalog

[Read this chapter](book/07-moe-models.md)


**[3.15 Self-supervised sparse expert language models](book/07-moe-models.md#315-self-supervised-sparse-expert-language-models)**

- [3.15.1 Switch Transformer](book/07-moe-models.md#3151-switch-transformer)
- [3.15.2 GLaM](book/07-moe-models.md#3152-glam)
- [3.15.3 Mixtral 8x7B](book/07-moe-models.md#3153-mixtral-8x7b)
- [3.15.4 Mixtral 8x22B](book/07-moe-models.md#3154-mixtral-8x22b)
- [3.15.5 DeepSeekMoE 16B](book/07-moe-models.md#3155-deepseekmoe-16b)
- [3.15.6 DeepSeek-V2](book/07-moe-models.md#3156-deepseek-v2)
- [3.15.7 DeepSeek-V3](book/07-moe-models.md#3157-deepseek-v3)
- [3.15.8 Grok-1](book/07-moe-models.md#3158-grok-1)
- [3.15.9 DBRX](book/07-moe-models.md#3159-dbrx)
- [3.15.10 Snowflake Arctic](book/07-moe-models.md#31510-snowflake-arctic)
- [3.15.11 OLMoE](book/07-moe-models.md#31511-olmoe)

[Coverage and continuation manifest](book/07-moe-models.md#coverage-and-continuation-manifest)

## 4. Mixture of Experts: A Dedicated Deep Dive

[Read this chapter](book/08-moe-deep-dive.md)

- [4.1 From adaptive local experts to sparse language models](book/08-moe-deep-dive.md#41-from-adaptive-local-experts-to-sparse-language-models)
- [4.2 Gating and routing mechanisms](book/08-moe-deep-dive.md#42-gating-and-routing-mechanisms)
- [4.3 Load balancing, capacity, overflow, and collapse](book/08-moe-deep-dive.md#43-load-balancing-capacity-overflow-and-collapse)
- [4.4 Where experts live in a Transformer](book/08-moe-deep-dive.md#44-where-experts-live-in-a-transformer)
- [4.5 Parameter ledger: total versus active](book/08-moe-deep-dive.md#45-parameter-ledger-total-versus-active)
- [4.6 Training challenges and distributed execution](book/08-moe-deep-dive.md#46-training-challenges-and-distributed-execution)
- [4.7 Inference: memory, arithmetic, and useful answers](book/08-moe-deep-dive.md#47-inference-memory-arithmetic-and-useful-answers)
- [4.8 A production-oriented case: Databricks DBRX](book/08-moe-deep-dive.md#48-a-production-oriented-case-databricks-dbrx)
- [4.9 A decision framework and remaining questions](book/08-moe-deep-dive.md#49-a-decision-framework-and-remaining-questions)

[Coverage and continuation manifest](book/08-moe-deep-dive.md#coverage-and-continuation-manifest)

## 5. Cross-Cutting Comparisons and Model Selection

[Read this chapter](book/09-comparative-guide.md)

- [5.1 A deliberately conditional ranking](book/09-comparative-guide.md#51-a-deliberately-conditional-ranking)
- [5.2 Choosing a first serious baseline](book/09-comparative-guide.md#52-choosing-a-first-serious-baseline)
- [5.3 What an apples-to-apples comparison requires](book/09-comparative-guide.md#53-what-an-apples-to-apples-comparison-requires)
- [5.4 Hardware and parallelism are part of the algorithmic choice](book/09-comparative-guide.md#54-hardware-and-parallelism-are-part-of-the-algorithmic-choice)
- [5.5 Vendor, community, and laboratory navigation](book/09-comparative-guide.md#55-vendor-community-and-laboratory-navigation)
- [5.6 Evidence gaps that remain useful to name](book/09-comparative-guide.md#56-evidence-gaps-that-remain-useful-to-name)

[Coverage and continuation manifest](book/09-comparative-guide.md#coverage-and-continuation-manifest)

## 6. Glossary

[Read this chapter](book/10-glossary.md)

- [6.1 Learning, objectives, and statistical reasoning](book/10-glossary.md#61-learning-objectives-and-statistical-reasoning)
- [6.2 Optimization and neural computation](book/10-glossary.md#62-optimization-and-neural-computation)
- [6.3 Representation, sequence, and generative modeling](book/10-glossary.md#63-representation-sequence-and-generative-modeling)
- [6.4 Adaptation, alignment, and evaluation](book/10-glossary.md#64-adaptation-alignment-and-evaluation)
- [6.5 Clustering, patterns, and sparse systems](book/10-glossary.md#65-clustering-patterns-and-sparse-systems)

[Coverage and continuation manifest](book/10-glossary.md#coverage-and-continuation-manifest)
