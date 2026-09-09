# 5. Cross-Cutting Comparisons and Model Selection

There is no best model for every job. A simple model may be easier to explain
and cheap to run. A larger model may handle harder patterns but need more
data, memory, and care.

It also matters where you start counting costs. Adapting a ready-made model
can be cheap because someone else paid for its first training. An MoE model
may use only some of its weights for each word, yet still need memory for
all of them. Here, **weights** are numbers the model has learned.

## 5.1 A deliberately conditional ranking

The table below is a rough guide, not a set of measured test scores.
The ratings assume the method suits the task and is used competently.
They help you ask better questions; they do not pick a winner for you.

| Rating | What it means | Which direction is better? |
|---|---|---|
| Interpretability | How easy it is to inspect and explain the model | 5 is easiest; 1 is hardest |
| Data efficiency | How far a small set of labeled examples can take it on this task | 5 usually needs fewer task labels |
| Compute cost | How much computer work it needs | 1 is relatively cheap; 5 is very costly |
| Production maturity | How well established its real-world use and support are | 5 is widely established; 1 is mainly experimental |

The data rating counts **labels for the current task**, not all data ever used
to train a ready-made model. A range means members of a family differ.
Do not average the four ratings: they describe different things.

| Algorithm family | Interpretability (1-5, higher better) | Labeled-data efficiency (1-5, higher better) | Compute cost (1-5, lower cheaper) | Typical production maturity (1-5) | Main qualification |
|---|---:|---:|---:|---:|---|
| Linear/logistic models and small GAMs | 4-5 | 4-5 | 1-2 | 5 | Input units and related inputs affect the learned numbers; links do not prove causes |
| Small decision trees | 4-5 | 3-4 | 1-2 | 5 | A large tree can become hard to read and change sharply with new data |
| Random forests and boosted trees | 2-3 | 4-5 | 2-3 | 5 | Often strong on tables; extra tools are usually needed to explain answers |
| Naive Bayes and small probability models | 3-5 | 4 | 1-2 | 5 | Simple assumptions about the data can help or hurt a lot |
| Kernel SVMs and Gaussian processes | 2-3 | 4-5 | 2-4 | 4-5 | Keeping and processing all pairwise relationships can be costly |
| Nearest-neighbor methods | 3-4 | 2-4 | 1 training / 2-4 inference | 5 | Their success depends on a useful meaning of "nearby" |
| HMMs, CRFs, and related sequence models | 3-4 | 3-4 | 2-3 | 4-5 | Their states and links help describe a sequence, but may not have one clear meaning |
| General supervised neural networks | 1-2 | 1-3 | 2-4 | 5 | Reusing a trained model or a suitable design can reduce label needs |
| CNNs and reused vision networks | 1-2 | 3-4 with transfer | 2-4 | 5 | A cheap fine-tune does not include the original training bill |
| Graph neural networks | 1-2 | 2-4 | 2-4 | 3-4 | Missing or changing links can matter as much as the network choice |
| Semi-supervised neural methods | 1-2 | 3-5 when assumptions hold | 3-4 | 3-4 | Wrong guesses or unsuitable unlabeled data can make learning worse |
| Clustering and prototype methods | 2-4 | Not comparable | 1-3 | 5 | They need no class labels to fit, but someone must check whether the groups are useful |
| PCA, NMF, and related compact summaries | 3-4 | Not comparable | 1-3 | 5 | A useful summary is not always easy to name or interpret |
| Nonlinear data plots | 1-2 | Not comparable | 2-4 | 4 for exploration | Separate islands in a plot do not prove separate real-world classes |
| Unsupervised anomaly detectors | 2-3 | 3-5 for fitting | 1-3 | 4-5 | Known unusual cases still help choose a sensible alert threshold |
| Autoencoders and VAEs | 1-2 | 3-5 for learning features | 2-4 | 4 | Rebuilding an input well does not prove the model understands its useful parts |
| GANs and diffusion generators | 1 | 3-5 for learning from raw data | 3-5 | 4-5 | Data choice, conditions, generation steps, and safety controls affect results |
| Pretrained language/multimodal models | 1 | 4-5 for adaptation | 5 pretraining / 2-5 serving | 5 for general APIs; task-specific reliability varies | Few new labels can hide a huge earlier training effort |
| Sparse MoE language models | 1 | 4-5 for adaptation | 5 total training infrastructure / variable serving | 4-5 | Using fewer weights at once does not mean storing fewer weights |
| Spiking neural networks | 1-3 | 1-3 | 1-4, hardware-dependent | 2-3 | Energy savings depend on the device and how often neurons fire |

Why does clustering get **not comparable** for data efficiency? It answers
a different question. Finding groups without labels is not the same task
as predicting a known class. Giving it a top score would hide that difference.

"Mature" also does not mean safe for every use. It means people have experience
building and supporting systems with that family.

## 5.2 Choosing a first serious baseline

A **baseline** is a sensible starting method. Before adding a harder method,
check whether it gives a useful improvement over that starting point.
The candidates below are explained in the main chapters.

| Problem and constraints | Reasonable starting candidates | What to compare before adding complexity |
|---|---|---|
| Predict a class from a small or medium table | Logistic regression, boosted trees, random forest | Missing values, rare classes, trustworthy probabilities, and fair test splits |
| Predict a number from a small dataset | Regularized regression, GAM, Gaussian process | Curved patterns, uncertainty, and predictions beyond the observed range |
| Recognize images with few labels | A pretrained CNN or ViT; compare changing only the output layer with changing all layers | Whether new images resemble the old training images; label quality and training cost |
| Learn from a few labels and many similar unlabeled examples | A supervised baseline, then consistency or pseudo-labeling | Whether the unlabeled examples really belong to the target problem |
| Assign linked labels in a sequence | CRF or a pretrained text encoder with a task-specific output | Neighboring-label mistakes, needed context, and speed |
| Find groups in customer or scientific data | Scaled k-means, mixtures, or suitable density/hierarchical methods | Whether the groups are stable, meaningful, and useful |
| Explore data with many measured features | PCA before t-SNE or UMAP | Whether the plot changes with settings and whether nearby points remain nearby |
| Detect rare events with few known examples | Simple rules and any possible labeled baseline, then anomaly detection | False alarms, review capacity, and changes in normal behavior |
| Generate text or code | A suitable small pretrained or hosted model | Correct answers, data rights, speed, cost, and human review |
| Serve many large-model requests | Dense and MoE candidates that meet the same quality goal | Full memory use, batching, slowest requests, and cost per useful answer |

This is a route into the catalog, not a second set of full entries.
The chapters provide the steps, examples, and sources for each method.

## 5.3 What an apples-to-apples comparison requires

Compare the **whole process**, not just the last prediction step.
Use the same training and test examples. Learn data-cleaning and scaling
steps from the training set only. Count the cost of tuning, extra data,
search, and any other tools the model uses.

For class prediction, look at which mistakes occur. High overall accuracy
can hide failure on a rare but important class. Check whether predicted
chances match real outcomes, not just whether the highest-scored item wins.

For number prediction, inspect errors of different sizes and kinds.
For event-time prediction, remember that some events have not happened
by the end of a study. Treating them as events that will never happen
gives a misleading comparison.

For grouping and data summaries, check meaning as well as the fitted score.
A model can rebuild noise very well. Groups can look neat without helping
any real decision. In t-SNE or UMAP plots, distances between separate islands
are not reliable rulers.

For generators, hold the prompt, search tools, number of attempts, and
answer-scoring rules fixed. Read failures as well as success scores.
A longer answer may cost more without helping the user finish the task.

## 5.4 Hardware and parallelism are part of the algorithmic choice

**Parallelism** means sharing work across processors. A model may need this
to run faster or to fit in memory. The methods below split different parts
of the work, so their costs differ.

| Technique | What is distributed or changed | What it helps | Cost or limitation |
|---|---|---|---|
| Data parallelism | Model copies process different batches of examples | More examples processed per second | Copies must share updates; each still stores model weights unless also sharded |
| Optimizer/parameter sharding | Devices divide weights or saved training state | Lower memory use per device | Pieces must be gathered or exchanged when needed |
| Tensor parallelism | Devices share one large array calculation | Layers too large for one device | Frequent messages can make slow links a bottleneck |
| Pipeline parallelism | Different devices own successive groups of layers | Deep models that do not fit on one device | Devices can sit idle while waiting for earlier stages |
| Expert parallelism | Devices store different small expert networks | MoE models with many experts | Inputs and results must travel to the right devices |
| Quantization | Numbers use fewer bits | Less memory and data movement | May change answers; needs suitable software and hardware |
| Activation checkpointing | Recalculate some intermediate results instead of saving them | Lower training memory use | More calculations |
| KV caching | Keep attention information about earlier tokens | Faster generation of later tokens | Longer contexts and more requests need more memory |
| Speculative decoding | Propose tokens, then have the target model check them | Faster generation when enough proposals are accepted | Checking rejected proposals can waste work |

Combining these methods is not automatically better. A model that is fast
on one machine can slow down when expert data must cross a network.
Low-bit weights still need working memory. Training also saves updates
and intermediate results that are not needed just to produce an answer.

The [MoE deep dive](08-moe-deep-dive.md) gives worked examples of these
tradeoffs and explains the limits of vendor speed claims.

## 5.5 Vendor, community, and laboratory navigation

This index helps you find work from different companies and research groups.
It does not mean they invented every method in their row or use it in every
product. Original papers and release notes give the detailed history.

| Organization or community | Examples discussed | Main chapter |
|---|---|---|
| Google / Google DeepMind | Inception, EfficientNet, Transformer, T5, Gemini, GShard, Switch, GLaM | [Neural supervised](02-supervised-neural.md), [foundation](06-foundation-models.md), [MoE](07-moe-models.md) |
| OpenAI | GPT, CLIP, DALL-E | [Foundation](06-foundation-models.md), [neural unsupervised](05-unsupervised-neural.md) |
| Meta AI and associated research | ConvNeXt, Llama, DINO, MAE | [Neural supervised](02-supervised-neural.md), [representation](05-unsupervised-neural.md), [foundation](06-foundation-models.md) |
| Microsoft and Microsoft Research | LightGBM, ResNet, Swin, Phi; DeepSpeed tools | [Classical supervised](01-supervised-classical.md), [neural supervised](02-supervised-neural.md), [foundation](06-foundation-models.md) |
| Anthropic | Claude and published work on training model behavior | [Foundation](06-foundation-models.md) |
| Amazon | Titan and Nova | [Foundation](06-foundation-models.md) |
| NVIDIA | StyleGAN, Nemotron, TensorRT-LLM serving | [Neural unsupervised](05-unsupervised-neural.md), [foundation](06-foundation-models.md), [MoE](08-moe-deep-dive.md) |
| Baidu | ERNIE | [Foundation](06-foundation-models.md) |
| Alibaba | Qwen | [Foundation](06-foundation-models.md) |
| Hugging Face, BigScience, and BigCode communities | Transformers software, BLOOM, StarCoder | [Foundation](06-foundation-models.md) |
| Mistral AI | Mistral dense models and Mixtral | [Foundation](06-foundation-models.md), [MoE catalog](07-moe-models.md) |
| Cohere | Command R | [Foundation](06-foundation-models.md) |
| xAI | Released Grok-1 base model | [MoE catalog](07-moe-models.md) |
| Stability AI and collaborating research groups | Stable Diffusion releases based on latent diffusion | [Neural unsupervised](05-unsupervised-neural.md) |
| IBM | Granite | [Foundation](06-foundation-models.md) |
| DeepSeek | Dense DeepSeek LLM, DeepSeekMoE, V2, V3 | [Foundation](06-foundation-models.md), [MoE catalog](07-moe-models.md) |
| Databricks / Mosaic research | DBRX and its published training/serving account | [MoE catalog](07-moe-models.md), [deep dive](08-moe-deep-dive.md) |
| Snowflake | Arctic and business-focused MoE design | [MoE catalog](07-moe-models.md) |
| Allen Institute for AI and collaborators | OLMoE and released data, code, and logs | [MoE catalog](07-moe-models.md) |
| Stanford research groups | GraphSAGE, GloVe, state-space work, MegaBlocks collaborations | [Neural supervised](02-supervised-neural.md), [representation](05-unsupervised-neural.md), [foundation](06-foundation-models.md), [MoE](08-moe-deep-dive.md) |
| Berkeley and collaborating academic groups | Diffusion research, image representations, and computing methods | [Neural unsupervised](05-unsupervised-neural.md) |
| MIT and University of Toronto researchers | The original adaptive mixture of local experts | [Supervised mixtures](02-supervised-neural.md#1111-adaptive-mixture-of-local-experts) |
| Carnegie Mellon and collaborators | Co-training and models for linked predictions | [Co-training](03-semi-supervised.md#212-co-training), [conditional random fields](01-supervised-classical.md#144-conditional-random-fields) |

ResNet began at Microsoft Research. Its researchers later working elsewhere
does not change that origin. BLOOM and StarCoder are shared community efforts,
not just products of the organization hosting their files.

## 5.6 Evidence gaps that remain useful to name

Public reports often focus on successful tests and major releases.
They less often share failed projects, full bills, or long-term repair costs.
That is why this book has more research tests than independently audited
business case studies.

We name that gap rather than fill it with guesses. Knowing how a method
works does not prove that it saved a company money. Missing business data
also do not mean the method failed.

## Coverage and continuation manifest

This chapter gives a rough comparison across families, starting choices,
fair-testing advice, a hardware guide, and a company/research index.
Continue to the [glossary](10-glossary.md) or the
[complete contents](../CONTENTS.md).

A future edition could compare methods on fixed data and hardware.
This chapter does not claim those new experiments have already been run.
