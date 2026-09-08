# 5. Cross-Cutting Comparisons and Model Selection

Catalogs are useful for finding candidates, but not for declaring a winner
without a task. An interpretable model with the wrong target is not a good
decision system. A powerful pretrained model can be an economical adaptation
choice while still depending on an enormous upstream training investment.
Sparse activation can reduce arithmetic without reducing the memory required
to host the complete model.

## 5.1 A deliberately conditional ranking

The following table supplies the requested ranking across algorithm families.
These are **editorial ordinal judgments**, not measured benchmark scores or
universal orderings. The reference setting is ordinary production use with
competent implementation, appropriate features, and a task that fits the
family's assumptions.

**Interpretability:** 5 means most directly inspectable, 1 least.
**Data efficiency:** 5 means relatively economical in *task-specific labeled
examples*, not in total historical data consumed.
**Compute cost:** 1 means relatively low, 5 very high; training and inference
can differ.
**Production maturity:** 5 means widely established operational practice,
1 mainly research or specialized experimental use.

Ranges reflect meaningful variation within a family. Do not average the four
columns into a purported objective score: their units and desirability differ.

| Algorithm family | Interpretability (1-5, higher better) | Labeled-data efficiency (1-5, higher better) | Compute cost (1-5, lower cheaper) | Typical production maturity (1-5) | Main qualification |
|---|---:|---:|---:|---:|---|
| Linear/logistic models and small GAMs | 4-5 | 4-5 | 1-2 | 5 | Coefficients need scaling, collinearity analysis, and a noncausal interpretation |
| Small decision trees | 4-5 | 3-4 | 1-2 | 5 | Deep trees cease to be readable and can be unstable |
| Random forests and boosted trees | 2-3 | 4-5 | 2-3 | 5 | Strong tabular defaults; explanations are usually post hoc |
| Naive Bayes and compact probabilistic models | 3-5 | 4 | 1-2 | 5 | Independence and distributional assumptions can dominate performance |
| Kernel SVMs and Gaussian processes | 2-3 | 4-5 | 2-4 | 4-5 | Exact kernel storage and factorization limit large-data regimes |
| Nearest-neighbor methods | 3-4 | 2-4 | 1 training / 2-4 inference | 5 | Metric quality and high-dimensional search determine usefulness |
| HMMs, CRFs, and structured statistical models | 3-4 | 3-4 | 2-3 | 4-5 | State structure can aid interpretation; latent states remain ambiguous |
| Generic supervised neural networks | 1-2 | 1-3 | 2-4 | 5 | Data efficiency improves with inductive bias or transfer learning |
| CNNs and transferred vision backbones | 1-2 | 3-4 with transfer | 2-4 | 5 | Pretraining cost is not included in adaptation-only comparisons |
| Graph neural networks | 1-2 | 2-4 | 2-4 | 3-4 | Graph availability, sampling, leakage, and changing topology matter |
| Semi-supervised deep recipes | 1-2 | 3-5 when assumptions hold | 3-4 | 3-4 | Unlabeled distribution shift or confirmation bias can erase gains |
| Clustering and prototype methods | 2-4 | Not comparable | 1-3 | 5 | No training labels required; semantic validation still requires evidence |
| PCA, NMF, and related low-rank models | 3-4 | Not comparable | 1-3 | 5 | Component interpretability varies with identifiability and constraints |
| Nonlinear visualization embeddings | 1-2 | Not comparable | 2-4 | 4 for exploration | Visual separation is not a calibrated class or distance claim |
| Unsupervised anomaly detectors | 2-3 | 3-5 for fitting | 1-3 | 4-5 | Rare labeled incidents are still valuable for threshold validation |
| Autoencoders and VAEs | 1-2 | 3-5 for representation fitting | 2-4 | 4 | Reconstruction quality is not equivalent to semantic usefulness |
| GANs and diffusion generators | 1 | 3-5 for raw-data fitting | 3-5 | 4-5 | Conditioning, data curation, sampling steps, and safety controls matter |
| Pretrained language/multimodal models | 1 | 4-5 for adaptation | 5 pretraining / 2-5 serving | 5 for general APIs; task-specific reliability varies | Small downstream label budgets hide large upstream data and compute |
| Sparse MoE language models | 1 | 4-5 for adaptation | 5 total training infrastructure / variable serving | 4-5 | Less active arithmetic does not imply a small memory footprint |
| Spiking neural networks | 1-3 | 1-3 | 1-4, hardware-dependent | 2-3 | Energy and latency advantages depend on event sparsity and implementation |

Clustering and dimensionality reduction receive **not comparable** rather than
a misleading top score for requiring no labels. Their output usually addresses
a different question from supervised prediction. Similarly, "production
maturity" means that deployment practices exist; it is not certification that
a particular model is safe or reliable for every use.

## 5.2 Choosing a first serious baseline

| Problem and constraints | Reasonable starting candidates | What to compare before adding complexity |
|---|---|---|
| Small or medium tabular classification | Logistic regression, boosted trees, random forest | Calibration, class imbalance, missingness, leakage, and group/time splits |
| Smooth numerical prediction with few observations | Regularized regression, GAM, Gaussian process | Nonlinearity, uncertainty calibration, extrapolation behavior |
| Image recognition with limited annotations | A pretrained CNN or ViT, then linear/head-only versus full fine-tuning | Domain shift, annotation consistency, augmentation, compute budget |
| Few labels and abundant same-domain unlabeled data | Supervised baseline, then consistency or pseudo-labeling | Whether unlabeled data truly follow the target distribution |
| Structured sequence labeling | CRF or a suitable pretrained encoder with a task head | Label dependencies, latency, context needs, boundary errors |
| Unlabeled customer or scientific grouping | Scaled k-means, mixtures, or a justified density/hierarchical method | Cluster stability, feature meaning, outliers, actionable interpretation |
| High-dimensional exploration | PCA before t-SNE or UMAP | Neighborhood preservation, seed sensitivity, confounding and overinterpretation |
| Rare-event detection with incomplete labels | Simple rules plus a labeled baseline where possible, then anomaly detection | Alert budget, false positives, changing normal behavior, delayed outcomes |
| Text or code generation | A small appropriate pretrained model or hosted model | Task success, factuality, licensing, latency, context cost, human review |
| High-throughput large-model serving | Dense and MoE candidates under the same quality requirement | Resident memory, batching, tail latency, communication, actual cost per useful answer |

This table is a search strategy, not an extra algorithm catalog. The linked
chapters contain the individual mechanisms, worked examples, and sources.

## 5.3 What an apples-to-apples comparison requires

Compare **the full pipeline**, not only the estimator. Fit feature transforms
using training data only. Keep splits and label budgets fixed. Include search,
pretraining access, augmentation, ensembling, decoding, retrieval, and hardware
in the resource accounting.

For classification, report a confusion matrix or class-sensitive metrics when
accuracy hides rare failures. For probabilistic output, evaluate calibration
and proper scoring rules, not just ranking. For regression, inspect residual
patterns and the cost of large errors. For survival models, account for
censoring rather than treating unobserved events as confirmed negatives.

For unsupervised models, combine internal objectives with stability,
domain-grounded inspection, and external criteria where available. Low
reconstruction loss can reflect copying nuisance variation. High silhouette
scores can reflect a convenient geometry rather than useful classes.
Neighborhood-preserving visualizations should not be used as distance rulers
across disconnected islands.

For generative models, hold prompt formats, tools, retrieval, sampling,
response length, and evaluation rules constant. Inspect representative
failures, not only aggregate scores. A model that produces a longer answer can
consume more tokens and evaluator attention without improving task completion.

## 5.4 Hardware and parallelism are part of the algorithmic choice

| Technique | What is distributed or changed | What it helps | Cost or limitation |
|---|---|---|---|
| Data parallelism | Replicas process different mini-batches | Training throughput | Gradient synchronization; does not by itself reduce per-replica weight memory |
| Optimizer/parameter sharding | States, gradients, and optionally weights are partitioned | Training memory | Collectives and parameter materialization |
| Tensor parallelism | A matrix operation is split across devices | Very large layers and compute | Frequent communication, especially across slow links |
| Pipeline parallelism | Layer groups live on different devices | Deep models that do not fit one device | Bubbles, micro-batch scheduling, activation transfers |
| Expert parallelism | Different expert weights live on different devices | Sparse models with many experts | Token dispatch/combine, imbalance, and all-to-all traffic |
| Quantization | Weights and/or activations use lower precision | Memory bandwidth and capacity | Kernel dependence, accuracy sensitivity, scale metadata |
| Activation checkpointing | Some activations are recomputed in backward | Training memory | Additional arithmetic |
| KV caching | Previous attention keys/values are retained | Autoregressive decoding | Context- and batch-dependent memory growth |
| Speculative decoding | A proposal mechanism is checked by the target model | Decoding latency in favorable regimes | Acceptance rate and verification overhead; not universally faster |

Distributed techniques compose, but not freely. A fast single-node benchmark
can become communication-bound when its experts cross a network boundary.
Quantized weights still require activation buffers and a KV cache. Optimizer
states and gradients make training memory substantially different from
weights-only inference memory.

For derivations, concrete routing examples, and vendor-specific boundaries,
see the [MoE deep dive](08-moe-deep-dive.md).

## 5.5 Vendor, community, and laboratory navigation

The following is an **index of represented contributions**, not a claim that
any organization invented a whole field or deploys every listed method.
Affiliations can change; source papers and release records remain the
authority.

| Organization or community | Examples discussed | Main chapter |
|---|---|---|
| Google / Google DeepMind | Inception, EfficientNet, Transformer, T5, Gemini, GShard, Switch, GLaM | [Neural supervised](02-supervised-neural.md), [foundation](06-foundation-models.md), [MoE](07-moe-models.md) |
| OpenAI | GPT, CLIP, DALL-E | [Foundation](06-foundation-models.md), [neural unsupervised](05-unsupervised-neural.md) |
| Meta AI and associated research | ConvNeXt, Llama, DINO, MAE | [Neural supervised](02-supervised-neural.md), [representation](05-unsupervised-neural.md), [foundation](06-foundation-models.md) |
| Microsoft and Microsoft Research | LightGBM, ResNet, Swin, Phi; DeepSpeed systems ecosystem | [Classical supervised](01-supervised-classical.md), [neural supervised](02-supervised-neural.md), [foundation](06-foundation-models.md) |
| Anthropic | Claude and publicly described alignment work | [Foundation](06-foundation-models.md) |
| Amazon | Titan and Nova | [Foundation](06-foundation-models.md) |
| NVIDIA | StyleGAN, Nemotron, TensorRT-LLM serving | [Neural unsupervised](05-unsupervised-neural.md), [foundation](06-foundation-models.md), [MoE](08-moe-deep-dive.md) |
| Baidu | ERNIE | [Foundation](06-foundation-models.md) |
| Alibaba | Qwen | [Foundation](06-foundation-models.md) |
| Hugging Face, BigScience, and BigCode communities | Transformers implementations, BLOOM, StarCoder | [Foundation](06-foundation-models.md) |
| Mistral AI | Mistral dense models and Mixtral | [Foundation](06-foundation-models.md), [MoE catalog](07-moe-models.md) |
| Cohere | Command R | [Foundation](06-foundation-models.md) |
| xAI | Released Grok-1 base model | [MoE catalog](07-moe-models.md) |
| Stability AI and collaborating research groups | Stable Diffusion releases built on latent diffusion | [Neural unsupervised](05-unsupervised-neural.md) |
| IBM | Granite | [Foundation](06-foundation-models.md) |
| DeepSeek | Dense DeepSeek LLM, DeepSeekMoE, V2, V3 | [Foundation](06-foundation-models.md), [MoE catalog](07-moe-models.md) |
| Databricks / Mosaic research | DBRX and associated training/serving evidence | [MoE catalog](07-moe-models.md), [deep dive](08-moe-deep-dive.md) |
| Snowflake | Arctic and enterprise-oriented MoE design | [MoE catalog](07-moe-models.md) |
| Allen Institute for AI and collaborators | OLMoE and open reproducibility artifacts | [MoE catalog](07-moe-models.md) |
| Stanford research groups | GraphSAGE, GloVe, state-space work, MegaBlocks collaborations | [Neural supervised](02-supervised-neural.md), [representation](05-unsupervised-neural.md), [foundation](06-foundation-models.md), [MoE](08-moe-deep-dive.md) |
| Berkeley and collaborating academic groups | Diffusion research, contrastive/generative work, systems contributions | [Neural unsupervised](05-unsupervised-neural.md) |
| MIT and University of Toronto researchers | The original adaptive mixture of local experts | [Supervised mixtures](02-supervised-neural.md#1111-adaptive-mixture-of-local-experts) |
| Carnegie Mellon and collaborators | Co-training and structured probabilistic prediction | [Co-training](03-semi-supervised.md#212-co-training), [conditional random fields](01-supervised-classical.md#144-conditional-random-fields) |

ResNet was introduced at Microsoft Research; later work by its researchers at
other institutions does not transfer its original attribution. Community
projects such as BLOOM and StarCoder are collaborative efforts, not solely
products of the organization hosting their weights.

## 5.6 Evidence gaps that remain useful to name

Public sources disproportionately report successful experiments, popular
datasets, and large-model releases. Failed deployments, full cost ledgers,
counterfactual procurement decisions, and long-term maintenance outcomes are
less often published. As a result, this book contains more research benchmarks
than audited production case studies.

That imbalance is stated rather than concealed. A missing production metric
does not make the mechanism unknowable, and a valid mechanism does not justify
inventing a business outcome. Future additions should improve the evidence,
not merely increase the number of names.

## Coverage and continuation manifest

This chapter supplies the cross-cutting ranking, a model-selection starting
point, evaluation caveats, systems comparisons, and vendor/community
navigation. Continue to the [glossary](10-glossary.md) or return to the
[complete contents](../CONTENTS.md).

A later edition could add standardized, reproducible task-specific comparisons
under fixed hardware and data budgets. This chapter does not claim to have
performed such experiments.
