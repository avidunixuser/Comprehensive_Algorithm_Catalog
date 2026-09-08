# Preface: How to Read an Algorithm Catalog

An algorithm is a way of learning or computing. An architecture is a pattern of
connections and operations. A trained model is the result of applying a learning
procedure to particular data. A product is a system that may contain several
models, retrieval, rules, human review, and application-specific infrastructure.
Confusing these objects is one of the quickest ways to misunderstand machine
learning.

This book treats them as related, but not interchangeable. Random forests are
not a vendor product; a Transformer is not synonymous with a chatbot; a model
with publicly downloadable weights does not necessarily have a reproducible
training dataset. A successful laboratory benchmark is not evidence of a
successful production deployment.

The aim is a working technical reference: enough history to understand an
idea's origin, enough mathematics to explain its behavior, enough systems detail
to anticipate deployment constraints, and enough evidence to distinguish a
documented result from an attractive story.

## P.1 The book's organization

The three primary learning sections appear in the requested order:

1. **Supervised Learning Algorithms:** fitting externally supplied targets,
   including classical predictors and representative supervised neural systems.
2. **Semi-Supervised Learning Algorithms:** using both labeled and unlabeled
   examples to improve the same learning task.
3. **Unsupervised Learning Algorithms:** discovering structure or modeling
   observations without task-specific target annotations, including an explicitly
   identified self-supervised pretraining division.

A fourth, standalone part examines **Mixture of Experts (MoE)** as an
architecture and a distributed-systems design. Its individual models also
appear in the appropriate learning section. The original supervised mixtures
and GShard translation are not incorrectly reclassified as unsupervised just
because later language models use experts.

The [complete contents](../CONTENTS.md) links to individual entries. Each
sub-category ends with a comparison table. The final reference chapters contain
a [cross-family comparison](09-comparative-guide.md) and a
[glossary](10-glossary.md). Continuation manifests state both what a chapter
covers and which extensions would require a later edition.

## P.2 What determines a learning category?

**The training signal, not the layer type.** A convolution can participate in a
supervised image classifier, a semi-supervised consistency model, or an
unlabeled denoising autoencoder. An architecture does not have an immutable
supervision category.

| Learning setting | Information supplied to learning | Representative objective | Important boundary |
|---|---|---|---|
| Supervised | Inputs and externally supplied targets | Minimize prediction loss against labels or continuous outcomes | Targets can be expensive, noisy, weak, or delayed |
| Semi-supervised | A labeled subset plus explicitly exploited unlabeled data | Supervised loss plus a consistency, graph, generative, or pseudo-label term | Merely having unlabeled records on disk does not make a method semi-supervised |
| Unsupervised | Observations without the downstream task's labels | Density likelihood, reconstruction, clustering, or a structural criterion | A discovered group is not automatically a scientifically meaningful class |
| Self-supervised | Targets constructed from observations themselves | Predict a masked token, next token, corrupted view, or related representation | The objective has a target, but not a manually supplied downstream class label |
| Weak or naturally paired supervision | Noisy rules, metadata, captions, or naturally associated modalities | Learn from imperfect labels or cross-modal matching | Not equivalent to an absence of supervision |
| Reinforcement learning | Actions, environmental transitions, and rewards | Improve expected return under a policy | A separate paradigm, not a fourth kind of unlabeled clustering |

For foundation models, the placement normally follows the **base pretraining
objective**. Subsequent instruction tuning, preference optimization,
distillation, or reinforcement learning is recorded in the training-paradigm
field. A supervised fine-tuning stage does not erase the self-supervised
pretraining history; neither does pretraining make the final assistant wholly
unsupervised.

There are genuine taxonomic boundary cases. CLIP's original description is
*natural language supervision*: a caption is informative supervision, even
when no one collected ImageNet-style class labels. The representation-learning
chapter explicitly groups such cross-modal pretraining with self-supervised
methods for comparison while identifying the paired signal. Text-conditioned
image generation likewise combines a self-generated denoising/reconstruction
target with conditioning from associated text. These are **cross-referenced
hybrids**, not evidence that paired language is unlabeled in every sense.

Graph learning needs similar care. A model trained with a small label mask on
a graph containing unlabeled nodes can be a semi-supervised, transductive
system. The same message-passing architecture trained on fully labeled graphs
is supervised. Each graph entry identifies its representative formulation.

## P.3 Scope and the meaning of comprehensive

This is a **broad first-edition reference with an explicit boundary**, not an
assertion that every algorithm, modification, vendor release, and application
ever published has been enumerated. Such a claim would be unverifiable.
Several entries intentionally cover a named family and distinguish its
important variants; a family entry is not counted once per mentioned
checkpoint.

The edition covers classical prediction, kernels and ensembles, structured
prediction, supervised neural architectures, major semi-supervised methods,
clustering, dimensionality reduction, density and anomaly modeling, pattern
mining, neural generation, representation pretraining, foundation-model
families, and sparse experts. It includes the neural families and MoE examples
specified in the research brief.

Standalone reinforcement-learning algorithms, the full causal-inference
literature, every recommender-system design, every optimizer, every neural
architecture-search method, and every domain-specific scientific model are
**not exhaustive sub-catalogs in this edition**. Training-related concepts such
as RLHF appear where they clarify included entries. The chapter manifests
identify further extensions without disguising them as completed coverage.

**Evidence cutoff: September 8, 2026.** This is an editorial boundary, not a
claim that every source was published recently or that every model is the
latest available version. Historical checkpoint names and dates are retained.
Proprietary families are described only to the level supported by their cited
public disclosures.

## P.4 Anatomy of an entry

Every individual catalog entry supplies nine common fields:

| Field | What the reader should learn |
|---|---|
| Name | The canonical name, abbreviation, aliases, and the entry's boundaries |
| Category & sub-category | The representative training signal and related uses |
| Originating paper/vendor/year | Where the idea was introduced and which implementation or release is being discussed |
| Core mechanism | The model, objective, update rule, or mathematical intuition |
| Inputs/outputs and typical data types | What enters the method and what its output actually means |
| Strengths and limitations | Conditions under which its assumptions help or fail |
| Computational complexity / scalability notes | Costs with explicit dimensions and computational assumptions |
| Real-world problem solved - REQUIRED WORKED EXAMPLE | A named application, research case, or clearly labeled illustrative calculation |
| Notable vendor implementations/libraries | Implementations, not unsupported claims of customer adoption |

Neural-network entries add architecture descriptions, activations, loss
functions, optimizers and schedules, regularization, backpropagation
considerations, parameter/scaling behavior, training stages, and
hardware/parallelism requirements.

An explicit **not publicly disclosed** is a substantive value, not a missing
field. An API model can be useful and extensively evaluated without exposing
its expert count, activation function, or training optimizer. Filling those
fields with a conventional Transformer recipe would manufacture knowledge.

## P.5 Evidence policy for worked examples

Every worked-example paragraph carries an evidence label:

| Label | Meaning | What it does not establish |
|---|---|---|
| **Sourced application** | A named organization or published scientific application actually used the method | A randomized causal estimate of business impact unless the source supplies one |
| **Research benchmark** | A real, named study evaluated the method on identifiable data | Commercial deployment, patient benefit, or production cost savings |
| **Illustrative (not a claimed deployment)** | A transparent teaching scenario or calculation where verified deployment evidence was not established | That the named hypothetical outcome happened in the world |

Some entries combine a documented application with an explicitly illustrative
arithmetic step. The two are distinguished locally. If a source does not
report downtime, profit, latency, or another business KPI, the entry says so.
The phrase **production KPI not reported** must never be interpreted as a zero
effect, a failed deployment, or an invitation to invent a percentage.

A source-backed example follows a chain:

**problem -> data -> learning mechanism -> output -> decision -> reported result**.

For a code model, for example, a programming benchmark is a real research
case: a function signature and problem statement become a generated function,
which is evaluated against the benchmark's tests. A reported pass rate
describes that protocol; it does not measure how much developer time an
organization saved.

The reason an algorithm fits is sometimes the author's technical analysis
rather than the original organization's documented procurement decision.
Entries distinguish these. "This method accommodates nonlinear tabular
interactions" does not imply that a hospital or manufacturer publicly gave that
reason for choosing it.

Primary papers, official model cards, technical reports, source repositories,
and official implementation documentation are preferred. Origin and
application claims carry nearby links. A vendor comparison remains a
**vendor-reported comparison**, even when the numbers are accurately quoted.
Versions, baselines, label budgets, shot counts, data splits, and metric
definitions are part of the result.

## P.6 Reading numbers without being misled

**Accuracy, error, and percentage points differ.** An error rate falling from
10% to 8% is a reduction of 2 percentage points and a relative error reduction
of 20%. It is not automatically a 20% gain in accuracy.

**Training cost, inference FLOPs, latency, throughput, and price differ.**
GPU-hours describe resource consumption under a stated implementation.
Inference FLOPs omit memory movement and communication. API prices include
commercial decisions. A lower active-parameter count is not an audited
cost-per-token reduction.

**Data and protocol matter.** A few-shot score with eight sampled answers and
majority voting is not a single-sample score. An ImageNet ensemble's top-5
error is not a single model's top-1 error. Different prompt parsers can change a
language-model benchmark without changing its weights.

**Uncertainty is not confined to confidence intervals.** Dataset contamination,
label errors, undisclosed training data, different serving kernels, and
nonrepresentative traffic can all limit a comparison. Cross-study numbers
should not be combined into an unofficial universal leaderboard.

**Scaling laws are empirical models, not guarantees.** The influential
[neural language-model scaling study](https://arxiv.org/abs/2001.08361) and
[compute-optimal training study](https://arxiv.org/abs/2203.15556) concern
particular objectives, datasets, budgets, and model families. Their fitted
relationships do not certify that a small specialized dataset, an MoE router,
or a long-context production service will behave identically.

## P.7 Mathematical and systems notation

| Symbol | Default meaning | Qualification |
|---|---|---|
| $`n`$ | Number of training examples | Tokens or observations are stated explicitly when used instead |
| $`d`$ | Input features or hidden width | A chapter declares the relevant interpretation |
| $`p`$ | Number of model parameters | Total and active MoE parameters are distinguished |
| $`k`$ | Neighbors, clusters, or selected experts | A local definition takes precedence |
| $`T`$ | Sequence length or time steps | Not automatically the number of training epochs |
| $`L`$ | Number of layers | A loss is written as $`\mathcal{L}`$ where ambiguity matters |
| $`B`$ | Batch size | Often counts sequences, not tokens |
| $`E`$ | Epochs or number of experts | Explicitly disambiguated in the MoE chapters |
| $`X,y`$ | Features and targets | Targets may be scalar, class-valued, structured, or self-generated |
| $`\theta`$ | Trainable parameters | Includes routers when they are learned |
| $`\lambda`$ | Regularization or auxiliary-loss weight | Not a universal hyperparameter shared across methods |

Big-O statements describe an algorithmic regime, not a benchmark on a
particular processor. Dense least-squares factorization, sparse iterative
least squares, and streaming SGD have different costs despite fitting closely
related statistical models. Exact Gaussian processes and inducing-point
approximations are likewise different computational regimes.

For dense self-attention, the attention operation scales quadratically in
sequence length, but the **whole Transformer block** also contains projection
and FFN costs. Memory-efficient exact attention can avoid materializing a
quadratic intermediate without turning all attention arithmetic into a
linear-time computation.

## P.8 From a catalog to a deployed system

Start with the decision and its failure cost, not a fashionable model name.
Establish an appropriate baseline, a leakage-resistant data split, a metric
that matches the decision, and a realistic operating threshold. Compare methods
at the same data and resource budget before attributing a gain to architecture.

For temporal, grouped, medical, or multi-site data, random row splitting can
be misleading. For clustering and visualization, convincing pictures do not
substitute for stability or external validity. For generative models, inspect
failure modes, data rights, privacy, factuality, and the consequences of acting
on generated output.

Hosted services such as Vertex AI, Azure Machine Learning, and Amazon
SageMaker are platforms, not learning algorithms. Their catalogs, regions,
versions, licenses, and serving capabilities change. A library listing in an
entry is a starting point for implementation, not a guarantee of service
availability or a deployment instruction.

## Coverage and continuation manifest

This guide establishes the taxonomy, evidence labels, entry schema, notation,
and boundaries used throughout the first edition. Continue with
[classical supervised methods](01-supervised-classical.md) or use the
[complete contents](../CONTENTS.md) to select a topic.

Further editions can deepen the deliberately bounded areas listed in P.3,
expand independently documented production case studies, and add newer
version-specific disclosures. None of those extensions is implied by the
word "comprehensive."
