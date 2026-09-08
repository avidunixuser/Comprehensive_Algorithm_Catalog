# 4. Mixture of Experts: A Dedicated Deep Dive

Mixture of Experts is not a synonym for an ensemble, a collection of chatbots,
or a model that has expert-level knowledge. It is a way to make computation
**conditional on the input**. A learned or fixed routing rule decides which
subnetworks contribute, and with what weights.

The central attraction is a separation between **stored capacity** and
**computation performed for one input**. The central difficulty is that the
separation is statistical and algorithmic, but the hardware must still store,
move, and coordinate the capacity.

Individual historical supervised mixtures and GShard appear in
[Section 1.11](02-supervised-neural.md). The self-supervised language-model
entries are in [Section 3.15](07-moe-models.md). This chapter connects their
mechanisms and explains which comparisons are justified.

## 4.1 From adaptive local experts to sparse language models

In [Adaptive Mixtures of Local Experts](https://doi.org/10.1162/neco.1991.3.1.79),
Robert Jacobs, Michael Jordan, Steven Nowlan, and Geoffrey Hinton (1991)
studied jointly trained expert networks and a gating network. The gate assigns
input-dependent responsibility; experts need not solve the entire input space
equally well.

A probabilistic mixture makes the idea explicit:


$$
p(y\mid x)=\sum_{e=1}^{E}\pi_e(x)\,p_e(y\mid x),\qquad
\pi_e(x)\geq 0,\quad \sum_e\pi_e(x)=1.
$$


Under a likelihood objective, an expert's training responsibility depends on
both its gate weight and how well it explains the target. This differs from
simply taking the mean of independently trained predictions. Minimizing the
loss of an averaged prediction is also not generally identical to maximizing
the likelihood of a mixture of predictive distributions.

The original formulation is **supervised**. It does not become
self-supervised retrospectively because modern language models use expert
FFNs.

| Period | Development | What changed |
|---|---|---|
| 1991 | Adaptive mixtures of local experts | Jointly learned input-dependent division of a supervised problem |
| 1994 | Hierarchical mixtures of experts | A hierarchy of gates and expert responsibilities |
| 2017 | Sparsely gated neural MoE | Large conditional subnetworks and noisy top-k routing at deep-learning scale |
| 2020 | GShard | Large multilingual translation with automated sharding and sparse Transformer FFNs |
| 2021-2022 | Switch and GLaM | Simple top-1 routing and sparse general-purpose language pretraining |
| 2022 | Expert-choice, ST-MoE, MegaBlocks | Routing balance, numerical/transfer stability, and dropless execution |
| 2023-2024 | Mixtral, DeepSeek, DBRX, Arctic, OLMoE, released Grok-1 | A wider range of publicly inspectable sparse language-model designs |

The modern scaling lineage includes
[Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538),
[GShard](https://arxiv.org/abs/2006.16668),
[Switch](https://arxiv.org/abs/2101.03961), and
[GLaM](https://arxiv.org/abs/2112.06905). The table is a selective historical
spine, not a claim that intermediate research or earlier conditional
computation did not exist.

**Dense versus sparse activation.** A dense mixture evaluates every expert
and combines their outputs. A sparse mixture evaluates only a subset for a
given token/input. Dense gating can still learn specialization; sparse
execution is the additional computational property. An ordinary ensemble may
be trained independently and combined only at the final output, whereas an
MoE layer can be one jointly trained component deep inside a network.

## 4.2 Gating and routing mechanisms

For a token representation $`h\in\mathbb{R}^d`$, let the router compute scores
$`s=W_rh`$. A token-choice router selects an index set $`\mathcal{S}(h)`$
and combines expert outputs:


$$
z(h)=\sum_{e\in\mathcal{S}(h)}g_e(h)\,F_e(h).
$$


The equations hide important implementation decisions: whether probabilities
are normalized before or after selection, whether noise is added, how capacity
is enforced, and whether the input is routed by token, sequence, task, or
modality.

| Routing mechanism | Selection rule | Benefit | Important limitation |
|---|---|---|---|
| Dense soft mixture | Evaluate every expert with soft weights | Fully differentiable combination and broad gradient flow | Arithmetic grows with expert count |
| Learned top-k token choice | Each token selects its k highest-scoring experts | Fixed expert arithmetic per token before overflow | Independent choices can overload the same experts |
| Noisy top-k | Perturb learned logits before selection during training | Exploration and smoother expected load estimates | Noise and balance objectives require careful tuning |
| Switch/top-1 | Each token selects one expert | Simple dispatch and reduced expert arithmetic | Capacity and gate-gradient details are especially important |
| Expert choice | Each expert selects a fixed-capacity set of tokens | Directly controls expert workload | Tokens can receive different numbers of experts, including none |
| Hash routing | Map token identities/features to experts by a fixed rule | Avoids learned-router optimization | Frequency skew and inflexible semantic assignment |
| Constrained or grouped routing | Restrict choices by device/node/group before selecting | Bounds expensive communication | Restricts the available expert combinations |

**Top-k and gradients.** The selected indices are discrete. Away from a
selection boundary, ordinary autodiff can propagate through the selected
scores, mixing weights, and expert functions. It does not differentiate the
change in rank ordering itself. A straight-through estimator is an optional
design, **not an automatic description of every top-k router**.

In [noisy top-k gating](https://arxiv.org/abs/1701.06538), the router can
add learned-scale Gaussian noise before applying top-k and softmax to the
retained scores. The noise supports exploration and differentiable estimates
of expected load. This does not mean that arbitrary hard assignment is fully
smooth.

**The top-1 normalization trap.** If one renormalizes a single selected score
to a probability of exactly one, the gate's output multiplier no longer
provides that route's usual task-loss gradient. Switch retains the selected
probability from the full softmax instead. A top-2 Mixtral-style selected-score
normalization is therefore not mechanically interchangeable with top-1
routing.

**Expert choice.** In
[Zhou et al. (2022)](https://arxiv.org/abs/2202.09368), each expert chooses
its highest-scoring tokens up to a fixed bucket size. Equal bucket sizes
balance expert loads, but do not guarantee coverage of every token: several
experts may choose the same inputs. For example, two experts each choosing
two of four tokens can both choose tokens 1 and 2, leaving tokens 3 and 4
without an expert contribution. Residual paths, coverage constraints, or
other design choices determine how this is handled.

**Hash routing.**
[Hash Layers for Large Sparse Models](https://arxiv.org/abs/2106.04426)
studies fixed token-to-expert assignments. A deterministic hash avoids
learning a router but does not make a highly frequent token disappear from
the load distribution. Collision and frequency behavior matter.

#### Worked routing calculation

This is an **illustrative calculation, not a production case study**. Suppose
a four-expert router produces probabilities $`(0.55,0.25,0.15,0.05)`$ and
selects two experts. Renormalizing the selected probabilities gives
$`(0.6875,0.3125)`$. If their output vectors are $`(2,0)`$ and $`(0,4)`$, the
expert contribution is


$$
0.6875(2,0)+0.3125(0,4)=(1.375,1.25).
$$


A residual block adds this contribution to its residual stream. These
components are hidden features, not human-readable class probabilities.
The computation uses two expert functions but still required router scoring
and whatever shared attention preceded it.

## 4.3 Load balancing, capacity, overflow, and collapse

Let $`N`$ be tokens dispatched together, $`E`$ experts, and $`k`$ selected experts
per token. A common token-choice capacity convention is


$$
C=\left\lceil c\,\frac{Nk}{E}\right\rceil
$$


assignments per expert, where $`c`$ is a capacity multiplier relative to the
average selected load. Implementations use different normalizations:
Switch's top-1 convention has $`k=1`$, while expert-choice papers may use a
capacity factor to denote the *average experts per token*. Always read the
definition before comparing the same numeric factor across systems.

**Illustrative capacity calculation.** For 1,024 tokens, eight experts,
top-2 routing, and $`c=1.25`$, the average load is 256 assignments and
capacity is 320 per expert. An expert receiving 370 assignments exceeds
its local capacity by 50 even though the global capacity of 2,560 exceeds
the 2,048 requested assignments. Sufficient aggregate slots do not guarantee
a feasible local allocation.

**Overflow policies are part of the model.** An implementation can skip
overflowed expert contributions, retain a residual bypass, try another expert,
prioritize assignments, raise capacity, or use dropless execution. These
policies have different training and inference semantics. "Token dropping"
usually means dropping an expert computation, not deleting the token from
the entire text sequence.

For top-1 routing, a Switch-style auxiliary objective can be written as


$$
\mathcal{L}_{\mathrm{balance}}
  =\alpha E\sum_{e=1}^{E}f_eP_e,
\qquad
f_e=\frac{1}{N}\sum_{t=1}^{N}\mathbf{1}[\operatorname{route}(t)=e],
\qquad
P_e=\frac{1}{N}\sum_{t=1}^{N}p(e\mid h_t).
$$


Here $`f_e`$ is the observed assignment fraction and $`P_e`$ is mean router
probability. The assignment indicator is normally treated as nondifferentiable;
the probability term supplies a gradient signal. Other routers use different
normalizations, importance/load penalties, or balance scopes. A global
average can hide imbalance within individual dispatch groups or devices.

**Why collapse happens.** Small initial differences can make a router favor
a subset of experts. Those experts receive more training, become more useful,
and attract more traffic, while other experts remain undertrained. Capacity
overflow, homogeneous batches, poor initialization, overly sharp scores, and
an unsuitable auxiliary coefficient can reinforce the feedback loop.
Perfectly uniform traffic is not always the best semantic specialization,
but chronic starvation wastes capacity and destabilizes execution.

**Balance is not numerical stability.**
[ST-MoE](https://arxiv.org/abs/2202.08906) introduces a router z-loss of
the form


$$
\mathcal{L}_z=\frac{\beta}{N}
\sum_{t=1}^{N}\left(\log\sum_{e=1}^{E}\exp(s_{te})\right)^2.
$$


This controls the log-partition magnitude and improves numerical stability.
It is **not the same loss as equalizing expert loads**. Softmax probabilities
are unchanged by a common shift of all logits, but z-loss changes, illustrating
why it constrains a different degree of freedom.

**Loss-free balancing needs precise wording.**
[DeepSeek-V3](https://arxiv.org/html/2412.19437v1) adjusts expert-specific
biases using observed load. These biases affect selection while underlying
affinities determine mixing. The report also retains a small **sequence-wise
auxiliary balance loss**. "Auxiliary-loss-free" describes the main balancing
strategy, not an absence of all auxiliary objectives; V3 additionally trains
with multi-token prediction.

**Dropless is a systems property, not infinite capacity.**
[MegaBlocks](https://arxiv.org/abs/2211.15841) uses block-sparse
computation to execute variable expert token counts without conventional
capacity dropping or padding every expert to a wasteful common maximum.
Finite memory, pathological skew, and communication remain constraints.
It does not make the router statistically optimal or remove the need to
monitor utilization.

## 4.4 Where experts live in a Transformer

The most common language-model placement replaces the **position-wise FFN**
inside a Transformer block, not its entire attention mechanism:

```text
residual stream
    |
    +--> normalization --> attention ------------------+
    |                                                  |
    +<---------------- residual addition <-------------+
    |
    +--> normalization --> router --> selected FFNs ---+
    |                                  |              |
    |                         weighted combination    |
    +<---------------- residual addition <-------------+
```

Attention mixes information across sequence positions. A standard FFN or
expert transforms each resulting position. Experts therefore see
contextualized representations even when routing is computed token by token.
One token can select different experts in different layers, and one word
can select different experts in different contexts.

| Placement pattern | Examples | Consequence |
|---|---|---|
| Alternating dense and MoE FFNs | GShard, GLaM; selected Switch configurations | Lower expert-storage/communication frequency than all-MoE FFNs |
| MoE FFN in every decoder block | Mixtral, OLMoE | More routing opportunities and communication boundaries |
| Initial dense blocks, then MoE | DeepSeekMoE, DeepSeek-V2, DeepSeek-V3 | A shared early representation precedes conditional capacity |
| Dense/shared path plus routed experts | DeepSeek shared experts; Arctic hybrid | Always-active computation must be included in active counts |
| Selected vision or modality blocks | Research extensions beyond this chapter's full catalog | Patch, modality, and routing granularity alter the systems tradeoffs |

The exact frequency is a checkpoint property. In the referenced releases,
Mixtral 8x7B uses 32 expert-FFN blocks and 8x22B uses 56; OLMoE uses 16.
DeepSeek-V2 keeps its first FFN dense, while V3 keeps its first three dense.
Arctic's released configuration uses an MoE frequency of one across its
35 hybrid layers. These details are sourced in the
[model catalog](07-moe-models.md), not inferred from a vendor name.

An expert number is an index, not a semantic job title. Empirical routing
patterns may correlate with token syntax, domain, language, or position, but
"expert 4 is the legal expert" requires evidence and is not generally a
guaranteed property of sparse training.

## 4.5 Parameter ledger: total versus active

Counts below are historical, approximate where indicated, and subject to the
sources' counting conventions. **B** means billion and **T** trillion
parameters, not bytes. For GShard and Switch-C, the active counts are from
GLaM's comparative accounting, as noted.

| Model / representative release | Organization | Total parameters | Active per token | Routing / placement | Source and qualification |
|---|---|---:|---:|---|---|
| GShard-M4, 2020 | Google | About 600B | About 1.5B | Top-2; alternating expert FFNs | [GLaM comparison](https://arxiv.org/abs/2112.06905); supervised translation, not the pretraining models in Part 3 |
| Switch-C, 2021 | Google | About 1.6T in Switch; 1.5T in GLaM's rounded table | About 1.5B in GLaM's table | Top-1; 2,048 experts per sparse layer | [Switch](https://arxiv.org/abs/2101.03961), [GLaM](https://arxiv.org/abs/2112.06905); cross-report convention |
| GLaM 64B/64E, 2021 | Google | About 1.2T | 96.6B | Top-2 of 64; alternating sparse FFNs | [Report](https://arxiv.org/abs/2112.06905) |
| Mixtral 8x7B, 2023 | Mistral AI | 46.7B | 12.9B | Top-2 of 8; every FFN | [Release](https://mistral.ai/news/mixtral-of-experts) |
| Mixtral 8x22B, 2024 | Mistral AI | 141B | 39B | Top-2 of 8; every FFN | [Release](https://mistral.ai/news/mixtral-8x22b) |
| DeepSeekMoE 16B, 2024 | DeepSeek | 16.4B | About 2.8B | Six of 64 routed plus two shared; after first dense block | [Report/release](https://github.com/deepseek-ai/DeepSeek-MoE) |
| DeepSeek-V2, 2024 | DeepSeek | 236B | 21B | Six of 160 routed plus two shared; MLA | [Release](https://github.com/deepseek-ai/DeepSeek-V2) |
| DeepSeek-V3, December 2024 | DeepSeek | 671B main model | 37B | Eight of 256 routed plus one shared; MLA | [Release](https://github.com/deepseek-ai/DeepSeek-V3); 685B including 14B MTP weights |
| Grok-1, released 2024 | xAI | 314B | Vendor says 25%; about 78.5B if applied literally | Top-2 of 8 | [Release](https://x.ai/news/grok-os); derived active approximation, shared-weight convention not independently reconciled |
| DBRX, 2024 | Databricks | 132B | 36B | Top-4 of 16 | [Technical release](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) |
| Arctic, April 2024 | Snowflake | 480B | 17B | Top-2 of 128 plus dense residual model | [Repository](https://github.com/Snowflake-Labs/snowflake-arctic); rounded counts |
| OLMoE-1B-7B-0924 | AI2 and collaborators | 6.9B | 1.3B | Top-8 of 64; dropless | [Repository](https://github.com/allenai/OLMoE) |

The Grok row deliberately exposes an accounting limitation rather than
pretending that a rounded activation percentage is an exact inventory.
Similarly, the additional V3 prediction module must not be silently counted
in one model's storage and omitted from another model's comparison.

Publicly described proprietary MoE variants, including specific Gemini
releases, may not disclose total or active counts. The
[foundation-model chapter](06-foundation-models.md) states those boundaries.
Rumored GPT, Claude, or other vendor expert counts are not entries in this
ledger.

For a simplified architecture with $`L_m`$ expert layers, identical per-expert
parameter count $`P_e`$, and always-counted shared parameters $`P_s`$:


$$
P_{\mathrm{total}}\approx P_s+L_mEP_e,\qquad
P_{\mathrm{active}}\approx P_s+L_mkP_e.
$$


For a gated FFN with width $`d`$ and intermediate size $`h`$, three projection
matrices contribute approximately $`3dh`$ expert parameters, ignoring biases.
These are **accounting approximations**, not a literal count of every memory
location touched: embeddings and output heads are especially
convention-dependent.

Increasing $`E`$ at fixed $`k`$, $`d`$, and $`h`$ can increase capacity while holding
selected FFN arithmetic roughly fixed. It does not hold router scoring,
storage, network behavior, data requirements, or training quality fixed.
Published scaling laws for dense models should not be transferred to sparse
models by replacing one parameter symbol without examining these assumptions.

## 4.6 Training challenges and distributed execution

**Expert parallelism.** Different devices own different experts. Tokens are
packed by destination, exchanged, processed, and returned to their originating
sequence positions. Dispatch and combine commonly involve all-to-all
communication. A device with the busiest experts can determine a step's
latency even if average utilization looks reasonable.

**Data parallelism** distributes examples across replicas. **Tensor
parallelism** distributes matrix operations inside layers. **Pipeline
parallelism** partitions layers into stages. **State sharding** distributes
optimizer/gradient/parameter storage. They solve different problems and
introduce different collective operations. Combining all of them is not
automatically better.

An illustrative forward dispatch/combine payload estimate is


$$
V_{\mathrm{forward}}\approx 2Nkd\,b
$$


bytes, where $`b`$ is bytes per activation element. This counts each logical
outgoing/return activation transfer once and assumes all selected assignments
are remote; it excludes gradients, protocol overhead, and local placements.
For $`N=4096`$, $`k=2`$, $`d=4096`$, and BF16 activations ($`b=2`$), this is
**128 MiB per MoE layer**. It is not a latency prediction. Network topology,
overlap, contention, collective algorithms, and backward communication matter.

**Small matrix inefficiency.** With many experts, each may see too few tokens
for efficient large matrix multiplication. Grouped GEMMs and block-sparse
executors improve packing, but skew and padding still affect effective
throughput. Increasing batch size can help arithmetic intensity while
worsening latency or memory demand.

**Optimization and precision.** Routers can be sensitive to score magnitude
and small numerical differences. Higher-precision router computations, stable
softmax, gradient clipping, careful initialization, and appropriately weighted
auxiliary losses are common controls. A model that loses fewer tokens can
still learn poorly if routing starves or over-specializes experts.

**Training state is larger than inference weights.** Gradients, optimizer
moments, master weights, saved activations, and communication buffers add to
the model's memory. Low precision, sharding, and activation recomputation
change the ledger but do not make it disappear.

**Hardware-aware routing.** Group or node limits can reduce expensive remote
transfers, at the cost of restricting possible expert combinations. The
DeepSeek-V3 report combines node-limited routing, expert/pipeline strategies,
FP8 computation, and overlap. Its near-hidden communication cost under its
conditions does not imply that all-to-all is free on a different cluster.

## 4.7 Inference: memory, arithmetic, and useful answers

An MoE is attractive when extra capacity improves quality enough to justify
the full memory and systems cost. It can be unattractive when requests are
small, batch sizes are low, expert weights cannot remain resident, or network
latency dominates.

| Resource | Primarily affected by | Why active parameter count is insufficient |
|---|---|---|
| Weight memory | Total stored weights and precision | Unselected experts must still be available |
| FFN arithmetic | Selected experts, width, depth, batch | A small number of large experts can cost more than many tiny ones |
| Attention/KV memory | Attention design, context, batch, KV heads | Expert sparsity does not itself compress attention state |
| Dispatch overhead | Top-k, token count, placement, interconnect | More destinations can mean more expensive communication |
| Latency | Critical path, batching, scheduling, cache locality | Less arithmetic can coexist with slower individual requests |
| Throughput | Utilization, concurrency, kernels, memory bandwidth | A low-batch latency result does not establish high-load throughput |
| Cost per useful answer | All resource costs, answer length, failure/retry rate | A cheap token can still produce an expensive failed task |

**Illustrative storage comparison, not a measured deployment.** Mixtral
8x7B's 46.7B total weights occupy about **93.4 GB decimal at two bytes per
weight**, before caches, scales, temporary buffers, and runtime overhead.
Idealized four-bit payload would be **23.35 GB**, but practical quantization
adds metadata and may retain some tensors at higher precision. A
12.9B-active model therefore does **not** have the same resident weight
footprint as a dense 12.9B model.

**Prefill and decode differ.** Long-prompt prefill can exploit large batches
of token operations. Autoregressive decode repeatedly produces new tokens,
often stressing weight/KV bandwidth and synchronization. An MoE system should
be evaluated in both regimes with actual prompt lengths and concurrency.

**Offloading changes the tradeoff.** Moving experts to host memory or storage
can make a checkpoint fit, but transferring selected weights on demand may
dominate generation time. Expert caching or replication helps some traffic
patterns and adds its own consistency, capacity, and scheduling costs.

**Quantization and distillation are not guaranteed equivalences.** Reduced
precision can alter router decisions or expert outputs; distillation into a
dense student changes the model. Both require task-level evaluation rather
than assuming that an active-parameter budget preserves all capability.

## 4.8 A production-oriented case: Databricks DBRX

**Evidence status: Sourced vendor application with separately identified
system benchmarks.** This case documents a real organization's public
deployment and stated architecture rationale. It does **not** claim an
independently audited customer cost saving.

**Industry and problem.** Databricks builds enterprise data and analytics
software. Its March 2024 release describes DBRX integration into
GenAI-powered products, including **early SQL applications**, and availability
through its Foundation Model APIs. The problem is to provide useful SQL,
code, and grounded language assistance without the active computation of a
comparably large dense model.

**Data and flow.** DBRX pretraining uses a reported **12T text/code
tokens**. For a SQL workflow, a user's request and relevant schema/context
are turned into a generated query, which the application must validate and
execute under appropriate permissions. The release does **not** publish the
internal SQL rollout dataset or an audited end-user success rate. Separately,
its public RAG benchmark uses Natural Questions/HotpotQA with retrieved
Wikipedia passages; those are not falsely substituted for the internal SQL
data.

**Why MoE rather than a dense alternative.** Databricks explicitly identifies
fine-grained MoE as a way to improve quality/compute tradeoffs: **132B total
parameters, 36B active, top-4 of 16 experts**. It also attributes progress to
training data, tokenization, optimization, and systems engineering. The
record therefore supports an architectural choice, not the stronger causal
claim that routing alone produced every gain.

| Public claim | Actual reported figure or observation | Correct interpretation |
|---|---|---|
| Service deployment | DBRX offered through Databricks APIs; early SQL-product integration described | A named vendor application, not a count of successful customer deployments |
| SQL rollout quality | Qualitative comparison with GPT-3.5 Turbo and GPT-4 Turbo | No public audited SQL business KPI in the cited release |
| Optimized service generation | **Up to 150 tokens/second/user**, with **8-bit quantization** | Vendor serving claim; not an SLA, average, or a result for all traffic |
| Comparison with Llama 2 70B | **Up to 2x inference throughput** in the described benchmark | Hardware/traffic/precision-specific comparison, not universal cost-per-token savings |
| Serving-comparison protocol | About **2,000 prompt tokens**, **256 output tokens**, **one new user per second**, **16-bit precision** and optimized TensorRT-LLM | Different protocol from the 8-bit 150-token service claim |
| Grounded-QA reference | DBRX **60.0%** versus compared GPT-3.5 Turbo **57.7%** on Natural Questions with top-ten retrieved passages | A research answer-matching outcome, not a SQL customer KPI |

All figures above come from the
[Databricks technical release, inference section, and Tables 3-4](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).
The benchmark runs tensor parallel across a node; the stated source should be
consulted before transferring a figure to another accelerator count or
configuration.

**What this case proves, and what it does not.** It provides a concrete
organization, use case, model design, public deployment statement, and actual
figures with their measurement conditions. It does not provide an
independent experiment where only MoE changes, a current API price, or a
verified percentage reduction in customer operating costs. That information
is not manufactured to make the case appear more complete.

Two related examples sharpen the boundary. DeepSeek-V2's
[release](https://github.com/deepseek-ai/DeepSeek-V2) reports **42.5%
training-cost savings, 93.3% KV reduction, and 5.76x maximum generation
throughput** relative to DeepSeek 67B, but combines MoE with MLA and system
changes. DeepSeek-V3's
[training ledger](https://arxiv.org/html/2412.19437v1) gives **2.788M
H800 GPU-hours** and an assumed **\$5.576M** rental cost, excluding earlier
R&D. Neither should be rewritten as an isolated MoE customer ROI statistic.

## 4.9 A decision framework and remaining questions

MoE should be compared with a dense candidate at a **specified quality
threshold and realistic workload**, not declared superior from parameter
counts.

1. Define the useful task outcome, acceptable failure rate, context lengths,
   response lengths, concurrency, and latency requirements.
2. Evaluate dense and sparse candidates with the same task data, prompts,
   retrieval, tools, and sampling budget.
3. Account for all resident weights, KV cache, quantization metadata,
   temporary memory, and replication/offloading requirements.
4. Measure prefill, decode, aggregate throughput, and tail latency under the
   intended hardware and traffic distribution.
5. Inspect expert load, overflow/drop behavior, kernel utilization, and
   inter-device traffic rather than only average GPU utilization.
6. Compute cost per successful task, including retries, longer outputs,
   failed calls, and operational complexity.

Several research questions remain open: how best to characterize expert
specialization; how to balance capacity, data, and training compute; how to
adapt routers without forgetting or collapse; how to route across modalities;
and how to make heterogeneous hardware practical without losing the
statistical advantages of flexible routing.

There is no universal MoE scaling law that settles these questions for every
dataset and system. The defensible conclusion is narrower and more useful:
**conditional computation is a powerful design option whose benefits depend
on the objective, the model, and the execution system together**.

## Coverage and continuation manifest

This standalone deep dive covers the 1991 foundation, dense/sparse activation,
top-k/noisy/Switch/expert-choice/hash routing, load balancing, capacity and
overflow, FFN placement, all requested representative vendors, total/active
counts with caveats, distributed training, inference tradeoffs, and a
source-grounded production-oriented case.

Continue to [cross-cutting comparisons](09-comparative-guide.md) and the
[glossary](10-glossary.md), or revisit the
[individual MoE entries](07-moe-models.md).

A later edition can add full vision/audio/task-routed model entries,
newer checkpoint-specific disclosures, and independently reproduced
cost-per-success measurements. This chapter does not pretend that such
unpublished or unperformed studies already exist.
