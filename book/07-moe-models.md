# 3. Unsupervised Learning Algorithms: Sparse MoE Model Catalog

## 3.15 Self-supervised sparse expert language models

This chapter catalogs representative MoE language models by their **base
pretraining signal**, usually next-token prediction or masked-span
reconstruction. Instruction and preference training are recorded separately.
The original 1991 supervised mixtures and GShard's supervised multilingual
translation appear in [Section 1.11](02-supervised-neural.md).

The [dedicated MoE deep dive](08-moe-deep-dive.md) derives routing, load
balancing, capacity, and communication behavior. Here, the same concepts are
attached to specific, dated implementations. None of these historical releases
is implicitly described as its vendor's latest model.

Parameter counts are reported using their source's conventions. **Total**
counts govern weights storage; **active per token** counts are a useful but
incomplete compute proxy. Shared attention, embeddings, dense layers, auxiliary
prediction heads, and routing overhead prevent the shortcut
`active = total * selected_experts / all_experts` from being generally valid.

### 3.15.1 Switch Transformer

**Name:** Switch Transformer; Switch-Base, Switch-Large, Switch-XXL, and
Switch-C are distinct scale configurations.

**Category & sub-category:** Unsupervised/self-supervised masked-span
pretraining; sparse encoder-decoder Transformers. Downstream fine-tuning is
supervised.

**Originating paper/vendor/year:** William Fedus, Barret Zoph, and Noam
Shazeer, Google, 2021 preprint and 2022 JMLR publication,
[Switch Transformers](https://arxiv.org/abs/2101.03961).

**Core mechanism:** Replace selected Transformer FFNs with expert banks. A
learned router sends each token to its single highest-scoring expert, retaining
the gate probability as an output multiplier. This top-1 design simplifies
dispatch relative to top-2 models. Capacity limits and an auxiliary
load-balancing objective keep a small number of experts from consuming most
of the traffic.

**Inputs/outputs and typical data types:** Corrupted text spans enter an
encoder-decoder; the decoder predicts the missing spans. Fine-tuned versions
map text questions or task prompts to answer sequences.

**Strengths and limitations:** Large conditional capacity at moderate active
compute; relatively simple routing. Memory, expert imbalance, overflow, and
transfer sensitivity remain substantial. Better pretraining perplexity does
not guarantee proportionate downstream improvement.

**Computational complexity / scalability notes:** Expert FFN arithmetic per
token uses one expert, but router scoring still considers the expert bank.
Training memory and optimizer states scale with total weights. Communication
and expert batch sizes determine realized speed.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Google's study addressed
closed-book question answering: a Natural Questions query is encoded and an
answer is generated without retrieving supporting documents. Pretraining
supplies broad text knowledge; supervised QA fine-tuning maps that knowledge
to answers. The paper's **Switch-XXL** experiment reports Natural Questions
exact match of **34.4 versus 32.8** for its T5-XXL comparison without salient
span masking. Sparse capacity is the technical rationale for improving
knowledge at a manageable active compute budget. This is a research result,
not an audited search-service business KPI, and it is not a result for
Switch-C. See the
[paper's Section 5.6](https://ar5iv.labs.arxiv.org/html/2101.03961).

**Notable vendor implementations/libraries:** Google's Mesh TensorFlow
research implementation; Hugging Face Transformers' SwitchTransformers
implementation and released Google checkpoints.

**Architecture diagram description:** `corrupted tokens -> encoder blocks
[attention + residual; alternating dense/switch FFN + residual] -> decoder
[causal attention; encoder cross-attention; dense/switch FFN] -> token head`.
Top-1 expert selection acts on each token representation, not on the entire
input document.

**Activation functions used and why:** Softmax normalizes router and attention
scores. FFN nonlinearities are configuration-dependent: the paper discusses
ReLU FFNs and GEGLU variants at larger scales. Do not assume every named
Switch checkpoint has an identical FFN.

**Loss function(s):** T5-style span reconstruction cross-entropy plus the
router's auxiliary load-balancing loss; supervised task losses during
fine-tuning.

**Optimization algorithm(s):** T5-style adaptive optimization, commonly
Adafactor, is the relevant implementation family. Learning rate, warmup/decay,
and fine-tuning settings must be taken from the selected configuration; a
single optimizer schedule is not specified for the entire family here.

**Regularization techniques:** Dropout and task-specific fine-tuning
regularization; the paper studies increased **expert dropout** rather than
indiscriminately increasing dropout everywhere. Capacity and balancing are
also operational constraints.

**Backpropagation considerations:** Hard expert indices have no ordinary
derivative. Gradients flow through selected expert outputs and gate weights.
Router computations in higher precision and careful initialization improve
stability; a skipped expert assignment does not receive its normal task
gradient.

**Parameter count / scaling behavior:** Switch-XXL is reported at about
**395B total** and Switch-C at about **1.6T total**, with **2,048 experts**
in Switch-C's expert layers. The
[GLaM comparative table](https://ar5iv.labs.arxiv.org/html/2112.06905)
tabulates Switch-C as **1.5T total / 1.5B activated** using its own rounded
accounting. These separately reported conventions should not be mistaken for
an exact inventory of one downloaded checkpoint.

**Training paradigm:** Self-supervised pretraining, supervised fine-tuning,
and optional distillation to a dense student; distillation does not preserve
all teacher capability automatically.

**Hardware/parallelism considerations:** Data and expert parallelism on TPU
meshes; model/tensor parallelism for wider variants. Single-expert arithmetic
does not make a trillion-weight model fit on a single accelerator.

### 3.15.2 GLaM

**Name:** GLaM, Generalist Language Model; the largest reported sparse variant
is GLaM 64B/64E.

**Category & sub-category:** Unsupervised/self-supervised autoregressive
language modeling; sparse decoder-only Transformers.

**Originating paper/vendor/year:** Nan Du and colleagues, Google,
[GLaM](https://arxiv.org/abs/2112.06905), December 2021 preprint and ICML 2022.

**Core mechanism:** Alternate dense Transformer FFNs with MoE FFNs. A
softmax router selects two experts in each sparse layer. The model learns
next-token probabilities while a large expert pool expands its representational
capacity without activating the entire pool for each token.

**Inputs/outputs and typical data types:** Text prompts, few-shot examples,
and partial sequences; output is a distribution over subsequent subword
tokens and generated continuations.

**Strengths and limitations:** A clear demonstration of conditional capacity
for general-purpose few-shot language tasks. Its large total footprint,
research-scale infrastructure, proprietary training execution, and
cross-study evaluation differences limit easy reproduction.

**Computational complexity / scalability notes:** Two selected expert FFNs
per sparse layer, plus dense layers and attention. Expert parallelism saves
neither the total weight storage nor the cost of dispatching representations.
The entire block's cost includes attention and router projections.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Google studied general-purpose
language understanding and generation across **29 public NLP tasks**, such as
question answering and reading comprehension. A task prompt and demonstrations
become a continuation whose answer is scored under each task's protocol.
The reported largest GLaM model required **180 GFLOPs per token versus
350 GFLOPs** in the report's GPT-3 comparison and a reported training-energy
estimate of **456 versus 1,287 MWh**. The technical motivation was better
language capability per computation rather than a dense model with all
weights active. These are the authors' research/system comparisons, not a
controlled production electricity bill or an isolation of MoE from hardware
and data differences.
[Source: GLaM Table 1](https://ar5iv.labs.arxiv.org/html/2112.06905).

**Notable vendor implementations/libraries:** Google's research stack,
GSPMD/XLA sharding and Cloud TPU execution. The paper is not evidence of a
public, downloadable GLaM checkpoint or a currently available GLaM API.

**Architecture diagram description:** `tokens -> repeated [causal attention
+ residual -> dense FFN or top-2-of-64 expert FFN + residual] -> vocabulary
head`. Every other Transformer layer uses MoE in the reported design.

**Activation functions used and why:** Softmax for attention and routing.
The report describes a gated-linear/GELU modification in its **non-MoE**
FFNs; that statement must not be silently extended to every expert subnetwork
without checking the relevant implementation.

**Loss function(s):** Autoregressive token cross-entropy, with expert-load
balancing in the sparse routing formulation.

**Optimization algorithm(s):** The report specifies **Adafactor**, no
first-moment accumulation, factored second moments, and update clipping.
Its learning rate is initially **0.01 for 10,000 steps**, followed by
inverse-square-root decay. These are historical GLaM settings, not a generic
recommendation for arbitrary models.

**Regularization techniques:** Data filtering and careful training-mixture
design; the reported pretraining **dropout is zero**. Normalization and
router-load controls remain present. Zero dropout does not mean an absence
of implicit or data-based regularization.

**Backpropagation considerations:** Selected experts and differentiable gate
weights receive gradients. Routing balance, low-precision activation
stability, and all-to-all synchronization affect useful training progress.

**Parameter count / scaling behavior:** The largest sparse model has about
**1.2T total and 96.6B active parameters per token**, with **64 experts** in
each MoE layer and two selected. The name 64B/64E is not its active parameter
count.

**Training paradigm:** From-scratch self-supervised pretraining followed by
zero-/one-/few-shot evaluation. In-context examples are not weight updates.

**Hardware/parallelism considerations:** The largest report uses **1,024
TPU-v4 chips**, two-dimensional weight/activation sharding, float32 weights,
and bfloat16 activations. Energy comparisons must retain these infrastructure
qualifications.

### 3.15.3 Mixtral 8x7B

**Name:** Mixtral 8x7B; Mixtral 8x7B Instruct is its separately adapted
instruction-following release.

**Category & sub-category:** Unsupervised/self-supervised autoregressive
language pretraining; sparse decoder Transformers.

**Originating paper/vendor/year:** Mistral AI, December 2023 release;
Albert Q. Jiang and colleagues,
[Mixtral of Experts](https://arxiv.org/html/2401.04088v1), January 2024.

**Core mechanism:** Each FFN is replaced by eight independently weighted
SwiGLU experts. A token-dependent router selects two, normalizes their selected
scores, and adds their weighted outputs. Attention is shared across the
experts. Experts are jointly trained subnetworks, not eight independently
trained chatbots.

**Inputs/outputs and typical data types:** Multilingual text and code;
next-token distributions, completions, or instruction-following responses
after adaptation.

**Strengths and limitations:** Strong capability at a lower active parameter
budget than large dense contemporaries, with public weights. All expert
weights still need to be stored or accessed. Performance depends on kernels,
batching, and hardware; the model name does not mean 56B independent weights.

**Computational complexity / scalability notes:** Top-2 expert FFN
arithmetic, dense attention, and routing. The paper explicitly notes that
active-parameter comparisons omit memory and utilization costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Mistral evaluated multilingual
knowledge and code capability as an alternative to much larger dense
inference models. In the paper's **5-shot MMLU** protocol, question text and
examples become scored answer choices: Mixtral 8x7B records **70.6%**, versus
**69.9%** for the authors' re-evaluated Llama 2 70B. This demonstrates the
research quality/active-compute tradeoff, not a measured customer productivity
gain. The
[release announcement](https://mistral.ai/news/mixtral-of-experts)
also documents its historical use behind the `mistral-small` endpoint; that
2023 mapping must not be assumed current.

**Notable vendor implementations/libraries:** Mistral reference code,
Hugging Face Transformers, vLLM/MegaBlocks kernels, and compatible serving
engines, subject to version and quantization support.

**Architecture diagram description:** `tokens -> 32 decoder blocks
[RMSNorm -> causal GQA -> residual; RMSNorm -> router -> two of eight SwiGLU
experts -> weighted sum -> residual] -> token head`. Full attention supports
the reported 32K context; do not copy Mistral 7B's sliding-window setting
without checking the Mixtral configuration.

**Activation functions used and why:** SiLU within SwiGLU experts provides
smooth multiplicative gating; attention and expert-selection weights use
softmax.

**Loss function(s):** Next-token cross-entropy. Implementations can expose
router auxiliary losses; the short report does not disclose a complete
pretraining coefficient schedule.

**Optimization algorithm(s):** AdamW with warmup and decay is a typical
adaptation recipe. The full original pretraining optimizer schedule is not
disclosed by the short release paper, and is not inferred here.

**Regularization techniques:** RMS normalization, curated data, and
fine-tuning choices; dropout and balancing settings must be read from the
selected training configuration rather than invented.

**Backpropagation considerations:** Gradients flow through selected SwiGLU
experts and normalized gate scores, not through hard top-2 index decisions.
Uneven routing can starve experts or waste capacity.

**Parameter count / scaling behavior:** **46.7B total / 12.9B active per
token**, as reported by Mistral. Every decoder FFN is an MoE layer. Shared
attention explains why eight-times-seven is not the total.

**Training paradigm:** From-scratch pretraining; the Instruct variant uses
supervised fine-tuning and **direct preference optimization**, not an
automatically assumed PPO/RLHF recipe.

**Hardware/parallelism considerations:** Dense/tensor parallel attention
can be combined with expert parallel FFNs. The roughly **93.4 GB** decimal
BF16 weights-only calculation is an illustrative storage lower bound, not
the complete runtime memory requirement.

### 3.15.4 Mixtral 8x22B

**Name:** Mixtral 8x22B; base and instructed April 2024 releases are distinct.

**Category & sub-category:** Unsupervised/self-supervised autoregressive
pretraining with a sparse decoder Transformer; instruction tuning is
supervised adaptation.

**Originating paper/vendor/year:** Mistral AI,
[April 2024 release](https://mistral.ai/news/mixtral-8x22b).

**Core mechanism:** Scale the Mixtral pattern to a larger shared decoder
and eight FFN experts per block, selecting two per token. The objective still
learns conditional token probabilities rather than selecting a complete
expert chatbot for an entire request.

**Inputs/outputs and typical data types:** Multilingual text, code, longer
documents, and structured/function-call responses under appropriate
instruction and serving formats.

**Strengths and limitations:** More capacity and a reported 64K context than
8x7B. Its large resident weight footprint can dominate low-concurrency serving;
more context also raises attention/cache costs.

**Computational complexity / scalability notes:** Two active experts per
block, but larger width and depth increase arithmetic relative to 8x7B.
Keeping top-k fixed does not keep compute fixed when the experts themselves
grow.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Mistral tested mathematical
problem solving using **GSM8K** and **MATH**. A written problem becomes a
generated solution and extracted answer. The instructed release reports
**90.8% on GSM8K with maj@8** and **44.6% on MATH with maj@4**. Multiple
samples and majority voting are part of these results; they are not
single-sample accuracy or evidence of classroom learning gains. The
organization's stated design rationale is capability at a favorable active
compute budget, while the independent dense-vs-MoE production cost
counterfactual remains unreported.
[Source](https://mistral.ai/news/mixtral-8x22b).

**Notable vendor implementations/libraries:** Mistral's platform and
released weights; Hugging Face Transformers and compatible MoE serving
runtimes.

**Architecture diagram description:** `tokens -> 56 residual decoder
blocks [GQA; top-2-of-8 SwiGLU FFN] -> vocabulary head`.
The [base configuration](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/config.json)
specifies width **6,144**, **48 query heads**, **8 KV heads**, and no
sliding-window setting.

**Activation functions used and why:** SiLU in gated expert MLPs; softmax
for attention and routing. The gate allows feature-dependent modulation
rather than only a pointwise nonlinearity.

**Loss function(s):** Autoregressive cross-entropy; a router auxiliary
coefficient is available in the released configuration. A configuration
default is not proof of every historical training stage's exact loss recipe.

**Optimization algorithm(s):** AdamW-style warmup/decay is typical for
fine-tuning. The release announcement does not disclose the complete original
optimization and post-training schedule.

**Regularization techniques:** RMSNorm and data/recipe-dependent
regularization. The cited base configuration has zero attention dropout;
that is not evidence that every possible adaptation should use zero dropout.

**Backpropagation considerations:** Top-2 assignment discontinuities,
expert-load skew, and large activation tensors; gradient clipping and careful
mixed precision are practical adaptation safeguards.

**Parameter count / scaling behavior:** **141B total / 39B active per
token**, rounded vendor figures. All 56 blocks have the Mixtral expert FFN
structure.

**Training paradigm:** Base pretraining followed by a separately released
instruction-tuned model. A complete alignment algorithm is not inferred
solely from the word "Instruct."

**Hardware/parallelism considerations:** Weights-only BF16 storage is
approximately **282 GB** decimal, before KV cache and buffers. Multi-device
tensor/expert parallelism or suitable quantization is usually necessary.

### 3.15.5 DeepSeekMoE 16B

**Name:** DeepSeekMoE 16B; distinct from DeepSeek-V2-Lite and later V2/V3.

**Category & sub-category:** Unsupervised/self-supervised autoregressive
language modeling; fine-grained expert specialization.

**Originating paper/vendor/year:** Damai Dai and colleagues, DeepSeek,
[DeepSeekMoE](https://arxiv.org/abs/2401.06066), January 2024.

**Core mechanism:** Split expert FFNs into finer-grained units and select
more of the smaller units. Keep a separate set of shared experts always
active, intending to represent common knowledge while routed experts learn
more differentiated transformations. Specialization is an empirical tendency,
not a guarantee of interpretable subject labels.

**Inputs/outputs and typical data types:** English and Chinese text and
code; token distributions and generated continuations.

**Strengths and limitations:** High parameter capacity at a modest active
compute budget and a public, comparatively accessible checkpoint. Small
experts may have inefficient kernel utilization, and shared experts do not
eliminate routing imbalance.

**Computational complexity / scalability notes:** Compute depends on the
sum of the selected small experts and always-active shared experts, not
simply the number selected. Increasing expert granularity changes dispatch
and matrix-size efficiency.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** DeepSeek's controlled comparison
trained the sparse 16B model and its dense DeepSeek 7B reference on the
**same 2T-token corpus**, evaluating text knowledge, mathematics, and code.
Prompts become answers or programs scored by the study's benchmark suite.
The release reports broadly comparable capability using **40.5% of the
computation** of DeepSeek 7B in its internal comparison. This supports the
authors' fine-grained/shared-expert design rationale, but is not a measured
40.5% API bill or proof that each individual task matches.
[Official results and explanation](https://github.com/deepseek-ai/DeepSeek-MoE).

**Notable vendor implementations/libraries:** DeepSeek's released model
code, Hugging Face integrations, and DeepSpeed-based fine-tuning examples.
Exact runtime support depends on model-code version.

**Architecture diagram description:** `tokens -> one dense decoder block
-> remaining blocks [attention; two shared expert equivalents + top-6-of-64
routed small experts; residuals] -> token head`. The
[released configuration](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/config.json)
has 28 layers and width 2,048.

**Activation functions used and why:** SiLU-based gated expert MLPs,
softmax routing, and softmax attention.

**Loss function(s):** Next-token cross-entropy with expert-balancing
objectives in the research design; instruction-response cross-entropy for
the Chat adaptation.

**Optimization algorithm(s):** AdamW-family optimization is appropriate
to this decoder training regime. The released fine-tuning example uses
warmup and cosine decay; its illustrative learning rate must not be
misreported as the original pretraining schedule.

**Regularization techniques:** RMSNorm, training-data construction, and
load balancing; attention dropout is zero in the cited released configuration.
Fine-tuning may require different regularization.

**Backpropagation considerations:** Shared experts receive broad task
gradients; routed experts receive selected subsets. Fine granularity can
increase specialization but does not guarantee that every expert receives
sufficient diverse gradients.

**Parameter count / scaling behavior:** **16.4B total, approximately
2.8B active** in the reported 16B design. Two shared and six selected routed
experts are active in its MoE blocks, whose expert widths are smaller than
a conventional dense FFN.

**Training paradigm:** From-scratch pretraining on 2T tokens, then a
separate supervised Chat fine-tune.

**Hardware/parallelism considerations:** The official release states that
the 16B model can be served on a **40 GB GPU without quantization** under
its supported inference conditions. This is not a guarantee for arbitrary
batch sizes or context lengths. Training requires substantially more memory
than weights-only inference.

### 3.15.6 DeepSeek-V2

**Name:** DeepSeek-V2, May 2024 full-size model; DeepSeek-V2-Lite is a
separate architecture scale.

**Category & sub-category:** Unsupervised/self-supervised language
pretraining; sparse experts plus compressed attention. Chat variants add
supervised and reinforcement-learning stages.

**Originating paper/vendor/year:** DeepSeek-AI,
[DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434), 2024.

**Core mechanism:** Combine fine-grained/shared DeepSeekMoE FFNs with
**Multi-head Latent Attention (MLA)**. MLA stores a compressed joint
key/value representation, reducing decoding cache requirements; MoE reduces
the expert arithmetic activated per token. These are distinct mechanisms.

**Inputs/outputs and typical data types:** English/Chinese language and
code prompts; continuations or chat responses, with long-context variants.

**Strengths and limitations:** Improves the coupled capacity/compute/cache
tradeoff. Efficient serving requires MLA-aware kernels and suitable
parallelism; a generic attention implementation may fail to realize the
published gains.

**Computational complexity / scalability notes:** Reduced KV storage does
not remove all context-dependent attention work. Routed expert FFNs retain
communication overhead and the full weight-storage requirement.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Sourced application.** DeepSeek released V2 through
its chat website and API to provide economical language/code assistance.
Requests are tokenized, processed by MLA and routed FFNs, and decoded into
responses. The
[official release](https://github.com/deepseek-ai/DeepSeek-V2)
documents the service and reports, **against DeepSeek 67B**, **42.5%
lower training cost, 93.3% less KV cache, and 5.76x maximum generation
throughput** in the authors' comparison. These are **system/research
measurements accompanying a deployed model**, not a measured customer
business KPI. They reflect MoE, MLA, and implementation together; attributing
all three gains to routing alone would be wrong.

**Notable vendor implementations/libraries:** DeepSeek's reference
implementation, Hugging Face model code, vLLM, and SGLang with appropriate
MLA/version support.

**Architecture diagram description:** `tokens -> 60 residual decoder
blocks [MLA; first-block dense FFN, then two shared experts +
top-6-of-160 routed experts] -> token head`, according to the
[released full-size configuration](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json).

**Activation functions used and why:** SiLU-gated MLPs, softmax router
scores, and attention normalization; RMS normalization stabilizes the
compressed and residual representations.

**Loss function(s):** Next-token prediction with routing-balancing
objectives; supervised response loss and RL objectives in identified
post-training variants.

**Optimization algorithm(s):** The
[V2 report](https://arxiv.org/html/2405.04434v1)
specifies **AdamW** with moment coefficients 0.9 and 0.95 and weight
decay 0.1. It warms up over 2,000 steps to **$`2.4\times10^{-4}`$**,
then applies step decay at approximately 60% and 90% of training tokens.
Context-extension and post-training schedules are separate.

**Regularization techniques:** RMSNorm, data curation, expert/device
balancing, and routing constraints. The public inference configuration is
not a complete record of every training regularizer.

**Backpropagation considerations:** Gradient flow must pass through MLA's
compression projections and selected expert paths. Expert/device imbalance
and unstable gate scores can negate capacity gains; clipping and precision
management remain important.

**Parameter count / scaling behavior:** **236B total / 21B active per
token** for full-size V2. **V2-Lite is 16B / 2.4B** and must not be used
as the configuration source for the 236B model.

**Training paradigm:** Pretraining on **8.1T tokens**, context extension,
SFT, and identified RL Chat variants. Base and Chat benchmarks are not
interchangeable.

**Hardware/parallelism considerations:** MLA-aware KV caching, expert
parallelism, and dense-operation parallelism. The release's example BF16
deployment specifies **eight 80 GB GPUs**; this is an implementation
configuration, not a universal minimum.

### 3.15.7 DeepSeek-V3

**Name:** DeepSeek-V3, December 2024 original release; base, post-trained,
and subsequent dated updates require separate provenance.

**Category & sub-category:** Unsupervised/self-supervised autoregressive
and multi-token pretraining; sparse decoder MoE with MLA.

**Originating paper/vendor/year:** DeepSeek-AI,
[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1), 2024.

**Core mechanism:** Retain MLA and fine-grained/shared experts, but adjust
per-expert routing biases based on observed load rather than relying mainly
on a batch-level auxiliary balancing gradient. Bias affects selection; the
underlying affinity determines the mixing weight. Multi-token prediction
adds a future-token training objective. The report still includes a small
**sequence-wise auxiliary balancing loss**.

**Inputs/outputs and typical data types:** Multilingual text and code;
next-token continuations or post-trained assistant responses.

**Strengths and limitations:** Strong capacity and carefully co-designed
training efficiency. Reproduction and deployment require substantial
distributed expertise, and headline training cost excludes broader R&D.
The phrase "auxiliary-loss-free" must not be read as "no auxiliary losses
of any kind."

**Computational complexity / scalability notes:** Eight small routed
experts plus one shared expert per MoE block; MLA cache compression and
overlapped communication change realized efficiency. Auxiliary prediction
modules can add checkpoint storage beyond the main model.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Sourced application.** DeepSeek released V3 on its
chat and API platforms for language, coding, and reasoning workloads. The
documented training path turns **14.8T pretraining tokens** into a base
model, then adapts it for user responses. The report records **2.788 million
H800 GPU-hours** for official training: 2.664M for pretraining, 119K for
context extension, and 5K for post-training. At the report's **assumed
\$2 per GPU-hour**, this is **\$5.576M**, **excluding prior research and
ablation experiments**. These are training-resource figures accompanying a
deployed model, not total company expenditure or a customer cost-per-answer
measurement. The design rationale is explicitly economical capability through
joint algorithm, framework, and hardware design.
[Report](https://arxiv.org/html/2412.19437v1);
[release](https://github.com/deepseek-ai/DeepSeek-V3).

**Notable vendor implementations/libraries:** DeepSeek reference inference,
SGLang, vLLM, LMDeploy, and hardware-vendor runtimes supporting the precise
V3/MLA/precision configuration.

**Architecture diagram description:** `tokens -> residual MLA decoder
blocks -> first three dense FFNs, then [one shared expert +
top-8-of-256 routed experts] -> LM head`; an additional multi-token
prediction module participates in training and optional speculative use.

**Activation functions used and why:** SiLU-based expert gating, attention
softmax, and sigmoid-based expert affinities in V3's routing formulation;
RMSNorm stabilizes residual/compressed paths.

**Loss function(s):** Autoregressive cross-entropy, multi-token prediction
loss, and a small sequence-wise balance term. SFT and RL objectives belong
to post-training, not the entire pretraining run.

**Optimization algorithm(s):** The report specifies **AdamW**
($`\beta_1=0.9,\beta_2=0.95`$, weight decay 0.1), warmup to
**$`2.2\times10^{-4}`$** over 2,000 steps, a long constant phase,
cosine decay, and lower-rate final stages.

**Regularization techniques:** Weight decay, data curation, load-control
bias updates, a small sequence balance penalty, and normalization.

**Backpropagation considerations:** Mixed-precision training retains
selected accumulations/master states in higher precision. The reported
gradient clipping norm is **1.0**. Routing-bias updates based on load must
not be confused with ordinary task-loss backpropagation.

**Parameter count / scaling behavior:** **671B main-model total / 37B
active per token**. The official release notes **685B** downloaded
parameters when **14B of multi-token prediction weights** are included.
Those figures answer different counting questions.

**Training paradigm:** From-scratch pretraining, context extension, SFT,
and RL; the report describes reasoning-data distillation from its R1-related
teacher work.

**Hardware/parallelism considerations:** **2,048 H800 GPUs** in the
reported cluster, FP8 mixed-precision computation, expert/pipeline/data
parallelism, DualPipe scheduling, and communication overlap. The report's
training design avoids costly tensor parallelism; it is not correct to
assume every large model uses every parallelism type.

### 3.15.8 Grok-1

**Name:** Grok-1, specifically the base checkpoint released in March 2024,
whose pretraining concluded in October 2023.

**Category & sub-category:** Unsupervised/self-supervised autoregressive
language modeling; sparse decoder MoE. This entry is not a disclosure claim
about later Grok models.

**Originating paper/vendor/year:** xAI, 2023 development and
[2024 open-weight release](https://x.ai/news/grok-os).

**Core mechanism:** A decoder uses a learned router to select two of eight
expert MLPs per token, with shared attention. The released code provides
enough architectural detail for inference but not a complete training recipe.

**Inputs/outputs and typical data types:** Text prompts and generated token
continuations. The released base checkpoint is not itself the full
search-connected Grok product.

**Strengths and limitations:** Public large-scale MoE weights and a
readable reference implementation. The implementation is explicitly not
optimized for MoE performance; datasets, optimization details, and some
parameter-accounting conventions remain insufficiently disclosed.

**Computational complexity / scalability notes:** Sparse FFN execution
does not eliminate the storage burden of 314B parameters. The official
reference may perform extra work for simplicity, so its runtime is not a
clean measurement of ideal sparse arithmetic.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** xAI's original Grok-1 capability
study evaluated code completion on **HumanEval**, using a function prompt
to produce Python code and scoring whether it passed the benchmark's
tests. The
[November 2023 announcement](https://x.ai/news/grok)
reports **63.2% zero-shot pass@1**, and **73.0% 5-shot MMLU** for
knowledge questions. Sparse capacity is the architectural rationale for
its compute class; no controlled production saving is disclosed. These
are historical **Grok-1 family reports**, not a new evaluation of the
downloaded base checkpoint or later live assistant versions.

**Notable vendor implementations/libraries:** xAI's
[JAX/Haiku reference repository](https://github.com/xai-org/grok-1)
and released weights; derivative runtime compatibility must be checked.

**Architecture diagram description:** `tokens -> 64 residual decoder
layers [GQA; top-2-of-8 expert MLPs] -> token head`. The release lists
width **6,144**, **48 query heads**, **8 KV heads**, and **8,192-token**
maximum context.

**Activation functions used and why:** The released
[model code](https://github.com/xai-org/grok-1/blob/main/model.py)
uses GELU in a multiplicatively gated expert MLP and float32 softmax
router probabilities. It is not simply a copied SwiGLU Mixtral block.

**Loss function(s):** Autoregressive next-token prediction is publicly
described. Exact original auxiliary losses and weights are not disclosed
by the inference release.

**Optimization algorithm(s):** The actual optimizer and learning-rate
schedule are **not publicly specified by the cited release**. A plausible
AdamW recipe is not evidence of xAI's historical recipe.

**Regularization techniques:** The code exposes normalization and
inference structure; training dropout, decay, and other regularization
settings are not established here.

**Backpropagation considerations:** Selected experts and gate weights
can be differentiated; hard routing is discontinuous. Higher-precision
router calculations are visible in the inference code, but are not a
complete account of historical training stability.

**Parameter count / scaling behavior:** **314B total**. xAI states
**25% of weights active per token**. Multiplying those rounded statements
gives approximately **78.5B**, but this is a **derived approximation**,
not a separately audited active-weight inventory; shared-weight counting
can change the interpretation.

**Training paradigm:** From-scratch pretraining on text; the released base
checkpoint is explicitly not dialogue fine-tuned.

**Hardware/parallelism considerations:** Large multi-accelerator memory,
JAX sharding, and optional released 8-bit weight support. The reference
code's deliberate inefficiency should not be marketed as production serving
performance.

### 3.15.9 DBRX

**Name:** DBRX Base and DBRX Instruct.

**Category & sub-category:** Unsupervised/self-supervised autoregressive
pretraining; fine-grained sparse decoder MoE, followed by instruction
adaptation.

**Originating paper/vendor/year:** Databricks Mosaic research,
[March 2024 release and technical account](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).

**Core mechanism:** Use sixteen relatively small experts and select four
per token, rather than eight larger experts with top-2 routing. Shared
attention, RoPE, GQA, and gated expert MLPs support a general-purpose
text/code model.

**Inputs/outputs and typical data types:** Text, code, questions with
retrieved passages, and instruction-following responses.

**Strengths and limitations:** A strong historical open-weight
quality/serving tradeoff with extensive vendor-described infrastructure.
Its results also reflect data, tokenization, and optimization improvements,
not MoE alone. The release license is a separate question from weight
availability.

**Computational complexity / scalability notes:** Four active expert
FFNs per token. Small expert matrices and dispatch overhead affect hardware
utilization. Batch/sequence lengths and quantization alter serving outcomes.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Databricks evaluated enterprise-style
retrieval-augmented question answering on **Natural Questions**: retrieve
the **top ten Wikipedia passages with bge-large-en-v1.5**, place them with
the question in the prompt, generate an answer, and evaluate answer matching.
DBRX Instruct achieves **60.0%** under that protocol versus **57.7%**
for the compared GPT-3.5 Turbo endpoint. This is a concrete grounded-QA
research case, not a customer support cost reduction. The same release
documents API availability and early internal SQL-product integration, with
no audited SQL business KPI. The rationale for sparse experts is higher
capacity with reduced active compute; retrieval and data quality also
contribute.
[Source: release Table 4 and deployment discussion](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).

**Notable vendor implementations/libraries:** Databricks Model Serving,
LLM Foundry, Composer, MegaBlocks, Hugging Face integrations, and
NVIDIA TensorRT-LLM in the reported serving comparison.

**Architecture diagram description:** `tokens -> residual decoder
blocks [GQA + RoPE; top-4-of-16 gated expert FFN] -> token head`.
Expert substitution occurs through the decoder rather than only at the
final output.

**Activation functions used and why:** Gated linear units provide
feature-dependent expert transformations; softmax normalizes attention and
router scores. The precise GLU implementation should be read from the
released model/configuration, not inferred from another vendor.

**Loss function(s):** Next-token cross-entropy plus recipe-dependent
routing regularization; instruction adaptation uses supervised response
training and other disclosed post-training procedures.

**Optimization algorithm(s):** Databricks describes a jointly optimized
training recipe; the release blog does not establish all optimizer
hyperparameters. AdamW-style warmup/decay is common in LLM Foundry
adaptations, not proof of every historical DBRX setting.

**Regularization techniques:** Curated and curriculum-scheduled data,
normalization, and MoE training controls. Claims of efficiency must retain
these accompanying changes.

**Backpropagation considerations:** Fine-grained expert batches benefit
from sparse kernels. Load imbalance, distributed synchronization, and
precision issues affect convergence; selected routes determine which
experts receive task gradients.

**Parameter count / scaling behavior:** **132B total / 36B active per
token**, with **16 experts and top-4 routing**.

**Training paradigm:** Pretraining on **12T tokens of text and code**,
followed by a separate instruction version; the public maximum training
context is 32K tokens.

**Hardware/parallelism considerations:** The release reports **3,072
NVIDIA H100s** for training. Reported inference comparisons use optimized
TensorRT-LLM and specified traffic/precision; see
[Section 4.8](08-moe-deep-dive.md) rather than assuming a universal speedup.

### 3.15.10 Snowflake Arctic

**Name:** Snowflake Arctic Base and Arctic Instruct, the April 2024
large language model; not every later product carrying "Arctic."

**Category & sub-category:** Unsupervised/self-supervised decoder
pretraining; dense-plus-sparse hybrid MoE with enterprise-oriented adaptation.

**Originating paper/vendor/year:** Snowflake AI Research,
[April 24, 2024 release](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/).

**Core mechanism:** Combine an always-active dense model with a large
residual expert component. A top-2 router selects from 128 expert choices.
The dense path carries broadly useful computation while sparse capacity
targets an enterprise-focused quality/compute tradeoff.

**Inputs/outputs and typical data types:** Natural-language requests,
database schemas, code, and instruction text; generated SQL, code, or
responses after suitable adaptation.

**Strengths and limitations:** A documented focus on SQL, coding, and
instruction following rather than only broad language leaderboards. The
large total footprint remains expensive to host; a favorable composite
benchmark does not guarantee correctness on a new schema or business query.

**Computational complexity / scalability notes:** Active computation
includes the dense path **and** selected experts. Long contexts, buffering,
and uneven expert load add costs beyond the nominal active count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Snowflake designed Arctic for
enterprise SQL/code copilots and evaluated SQL generation using **Spider**:
a natural-language question and schema become a SQL query whose output is
evaluated by the dataset protocol. The organization's
[technical repository](https://github.com/Snowflake-Labs/snowflake-arctic)
states that its selected enterprise metrics were on par with Llama 3 70B
while using **17x less training compute**. That is a **vendor-reported
training-budget comparison across recipes**, not a measured 17x SQL-query
cost saving or a customer deployment KPI. The explicit selection rationale
is enterprise task capability under a constrained training budget; schema
validation and controlled execution remain necessary.

**Notable vendor implementations/libraries:** Snowflake's released
weights and inference examples, Hugging Face model code, vLLM, and
associated DeepSpeed-oriented systems work. Later Arctic-branded models
must be identified separately.

**Architecture diagram description:** `tokens -> 35 residual hybrid
decoder layers [attention/dense path + top-2-of-128 expert MLP path] ->
token head`. The
[base configuration](https://huggingface.co/Snowflake/snowflake-arctic-base/blob/main/config.json)
uses an MoE frequency of one and a parallel attention/MLP residual design.

**Activation functions used and why:** SiLU-based expert MLPs, softmax
routing/attention, and RMSNorm; the nonlinear gate increases expressiveness
of the expert transformation.

**Loss function(s):** Next-token prediction with routing regularization;
the public configuration exposes a router auxiliary coefficient.
Instruction-specific losses belong to the adapted checkpoint.

**Optimization algorithm(s):** AdamW with warmup and decay is a typical
Transformer adaptation choice, not a claim about every original Arctic
setting. The repository introduction does not disclose every optimizer
hyperparameter; use the released training cookbook for a particular
reproduction.

**Regularization techniques:** Data-mixture design, normalization, and
router balancing. The cited configuration also has capacity and token-dropping
settings; one must not assume that every Arctic runtime is dropless.

**Backpropagation considerations:** The dense residual path supplies
gradient flow even when expert routing is sparse. Expert starvation,
overflow, and reduced precision still require monitoring.

**Parameter count / scaling behavior:** **480B total / 17B active**,
rounded vendor figures. The vendor describes a **10B dense model** plus
a residual **128 x 3.66B** expert component; arithmetic on rounded
components need not exactly equal the rounded total.

**Training paradigm:** Enterprise-oriented base pretraining and separate
instruction tuning; no assumption that every training token is a labeled
SQL example.

**Hardware/parallelism considerations:** Multi-accelerator storage and
expert/tensor sharding; about **960 GB** decimal BF16 main weights by a
simple illustrative count, before other runtime state. Sparse arithmetic
does not make this a small-memory model.

### 3.15.11 OLMoE

**Name:** OLMoE-1B-7B-0924, with separate SFT and Instruct derivatives.

**Category & sub-category:** Unsupervised/self-supervised language
pretraining; fine-grained, dropless sparse decoder MoE.

**Originating paper/vendor/year:** Niklas Muennighoff and collaborators,
Allen Institute for AI and collaborating institutions,
[OLMoE](https://arxiv.org/abs/2409.02060), September 2024.

**Core mechanism:** Use many small experts and select eight of 64 for
each token. Dropless sparse execution avoids capacity-based omission of
assigned token computation. Open data, code, checkpoints, and logs make
routing and training design choices more inspectable than weights-only
releases.

**Inputs/outputs and typical data types:** Text/code sequences;
continuations, benchmark answer probabilities, and adapted assistant
responses.

**Strengths and limitations:** Reproducibility artifacts and a relatively
small active footprint support research on sparse training. Its total
memory is still several times the active footprint, and dropless execution
does not imply unlimited memory or uniformly fast kernels.

**Computational complexity / scalability notes:** Eight small expert
evaluations per token, full attention/router costs, and dynamic sparse
packing. Performance depends on actual expert token counts and block-sparse
kernel support.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** AI2's study addresses the
scientific problem of **reproducible MoE training and analysis**, releasing
OLMoE-mix-0924 data, pretraining checkpoints/logs, and evaluation code.
Text batches train the model; held-out language and downstream QA tasks
measure its predictions. The
[dated base model card's evaluation snapshot](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
reports **54.1 on MMLU**, compared with **48.5 for DCLM-1B** in that
snapshot, with **1.3B versus 1.4B active parameters**. These are
release-protocol benchmark results, not a universal cross-harness ranking.
The [official repository](https://github.com/allenai/OLMoE) also publishes
routing-analysis artifacts and comparisons of design choices, addressing
the practical scientific problem of inspecting rather than merely trusting
sparse-model claims. **No audited production business KPI is reported.**

**Notable vendor implementations/libraries:** AI2 OLMo/Open Instruct,
MegaBlocks, Hugging Face Transformers, vLLM, SGLang, and llama.cpp
integrations identified in the release.

**Architecture diagram description:** `tokens -> 16 residual decoder
blocks [attention with normalization; top-8-of-64 SwiGLU FFN] -> output
head`, with width **2,048** and **16 attention heads**, according to
the [released training configuration](https://github.com/allenai/OLMoE/blob/main/configs/OLMoE-1B-7B-0924.yml).

**Activation functions used and why:** SwiGLU experts, RMS normalization,
and normalized attention/router scores; multiplicative gating supports
expressive small FFNs.

**Loss function(s):** Token cross-entropy, router balance loss, and router
z-loss. The cited configuration sets balance weight **0.01** and
z-loss weight **0.001**.

**Optimization algorithm(s):** **AdamW**, peak learning rate
**$`4\times10^{-4}`$** in the cited configuration, with token-based
warmup and cosine decay. The release also describes a separate final
annealing configuration.

**Regularization techniques:** Weight decay **0.1**, RMSNorm, data
filtering, and router penalties; attention, residual, and embedding dropout
are zero in that configuration.

**Backpropagation considerations:** Global gradient norm clipping at
**1.0** in the release, stable router normalization, and sparse expert
gradients. Dropless execution avoids a capacity-drop gradient omission but
does not cure poor expert specialization.

**Parameter count / scaling behavior:** **6.9B total / 1.3B active**
despite the rounded 1B-7B name. All 16 blocks use the MoE configuration;
eight of 64 routed experts are selected.

**Training paradigm:** From-scratch open pretraining; separately released
SFT and preference adaptations include DPO/KTO experiments. Do not combine
all adaptation methods into one presumed checkpoint recipe.

**Hardware/parallelism considerations:** The cited configuration uses
BF16 mixed precision, FSDP full sharding, and sparse MegaBlocks execution.
Expert parallelism is a possible systems design, not an automatic claim
about every released training run.

#### Comparative summary: sparse MoE language models

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Switch Transformer | Text-to-text tasks | Simple top-1 conditional capacity | Huge weight footprint and transfer sensitivity | Google Natural Questions research comparison |
| GLaM | General language prompts | Demonstrated quality/compute tradeoff | Research infrastructure and cross-study cost caveats | 29-task NLP study with reported FLOP/energy comparison |
| Mixtral 8x7B | Multilingual text and code | Public weights; 12.9B active | Routing and memory overhead | 5-shot MMLU; historical Mistral API deployment |
| Mixtral 8x22B | Larger text/code contexts | Greater capacity and 64K context | 141B total weight footprint | GSM8K/MATH majority-vote evaluations |
| DeepSeekMoE 16B | Text/code at smaller active scale | Fine-grained plus shared experts | Small-expert utilization and balance | Same-corpus comparison against dense DeepSeek 7B |
| DeepSeek-V2 | Language/code and longer contexts | MLA plus sparse FFN efficiency | Gains require suitable serving kernels | DeepSeek API and disclosed system comparisons |
| DeepSeek-V3 | Large-scale language/code assistance | Co-designed sparse/FP8 training | Major infrastructure and cost-accounting caveats | Deployed platform with reported training-resource ledger |
| Grok-1 | Research on released large MoE weights | Public architecture and checkpoint | Incomplete training recipe; unoptimized reference | xAI HumanEval/MMLU research report |
| DBRX | Enterprise text/code and RAG | Fine-grained experts and documented serving | Data/recipe improvements confound pure-MoE attribution | Natural Questions RAG; early SQL-product integration |
| Snowflake Arctic | SQL/code/instruction tasks | Explicit enterprise-focused design | Very large resident model | Spider-centered enterprise research comparison |
| OLMoE | Reproducible sparse-model research | Data, code, checkpoints, and logs | Active size understates memory | AI2 open training and routing-analysis study |

## Coverage and continuation manifest

This chapter supplies individual, fully schematized entries for eleven
representative self-supervised MoE language models/families. The original
supervised mixtures and GShard are covered in
[the supervised neural chapter](02-supervised-neural.md).

Continue with [Part 4: the MoE deep dive](08-moe-deep-dive.md) for routing,
balancing, expert placement, a consolidated parameter ledger, training and
inference costs, and a carefully qualified production-oriented case.
The [foundation-model chapter](06-foundation-models.md) covers dense models
and proprietary families, including publicly disclosed Gemini MoE information
without invented parameter counts.

Further extensions include a full vision/audio MoE catalog, multimodal and
task-routed expert systems, additional Qwen-family sparse releases, newer
Grok/DeepSeek checkpoints, and standardized independently reproduced serving
comparisons. Mentioning these extensions does not claim to have cataloged
every release.
