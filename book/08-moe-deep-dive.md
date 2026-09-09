# 4. Mixture of Experts: A Dedicated Deep Dive

**The main idea:** a large model can choose which small parts to use for
each input. Those parts are called **experts**, and the chooser is a
**router** or **gate**.

Think of selecting tools rather than using every tool for every job.
The analogy has a limit: an expert is a small learned network, not a named
skill or a separate chatbot. Its role is learned, not assigned a job title.

MoE separates two costs. **Total parameters** are all the learned numbers
stored in the model. **Active parameters** are the numbers used for one
input under a stated counting rule. Fewer active numbers can mean less
calculation, but the other numbers still need to be available.

The original supervised mixtures and GShard are in
[Section 1.11](02-supervised-neural.md). Later self-supervised language
models are in [Section 3.15](07-moe-models.md). This chapter explains the
shared ideas and the limits of the cost comparisons.

## 4.1 From adaptive local experts to sparse language models

In [Adaptive Mixtures of Local Experts](https://doi.org/10.1162/neco.1991.3.1.79),
Robert Jacobs, Michael Jordan, Steven Nowlan, and Geoffrey Hinton (1991)
trained experts and a gate together. The gate learned which experts to favor
for a given input. An expert did not have to do equally well on every kind
of example.

The original model used supplied answers, so it was **supervised**.
Modern language models using the same broad idea do not change that history.

**Optional math:** one way to combine experts is to mix their probability
predictions:

$$
p(y\mid x)=\sum_{e=1}^{E}\pi_e(x)\,p_e(y\mid x),\qquad
\pi_e(x)\geq 0,\quad \sum_e\pi_e(x)=1.
$$

Here, $`x`$ is the input and $`y`$ a possible answer. There are $`E`$
experts. Expert $`e`$ gives probability $`p_e(y\mid x)`$, and the gate
gives it share $`\pi_e(x)`$. The shares cannot be negative and add to one.
Multiply each expert prediction by its share, then add the results.

Under a likelihood-based loss, training credit depends on two things:
the gate's share and how well the expert explained the correct answer.
This is not just averaging separately trained models. Also, scoring an
average prediction is not generally the same as scoring a mixture of
probability distributions.

| Period | Development | What changed |
|---|---|---|
| 1991 | Adaptive mixtures of local experts | Experts and a gate learned to divide a supervised task |
| 1994 | Hierarchical mixtures of experts | Gates were arranged in a branching hierarchy |
| 2017 | Sparsely gated neural MoE | Very large models could choose only a few experts per input |
| 2020 | GShard | Expert Transformers and automatic work splitting scaled multilingual translation |
| 2021-2022 | Switch and GLaM | Top-1 routing and large sparse language pretraining |
| 2022 | Expert-choice, ST-MoE, MegaBlocks | Better ways to balance work, stabilize learning, and avoid dropping assignments |
| 2023-2024 | Mixtral, DeepSeek, DBRX, Arctic, OLMoE, released Grok-1 | More expert-model designs became publicly described or usable |

Important sources include
[Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538),
[GShard](https://arxiv.org/abs/2006.16668),
[Switch](https://arxiv.org/abs/2101.03961), and
[GLaM](https://arxiv.org/abs/2112.06905).
This timeline highlights a few developments. It does not deny the other
research between them.

**Dense versus sparse.** A dense mixture runs every expert and mixes their
outputs. A sparse mixture runs only selected experts for a given input.
Both can learn different expert roles, but only the sparse version skips
the unchosen calculations.

An **ensemble** often combines separately trained models at the end.
An MoE layer may instead sit deep inside one network and learn jointly
with the rest of it.

## 4.2 Gating and routing mechanisms

A text model first turns each token, often a word part, into a feature vector:
a list of numbers. A learned router reads that list and gives each expert
a score. It chooses experts and combines their outputs.

**Optional math:** let $`h`$ be a token's list of $`d`$ features.
A router with weight array $`W_r`$ computes scores $`s=W_rh`$.
For a chosen set $`\mathcal{S}(h)`$, the output is

$$
z(h)=\sum_{e\in\mathcal{S}(h)}g_e(h)\,F_e(h).
$$

$`F_e(h)`$ is expert $`e`$'s output and $`g_e(h)`$ its contribution weight.
The equation says: multiply each chosen output by its weight, then add.
It does not say how to choose the experts or what happens when one is full.

| Routing mechanism | Selection rule | Benefit | Important limitation |
|---|---|---|---|
| Dense soft mixture | Run all experts and weight their outputs | All experts can receive ordinary learning signals | More experts mean more work |
| Learned top-k token choice | Each token selects its k highest-scoring experts | A fixed number of experts per token before capacity limits | Many tokens may choose the same experts |
| Noisy top-k | Add noise to learned scores before selecting | Helps try different experts during learning | Noise size and balancing still need tuning |
| Switch/top-1 | Select one expert per token | Simpler routing and less expert work | Capacity and gate-learning details still matter |
| Expert choice | Each expert selects a fixed-size group of tokens | Direct control of each expert's workload | Some tokens get several experts; others may get none |
| Hash routing | Use a fixed rule to map tokens to experts | No need to learn the routing rule | Frequent tokens may overload their assigned experts |
| Constrained or grouped routing | Limit choices by device or group | Less costly communication | Fewer expert combinations are available |

**What top-k means.** Top-2 chooses two experts; top-1 chooses one.
The router may use softmax to turn scores into contribution weights.
Some methods do this before selection, others after. That choice changes
how the gate learns.

**How learning passes through a choice.** A gradient is a local slope used
to update weights. Chosen experts and their smooth contribution weights can
receive gradients. But the winner's index jumps when two scores change rank.
Ordinary backpropagation does not calculate a smooth slope for that jump.

A **straight-through estimator** supplies a stand-in slope in some designs.
It is not automatically used by every top-k router.

In [noisy top-k gating](https://arxiv.org/abs/1701.06538), the router can
add Gaussian noise with a learned scale before choosing. This helps explore
experts and estimate expected loads. It does not make every hard choice smooth.

**A top-1 trap.** Suppose the selected expert's weight is always rescaled
to exactly one. That constant multiplier no longer gives the router its
usual task-loss gradient. Switch instead keeps the expert's probability
from the full softmax. A top-2 weighting rule therefore cannot be copied
unchanged into top-1 without checking how learning works.

**Expert choice.** [Zhou et al. (2022)](https://arxiv.org/abs/2202.09368)
reverse the usual choice: each expert picks tokens. Equal-size groups balance
expert work, but do not guarantee that every token is picked.

For example, two experts each choose two of four tokens. Both might pick
tokens 1 and 2, leaving 3 and 4 with no expert contribution.
Shortcut paths or extra coverage rules decide what happens to those tokens.

**Hash routing.**
[Hash Layers for Large Sparse Models](https://arxiv.org/abs/2106.04426)
uses a fixed mapping rather than a learned chooser. A common token still
appears often, so its assigned expert can still become busy. The mapping
and the data's token frequencies matter.

#### Worked routing calculation

This is a **teaching calculation, not a production case study**.
A router gives four experts probabilities $`(0.55,0.25,0.15,0.05)`$.
It selects the first two.

Their probabilities add to 0.80. Divide each by 0.80 to get new shares
$`(0.6875,0.3125)`$ that add to one. Suppose their output vectors are
$`(2,0)`$ and $`(0,4)`$.

**Optional math:**

$$
0.6875(2,0)+0.3125(0,4)=(1.375,1.25).
$$

Multiply the two numbers in each vector by its share, then add matching
positions. A residual block adds this expert result to the values carried
through its shortcut. These numbers are hidden features, not answer probabilities.
The router and shared attention also did work; they are not free.

## 4.3 Load balancing, capacity, overflow, and collapse

Choosing experts is only part of the problem. They must also share the load.
If most tokens choose one expert, that expert can become a bottleneck while
others sit idle.

An expert's **capacity** is the work it is allowed to accept in a batch.
A **capacity factor** adds room above a defined average load. Different
papers define that factor differently, so the same number can mean different
things.

**Optional math:** for one common token-choice rule,

$$
C=\left\lceil c\,\frac{Nk}{E}\right\rceil.
$$

$`N`$ is the token count, $`k`$ the experts chosen per token, and $`E`$
the total experts. $`Nk/E`$ is the average assignments per expert.
Multiply by capacity factor $`c`$, then round up to get capacity $`C`$.
Switch uses top-1, so $`k=1`$. Expert-choice papers may instead use their
factor to mean average experts per token.

**Teaching example.** Take 1,024 tokens, eight experts, top-2 routing,
and $`c=1.25`$. Average load is 256 assignments; capacity is 320.
If one expert gets 370 assignments, it has 50 too many.
There are 2,560 total slots for only 2,048 assignments, yet some tokens
still cannot use their first choices. Free space elsewhere does not help
unless the routing rule can use it.

**What happens to overflow?** Software can skip that expert contribution,
try another expert, favor some assignments, raise capacity, or use dropless
execution. A residual shortcut may still carry the token's values forward.
These choices can change what the model learns and produces.

"Token dropping" usually means skipping some **expert work**. It does not
mean removing the token from the entire text.

**How a balancing loss helps.** An extra penalty makes overloaded choices
less attractive during training. It uses both actual assignments and the
router's probabilities.

**Optional math:** one Switch-style top-1 balance loss is

$$
\mathcal{L}_{\mathrm{balance}}
  =\alpha E\sum_{e=1}^{E}f_eP_e,
\qquad
f_e=\frac{1}{N}\sum_{t=1}^{N}\mathbf{1}[\operatorname{route}(t)=e],
\qquad
P_e=\frac{1}{N}\sum_{t=1}^{N}p(e\mid h_t).
$$

$`f_e`$ is the fraction of tokens assigned to expert $`e`$.
$`P_e`$ is its average router probability. $`\alpha`$ controls the
penalty's strength, $`E`$ counts experts, and $`N`$ counts tokens.
The indicator $`\mathbf{1}`$ is one if the assignment matches, otherwise zero.
The probability term supplies a smooth learning signal; the hard assignment
indicator normally does not.

Other routers use other penalties and counting rules. Also, a balanced
whole-batch average can hide a badly overloaded device or local group.

**Why routing collapse happens.** A small early advantage can send an
expert more tokens. It gets more practice and improves, so the router keeps
choosing it. Others receive too little practice. Poor starting weights,
very similar batches, overly sharp scores, or unsuitable penalties can
strengthen this loop.

Perfectly equal work is not always the best learned division of roles.
But long-term starvation wastes expert capacity and can hurt speed.

**Balance and numerical stability are different.**
[ST-MoE](https://arxiv.org/abs/2202.08906) adds a **router z-loss**.
Its goal is to control the size of a router calculation, not to make every
expert equally busy.

**Optional math:**

$$
\mathcal{L}_z=\frac{\beta}{N}
\sum_{t=1}^{N}\left(\log\sum_{e=1}^{E}\exp(s_{te})\right)^2.
$$

$`s_{te}`$ is token $`t`$'s score for expert $`e`$.
$`N`$ and $`E`$ count tokens and experts; $`\beta`$ sets the penalty.
The inner log-sum-exp is the router's log normalizer. The loss discourages
large magnitude in that quantity.

Adding the same constant to all scores leaves softmax shares unchanged,
but changes the z-loss. This shows that it controls something different
from those shares or the observed traffic.

**"Loss-free" needs care.**
[DeepSeek-V3](https://arxiv.org/html/2412.19437v1) adjusts per-expert
biases using measured load. The bias changes which experts win; the underlying
match scores still set their contributions. It also keeps a small
**sequence-wise auxiliary balance loss** and a multi-token prediction loss.
Its main balancing strategy is called loss-free, not every part of training.

**Dropless does not mean unlimited.**
[MegaBlocks](https://arxiv.org/abs/2211.15841) groups variable-sized
expert work into block-sparse calculations. This avoids the usual capacity
dropping without padding every expert to a wasteful common size.
Memory, skewed loads, and data transfers still have limits.
Good execution also does not guarantee that experts learn useful roles.

## 4.4 Where experts live in a Transformer

A Transformer often has two main jobs in each block. **Attention** mixes
information across text positions. An **FFN** then transforms the features
at each position. Most MoE language models replace the FFN with experts,
not the whole attention stage.

```text
values carried through the model
    |
    +--> rescale --> attention ------------------------+
    |                                                  |
    +<---------------- add shortcut <-----------------+
    |
    +--> rescale --> router --> selected expert FFNs --+
    |                               |                  |
    |                       mix their outputs          |
    +<---------------- add shortcut <-----------------+
```

Because attention has already shared context, a token's expert input includes
information from other positions. The same word can choose different experts
in different sentences or layers.

| Placement pattern | Examples | Consequence |
|---|---|---|
| Alternate dense and MoE FFNs | GShard, GLaM; selected Switch versions | Fewer expert layers to store and communicate through |
| MoE FFN in every decoder block | Mixtral, OLMoE | More routing choices, but more places where data may move |
| Dense blocks first, then MoE | DeepSeekMoE, DeepSeek-V2, DeepSeek-V3 | Shared early processing before expert choices |
| Dense/shared path plus routed experts | DeepSeek shared experts; Arctic hybrid | The always-used path must count in active work |
| Selected image or other input blocks | Research extensions beyond the full catalog here | Input units and routing choices change the costs |

Placement belongs to a model version, not a brand name.
Mixtral 8x7B uses **32** expert-FFN blocks and 8x22B uses **56**.
OLMoE uses **16**. DeepSeek-V2 leaves its first FFN dense; V3 leaves
its first three dense. Arctic uses MoE each layer across **35** hybrid layers.
The [model catalog](07-moe-models.md) gives their sources.

An expert number is an address, not a job title. Claiming "expert 4 handles
law" needs evidence. Experts may show patterns related to language, syntax,
or topic, but training does not promise a neat subject for each one.

## 4.5 Parameter ledger: total versus active

This table compares historical model sizes. **B** means billion parameters;
**T** means trillion. Neither means bytes. Some figures are rounded, and
reports do not always count shared or extra parts the same way.

| Model / representative release | Organization | Total parameters | Active per token | Routing / placement | Source and qualification |
|---|---|---:|---:|---|---|
| GShard-M4, 2020 | Google | About 600B | About 1.5B | Top-2; alternating expert FFNs | [GLaM comparison](https://arxiv.org/abs/2112.06905); supervised translation, not Part 3 pretraining |
| Switch-C, 2021 | Google | About 1.6T in Switch; 1.5T in GLaM's rounded table | About 1.5B in GLaM's table | Top-1; 2,048 experts per sparse layer | [Switch](https://arxiv.org/abs/2101.03961), [GLaM](https://arxiv.org/abs/2112.06905); different report counting |
| GLaM 64B/64E, 2021 | Google | About 1.2T | 96.6B | Top-2 of 64; alternating sparse FFNs | [Report](https://arxiv.org/abs/2112.06905) |
| Mixtral 8x7B, 2023 | Mistral AI | 46.7B | 12.9B | Top-2 of 8; every FFN | [Release](https://mistral.ai/news/mixtral-of-experts) |
| Mixtral 8x22B, 2024 | Mistral AI | 141B | 39B | Top-2 of 8; every FFN | [Release](https://mistral.ai/news/mixtral-8x22b) |
| DeepSeekMoE 16B, 2024 | DeepSeek | 16.4B | About 2.8B | Six of 64 routed plus two shared; after first dense block | [Report/release](https://github.com/deepseek-ai/DeepSeek-MoE) |
| DeepSeek-V2, 2024 | DeepSeek | 236B | 21B | Six of 160 routed plus two shared; MLA | [Release](https://github.com/deepseek-ai/DeepSeek-V2) |
| DeepSeek-V3, December 2024 | DeepSeek | 671B main model | 37B | Eight of 256 routed plus one shared; MLA | [Release](https://github.com/deepseek-ai/DeepSeek-V3); 685B with 14B MTP weights |
| Grok-1, released 2024 | xAI | 314B | Vendor says 25%; about 78.5B if applied literally | Top-2 of 8 | [Release](https://x.ai/news/grok-os); calculated approximation, not an exact checked inventory |
| DBRX, 2024 | Databricks | 132B | 36B | Top-4 of 16 | [Technical release](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) |
| Arctic, April 2024 | Snowflake | 480B | 17B | Top-2 of 128 plus a dense residual model | [Repository](https://github.com/Snowflake-Labs/snowflake-arctic); rounded counts |
| OLMoE-1B-7B-0924 | AI2 and collaborators | 6.9B | 1.3B | Top-8 of 64; dropless | [Repository](https://github.com/allenai/OLMoE) |

GShard and Switch-C's active counts come from GLaM's comparison.
Grok's active estimate multiplies a rounded size by a reported percentage;
we have not independently resolved how that statement counts shared weights.
DeepSeek-V3's extra multi-token prediction module also changes the stored total.
Keep those counting choices visible.

Some company-run MoE models, including specified Gemini versions, have
undisclosed counts. See the [foundation-model chapter](06-foundation-models.md).
We do not fill gaps with rumored GPT, Claude, or other model sizes.

**Optional math:** a simplified count separates shared weights from experts:

$$
P_{\mathrm{total}}\approx P_s+L_mEP_e,\qquad
P_{\mathrm{active}}\approx P_s+L_mkP_e.
$$

$`P_s`$ counts the always-used shared weights. $`L_m`$ is the number of
expert layers. Each has $`E`$ experts of size $`P_e`$, but a token chooses
only $`k`$. Total size counts every expert; active size counts chosen ones
plus the shared part.

A gated FFN with width $`d`$ and middle size $`h`$ has three main weight
arrays, totaling about $`3dh`$ values before biases. These are approximate
accounting rules. They are not literal counts of every memory location used.
Embedding and output layers can be counted differently across reports.

Adding experts while keeping chosen count and expert size fixed can add
capacity without much more selected FFN work. But it still changes storage,
router work, data needs, and network traffic. A scaling rule measured for
dense models cannot simply be copied onto MoE.

## 4.6 Training challenges and distributed execution

**Expert parallelism** stores experts on different devices. A token's
features travel to the chosen expert's device, then its results travel back.
This often uses **all-to-all communication**, where devices exchange different
pieces of data with many peers.

The busiest expert can slow an entire step. Average device use may look
fine even while one overloaded device makes others wait.

Other methods split different work. **Data parallelism** gives model copies
different examples. **Tensor parallelism** splits a large array calculation.
**Pipeline parallelism** puts successive layers on different devices.
**State sharding** divides saved training state. They can be combined,
but each adds communication and coordination.

**Optional math: a teaching estimate of data movement.**

$$
V_{\mathrm{forward}}\approx 2Nkd\,b.
$$

$`N`$ is tokens, $`k`$ chosen experts per token, $`d`$ features per token,
and $`b`$ bytes per feature. The factor two counts sending features out
and returning results. The estimate assumes every chosen assignment is
remote and counts each logical transfer once.

With $`N=4096`$, $`k=2`$, $`d=4096`$, and BF16 values using $`b=2`$
bytes, the result is **128 MiB per MoE layer**. It excludes backward
gradients, message overhead, and local assignments. It is not a prediction
of waiting time: link speed, layout, congestion, and overlap all matter.

**Small jobs can waste fast hardware.** Many experts may each receive
only a few tokens. A GPU often handles a larger grouped calculation better.
Grouped matrix operations and block-sparse software pack work more
efficiently. Larger batches can help, but may increase waiting time and memory.

**Routers need stable numbers.** Small rounding changes can affect which
expert wins. Higher-precision router math, stable softmax, careful starting
weights, gradient clipping, and suitable extra-loss weights can help.
Avoiding dropped work alone does not make learning good.

**Training needs more memory than running the weights.** It also saves
gradients, optimizer history, some high-precision weights, layer outputs,
and communication buffers. Using fewer bits, splitting storage, or
recalculating outputs can reduce the bill, not erase it.

**Device-aware choices can help.** Restricting experts to a few devices or
nodes can reduce costly transfers. It also limits which combinations are
available. DeepSeek-V3 combines such routing with FP8, pipeline/expert
work splitting, and overlap. Hiding much of its communication time does
not mean data transfers are free on every cluster.

## 4.7 Inference: memory, arithmetic, and useful answers

**Inference** means using a trained model to produce outputs.
MoE is useful when its extra capacity is worth the full storage and
software cost. It may be a poor fit for tiny batches, limited memory,
or slow links between devices.

| Resource | Primarily affected by | Why active parameter count is insufficient |
|---|---|---|
| Weight memory | All stored weights and their bit format | Unchosen experts still need to be available |
| FFN arithmetic | Chosen experts, their size, layers, and batch | A few large experts can cost more than many small ones |
| Attention/KV memory | Attention design, context, batch, and KV heads | Sparse experts do not themselves shrink the attention cache |
| Dispatch overhead | Destinations, chosen experts, and network | Sending data can be costly even when expert math is cheap |
| Latency | The request's slowest path and scheduling | Fewer calculations can still mean a longer wait |
| Throughput | Hardware use, concurrent requests, and software | One fast request does not prove high speed under heavy load |
| Cost per useful answer | Full costs, answer length, mistakes, and retries | Cheap tokens can still produce expensive failed tasks |

**Teaching storage calculation, not a measured deployment.** Mixtral
8x7B has **46.7B** total weights. At two bytes each, they take about
**93.4 GB** in decimal units. Caches, scales, temporary buffers, and
software need more space.

Ideal four-bit weight data would take **23.35 GB**. Real quantization also
stores metadata and may keep some arrays at higher precision. A model with
**12.9B active** weights does not therefore need the same storage as a
dense model with only 12.9B total.

**Prefill and decode have different jobs.** Prefill processes the prompt.
Decode generates new tokens, often one after another. Prompt work can use
large groups of calculations. Decode can spend much time moving weights
and attention data or waiting for other devices.

Test both using realistic prompt lengths and numbers of simultaneous users.
A speed figure for one stage is not a promise for the other.

**Offloading means moving some weights out of fast device memory.** This
can make a model fit, but loading chosen experts on demand can be slow.
Caching or copying popular experts may help certain traffic patterns.
Those choices add storage and scheduling work of their own.

**Lower precision and distillation change things.** Fewer bits can change
router choices or expert outputs. Teaching a dense student creates another
model. Neither method guarantees identical task quality from an active-size
number alone.

## 4.8 A production-oriented case: Databricks DBRX

**Evidence status: Sourced vendor application with separately identified
system benchmarks.** This is a real company's public account of a deployed
model and its design choice. It is not an independent audit of customer savings.

**The problem.** Databricks makes data and analytics software for businesses.
Its March 2024 release describes DBRX in early **SQL applications** and
through its Foundation Model APIs. SQL is a language for querying databases.
The goal is useful SQL, code, and document-based help without using every
weight of a similarly large dense model on every token.

**The data and steps.** DBRX pretraining used a reported **12T text/code
tokens**. In a SQL workflow, the user gives a request with a database
schema or other context. The model writes a query. The application still
must check it and run it only with allowed permissions.

The release does not share the internal SQL rollout dataset or an audited
end-user success rate. Its public question-answering tests instead use
Natural Questions/HotpotQA and retrieved Wikipedia passages. Those are not
the same data as the internal SQL rollout.

**Why choose MoE?** Databricks describes **132B total parameters, 36B
active, and top-4 of 16 experts** as a useful quality/compute choice.
It also credits better data, tokenization, optimization, and software.
The report supports that design choice, not the claim that MoE alone
caused all improvements.

| Public claim | Actual reported figure or observation | Correct interpretation |
|---|---|---|
| Service deployment | DBRX APIs and early SQL-product integration | A named vendor use, not a count of successful customer deployments |
| SQL rollout quality | A qualitative comparison with GPT-3.5 Turbo and GPT-4 Turbo | No public audited SQL business KPI in the cited release |
| Optimized service generation | **Up to 150 tokens/second/user**, using **8-bit quantization** | A vendor serving claim, not an average or a promised rate for every request |
| Comparison with Llama 2 70B | **Up to 2x inference throughput** in the stated test | Applies to that hardware, traffic, and precision; not a universal price saving |
| Serving-comparison protocol | About **2,000 prompt tokens**, **256 output tokens**, **one new user per second**, **16-bit precision**, optimized TensorRT-LLM | Different setup from the 8-bit, 150-token service claim |
| Document-based QA reference | DBRX **60.0%** versus GPT-3.5 Turbo **57.7%** on Natural Questions with ten retrieved passages | A research answer-matching score, not a SQL customer KPI |

These figures come from the
[Databricks release, inference section, and Tables 3-4](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).
Its comparison splits calculations across a node using tensor parallelism.
Changing the device count or setup can change the result.

**What we can conclude.** We have a named organization, use case, design,
deployment statement, and measured figures with conditions.
We do not have an experiment where only MoE changed, a current API price,
or a verified percentage drop in customer operating costs.

Two other examples show why care matters.
DeepSeek-V2's [release](https://github.com/deepseek-ai/DeepSeek-V2)
reports **42.5% training-cost savings, 93.3% KV reduction, and 5.76x
maximum generation throughput** against DeepSeek 67B. Its changes include
MLA and software as well as MoE.

DeepSeek-V3's [training ledger](https://arxiv.org/html/2412.19437v1)
reports **2.788M H800 GPU-hours** and an assumed **\$5.576M** rental
cost, excluding earlier research and development. Neither example proves
a customer return on investment caused by MoE alone.

## 4.9 A decision framework and remaining questions

Compare a dense model and an MoE at the **same required answer quality**
and under realistic use. Parameter counts are a starting clue, not a decision.

1. Define a useful result, acceptable mistakes, text lengths, user load,
   and maximum waiting time.
2. Give both models the same data, prompts, search tools, and number
   of attempts.
3. Count all weights, attention caches, extra quantization data, working
   memory, and copies or offloaded parts.
4. Measure prompt processing, token generation, total output rate,
   and the slowest requests on the intended hardware.
5. Check busy and underused experts, dropped work, device use, and
   data traffic. An average GPU-use number can hide problems.
6. Work out cost per successful task, including retries, long answers,
   failed requests, and support work.

Researchers still study what experts learn, how many to use, and how much
data they need. They also study how to retrain routers without losing
useful roles and how to handle images, audio, and mixed hardware.

No single scaling law answers all these questions.
The useful conclusion is simpler: **MoE can save selected calculation work,
but its value depends on the task, model, and computing system together.**

## Coverage and continuation manifest

This chapter covers the 1991 idea, dense and sparse use, the main routing
rules, balancing, overflow, expert placement, and total-versus-active counts.
It also explains training and serving costs and a carefully qualified
Databricks example.

Continue to [cross-cutting comparisons](09-comparative-guide.md), the
[glossary](10-glossary.md), or the [individual MoE entries](07-moe-models.md).

Later editions could add more image/audio experts, newer model details,
and independent cost-per-success studies. Those are future additions,
not results this book claims to have already obtained.
