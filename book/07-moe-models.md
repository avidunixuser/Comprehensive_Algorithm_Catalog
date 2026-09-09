# 3. Unsupervised Learning Algorithms: Sparse MoE Model Catalog

## 3.15 Self-supervised sparse expert language models

A large network does not have to use every part for every input.
**Mixture of Experts (MoE)** models choose which small networks, called
experts, will do part of the work. A **router** makes that choice.
An expert is a group of learned calculations, not a separate person or chatbot.

This chapter groups models by their first broad learning stage, called
**pretraining**. Most learn to predict the next piece of text or fill in
missing text. Later training on instructions or preferred answers is described
separately. The original 1991 mixtures and GShard translation used supplied
answers; see [Section 1.11](02-supervised-neural.md).

Some useful terms before you begin:

| Term | Plain meaning |
|---|---|
| Token | A unit the text model uses, often a word or part of a word |
| Parameter / weight | A number the model learns |
| Dense layer | Uses all of that layer's main weights for each input |
| Sparse expert layer | Chooses only some experts for each input |
| FFN | Feed-forward network: a stage that transforms the features at one text position |
| Attention | A calculation that combines information from different positions |
| Encoder / decoder | An encoder builds features from an input; a text decoder predicts output tokens |
| Residual connection | A shortcut that adds earlier values back after a layer |
| Normalization | Rescaling signals to make their sizes easier to work with |
| Softmax | Turns scores into positive shares that add to one |
| Active parameters | Weights used for one token, under the report's counting rule |
| Total parameters | All counted weights, including experts not used by that token |

**Total size and active size answer different questions.** You usually need
the full set of weights available in memory, even when each token uses only
some. Shared attention and other always-used parts also count. That is why
`active = total * selected_experts / all_experts` is not a safe shortcut.

The [MoE deep dive](08-moe-deep-dive.md) explains these choices in more detail.
These entries describe dated releases, not necessarily the latest model from
each company.

### 3.15.1 Switch Transformer

**In plain English:** Switch gives each token one chosen expert for part of
its processing. This lets the model store many experts without using them all
for every token.

**Name:** Switch Transformer. Switch-Base, Switch-Large, Switch-XXL, and
Switch-C are different size settings, not the same model.

**Category & sub-category:** Self-supervised text pretraining, grouped here
under unsupervised learning. It first learns to fill in missing text.
Later training for a specific task uses supplied answers.

**Originating paper/vendor/year:** William Fedus, Barret Zoph, and Noam
Shazeer at Google. The
[Switch Transformers paper](https://arxiv.org/abs/2101.03961)
appeared as a preprint in 2021 and in JMLR in 2022.

**Core mechanism:** Replace selected FFNs with groups of experts.
The router scores the experts and sends each token to the top one.
The chosen expert's output is multiplied by its router probability.
This is **top-1 routing**.

A top-2 router would choose two experts instead.

A limit controls how many tokens each expert can take. An extra training
penalty discourages sending almost everything to the same few experts.
These rules help spread the work.

**Inputs/outputs and typical data types:** Text with missing spans goes in.
The model predicts the missing text. A version trained further for questions
can take a question and generate an answer.

**Strengths and limitations:** It offers many learned weights without using
all of them at once. Choosing one expert also makes routing simpler.
But the full model still needs lots of memory. Busy experts can fill up,
and better first-stage learning does not always lead to equally better task scores.

**Computational complexity / scalability notes:** Each token uses one expert
per expert layer, but the router still scores the group. All experts' weights
and saved training state still take space. Real speed depends on how tokens
are grouped and moved between devices.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Google studied **closed-book question
answering**: answer from the trained model without looking up documents.
A Natural Questions prompt goes through the model, which writes an answer.
An exact-match score checks whether it matches a reference answer.

**Switch-XXL scored 34.4 versus 32.8 for T5-XXL** in the reported comparison
without *salient span masking*, a separate pretraining method. More stored
capacity at a manageable active cost is the technical reason for trying MoE.
These are research scores, not business savings, and they are not Switch-C
scores. See [Section 5.6 of the paper](https://ar5iv.labs.arxiv.org/html/2101.03961).

**Notable vendor implementations/libraries:** Google's Mesh TensorFlow
research code; Hugging Face Transformers' SwitchTransformers code and
released Google model checkpoints.

**Architecture diagram description:**

```text
text with missing parts
 -> encoder [attention; dense or Switch FFNs; residual shortcuts]
 -> decoder [attention to earlier outputs; attention to encoder;
             dense or Switch FFNs; residual shortcuts]
 -> predicted tokens
Switch FFN: router -> one chosen expert -> probability-weighted output
```

The diagram's dense and Switch layers alternate in the stated design.
Routing is for each token, not once for a whole document.

**Activation functions used and why:** Softmax creates attention and router
weights. The paper discusses ReLU FFNs and GEGLU variants at larger sizes.
These are different rules for changing signals inside the FFN; one recipe
does not cover every Switch version.

**Loss function(s):** Cross-entropy penalizes wrong missing-text predictions,
especially confident ones. A second loss encourages balanced expert use.
Later supervised training adds the relevant task loss.

**Optimization algorithm(s):** T5-style update rules, commonly Adafactor,
adjust the learned weights. The learning rate controls update size.
Its warmup, decay, and task-tuning settings depend on the exact version;
there is no one schedule for the whole family.

**Regularization techniques:** Dropout switches off some training signals
to reduce dependence on them. The paper studies extra **expert dropout**
inside experts, not simply more dropout everywhere. Capacity and balance
settings also control how training work is assigned.

**Backpropagation considerations:** Error signals can update the chosen
expert and its gate weight. The hard choice of an expert has no ordinary
smooth slope. More precise router calculations and careful starting weights
help stability. A skipped expert assignment misses its normal task update.

**Parameter count / scaling behavior:** Switch-XXL has about **395B total**
parameters. Switch-C is reported at about **1.6T total**, with **2,048 experts**
per expert layer. Here B means billion and T means trillion.

The [GLaM comparison](https://ar5iv.labs.arxiv.org/html/2112.06905)
lists Switch-C as **1.5T total / 1.5B activated** using its rounded count.
These are different reports' counting rules, not an exact inventory of
one downloaded file.

**Training paradigm:** Fill-in-the-blank pretraining, then supervised
fine-tuning. A smaller dense student may also learn from the larger model,
called distillation. It does not automatically keep all the teacher's skills.

**Hardware/parallelism considerations:** TPUs share training examples and
expert storage. Wider versions also split large calculations across devices.
Using one expert at a time does not make a trillion-weight model fit on one chip.

### 3.15.2 GLaM

**In plain English:** GLaM predicts text while using two chosen experts in
each expert layer. It studies how to get more capability without paying
the calculation cost of using all stored weights at once.

**Name:** GLaM, Generalist Language Model. Its largest reported sparse
version is GLaM 64B/64E.

**Category & sub-category:** Self-supervised next-token prediction.
It uses a sparse, decoder-only Transformer and sits in the unsupervised part.

**Originating paper/vendor/year:** Nan Du and colleagues at Google.
[GLaM](https://arxiv.org/abs/2112.06905) appeared as a December 2021
preprint and at ICML 2022.

**Core mechanism:** Alternate ordinary dense FFNs with expert FFNs.
In each expert layer, the router selects two experts and combines their
outputs. Training asks the model to predict the next token from earlier text.
The unused experts still store learned weights for other inputs.

**Inputs/outputs and typical data types:** A text prompt, perhaps with a few
example answers, goes in. Probabilities for later tokens and a generated
continuation come out.

**Strengths and limitations:** GLaM shows that sparse experts can improve
the balance between language quality and compute. Its total size remains huge.
The research setup and incomplete public training access make the full result
hard to repeat.

**Computational complexity / scalability notes:** Each sparse layer uses
two experts. The dense layers, attention, and routing still add work.
Putting experts on different devices divides their storage, but it also
requires sending token features between those devices.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Google tested **29 public NLP tasks**,
including questions and reading comprehension. NLP means natural language
processing. A prompt and any demonstrations lead to an answer, scored using
each task's rules.

The largest model's report gives **180 GFLOPs per token versus 350 GFLOPs**
for its GPT-3 comparison. GFLOPs count billions of arithmetic operations.
The reported training-energy estimates are **456 versus 1,287 MWh**.
These are research comparisons, not a controlled business electricity bill.
The hardware and data differ too, so the whole gap cannot be assigned to
MoE alone. [Source: Table 1](https://ar5iv.labs.arxiv.org/html/2112.06905).

**Notable vendor implementations/libraries:** Google's research software,
GSPMD/XLA tools for dividing work, and Cloud TPUs. The paper does not establish
a public downloadable GLaM model or a currently available GLaM API.

**Architecture diagram description:**

```text
tokens
 -> repeated [attention to earlier tokens + residual shortcut
              -> dense FFN or top-2-of-64 expert FFN + residual shortcut]
 -> next-token scores
```

Every other Transformer layer uses MoE in the reported design.

**Activation functions used and why:** Softmax weights attention and experts.
The paper also describes a gated-linear/GELU change in its **non-MoE FFNs**.
Gates change the strength of feature signals. This statement should not be
copied onto every expert without checking its actual code.

**Loss function(s):** Next-token cross-entropy, which rewards probability
on the real next token, plus expert-load balancing.

**Optimization algorithm(s):** The report uses **Adafactor** with no
first-moment accumulation. It stores compact second-moment summaries and
clips updates. In plain terms, it adjusts step sizes from recent gradients
while saving memory and limiting large steps.

The learning rate stays at **0.01 for 10,000 steps**, then follows
inverse-square-root decay: steps shrink as training goes on. These are
GLaM's historical settings, not advice for every model.

**Regularization techniques:** Filtered data, a chosen mix of training
sources, and controls on expert load. Pretraining **dropout is zero**.
That does not mean all controls on learning were removed; normalization
and data choices still matter.

**Backpropagation considerations:** The selected experts and gate weights
receive learning signals. Uneven routing and rounding errors can hurt learning.
Devices also have to exchange and combine results correctly.

**Parameter count / scaling behavior:** About **1.2T total and 96.6B active
parameters per token** in the largest sparse version. Each MoE layer has
**64 experts**, with two chosen. "64B/64E" is the model's label, not its
active parameter count.

**Training paradigm:** Train from scratch by predicting text, then test with
zero, one, or a few demonstrations in the prompt. Those prompt examples do
not update the model's weights.

**Hardware/parallelism considerations:** The largest reported run uses
**1,024 TPU-v4 chips**. It divides weight and signal arrays in two dimensions.
Weights use float32 and intermediate signals use bfloat16, which uses fewer
bits. Those details matter when interpreting its energy figures.

### 3.15.3 Mixtral 8x7B

**In plain English:** Mixtral 8x7B uses two of eight experts at each text
processing block. Its public weights let others run and adapt the model,
but all eight experts still take storage.

**Name:** Mixtral 8x7B. Mixtral 8x7B Instruct is a separate version trained
further to follow instructions.

**Category & sub-category:** Self-supervised next-token pretraining with
a sparse decoder Transformer; grouped here under unsupervised learning.

**Originating paper/vendor/year:** Mistral AI released it in December 2023.
Albert Q. Jiang and colleagues published
[Mixtral of Experts](https://arxiv.org/html/2401.04088v1) in January 2024.

**Core mechanism:** Each FFN becomes eight experts with separate weights.
For each token, a router selects two and makes their selected weights sum to
one. It then adds their weighted outputs. The experts share attention and
learn together. They are not eight separately trained chatbots.

**Inputs/outputs and typical data types:** Text and code in several languages
go in. The model gives next-token probabilities and continuations. The
Instruct version is adapted to answer requests.

**Strengths and limitations:** It gives strong historical results with fewer
active weights than some larger dense models. Public weights make local use
possible. But storage, routing software, and batching still affect its speed.
The name does not mean it has 56B separate weights.

**Computational complexity / scalability notes:** Each token uses two expert
FFNs per block, plus shared attention and routing. Comparing only active
weight counts leaves out memory traffic and how well the hardware is used.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Mistral tested knowledge and code
as an alternative to larger dense models. In **5-shot MMLU**, the model gets
five demonstrations before answering multiple-choice questions.
Mixtral 8x7B scores **70.6%**, versus **69.9%** for Llama 2 70B as
re-evaluated by the authors.

This is a test of quality versus active compute, not measured time saved
by customers. The [release](https://mistral.ai/news/mixtral-of-experts)
also says it powered the `mistral-small` endpoint then. That **2023**
endpoint mapping must not be assumed current.

**Notable vendor implementations/libraries:** Mistral reference code,
Hugging Face Transformers, vLLM/MegaBlocks, and compatible serving tools.
Support depends on model version and number precision.

**Architecture diagram description:**

```text
tokens -> 32 decoder blocks
each block:
  RMSNorm -> causal GQA attention -> add residual shortcut
  RMSNorm -> router -> two of eight SwiGLU experts
                       -> weighted sum -> add residual shortcut
-> next-token scores
```

RMSNorm rescales signals. GQA shares some attention data across heads.
The model has full attention over its reported 32K context; do not assume
it uses Mistral 7B's sliding-window attention.

**Activation functions used and why:** SwiGLU experts use a smooth SiLU
gate to control feature strength. Softmax turns attention and router scores
into weights.

**Loss function(s):** Next-token cross-entropy. Software may also expose
extra router losses. The short report does not reveal every original
pretraining loss setting.

**Optimization algorithm(s):** AdamW with warmup and decay is a typical
way to adapt the model. Warmup starts with small steps; decay reduces them
later. The full original pretraining schedule is not disclosed here.

**Regularization techniques:** RMSNorm, selected training data, and
fine-tuning choices. Exact dropout and balance settings need the chosen
training configuration; they should not be guessed.

**Backpropagation considerations:** Error signals pass through the chosen
experts and their contribution weights. The hard top-2 choice itself is
not smooth. Experts that receive too few tokens may learn poorly.

**Parameter count / scaling behavior:** **46.7B total / 12.9B active per
token**, according to Mistral. Every decoder FFN uses MoE. Shared attention
is one reason eight-times-seven does not give the total.

**Training paradigm:** Pretraining from scratch. Instruct adds supervised
fine-tuning and **direct preference optimization (DPO)**, which learns from
preferred and less-preferred answers. Do not assume it uses a PPO/RLHF
training loop.

**Hardware/parallelism considerations:** Devices can split attention
calculations and store separate experts. A simple BF16 weights-only count is
about **93.4 GB** in decimal units. This is a teaching calculation of the
weight storage, not the complete memory needed to run it.

### 3.15.4 Mixtral 8x22B

**In plain English:** This is a larger Mixtral that still chooses two of
eight experts per block. The experts and shared network are larger, so two
chosen experts do not mean the same cost as in 8x7B.

**Name:** Mixtral 8x22B. Its base and instructed April 2024 releases are
different versions.

**Category & sub-category:** Self-supervised next-token pretraining with
a sparse decoder. Instruction tuning adds supervised training afterward.

**Originating paper/vendor/year:** Mistral AI,
[April 2024 release](https://mistral.ai/news/mixtral-8x22b).

**Core mechanism:** A larger shared decoder works with eight expert FFNs
per block. Each token uses two. The model still predicts text token by token;
it does not choose one complete chatbot for a whole request.

**Inputs/outputs and typical data types:** Text, code, and longer documents
in several languages. Suitable instruction and serving formats can also
support structured outputs and requests to call software functions.

**Strengths and limitations:** More stored capacity and a reported **64K
context** than 8x7B. But its large full size can be costly when few requests
run at once. Longer prompts also need more attention work and saved state.

**Computational complexity / scalability notes:** Two experts run per block,
but they sit in a wider, deeper model. Keeping the same number of chosen
experts does not keep work fixed when each expert grows.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Mistral tested math solving on
**GSM8K** and **MATH**. A problem goes in; generated solutions produce
answers that are scored against the references.

The instructed model reports **90.8% on GSM8K with maj@8** and **44.6%
on MATH with maj@4**. These use majority votes from eight or four attempts.
They are not single-attempt accuracy or proof of better classroom learning.
Mistral describes a quality/active-compute goal, but does not give an
independent comparison of live business costs for dense versus MoE models.
[Source](https://mistral.ai/news/mixtral-8x22b).

**Notable vendor implementations/libraries:** Mistral's platform and
released weights; Hugging Face Transformers and compatible MoE serving tools.

**Architecture diagram description:**

```text
tokens -> 56 decoder blocks with residual shortcuts
          [GQA attention; top-2-of-8 SwiGLU FFN]
       -> next-token scores
```

The [base configuration](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/config.json)
has width **6,144**, **48 query heads**, and **8 KV heads**.
Those heads are parallel attention channels; some share keys and values.
The configuration has no sliding-window setting.

**Activation functions used and why:** SiLU gates control expert feature
strength. Softmax supplies attention and routing weights. Multiplying by
a learned gate lets the network change a feature's influence.

**Loss function(s):** Next-token cross-entropy. The released configuration
has a setting for an extra router loss. A default setting does not prove
that every original training stage used that value.

**Optimization algorithm(s):** AdamW with a rising then falling learning
rate is a typical fine-tuning choice. The announcement does not disclose
the complete original training or post-training schedule.

**Regularization techniques:** RMSNorm and choices of data and training
controls. The cited configuration has zero attention dropout. That does
not mean zero dropout is best for every new fine-tune.

**Backpropagation considerations:** The hard top-2 choice can change
suddenly. Uneven expert use and large saved layer outputs also complicate
training. Clipping large gradients and managing number precision can help.

**Parameter count / scaling behavior:** **141B total / 39B active per
token**, rounded vendor counts. All 56 blocks use the Mixtral expert FFN.

**Training paradigm:** Base pretraining followed by a separately released
instruction-tuned model. The word "Instruct" alone does not reveal its
complete training recipe.

**Hardware/parallelism considerations:** BF16 weights alone take about
**282 GB** in decimal units. Attention caches and working space add more.
It usually needs several devices or suitable lower-bit storage.

### 3.15.5 DeepSeekMoE 16B

**In plain English:** DeepSeekMoE divides expert work into smaller pieces.
It combines chosen small experts with shared experts that every token uses.

**Name:** DeepSeekMoE 16B. It is not DeepSeek-V2-Lite or the later V2/V3 models.

**Category & sub-category:** Self-supervised next-token learning with
fine-grained experts, grouped here under unsupervised learning.

**Originating paper/vendor/year:** Damai Dai and colleagues at DeepSeek,
[DeepSeekMoE](https://arxiv.org/abs/2401.06066), January 2024.

**Core mechanism:** Make the expert FFNs smaller and choose more of them
per token. Also keep shared experts active for every token. The aim is for
shared experts to handle common patterns and chosen experts to learn more
distinct patterns. This does not guarantee that each expert gets a clear
human-readable subject.

**Inputs/outputs and typical data types:** English and Chinese text and code.
It produces next-token probabilities and continuations.

**Strengths and limitations:** It offers many stored weights at a smaller
active compute cost, with a public model. Very small expert calculations
can be inefficient on some hardware. Shared experts do not automatically
balance the work sent to the other experts.

**Computational complexity / scalability notes:** Count both the selected
experts and the always-used shared experts. The number of experts alone
does not tell you the work: their individual sizes matter too.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** DeepSeek compared the sparse
16B model with its dense DeepSeek 7B model trained on the **same 2T-token
corpus**. It used knowledge, math, and code tests. Prompts became answers
or programs, then the tests scored those outputs.

The release reports broadly similar ability using **40.5% of the computation**
of DeepSeek 7B in its internal comparison. This supports trying smaller
and shared experts. It does not mean an API bill was 40.5% as large or that
every individual task tied. [Official results](https://github.com/deepseek-ai/DeepSeek-MoE).

**Notable vendor implementations/libraries:** DeepSeek model code,
Hugging Face tools, and DeepSpeed fine-tuning examples. Support depends
on the exact model-code version.

**Architecture diagram description:**

```text
tokens -> one dense decoder block
       -> remaining blocks:
          [attention; two shared expert equivalents
           + top-6-of-64 small routed experts; residual shortcuts]
       -> next-token scores
```

The [released configuration](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/config.json)
has **28 layers** and width **2,048**.

**Activation functions used and why:** SiLU-based gates control expert
signals. Softmax supplies weights for routing and attention.

**Loss function(s):** Next-token cross-entropy plus expert-balancing goals.
The Chat version also learns from supplied instruction-and-answer pairs.

**Optimization algorithm(s):** AdamW-family update rules fit this kind
of decoder. The released fine-tuning example starts with warmup, then uses
cosine decay to shrink steps smoothly. That example's learning rate is not
proof of the original pretraining schedule.

**Regularization techniques:** RMSNorm, selected training data, and load
balancing. The cited configuration uses zero attention dropout. A new
fine-tuning task may need different controls.

**Backpropagation considerations:** Shared experts learn from broad traffic;
routed experts learn from their selected tokens. Making experts smaller
can encourage different roles, but some may still get too little useful
training.

**Parameter count / scaling behavior:** **16.4B total, about 2.8B active**.
Each MoE block uses two shared experts and six selected routed experts.
Each expert is narrower than a conventional dense FFN.

**Training paradigm:** Pretraining from scratch on **2T tokens**, then
a separate supervised Chat fine-tune.

**Hardware/parallelism considerations:** The release says the model can
run on a **40 GB GPU without quantization** under its supported conditions.
That does not promise the same for any batch or context length. Training
needs much more saved state than simply running the weights.

### 3.15.6 DeepSeek-V2

**In plain English:** DeepSeek-V2 tackles two costs: which expert calculations
to run and how much information to save about earlier text. It combines sparse
experts with compressed attention storage.

**Name:** DeepSeek-V2, the full-size May 2024 model. DeepSeek-V2-Lite is
a separate smaller version.

**Category & sub-category:** Self-supervised language pretraining using
sparse experts and compressed attention. Chat versions add supervised and
reward-based training.

**Originating paper/vendor/year:** DeepSeek-AI,
[DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434), 2024.

**Core mechanism:** Use DeepSeekMoE's small and shared experts for FFNs.
Use **Multi-head Latent Attention (MLA)** to save a compact joint form of
attention keys and values. These are stored features of earlier tokens
used when generating later ones.

MLA reduces that cache's storage. MoE reduces the expert work used for each
token. They solve different parts of the cost problem.

**Inputs/outputs and typical data types:** English/Chinese text and code
prompts become continuations or chat answers, including longer-context uses.

**Strengths and limitations:** It improves the balance of stored capacity,
active work, and attention memory. The speed gains need software built to
use MLA well. Generic attention code may not achieve them.

**Computational complexity / scalability notes:** A smaller cache does
not remove all attention work on long text. Experts still need their full
stored weights and must exchange data when split across devices.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Sourced application.** DeepSeek offered V2 through its
chat site and API for language and code help. A request is split into tokens,
processed through MLA and experts, then decoded into a response.

The [official release](https://github.com/deepseek-ai/DeepSeek-V2)
reports **42.5% lower training cost, 93.3% less KV cache, and 5.76x
maximum generation throughput**, compared with **DeepSeek 67B**.
Throughput means output produced per unit time.

These are reported system measurements alongside a deployed model, not a
customer business result. They reflect MoE, MLA, and the software together.
They do not show that routing alone caused all three gains.

**Notable vendor implementations/libraries:** DeepSeek reference code,
Hugging Face model code, vLLM, and SGLang versions that support its MLA design.

**Architecture diagram description:**

```text
tokens -> 60 decoder blocks with residual shortcuts
          [MLA attention;
           first block: dense FFN
           later blocks: two shared experts + top-6-of-160 routed experts]
       -> next-token scores
```

This follows the [full-size configuration](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json),
not the Lite model.

**Activation functions used and why:** SiLU-gated expert layers control
feature strength. Softmax weights routing and attention. RMSNorm rescales
the compressed and shortcut signals to help keep training stable.

**Loss function(s):** Next-token prediction plus balancing terms.
Identified Chat versions also use supervised answer losses and
reinforcement-learning objectives.

**Optimization algorithm(s):** The [V2 report](https://arxiv.org/html/2405.04434v1)
uses **AdamW**, with moment settings **0.9 and 0.95** and weight decay
**0.1**. The moments track recent gradient direction and size.

**Optional math:** the peak learning rate is $`2.4\times10^{-4}`$.
It rises to that value over **2,000 steps**, then drops at about **60%**
and **90%** of training tokens. Context extension and later training use
their own schedules.

**Regularization techniques:** RMSNorm, chosen data, expert/device
balancing, and limits on routing. The public inference configuration does
not tell us every original training control.

**Backpropagation considerations:** Learning signals pass through MLA's
compression layers and the chosen experts. Uneven traffic and unstable
router scores can waste the added capacity. Gradient clipping and suitable
number precision remain important.

**Parameter count / scaling behavior:** Full V2 has **236B total / 21B
active per token**. **V2-Lite has 16B / 2.4B**. Its settings must not be
substituted for the 236B model's settings.

**Training paradigm:** Pretraining on **8.1T tokens**, extending context
length, supervised fine-tuning (SFT), and the stated RL Chat versions.
Base and Chat scores describe different models.

**Hardware/parallelism considerations:** It needs MLA-aware caching and
ways to divide experts and dense calculations across devices. The release's
example BF16 setup uses **eight 80 GB GPUs**. That is one setup, not a
universal minimum.

### 3.15.7 DeepSeek-V3

**In plain English:** DeepSeek-V3 keeps the compressed attention and small
experts from V2. It also adjusts routing to share work more evenly and learns
to predict an extra future token during training.

**Name:** DeepSeek-V3, the original December 2024 release. Base, post-trained,
and later updated versions need separate names and results.

**Category & sub-category:** Self-supervised next-token and multi-token
pretraining; a sparse decoder with MLA.

**Originating paper/vendor/year:** DeepSeek-AI,
[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1), 2024.

**Core mechanism:** Monitor how much work each expert gets. Adjust a
per-expert bias to make overloaded experts less likely to be chosen and
underused ones more likely. The bias changes the selection, while the
underlying match score controls the contribution weight.

An extra prediction task teaches future-token patterns. The model also
keeps a small **sequence-wise auxiliary balancing loss**, even though its
main balancing method is called "auxiliary-loss-free."

**Inputs/outputs and typical data types:** Text and code in several
languages; continuations or post-trained assistant responses.

**Strengths and limitations:** The model, software, and hardware are designed
together for efficient training. Repeating or serving it still requires
major computing resources. Its headline cost leaves out earlier research.
"Auxiliary-loss-free" does not mean no extra losses of any kind.

**Computational complexity / scalability notes:** Each MoE block uses
eight small routed experts and one shared expert. MLA saves cache space.
Overlapping calculations with data transfers helps speed. Extra prediction
modules also add storage beyond the main model.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Sourced application.** DeepSeek offered V3 through its
chat and API platforms for language, code, and reasoning tasks. It trained
a base model on **14.8T tokens**, then adapted it to answer users.

The report lists **2.788 million H800 GPU-hours** for official training:
**2.664M** for pretraining, **119K** for context extension, and **5K**
for post-training. At an **assumed \$2 per GPU-hour**, the total is
**\$5.576M**. It **excludes prior research and ablation experiments**.

This is a training-resource figure, not all company spending or a customer's
cost per answer. DeepSeek's stated goal is lower cost through joint choices
of algorithms, software, and hardware.
[Report](https://arxiv.org/html/2412.19437v1);
[release](https://github.com/deepseek-ai/DeepSeek-V3).

**Notable vendor implementations/libraries:** DeepSeek reference inference,
SGLang, vLLM, LMDeploy, and compatible vendor tools. They must support the
right V3 version, MLA design, and number formats.

**Architecture diagram description:**

```text
tokens -> MLA decoder blocks with residual shortcuts
          first three FFNs: dense
          later FFNs: one shared expert + top-8-of-256 routed experts
       -> next-token scores
extra multi-token prediction module: training and optional speculative use
```

Speculative use means proposing tokens that the target model can check.

**Activation functions used and why:** SiLU gates control expert signals.
Attention uses softmax. V3 uses sigmoid-based expert match scores in its
router. RMSNorm rescales signals in the compressed and residual paths.

**Loss function(s):** Next-token cross-entropy, multi-token prediction,
and a small sequence balance loss. SFT and RL losses belong to later stages,
not the whole pretraining run.

**Optimization algorithm(s):** **AdamW**, with moment settings
$`\beta_1=0.9,\beta_2=0.95`$ and weight decay **0.1**.
Those settings control the optimizer's moving averages and weight shrinking.

**Optional math:** the rate warms up to $`2.2\times10^{-4}`$ over
**2,000 steps**. It then stays constant for a long phase, follows cosine
decay, and uses lower rates at the end.

**Regularization techniques:** Weight decay, chosen data, routing-bias
updates, a small sequence balance penalty, and normalization.
These control different risks; none guarantees good output by itself.

**Backpropagation considerations:** Some calculations use fewer bits to
save work and memory; important saved values keep more precision.
The reported gradient clipping norm is **1.0**. Load-based router-bias
updates are separate from ordinary error-based gradient updates.

**Parameter count / scaling behavior:** **671B main-model total / 37B
active per token**. The release counts **685B** downloaded parameters when
**14B of multi-token prediction weights** are included. One figure counts
the main model; the other also counts the extra module.

**Training paradigm:** Pretraining from scratch, context extension, SFT,
and RL. The report also describes learning from reasoning data supplied
by its R1-related teacher work, a form of distillation.

**Hardware/parallelism considerations:** The reported cluster has
**2,048 H800 GPUs**. It uses FP8 mixed precision, expert/pipeline/data
parallelism, and DualPipe scheduling to overlap transfers and computation.
It avoids costly tensor parallelism in this training setup. Large models
do not all use the same combination of work-sharing methods.

### 3.15.8 Grok-1

**In plain English:** Grok-1 is a large public-weight model that chooses
two of eight experts for each token. Its released base model is not the
same thing as the full Grok chat product.

**Name:** The Grok-1 base checkpoint released in **March 2024**.
Its pretraining ended in **October 2023**.

**Category & sub-category:** Self-supervised next-token language learning
with a sparse decoder. This entry makes no claim about later Grok designs.

**Originating paper/vendor/year:** xAI, development in **2023** and
[open-weight release in 2024](https://x.ai/news/grok-os).

**Core mechanism:** Shared attention processes text context. A router then
selects two of eight expert MLPs, which are small feed-forward networks.
The public code shows how to run this design, but not the full original
training recipe.

**Inputs/outputs and typical data types:** Text prompts become token
continuations. Search tools and a finished chat application are not built
into the released base weights.

**Strengths and limitations:** Researchers can inspect its architecture
and use public weights. The reference code is deliberately not a fast MoE
implementation. Training data, update settings, and some counting details
are not fully disclosed.

**Computational complexity / scalability notes:** Sparse expert use does
not remove storage for **314B** weights. The simple reference code may do
extra work, so its runtime does not measure an ideal fast MoE system.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** xAI's original Grok-1 study used
**HumanEval** code-completion tasks. A function prompt led to Python code,
and tests checked the result.

The [November 2023 announcement](https://x.ai/news/grok) reports
**63.2% zero-shot pass@1**: no task demonstrations, with the success measure
for one attempt. It also reports **73.0% on 5-shot MMLU** knowledge questions.
The design aims to get capability at a lower active compute cost, but no
controlled business saving is disclosed.

These are historical Grok-1 family reports. They are not new tests of the
downloaded base checkpoint or of later live assistants.

**Notable vendor implementations/libraries:** xAI's
[JAX/Haiku reference code](https://github.com/xai-org/grok-1) and weights.
Other tools must be checked for compatible model support.

**Architecture diagram description:**

```text
tokens -> 64 decoder layers with residual shortcuts
          [GQA attention; top-2-of-8 expert MLPs]
       -> next-token scores
```

The release lists width **6,144**, **48 query heads**, **8 KV heads**,
and a maximum context of **8,192 tokens**. GQA means some query heads
share the stored keys and values.

**Activation functions used and why:** The
[released code](https://github.com/xai-org/grok-1/blob/main/model.py)
uses GELU in a multiplicative expert gate and float32 softmax for routing.
This is not simply Mixtral's SwiGLU expert design.

**Loss function(s):** Next-token prediction is publicly described.
The inference release does not give all original extra losses or their weights.

**Optimization algorithm(s):** The optimizer and learning-rate schedule
are **not publicly specified by the cited release**. AdamW may sound
plausible, but plausibility is not evidence.

**Regularization techniques:** The code shows normalization and the
inference layout. It does not establish the original dropout, weight decay,
or all other controls on training.

**Backpropagation considerations:** The chosen experts and gate weights
can receive gradients. The hard routing choice changes in discrete jumps.
More precise router code is visible, but that is not a complete record
of training stability.

**Parameter count / scaling behavior:** **314B total**. xAI says **25%
of weights are active per token**. Multiplying those rounded statements
gives about **78.5B**. That is a **derived approximation**, not an
independent exact count; treatment of shared weights can affect it.

**Training paradigm:** Text pretraining from scratch. The released base
checkpoint is explicitly not fine-tuned for dialogue.

**Hardware/parallelism considerations:** It needs substantial memory across
accelerators, JAX tools for splitting arrays, and optionally the released
8-bit weight support. The reference code's speed should not be presented
as production-serving performance.

### 3.15.9 DBRX

**In plain English:** DBRX uses four of sixteen smaller experts per token.
Databricks built it for general text and code tasks, including answers that
use retrieved documents.

**Name:** DBRX Base and DBRX Instruct.

**Category & sub-category:** Self-supervised next-token pretraining with
a fine-grained sparse decoder, followed by instruction training.

**Originating paper/vendor/year:** Databricks Mosaic research,
[March 2024 technical release](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).

**Core mechanism:** Choose four small experts from sixteen, instead of
top-2 routing through two larger experts from eight. Shared attention handles context. RoPE
supplies position information, GQA shares some attention storage, and
gated MLPs control the strength of expert features.

**Inputs/outputs and typical data types:** Text, code, and questions with
retrieved passages go in; generated text or instruction-following responses
come out.

**Strengths and limitations:** It offers public weights and a detailed
historical serving account. Its results also rely on better data,
tokenization, and training choices. They cannot all be credited to MoE.
Public weights do not remove the need to read the license.

**Computational complexity / scalability notes:** Four expert FFNs run
per token. Small calculations and moving tokens between experts affect
hardware use. Batch size, text length, and lower-bit formats change speed.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Databricks tested question
answering on **Natural Questions** with document search. It retrieved the
**top ten Wikipedia passages using bge-large-en-v1.5**, added them to
the question, generated an answer, and checked answer matching.

DBRX Instruct scored **60.0%**, versus **57.7%** for the compared
GPT-3.5 Turbo endpoint. This is a research result for answers supported
by retrieved text, not a customer support cost saving.
The release also describes API access and early internal SQL-product use,
but gives no audited SQL business KPI.

The reason for trying MoE is more stored capacity with fewer active weights.
Search and data quality matter too.
[Source: Table 4 and deployment account](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).

**Notable vendor implementations/libraries:** Databricks Model Serving,
LLM Foundry, Composer, MegaBlocks, Hugging Face tools, and NVIDIA
TensorRT-LLM in the reported serving comparison.

**Architecture diagram description:**

```text
tokens -> decoder blocks with residual shortcuts
          [GQA attention + RoPE position information;
           top-4-of-16 gated expert FFN]
       -> next-token scores
```

Experts replace FFNs through the decoder, not only at the last output.

**Activation functions used and why:** Gated linear units control feature
strength. Softmax weights attention and routing. The exact gate variant
comes from the released configuration, not another vendor's recipe.

**Loss function(s):** Next-token cross-entropy and recipe-dependent router
controls. Instruction adaptation learns from supplied answers and other
disclosed later training steps.

**Optimization algorithm(s):** Databricks describes several training
choices working together. Its release blog does not give every optimizer
setting. AdamW-style warmup and decay are common in LLM Foundry adaptations,
not proof of every original DBRX setting.

**Regularization techniques:** Carefully chosen data, a data mix that
changes during training, normalization, and MoE controls. These changes
must stay part of any explanation of its efficiency gains.

**Backpropagation considerations:** Sparse software kernels, the small
programs doing low-level calculations, help run expert batches efficiently.
Uneven loads, rounding, and exchanging updates across devices can affect
learning. Only selected routes give each expert its task gradient.

**Parameter count / scaling behavior:** **132B total / 36B active per
token**, with **16 experts and top-4 routing**.

**Training paradigm:** Pretraining on **12T text and code tokens**,
then a separate instruction version. The reported maximum training context
is **32K tokens**.

**Hardware/parallelism considerations:** The release reports **3,072 NVIDIA
H100s** for training. Its speed comparisons use tuned TensorRT-LLM software
and stated traffic and precision settings. See
[Section 4.8](08-moe-deep-dive.md) before applying those figures elsewhere.

### 3.15.10 Snowflake Arctic

**In plain English:** Arctic combines an always-used network with a large
set of optional experts. Snowflake designed this release around business
tasks such as writing SQL queries and code.

**Name:** Snowflake Arctic Base and Arctic Instruct, the **April 2024**
large language model. Later products called Arctic may be different models.

**Category & sub-category:** Self-supervised decoder pretraining, combining
dense and sparse paths, followed by business-focused adaptation.

**Originating paper/vendor/year:** Snowflake AI Research,
[April 24, 2024 release](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/).

**Core mechanism:** Run the dense path for every token. Add a contribution
from two chosen experts out of **128**. The shared path handles common
work while the expert group provides extra learned capacity.

**Inputs/outputs and typical data types:** Text requests, code, and database
schemas, which describe tables and columns. Outputs can include SQL queries,
code, and text after suitable training.

**Strengths and limitations:** It focuses on SQL, coding, and following
instructions, rather than only broad language scores. Its full model is
still expensive to store. Good test scores do not guarantee a correct query
for a new database.

**Computational complexity / scalability notes:** Count the dense path
**and** the selected experts. Long prompts, saved working data, and
uneven expert traffic add costs beyond an active-parameter total.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** Snowflake used **Spider** to
test SQL generation. A question and database schema go into the model.
It writes a SQL query, which is checked under the dataset's rules.

The [technical repository](https://github.com/Snowflake-Labs/snowflake-arctic)
reports similar selected enterprise scores to Llama 3 70B with
**17x less training compute**. That is Snowflake's comparison of whole
training recipes. It is not a measured 17x saving on live SQL queries
or a customer KPI.

The stated goal is useful business-task capability on a limited training
budget. Generated queries still need checks and controlled execution.

**Notable vendor implementations/libraries:** Snowflake's weights and
inference examples, Hugging Face code, vLLM, and related DeepSpeed work.
Identify later Arctic-branded models separately.

**Architecture diagram description:**

```text
tokens -> 35 hybrid decoder layers with residual shortcuts
          [attention/dense path + top-2-of-128 expert MLP path]
       -> next-token scores
```

The [base configuration](https://huggingface.co/Snowflake/snowflake-arctic-base/blob/main/config.json)
uses MoE every layer, with parallel attention/MLP residual paths.

**Activation functions used and why:** SiLU expert gates, softmax attention
and routing, and RMSNorm. Gates let the expert change which feature signals
matter most.

**Loss function(s):** Next-token prediction plus router controls.
The public configuration exposes an extra router-loss coefficient.
Instruction losses apply to the adapted version.

**Optimization algorithm(s):** AdamW with warmup and decay is a typical
Transformer adaptation choice, not a claim about every original Arctic
setting. The repository introduction does not give all optimizer settings.
The training cookbook is needed for a specific reproduction.

**Regularization techniques:** Training-data mix, normalization, and
router balance. The configuration has capacity and token-dropping settings.
Do not assume every Arctic runtime runs without dropping assignments.

**Backpropagation considerations:** The dense shortcut path carries learning
signals even with sparse experts. Underused experts, overflow, and
lower-precision rounding still need attention.

**Parameter count / scaling behavior:** **480B total / 17B active**,
rounded vendor figures. The description combines a **10B dense model**
with a **128 x 3.66B** residual expert component. Rounded parts need not
add up exactly to the rounded total.

**Training paradigm:** Business-focused base pretraining, then separate
instruction tuning. This does not mean every training token is a labeled
SQL example.

**Hardware/parallelism considerations:** Weight storage and calculations
are spread across devices. A simple BF16 count gives about **960 GB**
in decimal units for main weights alone. That is an illustrative count
before other working memory, not a small-memory model.

### 3.15.11 OLMoE

**In plain English:** OLMoE is a smaller-active-size expert model with
released data, code, and training records. This helps researchers inspect
how it learned, not just download its final weights.

**Name:** OLMoE-1B-7B-0924, with separate SFT and Instruct versions.

**Category & sub-category:** Self-supervised language pretraining with
small sparse experts. Its execution is **dropless**: chosen assignments
are not skipped because of the usual fixed expert capacity.

**Originating paper/vendor/year:** Niklas Muennighoff and collaborators
at the Allen Institute for AI and partner institutions,
[OLMoE](https://arxiv.org/abs/2409.02060), September 2024.

**Core mechanism:** Select eight of **64** small experts per token.
Special sparse software handles different expert workloads without the usual
capacity-based dropping. Released training data, checkpoints, code, and logs
make its choices easier to study than a weights-only release.

**Inputs/outputs and typical data types:** Text and code sequences produce
continuations or answer probabilities. Later versions are adapted as assistants.

**Strengths and limitations:** Its open training record and small active
size help research. The whole model still needs much more memory than its
active size suggests. Dropless does not mean unlimited memory or the same
speed on every device.

**Computational complexity / scalability notes:** Eight small experts run
per token, along with full attention and router work. Speed depends on
how many tokens reach each expert and on the sparse software's support.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**
**Evidence status: Research benchmark.** AI2 wanted MoE training and analysis
to be repeatable. It released **OLMoE-mix-0924** data, checkpoints, logs,
and test code. Training uses text batches; separate language and question
tests measure its predictions.

The [dated base model card](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
reports **54.1 on MMLU**, versus **48.5 for DCLM-1B** in that snapshot,
with **1.3B versus 1.4B active parameters**. These scores belong to the
release's test rules, not every possible evaluation setup.

The [repository](https://github.com/allenai/OLMoE) also provides routing
analysis and design comparisons. That helps researchers check claims
instead of just trusting them. **No audited production business KPI is reported.**

**Notable vendor implementations/libraries:** AI2 OLMo/Open Instruct,
MegaBlocks, Hugging Face Transformers, vLLM, SGLang, and llama.cpp
integrations named in the release.

**Architecture diagram description:**

```text
tokens -> 16 decoder blocks with residual shortcuts
          [normalized attention; top-8-of-64 SwiGLU FFN]
       -> next-token scores
```

Width is **2,048**, with **16 attention heads**, according to the
[training configuration](https://github.com/allenai/OLMoE/blob/main/configs/OLMoE-1B-7B-0924.yml).

**Activation functions used and why:** SwiGLU gates control expert
feature strength. RMSNorm rescales signals; normalized attention and router
scores set their contribution weights.

**Loss function(s):** Token cross-entropy, a router balance loss, and a
router z-loss. The balance term spreads work; the z-loss helps control
router-number size. Their cited weights are **0.01** and **0.001**.

**Optimization algorithm(s):** **AdamW** with token-based warmup and
cosine decay. The release also describes a separate final annealing stage,
meaning a lower-rate finishing stage.

**Optional math:** the cited peak learning rate is $`4\times10^{-4}`$.
It controls the size of the optimizer's updates.

**Regularization techniques:** Weight decay **0.1**, RMSNorm, filtered
data, and router penalties. Attention, residual, and embedding dropout
are zero in this configuration.

**Backpropagation considerations:** The release clips the global gradient
norm at **1.0**. It uses stable router normalization and sparse expert
updates. Not dropping assignments preserves their task gradients, but it
does not guarantee useful expert roles.

**Parameter count / scaling behavior:** **6.9B total / 1.3B active**,
despite the rounded 1B-7B name. All **16** blocks use MoE with eight
of 64 routed experts chosen.

**Training paradigm:** Open pretraining from scratch. Separate later
versions use supervised fine-tuning and preference methods, including
DPO/KTO experiments. These are not all one checkpoint's recipe.

**Hardware/parallelism considerations:** The cited setup uses BF16 mixed
precision, **FSDP full sharding** to split training state across devices,
and sparse MegaBlocks code. Expert parallelism is another possible design;
it should not be assumed for every released run.

#### Comparative summary: sparse MoE language models

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Switch Transformer | Text-to-text tasks | One chosen expert keeps routing simple | Huge full size; task gains can vary | Google Natural Questions research comparison |
| GLaM | General text prompts | Published quality/compute study | Hard-to-repeat research setup | 29-task NLP study with work and energy figures |
| Mixtral 8x7B | Text and code in several languages | Public weights; 12.9B active | All experts still need memory | 5-shot MMLU; historical Mistral API use |
| Mixtral 8x22B | Longer text/code inputs | More capacity and 64K context | 141B total weights | GSM8K/MATH tests with several attempts and voting |
| DeepSeekMoE 16B | Text/code at a smaller active size | Small experts plus shared experts | Small calculations and uneven loads can slow it | Same-data comparison with dense DeepSeek 7B |
| DeepSeek-V2 | Text/code and longer prompts | Compresses attention storage and uses sparse experts | Needs suitable MLA software | DeepSeek API and reported system comparisons |
| DeepSeek-V3 | Large-scale language/code help | Joint model/software/hardware design | Large setup; headline cost excludes earlier research | Live platform with a published training-cost account |
| Grok-1 | Research on public large-model weights | Released layout and weights | Incomplete training recipe and simple reference code | xAI HumanEval/MMLU report |
| DBRX | Business text/code and searched-document answers | Small experts and described serving setup | Data and other changes also explain gains | Natural Questions with search; early SQL-product use |
| Snowflake Arctic | SQL, code, and instructions | Focus on business tasks | Very large full weight storage | Spider-based enterprise research comparison |
| OLMoE | Research that others can inspect | Data, code, checkpoints, and logs | Active size does not give full memory cost | AI2 open training and routing study |

## Coverage and continuation manifest

This chapter covers eleven self-supervised MoE models and families.
The original supervised mixtures and GShard are in
[the supervised neural chapter](02-supervised-neural.md).

Continue to [Part 4: the MoE deep dive](08-moe-deep-dive.md) for how
routers choose, how experts share work, and how costs should be compared.
The [foundation-model chapter](06-foundation-models.md) covers dense and
closed families, including disclosed Gemini MoE details without invented sizes.

Later additions could cover more image/audio experts, task-based routing,
Qwen sparse releases, newer Grok/DeepSeek versions, and independent serving
comparisons. This edition does not claim to include every release.
