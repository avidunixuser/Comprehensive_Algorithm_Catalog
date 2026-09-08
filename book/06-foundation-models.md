# 3. Unsupervised Learning Algorithms: Language and Multimodal Foundation Models

This volume continues Part 3 with sections 3.11-3.14. Its organizing principle is representation and generative pretraining, not the assertion that every model sold under a foundation-model brand is unsupervised. The evidence policy is dated **2026-09-08**. The releases examined below are explicitly historical; none is described as the latest model available on that date.

**What an entry names.** An algorithm is a procedure, such as masked-token prediction. An architecture is a computational graph, such as a Transformer encoder. A checkpoint is a particular set of learned weights. Instruction tuning, distillation, and preference optimization are training stages. An assistant product adds interfaces, policies, retrieval, tools, and sometimes several models. A serving API transports requests to an implementation; its model identifier is not a disclosure of that implementation. Thus ChatGPT is not synonymous with the GPT-3 architecture, Claude's API is not its training algorithm, and Amazon Bedrock is not an Amazon Titan neural network.

**Training signals.** For tokenized text $`x_{1:T}`$, causal language-model pretraining commonly minimizes


$$
\mathcal L_{\mathrm{CLM}}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$


The target is taken from the text itself: this is self-supervision, not human annotation of each prediction. Masked language modeling instead predicts selected original tokens from a corrupted input; span-denoising encoder-decoders predict missing spans or reconstruct the complete original sequence. These objectives do not establish factual truth, copyright permission, or neutrality of the corpus.

Supervised fine-tuning (SFT) fits desired responses to prompts, usually with conditional token cross-entropy and an explicit choice about which prompt tokens contribute to loss. Distillation trains a student on a teacher's probability distributions, generated answers, or selected reasoning demonstrations. Training on synthetic text alone does not establish that a particular checkpoint used logit-matching distillation. RLHF uses human feedback, often through a learned reward model and a policy-optimization stage; RLAIF substitutes or supplements AI-generated feedback. Constitutional AI is one published approach to producing critiques, revisions, and preference feedback. DPO directly optimizes preference pairs rather than requiring an online policy-gradient loop. These are not interchangeable names for all alignment. Standalone reinforcement learning lies outside the book's three primary supervision categories; its use after pretraining is identified rather than relabeled as unsupervised learning.

**Shared computational assumptions.** Let $`B`$ be batch size, $`T`$ sequence length, $`d`$ hidden width, $`L`$ layers, $`h`$ query heads, $`p`$ parameters, and $`V`$ vocabulary size. With feed-forward width proportional to $`d`$, a dense full-attention Transformer forward pass costs approximately $`O(BL(Td^2+T^2d))`$, plus vocabulary scoring of up to $`O(BTdV)`$ when that head executes. Tying input/output weights saves parameters, not this projection arithmetic. Conventional materialized attention needs $`O(BLhT^2)`$ attention storage; memory-efficient kernels need not materialize that matrix, but do not make exact full-attention arithmetic linear. Backpropagation adds activations, gradients, and optimizer state, not just the parameter storage.

For an encoder-decoder with source length $`T_s`$ and target length $`T_o`$, the attention terms also include $`T_sT_o`$ cross-attention. Autoregressive generation is sequential in output tokens. With a KV cache, a dense decoder's next-token block work is approximately $`O(L(d^2+Td))`$, plus the vocabulary head. Grouped-query attention (GQA) reduces stored keys and values, not the entire feed-forward computation. Sliding windows change attention connectivity; mixture-of-experts (MoE) routing changes which parameter blocks execute. Neither is a synonym for the other.

These formulas are analytic reference models, **not estimates of undisclosed vendor architectures**. A weight-only BF16 storage estimate is $`2p`$ bytes, excluding cache, temporary buffers, gradients, and optimizer state. Benchmark scores below are author-reported results whose source and protocol were checked; the models were not rerun for this book. Results across different prompts, contamination controls, sampling schemes, or dataset versions are not a universal leaderboard.

## 3.11 Encoder and Encoder-Decoder Pretraining

These methods learn from unlabelled text by constructing prediction targets. Their downstream classifiers, answer extractors, and summarizers can nevertheless require supervised training.

### 3.11.1 BERT

**Name:** BERT, Bidirectional Encoder Representations from Transformers; representative checkpoints are the original English BERT-Base and BERT-Large, not later domain adaptations.

**Category & sub-category:** Unsupervised/self-supervised representation learning; bidirectional encoder pretraining with masked-token and sentence-pair objectives.

**Originating paper/vendor/year:** Jacob Devlin and colleagues at Google, [2018 preprint](https://arxiv.org/html/1810.04805v2), subsequently published at NAACL 2019. Google released the implementation and pretrained weights; BERT is an architecture-and-training formulation, not a hosted assistant.

**Core mechanism:** WordPiece tokens receive token, segment, and position embeddings. Bidirectional self-attention lets the representation of a candidate answer use context on both sides. The original procedure selects 15% of tokens for prediction, replacing most with a mask while sometimes substituting a random token or retaining the original. A separate next-sentence prediction (NSP) head discriminates actual sentence continuations from sampled alternatives.

**Inputs/outputs and typical data types:** Text sequences or sentence pairs enter as token IDs and masks. Outputs are contextual token vectors, masked-token distributions, or task-head predictions such as entity labels, sentence classes, and answer-span endpoints. BERT is not natively a left-to-right response generator.

**Strengths and limitations:** It is effective when the entire input is available, particularly for classification and extraction. A span head can restrict an answer to source text, unlike unconstrained generation. Original checkpoints have short context, English corpus biases, and a pretraining/fine-tuning mismatch caused by artificial masks; source extraction still does not ensure that the selected statement is true.

**Computational complexity / scalability notes:** The full-attention encoder follows the shared $`O(BL(Td^2+T^2d))`$ model. Classification adds a small head; answer extraction adds endpoint scores rather than an autoregressive loop. Long documents need windows or an architectural modification, not merely a larger input buffer.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In the Google BERT study, **SQuAD 1.1** tests reading comprehension on Wikipedia passages. Passage plus question -> fine-tuned BERT -> start/end token scores -> highest-scoring valid answer span -> comparison with reference answers. Bidirectional representations are a technical fit for locating an answer whose identity depends on surrounding words; this is the authors' benchmark experiment, not a deployment claim. [Table 2](https://arxiv.org/html/1810.04805v2#S4.T2) reports the single BERT-Large model at **84.1 exact match / 90.9 F1 on development**, compared with BERT-Base's **80.8 / 88.5**. The often-quoted **93.2 test F1** belongs to an **ensemble with additional TriviaQA training**, not to an ordinary single BERT checkpoint. No business KPI is reported.

**Notable vendor implementations/libraries:** Google's [original BERT repository](https://github.com/google-research/bert) and Hugging Face Transformers. A library's support is not evidence that every Google product uses this exact checkpoint.

**Architecture diagram description:** `WordPiece + position + segment embeddings -> [bidirectional attention -> residual/LayerNorm -> GELU feed-forward -> residual/LayerNorm] x L -> MLM/NSP or downstream head`.

**Activation functions used and why:** GELU in the feed-forward network provides smooth nonlinear gating; attention and classification use softmax. The sigmoid-like interpretation of a binary output does not replace the encoder's GELU.

**Loss function(s):** Sum of mean masked-token negative log likelihood and NSP classification loss during pretraining. SQuAD fine-tuning uses the sum of the correct start- and end-position log-losses.

**Optimization algorithm(s):** The original paper specifies Adam with learning rate $`10^{-4}`$, $`\beta_1=0.9,\beta_2=0.999`$, 10,000 warmup steps, and linear decay. Task fine-tuning uses separate, much shorter schedules.

**Regularization techniques:** Weight decay 0.01, dropout 0.1, LayerNorm, and stochastic corruption in the original recipe. LayerNorm stabilizes representations; it is not itself evidence of improved generalization.

**Backpropagation considerations:** All encoder layers can be fine-tuned. Small labeled datasets are sensitive to learning rate and initialization; padding must not receive prediction loss. Residual paths aid gradient transport through the deep encoder.

**Parameter count / scaling behavior:** Original Base: **12 layers, width 768, about 110M parameters**. Large: **24 layers, width 1,024, about 340M**. These figures do not describe every model with "BERT" in its name.

**Training paradigm:** Self-supervised pretraining on BooksCorpus and English Wikipedia, followed by supervised task adaptation. NSP uses automatically constructed labels, not manually annotated sentence relations.

**Hardware/parallelism considerations:** The paper reports TPU-pod pretraining. Modern fine-tuning can use accelerators with data parallelism and mixed precision; full-length attention, rather than the small output head, usually drives activation memory.

### 3.11.2 RoBERTa

**Name:** RoBERTa, a Robustly Optimized BERT Pretraining Approach; the representative is the original 2019 large English encoder.

**Category & sub-category:** Unsupervised/self-supervised representation learning; optimized masked-language-model encoder pretraining.

**Originating paper/vendor/year:** Yinhan Liu and colleagues, Facebook AI and the University of Washington, [2019 technical report](https://arxiv.org/html/1907.11692v1). This is primarily a controlled recipe study and set of checkpoints, not a new attention algorithm.

**Core mechanism:** Retain BERT-style bidirectional encoding but remove NSP, dynamically resample masks, use more text and larger batches, and train longer with full-length sequences. Byte-level BPE replaces BERT's original tokenization. The study demonstrates that data and optimization can explain apparent architectural progress.

**Inputs/outputs and typical data types:** Byte-BPE text tokens produce contextual embeddings and masked-token predictions. Supervised heads produce sentiment classes, entailment labels, multiple-choice scores, or answer spans.

**Strengths and limitations:** It is a strong task-adaptable encoder with relatively simple architecture. However, its increased compute and data requirements are central to the result: one cannot attribute all improvements to removing NSP. Like BERT, it is not a general conversational generator, and larger web corpora retain bias and contamination risks.

**Computational complexity / scalability notes:** Same dense encoder asymptotics as BERT, but a bigger vocabulary and training budget change actual costs. Dynamic masking increases the diversity of targets without reducing per-step attention or feed-forward computation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Facebook AI's **SQuAD 2.0** experiment requires both answer extraction and detecting questions unsupported by a passage. Question plus Wikipedia passage -> RoBERTa span and answerability predictions -> return a span or select no-answer -> score against the dataset. An extractive model is technically suitable when provenance must stay inside the passage; a generative decoder can instead invent an unsupported answer. [Table 6](https://arxiv.org/html/1907.11692v1#S5.T6) reports the single-model development result **86.5 EM / 89.4 F1**, versus **79.0 / 81.8** for the listed BERT-Large baseline. The RoBERTa result uses the provided SQuAD data without extra task-training data. The **86.8 / 89.8 test** result is a different row and must not be relabeled as development performance. This is not a measured customer-support deployment.

**Notable vendor implementations/libraries:** Facebook/Meta fairseq and Hugging Face Transformers. The [released configuration](https://huggingface.co/FacebookAI/roberta-large/blob/main/config.json) identifies the large model's graph; the Hugging Face explanatory card is implementation-provider documentation rather than an additional originating paper.

**Architecture diagram description:** `byte-BPE + learned positions -> bidirectional Transformer encoder blocks -> contextual vectors -> masked-token or supervised task head`; there is no pretrained NSP head in the RoBERTa recipe.

**Activation functions used and why:** GELU feed-forward activations preserve the BERT-style nonlinear transform; attention and vocabulary prediction use softmax. The released large configuration specifies GELU.

**Loss function(s):** Cross-entropy at selected masked positions. NSP loss is removed. SQuAD adaptation adds span and answerability training; it is not still "unsupervised" simply because the backbone was pretrained.

**Optimization algorithm(s):** Adam with warmup and linear learning-rate decay. The report tunes peak learning rate, warmup, and epsilon by setting and finds $`\beta_2=0.98`$ helpful for large-batch stability. A single numerical schedule should not be imposed on every ablation in the paper.

**Regularization techniques:** Dynamic masking, weight decay, and dropout; the released large configuration has 0.1 hidden and attention dropout. LayerNorm and residual connections remain.

**Backpropagation considerations:** Large-batch mixed-precision training requires stable optimizer numerics and loss handling. Fine-tuning can update the whole encoder; an apparent architecture comparison is confounded if its optimization and training-token budgets differ.

**Parameter count / scaling behavior:** The [official fairseq release](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md) lists approximately **355M parameters for Large** and **125M for Base**. Large has **24 layers and width 1,024**; its larger vocabulary makes its total different from BERT-Large's. Large-model results do not automatically apply to Base.

**Training paradigm:** Self-supervised MLM, with downstream supervised adaptation. The strongest reported pretraining setting uses **160 GB of text and 500,000 updates**, not the smaller Books/Wikipedia-only ablation.

**Hardware/parallelism considerations:** The report describes mixed-precision training on DGX-1 systems with V100 GPUs and inter-node communication. Data parallelism increases throughput but does not remove the need to store each replica or shard its state.

### 3.11.3 T5

**Name:** T5, Text-to-Text Transfer Transformer; original T5-Base is the architectural reference, with the original T5-11B system used to illustrate scaling.

**Category & sub-category:** Unsupervised/self-supervised pretraining; span-denoising encoder-decoder and a unified text-to-text task interface.

**Originating paper/vendor/year:** Colin Raffel and colleagues at Google, 2019 preprint and [JMLR 2020 paper](https://jmlr.org/papers/v21/20-074.html). The [full report](https://arxiv.org/html/1910.10683v4) distinguishes the baseline experiments from its final scaled systems.

**Core mechanism:** Express tasks as text input and text output, often with a task prefix. In span corruption, replace contiguous missing spans with distinct sentinel tokens; the target contains the removed spans and sentinels rather than a complete copy of the input. A bidirectional encoder supplies representations to an autoregressive decoder.

**Inputs/outputs and typical data types:** SentencePiece text inputs can encode questions, passages, translation instructions, or classification tasks. Outputs are generated text: a translation, short answer, summary, or label word. A label such as `entailment` is an output token sequence, not a separate indispensable classifier architecture.

**Strengths and limitations:** One interface supports varied tasks and efficient corruption targets. Unlike an extractive span head, it can answer in a normalized form not literally present in the input. That flexibility also permits unsupported generation, invalid class strings, and exposure bias; task formatting and decoding policy matter.

**Computational complexity / scalability notes:** Encoder self-attention, decoder self-attention, and source-target cross-attention all contribute. Short denoising targets save decoder work relative to reconstructing every source token. Long-input inference still requires encoding the source and maintaining decoder state.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Google's T5 study evaluates **SQuAD 1.1** reading comprehension through text generation. A question and its Wikipedia context, formatted as a text task, enter T5; the decoder emits an answer string; the evaluator normalizes it and compares it with reference answers. This tests whether a uniform generative interface can replace a task-specific BERT span head. The final-results **Table 14** reports **T5-11B at 91.26 EM / 96.22 F1**, compared with **T5-Base at 85.44 / 92.08**, on the **SQuAD validation set**. The [report explicitly makes SQuAD an exception](https://arxiv.org/html/1910.10683v4) to its final test-set evaluations because the benchmark server could not run the largest models. These are scaled, task-adapted systems, not raw denoising checkpoints. The final recipe includes supervised task mixtures and fine-tuning, so the result cannot be credited to unlabelled C4 alone. No production QA KPI is established.

**Notable vendor implementations/libraries:** Google's [text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer), Mesh TensorFlow in the original work, and Hugging Face Transformers. FLAN-T5 is a later instruction-tuned derivative, not another name for the original T5 training recipe.

**Architecture diagram description:** `corrupted text + sentinels -> bidirectional encoder -> cross-attended causal decoder -> removed spans + sentinels`; downstream tasks replace the corruption input with task text.

**Activation functions used and why:** Original T5 uses ReLU feed-forward networks. Later T5 variants can change the feed-forward nonlinearity; their gated activations must not be back-projected onto the original model. Attention and output distributions use softmax.

**Loss function(s):** Teacher-forced conditional token cross-entropy over target strings. The baseline learns span denoising; supervised text-to-text tasks supply different targets under the same loss.

**Optimization algorithm(s):** Adafactor. The baseline pretraining schedule is $`1/\sqrt{\max(s,10^4)}`$, with $`s`$ the update index, followed by separately specified fine-tuning at constant 0.001. This equation describes inverse-square-root decay, not exponential decay.

**Regularization techniques:** Baseline dropout 0.1, span corruption, residual paths, relative position biases, and T5's simplified pre-normalization without mean subtraction. Data cleaning is separately important and is not a dropout substitute.

**Backpropagation considerations:** Teacher forcing permits parallel target-token loss calculation during training, but inference remains autoregressive. Gradients cross the decoder-to-encoder attention interface; shared embeddings and masked targets need consistent tokenization.

**Parameter count / scaling behavior:** Original T5-Base is about **220M parameters** with **12 encoder and 12 decoder blocks**, width 768. The named **11B** system is much larger; its scores and hardware costs are not Base-model properties.

**Training paradigm:** Primarily self-supervised denoising, followed by task adaptation; the paper also explicitly investigates multitask supervised/self-supervised mixtures. Instruction tuning and distillation are optional later stages, not intrinsic consequences of a text-to-text interface.

**Hardware/parallelism considerations:** The original study uses TPU infrastructure and model parallelism for large configurations. Adafactor reduces optimizer-state requirements; it does not eliminate activation, attention, or encoder-decoder memory.

### 3.11.4 BART

**Name:** BART, Bidirectional and Auto-Regressive Transformers; representative releases are `bart.base`, `bart.large`, and the separately fine-tuned `bart.large.cnn`.

**Category & sub-category:** Unsupervised/self-supervised learning; denoising sequence-to-sequence pretraining.

**Originating paper/vendor/year:** Mike Lewis and colleagues at Facebook AI, [2019 preprint](https://arxiv.org/html/1910.13461v1), published at [ACL 2020](https://aclanthology.org/2020.acl-main.703/).

**Core mechanism:** Corrupt a document, then reconstruct the original complete text. The paper compares deletion, masking, infilling, sentence permutation, and other noise functions. Its large model combines text infilling with sentence permutation. Unlike T5's sentinel-target formulation, the decoder reconstructs the whole original sequence.

**Inputs/outputs and typical data types:** Corrupted token sequences during pretraining; articles, conversations, or other source text during supervised adaptation. Outputs are reconstructed documents or generated summaries and responses. Additional task heads can support classification.

**Strengths and limitations:** Bidirectional source encoding plus causal output generation is well matched to rewriting and summarization. Flexible corruption need not preserve source length. Autoregressive generation is slower than a single extractive pass and can hallucinate; ROUGE overlap is not a measure of factual consistency.

**Computational complexity / scalability notes:** Use the shared encoder-decoder model with $`T_s^2`$, $`T_o^2`$, and $`T_sT_o`$ attention terms and dense feed-forward costs. Complete-text reconstruction can involve more target computation than T5-style removed-span prediction. Beam search increases inference work and cache storage.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Facebook's **CNN/Daily Mail** experiment compresses news articles into highlights. Article -> supervised `bart.large.cnn` -> generated summary -> ROUGE comparison with the held-out reference summary. Generation is technically preferable to lead-sentence extraction when the summary must fuse information; it is not guaranteed to preserve every fact. The [official release table](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md) identifies the **test set, no additional task data** and reports **44.16 ROUGE-1, 21.28 ROUGE-2, 40.90 ROUGE-L**, versus **42.13 / 19.60 / 39.18** for BERTSUMEXTABS. This is a benchmark result, not evidence that a newsroom reduced editing time. A human editor or factuality check would remain a deployment requirement, not a measured step in this benchmark.

**Notable vendor implementations/libraries:** Facebook/Meta fairseq and Hugging Face Transformers, including `facebook/bart-large-cnn`. The `.cnn` and `.xsum` suffixes identify supervised adaptation, not distinct pretraining architectures.

**Architecture diagram description:** `noised document -> bidirectional encoder -> causal decoder with encoder cross-attention -> original document`; replace the reconstruction target with a summary during fine-tuning.

**Activation functions used and why:** GELU replaces the original Transformer's ReLU in BART's feed-forward layers, providing smooth nonlinear transformations; softmax supplies attention weights and output probabilities.

**Loss function(s):** Conditional negative log likelihood of the original document during pretraining. The official summarization fine-tuning example uses label-smoothed cross-entropy with smoothing 0.1.

**Optimization algorithm(s):** The paper does not independently enumerate every large-model pretraining optimizer setting. The [published CNN/Daily Mail fine-tuning recipe](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md) specifies Adam, $`(0.9,0.999)`$, learning rate $`3\times10^{-5}`$, 500 warmup updates, and polynomial decay over 20,000 updates. These are **fine-tuning**, not reconstructed pretraining hyperparameters.

**Regularization techniques:** Corruption, residuals, and LayerNorm; the large pretraining run disables dropout for its final 10%. The cited summarization recipe uses dropout 0.1 and weight decay 0.01.

**Backpropagation considerations:** Teacher forcing trains all target positions in parallel; gradients flow through both stacks. The summarization recipe clips gradient norm at 0.1. Padding, source truncation, and label smoothing change the effective training problem.

**Parameter count / scaling behavior:** The official release lists approximately **140M** for Base and **400M** for Large, with respectively **6/6** and **12/12 encoder/decoder layers**. These rounded release figures are not exact tensor counts.

**Training paradigm:** Self-supervised denoising on a large text corpus, followed by supervised generation or classification. A pretrained BART model alone is not the CNN/Daily Mail benchmark system.

**Hardware/parallelism considerations:** The documented fine-tuning example targets eight 32-GB V100 GPUs with FP16 and gradient accumulation. This is a reproducible example configuration, not a universal hardware minimum or the original pretraining cluster specification.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| BERT | Text pairs and token-level annotation | Bidirectional features and source-constrained span heads | Short original context; not a native generator | Google SQuAD 1.1 benchmark |
| RoBERTa | Text classification and extractive QA | Stronger MLM recipe without changing the basic encoder | Gains depend heavily on data and training budget | Facebook AI SQuAD 2.0 benchmark |
| T5 | Tasks expressible as text-to-text mappings | Unified interface and efficient span-denoising targets | Generative answers need validation; two stacks cost memory | Google SQuAD text-to-text benchmark |
| BART | Documents, summaries, and rewriting pairs | Bidirectional source understanding with flexible generation | Hallucination and sequential decoding | CNN/Daily Mail summarization benchmark |

## 3.12 Public Decoder Research and Base-Model Families

"Public" here means that specified research facts, implementations, or checkpoints are public. It does **not** mean that GPT-3 weights or GPT-4 internals are open. The dense models in this category must also be distinguished from similarly branded MoE releases in [the MoE model volume](07-moe-models.md).

### 3.12.1 GPT Family

**Name:** Generative Pre-trained Transformer family, covering GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), and the boundary of public GPT-4 (2023) disclosure. GPT-3 175B supplies the detailed large-scale training example.

**Category & sub-category:** Self-supervised autoregressive language modeling, with release-dependent supervised or preference-based adaptation. Sparse attention in GPT-3 is not mixture-of-experts routing.

**Originating paper/vendor/year:** OpenAI's [GPT-1 implementation](https://github.com/openai/finetune-transformer-lm), [GPT-2 model card](https://github.com/openai/gpt-2/blob/master/model_card.md), Brown and colleagues' [2020 GPT-3 paper](https://arxiv.org/html/2005.14165v4), and the [2023 GPT-4 technical report](https://arxiv.org/html/2303.08774v6). These sources expose different amounts of information.

**Core mechanism:** GPT-1 combines next-token pretraining with supervised task fine-tuning. GPT-2 investigates task performance from text prompts after WebText pretraining. GPT-3 scales this approach and evaluates in-context demonstrations **without gradient updates**. GPT-4 is publicly described as Transformer-based, next-token pretrained, multimodal, and subsequently aligned with RLHF; its detailed architecture is withheld. None of these facts makes a contemporary ChatGPT session equivalent to a raw GPT-3 model.

**Inputs/outputs and typical data types:** GPT-1/2/3 operate on text tokens, including code and task examples. A causal head returns next-token probabilities and generated continuations. GPT-4's report includes image-plus-text input and text output; this must not be generalized to every historical or modern endpoint's modality support.

**Strengths and limitations:** A single generative interface can perform tasks without a task-specific head, especially with demonstrations. Its outputs remain probabilistic and can be fabricated. In-context examples consume context; they do not durably retrain the weights. Proprietary endpoint revisions and hidden system behavior complicate reproducibility.

**Computational complexity / scalability notes:** GPT-1/2 use causal Transformer computation. GPT-3 alternates dense and locally banded sparse attention, so its exact attention cost depends on that pattern; dense feed-forward work remains. Cached decoding still pays for each generated token. GPT-4's actual FLOPs, cache layout, and memory requirements are **not publicly disclosed**.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In OpenAI's **LAMBADA** evaluation, a passage provides context for predicting its final word. Passage -> GPT-3 continuation probabilities -> predicted final word -> exact-match decision on the test item. Unlike a separately trained cloze classifier, the language model can attempt this using its existing prediction interface. [Table 3.2](https://arxiv.org/html/2005.14165v4) reports **76.2% zero-shot test accuracy** and **86.4% few-shot**, versus the cited prior-best **68.0%**. The few-shot condition uses a different cloze-style format and development-set demonstrations, not a gradient-trained adaptation. The table is not proof that prompting alone always improves performance; one-shot scores are lower here. Benchmark contamination limitations are discussed in the paper. No business productivity gain is established.

**Notable vendor implementations/libraries:** OpenAI released GPT-1 and GPT-2 code/weights; Hugging Face implements these historical graphs. OpenAI's hosted API and ChatGPT are access/product layers, not public GPT-4 architecture implementations.

**Architecture diagram description:** GPT-1: `embeddings -> causal attention/MLP blocks with post-normalization -> LM or supervised head`. GPT-2/3 use pre-normalized residual blocks; GPT-3 adds alternating attention patterns. GPT-4: `text/image input -> proprietary Transformer-based model -> text`; the interior graph is not publicly disclosed.

**Activation functions used and why:** The public GPT-1/2/3 formulations use GELU feed-forward nonlinearities and softmax distributions. GPT-4's activation functions are **not publicly disclosed**; GELU must not be inferred from the family name.

**Loss function(s):** GPT-1/2/3 base models use causal token negative log likelihood; GPT-1 also has supervised task losses during adaptation. GPT-4 discloses next-token pretraining and RLHF, but not a reproducible complete loss specification or preference-stage weighting.

**Optimization algorithm(s):** GPT-3 uses Adam, $`(0.9,0.95)`$, gradient-norm clipping 1.0, 375M warmup tokens, and cosine decay to 10% of the peak over 260B tokens. The 175B model's peak learning rate is $`6\times10^{-5}`$. GPT-1 fine-tuning defaults in its repository are a different recipe. GPT-4's optimizer and schedule are **not publicly disclosed**.

**Regularization techniques:** GPT-1's released training code includes dropout and weight decay. GPT-3 specifies weight decay 0.1, corpus filtering/deduplication, pre-normalization, and careful initialization. GPT-4's dropout, normalization layout, and regularization coefficients are **not publicly disclosed**.

**Backpropagation considerations:** Large historical models need distributed gradients and activation management. GPT-3 benchmark in-context learning performs forward inference, not backpropagation. No claim about GPT-4's gradient precision or checkpointing strategy follows from its API.

**Parameter count / scaling behavior:** GPT-1 has 12 layers of width 768; its [published tensor shapes](https://github.com/openai/finetune-transformer-lm/blob/master/model/params_shapes.json) sum to approximately 117M parameters. GPT-2's final card lists **124M, 355M, 774M, and 1.5B** releases. GPT-3 175B has **96 layers, width 12,288**, and was trained for **300B tokens**. GPT-4 total/active counts are **not publicly disclosed**.

**Training paradigm:** Historical base self-supervision; GPT-1 task SFT; GPT-3 prompt-only evaluation; GPT-4 disclosed RLHF after pretraining. Distillation and later reasoning-model recipes must be identified separately rather than attributed to every GPT.

**Hardware/parallelism considerations:** The GPT-3 paper reports V100 training with matrix and layer model parallelism on a high-bandwidth cluster. GPT-4 hardware and training compute are explicitly withheld. API latency cannot recover either reliably.

### 3.12.2 LLaMA / Llama

**Name:** Meta's LLaMA/Llama family. The detailed reference is **LLaMA 1, February 2023**, especially its nominal 7B and 65B base checkpoints; later generations are distinguished explicitly.

**Category & sub-category:** Self-supervised causal language-model pretraining; dense base-model family with separately post-trained assistants.

**Originating paper/vendor/year:** Hugo Touvron and colleagues, Meta AI, [LLaMA, 2023](https://arxiv.org/html/2302.13971v1). [Llama 2](https://arxiv.org/html/2307.09288v2) is a separate 2023 release; [the Llama 3 report](https://arxiv.org/html/2407.21783v1) documents the 2024 generation, including its dense 405B model.

**Core mechanism:** Predict the next token with a causal Transformer. LLaMA 1 combines pre-normalization with RMSNorm, rotary positions, and a gated feed-forward network, emphasizing strong smaller models trained on substantial token budgets. Llama 2-Chat adds dialogue alignment; Llama 3 changes scale, data, tokenization, and post-training. These are not merely aliases for the original 7B checkpoint.

**Inputs/outputs and typical data types:** Base text and code tokens -> continuation probabilities or generated text. Chat variants require their documented conversation templates. The 2024 report's multimodal experiments must not be treated as evidence that the original text-only LLaMA checkpoint accepts images.

**Strengths and limitations:** Downloadable weights enable local evaluation, adaptation, and control of inference. Dense models are comparatively straightforward to shard. License terms and release availability vary, and an open-weight base model is neither a complete assistant nor a guarantee of transparent training data.

**Computational complexity / scalability notes:** LLaMA 1 follows the dense full-attention model. Larger $`p`$ increases weight traffic and feed-forward work; longer context adds attention/cache costs. Later use of GQA in specified Llama models is not a property of every LLaMA 1 attention layer.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Meta evaluates **MMLU**, multiple-choice questions spanning academic and professional subjects. Five task demonstrations plus a question and options -> LLaMA scoring -> selected answer -> comparison with the test key. This probes reusable knowledge without training a separate subject classifier. [The original paper's Table 9](https://arxiv.org/html/2302.13971v1) reports **63.4% average five-shot accuracy for LLaMA-65B**, versus **46.9% for LLaMA-13B**. Greater capacity improves this benchmark, but the paper also shows larger competitors ahead; it does not establish that 65B is optimal under a particular organization's budget. These scores describe 2023 LLaMA 1, not Llama 2, Llama 3, or a clinical/professional qualification.

**Notable vendor implementations/libraries:** Meta's release repositories, Hugging Face Transformers, and local runtimes such as llama.cpp. Runtime support does not imply that quantized derivatives retain the original benchmark score.

**Architecture diagram description:** LLaMA 1: `token embeddings -> [RMSNorm -> causal self-attention with RoPE -> residual; RMSNorm -> SwiGLU FFN -> residual] x L -> RMSNorm -> token head`.

**Activation functions used and why:** SwiGLU multiplies a SiLU-transformed gate by a second projection, supplying a learnable nonlinear feature gate. The original report adjusts intermediate width to manage parameter cost. Softmax forms attention weights and token distributions.

**Loss function(s):** Causal token cross-entropy for the base reference. Llama 2-Chat's SFT and preference/RLHF stages are additional objectives, not losses secretly present in all original base checkpoints.

**Optimization algorithm(s):** LLaMA 1 uses AdamW, $`(0.9,0.95)`$, 2,000 warmup updates, and cosine decay to 10% of peak. The nominal 7B/13B runs use peak $`3\times10^{-4}`$; the larger two use $`1.5\times10^{-4}`$. These are **LLaMA 1** settings.

**Regularization techniques:** The original recipe specifies weight decay 0.1 and gradient clipping 1.0. RMSNorm, residuals, data filtering, and deduplication address different numerical and data-quality concerns; an undocumented family-wide dropout rate is not assumed.

**Backpropagation considerations:** Efficient attention, selective activation recomputation, and communication overlap reduce training overhead. Local SFT can update all weights or use adapters, but parameter-efficient tuning does not remove the frozen backbone's inference-memory cost.

**Parameter count / scaling behavior:** The original table lists approximately **6.7B, 13.0B, 32.5B, and 65.2B**, often named 7B/13B/33B/65B. The first two see **1.0T tokens**, the latter two **1.4T**. Llama 2 spans 7B-70B; the 2024 Llama 3 report's **405B dense** model is a different checkpoint.

**Training paradigm:** Self-supervised base pretraining; instruction and preference stages are separately named. MoE Llama releases and their routing are continued in [the MoE volume](07-moe-models.md), not folded into this dense recipe.

**Hardware/parallelism considerations:** LLaMA 1 reports the 65B training run on **2,048 A100 80-GB GPUs**. That is a training configuration, not a minimum for inference. Quantization, tensor sharding, batch size, and context determine local serving requirements.

### 3.12.3 Mistral Dense Models

**Name:** Mistral dense-language-model family, represented by **Mistral-7B-v0.1, 2023**. Mixtral is excluded from this entry.

**Category & sub-category:** Self-supervised causal base modeling; dense decoder with grouped-query and sliding-window attention.

**Originating paper/vendor/year:** Albert Q. Jiang and colleagues at Mistral AI, [Mistral 7B technical report, 2023](https://arxiv.org/html/2310.06825v1), with the [v0.1 base model card](https://huggingface.co/mistralai/Mistral-7B-v0.1).

**Core mechanism:** A dense Transformer predicts subsequent tokens while reducing attention-state costs through GQA and local sliding windows. Across stacked layers, information can propagate beyond one layer's window. No expert router chooses among alternative FFNs: sliding attention is not MoE.

**Inputs/outputs and typical data types:** Byte-fallback BPE text and code -> token probabilities and continuations. The base model lacks the conversation tuning of Mistral-Instruct. Later dense Mistral checkpoints may change vocabulary, context, or multimodal support; the v0.1 settings below do not specify them.

**Strengths and limitations:** Useful capability in a relatively small dense checkpoint, with lower KV storage from GQA. Local attention can improve long-sequence efficiency but does not make distant-token access equivalent to full attention at every layer. A base continuation can be unsafe, ungrounded, or poorly instruction-following.

**Computational complexity / scalability notes:** With window $`W`$, local attention work is approximately $`O(BLT\min(T,W)d)`$, alongside $`O(BLTd^2)`$ projection/FFN work. A rolling local cache can cap per-layer KV history at $`W`$ in a compatible implementation. Chunking and cache correctness matter; the dense FFN still executes for every token.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Mistral's **MMLU** experiment evaluates multi-subject question answering under a shared re-run pipeline. Five demonstrations plus a test question and candidate answers -> Mistral-7B-v0.1 -> answer score/selection -> test accuracy. A smaller dense model is technically attractive where local memory and latency constrain a knowledge assistant; that is this book's deployment rationale, not a documented customer purchase decision. [Table 2](https://arxiv.org/html/2310.06825v1) reports **60.1% five-shot MMLU**, versus the paper's **55.6% Llama 2 13B** baseline. Those figures cannot be mixed with another paper's differently prompted Mistral scores. No production KPI or proof of universal superiority is reported.

**Notable vendor implementations/libraries:** Mistral's public inference code, Hugging Face Transformers, and compatible serving runtimes. A hosted Mistral API alias is not necessarily the downloadable v0.1 model.

**Architecture diagram description:** `BPE embeddings -> dense causal Transformer blocks: RoPE/GQA with window W + gated FFN and RMSNorm/residual paths -> token head`. The [released configuration](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json) specifies 32 layers, width 4,096, 32 query heads, and eight KV heads.

**Activation functions used and why:** The published configuration uses SiLU; the gated FFN is conventionally described as SwiGLU. Smooth gating transforms features while preserving an efficient dense implementation. Softmax operates inside attention and token prediction.

**Loss function(s):** Causal token prediction for the base language model. Full original data weighting and any auxiliary training losses are not publicly disclosed in the cited short release materials. Instruction-model losses cannot be presumed for v0.1 Base.

**Optimization algorithm(s):** The original optimizer, numerical learning-rate schedule, warmup, and clipping recipe are **not publicly disclosed in the cited release report/card**. Compatibility with AdamW-based fine-tuning code is not evidence of the original optimizer.

**Regularization techniques:** RMSNorm and residual paths are public architectural facts. Original dropout, weight-decay coefficients, and detailed data-filtering recipe are not publicly disclosed in those materials.

**Backpropagation considerations:** Correctly masking local attention is necessary for the intended receptive field; KV caches used for inference should not be confused with the training computation graph. Original gradient-recomputation and precision details are not assumed.

**Parameter count / scaling behavior:** Nominally **7B parameters**. The report gives **8,192 training context** and **4,096 attention window**; the released configuration also permits a larger positional limit. Configuration limits alone are not evidence of equal-quality long-context reasoning.

**Training paradigm:** Self-supervised base training, with separately released instruction-tuned derivatives. Mixtral's sparse experts and balancing objectives belong in [section 3.15](07-moe-models.md).

**Hardware/parallelism considerations:** Local weights can be sharded or quantized, while GQA reduces KV traffic. No unpublished training GPU count is asserted. Runtime support for the specific sliding-window semantics must be checked before relying on projected cache savings.

### 3.12.4 Qwen Dense Models

**Name:** Alibaba's Qwen dense family, represented by **Qwen2.5-7B Base, September 2024**; not Qwen2.5-Instruct, Qwen-VL, or a Qwen MoE checkpoint.

**Category & sub-category:** Self-supervised multilingual causal language modeling; dense base models with separately documented post-training.

**Originating paper/vendor/year:** Qwen Team, Alibaba, [Qwen2.5 release/model card, 2024](https://huggingface.co/Qwen/Qwen2.5-7B) and [December 2024 technical report](https://arxiv.org/html/2412.15115v1).

**Core mechanism:** Dense autoregressive Transformers combine GQA, RoPE, QKV bias, RMSNorm, and SwiGLU. The report emphasizes data selection, training scale, mathematical/code data, and long-context adaptation. The family also includes other architectures and specialized branches; "Qwen" alone is insufficient to determine the graph or training stage.

**Inputs/outputs and typical data types:** Multilingual prose, source code, and serialized structured text -> next-token distributions and generated text. A Base checkpoint is not recommended as a ready conversation agent; native image input belongs to specifically multimodal releases.

**Strengths and limitations:** A strong multilingual and technical base can be adapted locally and used under a checkpoint-specific license. However, generated JSON can be invalid, numerical reasoning can fail, and broad language coverage is not a uniform accuracy guarantee. An advertised context limit is not a benchmark of recall at every position.

**Computational complexity / scalability notes:** Dense decoder costs follow the shared model; GQA reduces KV state, not $`p`$ to an "active expert" subset. Long-context positional scaling changes positional handling but does not erase quadratic full-attention arithmetic. Use version-appropriate runtime settings for long context.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The Qwen2.5 report evaluates **GSM8K**, held-out grade-school mathematical word problems. Four demonstrations plus a problem -> Qwen2.5-7B Base -> generated solution/answer -> comparison of the extracted answer with the test key. Reusing a pretrained generator avoids creating a bespoke symbolic parser for every problem wording, although a calculator or solver can be more dependable for arithmetic. The [7B+ base-model table](https://arxiv.org/html/2412.15115v1) reports **85.4% for Qwen2.5-7B**, versus **80.2% for Qwen2-7B**, under the stated **four-shot** protocol. These are base-model benchmark results, not the scores of the Instruct checkpoint and not evidence of improved school outcomes.

**Notable vendor implementations/libraries:** Alibaba's Qwen repositories, Hugging Face Transformers, and compatible inference runtimes. Qwen API products and local base weights have different operational interfaces. License terms must be checked per size and release rather than inferred from this 7B example.

**Architecture diagram description:** `token embedding -> [RMSNorm -> RoPE/GQA with QKV bias -> residual; RMSNorm -> SwiGLU dense FFN -> residual] x 28 -> vocabulary head`.

**Activation functions used and why:** SwiGLU, using SiLU in its gate, supplies nonlinear feature selection; attention and output distributions use softmax. These are public Qwen2.5 dense-model details, not claims about every Qwen vision or MoE component.

**Loss function(s):** Causal token cross-entropy for Base. The report separately discusses SFT and preference/RL stages for post-trained models; their response losses and preference objectives must not be added to the Base checkpoint's description.

**Optimization algorithm(s):** The report discusses scaling-law-based selection of learning rate and batch size, but does not give a complete optimizer and numerical warmup/decay recipe for this specific 7B release. Those settings are **not publicly disclosed in the cited materials**; an earlier Qwen recipe is not substituted.

**Regularization techniques:** Public pre-normalization/RMSNorm and residual paths stabilize the graph. The report documents filtering and data-quality selection. Checkpoint-specific dropout and weight-decay settings are not established by the model card.

**Backpropagation considerations:** Dense training still updates the model's full trainable parameter set. Fine-tuning on conversation text needs role-aware loss masking. Synthetic-data filtering and benchmark decontamination concern target quality, not a different backpropagation rule.

**Parameter count / scaling behavior:** The card specifies **7.61B total parameters**, **6.53B non-embedding**, **28 layers**, and **28 query / four KV heads**. The report's expanded **18T-token corpus scale** is a family-level data statement, not evidence that every size saw an identical token budget.

**Training paradigm:** Self-supervised Base; separate instruction and preference stages. The report includes AI-assisted data generation and selection, but that alone does not establish logit-level distillation of every checkpoint.

**Hardware/parallelism considerations:** Weight-only BF16 storage is approximately **15.2 GB**, calculated as $`2\times7.61`$ billion bytes; useful serving needs extra cache and buffers. Sharding, quantization, and optimized attention change practical capacity. No undocumented original GPU count is assumed.

### 3.12.5 DeepSeek LLM Dense Base

**Name:** **DeepSeek LLM 7B/67B Base**, the original dense family described in the January 2024 report. DeepSeek-V2, V3, and R1 are not alternative names for it.

**Category & sub-category:** Self-supervised bilingual causal language modeling; dense base pretraining with separately released Chat derivatives.

**Originating paper/vendor/year:** DeepSeek AI, [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism, 2024](https://arxiv.org/html/2401.02954v1), with the [67B Base model card](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base).

**Core mechanism:** A LLaMA-like pre-normalized Transformer learns next-token prediction over English and Chinese text. The design study investigates scaling and a multi-step learning-rate schedule. Its dense 67B model uses GQA; the smaller 7B configuration uses ordinary multi-head attention.

**Inputs/outputs and typical data types:** Chinese/English prose, mathematical text, and code tokens -> continuation probabilities and generated text. Chat models add response-oriented adaptation. These text-only base weights do not implement DeepSeek's later MoE routing or reasoning-RL procedures.

**Strengths and limitations:** A published dense architecture and explicit schedule make this a useful scaling-study reference. Bilingual coverage and technical tasks benefit from substantial pretraining, but open weights do not reveal every document, and a base model is not aligned for every interactive use.

**Computational complexity / scalability notes:** Both reference sizes follow dense Transformer scaling; GQA in 67B lowers KV memory but does not sparsify its feed-forward weights. A deep 95-layer stack adds sequential dependency and communication costs even when parameter totals resemble shallower competitors.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** DeepSeek's **HumanEval** experiment tests Python function synthesis. A function signature and natural-language specification -> the 67B Base model -> candidate Python implementation -> unit-test execution -> pass/fail. Generative code completion fits underspecified natural-language interfaces better than a fixed template library, but executable tests and human review remain essential. [Table 5](https://arxiv.org/html/2401.02954v1) reports **42.7% zero-shot pass@1** for DeepSeek LLM 67B Base, versus the authors' **28.7% Llama 2 70B** result. This comparison uses their internal evaluation framework with **greedy generation**; it is not interchangeable with other HumanEval sampling pipelines. It is also not the much higher score reported for a separately adapted Chat model. No commercial engineering-time reduction is claimed.

**Notable vendor implementations/libraries:** DeepSeek's [DeepSeek-LLM repository](https://github.com/deepseek-ai/DeepSeek-LLM) and Hugging Face Transformers. The historical repository's code license and model license are distinct; neither specifies the internals behind every later DeepSeek API endpoint.

**Architecture diagram description:** `token embeddings -> pre-normalized RMSNorm/RoPE causal attention + SwiGLU FFN residual blocks -> token head`; 67B uses grouped KV heads, while 7B has as many KV as query heads.

**Activation functions used and why:** SwiGLU supplies gated nonlinear FFN transformations; softmax forms attention and vocabulary probabilities. This is the original dense report's published choice, not an extrapolation from V3.

**Loss function(s):** Causal token cross-entropy for Base. The report describes SFT and DPO for Chat as additional stages; DPO is not the same procedure as the later R1 reasoning-RL training.

**Optimization algorithm(s):** AdamW, $`(0.9,0.95)`$, weight decay 0.1, and 2,000 warmup steps. Peak rates are **$`4.2\times10^{-4}`$ for 7B** and **$`3.2\times10^{-4}`$ for 67B**. At 80% of the token budget the rate drops to 31.6% of peak; at 90%, to 10%.

**Regularization techniques:** Weight decay, gradient-norm clipping 1.0, RMSNorm, and residual paths. Corpus construction and benchmark overlap controls are separate from numerical regularization.

**Backpropagation considerations:** The report's clipping limits gradient spikes. The stepwise schedule permits reuse of an earlier training phase when extending a run; this is a documented engineering motivation, not proof that it universally beats cosine decay.

**Parameter count / scaling behavior:** Nominal **7B and 67B**; both are trained on **2T tokens**. The 7B specification has **30 layers, width 4,096, 32 query/KV heads**. The 67B has **95 layers, width 8,192, 64 query and eight KV heads**; both report 4,096-token context.

**Training paradigm:** Self-supervised base pretraining; separate supervised/preference-adapted Chat. Continue to [MoE model families](07-moe-models.md) for DeepSeekMoE/V2/V3 and the explicit discussion of reasoning post-training, rather than duplicating those entries here.

**Hardware/parallelism considerations:** Dense 67B training needs distributed state and activation management; GQA helps inference cache efficiency. A multi-GPU or quantized inference implementation is not evidence of the original cluster size or training precision.

### 3.12.6 BLOOM

**Name:** BLOOM, BigScience Large Open-science Open-access Multilingual Language Model; reference checkpoint **BLOOM-176B, 2022**, distinct from instruction-tuned BLOOMZ.

**Category & sub-category:** Self-supervised multilingual causal modeling; a dense, collaboratively developed research base model.

**Originating paper/vendor/year:** The BigScience collaboration, including Hugging Face and many academic/industry contributors, [2022 report](https://arxiv.org/html/2211.05100v4) and [official model card](https://huggingface.co/bigscience/bloom). Hugging Face is an organizer and implementation provider, not the sole inventor of the collective project.

**Core mechanism:** Train a decoder-only Transformer to continue text over the ROOTS multilingual corpus. ALiBI supplies attention-position biases without a learned absolute position embedding. An additional embedding LayerNorm helps stabilize large-scale training.

**Inputs/outputs and typical data types:** Text in the documented natural languages and programming languages -> next-token distributions and continuations. Task instructions are text prompts; BLOOMZ adds explicit multilingual multitask instruction training.

**Strengths and limitations:** Public weights, corpus documentation, and a multilingual research process support scrutiny and adaptation. Training-language inclusion does not guarantee strong performance in every language or task. The very large dense checkpoint is expensive to serve, and its base outputs can be biased or unsafe.

**Computational complexity / scalability notes:** Dense attention and FFNs follow the shared cost model. ALiBI changes attention scores, not the quadratic number of attention interactions. A short original training sequence does not become a validated long-context model simply because the positional formula extrapolates.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** BigScience evaluates **HumanEval** Python code completion. A held-out function specification -> BLOOM-176B -> candidate function body -> execution against unit tests -> pass@k estimate. This probes whether a broad multilingual model can also support programming; specialized code training is a reasonable alternative when code correctness is the priority. [Table 9 of the report's v4 edition](https://arxiv.org/html/2211.05100v4) gives **15.52% pass@1** for BLOOM and **12.06% for BLOOMZ**. The report explains that BLOOMZ's instruction mixture is not chiefly pure code completion, so general instruction tuning need not improve this task. Non-BLOOM baselines in that table come from prior work, not a common re-run. Sampling details and paper/card revisions must be retained when reproducing the result; no production coding KPI is established.

**Notable vendor implementations/libraries:** BigScience's Megatron-DeepSpeed fork, Hugging Face Transformers, and model-serving integrations. BLOOM's RAIL model license is not equivalent to an unrestricted software-code license.

**Architecture diagram description:** `token embeddings + embedding LayerNorm -> 70 causal Transformer blocks with ALiBI attention biases and GELU FFNs -> vocabulary distribution`.

**Activation functions used and why:** GELU in the FFN and softmax in attention/output. Fused bias/GELU kernels improve implementation efficiency without changing the high-level activation's purpose.

**Loss function(s):** Mean-reduced token cross-entropy for autoregressive pretraining. BLOOMZ's later multitask targets define a separate supervised instruction stage.

**Optimization algorithm(s):** The 176B training table specifies Adam, $`(0.9,0.95)`$, peak learning rate **$`6\times10^{-5}`$**, **375M warmup tokens**, and cosine decay scheduled over **410B tokens**, with a $`6\times10^{-6}`$ floor. The checkpoint reports **366B tokens seen**; it is incorrect to say it necessarily completed the entire decay horizon.

**Regularization techniques:** Weight decay 0.1, gradient clipping 1.0, LayerNorm, and carefully designed data processing. Positional bias and embedding normalization are architectural/numerical choices rather than proof against memorization.

**Backpropagation considerations:** Distributed optimizer state, activation checkpointing, and fused operations manage a large dense graph. The paper describes engineering work on numerical stability and communication; finite-precision training must be tested at the actual scale.

**Parameter count / scaling behavior:** The model card specifies **176,247,271,424 parameters**, **70 layers**, **112 attention heads**, width **14,336**, and **2,048 training sequence length**. Its data spans **46 natural and 13 programming languages**. These counts do not describe smaller BLOOM checkpoints.

**Training paradigm:** Self-supervised language modeling; BLOOMZ uses the separate xP3 multitask instruction stage. A multilingual prompt demonstration at inference does not retrain the base model.

**Hardware/parallelism considerations:** The reference run uses **384 A100 80-GB GPUs on the Jean Zay supercomputer**, with Megatron-DeepSpeed parallelism. Weight-only BF16 storage is about 352.5 GB by arithmetic, before caches and execution buffers; training state is much larger.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| GPT family | Text/code; image input in specified GPT-4 systems | General generative and in-context task interface | Modern internals undisclosed; endpoints are not historical checkpoints | GPT-3 LAMBADA test benchmark |
| LLaMA / Llama | Text/code with local adaptation needs | Public dense checkpoints and documented research generations | Recipes, licenses, and modalities change across releases | LLaMA 1 five-shot MMLU benchmark |
| Mistral dense | Text/code under serving-memory constraints | GQA and local attention in a small dense model | Original v0.1 training recipe only partly disclosed | Mistral-7B-v0.1 MMLU benchmark |
| Qwen dense | Multilingual and mathematical/code text | Strong checkpoint-specific multilingual/technical capabilities | Base, Instruct, VL, and MoE variants must be separated | Qwen2.5-7B GSM8K benchmark |
| DeepSeek LLM dense base | Chinese/English prose and code | Explicit dense scaling and stepwise schedule | Not the architecture or RL recipe of V2/V3/R1 | DeepSeek LLM 67B Base HumanEval benchmark |
| BLOOM | Multilingual text and research reproducibility | Collective documentation and broad language coverage | Very large dense serving cost; uneven task/language quality | BLOOM-176B HumanEval benchmark |

## 3.13 Vendor Multimodal and Assistant Families

This category groups vendor-facing families, not a single supervision regime or shared neural architecture. Command R is text-only in the release examined; the original ERNIE reference is an encoder; Titan includes embedding and generation products. Multimodal paired data supplies cross-modal supervision. The evidence distinguishes observable interfaces, published training stages, public reference implementations, and genuinely undisclosed internals.

### 3.13.1 Anthropic Claude

**Name:** Claude, represented by the **Claude 3 family announced in March 2024**, with Claude 3 Opus used for the worked benchmark. Haiku, Sonnet, and Opus are separate model offerings, not disclosed parameter-count bins.

**Category & sub-category:** Proprietary pretrained assistant family with multimodal input and post-training alignment. Its placement alongside self-supervised foundations does not classify all Claude training as unsupervised.

**Originating paper/vendor/year:** Anthropic's [Claude 3 release and linked model card, 2024](https://www.anthropic.com/news/claude-3-family). Anthropic's [Constitutional AI research, 2022](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) explains a related alignment method, not a complete recipe for every Claude release.

**Core mechanism:** The public interface conditions generated text on a conversation and, for Claude 3, image inputs. Anthropic describes work on Constitutional AI and safety tuning. It does not disclose a complete Claude 3 layer graph, dataset manifest, or optimizer recipe; observed fluency is not evidence for a specific decoder-only or MoE design.

**Inputs/outputs and typical data types:** Text conversations and images such as photographs, charts, and technical diagrams -> generated text. Product handling of PDFs, retrieval, or tools can add preprocessing and orchestration beyond a single model call.

**Strengths and limitations:** Long-context document handling and image interpretation support analysis workflows. However, proprietary internals limit independent reproducibility, citations or confident prose can still be wrong, and benchmark needle retrieval is much narrower than reasoning reliably over an entire document collection.

**Computational complexity / scalability notes:** Internal layer count, width, attention pattern, routing, and compute are **not publicly disclosed** for the reference. Assess an endpoint through measured prompt length, output length, latency, request limits, and cost under a specified workload. Do not assign the shared dense-Transformer FLOP formula as Claude's known implementation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Anthropic's Claude 3 release evaluates **Needle In A Haystack (NIAH)** long-document retrieval. A crowdsourced document corpus receives an inserted fact and a question chosen from **30 needle/question pairs**; Opus processes the context, answers the question, and the experiment checks retrieval accuracy. Long-context processing is technically useful when the answer must be recovered from supplied documents rather than guessed from pretraining; retrieval-plus-a-shorter-model is an alternative whose indexing adds another failure point. The [release reports Opus exceeding 99% accuracy](https://www.anthropic.com/news/claude-3-family) in its NIAH evaluation. This is a vendor-reported synthetic-insertion benchmark, **not** 99% accuracy on arbitrary documents, an independently repeated evaluation, or a production search KPI. The announcement does not provide enough detail to reconstruct every scoring condition.

**Notable vendor implementations/libraries:** Anthropic's API and SDKs, Claude's hosted application, and specified partner-hosted access. These transport/product layers do not expose Claude's weights or establish a local architectural implementation.

**Architecture diagram description:** Functional, not a claimed layer graph: `conversation + supported images -> proprietary Claude 3 inference -> response text -> optional application-side validation/tools`. Modality encoders, fusion design, attention topology, and expert routing are **not publicly disclosed** here.

**Activation functions used and why:** **Not publicly disclosed.** Neither GELU nor SwiGLU can be justified from a different vendor's model. Response probability controls exposed by an API do not reveal internal activations.

**Loss function(s):** The complete Claude 3 loss mixture is **not publicly disclosed**. Constitutional AI research separately demonstrates supervised learning from self-critiques/revisions and preference-model-based RLAIF. Those published experimental stages do not specify Claude 3's full pretraining or alignment losses.

**Optimization algorithm(s):** The underlying gradient optimizer, learning rates, warmup, decay schedule, and per-stage coefficients are **not publicly disclosed** for Claude 3.

**Regularization techniques:** Internal dropout, weight decay, normalization, and numerical regularization are **not publicly disclosed**. Safety evaluation and content policies are documented system practices, not substitutes for these missing neural details.

**Backpropagation considerations:** Hosted inference exposes no training gradients. Internal precision, gradient clipping, recomputation, and parallel gradient aggregation are **not publicly disclosed**. A student's training on permitted model outputs would not grant access to Claude's computation graph.

**Parameter count / scaling behavior:** Total and active parameter counts are **not publicly disclosed**. The launch's **200K context offering** is an interface limit, not a model-size estimate. Capability tiers do not reveal how parameters or compute differ.

**Training paradigm:** Proprietary foundation pretraining plus post-training alignment; Anthropic publishes relevant human/AI-feedback research. The exact balance of self-supervision, SFT, RLHF/RLAIF, or distillation for this release is not completely disclosed.

**Hardware/parallelism considerations:** Training devices, cluster count, sharding, and serving parallelism are **not publicly disclosed** in the cited release materials. Cloud availability and corporate hardware partnerships are not evidence of a particular Claude training configuration.

### 3.13.2 Google / DeepMind Gemini

**Name:** Gemini, represented by the **Gemini 1.5 Pro and 1.5 Flash models in the 2024 technical report**, with Gemini 1.0 identified as the preceding generation.

**Category & sub-category:** Proprietary multimodal pretrained model family with instruction/preference adaptation. Gemini 1.5 Pro's disclosed MoE architecture is cross-referenced, not expanded into a duplicate MoE entry.

**Originating paper/vendor/year:** Google/Google DeepMind, [Gemini 1.5 technical report, 2024, v5](https://arxiv.org/html/2403.05530v5), following Gemini 1.0 in 2023. Gemini product applications, Gemini API, and Vertex AI are distinct from the trained checkpoints.

**Core mechanism:** Joint handling of text, images, audio, and video permits mixed-modality context. The report explicitly identifies **Gemini 1.5 Pro as a sparse MoE Transformer-based model**. It describes **1.5 Flash as a Transformer decoder**, with parallel attention/feed-forward computation and online distillation from Pro. Those disclosures are narrower than a reproducible complete architecture.

**Inputs/outputs and typical data types:** Interleaved text/code, images, audio, and video -> generated text and task responses in the examined evaluations. Supported API modalities and context quotas must be tied to a particular release rather than inferred from the brand.

**Strengths and limitations:** Very long contexts can incorporate complete reference materials without training a task-specific model. Multimodal input avoids forcing every signal through an external text transcription first. Nevertheless, long context does not guarantee attention to all relevant evidence, and proprietary training details constrain reproducibility.

**Computational complexity / scalability notes:** MoE can decouple total stored parameters from the subset executing per token, but Gemini 1.5 Pro's total/active counts, routing fan-out, and topology are **not publicly disclosed**. Long-context architecture changes are reported without a full reproducible cost model. Flash latency figures cannot reveal Pro's parameter count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Google's **Machine Translation from One Book (MTOB)** experiment studies learning to translate English into Kalamang from reference materials. Approximately a **500-page grammar**, a bilingual wordlist, and parallel example sentences enter the context; a new sentence enters as the task; Gemini emits a translation for evaluation. This is a technical alternative to fine-tuning a translation model when parallel training data are scarce, not a demonstrated translation-service selection decision. In the report's **full-book English-to-Kalamang evaluation**, [Table 5](https://arxiv.org/html/2403.05530v5) gives Gemini 1.5 Pro a **5.46/6 human rating** and **59.0 chrF**. The human rater is a non-native, non-fluent learner who can recognize their own comparison translations; these scores are not native-speaker certification. Context-free performance is poor, and the reverse direction has a larger human-model gap. No community-service or business KPI is reported.

**Notable vendor implementations/libraries:** Google's Gemini API and SDKs and Vertex AI serving. Public client libraries expose requests, not the proprietary training stack.

**Architecture diagram description:** Published-level view: `mixed-modal inputs -> Gemini 1.5 Pro sparse-MoE Transformer -> text`; separately, `inputs -> 1.5 Flash Transformer decoder with parallel attention/FFN -> text`. Modality front ends, layer dimensions, routing details, and all internal connections are not fully disclosed. See [MoE mechanisms](08-moe-deep-dive.md).

**Activation functions used and why:** Per-component nonlinearities are **not publicly disclosed** in the cited report. Neither a standard dense-model SwiGLU block nor an assumed vision encoder is substituted.

**Loss function(s):** The report discusses language-model next-token prediction, multimodal instruction-response tuning, human-preference tuning, and Flash distillation. Exact complete objective mixtures, teacher/student loss weights, and auxiliary routing losses are **not publicly disclosed**.

**Optimization algorithm(s):** Flash is described as trained with **higher-order preconditioned methods**. That is a genuine disclosure, but not a license to name an exact optimizer or learning-rate schedule; those numerical details, and Pro's complete optimizer recipe, are **not publicly disclosed**.

**Regularization techniques:** Dataset curation and post-training safety work are described. Dropout, weight decay, normalization placement, and other checkpoint-specific numerical settings are **not publicly disclosed**.

**Backpropagation considerations:** Online distillation is explicitly reported for Flash, unlike mere speculation based on its size. The exact teacher-gradient treatment, distributed gradient layout, and activation checkpointing are not publicly disclosed. MTOB in-context adaptation itself involves no task-specific gradient updates.

**Parameter count / scaling behavior:** Pro total/active counts and Flash's count are **not publicly disclosed** in this 2024 report. Context-length demonstrations are not weight counts or guarantees that every API user had that context allowance.

**Training paradigm:** Multimodal pretraining; supervised multimodal instructions; further human-preference adaptation; explicitly documented Flash distillation. Paired multimodal data supplies supervision, so the family is not purely unsupervised.

**Hardware/parallelism considerations:** The report states training used **multiple 4,096-chip TPUv4 pods across datacenters**. It does not disclose a complete per-model chip count or sharding plan. This public infrastructure detail should be preserved without inventing the missing quantities.

### 3.13.3 Cohere Command R

**Name:** Command R, represented by **`c4ai-command-r-v01`, March 2024**, the 35B research-weight release; not Command R+ or a later hosted alias.

**Category & sub-category:** Autoregressive pretrained language model with supervised/preference post-training for grounded generation and tool use. This reference is **text-only**, despite its placement among vendor assistant families.

**Originating paper/vendor/year:** Cohere and Cohere For AI, now Cohere Labs, [Command R model card, 2024](https://huggingface.co/CohereLabs/c4ai-command-r-v01). The publicly released [Transformers v4.40.0 implementation](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/cohere/modeling_cohere.py) provides architectural details, not the original training dataset or optimizer.

**Core mechanism:** Generate responses conditioned on conversations, retrieved snippets, or tool descriptions. Grounded generation can first identify relevant/cited documents, produce an answer, and insert source spans. Tool selection emits proposed actions; the **application**, not the neural weights, executes an API call and returns its result.

**Inputs/outputs and typical data types:** Text conversations, document snippets with metadata, and tool schemas -> answer text, citation spans, or structured tool requests. A separate retriever supplies documents; Command R is not itself the vector index.

**Strengths and limitations:** Explicitly trained grounding and tool formats reduce the mismatch between generic chat and document-based assistance. Citations remain predictions that require checking against sources. Missing or adversarial retrieved content, prompt-template changes, and tool failures can all corrupt the final answer.

**Computational complexity / scalability notes:** The public causal Transformer follows dense decoder scaling; long prompts add attention and KV-cache costs. Retrieval can shorten the supplied evidence, but its indexing/search cost belongs to another component. A maximum context claim is not a fixed per-request memory footprint.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** Cohere's **official grounded-generation demonstration** asks which penguin is largest and supplies two snippets about emperor penguin height and habitat. Conversation plus snippets -> Command R's documented grounding template -> answer with separate citations to the relevant documents -> reader follows the source spans to check the claim. The [model card includes a rendered completion](https://huggingface.co/CohereLabs/c4ai-command-r-v01) identifying the emperor penguin and linking supporting statements to document IDs. Grounded generation is technically preferable to an unconditioned answer when source traceability matters, but this is a **vendor reference demonstration, not a customer deployment or independently executed test**. No factuality rate, citation-accuracy aggregate, or business KPI is reported.

**Notable vendor implementations/libraries:** Cohere's hosted API/SDK, Cohere Labs research weights, and Hugging Face Transformers. The research release's license and the commercial hosted service's terms are separate.

**Architecture diagram description:** Public reference implementation: `tokens -> repeated pre-LayerNorm -> {causal RoPE attention || gated MLP} -> sum both branches with residual -> final normalization -> token logits`. Parallel branches distinguish this block from a simple sequential Llama block.

**Activation functions used and why:** The [public reference configuration](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/cohere/configuration_cohere.py) selects SiLU, used in a gated MLP; attention uses softmax. These are implementation facts, not an invented vendor-wide recipe.

**Loss function(s):** The public causal-LM implementation supports shifted-token cross-entropy. The model card states SFT and preference training for assistant behavior, including grounding; it does not disclose a complete original preference loss or its coefficients.

**Optimization algorithm(s):** Original pretraining/post-training optimizers and numerical learning-rate schedules are **not publicly disclosed in the cited card**. A runtime's training API does not establish which optimizer Cohere used.

**Regularization techniques:** LayerNorm and residual structure are public. The reference configuration defaults attention dropout to zero; this is not a complete account of the original run's weight decay, data filtering, or other regularization.

**Backpropagation considerations:** Public weights permit local supervised adaptation subject to license. Grounding markers and response templates must be preserved in targets. Gradients do not flow through ordinary external document retrieval or tool execution unless an additional training method explicitly supplies that connection.

**Parameter count / scaling behavior:** The card specifies **35B parameters and a 128K context window**. It does not say that every Command family model has this size. BF16 weights alone require approximately 70 GB by arithmetic.

**Training paradigm:** Autoregressive pretraining followed by SFT and preference training. Exact RLHF/RLAIF algorithms, dataset proportions, and any distillation recipe are not publicly disclosed in this card.

**Hardware/parallelism considerations:** The card documents quantized loading options; tensor/model sharding and KV-cache planning matter for local serving. Original training hardware and full parallelism strategy are not publicly disclosed.

### 3.13.4 Baidu ERNIE

**Name:** Baidu's ERNIE family. The algorithmic reference is **ERNIE 1.0, 2019**; the public `ernie-1.0-base-zh` implementation and **ERNIE 4.0/ERNIE Bot, 2023**, are separately identified.

**Category & sub-category:** Knowledge-aware self-supervised encoder pretraining in the original formulation; a broader vendor generative/assistant family in later products.

**Originating paper/vendor/year:** Yu Sun and colleagues at Baidu, [ERNIE: Enhanced Representation through Knowledge Integration, 2019](https://arxiv.org/html/1904.09223v1). This is not the different same-acronym knowledge-graph paper from another research group. Baidu's [October 2023 ERNIE 4.0 announcement](https://en.prnasia.com/releases/global/baidu-launches-ernie-4-0-foundation-model-leading-a-new-wave-of-ai-native-applications-422575.shtml) is a vendor-issued product disclosure, not the 2019 encoder's technical specification.

**Core mechanism:** Instead of masking only independent Chinese characters or tokens, mask complete phrases and entities. This makes reconstruction depend on broader context rather than another visible piece of the same entity. The original model remains a bidirectional Transformer; it does not require injecting an explicit knowledge-graph vector at every layer.

**Inputs/outputs and typical data types:** Original Chinese text and sentence pairs -> contextual vectors, masked-token predictions, and task labels. ERNIE Bot's generated responses and multimodal product demonstrations belong to later systems; their interface does not reveal how many models or modality components are involved.

**Strengths and limitations:** Meaningful masking units can improve Chinese semantic representations compared with character-level shortcuts. Segmentation/entity identification can itself introduce errors and upstream supervision. The original encoder cannot be treated as a specification for a contemporary generative assistant.

**Computational complexity / scalability notes:** ERNIE 1.0 retains dense encoder attention/FFN scaling; phrase/entity preprocessing adds separate work. No supported FLOP, KV-cache, or routing estimate for ERNIE 4.0 follows from the original encoder's dimensions.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Baidu's 2019 **Chinese XNLI** evaluation classifies a premise/hypothesis pair as entailment, contradiction, or neutral. Pair -> supervised classifier on ERNIE's pretrained encoder -> class scores -> selected relation compared with the test label. Entity/phrase-aware masking is technically relevant because semantic relations depend on complete expressions, not just character overlap. The [paper's results table](https://arxiv.org/html/1904.09223v1) reports **78.4% test accuracy** for ERNIE versus **77.2% for its BERT baseline**; the **79.9%** ERNIE figure is development, not test. This is evidence about the original encoder study, not ERNIE 4.0, search-product conversion, or a measured business gain.

**Notable vendor implementations/libraries:** Baidu's PaddlePaddle/PaddleNLP ERNIE implementations; ERNIE Bot and Qianfan are product/platform layers. Platform access to multiple foundation models is not proof they share one ERNIE architecture.

**Architecture diagram description:** ERNIE 1.0: `token + position + segment embeddings -> bidirectional Transformer encoder -> MLM/task head`; phrase/entity selection supplies the masking pattern. ERNIE 4.0's full internal diagram is **not publicly disclosed in the cited announcement**.

**Activation functions used and why:** The named [PaddleNLP `ernie-1.0-base-zh` configuration](https://github.com/PaddlePaddle/PaddleNLP/blob/v2.8.1/paddlenlp/transformers/ernie/configuration.py) uses **ReLU**, not BERT's GELU. Other explicitly named ERNIE configurations in that file differ. This is a public reference-checkpoint detail, not a claim that all ERNIE assistants use ReLU.

**Loss function(s):** Masked-token reconstruction, with the original paper also exploring dialogue-related pretraining and supervised task losses. ERNIE 4.0's complete pretraining and alignment loss mixture is **not publicly disclosed** by its announcement.

**Optimization algorithm(s):** The cited 2019 paper does not enumerate a full optimizer/learning-rate schedule; the 2023 product announcement does not disclose ERNIE 4.0's either. Neither is silently filled with BERT's numerical recipe.

**Regularization techniques:** The named PaddleNLP reference has **0.1 hidden and attention dropout**, with Transformer normalization/residual structure. Knowledge-aware corruption changes the learning task. Modern product-specific dropout, weight decay, and normalization remain undisclosed here.

**Backpropagation considerations:** Gradients update the encoder and supervised head, not the discrete external selection of masking units. Weak upstream annotations should not be mistaken for a differentiable knowledge-retrieval module.

**Parameter count / scaling behavior:** The original paper specifies **12 layers, width 768, and 12 attention heads**. Its vocabulary differs from English BERT's, so copying the latter's parameter total would be wrong. A parameter count or total/active split for ERNIE 4.0 is **not publicly disclosed in the cited release**.

**Training paradigm:** Original self-supervised reconstruction over heterogeneous Chinese corpora, followed by supervised task adaptation. Later assistant stages require release-specific evidence; the ERNIE name alone does not identify SFT, RLHF, RLAIF, or distillation.

**Hardware/parallelism considerations:** Public encoder implementations can use GPU data parallelism. The original paper and the cited ERNIE 4.0 announcement do not provide a complete training-hardware or distributed-optimization specification.

### 3.13.5 Amazon Titan

**Name:** Amazon Titan, represented by **Titan Multimodal Embeddings G1** and contrasted with **Titan Text Embeddings V2**. Titan Text generation and Titan Image Generator are different models, not output modes of one disclosed network.

**Category & sub-category:** Proprietary foundation-model family; multimodal/text representation learning and separate generative products. Paired image-caption customization is supervised cross-modal learning, not pure unlabelled learning.

**Originating paper/vendor/year:** Amazon/AWS, [2023 multimodal release](https://aws.amazon.com/blogs/aws/amazon-titan-image-generator-multimodal-embeddings-and-text-models-are-now-available-in-amazon-bedrock/); [Multimodal Embeddings G1 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html) and [Text Embeddings V2 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html). The latter documents the separate 2024-generation text model.

**Core mechanism:** Map image and/or short text input into a shared semantic vector space for retrieval. The actual embedding architecture and objective are not disclosed sufficiently to call it CLIP, a specific two-tower encoder, or a decoder-only Transformer. Titan generation products have different input/output behavior.

**Inputs/outputs and typical data types:** Multimodal G1 accepts an image, English text, or both and returns a numerical vector; Text V2 maps text to a vector. Neither embedding endpoint itself returns a natural-language answer or performs database nearest-neighbor search.

**Strengths and limitations:** Shared embeddings enable text-to-image and image-to-image search without requiring exact keyword overlap. Similarity is not verified object identity or semantic truth, and short vectors can change retrieval quality. Proprietary internals and service revisions constrain reproducibility.

**Computational complexity / scalability notes:** Model inference cost is **not publicly disclosed**. Separately, brute-force search over $`N`$ indexed vectors of width $`m`$ costs $`O(Nm)`$ per query and stores $`O(Nm)`$ numbers; an approximate index trades recall, memory, and latency. This is database complexity, not Titan's neural complexity.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** AWS's **Titan Multimodal Embeddings workshop** implements visual product search over a subset of the **Amazon Berkeley Objects (ABO)** catalog. Product images and English metadata are indexed; the notebook's concrete text query **"A bed"** -> Titan vector -> OpenSearch Serverless nearest neighbors -> displayed product images/metadata for a user to inspect. A shared image/text space is technically useful where visual resemblance or phrasing differs from catalog keywords. The [workshop](https://github.com/aws-samples/titan-multimodal-embeddings-workshop/tree/main/3-content-search) and [query notebook](https://github.com/aws-samples/titan-multimodal-embeddings-workshop/blob/main/3-content-search/3b-content-search.ipynb) document the workflow and result-display code. This is a **reference implementation, not Amazon retail production evidence**. No Recall@k, sales uplift, or other public KPI is reported, and it was not rerun for this book.

**Notable vendor implementations/libraries:** Amazon Bedrock's `InvokeModel` interface, Boto3, and the AWS workshop's OpenSearch integration. Bedrock hosts multiple vendors' models and is not itself Titan's architecture.

**Architecture diagram description:** Functional view: `image and/or text -> proprietary Titan embedding model -> vector -> external index -> ranked objects`. The internal image/text encoders, fusion mechanism, and projection layers are **not publicly disclosed** here.

**Activation functions used and why:** **Not publicly disclosed.** An output embedding or similarity metric does not reveal whether the underlying network uses ReLU, GELU, SwiGLU, or another activation.

**Loss function(s):** The exact pretraining loss, contrastive-negative sampling, temperature, and auxiliary objectives are **not publicly disclosed**. Contrastive learning is a possible general approach to this task, not a verified Titan recipe.

**Optimization algorithm(s):** Original optimizer and pretraining learning-rate schedule are **not publicly disclosed**. The customization documentation exposes a default learning rate of **$`5\times10^{-5}`$** for Multimodal G1; that is a customer fine-tuning control, not disclosure of the original optimizer or schedule.

**Regularization techniques:** Dropout, weight decay, normalization, and model-specific regularizers are **not publicly disclosed**. A validation set in the customization workflow helps model selection but does not reveal the hidden regularization design.

**Backpropagation considerations:** Hosted embedding inference exposes no gradients. Documented image-caption fine-tuning supplies paired data to a managed training process; the external nearest-neighbor index is not automatically trained end-to-end with the embedding model.

**Parameter count / scaling behavior:** Parameter counts are **not publicly disclosed**. Multimodal G1 offers vector dimensions **1,024, 384, or 256**; Text V2 offers **1,024, 512, or 256**. These are **embedding widths, not parameter totals**. Text V2's documented maximum input is 8,192 tokens, not Multimodal G1's text limit.

**Training paradigm:** Proprietary foundation pretraining with an incompletely disclosed signal mixture; explicitly supported multimodal customization uses image-caption pairs. No unverified RLHF or distillation stage is asserted for these embedding checkpoints.

**Hardware/parallelism considerations:** AWS manages inference and training infrastructure. Original accelerator type/count and parallelism are not publicly disclosed in these materials. Batch indexing and request concurrency affect application throughput independently of the unknown neural implementation.

### 3.13.6 Amazon Nova

**Name:** Amazon Nova, represented by the **2024 understanding releases Micro, Lite, and Pro**. Canvas and Reel are separately named image/video generation models.

**Category & sub-category:** Proprietary text/multimodal foundation models with instruction and preference alignment; not a single architecture covering every Nova-branded service.

**Originating paper/vendor/year:** Amazon/AWS, [December 2024 launch](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-frontier-intelligence-and-industry-leading-price-performance/), and [The Amazon Nova Family of Models: Technical Report and Model Card, 2025](https://arxiv.org/html/2506.12103v1). The launch article notes its April 2025 benchmark update; the report is not evidence that these are the latest releases in 2026.

**Core mechanism:** The report identifies Micro/Lite/Pro as Transformer-based and describes multilingual/multimodal pretraining, instruction SFT, reward modeling, and preference optimization. It does not publish a complete layer graph, activation list, parameter count, or gradient-optimizer recipe.

**Inputs/outputs and typical data types:** Micro is text-only; Lite and Pro accept text, images, and video and generate text in the examined release. Canvas generates images and Reel generates video. A video-understanding input path is not a video-generation architecture.

**Strengths and limitations:** Document/image/video understanding and managed customization support application-specific workflows. Errors in reading tiny text, grounding responses, or interpreting temporal context remain possible. Service access and advertised price/performance do not provide reproducible model internals or a guarantee about a customer's costs.

**Computational complexity / scalability notes:** Exact model complexity is **not publicly disclosed**. Tokenized media, output length, concurrency, and endpoint limits influence observable work; parameter count cannot be inferred reliably from tokens per second. The dense reference formula is not an established Nova implementation model.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Amazon's **DocVQA** evaluation tests answering questions about document images. Document image plus question -> Nova Pro -> a short answer phrase -> comparison with the reference using **Average Normalized Levenshtein Similarity (ANLS)**. Direct multimodal processing is technically attractive compared with an OCR-then-text pipeline when layout and visual context matter, though it still needs document-level validation. [Table 3](https://arxiv.org/html/2506.12103v1) reports **93.5 ANLS for Nova Pro** and **92.4 for Nova Lite** on the **DocVQA test set**, under the report's **zero-shot** protocol. Appendix B.2.2 requests a single word or phrase. ANLS rewards near-matching strings; it is **not 93.5% perfectly correct business documents**. Other vendors' rows can use different OCR or prompting protocols. No document-processing ROI is established.

**Notable vendor implementations/libraries:** Amazon Bedrock, its Converse/InvokeModel interfaces, and AWS SDKs. The model identifier selects a hosted release; the API does not disclose its parameter tensors.

**Architecture diagram description:** Published-level functional view: `text (Micro), or text/images/video (Lite/Pro) -> proprietary Transformer-based understanding model -> text`. Full modality encoders, fusion, layer layouts, and any routing are **not publicly disclosed**.

**Activation functions used and why:** **Not publicly disclosed** for these understanding models in the cited report. Do not import Titan, Llama, or generic decoder activations.

**Loss function(s):** The report names instruction SFT, reward models trained from human preferences, DPO, and PPO. Exact pretraining/auxiliary multimodal losses and full stage weighting are **not publicly disclosed**. Named alignment methods are genuine disclosures, not a complete reproducible loss specification.

**Optimization algorithm(s):** The underlying gradient optimizer and numerical learning-rate schedules are **not publicly disclosed**. PPO and DPO name alignment procedures; they do not identify an Adam/SGD implementation or learning-rate policy.

**Regularization techniques:** The report documents data curation and safety-oriented SFT/RLHF work. Neural dropout, normalization, weight-decay coefficients, and other internal settings are **not publicly disclosed**. Runtime moderation is a separate system control.

**Backpropagation considerations:** Hosted inference exposes no gradients; original precision, clipping, and recomputation are not publicly disclosed. Bedrock customization and distillation are managed training capabilities, not proof that every released base model was distilled.

**Parameter count / scaling behavior:** Total/active parameter counts are **not publicly disclosed**. Launch context limits are **128K for Micro** and **300K for Lite/Pro**; context length is not parameter count. These interface facts do not apply automatically to later Nova releases.

**Training paradigm:** Multilingual/multimodal pretraining, supervised instruction demonstrations, and human-preference alignment are reported. The launch describes Pro as a teacher for custom Micro/Lite distillation; this is distinct from a disclosed original pretraining recipe.

**Hardware/parallelism considerations:** Original hardware inventory and sharding are **not publicly disclosed** in the cited report. Amazon's ownership of accelerator technology does not establish which hardware trained this checkpoint. Measure serving behavior on a versioned endpoint under the intended workload.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Anthropic Claude | Conversations, documents, and supported images | Long-context assistant interface and published safety research | Detailed internals undisclosed; needle recall is a narrow test | Claude 3 Opus NIAH benchmark |
| Google / DeepMind Gemini | Interleaved text, image, audio, and video | Long-context multimodal processing; published Pro/Flash distinctions | MoE size/routing and much of the recipe undisclosed | Gemini 1.5 Pro MTOB translation benchmark |
| Cohere Command R | Text snippets, conversations, and tool schemas | Explicit grounding/citation and tool-request training | Retrieval quality and citation correctness still need validation | Cohere's sourced penguin-QA grounding demonstration |
| Baidu ERNIE | Chinese semantic tasks; later versioned assistant interfaces | Original phrase/entity-aware masking | Original encoder facts do not specify later products | ERNIE 1.0 Chinese XNLI benchmark |
| Amazon Titan | Text/image retrieval representations | Shared-space multimodal search through a managed API | Embedding architecture and losses undisclosed | AWS ABO catalog-search reference implementation |
| Amazon Nova | Documents and image/video understanding | Published modality and alignment-stage distinctions | Counts, activations, and detailed graph undisclosed | Nova Pro/Lite DocVQA test benchmark |

## 3.14 Open Code, Efficient, and Enterprise Families

These entries concern public neural implementations and named weights, not a guarantee that every release is fully open in its data, training recipe, or licensing. "Efficient" must be scoped to inference, training, memory, or task quality; a compact serving model can still have required substantial pretraining compute. The concluding S4 and Mamba entries broaden the coverage from branded checkpoint families to non-attention sequence architectures, using explicitly self-supervised generative instantiations.

### 3.14.1 Microsoft Phi

**Name:** Phi family, represented by **Phi-3-mini-4k-instruct in the original April 2024 report** and the corresponding pre-June-update model-card revision. Phi-1/2 and later Phi variants have different recipes.

**Category & sub-category:** Data-curated causal foundation modeling with supervised/preference post-training; compact dense assistant model.

**Originating paper/vendor/year:** Microsoft's Phi research began with the 2023 "Textbooks Are All You Need" line; the detailed reference is the [April 2024 Phi-3 technical report, v1](https://arxiv.org/html/2404.14219v1) and [pinned original model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/ff07dc01615f8113924aed013115ab2abd32115b/README.md). The pinned revision prevents later training-token and score updates from overwriting historical provenance.

**Core mechanism:** Train a small causal Transformer on heavily filtered web material and synthetic, educationally structured data, then adapt it for useful responses. The key emphasis is allocating limited model capacity to useful knowledge and reasoning patterns, not introducing a universally new attention algorithm.

**Inputs/outputs and typical data types:** Text, code, mathematical prompts, and role-delimited conversations -> generated text. This Mini checkpoint is text-only; separately named Phi vision or MoE releases must not inherit its complete architecture description.

**Strengths and limitations:** Small weights make local/offline serving and constrained-memory applications feasible. Data curation can improve capability per parameter, but does not prove all factual knowledge or languages are covered. Reasoning benchmarks, safety behavior, and instruction following can differ across post-training revisions.

**Computational complexity / scalability notes:** Dense decoder scaling applies. A smaller $`p`$ lowers weight traffic, but sequential output generation and growing KV history remain. Quantization lowers storage/compute precision and can change accuracy; it is not a free preservation of the BF16 benchmark system.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Microsoft's original **GSM8K** experiment tests a compact model on grade-school mathematical word problems. Problem text -> Phi-3-mini with a chain-of-thought prompting protocol -> generated solution/final answer -> answer extraction and comparison with the test key. A compact generator is technically relevant when language understanding is needed locally; a symbolic calculator is preferable for guaranteed arithmetic once the problem has been parsed. The [v1 report](https://arxiv.org/html/2404.14219v1) gives **82.5% under zero-shot CoT**, versus **61.1% for Phi-2** in the same table. This is the adapted Phi-3-mini system, not an unaligned base checkpoint or an education deployment. The result does not establish a student-learning KPI.

**Notable vendor implementations/libraries:** Microsoft Phi releases, Hugging Face Transformers, and compatible ONNX/local runtimes. A hosted model catalog or inference wrapper is not a different Phi neural algorithm.

**Architecture diagram description:** The [pinned Mini configuration](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/ff07dc01615f8113924aed013115ab2abd32115b/config.json) specifies a dense causal Transformer: `tokens -> RMSNorm/RoPE attention and gated FFN residual blocks -> token head`, with **32 layers, width 3,072**, and **32 query/KV heads**.

**Activation functions used and why:** SiLU in the gated FFN supplies smooth nonlinear feature selection; attention/output use softmax. Phi-2's architecture and activation choices are not substituted for this Phi-3 reference.

**Loss function(s):** Autoregressive language-model prediction for pretraining, then SFT and DPO as documented in the card. Synthetic training examples do not establish that Mini minimized a teacher-logit KL loss.

**Optimization algorithm(s):** The original report/card does not disclose a complete pretraining optimizer and numerical learning-rate schedule. These are **not publicly disclosed in the cited release materials**; later fine-tuning sample scripts are not the original training recipe.

**Regularization techniques:** The pinned configuration sets attention, embedding, and residual dropout to **0.0** and uses RMSNorm. Data filtering and staged target selection are central; zero dropout does not mean absence of data-quality controls or imply other unpublished regularization coefficients.

**Backpropagation considerations:** Residuals and normalization support deep training; no original clipping or gradient-precision recipe is inferred where undisclosed. DPO updates policy parameters using preference pairs and should not be mislabeled as a documented online PPO loop.

**Parameter count / scaling behavior:** **3.8B parameters**, **4K context**, and **3.3T training tokens** in the original reference. The model card's later June update reports a different token budget; neither figure describes every Phi release.

**Training paradigm:** Curated/synthetic self-supervised pretraining, then supervised instructions and preference optimization. Synthetic-data use and distillation overlap in some workflows but are not synonymous.

**Hardware/parallelism considerations:** The pinned card reports **512 H100-80G GPUs** for training. Separately, the paper demonstrates a four-bit Mini running offline on an **iPhone 14 with A16 Bionic at over 12 tokens/second**. That is a hardware-specific prototype measurement, not a universal phone throughput or production reliability guarantee.

### 3.14.2 NVIDIA Nemotron

**Name:** Nemotron family, represented by **Nemotron-4-340B-Base, June 2024**. Its Instruct and Reward releases are separate adaptations; later models bearing "Nemotron" need their own provenance.

**Category & sub-category:** Dense causal base modeling and a supporting synthetic-data/alignment ecosystem.

**Originating paper/vendor/year:** NVIDIA, [Nemotron-4 340B Technical Report, 2024](https://arxiv.org/html/2406.11704v1), [Base model card](https://huggingface.co/nvidia/Nemotron-4-340B-Base), and the public NeMo implementation.

**Core mechanism:** A large dense causal Transformer provides a base for text generation and adaptation. The wider release also supplies an instruction model for generating responses and a reward model for judging/filtering them. A reward model is not merely a different temperature setting on Base, and synthetic-data generation is a workflow rather than a unique architecture.

**Inputs/outputs and typical data types:** Multilingual text/code -> continuation probabilities and generated text for Base. Instruct consumes conversation-style prompts; Reward scores candidate responses using its separate trained head and objectives.

**Strengths and limitations:** Public large-model weights and compatible training infrastructure support teacher-model and synthetic-data research. The 340B dense model is expensive; synthetic outputs can amplify teacher errors and require quality/diversity checks. An NVIDIA serving product does not reveal the architecture of all models it hosts.

**Computational complexity / scalability notes:** Dense per-token execution involves the full set of applicable layer weights, unlike an MoE active subset. GQA reduces KV state. The large embedding vocabulary and 96-layer stack add costs beyond a simplistic "quadratic attention" characterization.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** NVIDIA evaluates **MMLU** to measure the Base model's reusable multi-subject knowledge. Five demonstrations plus a test question/options -> Nemotron-4-340B-Base -> selected answer -> accuracy against the key. Such evaluation is relevant before selecting a teacher for synthetic knowledge-task data; that is a technical screening rationale, not proof that high MMLU guarantees high-quality synthetic examples. The [Base card](https://huggingface.co/nvidia/Nemotron-4-340B-Base) reports **81.1% five-shot MMLU**, and the report's table places it below its listed **84.2% Qwen-2 72B** comparison on this benchmark. Model size alone therefore does not determine task ranking. The benchmark is not a measured customer deployment or synthetic-data ROI.

**Notable vendor implementations/libraries:** NVIDIA NeMo, NeMo-Aligner, and compatible inference integrations. NIM is a serving packaging/interface layer; it is not the Nemotron-4 architecture itself.

**Architecture diagram description:** `tokens -> causal RoPE/GQA Transformer blocks with LayerNorm and squared-ReLU FFNs -> untied vocabulary head`. The [NeMo `Nemotron4Config340B`](https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/llm/gpt/model/nemotron.py) makes the layer dimensions and normalization explicit.

**Activation functions used and why:** **Squared ReLU**, $`f(x)=\max(0,x)^2`$, is the published FFN choice, not SwiGLU. It gives zero response to negative preactivations and a nonlinear positive branch. Large positive activations also have large derivatives, making numerical management important.

**Loss function(s):** Causal language-model loss for Base, retained during its continued pretraining phase. Instruct and Reward have additional alignment/rating objectives described separately in the report; Reward's scoring loss is not the Base model's next-token objective.

**Optimization algorithm(s):** The 340B report describes a distributed optimizer and a changed learning-rate decay schedule during continued training, but does not enumerate a complete named gradient-optimizer/numerical schedule. These details are **not fully publicly disclosed there**; the 15B report or NeMo defaults must not silently stand in for them.

**Regularization techniques:** The report and implementation specify **zero dropout**, untied input/output embeddings, and bias-free attention/FFN linear projections; the public implementation uses parameterized LayerNorm. Data reweighting in continued training is a separate intervention. Exact unpublished weight-decay/clipping settings are not asserted.

**Backpropagation considerations:** Large dense gradients and optimizer states require sharding; the report explicitly distributes optimizer state across data-parallel replicas. The 8T-to-1T data transition changes targets and sampling emphasis while retaining the base prediction loss.

**Parameter count / scaling behavior:** Nominal **340B**, with **96 layers**, width **18,432**, **96 query heads and eight KV heads**, and **4,096 context**. Training comprises **8T initial tokens plus 1T continued-training tokens**. These are not the dimensions of every Nemotron or Llama-Nemotron release.

**Training paradigm:** Primarily autoregressive self-supervision; the continued phase introduces some QA/alignment-style examples. Separate Instruct/Reward training supplies SFT and preference-related capabilities. The label Base does not prove that no instruction-like text appeared in its data.

**Hardware/parallelism considerations:** The report ramps data parallelism with **eight-way tensor and twelve-way pipeline parallelism**, reaching **6,144 H100 GPUs**. It separately targets eight-H100 inference in **FP8**; the card's BF16 serving examples require more memory. Training topology, weight precision, and inference footprint must not be conflated.

### 3.14.3 IBM Granite

**Name:** IBM Granite, represented by **Granite-3.0-8B-Base, released October 21, 2024**. Instruct, code-specialized, vision, and MoE variants are not all this same graph or recipe.

**Category & sub-category:** Dense causal foundation pretraining with an enterprise-oriented release/documentation approach; separate supervised/preference-adapted instruction models.

**Originating paper/vendor/year:** Granite Team at IBM, [Granite 3.0 Language Models technical report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) and [8B Base model card, 2024](https://huggingface.co/ibm-granite/granite-3.0-8b-base).

**Core mechanism:** A dense GQA/RoPE/SwiGLU Transformer is trained in two phases with curated data. Maximal update parameterization, written $`\mu`$P, controls scaling and hyperparameter transfer rather than changing next-token prediction into a different supervision class. The 3.0 family also contains MoE models, which are not described by this 8B dense configuration.

**Inputs/outputs and typical data types:** Multilingual prose, code, and serialized structured text -> generated continuations. Base does not inherit the instruction release's complete conversation behavior or safety alignment. Enterprise-facing software may add retrieval, access control, and task-specific models.

**Strengths and limitations:** Public weights, an Apache 2.0 release, and relatively detailed technical documentation help local adaptation and governance. These advantages do not prove universal benchmark leadership, complete training-data transparency, or fitness for an untested enterprise workflow.

**Computational complexity / scalability notes:** Dense Transformer costs apply; GQA and shared input/output embeddings reduce particular state/parameter costs. $`\mu`$P multipliers alter numerical scaling, not the number of sequence pairs in attention. Matching layer counts while omitting these multipliers is not necessarily a faithful reimplementation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** IBM's **HumanEval** evaluation asks models to synthesize Python functions from specifications. A held-out signature/docstring -> Granite-3.0-8B-Base -> candidate function -> unit-test execution -> pass@1 assessment. This is technically appropriate for assessing a code-capable base model before adaptation; a compiler alone checks syntax/types but does not synthesize an implementation from prose. [Table 8 of the report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) reports **52.44% pass@1**, compared with **31.71% for its Llama-3.1-8B Base baseline**. The model card independently repeats 52.44. The table does not provide every code-sampling setting, so it is not a complete reproduction recipe or a cross-paper ranking. No watsonx customer productivity KPI follows from this benchmark.

**Notable vendor implementations/libraries:** IBM's Granite repositories, Hugging Face Transformers, and IBM deployment/customization integrations. watsonx is a platform, not the architecture name of every model used within it.

**Architecture diagram description:** `shared token embeddings -> 40 dense RMSNorm/RoPE-GQA/SwiGLU residual blocks -> shared token head`, with release-specific embedding, attention, residual, and logit multipliers. The [published configuration](https://huggingface.co/ibm-granite/granite-3.0-8b-base/blob/main/config.json) is important for exact numerical behavior.

**Activation functions used and why:** SwiGLU, using SiLU in its gate, provides nonlinear feature selection; softmax handles attention and token prediction. The released config should take precedence over assuming an unmodified Llama block.

**Loss function(s):** Causal token cross-entropy in Base pretraining. The report separately describes curriculum SFT and preference/RL methods for Instruct; those additional alignment objectives must not be assigned to the Base result.

**Optimization algorithm(s):** The report specifies **AdamW**, $`(0.9,0.95)`$, weight decay 0.1, and a **Power scheduler** with **2,500 warmup iterations**, a slow power-law phase, and a final faster decay. Its scheduler settings are $`a=4`$, $`b=-0.51`$, and an upper bound 0.02 under the report's parameterization. These must be interpreted with $`\mu`$P's layer-wise scaling, not copied as a universal unscaled AdamW rate.

**Regularization techniques:** Weight decay 0.1, RMSNorm, curated data, and the released configuration's **0.1 attention dropout**. $`\mu`$P and explicit residual/logit scaling concern stable numerical behavior and hyperparameter transfer, not proof against memorization.

**Backpropagation considerations:** Faithful gradient updates require preserving the parameterization's multipliers and learning-rate scaling. The report uses distributed optimizer-state management and optimized attention/normalization kernels; changing precision or scaling requires validation.

**Parameter count / scaling behavior:** The card gives **8.1B parameters**, **40 layers**, width **4,096**, **32 query/eight KV heads**, FFN width **12,800**, and **4,096 context**. The model sees **10T tokens in stage 1 plus 2T in stage 2**. MoE total/active figures in the family are separate.

**Training paradigm:** Predominantly self-supervised base training; the second phase includes curated multilingual and instruction data. A separate Instruct release receives additional SFT/alignment. Base is explicitly not a fully safety-aligned assistant.

**Hardware/parallelism considerations:** IBM reports training on its **Blue Vela H100 infrastructure**, with tensor, pipeline, and data parallelism. Local serving can use quantization or sharding, but must budget for KV state beyond the approximately 16.2-GB BF16 weight-only calculation.

### 3.14.4 StarCoder

**Name:** StarCoder and StarCoderBase, the **2023 15.5B BigCode releases**. StarCoder2 is a later family and is not assigned this recipe.

**Category & sub-category:** Self-supervised code language modeling with fill-in-the-middle (FIM); Python-focused continued training for the named StarCoder checkpoint.

**Originating paper/vendor/year:** Raymond Li and the BigCode collaboration, organized by Hugging Face and ServiceNow with many contributors, [StarCoder: May the Source Be with You!, 2023](https://arxiv.org/html/2305.06161v1), and [official model card](https://huggingface.co/bigcode/starcoder).

**Core mechanism:** A causal Transformer learns from code while FIM transforms some sequences into prefix/suffix/middle order with special markers. It can therefore fill a missing code region using text both before and after the insertion point without replacing the entire model with a bidirectional encoder.

**Inputs/outputs and typical data types:** Source code, function signatures/docstrings, and FIM-delimited prefix/suffix text -> code completions or inserted code. Plain instruction prompts are not equivalent to training the model as an assistant.

**Strengths and limitations:** Code-specialized data and FIM fit editor completion, where the suffix already exists. Outputs can still be incorrect, insecure, or copied from training examples. Permissive-source filtering and opt-out processing do not eliminate downstream attribution or license obligations.

**Computational complexity / scalability notes:** Dense causal modeling still pays attention and FFN costs. Multi-query attention (MQA) shares a single KV head across query heads, substantially reducing cache state; it does not make the FFN sparse or guarantee linear full-attention arithmetic. FIM rearranges the training sequence rather than adding a separate inference pass.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** BigCode evaluates **HumanEval** Python function synthesis. A function signature/docstring -> StarCoder -> sampled body -> unit tests -> pass@1 estimate. Specialized code training is technically preferable to a broad multilingual base when the objective is executable completion rather than conversation. [Table 12 and section 6.1.1](https://arxiv.org/html/2305.06161v1) report **33.6% pass@1 for StarCoder**, versus **30.4% for StarCoderBase**. The open-model evaluation uses **200 samples per problem and temperature 0.2 for pass@1 estimation**. Its **40.8% StarCoder-Prompted** row adds a separate prompting intervention and must not be presented as the unprompted score. These are benchmark test results, not measured developer-time savings or production deployment evidence.

**Notable vendor implementations/libraries:** BigCode's training/evaluation repositories, Hugging Face Transformers' GPTBigCode implementation, and compatible code-completion runtimes. BigCode is a collaboration; neither organizing company alone accounts for the full authorship.

**Architecture diagram description:** `code/FIM tokens + learned positions -> causal GPT-2-style pre-normalized blocks with MQA and dense FFNs -> token head`. The suffix is placed before the target middle in the model's causal order.

**Activation functions used and why:** The public [GPTBigCode implementation configuration](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py) uses tanh-approximated GELU for smooth FFN nonlinearities; attention and vocabulary prediction use softmax. Later StarCoder architectures must be checked independently.

**Loss function(s):** Next-token cross-entropy on ordinary and FIM-transformed code. StarCoder's additional Python training keeps a language-model objective; "fine-tuned" here does **not** automatically mean supervised instruction tuning.

**Optimization algorithm(s):** StarCoderBase uses Adam, $`(0.9,0.95)`$, epsilon $`10^{-8}`$, and cosine decay from **$`3\times10^{-4}`$ to $`3\times10^{-5}`$** after **2,000 warmup iterations**. Python adaptation uses **$`5\times10^{-5}`$ to $`5\times10^{-6}`$** with **1,000 warmup iterations**, as specified separately in the paper.

**Regularization techniques:** Weight decay 0.1, LayerNorm/residuals, and stochastic FIM transformations. Dataset filtering and deduplication reduce particular risks but do not prove code originality. Generic GPTBigCode dropout defaults are not treated as a disclosed original run setting.

**Backpropagation considerations:** FIM markers and segment order must match training conventions; otherwise gradients optimize an unintended task. The paper reports BF16 training with FP32 gradient reduction, choosing stability despite a throughput cost.

**Parameter count / scaling behavior:** Both named 2023 models are approximately **15.5B**, with **40 layers**, width **6,144**, **48 query heads**, FFN width **24,576**, and **8,192 context**. StarCoderBase sees **1T tokens**; StarCoder adds **35B Python tokens**, not an entirely different-sized network.

**Training paradigm:** Self-supervised code pretraining and Python continued training. A technical-assistant prompt changes inference conditioning, not the weights. Distillation, RLHF, or instruction alignment are not established for these base-code releases.

**Hardware/parallelism considerations:** The paper reports **512 A100 80-GB GPUs**, with four-way tensor, four-way pipeline, and 32-way data parallelism. Local quantized deployment is a different setting and needs independent correctness, latency, and memory checks.

### 3.14.5 S4

**Name:** S4, Structured State-Space Sequence Model. The detailed neural instantiation is the **249M-parameter autoregressive WikiText-103 system in version 3 of the original paper**, not a Speech Commands classifier, S4D, or the later SaShiMi architecture.

**Category & sub-category:** Self-supervised generative sequence modeling; structured, linear-time-invariant state-space sequence operators inside a nonlinear language-model backbone. S4 is an architecture/parameterization family, not a hosted assistant product.

**Originating paper/vendor/year:** Albert Gu, Karan Goel, and Christopher Re at Stanford, [Efficiently Modeling Long Sequences with Structured State Spaces, 2021 preprint / ICLR 2022](https://arxiv.org/html/2111.00396v3). The authors' [state-spaces/s4 repository](https://github.com/state-spaces/s4) provides implementations, experiment configurations, and generation examples.

**Core mechanism:** Start from a continuous-time linear system, $`\dot{s}=As+bu`$, with readout $`y=cs+Du`$. Discretization produces $`s_t=\bar A s_{t-1}+\bar b u_t`$ and a causal convolution kernel $`K_j=c\bar A^j\bar b`$. S4 parameterizes the state matrix using normal-plus-low-rank structure related to HiPPO, permitting a well-conditioned diagonal-plus-low-rank representation and efficient Cauchy-kernel computations. It can train through convolution and generate through recurrent state updates. **The SSM operator is linear and time-invariant; the surrounding neural network is nonlinear. There is no query-key attention matrix.**

**Inputs/outputs and typical data types:** In the reference language model, discrete text tokens enter adaptive embeddings; the model returns next-token probabilities or generated continuations. Other S4 systems process real-valued waveforms, flattened images, or sensor sequences, but their heads and supervision differ.

**Strengths and limitations:** Structured initialization and recurrent state support long dependencies and streaming generation without retaining a growing KV cache. However, an LTI kernel cannot change its sequence-mixing coefficients according to the current token; fixed-state compression can lose details that full attention could access directly. Nonlinear output gates do not make the internal SSM input-selective in Mamba's sense.

**Computational complexity / scalability notes:** Let $`N`$ denote state order. After kernel construction, FFT convolution costs approximately $`O(BdT\log T)`$ per S4 layer, plus $`O(BTd^2)`$ channel/FFN mixing. Kernel construction is additional: the paper derives fast structured Cauchy methods, while naive kernel evaluation can cost $`O(dNT)`$. For fixed low rank, recurrent SSM updates need $`O(dN)`$ state and work per decoding token, plus channel mixing and vocabulary scoring. Constant cost in prior sequence length does not mean constant cost in width, state order, or model size.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The Stanford S4 study evaluates **WikiText-103** next-word language modeling on Wikipedia text. A held-out token prefix -> embedded inputs and recurrent/convolutional S4 backbone -> next-token distribution -> likelihood assigned to the actual next token -> aggregate test perplexity. This tests an attention-free alternative for generative text modeling; the technical attraction is recurrent-state serving, not a claim that S4 improves every quality metric. [Figure 8 and Appendix D.3.2](https://arxiv.org/html/2111.00396v3) report **20.95 test perplexity for S4, 249M parameters**, versus **20.51 for the 247M Transformer baseline**. Lower is better. The evaluation uses the stated sliding, non-overlapping-window protocol rather than additional-context scoring variants. The baseline is the cited adaptive-input Transformer setting, not every possible Transformer. No production latency or business KPI follows from these perplexities.

**Notable vendor implementations/libraries:** The authors' PyTorch S4 repository, with custom CUDA or PyKeOps support for structured kernels. These implementations are not an attention API, and the generic S4 module is not the trained WikiText checkpoint.

**Architecture diagram description:** `tokens -> adaptive embeddings -> [pre-LayerNorm/residual S4 -> pre-LayerNorm/residual S4 -> position-wise FFN] x 16 macroblocks -> tied adaptive softmax`. The reference uses **two S4 layers per macroblock**, not 16 total SSM layers; [the public block configuration](https://github.com/state-spaces/s4/blob/main/configs/model/layer/s4s4ff.yaml) makes the distinction explicit.

**Activation functions used and why:** The SSM recurrence itself is linear. The [S4 layer configuration](https://github.com/state-spaces/s4/blob/main/configs/model/layer/s4.yaml) uses GELU and a final GLU; the language-model FFN also uses GELU. These supply nonlinear feature transformation and gating around the sequence operator. Adaptive softmax normalizes vocabulary predictions, not attention weights.

**Loss function(s):** Autoregressive next-token negative log likelihood through the adaptive softmax; targets come from the text itself. Perplexity is the exponential of average token log-loss, not a classification accuracy.

**Optimization algorithm(s):** Appendix D.3.2 specifies **AdamW, learning rate $`5\times10^{-4}`$, and one cosine cycle with an 800,000-update maximum**. The [published WikiText reproduction configuration](https://github.com/state-spaces/s4/blob/main/configs/experiment/lm/s4-wt103.yaml) specifies **1,000 warmup updates**. This recipe is for that language experiment, not all S4 classifiers.

**Regularization techniques:** The paper's language run uses dropout **0.25**, pre-LayerNorm, and ordinary-weight decay **0.1**. The repository separately treats sensitive SSM dynamics with special optimizer groups, including zero weight decay; blindly applying one decay rule to every state-space parameter is unsafe.

**Backpropagation considerations:** Convolutional training backpropagates through kernel generation and FFT operations without a token-by-token recurrent training loop. Conditioning of the state representation, discretization, and complex arithmetic matters. The cited language run uses **no gradient clipping**; its recipe must not inherit Mamba's clipping value.

**Parameter count / scaling behavior:** The reference has **249M parameters**, width **1,024**, and **16 macroblocks**; the public S4 layer default has state order **64**. State order is not total parameter count. Embeddings, adaptive output layers, channel mixing, and the number of S4 layers all contribute to $`p`$.

**Training paradigm:** The primary instance here is **self-supervised causal language modeling**. The original paper's LRA and Speech Commands classifiers instead use labeled targets; see [supervised neural sequence learning](02-supervised-neural.md) for that supervision regime. Bidirectional or pooled classifiers are not this causal generator.

**Hardware/parallelism considerations:** The language experiment reports **eight A100 GPUs**, batch size **one per GPU**, and **8,192-token context**. FFT/Cauchy kernels favor parallel training, while recurrent execution supports stateful inference. Hardware-specific generation speedups require their exact batching and memory protocol; they are not universal S4 guarantees.

### 3.14.6 Mamba

**Name:** Mamba, the selective state-space architecture, represented by the **original `state-spaces/mamba-2.8b` Pile base checkpoint**. This is Mamba-1, not Mamba-2, Mamba-3, a SlimPajama retraining, or an attention-hybrid derivative.

**Category & sub-category:** Self-supervised autoregressive foundation modeling; input-selective recurrent state-space sequence modeling with hardware-aware parallel scans.

**Originating paper/vendor/year:** Albert Gu and Tri Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023 preprint, v2](https://arxiv.org/html/2312.00752v2), with the authors' [versioned v1.0.1 release documentation](https://github.com/state-spaces/mamba/blob/v1.0.1/README.md) and public checkpoint configuration.

**Core mechanism:** Unlike S4's fixed kernel, Mamba computes input-dependent $`B_t`$, $`C_t`$, and positive step size $`\Delta_t`$. Its conceptual recurrence is $`s_t=\bar A_t s_{t-1}+\bar B_t u_t`$, with $`y_t=C_t s_t+Du_t`$; a learned diagonal $`A`$ and the selected step size determine the discrete transition. Selection controls retention, input writing, and readout. Because these coefficients vary with the sequence, a single stationary FFT convolution no longer computes the operator. An efficient associative scan implements the recurrence. **This is not softmax attention, and the original homogeneous Mamba model contains no attention layers.**

**Inputs/outputs and typical data types:** Tokenized text -> vocabulary probabilities and generated continuations in the chosen checkpoint. The paper also trains separate audio and DNA systems. Their tokenization, state configuration, sequence length, and task heads must not be transferred to the 2.8B language model.

**Strengths and limitations:** Content-dependent state updates address an important limitation of LTI models while retaining a fixed-size recurrent inference state. However, compressed state is not a lossless archive of all preceding tokens. Million-position DNA experiments do not prove perfect million-token recall for this language checkpoint; hybrid models can have different attention and cache behavior.

**Computational complexity / scalability notes:** With fixed expansion factor, state order $`N`$, convolution width $`w`$, and width $`d`$, per-block sequence work is approximately $`O(BT(d^2+dN+dw))`$, excluding the vocabulary head. It is linear in $`T`$ under those assumptions, not independent of model width. Recurrent decoding keeps $`O(Bd(N+w))`$ state per block rather than a KV cache growing with $`T`$. Parallel scan and backward recomputation reduce hardware-memory traffic; a naive materialization of every intermediate state can still be expensive.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Gu and Dao evaluate the original Pile-trained models on **LAMBADA** final-word prediction. A narrative prefix -> Mamba's token-dependent state updates -> next-token/word probabilities -> predicted final word -> exact-match test decision. The scientific question is whether an attention-free generative backbone can retain the context needed for word completion; a full-attention decoder is the natural alternative. [The v2 paper's zero-shot Table 1](https://arxiv.org/html/2312.00752v2) reports **69.2% LAMBADA accuracy and 4.23 perplexity for Mamba-2.8B**, versus **64.7% and 5.04 for Pythia-2.8B**. Both use the same Pile training budget and tokenizer in the cited comparison; the protocol uses the EleutherAI evaluation harness. The separate **6.22** Mamba score in that table is **Pile validation perplexity**, not LAMBADA or a test score. No customer deployment or business KPI is claimed.

**Notable vendor implementations/libraries:** The authors' `mamba-ssm` package, CUDA/Triton selective-scan implementation, and Hugging Face-hosted weights. A Mamba block, a trained language-model checkpoint, and a serving wrapper are three different objects.

**Architecture diagram description:** `embedding -> repeated RMSNorm/residual blocks: linear expansion -> {causal depthwise convolution + SiLU -> selective SSM scan} multiplied by {SiLU gate} -> output projection -> final norm -> vocabulary head`. The [original block implementation](https://github.com/state-spaces/mamba/blob/v1.0.1/mamba_ssm/modules/mamba_simple.py) has no query-key-product or attention-softmax stage.

**Activation functions used and why:** SiLU/Swish transforms the convolutional branch and output gate. Softplus makes $`\Delta_t`$ positive; the reference parameterizes the real diagonal dynamics with $`A=-\exp(A_{\log})`$. These choices control nonlinear selection and decay. Vocabulary softmax remains valid even though **attention softmax is absent**.

**Loss function(s):** Causal next-token cross-entropy for the Pile base model. Selectivity is learned through this prediction loss, not through a separate reward or an attention-alignment target.

**Optimization algorithm(s):** The paper's improved recipe uses **AdamW with $`(0.9,0.95)`$**, linear warmup, and cosine decay to **$`10^{-5}`$**; peak learning rates follow the stated size-dependent, five-times-GPT-3 rule. The cited materials do not separately enumerate every 2.8B warmup override, so no universal numerical warmup duration is invented.

**Regularization techniques:** Reported weight decay **0.1**, **no dropout**, and RMSNorm in the chosen checkpoint. The implementation exempts $`A_{\log}`$ and the learned skip parameter $`D`$ from weight decay. Preserving the specialized step-size initialization is important; generic bias-zeroing can overwrite it.

**Backpropagation considerations:** Gradients flow through the input-dependent selectors and scan; the reported gradient-clipping value is **1.0**. Fused kernels recompute intermediate states during backward to avoid large memory transfers. The [checkpoint configuration](https://huggingface.co/state-spaces/mamba-2.8b/blob/main/config.json) preserves residuals in FP32. Recurrent state must be reset between independent sequences to avoid accidental context leakage.

**Parameter count / scaling behavior:** Nominally **2.8B parameters**, width **2,560**, and **64 actual Mamba blocks**. The historical README's 32 Transformer-equivalent layers explicitly require doubling to count Mamba blocks. The original module defaults are **state order 16, convolution width four, expansion two**; these are not Mamba-2 defaults or attention-head counts.

**Training paradigm:** **Self-supervised Pile pretraining for 300B tokens at sequence length 2,048**, without instruction or preference tuning for the reference release. The paper separately fine-tunes DNA models on labeled great-ape species classification; that is a supervised variant, cross-referenced to [supervised neural learning](02-supervised-neural.md), not a property of every Mamba checkpoint.

**Hardware/parallelism considerations:** Hardware-aware fused convolution/scan kernels and parallel prefix computation are central to efficient accelerator training. Recurrent decoding avoids a growing attention cache, but output projection and weight bandwidth still cost work. Pure Python scans, different batch sizes, and different devices cannot inherit the paper's GPU throughput claims.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Microsoft Phi | Compact local text/code reasoning | Capability-oriented data curation and small serving footprint | Small serving size does not imply cheap training or full knowledge coverage | Original Phi-3-mini GSM8K benchmark |
| NVIDIA Nemotron | Large-scale text/code and teacher-model workflows | Public Base/Instruct/Reward distinctions and infrastructure | 340B dense reference has very large compute/memory costs | Nemotron-4-340B-Base MMLU benchmark |
| IBM Granite | Locally adaptable enterprise text/code workloads | Public dense configuration, parameterization, and training report | Base is not a complete safety-aligned enterprise application | Granite-3.0-8B-Base HumanEval benchmark |
| StarCoder | Code completion and insertion with a known suffix | Code-specific pretraining and FIM | Generated code still needs tests, security review, and attribution checks | BigCode HumanEval benchmark |
| S4 | Long text, audio, and other ordered sequences | Structured LTI kernels with convolutional training and recurrent generation | Fixed kernel lacks token-dependent sequence mixing | S4 WikiText-103 test perplexity benchmark |
| Mamba | Causal text, audio, and DNA sequences in separately trained models | Input-selective state updates without attention or growing KV cache | Fixed-state compression and implementation-sensitive efficiency | Original Mamba-2.8B LAMBADA benchmark |

## Coverage and continuation manifest

- **Covered: 22 neural family/formulation entries.** Section **3.11.1-3.11.4** covers BERT, RoBERTa, T5, and BART. Section **3.12.1-3.12.6** covers GPT, LLaMA/Llama, Mistral dense, Qwen dense, DeepSeek LLM dense Base, and BLOOM. Section **3.13.1-3.13.6** covers Claude, Gemini, Command R, ERNIE, Titan, and Nova. Section **3.14.1-3.14.6** covers Phi, Nemotron, Granite, StarCoder, S4, and Mamba. S4's structured LTI operator and Mamba's selective recurrence are individually distinguished from attention.
- **Evidence boundary:** Twenty worked examples are research benchmarks; two are sourced vendor reference demonstrations, Command R grounding and the Titan ABO workshop. None is represented as a measured customer production deployment. Scores were checked against public reports/cards, not independently reproduced. Proprietary architecture, parameter, activation, optimizer, and dataset unknowns remain explicit. Model-card/configuration revisions are identified where they differ from historical papers.
- **Previous foundation:** Continue back to [unsupervised neural and self-supervised representation methods, sections 3.6-3.10](05-unsupervised-neural.md) for related neural mechanisms, and [unsupervised classical methods](04-unsupervised-classical.md) for clustering and other non-neural structure discovery. The original supervised Transformer and task-specific neural training belong in [supervised neural methods](02-supervised-neural.md); mixed-label regimes are discussed in [semi-supervised learning](03-semi-supervised.md).
- **Next and cross-cutting volumes:** [MoE model families, section 3.15](07-moe-models.md) distinguishes Mixtral, sparse DeepSeek and other expert-model releases from the dense references here. [The dedicated MoE deep dive](08-moe-deep-dive.md) explains routing, expert capacity, balancing, training, and systems trade-offs. [The comparative guide](09-comparative-guide.md), [glossary](10-glossary.md), and [reading guide](00-reading-guide.md) provide selection criteria, terminology, and shared evidence conventions.
- **Further depth not included in this edition:** Exhaustive release histories through September 2026; every regional, quantized, distilled, instruction, vision, speech, code, embedding, or reasoning derivative; a full catalog of S4D/S5/S4ND, SaShiMi, Mamba-2/-3, and attention-hybrid architectures; complete tokenizer specifications and corpus manifests; all licenses and changing endpoint quotas; full contrastive/multimodal objective derivations; reproducible end-to-end pretraining scripts; comprehensive PEFT, distillation, preference-optimization, or standalone reinforcement-learning tutorials; and independent production cost/latency or contamination audits. These are explicit scope limits, not claims that the named historical releases are current leaders.
