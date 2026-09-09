# 3. Unsupervised Learning Algorithms: Language and Multimodal Foundation Models

This volume continues Part 3 with sections 3.11-3.14. Foundation models first learn patterns that can support many later tasks. This chapter groups them by how they learn useful representations or generate content. It does **not** say that every model under these brands learns without supervision. The evidence policy is dated **2026-09-08**. All releases discussed here are historical examples, not claims about the latest available models.

**What an entry names.** An algorithm is a set of steps, such as hiding and predicting words. An architecture is the arrangement of a model's layers and connections. A layer transforms the numbers it receives. Weights, also called parameters, are the numbers adjusted during learning. A checkpoint is one saved set of those learned numbers. In model sizes, M means million, B means billion, and T means trillion.

A model is not the same thing as the application around it. An assistant application can add a chat screen, search, safety rules, tools, and several models. An API is an interface that lets software send requests and receive results. Its model name does not reveal the model's internal design. ChatGPT is therefore not another name for GPT-3's architecture. Claude's API is not its training algorithm. Amazon Bedrock is a hosting platform, not an Amazon Titan neural network.

**Reading and writing text.** A token is a small piece of text: perhaps a word, part of a word, or punctuation. A tokenizer splits text into these pieces and assigns each a numerical ID. An embedding turns an ID into a learned list of numbers, or vector. An encoder reads the available input and builds vectors that reflect its context. A decoder generates an output one token at a time. A causal decoder can use earlier tokens, but not future ones. An encoder-decoder combines both jobs: read the source, then write an output using it.

Most models here use Transformers. Attention compares token representations and combines information from the allowed positions. Multiple attention heads make separate comparisons; they are not separate assistants. A feed-forward network, also called an FFN or MLP, then transforms each position's vector. An activation function lets this transformation learn more than a straight-line relationship. Softmax turns scores into positive weights that sum to one. Those weights are **not** a guarantee that an answer is true.

Residual connections add a shortcut around a transformation. Normalization keeps internal number scales manageable. LayerNorm centers and rescales values; RMSNorm rescales using their root-mean-square size. Their exact placement matters. Neither technique proves that a model will work well on new data.

**Training signals.** Pretraining is the broad initial learning stage. In causal language modeling, the model reads earlier tokens and predicts the next one. The text itself supplies the correct next token. This is **self-supervision**, not a person labeling every prediction.

**Optional math:** For a token sequence $`x_{1:T}`$, the common causal-language-model loss is:

$$
\mathcal L_{\mathrm{CLM}}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

Here, $`T`$ is the number of tokens and $`t`$ is a position. The token at that position is $`x_t`$; $`x_{<t}`$ means all earlier tokens. The learned weights are $`\theta`$, and $`p_\theta`$ is the model's predicted probability. The loss $`\mathcal L_{\mathrm{CLM}}`$ adds a penalty for assigning low probability to each actual next token. Minimizing it encourages better predictions, not guaranteed factual truth.

Masked language modeling hides or changes selected tokens, then predicts the originals. Span denoising removes stretches of text. An encoder-decoder predicts those missing stretches or rebuilds the entire original text. None of these tasks establishes that the training text is true, unbiased, or legally reusable.

Fine-tuning means further training a checkpoint for a narrower purpose. **Supervised fine-tuning (SFT)** learns from prompts paired with desired answers. Its token cross-entropy, or log-loss, penalizes low probabilities for the desired words. The training setup must also decide whether prompt tokens contribute to that loss. During **teacher forcing**, training supplies the correct earlier output tokens. During generation, the model instead has to use its own earlier outputs.

**Distillation** trains a student model using a teacher model's probability distributions, generated answers, or selected reasoning examples. Synthetic text is text produced by a model. Using it does not, by itself, prove that a checkpoint matched the teacher's internal output scores, called logits.

**Preference feedback** compares answers rather than supplying only one ideal answer. RLHF means reinforcement learning from human feedback. It often trains a reward model to score answers, then adjusts the response-producing model, or policy. RLAIF uses AI feedback instead of, or alongside, human feedback. Constitutional AI is one published way to make critiques, revisions, and preference feedback. DPO learns directly from preferred/rejected answer pairs; it does not require an online policy-gradient loop. PPO is a different policy-training procedure. These names are not interchangeable labels for all safety or instruction training. Standalone reinforcement learning is outside the book's three main supervision categories. Its use after pretraining remains labeled, not renamed unsupervised learning.

**Shared computational assumptions.** Longer text and larger models need more computation and memory. In full attention, each position can compare with every allowed position. Doubling text length can roughly quadruple that comparison work, with other dimensions fixed. More layers also mean more repeated work. An encoder-decoder must additionally compare output positions with source positions.

**Optional math:** Let $`B`$ be examples per batch, $`T`$ tokens per sequence, $`d`$ vector width, $`L`$ layers, $`h`$ query heads, $`p`$ parameters, and $`V`$ vocabulary size. If FFN width grows in proportion to $`d`$, a dense full-attention forward pass costs approximately $`O(BL(Td^2+T^2d))`$. Big-O describes how work grows, not an exact runtime. Scoring the vocabulary can add up to $`O(BTdV)`$ when that output layer runs. Sharing input/output weights saves parameters, but not this scoring work. Storing all attention weights takes $`O(BLhT^2)`$ memory. Memory-efficient kernels can avoid storing that whole table; exact full attention still does not become linear-time arithmetic.

Training also needs intermediate layer outputs, gradients, and optimizer records. A gradient tells an optimizer how a small weight change would affect loss. Backpropagation calculates these gradients backward through the model. The learning rate controls update size. Adam, AdamW, and Adafactor are optimizer choices used below. Warmup gradually raises the learning rate; decay later lowers it. Dropout temporarily omits some internal values during training. Weight decay discourages large weights. Gradient clipping caps gradient size to limit spikes. These tools address different problems.

**Optional math:** With $`T_s`$ source tokens and $`T_o`$ output tokens, encoder-decoder attention includes $`T_sT_o`$ source-output comparisons. A decoder can cache earlier attention keys and values, abbreviated **KV**, rather than recalculate them. Using the same layer count $`L`$, width $`d`$, and cached length $`T`$, its next-token block work is approximately $`O(L(d^2+Td))`$, plus vocabulary scoring. It still produces output tokens sequentially.

Grouped-query attention (**GQA**) lets several query heads share stored keys and values. It saves KV memory, not all FFN computation. Sliding-window attention limits which positions a layer can compare. Mixture-of-experts (**MoE**) routing instead selects which weight blocks run. Experts are small subnetworks, not independent chatbots. These are different mechanisms.

**Resource requirements.** GPUs and TPUs are accelerators for large amounts of numerical work. Training can split examples across devices, called data parallelism. It can split large calculations, called tensor parallelism, or split layers into pipeline stages. Sharding means dividing stored model or training state across devices. Recomputing saved intermediate results can reduce memory at the cost of extra work. A hardware kernel is a low-level calculation routine; fusing combines steps into one routine. FP16 and BF16 are different 16-bit number formats; FP32 uses 32 bits. Mixed precision uses different formats where suitable. Quantization stores numbers with fewer bits and can change accuracy. Training hardware counts below are not automatic minimums for running a model.

**Optional math:** BF16 stores each weight in two bytes, so weights alone take about $`2p`$ bytes for $`p`$ parameters. This excludes caches, working buffers, gradients, and optimizer records. The cost formulas here are reference calculations, **not estimates of undisclosed vendor architectures**.

All benchmark scores below come from the named authors' reports. Their sources and test procedures were checked, but the models were not rerun for this book. Development or validation sets help check training choices. Test sets are reserved for final evaluation. A held-out split does not itself prove that similar material never appeared in pretraining. Different prompts, sampling rules, dataset versions, or checks for test data in training can change scores. These numbers are not one universal leaderboard. A business KPI means a measured business outcome, such as time saved; benchmark scores alone do not establish one.

## 3.11 Encoder and Encoder-Decoder Pretraining

These models make learning targets from unlabelled text. Later tasks can still need labeled examples. For instance, training an answer finder may require questions with known answers.

### 3.11.1 BERT

**In plain English:** BERT reads a passage in both directions to work out what each word means there. It is useful for labeling text or finding an answer already present in a passage.

**Name:** BERT stands for Bidirectional Encoder Representations from Transformers. This entry uses the original English BERT-Base and BERT-Large checkpoints, not later models adapted to specialist fields.

**Category & sub-category:** Unsupervised/self-supervised representation learning. BERT pretrains a bidirectional encoder by predicting hidden tokens and relationships between sentence pairs.

**Originating paper/vendor/year:** Jacob Devlin and colleagues at Google described BERT in a [2018 preprint](https://arxiv.org/html/1810.04805v2), published at NAACL 2019. Google released code and pretrained weights. BERT names a model design and learning method, not a hosted assistant.

**Core mechanism:** WordPiece splits text into tokens. BERT adds learned numbers for each token, its position, and which sentence segment contains it. Bidirectional attention uses words on both sides. The original method selects 15% of tokens for prediction. Most selected tokens are hidden with a mask; others are replaced randomly or left unchanged. A separate next-sentence prediction (NSP) output checks whether one sentence really follows another or was sampled as an alternative.

**Inputs/outputs and typical data types:** Text or sentence pairs enter as token IDs and masks marking usable positions. BERT returns context-aware token vectors. Added output layers, called heads, can predict hidden tokens, entity labels, sentence classes, or answer start/end positions. BERT does not naturally write a response from left to right.

**Strengths and limitations:** BERT fits tasks where the whole input is available. An answer-span head selects text from the source instead of freely inventing an answer. However, the original checkpoints have short context limits and biases from their English data. Artificial masks also make pretraining differ from later use. Selecting a statement from a passage does not prove it is true.

**Computational complexity / scalability notes:** Longer passages increase full-attention comparisons quickly. Classification adds a small output head; answer extraction scores endpoints without generating tokens one by one. Long documents require windows or a changed architecture, not just a larger input buffer. **Optional math:** The shared cost is $`O(BL(Td^2+T^2d))`$, where $`B`$ is batch size, $`L`$ layers, $`T`$ tokens, and $`d`$ vector width.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Google's **SQuAD 1.1** test asks questions about Wikipedia passages. A passage and question enter fine-tuned BERT. It scores start/end positions, selects the highest-scoring valid span, and compares that span with reference answers. Reading both sides helps locate an answer whose meaning depends on surrounding words. This was the authors' research test, not a deployment.

[Table 2](https://arxiv.org/html/1810.04805v2#S4.T2) reports single-model BERT-Large at **84.1 exact match / 90.9 F1 on development**, versus BERT-Base's **80.8 / 88.5**. Exact match requires the reference answer's normalized wording. F1 balances matching answer words against missing or extra words; higher is better. The widely quoted **93.2 test F1** came from an **ensemble with additional TriviaQA training**: several models combined, with extra training data. It was not an ordinary single BERT checkpoint. No business KPI was reported.

**Notable vendor implementations/libraries:** Google's [original BERT repository](https://github.com/google-research/bert) and Hugging Face Transformers provide implementations. Library support does not show that every Google product uses these exact weights.

**Architecture diagram description:** `WordPiece + position + segment embeddings -> [bidirectional attention -> residual/LayerNorm -> GELU feed-forward -> residual/LayerNorm] x L -> MLM/NSP or downstream head`. Each repeated block reads context, transforms vectors, and adds shortcuts. Here, L is the number of blocks; MLM means masked language modeling.

**Activation functions used and why:** GELU smoothly changes how strongly values pass through the FFN. Softmax turns attention and classification scores into weights or probabilities. A two-class output can be interpreted like a sigmoid, which maps a score between zero and one. That interpretation does not replace GELU inside the encoder.

**Loss function(s):** Pretraining adds the average masked-token log-loss to the NSP classification loss. These penalize wrong token and sentence-pair predictions. SQuAD fine-tuning adds the log-losses for the correct start and end positions.

**Optimization algorithm(s):** The original uses Adam, 10,000 warmup steps, then linear learning-rate decay. **Optional math:** Its learning rate is $`10^{-4}`$, with $`\beta_1=0.9,\beta_2=0.999`$. The two beta settings control Adam's running averages of gradients and squared gradients. Task fine-tuning uses separate, much shorter schedules.

**Regularization techniques:** The original recipe uses weight decay 0.01, dropout 0.1, and randomly changed input tokens. LayerNorm keeps internal scales stable. Its presence alone does not prove better results on unseen examples.

**Backpropagation considerations:** Fine-tuning can adjust every encoder layer. With few labeled examples, learning rate and starting weights can strongly affect results. Padding tokens must not receive prediction loss. Residual shortcuts help gradients pass through the deep encoder.

**Parameter count / scaling behavior:** Original Base has **12 layers, width 768, about 110M parameters**. Large has **24 layers, width 1,024, about 340M**. Width is the length of an internal token vector. Other models with "BERT" in their names can have different sizes.

**Training paradigm:** BooksCorpus and English Wikipedia supply self-supervised pretraining text. Labeled task examples then support supervised adaptation. NSP labels are constructed automatically, not manually assigned sentence relationships.

**Hardware/parallelism considerations:** The paper reports pretraining on a TPU pod. Later fine-tuning can use accelerators, data parallelism, and mixed precision. Full-length attention usually needs much more intermediate-result memory than the small task head.

### 3.11.2 RoBERTa

**In plain English:** RoBERTa is a BERT-like reader trained with a stronger learning recipe. It is useful for text classification and finding answers in supplied passages.

**Name:** RoBERTa means a Robustly Optimized BERT Pretraining Approach. The reference here is the original 2019 large English encoder.

**Category & sub-category:** Unsupervised/self-supervised representation learning. It improves how a masked-language-model encoder is pretrained.

**Originating paper/vendor/year:** Yinhan Liu and colleagues at Facebook AI and the University of Washington published the [2019 technical report](https://arxiv.org/html/1907.11692v1). The main contribution is a controlled training-recipe study and released checkpoints, not a new attention algorithm.

**Core mechanism:** Keep BERT's two-direction reading, but remove NSP. Choose fresh masks during training, use more text and larger batches, and train longer on full-length sequences. Byte-level BPE splits text into commonly used pieces represented through bytes. The study shows why better data and training can look like progress from a new architecture.

**Inputs/outputs and typical data types:** Byte-BPE token IDs become context-aware vectors and hidden-token predictions. Supervised output heads can classify sentiment or sentence relationships, score multiple-choice answers, or locate answer spans.

**Strengths and limitations:** RoBERTa offers a strong, fairly simple encoder that can learn many labeled tasks. Its extra data and compute are part of the result. Removing NSP alone cannot receive all the credit. Like BERT, it is not a general conversation generator. Larger web datasets still bring bias and possible overlap with test data.

**Computational complexity / scalability notes:** Work grows with sequence length and model width much as it does for BERT. The larger vocabulary and training budget change actual costs. Choosing new masks gives more varied prediction targets, but does not reduce a training step's attention or FFN work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Facebook AI's **SQuAD 2.0** test includes questions whose answers are absent from the passage. A question and Wikipedia passage enter RoBERTa. It predicts answer endpoints and whether an answer exists, then returns a span or no-answer. Keeping the answer inside a passage helps source tracing; a free-writing decoder could instead invent an unsupported answer.

[Table 6](https://arxiv.org/html/1907.11692v1#S5.T6) reports **86.5 EM / 89.4 F1** for one model on development, versus **79.0 / 81.8** for the listed BERT-Large baseline. EM means exact match after answer normalization. F1 rewards matching answer words while accounting for missing and extra words. RoBERTa used the supplied SQuAD data, without extra task-training data. Its **86.8 / 89.8 test** result belongs to a separate row, not development. This was not a measured customer-support deployment.

**Notable vendor implementations/libraries:** Facebook/Meta fairseq and Hugging Face Transformers support RoBERTa. The [released configuration](https://huggingface.co/FacebookAI/roberta-large/blob/main/config.json) records the large model's design. Hugging Face's explanatory card documents an implementation; it is not another originating paper.

**Architecture diagram description:** `byte-BPE + learned positions -> bidirectional Transformer encoder blocks -> contextual vectors -> masked-token or supervised task head`. The blocks read both sides of each token. This recipe has no pretrained NSP head.

**Activation functions used and why:** The released large configuration uses GELU to transform FFN values smoothly. Softmax supplies attention weights and vocabulary probabilities. This keeps the basic BERT-style nonlinear transformation.

**Loss function(s):** Cross-entropy penalizes wrong predictions at selected masked positions. There is no NSP loss. SQuAD adaptation adds training for answer spans and answerability. That stage is supervised even though the main encoder was pretrained without task labels.

**Optimization algorithm(s):** Adam uses warmup and linear learning-rate decay. The report varies peak rate, warmup, and epsilon, a small numerical-stability setting. **Optional math:** $`\beta_2=0.98`$ was helpful for large batches; beta-two controls the running average of squared gradients. Different comparison experiments use different schedules, so one schedule cannot describe them all.

**Regularization techniques:** Training uses fresh masks, weight decay, and dropout. The large release sets hidden and attention dropout to 0.1. LayerNorm manages scales, while residual connections provide shortcuts.

**Backpropagation considerations:** Large batches and mixed precision need stable optimizer calculations and correct loss handling. Fine-tuning can update the entire encoder. Comparing two architectures is not a clean test of their designs if their training-token or optimization budgets also differ.

**Parameter count / scaling behavior:** The [official fairseq release](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md) lists about **355M parameters for Large** and **125M for Base**. Large has **24 layers and width 1,024**. Its larger vocabulary helps explain why its count differs from BERT-Large. Large's results do not automatically apply to Base.

**Training paradigm:** Pretraining predicts masked tokens using self-supervision; later task adaptation uses labels. The strongest reported setting uses **160 GB of text and 500,000 updates**. That is not the smaller comparison trained only on Books/Wikipedia.

**Hardware/parallelism considerations:** The report uses DGX-1 systems with V100 GPUs, mixed precision, and communication between machines. Splitting batches across devices increases throughput. Each model copy still needs storage unless its state is divided across devices.

### 3.11.3 T5

**In plain English:** T5 treats many tasks as "read some text, then write some text." The same basic model can write an answer, a translation, a summary, or a label.

**Name:** T5 means Text-to-Text Transfer Transformer. Original T5-Base supplies the design reference; original T5-11B shows what changes at a much larger scale.

**Category & sub-category:** Unsupervised/self-supervised pretraining with an encoder-decoder. It learns to restore missing text spans and uses a common text-input/text-output task format.

**Originating paper/vendor/year:** Colin Raffel and colleagues at Google released a 2019 preprint and [JMLR 2020 paper](https://jmlr.org/papers/v21/20-074.html). The [full report](https://arxiv.org/html/1910.10683v4) separates baseline experiments from its final, larger systems.

**Core mechanism:** A task is written as text, often with a prefix naming the task. During span corruption, missing stretches are replaced by distinct markers called sentinels. A bidirectional encoder reads this damaged input. A causal decoder writes the removed spans and their markers, **not** a full copy of the input. It uses cross-attention to read the encoder's vectors while writing.

**Inputs/outputs and typical data types:** SentencePiece tokens represent questions, passages, translation requests, or classification tasks. Outputs are text such as answers, translations, summaries, or label words. For example, `entailment` is a generated token sequence. It does not require a separate, essential classifier architecture.

**Strengths and limitations:** One interface supports varied tasks, and missing-span targets can be short. An answer can use a standard form that does not appear word-for-word in the source. But T5 can also generate unsupported answers or invalid label strings. Training with correct earlier output tokens differs from using its own possibly wrong outputs later; this is exposure bias. Task formatting and output-selection rules matter.

**Computational complexity / scalability notes:** The encoder compares input positions, the decoder compares earlier output positions, and cross-attention connects the two. Short missing-span targets save decoder work compared with rebuilding all source tokens. Long-input use still needs source encoding and decoder state.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Google's T5 study answers **SQuAD 1.1** questions by generating text. T5 reads a question and Wikipedia context in the task format. It writes an answer string, which the evaluator normalizes and compares with reference answers. This tests whether one generative interface can replace BERT's special answer-span head.

Final-results **Table 14** reports **T5-11B at 91.26 EM / 96.22 F1**, versus **T5-Base at 85.44 / 92.08**, on the **SQuAD validation set**. EM checks exact normalized wording; F1 measures answer-word overlap while accounting for omissions and additions. The [report makes SQuAD an explicit exception](https://arxiv.org/html/1910.10683v4) to final test-set evaluation: the benchmark server could not run the largest models. These systems received task adaptation; they were not raw denoising checkpoints. The final recipe includes supervised task mixtures and fine-tuning, so unlabelled C4 alone cannot receive the credit. No production question-answering KPI was established.

**Notable vendor implementations/libraries:** Google's [text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer), the original Mesh TensorFlow implementation, and Hugging Face Transformers support T5. FLAN-T5 is a later instruction-tuned derivative, not the original T5 recipe under another name.

**Architecture diagram description:** `corrupted text + sentinels -> bidirectional encoder -> cross-attended causal decoder -> removed spans + sentinels`. The decoder reads the encoder's results through cross-attention. For later tasks, task text replaces the damaged-text input.

**Activation functions used and why:** Original T5 uses ReLU in its FFNs: negative values become zero, while positive values pass through. Attention and output probabilities use softmax. Later T5 variants can use different or gated activations. Those later choices must not be assigned to the original.

**Loss function(s):** Teacher-forced token cross-entropy rewards the desired output string, given the input and correct earlier target tokens. Baseline targets contain missing spans. Supervised text-to-text tasks use different targets with the same kind of loss.

**Optimization algorithm(s):** T5 uses Adafactor. **Optional math:** Baseline pretraining sets the learning rate to $`1/\sqrt{\max(s,10^4)}`$, where $`s`$ counts updates. The denominator uses at least 10,000, then grows as the square root of the update count. This is inverse-square-root decay, not exponential decay. Separately specified fine-tuning uses a constant 0.001 rate.

**Regularization techniques:** The baseline uses dropout 0.1, missing-span corruption, and residual shortcuts. Relative position biases tell attention about token spacing. T5's simplified pre-normalization rescales without subtracting the mean. Cleaning the data remains a separate need; dropout does not replace it.

**Backpropagation considerations:** Teacher forcing allows losses for all target positions to be computed in parallel during training. Generation still happens token by token. Gradients pass through cross-attention into the encoder. Shared embeddings and masked targets must agree on tokenization.

**Parameter count / scaling behavior:** Original T5-Base has about **220M parameters**, **12 encoder and 12 decoder blocks**, and width 768. The named **11B** system is much larger. Neither its scores nor its hardware costs describe Base.

**Training paradigm:** The main method is self-supervised denoising followed by task adaptation. The paper also tests mixtures of supervised tasks and self-supervision. A text-to-text interface does not automatically imply later instruction tuning or distillation.

**Hardware/parallelism considerations:** The original study uses TPUs and splits large models across devices. Adafactor reduces the storage needed for optimizer records. It does not remove memory needs for intermediate results, attention, or the two model stacks.

### 3.11.4 BART

**In plain English:** BART learns by repairing damaged documents. After training on article-summary pairs, it can turn an article into a shorter piece of text.

**Name:** BART stands for Bidirectional and Auto-Regressive Transformers. The references are `bart.base`, `bart.large`, and the separately fine-tuned `bart.large.cnn`.

**Category & sub-category:** Unsupervised/self-supervised learning through denoising sequence-to-sequence pretraining. It reads a damaged sequence and learns to write the original.

**Originating paper/vendor/year:** Mike Lewis and colleagues at Facebook AI described BART in a [2019 preprint](https://arxiv.org/html/1910.13461v1), published at [ACL 2020](https://aclanthology.org/2020.acl-main.703/).

**Core mechanism:** Damage a document, then reconstruct all of its original text. The paper compares deleting tokens, masking tokens, filling gaps, shuffling sentences, and other changes. The large model combines gap filling with sentence shuffling. Unlike T5's removed-span targets, BART's decoder writes the **whole** original sequence.

**Inputs/outputs and typical data types:** Pretraining takes damaged token sequences and returns restored documents. Supervised adaptation can take articles, conversations, or other text and return summaries or responses. Extra output heads can also support classification.

**Strengths and limitations:** An encoder reads the whole source, while a causal decoder writes a flexible new version. The damage process need not keep the source length unchanged. However, writing token by token is slower than one-pass extraction and can invent facts. ROUGE compares wording with reference summaries; it does not check factual consistency.

**Computational complexity / scalability notes:** Both stacks and their cross-attention require work. Rebuilding a whole document can cost more than predicting only T5-style missing spans. Beam search keeps several possible outputs in play, increasing work and cache storage. **Optional math:** For source length $`T_s`$ and output length $`T_o`$, attention includes $`T_s^2`$, $`T_o^2`$, and $`T_sT_o`$ terms, plus dense FFN work. These count within-source, within-output, and source-output comparisons.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Facebook's **CNN/Daily Mail** experiment turns news articles into highlights. An article enters the supervised `bart.large.cnn` model. Its generated summary is compared with a held-out reference summary using ROUGE. Generation can combine information from across an article instead of simply copying its opening sentences. It is not guaranteed to keep every fact correct.

The [official release table](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md) specifies the **test set, no additional task data**. It reports **44.16 ROUGE-1, 21.28 ROUGE-2, 40.90 ROUGE-L**, versus **42.13 / 19.60 / 39.18** for BERTSUMEXTABS. ROUGE-1 compares single words, ROUGE-2 word pairs, and ROUGE-L shared word order through a longest common subsequence. Higher overlap is better on these measures, but is not a truth check. The experiment does not show that a newsroom saved editing time. A human editor or factuality check would be a deployment requirement, not a measured benchmark step.

**Notable vendor implementations/libraries:** Facebook/Meta fairseq and Hugging Face Transformers provide BART, including `facebook/bart-large-cnn`. The `.cnn` and `.xsum` suffixes name supervised adaptations, not different pretraining architectures.

**Architecture diagram description:** `noised document -> bidirectional encoder -> causal decoder with encoder cross-attention -> original document`. The encoder reads the damaged source; the decoder uses its vectors to rebuild it. Summary fine-tuning replaces the reconstruction target with a summary.

**Activation functions used and why:** BART's FFNs use smooth GELU transformations instead of the original Transformer's ReLU. Softmax turns attention scores and output scores into weights and probabilities.

**Loss function(s):** Pretraining penalizes low probability for the original document, given its damaged version. The official summary-training example uses cross-entropy with label smoothing 0.1. Smoothing avoids putting the entire target probability on just one token.

**Optimization algorithm(s):** The paper does not list every large-model pretraining optimizer setting. The [published CNN/Daily Mail fine-tuning recipe](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md) uses Adam, 500 warmup updates, and polynomial rate decay over 20,000 updates. **Optional math:** Its averaging settings are $`(0.9,0.999)`$, for gradients and squared gradients, and its learning rate is $`3\times10^{-5}`$. Polynomial decay follows a power-based curve. These are **fine-tuning** settings, not recovered pretraining settings.

**Regularization techniques:** Training uses damaged inputs, residual shortcuts, and LayerNorm. The large pretraining run turns dropout off for its final 10%. The cited summary recipe uses dropout 0.1 and weight decay 0.01.

**Backpropagation considerations:** Teacher forcing lets training handle target-position losses in parallel, with gradients passing through both stacks. The summary recipe clips gradient norm at 0.1. Padding, cutting off long sources, and label smoothing affect what the model learns.

**Parameter count / scaling behavior:** Official rounded sizes are about **140M** for Base and **400M** for Large. Their encoder/decoder layer counts are **6/6** and **12/12**, respectively. These release figures are rounded, not exact counts of stored tensor values.

**Training paradigm:** BART first learns to repair text through self-supervision, then learns supervised generation or classification tasks. A pretrained BART checkpoint alone is not the CNN/Daily Mail benchmark system.

**Hardware/parallelism considerations:** The documented fine-tuning example uses eight 32-GB V100 GPUs, FP16, and gradient accumulation. Accumulation combines gradients from smaller batches before updating weights. This example is not a universal hardware minimum or the original pretraining cluster specification.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| BERT | Text pairs and labels for individual tokens | Reads both sides; can select an answer from the source | Original context is short; does not naturally write responses | Google's SQuAD 1.1 reading test |
| RoBERTa | Text labels and questions answered from passages | Improves masked-token training without replacing the basic encoder | Improvements depend strongly on more data and training | Facebook AI's SQuAD 2.0 reading test |
| T5 | Tasks with text inputs and text outputs | Uses one task format and short missing-span targets | Written answers need checking; two stacks need memory | Google's SQuAD text-to-text reading test |
| BART | Documents, summaries, and text-rewriting pairs | Reads the whole source before writing a flexible output | Can invent facts; writes one token at a time | CNN/Daily Mail summary-writing test |

## 3.12 Public Decoder Research and Base-Model Families

"Public" means that particular research findings, code, or weights are available. It does **not** mean GPT-3's weights or GPT-4's internals are open. The dense models here use their applicable layer weights for each token. Similarly named MoE releases select among expert subnetworks and belong in [the MoE model volume](07-moe-models.md).

### 3.12.1 GPT Family

**In plain English:** GPT models learn to continue text, then use that ability to attempt many tasks. A prompt can show the task, but a fluent continuation can still be wrong.

**Name:** Generative Pre-trained Transformer family. This entry distinguishes GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), and what was publicly disclosed about GPT-4 (2023). GPT-3 175B supplies the detailed large-training example.

**Category & sub-category:** Self-supervised autoregressive language modeling: predicting text from earlier text. Some releases add supervised or preference-based training. GPT-3's sparse attention limits token comparisons; it is not MoE routing among weight blocks.

**Originating paper/vendor/year:** OpenAI's [GPT-1 implementation](https://github.com/openai/finetune-transformer-lm), [GPT-2 model card](https://github.com/openai/gpt-2/blob/master/model_card.md), Brown and colleagues' [2020 GPT-3 paper](https://arxiv.org/html/2005.14165v4), and the [2023 GPT-4 technical report](https://arxiv.org/html/2303.08774v6). Each source reveals a different amount of detail.

**Core mechanism:** GPT-1 first predicts next tokens, then learns labeled tasks through fine-tuning. GPT-2 studies task performance from prompts after training on WebText. GPT-3 scales this idea and tests learning from examples placed in the prompt, **without gradient updates**. The GPT-4 report describes a Transformer-based, multimodal model pretrained on next-token prediction and later trained with RLHF. Its detailed design is withheld. A contemporary ChatGPT session is not therefore a raw GPT-3 model.

**Inputs/outputs and typical data types:** GPT-1/2/3 take text tokens, including code and task examples. They output next-token probabilities and continuations. The GPT-4 report includes image-plus-text input and text output. That does not specify the supported inputs of every earlier or later API endpoint.

**Strengths and limitations:** A prompt-based generator can attempt many tasks without a separate output head for each. Examples in the prompt can help, but consume context space and do not permanently change the weights. Generated answers can be fabricated. Changed hosted versions and hidden application behavior make exact repetition harder.

**Computational complexity / scalability notes:** GPT-1/2 use causal Transformer computation. GPT-3 alternates full attention with sparse attention to nearby positions. Its exact attention cost depends on that pattern, while dense FFN work remains. Cached decoding still requires work for every output token. GPT-4's arithmetic operation count, cache layout, and memory needs are **not publicly disclosed**.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** OpenAI's **LAMBADA** test provides a passage and asks for its final word. GPT-3 scores possible continuations, predicts that word, and receives an exact-match test decision. It uses its existing text-prediction ability instead of a separately trained fill-the-blank classifier.

[Table 3.2](https://arxiv.org/html/2005.14165v4) reports **76.2% zero-shot test accuracy** and **86.4% few-shot**, versus the cited prior-best **68.0%**. Zero-shot supplies no task demonstrations; few-shot supplies several. The few-shot setup uses a different fill-the-blank format and development-set demonstrations. It does not train the weights. Prompt examples do not always improve scores: the one-shot score is lower here. The paper also discusses possible benchmark material in training data. No business productivity gain was established.

**Notable vendor implementations/libraries:** OpenAI released GPT-1 and GPT-2 code and weights; Hugging Face implements those historical designs. OpenAI's hosted API provides access, while ChatGPT is an application. Neither is a public implementation of GPT-4's architecture.

**Architecture diagram description:** GPT-1: `embeddings -> causal attention/MLP blocks with post-normalization -> LM or supervised head`. Post-normalization means normalization follows the block transformation. GPT-2/3 instead normalize before transformations within residual blocks; GPT-3 alternates attention patterns. GPT-4's functional view is `text/image input -> proprietary Transformer-based model -> text`. Its interior layer graph is not publicly disclosed.

**Activation functions used and why:** Published GPT-1/2/3 designs use GELU for smooth FFN transformations and softmax for probability distributions. GPT-4's activations are **not publicly disclosed**. The family name does not establish that it uses GELU.

**Loss function(s):** GPT-1/2/3 base training penalizes low probability for the actual next token. GPT-1 adaptation also uses labeled-task losses. GPT-4 discloses next-token pretraining and RLHF, but not a complete, repeatable loss recipe or the weights assigned to preference-stage losses.

**Optimization algorithm(s):** GPT-3 uses Adam, gradient-norm clipping 1.0, 375M warmup tokens, and cosine decay to 10% of peak over 260B tokens. Cosine decay follows a smooth cosine-shaped curve. **Optional math:** Adam's gradient and squared-gradient averaging settings are $`(0.9,0.95)`$. The 175B model's peak learning rate is $`6\times10^{-5}`$. GPT-1 repository fine-tuning defaults are a different recipe. GPT-4's optimizer and schedule are **not publicly disclosed**.

**Regularization techniques:** GPT-1's released code includes dropout and weight decay. GPT-3 specifies weight decay 0.1, data filtering, removal of duplicate text, pre-normalization, and careful starting weights. GPT-4's dropout, normalization placement, and regularization coefficients are **not publicly disclosed**.

**Backpropagation considerations:** Large historical models need gradients and intermediate training results managed across devices. GPT-3's prompt-example tests run predictions only, not backpropagation. GPT-4's API does not reveal gradient precision or which intermediate results are saved or recomputed.

**Parameter count / scaling behavior:** GPT-1 has 12 layers and width 768; its [published tensor shapes](https://github.com/openai/finetune-transformer-lm/blob/master/model/params_shapes.json) add up to about 117M parameters. GPT-2's final card lists **124M, 355M, 774M, and 1.5B** releases. GPT-3 175B has **96 layers, width 12,288**, and saw **300B tokens** during training. GPT-4's total and active parameter counts are **not publicly disclosed**.

**Training paradigm:** Historical base models use self-supervision. GPT-1 adds task SFT; GPT-3's cited evaluation learns only through the prompt. GPT-4 discloses RLHF after pretraining. Distillation and later reasoning-model recipes need their own evidence, not a family-wide assumption.

**Hardware/parallelism considerations:** The GPT-3 paper reports V100 training on a high-bandwidth cluster. It divides matrix calculations and layers across devices. GPT-4's hardware and training compute are explicitly withheld. API response time cannot reliably recover either.

### 3.12.2 LLaMA / Llama

**In plain English:** LLaMA models learn to continue text and make their weights available for local use. A base model can be adapted for a task, but is not already a complete assistant.

**Name:** Meta's LLaMA/Llama family. The detailed reference is **LLaMA 1, February 2023**, especially the models named 7B and 65B Base. Later generations are kept separate.

**Category & sub-category:** Self-supervised causal language-model pretraining. These are dense base models, with assistants produced through additional training.

**Originating paper/vendor/year:** Hugo Touvron and colleagues at Meta AI published [LLaMA, 2023](https://arxiv.org/html/2302.13971v1). [Llama 2](https://arxiv.org/html/2307.09288v2) is a separate 2023 release. [The Llama 3 report](https://arxiv.org/html/2407.21783v1) describes the 2024 generation, including a dense 405B model.

**Core mechanism:** A causal Transformer repeatedly predicts the next token. LLaMA 1 rescales inputs with RMSNorm before block transformations. Rotary positions, or RoPE, represent token positions by rotating parts of attention vectors. A gated FFN controls which transformed values pass through. The study emphasizes smaller models trained on substantial text budgets. Llama 2-Chat adds dialogue training; Llama 3 changes scale, data, tokenization, and later training. Neither is an alias for original 7B.

**Inputs/outputs and typical data types:** Base checkpoints take text and code tokens and return continuation probabilities or generated text. Chat variants need their documented conversation formats. Multimodal experiments in the 2024 report do not make original text-only LLaMA accept images.

**Strengths and limitations:** Downloadable weights support local testing, adaptation, and control over inference. Dense models are relatively straightforward to divide across devices. However, licenses and access vary by release. Open weights do not guarantee fully visible training data or a finished assistant.

**Computational complexity / scalability notes:** LLaMA 1 follows dense full-attention growth: longer inputs add comparisons and cache space. More weights also require more movement from memory and more FFN work. **Optional math:** $`p`$ denotes the parameter count, so increasing $`p`$ increases those weight-related costs. GQA in specified later Llama models must not be assumed for every LLaMA 1 layer.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Meta uses **MMLU**, a collection of academic and professional multiple-choice questions. Five demonstrations, a question, and answer choices enter LLaMA. It scores the choices, selects one, and compares it with the test key. This checks whether one model can reuse learned knowledge without a separate classifier for each subject.

[The original paper's Table 9](https://arxiv.org/html/2302.13971v1) reports **63.4% average five-shot accuracy for LLaMA-65B**, versus **46.9% for LLaMA-13B**. Accuracy is the share of correctly selected answers. More capacity helps in this comparison, but larger competitors also score higher in the paper. The result does not establish the best size for an organization's budget. It describes 2023 LLaMA 1, not Llama 2 or Llama 3, and is not a clinical or professional qualification.

**Notable vendor implementations/libraries:** Meta's release repositories, Hugging Face Transformers, and local runtimes such as llama.cpp support these models. Running quantized weights in a supported runtime does not guarantee the original score.

**Architecture diagram description:** LLaMA 1: `token embeddings -> [RMSNorm -> causal self-attention with RoPE -> residual; RMSNorm -> SwiGLU FFN -> residual] x L -> RMSNorm -> token head`. Here, L is the layer count. Each block first gathers allowed context, then transforms each token's vector.

**Activation functions used and why:** SwiGLU multiplies a SiLU-transformed gate by a second learned transformation. The gate changes how strongly values pass through. The original report adjusts the FFN's inner width to control parameter cost. Softmax forms attention weights and token probabilities.

**Loss function(s):** The base reference uses causal token cross-entropy: a penalty for poor next-token predictions. Llama 2-Chat adds SFT and preference/RLHF losses later. They are not hidden stages of every original base checkpoint.

**Optimization algorithm(s):** LLaMA 1 uses AdamW, 2,000 warmup updates, and cosine decay to 10% of peak. **Optional math:** Its gradient and squared-gradient averaging settings are $`(0.9,0.95)`$. The nominal 7B/13B runs peak at $`3\times10^{-4}`$; the larger two peak at $`1.5\times10^{-4}`$. These settings describe **LLaMA 1**, not every generation.

**Regularization techniques:** The original recipe specifies weight decay 0.1 and gradient clipping 1.0. RMSNorm and residual shortcuts help numerical behavior; filtering and duplicate removal address data quality. No undocumented dropout rate is assigned to the whole family.

**Backpropagation considerations:** Efficient attention, selective recomputation, and overlapping communication with calculations reduce overhead. Local SFT can update every weight or train small added modules called adapters. Updating fewer parameters does not remove the unchanged main model from inference memory.

**Parameter count / scaling behavior:** The original table gives approximately **6.7B, 13.0B, 32.5B, and 65.2B**, commonly named 7B/13B/33B/65B. The first two see **1.0T tokens**; the latter two see **1.4T**. Llama 2 spans 7B-70B. The **405B dense** model in the 2024 Llama 3 report is a different checkpoint.

**Training paradigm:** Base pretraining uses self-supervision. Instruction and preference training are separate named stages. MoE Llama releases and routing are covered in [the MoE volume](07-moe-models.md), not added to this dense recipe.

**Hardware/parallelism considerations:** LLaMA 1's 65B training run uses **2,048 A100 80-GB GPUs**. This is a training setup, not a minimum for running predictions. Quantization, splitting weights, batch size, and context length determine local serving needs.

### 3.12.3 Mistral Dense Models

**In plain English:** This Mistral model writes text while saving some attention memory. It shares stored attention information and limits each layer's direct view to a nearby window.

**Name:** Mistral's dense-language-model family, represented by **Mistral-7B-v0.1, 2023**. This entry does not cover Mixtral.

**Category & sub-category:** Self-supervised causal base modeling. The dense decoder uses grouped-query attention and sliding-window attention.

**Originating paper/vendor/year:** Albert Q. Jiang and colleagues at Mistral AI published the [Mistral 7B technical report, 2023](https://arxiv.org/html/2310.06825v1) and [v0.1 base model card](https://huggingface.co/mistralai/Mistral-7B-v0.1).

**Core mechanism:** The model predicts following tokens. GQA lets several attention queries share keys and values, while local windows limit direct comparisons. Across multiple layers, information can travel farther than one window. There is no router selecting among alternative expert FFNs. Sliding attention is not MoE.

**Inputs/outputs and typical data types:** BPE splits text and code into tokens, using bytes as a fallback for otherwise unrepresented pieces. The model returns token probabilities and continuations. Base does not have Mistral-Instruct's conversation tuning. Later dense releases can change vocabulary, context, or supported media; v0.1 does not specify them.

**Strengths and limitations:** A relatively small dense checkpoint can offer useful capabilities with less KV storage from GQA. Local attention can make long-sequence processing cheaper. It does not give every layer the same direct access to distant tokens as full attention. Base outputs can be unsafe, unsupported, or poor at following instructions.

**Computational complexity / scalability notes:** Once text exceeds the window, each token compares directly with only a limited local history. A compatible rolling cache can keep only that window per layer. Splitting sequences into chunks and updating the cache must be done correctly. Dense FFNs still run for every token.

**Optional math:** Local attention work is approximately $`O(BLT\min(T,W)d)`$, plus $`O(BLTd^2)`$ for projections and FFNs. Here, $`B`$ is batch size, $`L`$ layers, $`T`$ sequence length, $`W`$ window length, and $`d`$ width. The minimum selects the smaller of sequence and window lengths. A rolling cache can cap per-layer history at $`W`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Mistral's **MMLU** test covers multiple subjects using a shared re-run pipeline. Five demonstrations plus a test question and answer choices enter Mistral-7B-v0.1. It scores/selects an answer, then the evaluator checks the test key. A smaller local model may fit limited memory and response-time budgets. That is this book's technical rationale, not a documented customer purchase.

[Table 2](https://arxiv.org/html/2310.06825v1) reports **60.1% five-shot MMLU**, versus **55.6% for Llama 2 13B** in that paper. These are shares of correct answers under that prompting setup. They cannot be mixed with differently prompted Mistral results from other papers. No production KPI or universal superiority was shown.

**Notable vendor implementations/libraries:** Mistral's public inference code, Hugging Face Transformers, and compatible runtimes can run the model. A hosted API alias need not be the downloadable v0.1 checkpoint.

**Architecture diagram description:** `BPE embeddings -> dense causal Transformer blocks: RoPE/GQA with window W + gated FFN and RMSNorm/residual paths -> token head`. W is the attention-window length. The [released configuration](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json) specifies 32 layers, width 4,096, 32 query heads, and eight KV heads.

**Activation functions used and why:** The published configuration uses SiLU in the gate; the gated FFN is commonly called SwiGLU. This smoothly controls which transformed values pass through. Attention and token prediction use softmax.

**Loss function(s):** Base learns causal token prediction. The short release materials do not fully disclose original data weighting or any extra training losses. Losses used for instruction models must not be assumed for v0.1 Base.

**Optimization algorithm(s):** The original optimizer, numerical learning-rate schedule, warmup, and clipping recipe are **not publicly disclosed in the cited release report/card**. Compatibility with AdamW fine-tuning code does not identify the original optimizer.

**Regularization techniques:** RMSNorm and residual shortcuts are public design details. Original dropout, weight-decay coefficients, and the full data-filtering recipe are not publicly disclosed in those materials.

**Backpropagation considerations:** Attention masks must enforce the intended local view during training. An inference KV cache is stored prediction state, not the training computation graph. Original gradient-recomputation and numerical-precision details are not assumed.

**Parameter count / scaling behavior:** The model is nominally **7B parameters**. The report specifies **8,192 training context** and a **4,096 attention window**. The released configuration permits a larger positional limit. That limit alone does not prove equally good reasoning across longer contexts.

**Training paradigm:** Base training is self-supervised; instruction-tuned versions are separate releases. Mixtral's sparse experts and training to balance their use belong in [section 3.15](07-moe-models.md).

**Hardware/parallelism considerations:** Local weights can be split across devices or quantized. GQA reduces movement of KV data. No unpublished training GPU count is supplied. Check that the runtime implements v0.1's window behavior before counting on cache savings.

### 3.12.4 Qwen Dense Models

**In plain English:** This Qwen model learns to continue multilingual text, code, and mathematical writing. Its base weights are a starting point for adaptation, not a ready-made chat assistant.

**Name:** Alibaba's dense Qwen family, represented by **Qwen2.5-7B Base, September 2024**. It is not Qwen2.5-Instruct, Qwen-VL, or a Qwen MoE checkpoint.

**Category & sub-category:** Self-supervised multilingual causal language modeling. These are dense base models; later training stages are documented separately.

**Originating paper/vendor/year:** Alibaba's Qwen Team released the [Qwen2.5 model card, 2024](https://huggingface.co/Qwen/Qwen2.5-7B) and [December 2024 technical report](https://arxiv.org/html/2412.15115v1).

**Core mechanism:** A dense causal Transformer predicts next tokens. GQA shares keys/values; RoPE represents positions through rotations. QKV bias adds learned offsets to query, key, and value calculations. RMSNorm rescales values and SwiGLU gates the FFN. The report emphasizes data selection, scale, math/code text, and long-context adaptation. The name "Qwen" alone does not identify a design or training stage.

**Inputs/outputs and typical data types:** Multilingual prose, source code, and structured data written as text become token probabilities and generated text. Base is not recommended as a ready conversation agent. Native image input belongs to specifically multimodal releases.

**Strengths and limitations:** The checkpoint supports local adaptation for multilingual and technical tasks under its own license. It can still produce invalid JSON or wrong numerical reasoning. Language coverage does not mean equal accuracy across languages. A context limit is not proof of recall at every position.

**Computational complexity / scalability notes:** Dense-decoder costs apply. GQA shrinks stored KV state; it does not turn the model into a partly active expert system. **Optional math:** $`p`$ means the total parameter count, not a smaller active-expert count here. Extending positional handling does not remove full attention's roughly quadratic length cost. Long contexts need settings appropriate to the release.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The Qwen2.5 report uses **GSM8K**, held-out grade-school math word problems. Four demonstrations and a problem enter Qwen2.5-7B Base. It generates a solution/answer, and the evaluator extracts the answer and checks the test key. This avoids writing a separate symbolic parser for every wording. A calculator or solver can still be more reliable for arithmetic.

The [7B+ base-model table](https://arxiv.org/html/2412.15115v1) reports **85.4% for Qwen2.5-7B**, versus **80.2% for Qwen2-7B**, under its **four-shot** protocol. These percentages measure correct extracted answers. They are Base results, not Instruct scores, and do not show improved school outcomes.

**Notable vendor implementations/libraries:** Alibaba's Qwen repositories, Hugging Face Transformers, and compatible runtimes support these weights. Qwen API products and local base weights have different interfaces. Check licenses for each size and release, rather than assuming this 7B example's terms apply throughout.

**Architecture diagram description:** `token embedding -> [RMSNorm -> RoPE/GQA with QKV bias -> residual; RMSNorm -> SwiGLU dense FFN -> residual] x 28 -> vocabulary head`. Each of the 28 blocks gathers context, then transforms each position's vector.

**Activation functions used and why:** SwiGLU uses a SiLU gate to control transformed values. Softmax supplies attention weights and output probabilities. These disclosed dense Qwen2.5 details do not specify every Qwen vision or MoE component.

**Loss function(s):** Base uses causal token cross-entropy. The report discusses SFT and preference/reinforcement-learning stages separately for post-trained models. Their answer and preference losses must not be added to Base's description.

**Optimization algorithm(s):** The report uses measured scaling patterns to guide learning-rate and batch-size selection. It does not give a complete optimizer and numerical warmup/decay recipe for this 7B release. Those settings are **not publicly disclosed in the cited materials**. An earlier Qwen recipe is not substituted.

**Regularization techniques:** Pre-normalization with RMSNorm and residual shortcuts help stabilize calculations. The report describes filtering and selecting higher-quality data. The model card does not establish this checkpoint's dropout and weight-decay settings.

**Backpropagation considerations:** Dense training still updates the full trainable weight set. Conversation fine-tuning must correctly mark which roles' tokens receive loss. Filtering synthetic text and checking for benchmark overlap improve targets; they do not change backpropagation's basic rule.

**Parameter count / scaling behavior:** The card lists **7.61B total parameters**, **6.53B non-embedding**, **28 layers**, and **28 query / four KV heads**. Non-embedding excludes the token lookup-related weights. The report's expanded **18T-token corpus scale** describes family-level data. It does not prove that every size saw the same token budget.

**Training paradigm:** Base uses self-supervision. Instruction and preference training are separate stages. The report includes AI-assisted data generation and selection, but this does not establish teacher-logit matching for every checkpoint.

**Hardware/parallelism considerations:** BF16 weights alone need about **15.2 GB**; serving also needs caches and buffers. **Optional math:** $`2\times7.61`$ billion bytes multiplies two bytes per weight by 7.61 billion weights. Sharding, quantization, and optimized attention change practical capacity. No undocumented original GPU count is assumed.

### 3.12.5 DeepSeek LLM Dense Base

**In plain English:** These original DeepSeek models learn to continue English, Chinese, and code. They are dense text models, not the later DeepSeek expert models or reasoning-trained releases.

**Name:** **DeepSeek LLM 7B/67B Base**, the original dense family in the January 2024 report. DeepSeek-V2, V3, and R1 are different releases.

**Category & sub-category:** Self-supervised bilingual causal language modeling. Dense Base pretraining and the separately released Chat adaptations are distinguished.

**Originating paper/vendor/year:** DeepSeek AI published [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism, 2024](https://arxiv.org/html/2401.02954v1), alongside the [67B Base model card](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base).

**Core mechanism:** A LLaMA-like Transformer normalizes before transformations and predicts next tokens in English/Chinese text. The study tests scale and a learning rate that drops in steps. The dense 67B model uses GQA to share keys and values; 7B uses ordinary multi-head attention.

**Inputs/outputs and typical data types:** Chinese/English prose, mathematical text, and code tokens produce continuation probabilities and text. Chat models add response-focused adaptation. These text-only base weights do not contain later DeepSeek MoE routing or reasoning-RL procedures.

**Strengths and limitations:** Public design details and a stated schedule make the family useful for studying scale. Substantial pretraining supports bilingual and technical tasks. Open weights still do not reveal every training document. A base model is not trained for every requirement of interactive use.

**Computational complexity / scalability notes:** Both sizes have dense Transformer costs. GQA in 67B reduces KV memory, but its FFN weights still run densely. Its 95 layers also add a long chain of dependent computations and communication. Similar total parameter counts need not imply similar costs for shallower models.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** DeepSeek's **HumanEval** test asks for Python functions. A function signature and written specification enter 67B Base. It generates an implementation, which is run against unit tests. This can handle varied prose requests better than a fixed template library, but tests and human review remain necessary.

[Table 5](https://arxiv.org/html/2401.02954v1) reports **42.7% zero-shot pass@1** for DeepSeek LLM 67B Base, versus the authors' **28.7% Llama 2 70B** result. Pass@1 measures success from one candidate under the evaluation procedure. Here, their internal framework uses **greedy generation**, choosing the highest-scoring next token, without task demonstrations. Other HumanEval sampling procedures are not interchangeable with this one. A separately adapted Chat model's much higher score is not the Base score. No commercial engineering-time saving was claimed.

**Notable vendor implementations/libraries:** DeepSeek's [DeepSeek-LLM repository](https://github.com/deepseek-ai/DeepSeek-LLM) and Hugging Face Transformers support the historical models. The repository's code and model licenses are separate. Neither reveals the internals of every later DeepSeek API endpoint.

**Architecture diagram description:** `token embeddings -> pre-normalized RMSNorm/RoPE causal attention + SwiGLU FFN residual blocks -> token head`. RMSNorm rescales, RoPE represents positions, and SwiGLU gates FFN values. The 67B model groups KV heads; 7B has as many KV heads as query heads.

**Activation functions used and why:** SwiGLU provides a gated nonlinear FFN transformation. Softmax creates attention weights and vocabulary probabilities. These choices come from the original dense report, not guesses based on V3.

**Loss function(s):** Base uses causal token cross-entropy. The report adds SFT and DPO for Chat. DPO learns from preferred/rejected answer pairs; it is not the later R1 reasoning-RL procedure.

**Optimization algorithm(s):** AdamW uses weight decay 0.1 and 2,000 warmup steps. **Optional math:** Its gradient and squared-gradient averaging settings are $`(0.9,0.95)`$. Peak rates are **$`4.2\times10^{-4}`$ for 7B** and **$`3.2\times10^{-4}`$ for 67B**. At 80% of the token budget, the rate falls to 31.6% of peak; at 90%, it falls to 10%.

**Regularization techniques:** The recipe uses weight decay, gradient-norm clipping 1.0, RMSNorm, and residual shortcuts. Building the corpus and checking benchmark overlap are separate data controls, not numerical regularization.

**Backpropagation considerations:** Clipping limits large gradient spikes. The stepped rate schedule allows an earlier training phase to be reused when extending a run. That is the report's engineering reason, not proof that it always beats cosine decay.

**Parameter count / scaling behavior:** Nominal sizes are **7B and 67B**, each trained on **2T tokens**. The 7B model has **30 layers, width 4,096, 32 query/KV heads**. The 67B model has **95 layers, width 8,192, 64 query and eight KV heads**. Both report 4,096-token context.

**Training paradigm:** Base pretraining is self-supervised; Chat adds supervised and preference-based adaptation. [MoE model families](07-moe-models.md) covers DeepSeekMoE/V2/V3 and separately discusses reasoning post-training. Those entries are not duplicated here.

**Hardware/parallelism considerations:** Dense 67B training needs state and intermediate results managed across devices. GQA helps inference-cache efficiency. A multi-GPU or quantized serving example does not establish the original training cluster size or precision.

### 3.12.6 BLOOM

**In plain English:** BLOOM is a large text generator built through a multilingual research collaboration. Its public weights and documentation support study, but running the largest version needs substantial memory.

**Name:** BLOOM means BigScience Large Open-science Open-access Multilingual Language Model. The reference is **BLOOM-176B, 2022**, not the instruction-tuned BLOOMZ.

**Category & sub-category:** Self-supervised multilingual causal modeling. This is a dense research base model developed collaboratively.

**Originating paper/vendor/year:** The BigScience collaboration includes Hugging Face and many academic and industry contributors. Sources are its [2022 report](https://arxiv.org/html/2211.05100v4) and [official model card](https://huggingface.co/bigscience/bloom). Hugging Face organized and supplied implementations; it was not the project's sole inventor.

**Core mechanism:** A decoder-only Transformer learns to continue text from the multilingual ROOTS corpus. ALiBI adds position-related biases to attention scores instead of learned absolute position embeddings. An extra LayerNorm after token embedding helps stabilize large-model training.

**Inputs/outputs and typical data types:** Text in the documented natural and programming languages becomes next-token probabilities and continuations. A task instruction can be placed in a prompt. BLOOMZ separately adds explicit multilingual, multitask instruction training.

**Strengths and limitations:** Public weights, corpus documentation, and the multilingual research process support scrutiny and adaptation. Including a language in training does not guarantee strong results on every task in it. The large dense checkpoint is costly to serve, and Base can produce biased or unsafe text.

**Computational complexity / scalability notes:** Dense attention and FFNs follow the shared cost model. ALiBI changes comparison scores, not the roughly quadratic number of full-attention comparisons. A position formula that extends beyond training length does not by itself establish a reliable long-context model.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** BigScience uses **HumanEval** to test Python code completion. BLOOM-176B reads a held-out function specification, generates a body, and has it checked with unit tests. The evaluation estimates pass@k: the chance that at least one of k candidates passes. This asks whether a broad multilingual model can also help with programming. Specialized code training is an alternative when code correctness is the priority.

[Table 9 in report version v4](https://arxiv.org/html/2211.05100v4) gives **15.52% pass@1** for BLOOM and **12.06% for BLOOMZ**. Pass@1 concerns one candidate. BLOOMZ's instruction mixture is not mainly pure code completion, so general instruction tuning need not help this task. Non-BLOOM comparison scores come from prior work, not a shared re-run. Sampling settings and paper/card revisions must be retained when reproducing results. No production coding KPI was established.

**Notable vendor implementations/libraries:** BigScience's Megatron-DeepSpeed fork, Hugging Face Transformers, and serving integrations support BLOOM. Its RAIL model license is not the same as an unrestricted software-code license.

**Architecture diagram description:** `token embeddings + embedding LayerNorm -> 70 causal Transformer blocks with ALiBI attention biases and GELU FFNs -> vocabulary distribution`. The extra normalization acts before the 70 repeated text-processing blocks.

**Activation functions used and why:** GELU smoothly transforms FFN values. Softmax supplies attention weights and output probabilities. Combining bias and GELU operations in a fused kernel improves execution efficiency without changing the activation's purpose.

**Loss function(s):** Pretraining averages token cross-entropy losses for next-token prediction. BLOOMZ's later multitask targets belong to a separate supervised instruction stage.

**Optimization algorithm(s):** The 176B training table specifies Adam, **375M warmup tokens**, and cosine decay scheduled over **410B tokens**. **Optional math:** Gradient and squared-gradient averaging settings are $`(0.9,0.95)`$. Peak learning rate is **$`6\times10^{-5}`$**, with a $`6\times10^{-6}`$ floor. The checkpoint reports **366B tokens seen**. A planned 410B-token decay schedule does not mean the checkpoint completed that schedule.

**Regularization techniques:** The recipe uses weight decay 0.1, gradient clipping 1.0, LayerNorm, and carefully designed data processing. Position biases and embedding normalization help design or numerical behavior; they do not prove the model cannot memorize text.

**Backpropagation considerations:** Distributed optimizer records, recomputing intermediate results, and fused operations help manage the large dense model. The paper describes numerical-stability and communication work. Lower-precision training must be checked at the actual scale.

**Parameter count / scaling behavior:** The model card lists **176,247,271,424 parameters**, **70 layers**, **112 attention heads**, width **14,336**, and **2,048 training sequence length**. The data spans **46 natural and 13 programming languages**. These are not the sizes or coverage of every smaller BLOOM checkpoint.

**Training paradigm:** BLOOM uses self-supervised language modeling. BLOOMZ adds the separate xP3 multitask instruction stage. Putting a multilingual demonstration in an inference prompt does not retrain Base.

**Hardware/parallelism considerations:** The reference run uses **384 A100 80-GB GPUs on the Jean Zay supercomputer**, with Megatron-DeepSpeed parallelism. Two-byte BF16 weights alone need about 352.5 GB by arithmetic. Caches and working buffers need more; training state is much larger again.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| GPT family | Text/code; images in specified GPT-4 systems | Can attempt tasks through generated text and prompt examples | Newer internals are undisclosed; hosted versions differ from historical weights | GPT-3's LAMBADA final-word test |
| LLaMA / Llama | Text/code needing local adaptation | Provides dense weights and documented research generations | Recipes, licenses, and accepted media change by release | LLaMA 1's five-shot MMLU question test |
| Mistral dense | Text/code where serving memory is limited | Shares KV state and limits attention windows in a small dense model | Original v0.1 training details are only partly disclosed | Mistral-7B-v0.1's MMLU question test |
| Qwen dense | Multilingual text, mathematics, and code | Offers multilingual and technical abilities in named checkpoints | Base, Instruct, VL, and MoE versions must stay separate | Qwen2.5-7B's GSM8K math test |
| DeepSeek LLM dense base | Chinese/English prose and code | Documents dense-model scaling and stepped learning rates | Does not describe V2/V3/R1 designs or reinforcement learning | DeepSeek LLM 67B Base's HumanEval code test |
| BLOOM | Multilingual text and research needing public documentation | Offers collaborative documentation and broad language coverage | Large dense weights are costly; language/task quality varies | BLOOM-176B's HumanEval code test |

## 3.13 Vendor Multimodal and Assistant Families

These entries group vendor families, not one shared design or learning method. Multimodal means working with more than one kind of input, such as text and images. Command R is text-only in the examined release. Original ERNIE is an encoder. Titan includes separate embedding and generation products. Paired media, such as an image and caption, supplies a learning signal across input types. Each entry separates visible service behavior, published training stages, public code, and genuinely unknown internals.

### 3.13.1 Anthropic Claude

**In plain English:** Claude can read conversations and supported images, then write a response. It can help examine supplied documents, but its answers still need checking.

**Name:** Claude, represented by the **Claude 3 family announced in March 2024**. Claude 3 Opus supplies the benchmark example. Haiku, Sonnet, and Opus are separate offerings; their names do not disclose parameter-count ranges.

**Category & sub-category:** A proprietary pretrained assistant family with multimodal input and later training for desired behavior, often called alignment. Grouping it with self-supervised foundations does not make every Claude training stage unsupervised.

**Originating paper/vendor/year:** Anthropic's [Claude 3 release and linked model card, 2024](https://www.anthropic.com/news/claude-3-family) describe the family. Its [Constitutional AI research, 2022](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) explains a related alignment method. That research is not a complete recipe for every Claude release.

**Core mechanism:** The interface generates text using a conversation and, for Claude 3, supplied images. Anthropic describes Constitutional AI and safety-tuning work. It does not publish Claude 3's complete layer graph, dataset list, or optimizer recipe. Fluent writing does not establish a decoder-only or MoE design.

**Inputs/outputs and typical data types:** Conversations and images, including photos, charts, and technical diagrams, produce generated text. A product may separately prepare PDFs, retrieve documents, or coordinate tools. Those steps can happen outside one model call.

**Strengths and limitations:** Long-document handling and image interpretation can support analysis. However, hidden internals limit independent reproduction. Citations and confident wording can still be wrong. Finding one inserted fact is much narrower than reliably reasoning over an entire document collection.

**Computational complexity / scalability notes:** Layer count, width, attention pattern, routing, and compute are **not publicly disclosed** for this reference. Measure a named endpoint using specified prompt/output lengths, response times, request limits, and costs. The shared dense-Transformer formula is not a known Claude implementation formula.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Anthropic's **Needle In A Haystack (NIAH)** test checks long-document retrieval. The experiment inserts a fact into a crowdsourced document corpus and chooses from **30 needle/question pairs**. Opus reads the context, answers the question, and is scored on recovering the fact. This tests finding supplied evidence rather than guessing from pretraining. A separate search system feeding a shorter context is an alternative, but that index can introduce another failure.

The [release reports Opus exceeding 99% accuracy](https://www.anthropic.com/news/claude-3-family) on this NIAH evaluation. This is a vendor-reported test with deliberately inserted facts. It is **not** 99% accuracy on arbitrary documents, an independently repeated evaluation, or a production search KPI. The announcement does not give enough detail to reconstruct every scoring condition.

**Notable vendor implementations/libraries:** Anthropic's API and SDKs, the hosted Claude application, and specified partner services provide access. They are product or communication layers. They do not expose Claude's weights or provide a public local architecture implementation.

**Architecture diagram description:** Functional view only: `conversation + supported images -> proprietary Claude 3 inference -> response text -> optional application-side validation/tools`. This shows the workflow, not internal layers. Media encoders, how media are combined, attention connections, and expert routing are **not publicly disclosed** here.

**Activation functions used and why:** **Not publicly disclosed.** Another vendor's GELU or SwiGLU choice cannot fill this gap. API controls for response probabilities do not reveal internal activation functions.

**Loss function(s):** Claude 3's complete loss mixture is **not publicly disclosed**. Separate Constitutional AI experiments train on self-critiques and revised answers, then use preference-model-based RLAIF. Those experiments do not specify all Claude 3 pretraining or alignment losses.

**Optimization algorithm(s):** The gradient optimizer, learning rates, warmup, rate decay, and per-stage coefficients are **not publicly disclosed** for Claude 3. No other model's numerical recipe is substituted.

**Regularization techniques:** Internal dropout, weight decay, normalization, and other numerical controls are **not publicly disclosed**. Documented safety tests and content policies are system practices, not answers to these missing neural-design questions.

**Backpropagation considerations:** Hosted inference returns no training gradients. Gradient precision, clipping, recomputation, and how gradients are combined across devices are **not publicly disclosed**. Training a student on permitted outputs would still not expose Claude's internal computation graph.

**Parameter count / scaling behavior:** Total parameters and parameters active for a request are **not publicly disclosed**. The launch's **200K context offering** describes how much fits in context, not how many weights the model has. Capability tiers do not reveal differences in parameter numbers or compute.

**Training paradigm:** Proprietary foundation pretraining is followed by alignment stages. Anthropic publishes relevant human- and AI-feedback research. The exact balance of self-supervision, SFT, RLHF/RLAIF, and any distillation for this release is not completely disclosed.

**Hardware/parallelism considerations:** Training devices, cluster count, sharding, and serving parallelism are **not publicly disclosed** in the cited release materials. Cloud access and corporate hardware partnerships do not identify the hardware setup used to train Claude 3.

### 3.13.2 Google / DeepMind Gemini

**In plain English:** Gemini can use text together with images, audio, and video to answer tasks. The examined versions can read long reference material, but long context does not guarantee complete understanding.

**Name:** Gemini, represented by **Gemini 1.5 Pro and 1.5 Flash in the 2024 technical report**. Gemini 1.0 is the preceding generation, not the same checkpoint.

**Category & sub-category:** Proprietary multimodal pretraining followed by instruction and preference adaptation. Gemini 1.5 Pro's disclosed MoE design is linked to the MoE discussion, not duplicated as another entry here.

**Originating paper/vendor/year:** Google/Google DeepMind published the [Gemini 1.5 technical report, 2024, v5](https://arxiv.org/html/2403.05530v5), following Gemini 1.0 in 2023. Gemini applications, the Gemini API, and Vertex AI are distinct from the trained checkpoints.

**Core mechanism:** The model can use a context that mixes text, images, audio, and video. The report names **Gemini 1.5 Pro as a sparse MoE Transformer-based model**. It names **1.5 Flash as a Transformer decoder**, with attention and FFN calculations in parallel and online distillation from Pro. Online distillation means the teacher supplies learning information during student training. These disclosures do not give a complete reproducible architecture.

**Inputs/outputs and typical data types:** Mixed text/code, images, audio, and video lead to generated text and task responses in the examined evaluations. Which inputs an API accepts, and how much context it permits, must be checked for the specific release.

**Strengths and limitations:** A long context can hold reference materials without task-specific fine-tuning. Multimodal input need not first turn every signal into text through an outside transcription system. Still, relevant evidence can be missed. Proprietary training details limit reproducibility.

**Computational complexity / scalability notes:** MoE can store many parameters while running only a subset per token. Pro's total/active counts, number of selected experts, and connection layout are **not publicly disclosed**. Reported long-context changes do not supply a complete cost model. Flash's response time does not reveal Pro's parameter count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Google's **Machine Translation from One Book (MTOB)** experiment tests English-to-Kalamang translation using reference material. The context contains approximately a **500-page grammar**, a bilingual wordlist, and parallel example sentences. Gemini reads a new sentence and writes a translation. This offers a way to attempt translation when paired training examples are scarce. It is not a documented decision to deploy a translation service.

For the **full-book English-to-Kalamang evaluation**, [Table 5](https://arxiv.org/html/2403.05530v5) reports a **5.46/6 human rating** and **59.0 chrF** for Gemini 1.5 Pro. chrF compares overlapping character sequences with reference translations; it is not a percent-perfect translation score. The human rater is a non-native, non-fluent learner who can recognize their own comparison translations. These scores are not native-speaker certification. Performance without the reference context is poor, and the reverse translation direction has a larger human-model gap. No community-service or business KPI was reported.

**Notable vendor implementations/libraries:** Google's Gemini API, SDKs, and Vertex AI provide serving access. Public client code explains requests, not the proprietary training implementation.

**Architecture diagram description:** Published-level views: `mixed-modal inputs -> Gemini 1.5 Pro sparse-MoE Transformer -> text`; separately, `inputs -> 1.5 Flash Transformer decoder with parallel attention/FFN -> text`. Input-processing components, layer dimensions, routing, and all internal connections are not fully disclosed. See [MoE mechanisms](08-moe-deep-dive.md).

**Activation functions used and why:** Per-component activations are **not publicly disclosed** in the cited report. A familiar SwiGLU block or a guessed image encoder must not be inserted as if verified.

**Loss function(s):** The report discusses next-token prediction, multimodal instruction-response tuning, human-preference tuning, and Flash distillation. Complete loss mixtures, teacher/student loss weights, and extra routing losses are **not publicly disclosed**.

**Optimization algorithm(s):** Flash uses **higher-order preconditioned methods**. In basic terms, these use information about the loss's shape to rescale update directions. That is a real disclosure, but it does not identify an exact optimizer or learning-rate schedule. Those numerical details, and Pro's full optimizer recipe, are **not publicly disclosed**.

**Regularization techniques:** The report describes selecting and cleaning data and later safety training. Dropout, weight decay, normalization placement, and other checkpoint-specific numerical settings are **not publicly disclosed**.

**Backpropagation considerations:** Flash's online distillation is explicitly reported, not guessed from its smaller role. The handling of teacher gradients, distribution of gradients, and saved/recomputed layer outputs are not publicly disclosed. MTOB's use of reference material in the prompt does **not** update weights for the translation task.

**Parameter count / scaling behavior:** Pro's total/active counts and Flash's parameter count are **not publicly disclosed** in the 2024 report. Long-context demonstrations are neither weight counts nor guarantees of every API user's allowance.

**Training paradigm:** The reported stages include multimodal pretraining, supervised multimodal instructions, human-preference adaptation, and Flash distillation. Paired media supplies supervision, so the family is not purely unsupervised.

**Hardware/parallelism considerations:** The report states **multiple 4,096-chip TPUv4 pods across datacenters** were used for training. It does not give a complete per-model chip count or sharding plan. The published hardware detail is retained without inventing the missing quantities.

### 3.13.3 Cohere Command R

**In plain English:** Command R can answer using supplied documents and point to supporting passages. It can also request tools, but the surrounding application must actually run them.

**Name:** Command R, represented by **`c4ai-command-r-v01`, March 2024**, a 35B research-weight release. It is not Command R+ or a later hosted alias.

**Category & sub-category:** An autoregressive pretrained language model with supervised/preference training for source-based answers and tool use. This particular release is **text-only**, despite its location among vendor assistant families.

**Originating paper/vendor/year:** Cohere and Cohere For AI, now Cohere Labs, provide the [Command R model card, 2024](https://huggingface.co/CohereLabs/c4ai-command-r-v01). The [Transformers v4.40.0 implementation](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/cohere/modeling_cohere.py) exposes model-design details, not the original training data or optimizer.

**Core mechanism:** Conversations, retrieved snippets, or tool descriptions guide generation. In grounded generation, the model can identify relevant documents, write an answer, and mark supporting source spans. A tool request is a proposed action. The **application**, not the neural weights, makes the API call and returns its result.

**Inputs/outputs and typical data types:** Text conversations, document snippets with metadata, and descriptions of tool inputs produce answer text, citations, or structured tool requests. A separate retriever finds the documents. Command R is not itself a vector-search index.

**Strengths and limitations:** Training specific source and tool formats better matches document assistance than generic chat alone. Yet a citation is still a prediction, not proof of support. Missing or malicious retrieved text, changed prompt templates, and failed tools can lead to wrong final answers.

**Computational complexity / scalability notes:** The public causal Transformer has dense-decoder costs. Longer prompts require more attention work and KV memory. Retrieval can reduce how much evidence is passed in, but indexing/search has its own cost. A maximum context window does not mean every request uses the same memory.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** Cohere's **official grounded-generation demonstration** asks which penguin is largest. It supplies two snippets about emperor penguin height and habitat. The conversation and snippets enter Command R's grounding template. The generated answer cites the relevant documents separately, letting the reader follow the marked sources.

The [model card shows a rendered completion](https://huggingface.co/CohereLabs/c4ai-command-r-v01) identifying the emperor penguin and linking supporting statements to document IDs. This makes an answer easier to check than one produced without sources. It is a **vendor reference demonstration, not a customer deployment or independently executed test**. No overall factuality rate, citation-accuracy score, or business KPI was reported.

**Notable vendor implementations/libraries:** Cohere's hosted API/SDK, Cohere Labs research weights, and Hugging Face Transformers provide different ways to use the model. The research license and commercial service terms are separate.

**Architecture diagram description:** Public reference: `tokens -> repeated pre-LayerNorm -> {causal RoPE attention || gated MLP} -> sum both branches with residual -> final normalization -> token logits`. The two branches run in parallel and are added to the shortcut. This differs from a simple Llama block that runs them sequentially. Logits are output scores before conversion to probabilities.

**Activation functions used and why:** The [public reference configuration](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/cohere/configuration_cohere.py) uses SiLU in a gated MLP. The gate controls transformed values; attention uses softmax. These are implementation details for this reference, not a vendor-wide training recipe.

**Loss function(s):** The public implementation supports cross-entropy with each token predicting the next, called shifted-token loss. The model card reports SFT and preference training, including grounded answers. It does not disclose the complete original preference loss or its coefficients.

**Optimization algorithm(s):** Original pretraining and post-training optimizers and numerical learning-rate schedules are **not publicly disclosed in the cited card**. A library's training interface does not establish what optimizer Cohere used.

**Regularization techniques:** LayerNorm and residual shortcuts are visible in the implementation. Attention dropout defaults to zero in the reference configuration. That default is not a full record of the original run's weight decay, filtering, or other controls.

**Backpropagation considerations:** Public weights allow local supervised adaptation under the license. Training targets must keep the grounding markers and response formats. Ordinary external retrieval and tool calls do not automatically pass gradients back. An added training method would have to provide that connection.

**Parameter count / scaling behavior:** The card specifies **35B parameters and a 128K context window**. Not every Command model has that size. At two bytes per weight, BF16 weights alone need approximately 70 GB.

**Training paradigm:** Autoregressive pretraining is followed by SFT and preference training. The exact RLHF/RLAIF procedures, dataset shares, and any distillation recipe are not publicly disclosed in this card.

**Hardware/parallelism considerations:** The card describes options for loading quantized weights. Local serving needs a plan for splitting model calculations/weights and storing the KV cache. Original training hardware and the full parallelism strategy are not publicly disclosed.

### 3.13.4 Baidu ERNIE

**In plain English:** Original ERNIE learns by hiding whole phrases and named things, then rebuilding them. This helps it learn meaning beyond isolated Chinese characters; later ERNIE assistants are different systems.

**Name:** Baidu's ERNIE family. The algorithm reference is **ERNIE 1.0, 2019**. The public `ernie-1.0-base-zh` implementation and **ERNIE 4.0/ERNIE Bot, 2023**, are identified separately.

**Category & sub-category:** Original ERNIE is a self-supervised encoder using phrase/entity knowledge to choose what to hide. Later products form a broader text-generation and assistant family.

**Originating paper/vendor/year:** Yu Sun and colleagues at Baidu published [ERNIE: Enhanced Representation through Knowledge Integration, 2019](https://arxiv.org/html/1904.09223v1). Another group's knowledge-graph paper uses the same acronym but is different work. Baidu's [October 2023 ERNIE 4.0 announcement](https://en.prnasia.com/releases/global/baidu-launches-ernie-4-0-foundation-model-leading-a-new-wave-of-ai-native-applications-422575.shtml) is a vendor product disclosure, not a specification of the 2019 encoder.

**Core mechanism:** Instead of hiding only separate tokens or Chinese characters, ERNIE hides complete phrases and entities, such as named people or places. The model must then use wider context rather than copy a visible part of the same entity. It remains a bidirectional Transformer. It does not require adding a knowledge-graph vector at every layer.

**Inputs/outputs and typical data types:** Original ERNIE takes Chinese text and sentence pairs. It returns context-aware vectors, hidden-token predictions, or task labels. ERNIE Bot's generated answers and multimodal product demonstrations belong to later systems. Their interfaces do not reveal how many models or media-processing parts they contain.

**Strengths and limitations:** Hiding meaningful units can improve Chinese meaning representations compared with character-level shortcuts. However, deciding phrase boundaries or identifying entities can introduce errors and use upstream supervision. Original ERNIE is not a blueprint for a later generative assistant.

**Computational complexity / scalability notes:** ERNIE 1.0 keeps dense-encoder attention and FFN costs. Finding phrases and entities adds separate preprocessing work. Its dimensions do not establish ERNIE 4.0's operation count, KV-cache needs, or routing.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Baidu's 2019 **Chinese XNLI** test compares a premise with a proposed statement, or hypothesis. A supervised head on ERNIE classifies whether the premise supports it, contradicts it, or leaves it undecided: entailment, contradiction, or neutral. The chosen class is compared with the test label. Whole expressions matter for this task, not just shared characters.

The [paper's results table](https://arxiv.org/html/1904.09223v1) reports **78.4% test accuracy** for ERNIE versus **77.2% for its BERT baseline**. Accuracy is the share of correct relation labels. ERNIE's **79.9%** is a development score, not test. These findings concern the original encoder, not ERNIE 4.0, search-product conversion, or measured business gains.

**Notable vendor implementations/libraries:** Baidu supplies PaddlePaddle/PaddleNLP ERNIE implementations. ERNIE Bot and Qianfan are product/platform layers. A platform hosting multiple foundation models does not show they share one ERNIE architecture.

**Architecture diagram description:** ERNIE 1.0: `token + position + segment embeddings -> bidirectional Transformer encoder -> MLM/task head`. Phrase/entity selection determines which input units are hidden for masked language modeling. ERNIE 4.0's full internal diagram is **not publicly disclosed in the cited announcement**.

**Activation functions used and why:** The named [PaddleNLP `ernie-1.0-base-zh` configuration](https://github.com/PaddlePaddle/PaddleNLP/blob/v2.8.1/paddlenlp/transformers/ernie/configuration.py) uses **ReLU**, not BERT's GELU. ReLU sets negative values to zero and passes positive ones through. Other named configurations in that file differ. This does not mean all ERNIE assistants use ReLU.

**Loss function(s):** Original ERNIE reconstructs hidden tokens. Its paper also explores dialogue-related pretraining and supervised task losses. ERNIE 4.0's full mixture of pretraining and alignment losses is **not publicly disclosed** by its announcement.

**Optimization algorithm(s):** The cited 2019 paper does not list a complete optimizer and learning-rate schedule. The 2023 announcement also does not disclose ERNIE 4.0's. BERT's numbers cannot silently fill either gap.

**Regularization techniques:** The named PaddleNLP reference uses **0.1 hidden and attention dropout**, plus normalization and residual shortcuts. Phrase/entity-aware masking changes the learning task. Later product-specific dropout, weight decay, and normalization remain undisclosed here.

**Backpropagation considerations:** Gradients update the encoder and labeled-task head. They do not update the separate, discrete choice of which phrases/entities to hide. Upstream weak annotations are not the same thing as a knowledge-retrieval module trained through gradients.

**Parameter count / scaling behavior:** The original paper specifies **12 layers, width 768, and 12 attention heads**. Its vocabulary differs from English BERT's, so copying BERT's parameter total would be wrong. ERNIE 4.0's parameter count or total/active split is **not publicly disclosed in the cited release**.

**Training paradigm:** Original ERNIE reconstructs Chinese text from varied corpora using self-supervision, then adapts to labeled tasks. Later assistant stages need release-specific evidence. The brand alone does not establish SFT, RLHF, RLAIF, or distillation.

**Hardware/parallelism considerations:** Public encoder implementations can split batches across GPUs. The original paper and cited ERNIE 4.0 announcement do not give complete training-hardware or distributed-optimizer specifications.

### 3.13.5 Amazon Titan

**In plain English:** The Titan embedding models turn text or images into lists of numbers for comparison. An application can use those lists to find similar catalog items; the embedding model itself does not run the search.

**Name:** Amazon Titan, represented by **Titan Multimodal Embeddings G1** and compared with **Titan Text Embeddings V2**. Titan Text generation and Titan Image Generator are different models, not modes of one disclosed network.

**Category & sub-category:** A proprietary family with multimodal/text representation models and separate generation products. Customizing with paired images and captions supplies supervised learning across media; it is not pure unlabelled learning.

**Originating paper/vendor/year:** Amazon/AWS's [2023 multimodal release](https://aws.amazon.com/blogs/aws/amazon-titan-image-generator-multimodal-embeddings-and-text-models-are-now-available-in-amazon-bedrock/), [Multimodal Embeddings G1 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html), and [Text Embeddings V2 documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html). Text V2 is a separate 2024-generation text model.

**Core mechanism:** The model turns images and/or short text into vectors in a shared comparison space. Nearby vectors can represent related content. The embedding architecture and learning objective are not sufficiently disclosed to call it CLIP, a particular two-encoder design, or a decoder-only Transformer. Titan's generation products do different jobs.

**Inputs/outputs and typical data types:** Multimodal G1 accepts an image, English text, or both and returns a vector, meaning a list of numbers. Text V2 turns text into a vector. Neither embedding endpoint itself writes a natural-language answer or searches a database for nearest neighbors.

**Strengths and limitations:** Shared vectors support text-to-image and image-to-image search without exact keyword matches. Similarity does not prove object identity or semantic truth. Shorter vectors can change retrieval quality. Hidden internals and service revisions make reproduction harder.

**Computational complexity / scalability notes:** The neural inference cost is **not publicly disclosed**. Database search is separate: comparing against every stored vector takes more work as either the catalog or vector width grows. An approximate index trades search completeness, memory, and response time.

**Optional math:** For $`N`$ indexed vectors, each with $`m`$ numbers, brute-force search costs $`O(Nm)`$ work per query and stores $`O(Nm)`$ numbers. Doubling the catalog roughly doubles this comparison work, with width fixed. This is the database's complexity, not Titan's hidden neural complexity.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** AWS's **Titan Multimodal Embeddings workshop** builds visual product search over a subset of **Amazon Berkeley Objects (ABO)**. Product images and English metadata are indexed. The notebook query **"A bed"** becomes a Titan vector. OpenSearch Serverless finds nearby indexed vectors, then displays product images and metadata for inspection. A shared text/image space helps when wording or visual resemblance differs from exact catalog keywords.

The [workshop](https://github.com/aws-samples/titan-multimodal-embeddings-workshop/tree/main/3-content-search) and [query notebook](https://github.com/aws-samples/titan-multimodal-embeddings-workshop/blob/main/3-content-search/3b-content-search.ipynb) show the workflow and result-display code. This is a **reference implementation, not Amazon retail production evidence**. No Recall@k score, sales uplift, or other public KPI was reported. Recall@k would measure the fraction of all relevant items found in the top k retrieved results. The workshop was not rerun for this book.

**Notable vendor implementations/libraries:** Amazon Bedrock's `InvokeModel`, Boto3, and the workshop's OpenSearch integration connect the pieces. Bedrock hosts several vendors' models. It is not Titan's architecture.

**Architecture diagram description:** Functional view: `image and/or text -> proprietary Titan embedding model -> vector -> external index -> ranked objects`. Internal image/text encoders, how their information is combined, and vector-output layers are **not publicly disclosed** here.

**Activation functions used and why:** **Not publicly disclosed.** A returned vector or a similarity calculation does not show whether the network uses ReLU, GELU, SwiGLU, or another activation.

**Loss function(s):** The exact pretraining loss, choice of contrasting negative examples, temperature setting, and extra objectives are **not publicly disclosed**. Temperature here would control score scaling. Contrastive learning, which pulls related examples closer and separates others, is one possible general approach. It is not a verified Titan recipe.

**Optimization algorithm(s):** The original optimizer and pretraining learning-rate schedule are **not publicly disclosed**. **Optional math:** Multimodal G1 customization exposes a default learning rate of **$`5\times10^{-5}`$**, controlling customer fine-tuning update size. That setting does not identify the original optimizer or schedule.

**Regularization techniques:** Dropout, weight decay, normalization, and model-specific controls are **not publicly disclosed**. Holding out a validation set during customization helps choose a model. It does not reveal hidden regularization choices.

**Backpropagation considerations:** Hosted embedding inference exposes no gradients. Documented image-caption fine-tuning sends paired data to a managed training process. The outside nearest-neighbor index is not automatically trained end-to-end with the embedding model.

**Parameter count / scaling behavior:** Parameter counts are **not publicly disclosed**. Multimodal G1 offers vector widths **1,024, 384, or 256**; Text V2 offers **1,024, 512, or 256**. These are **embedding widths, not parameter totals**. Text V2's documented maximum input is 8,192 tokens. That is not Multimodal G1's text limit.

**Training paradigm:** Foundation pretraining uses an incompletely disclosed mixture of learning signals. Supported multimodal customization explicitly uses image-caption pairs. No unverified RLHF or distillation stage is assigned to these embedding checkpoints.

**Hardware/parallelism considerations:** AWS manages the serving and training infrastructure. Original accelerator types/counts and parallelism are not publicly disclosed in these materials. Batch indexing and concurrent requests affect application throughput separately from the unknown neural design.

### 3.13.6 Amazon Nova

**In plain English:** The examined Nova models read text and, in some versions, images or video, then write answers. Reading a video is different from generating one; those are separate Nova models.

**Name:** Amazon Nova, represented by the **2024 understanding releases Micro, Lite, and Pro**. Canvas and Reel separately generate images and video.

**Category & sub-category:** Proprietary text/multimodal foundation models with instruction and preference training. The Nova brand does not identify one architecture used by every service.

**Originating paper/vendor/year:** Amazon/AWS's [December 2024 launch](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-frontier-intelligence-and-industry-leading-price-performance/) and [The Amazon Nova Family of Models: Technical Report and Model Card, 2025](https://arxiv.org/html/2506.12103v1). The launch article notes an April 2025 benchmark update. The report does not establish that these are the latest releases in 2026.

**Core mechanism:** The report describes Micro/Lite/Pro as Transformer-based. It names multilingual/multimodal pretraining, instruction SFT, reward modeling, and preference optimization. It does not provide a complete layer diagram, activation list, parameter count, or gradient-optimizer recipe.

**Inputs/outputs and typical data types:** Micro takes text only. Lite and Pro take text, images, and video and output text in the examined release. Canvas generates images; Reel generates video. A video-input path does not specify a video-generation design.

**Strengths and limitations:** Reading documents, images, and video, plus managed customization, can support specialized workflows. Models can still misread tiny text, give unsupported answers, or misunderstand event timing. Service access and advertised price/performance do not reveal reproducible internals or guarantee a customer's costs.

**Computational complexity / scalability notes:** Exact model complexity is **not publicly disclosed**. Media tokenization, output length, concurrency, and endpoint limits affect visible work. Tokens per second do not reliably reveal parameter count. The dense reference formula is not an established Nova implementation model.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Amazon's **DocVQA** evaluation asks questions about document images. An image and question enter Nova Pro; it returns a short answer phrase. The evaluator compares the answer with the reference using **Average Normalized Levenshtein Similarity (ANLS)**. ANLS measures string similarity using the edits needed to match strings, adjusted for length. Reading images directly can use layout and visual context that an OCR-then-text pipeline may handle differently. OCR means converting text in images into text characters. Document-level validation is still needed.

[Table 3](https://arxiv.org/html/2506.12103v1) reports **93.5 ANLS for Nova Pro** and **92.4 for Nova Lite** on the **DocVQA test set**, using the report's **zero-shot** setup: no task demonstrations. Appendix B.2.2 asks for a single word or phrase. ANLS rewards near-matching strings; it is **not 93.5% perfectly correct business documents**. Other vendors' rows may use different OCR or prompting procedures. No document-processing return on investment was established.

**Notable vendor implementations/libraries:** Amazon Bedrock, Converse/InvokeModel, and AWS SDKs provide access. A model identifier selects a hosted release. It does not expose the arrays containing its parameters.

**Architecture diagram description:** Published-level functional view: `text (Micro), or text/images/video (Lite/Pro) -> proprietary Transformer-based understanding model -> text`. Complete media encoders, how media are combined, layer layouts, and any routing are **not publicly disclosed**.

**Activation functions used and why:** **Not publicly disclosed** for these understanding models in the cited report. Titan, Llama, or generic decoder activations must not fill the gap.

**Loss function(s):** The report names instruction SFT, reward models learned from human preferences, DPO, and PPO. Exact pretraining/extra multimodal losses and the full weighting of stages are **not publicly disclosed**. The named alignment methods are real disclosures, but not a complete recipe for reproducing all losses.

**Optimization algorithm(s):** The underlying gradient optimizer and numerical learning-rate schedules are **not publicly disclosed**. PPO and DPO describe alignment procedures. They do not tell us whether an Adam/SGD implementation was used or how its learning rate changed.

**Regularization techniques:** The report describes data selection and safety-focused SFT/RLHF work. Dropout, normalization, weight-decay coefficients, and other internal settings are **not publicly disclosed**. Moderating requests or responses during use is a separate system control.

**Backpropagation considerations:** Hosted inference exposes no gradients. Original numerical precision, clipping, and recomputation details are not publicly disclosed. Bedrock offers managed customization and distillation. Those options do not prove that every released base model was distilled.

**Parameter count / scaling behavior:** Total and active parameter counts are **not publicly disclosed**. Launch context limits are **128K for Micro** and **300K for Lite/Pro**. Context length is not parameter count, and these interface facts do not automatically describe later releases.

**Training paradigm:** The report describes multilingual/multimodal pretraining, supervised instruction examples, and human-preference alignment. The launch offers Pro as a teacher for custom Micro/Lite distillation. That is distinct from disclosing their complete original pretraining recipe.

**Hardware/parallelism considerations:** Original hardware inventory and sharding are **not publicly disclosed** in the cited report. Amazon's own accelerator technology does not prove which hardware trained a checkpoint. Measure a versioned endpoint using the workload the application actually needs.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Anthropic Claude | Conversations, documents, and supported images | Offers long-context assistance and published safety research | Detailed internals are hidden; inserted-fact recall is a narrow test | Claude 3 Opus's NIAH inserted-fact test |
| Google / DeepMind Gemini | Mixed text, images, audio, and video | Handles long mixed-media context; distinguishes Pro and Flash publicly | MoE size/routing and much training detail remain hidden | Gemini 1.5 Pro's MTOB translation test |
| Cohere Command R | Text snippets, conversations, and tool descriptions | Trains explicit source citations and tool-request formats | Retrieved content and cited support still need checking | Cohere's documented penguin-answer demonstration |
| Baidu ERNIE | Chinese meaning tasks; later named assistant products | Original model hides whole phrases and entities | Original encoder details do not specify later assistants | ERNIE 1.0's Chinese XNLI relation test |
| Amazon Titan | Text/image vectors for finding related items | Offers shared-space image/text comparisons through a managed API | Embedding design and losses remain undisclosed | AWS's ABO catalog-search reference demo, not retail deployment |
| Amazon Nova | Documents, images, and video to be understood | Publishes input-type and alignment-stage distinctions | Counts, activations, and the detailed layer graph remain hidden | Nova Pro/Lite's DocVQA document-question test |

## 3.14 Open Code, Efficient, and Enterprise Families

These entries have public neural implementations and named weights. That does not make every dataset, training recipe, or license fully open. "Efficient" must say what it saves: training work, serving work, memory, or resources for a given task quality. A small model can be cheap to run after expensive pretraining. S4 and Mamba broaden this group beyond branded checkpoints. They are sequence architectures that use running states instead of attention; the chosen examples learn to generate text through self-supervision.

### 3.14.1 Microsoft Phi

**In plain English:** Phi-3-mini aims to fit useful text and code abilities into a small model. Its size can support local use, but small weights do not mean cheap training or complete knowledge.

**Name:** Phi family, represented by **Phi-3-mini-4k-instruct in the original April 2024 report** and its model-card revision before the June update. Phi-1/2 and later Phi variants use different recipes.

**Category & sub-category:** Causal foundation modeling with carefully selected data, followed by supervised and preference training. The reference is a compact dense assistant model.

**Originating paper/vendor/year:** Microsoft's Phi work began with the 2023 "Textbooks Are All You Need" line. Sources here are the [April 2024 Phi-3 technical report, v1](https://arxiv.org/html/2404.14219v1) and [pinned original model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/ff07dc01615f8113924aed013115ab2abd32115b/README.md). Pinning a revision keeps later token-budget and score updates from replacing the historical record.

**Core mechanism:** Train a small causal Transformer using heavily filtered web text and synthetic data organized like educational material. Then train it for useful responses. The emphasis is choosing knowledge and reasoning patterns that make good use of limited capacity. It is not a claim to have introduced a universally new attention algorithm.

**Inputs/outputs and typical data types:** Text, code, math prompts, and conversations with role markers produce generated text. This Mini checkpoint is text-only. Separate Phi vision and MoE releases must not inherit all its architecture details.

**Strengths and limitations:** Small weights support local/offline use and limited-memory applications. Careful data selection can improve ability per parameter, but does not guarantee coverage of every fact or language. Reasoning-test results, safety behavior, and instruction following can change between later-training revisions.

**Computational complexity / scalability notes:** Dense-decoder costs apply. Fewer weights reduce storage and memory traffic, but tokens are still generated sequentially and KV history grows. **Optional math:** $`p`$ denotes parameter count; a smaller $`p`$ lowers weight-related costs, not every cost. Quantization lowers numerical precision and may change accuracy. It is not a free copy of the BF16 benchmark system.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Microsoft's original **GSM8K** test gives a compact model grade-school math word problems. Phi-3-mini receives a problem with a chain-of-thought (CoT) prompting setup, which asks for written solution steps. It writes a solution and final answer. The evaluator extracts the answer and checks the test key. A local generator can help interpret varied wording; a symbolic calculator is preferable for guaranteed arithmetic once that wording has been parsed.

The [v1 report](https://arxiv.org/html/2404.14219v1) gives **82.5% under zero-shot CoT**, versus **61.1% for Phi-2** in the same table. Zero-shot means no worked task demonstrations were supplied. These are correct-answer rates for the adapted Phi-3-mini system, not an unaligned base checkpoint. The experiment is not an education deployment and does not measure student learning.

**Notable vendor implementations/libraries:** Microsoft Phi releases, Hugging Face Transformers, and compatible ONNX/local runtimes support the checkpoint. A hosted catalog or inference wrapper is a way to run Phi, not another Phi neural algorithm.

**Architecture diagram description:** The [pinned Mini configuration](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/ff07dc01615f8113924aed013115ab2abd32115b/config.json) specifies `tokens -> RMSNorm/RoPE attention and gated FFN residual blocks -> token head`. It has **32 layers, width 3,072**, and **32 query/KV heads**. The blocks rescale vectors, gather allowed context, and apply gated transformations with shortcuts.

**Activation functions used and why:** SiLU in the FFN gate smoothly controls transformed values. Attention and output probabilities use softmax. Phi-2's design and activations are not substituted for this Phi-3 checkpoint.

**Loss function(s):** Pretraining predicts next tokens; the card then documents SFT and DPO. Synthetic examples do not prove Mini used a teacher-logit KL loss, which would compare teacher and student probability distributions.

**Optimization algorithm(s):** A complete original pretraining optimizer and numerical learning-rate schedule are **not publicly disclosed in the cited release materials**. Later sample fine-tuning scripts are not evidence of the original recipe.

**Regularization techniques:** The pinned configuration sets attention, embedding, and residual dropout to **0.0** and uses RMSNorm. Filtering data and choosing targets in stages are central. Zero dropout does not mean there were no data-quality controls, nor does it reveal other unpublished coefficients.

**Backpropagation considerations:** Residual shortcuts and normalization support deep-model training. Original clipping and gradient-precision settings are not guessed where undisclosed. DPO updates response-model weights from preference pairs; it should not be relabeled as a documented online PPO loop.

**Parameter count / scaling behavior:** The original reference has **3.8B parameters**, **4K context**, and **3.3T training tokens**. The later June model-card update gives a different token budget. Neither budget describes every Phi release.

**Training paradigm:** Selected real and synthetic text supplies self-supervised pretraining. Supervised instructions and preference optimization follow. Synthetic-data training can overlap with distillation, but the terms do not mean the same thing.

**Hardware/parallelism considerations:** The pinned card reports **512 H100-80G GPUs** for training. Separately, the paper runs a four-bit Mini offline on an **iPhone 14 with A16 Bionic at over 12 tokens/second**. Four-bit storage is a quantized setup. This is a device-specific prototype measurement, not a guarantee of phone speed or production reliability.

### 3.14.2 NVIDIA Nemotron

**In plain English:** This Nemotron base model is a large text generator that can support further training. Separate instruction and reward models help generate and judge synthetic examples; they are not just different settings on Base.

**Name:** Nemotron family, represented by **Nemotron-4-340B-Base, June 2024**. Instruct and Reward are separate adaptations. Later models with "Nemotron" in their names need their own release evidence.

**Category & sub-category:** Dense causal base modeling, with related tools and models for synthetic-data creation and alignment.

**Originating paper/vendor/year:** NVIDIA's [Nemotron-4 340B Technical Report, 2024](https://arxiv.org/html/2406.11704v1), [Base model card](https://huggingface.co/nvidia/Nemotron-4-340B-Base), and public NeMo implementation describe this reference.

**Core mechanism:** A large dense causal Transformer predicts text and supplies a starting point for adaptation. In the wider release, Instruct generates responses and Reward judges or filters responses. Reward has its own trained scoring behavior. It is not Base with a different temperature setting. Generating synthetic data is a workflow, not one unique neural architecture.

**Inputs/outputs and typical data types:** Base turns multilingual text/code into continuation probabilities and text. Instruct takes conversation-style prompts. Reward scores candidate answers using its separately trained output head and objectives.

**Strengths and limitations:** Public large-model weights and matching infrastructure support teacher-model and synthetic-data studies. The dense 340B model is expensive to run. Synthetic text can repeat or amplify teacher errors, so quality and variety need checks. An NVIDIA serving product does not identify every hosted model's architecture.

**Computational complexity / scalability notes:** Dense execution uses all applicable layer weights for each token, rather than selecting an MoE subset. GQA reduces KV storage. The large vocabulary and 96-layer stack add costs beyond attention alone. Calling all the work "quadratic attention" misses these costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** NVIDIA uses **MMLU** multiple-choice questions to test Base's knowledge across subjects. Five demonstrations, a question, and its choices enter Nemotron-4-340B-Base. It selects an answer and is checked against the test key. Such testing can help screen a possible teacher for synthetic knowledge-task data. A high score does not guarantee high-quality synthetic examples.

The [Base card](https://huggingface.co/nvidia/Nemotron-4-340B-Base) reports **81.1% five-shot MMLU**. The report's table places it below the listed **84.2% Qwen-2 72B** comparison. These are correct-answer rates, showing that size alone does not determine task ranking. This benchmark is neither a customer deployment nor a measured return on synthetic-data investment.

**Notable vendor implementations/libraries:** NVIDIA NeMo, NeMo-Aligner, and compatible inference integrations support the release. NIM packages and serves models; it is not the Nemotron-4 architecture itself.

**Architecture diagram description:** `tokens -> causal RoPE/GQA Transformer blocks with LayerNorm and squared-ReLU FFNs -> untied vocabulary head`. "Untied" means the output weights are not shared with input embeddings. The [NeMo `Nemotron4Config340B`](https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/llm/gpt/model/nemotron.py) specifies the layer dimensions and normalization.

**Activation functions used and why:** The published FFN uses **squared ReLU**, not SwiGLU. It sets negative inputs to zero and squares positive ones. Large positive inputs also produce large gradients, so numerical care matters. **Optional math:** $`f(x)=\max(0,x)^2`$, where $`x`$ is an input value and $`f(x)`$ is its transformed output.

**Loss function(s):** Base uses causal language-model loss, including during continued pretraining. Instruct and Reward add alignment or rating objectives described separately. Reward's scoring loss is not Base's next-token loss.

**Optimization algorithm(s):** The 340B report describes an optimizer distributed across devices and a changed learning-rate decay during continued training. It does not give a complete named gradient optimizer and numerical schedule. These details are **not fully publicly disclosed there**. Neither the 15B report nor NeMo defaults can silently fill the gap.

**Regularization techniques:** The report and implementation specify **zero dropout**, separate input/output embeddings, and attention/FFN linear transformations without bias offsets. The public implementation uses LayerNorm with learned parameters. Changing data proportions during continued training is a separate intervention. Unpublished weight-decay or clipping settings are not guessed.

**Backpropagation considerations:** Large dense gradients and optimizer records need to be divided across devices. The report explicitly spreads optimizer records among data-parallel model copies. The 8T-to-1T data transition changes target selection and sampling emphasis while keeping the same base prediction loss.

**Parameter count / scaling behavior:** The nominal size is **340B**, with **96 layers**, width **18,432**, **96 query heads and eight KV heads**, and **4,096 context**. Training uses **8T initial tokens plus 1T continued-training tokens**. Other Nemotron or Llama-Nemotron releases can have different dimensions.

**Training paradigm:** The main learning signal is autoregressive self-supervision. Continued training includes some question-answer and alignment-style examples. Separate Instruct/Reward stages add SFT and preference-related abilities. The label Base does not prove that all instruction-like text was absent.

**Hardware/parallelism considerations:** The report increases data parallelism while using **eight-way tensor and twelve-way pipeline parallelism**, reaching **6,144 H100 GPUs**. Separately, it targets eight-H100 inference in **FP8**, an eight-bit floating-point format. The card's BF16 serving examples require more memory. Training layout, weight precision, and inference requirements are different facts.

### 3.14.3 IBM Granite

**In plain English:** This Granite model offers locally adaptable text and code generation for enterprise work. Its public weights and documentation help inspection, but an application still needs its own safety and task checks.

**Name:** IBM Granite, represented by **Granite-3.0-8B-Base, released October 21, 2024**. Instruct, code-specialized, vision, and MoE versions do not all share this exact design or recipe.

**Category & sub-category:** Dense causal foundation pretraining with enterprise-focused release documentation. Instruction models separately receive supervised and preference training.

**Originating paper/vendor/year:** IBM's Granite Team published the [Granite 3.0 Language Models technical report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) and [8B Base model card, 2024](https://huggingface.co/ibm-granite/granite-3.0-8b-base).

**Core mechanism:** A dense Transformer with GQA, RoPE, and SwiGLU learns in two phases from selected data. Maximal update parameterization sets scaling rules for transferring useful training settings between model sizes. It does not replace next-token prediction with another supervision type. **Technical detail (optional):** This parameterization is written $`\mu`$P, pronounced "mu-P"; the Greek letter names the method. Granite 3.0 also includes MoE models, which this dense 8B configuration does not describe.

**Inputs/outputs and typical data types:** Multilingual prose, code, and structured information written as text produce continuations. Base does not inherit the instruction release's full conversation behavior or safety alignment. Enterprise software may separately add retrieval, access controls, and specialist models.

**Strengths and limitations:** Public weights, the Apache 2.0 release, and relatively detailed documentation support local adaptation and oversight. They do not prove universal benchmark leadership, full visibility of training data, or suitability for an untested business workflow.

**Computational complexity / scalability notes:** Dense Transformer costs still apply. GQA saves some stored attention state, while shared input/output embeddings save parameters. Mu-P's multipliers rescale numbers; they do not reduce how many token pairs attention compares. Copying layer counts without the multipliers may give different behavior. **Technical detail (optional):** $`\mu`$P is the report's name for these parameterization rules.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** IBM's **HumanEval** test asks for Python functions from specifications. A held-out function signature and docstring enter Granite-3.0-8B-Base. It writes a candidate, which is checked with unit tests. Pass@1 assesses success for one candidate under the test procedure. This helps evaluate code ability before adaptation. A compiler checks syntax/types but does not itself write the requested implementation from prose.

[Table 8 of the report](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) reports **52.44% pass@1**, compared with **31.71% for its Llama-3.1-8B Base baseline**. The model card independently repeats 52.44. The table does not disclose every code-sampling setting. It is therefore not a complete reproduction recipe or a fair cross-paper ranking. No watsonx customer productivity KPI follows from it.

**Notable vendor implementations/libraries:** IBM's Granite repositories, Hugging Face Transformers, and IBM deployment/customization integrations support these models. watsonx is a platform, not the architecture of every model it uses.

**Architecture diagram description:** `shared token embeddings -> 40 dense RMSNorm/RoPE-GQA/SwiGLU residual blocks -> shared token head`. Release-specific multipliers rescale embeddings, attention, residuals, and output scores, or logits. The [published configuration](https://huggingface.co/ibm-granite/granite-3.0-8b-base/blob/main/config.json) matters for reproducing those exact calculations.

**Activation functions used and why:** SwiGLU uses a SiLU gate to control transformed FFN values. Softmax supplies attention weights and token probabilities. Use the released configuration instead of assuming an unchanged Llama block.

**Loss function(s):** Base pretraining uses causal token cross-entropy. The report separately describes staged, or curriculum, SFT and preference/RL methods for Instruct. These extra alignment losses must not be assigned to the Base benchmark.

**Optimization algorithm(s):** The report uses **AdamW**, weight decay 0.1, and a **Power scheduler** with **2,500 warmup iterations**. After warmup, the rate follows a slow power-law phase and then falls faster near the end.

**Optional math:** AdamW's gradient and squared-gradient averaging settings are $`(0.9,0.95)`$. Scheduler settings are $`a=4`$, $`b=-0.51`$, and an upper bound 0.02. Here, $`a`$ and $`b`$ name the report's power-schedule settings. They must be read with $`\mu`$P's layer-wise scaling: different layers receive appropriately rescaled updates. The upper bound is not a universal unscaled AdamW learning rate.

**Regularization techniques:** The recipe includes weight decay 0.1, RMSNorm, selected data, and **0.1 attention dropout** in the released configuration. Mu-P and residual/logit multipliers help numerical stability and transfer of settings between sizes. They do not prove the model cannot memorize training material.

**Backpropagation considerations:** Reproducing weight updates requires the stated multipliers and learning-rate scaling. The report distributes optimizer records and uses optimized attention/normalization kernels. Changing precision or scaling needs validation, even if the layer diagram looks unchanged.

**Parameter count / scaling behavior:** The card specifies **8.1B parameters**, **40 layers**, width **4,096**, **32 query/eight KV heads**, FFN width **12,800**, and **4,096 context**. It sees **10T tokens in stage 1 plus 2T in stage 2**. The family's MoE total/active counts are separate.

**Training paradigm:** Base training is mostly self-supervised. The second phase includes selected multilingual and instruction data. The separate Instruct release receives more SFT/alignment. Base is explicitly not a fully safety-aligned assistant.

**Hardware/parallelism considerations:** IBM reports **Blue Vela H100 infrastructure**, using tensor, pipeline, and data parallelism. Local serving can quantize or split weights. Its budget must include KV state beyond the approximately 16.2-GB BF16 weight-only calculation.

### 3.14.4 StarCoder

**In plain English:** StarCoder writes code continuations and fills gaps using code before and after them. It is designed for code completion, but generated code still needs tests and review.

**Name:** StarCoder and StarCoderBase, the **2023 15.5B BigCode releases**. StarCoder2 is a later family with a separate recipe.

**Category & sub-category:** Self-supervised code language modeling with fill-in-the-middle (FIM). The named StarCoder checkpoint also receives Python-focused continued training.

**Originating paper/vendor/year:** Raymond Li and the BigCode collaboration published [StarCoder: May the Source Be with You!, 2023](https://arxiv.org/html/2305.06161v1), with an [official model card](https://huggingface.co/bigcode/starcoder). Hugging Face and ServiceNow organized the project with many contributors.

**Core mechanism:** A causal Transformer predicts code tokens. For some training sequences, FIM rearranges code as prefix, suffix, then missing middle, separated by special markers. The model thus sees both sides before writing the gap. It remains a causal decoder, not a replacement bidirectional encoder.

**Inputs/outputs and typical data types:** Source code, function signatures/docstrings, or marked FIM prefix/suffix text produce completions or inserted code. Giving an instruction prompt is not the same as training an assistant.

**Strengths and limitations:** Code-focused data and FIM fit editors where later code already exists. Output can still be wrong, insecure, or copied from training examples. Filtering for permissive sources and processing opt-outs do not remove later attribution or licensing obligations.

**Computational complexity / scalability notes:** Dense causal attention and FFN costs remain. Multi-query attention (MQA) shares a single KV head among query heads, saving cache memory. It does not make FFNs sparse or full-attention arithmetic linear in sequence length. FIM reorders training text rather than adding another inference pass.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** BigCode's **HumanEval** test supplies a Python signature/docstring. StarCoder samples a function body, which is run against unit tests. Pass@1 estimates how often one sampled completion succeeds. Code-specific training fits executable completion better than general conversation when that is the goal.

[Table 12 and section 6.1.1](https://arxiv.org/html/2305.06161v1) report **33.6% pass@1 for StarCoder**, versus **30.4% for StarCoderBase**. The open-model procedure uses **200 samples per problem and temperature 0.2 for pass@1 estimation**. Temperature controls how strongly sampling favors higher-scoring next tokens; the many samples estimate one-candidate success. The **40.8% StarCoder-Prompted** row adds a separate prompting change, so it is not the unprompted score. These test results do not measure developer-time savings or production deployment.

**Notable vendor implementations/libraries:** BigCode's training/evaluation repositories, Hugging Face Transformers' GPTBigCode implementation, and compatible code-completion runtimes support the models. Full authorship belongs to the collaboration, not either organizing company alone.

**Architecture diagram description:** `code/FIM tokens + learned positions -> causal GPT-2-style pre-normalized blocks with MQA and dense FFNs -> token head`. The suffix appears before the target middle in the causal sequence. This ordering lets the model use that suffix without reading future tokens in its own input order.

**Activation functions used and why:** The [public GPTBigCode configuration](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py) uses a tanh approximation to GELU for smooth FFN transformations. Tanh is the smooth curve used to approximate the activation. Attention and output probabilities use softmax. Later StarCoder architectures require separate checks.

**Loss function(s):** Training uses next-token cross-entropy for ordinary and FIM-reordered code. Extra Python training keeps the language-model objective. "Fine-tuned" here does **not** automatically mean supervised instruction tuning.

**Optimization algorithm(s):** StarCoderBase uses Adam and cosine rate decay after **2,000 warmup iterations**. Python adaptation has a separate schedule with **1,000 warmup iterations**. **Optional math:** Base uses gradient/squared-gradient averaging settings $`(0.9,0.95)`$ and epsilon $`10^{-8}`$, a numerical-stability setting. Its rate falls from **$`3\times10^{-4}`$ to $`3\times10^{-5}`$**. Python adaptation instead falls from **$`5\times10^{-5}`$ to $`5\times10^{-6}`$**.

**Regularization techniques:** Training uses weight decay 0.1, LayerNorm/residual shortcuts, and randomly chosen FIM transformations. Filtering and removing duplicates reduce particular risks but do not prove output code is original. Generic GPTBigCode dropout defaults are not treated as original-run disclosures.

**Backpropagation considerations:** FIM markers and segment order must match the training conventions, or the loss teaches a different task. The paper reports BF16 training with FP32 gradient reduction. Combining gradients in higher 32-bit precision improved stability at a throughput cost.

**Parameter count / scaling behavior:** Both named 2023 models have about **15.5B parameters**, **40 layers**, width **6,144**, **48 query heads**, FFN width **24,576**, and **8,192 context**. StarCoderBase sees **1T tokens**. StarCoder adds **35B Python tokens**, rather than changing to a differently sized network.

**Training paradigm:** Both code pretraining and Python continued training are self-supervised. A technical-assistant prompt changes the context at inference, not the weights. Distillation, RLHF, and instruction alignment are not established for these base-code releases.

**Hardware/parallelism considerations:** The paper uses **512 A100 80-GB GPUs**, with four-way tensor, four-way pipeline, and 32-way data parallelism. Local quantized serving is a different setup. It needs its own correctness, response-time, and memory checks.

### 3.14.5 S4

**In plain English:** S4 carries a fixed-size running state as it reads a sequence. It can predict the next token without keeping an attention record of every earlier token, though its compressed state can lose details.

**Name:** S4 means Structured State-Space Sequence Model. The detailed example is the **249M-parameter autoregressive WikiText-103 system in version 3 of the original paper**. It is not a Speech Commands classifier, S4D, or the later SaShiMi architecture.

**Category & sub-category:** Self-supervised generative sequence modeling. A state-space operator updates a running collection of numbers. S4's operator is linear and time-invariant (LTI): it adds weighted values, using the same weights at each position. Those internal weights do not change with the current token. The larger language network around it is nonlinear. S4 is a design family, not a hosted assistant.

**Originating paper/vendor/year:** Albert Gu, Karan Goel, and Christopher Re at Stanford published [Efficiently Modeling Long Sequences with Structured State Spaces, 2021 preprint / ICLR 2022](https://arxiv.org/html/2111.00396v3). The authors' [state-spaces/s4 repository](https://github.com/state-spaces/s4) provides code, experiment configurations, and generation examples.

**Core mechanism:** At each step, the operator combines the previous state with the new input. It reads an output from the updated state, with an additional direct path from the input. The state is a compressed running record, not a searchable copy of all earlier text. The same coefficients apply at every position. For training, S4 can use convolution: combining earlier inputs with the same weight pattern at each position. For generation, it can use the matching step-by-step state updates. **There is no query-key attention matrix. The state-space operator is linear and time-invariant; the surrounding network is nonlinear.**

**Optional math:** The starting continuous-time system is $`\dot{s}=As+bu`$, with readout $`y=cs+Du`$. Here, $`s`$ is the state, $`u`$ the input, and $`y`$ the output. The dot means the rate at which the state changes. $`A`$ transforms the state, $`b`$ writes input into it, $`c`$ reads from it, and $`D`$ controls the direct input path. Converting to discrete steps gives $`s_t=\bar A s_{t-1}+\bar b u_t`$. The index $`t`$ marks a step; bars mark the converted coefficients. The convolution kernel is $`K_j=c\bar A^j\bar b`$: $`K_j`$ is the weight for a lag of $`j`$ steps, and the power applies the state transition repeatedly.

**Technical detail (optional):** S4 uses normal-plus-low-rank matrix structure related to HiPPO. It can rewrite the state matrix in a well-conditioned diagonal-plus-low-rank form: separate diagonal entries plus a compact correction. "Well-conditioned" means avoiding excessive sensitivity to small numerical errors. This structure enables efficient Cauchy-kernel computations for the sequence weights; it does not turn the operator into attention.

**Inputs/outputs and typical data types:** The reference takes text tokens through adaptive embeddings and outputs next-token probabilities or continuations. Adaptive embeddings allocate representation capacity across vocabulary groups. Other S4 models handle numerical waveforms, flattened images, or sensor sequences. Their output heads and learning signals differ.

**Strengths and limitations:** Structured starting weights and a running state support long dependencies and streaming generation without a growing KV cache. However, an LTI operator cannot change its sequence-mixing coefficients to suit the current token. A compressed state can lose details that full attention could access directly. Nonlinear output gates do not make S4's internal operator token-selective in Mamba's sense.

**Computational complexity / scalability notes:** Once the kernel is built, fast Fourier transform (FFT) convolution processes a long sequence in parallel. Its cost grows somewhat faster than the sequence length, with model dimensions fixed. Building the kernel adds separate work. During recurrent generation, the state and update work need not grow with the length of the already-read text.

**Optional math:** Let $`B`$ be batch size, $`d`$ channel width, $`T`$ sequence length, and $`N`$ state order, the number of state values per channel. FFT convolution costs about $`O(BdT\log T)`$ per S4 layer, plus $`O(BTd^2)`$ for channel/FFN mixing. The paper derives fast structured Cauchy methods for kernel construction; a naive evaluation can cost $`O(dNT)`$. With fixed low rank, recurrent updates need $`O(dN)`$ state and work per decoding token, plus channel mixing and vocabulary scoring. Constant cost in earlier sequence length does not mean constant cost in width, state order, or model size.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Stanford's S4 study predicts next words in **WikiText-103** Wikipedia text. A held-out token prefix enters adaptive embeddings and the recurrent/convolutional S4 network. The model assigns probabilities to next tokens. The evaluator checks the probability assigned to the actual next token and combines these into test perplexity. Perplexity measures prediction surprise; lower is better, not a percentage of correct answers. This tests attention-free generation, with running-state serving as its attraction, not an assumed win on every quality measure.

[Figure 8 and Appendix D.3.2](https://arxiv.org/html/2111.00396v3) report **20.95 test perplexity for S4, 249M parameters**, versus **20.51 for the 247M Transformer baseline**. The stated evaluation uses a sliding, non-overlapping-window protocol, not scoring variants that supply extra context. The comparison is the cited adaptive-input Transformer setting, not every possible Transformer. These perplexities establish no production latency or business KPI.

**Notable vendor implementations/libraries:** The authors' PyTorch S4 repository provides custom CUDA or PyKeOps support for structured kernels. It is not an attention API. A reusable S4 code module is not itself the trained WikiText checkpoint.

**Architecture diagram description:** `tokens -> adaptive embeddings -> [pre-LayerNorm/residual S4 -> pre-LayerNorm/residual S4 -> position-wise FFN] x 16 macroblocks -> tied adaptive softmax`. Each macroblock is a larger group with **two S4 layers**, not one. The [public block configuration](https://github.com/state-spaces/s4/blob/main/configs/model/layer/s4s4ff.yaml) confirms this; there are not just 16 total SSM layers. The output shares weights with adaptive embeddings.

**Activation functions used and why:** The state recurrence itself is linear. The [S4 layer configuration](https://github.com/state-spaces/s4/blob/main/configs/model/layer/s4.yaml) uses GELU and a final GLU gate; the language-model FFN also uses GELU. A GLU multiplies one stream of values by gate values that control what passes through. These add nonlinear transformations around the sequence operator. Adaptive softmax converts grouped vocabulary scores to probabilities, not attention weights.

**Loss function(s):** The model penalizes low probability for the next token through adaptive softmax. The text supplies its own targets. Perplexity is the exponential of the average token log-loss: a different way to express prediction error, not classification accuracy.

**Optimization algorithm(s):** Appendix D.3.2 specifies **AdamW and one cosine cycle with an 800,000-update maximum**. The [published WikiText reproduction configuration](https://github.com/state-spaces/s4/blob/main/configs/experiment/lm/s4-wt103.yaml) gives **1,000 warmup updates**. **Optional math:** The learning rate is **$`5\times10^{-4}`$**, which controls weight-update size. This recipe describes that language experiment, not every S4 classifier.

**Regularization techniques:** The language run uses dropout **0.25**, pre-LayerNorm, and ordinary-weight decay **0.1**. The repository puts sensitive state-space dynamics into separate optimizer groups, including zero weight decay. Applying the same decay rule to every parameter would ignore this distinction.

**Backpropagation considerations:** Convolutional training sends gradients through kernel generation and FFT operations without a token-by-token recurrent training loop. Stable state representations, conversion to discrete steps, and complex-number arithmetic matter. This language run uses **no gradient clipping**. It must not inherit Mamba's clipping setting.

**Parameter count / scaling behavior:** The reference has **249M parameters**, width **1,024**, and **16 macroblocks**. The public S4 layer default has state order **64**. State order is not total model size. **Technical detail (optional):** The total parameter count $`p`$ includes embeddings, adaptive output layers, channel mixing, and all S4 layers, not just state dynamics.

**Training paradigm:** The chosen example is **self-supervised causal language modeling**. The original paper also has Long Range Arena (LRA) and Speech Commands classifiers trained with labels; see [supervised neural sequence learning](02-supervised-neural.md). Bidirectional readers or classifiers that pool a sequence into one result are not this causal generator.

**Hardware/parallelism considerations:** The language experiment uses **eight A100 GPUs**, batch size **one per GPU**, and **8,192-token context**. FFT/Cauchy kernels support parallel training; recurrent updates support stateful generation. Reported hardware speedups depend on exact batching and memory procedures. They are not universal S4 guarantees.

### 3.14.6 Mamba

**In plain English:** Mamba keeps a running state, but changes how it updates that state based on the current token. It can choose what to retain, write, and read without using attention.

**Name:** Mamba, the selective state-space architecture, represented by the **original `state-spaces/mamba-2.8b` Pile base checkpoint**. This is Mamba-1, not Mamba-2, Mamba-3, a SlimPajama retraining, or an attention-hybrid version.

**Category & sub-category:** Self-supervised autoregressive foundation modeling. Its recurrent state updates depend on the input. Parallel scan calculations help run those updates efficiently on accelerators.

**Originating paper/vendor/year:** Albert Gu and Tri Dao published [Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023 preprint, v2](https://arxiv.org/html/2312.00752v2). Sources also include their [versioned v1.0.1 release documentation](https://github.com/state-spaces/mamba/blob/v1.0.1/README.md) and public checkpoint configuration.

**Core mechanism:** Mamba reads a token and selects how strongly to retain state, write input, and read an output. Unlike S4's fixed kernel, these settings change with the input sequence. The state remains a compressed running record, not a store of all past tokens. An associative scan groups update calculations so hardware can combine them in parallel. It gives the same state updates as the step-by-step rule. One fixed FFT convolution cannot represent these changing coefficients. **This is not softmax attention. The original all-Mamba model contains no attention layers.**

**Optional math:** Mamba selects input-dependent $`B_t`$, $`C_t`$, and positive step size $`\Delta_t`$. Its conceptual update is $`s_t=\bar A_t s_{t-1}+\bar B_t u_t`$, with output $`y_t=C_t s_t+Du_t`$. Here, $`t`$ is the token step, $`u_t`$ the input, $`s_t`$ the state, and $`y_t`$ the output. $`B_t`$ controls writing, $`C_t`$ reading, and $`D`$ a direct input path. Bars mark coefficients converted to discrete updates. A learned diagonal matrix $`A`$ and selected step size $`\Delta_t`$ determine $`\bar A_t`$; the step size also affects the converted input update $`\bar B_t`$. Selection changes retention, writing, and readout rather than building query-key comparisons.

**Inputs/outputs and typical data types:** This checkpoint turns text tokens into vocabulary probabilities and continuations. The paper trains other models for audio and DNA. Their tokenization, state settings, sequence lengths, and heads must not be copied into the 2.8B text model's description.

**Strengths and limitations:** Input-dependent updates address a limitation of fixed LTI operators while keeping a fixed-size inference state. But that state cannot perfectly archive all earlier tokens. Million-position DNA experiments do not prove perfect million-token recall for this language checkpoint. Hybrids that add attention can have different cache behavior.

**Computational complexity / scalability notes:** With other dimensions fixed, reading twice as many tokens roughly doubles this block's sequence work. This does not make wide models free. The inference state does not grow with earlier sequence length like an attention KV cache. A naive implementation that stores every intermediate state can still need substantial memory.

**Optional math:** With fixed expansion factor, batch size $`B`$, sequence length $`T`$, width $`d`$, state order $`N`$, and convolution width $`w`$, per-block work is about $`O(BT(d^2+dN+dw))`$, excluding vocabulary scoring. State order counts state values per channel; convolution width counts nearby positions used by the short convolution. Recurrent decoding keeps $`O(Bd(N+w))`$ state per block, rather than a KV cache growing with $`T`$. Parallel scan and backward recomputation reduce movement between hardware memory and computation units.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Gu and Dao test the original Pile-trained models on **LAMBADA** final-word prediction. A narrative prefix updates Mamba's state token by token. The model predicts the last word from its next-token/word probabilities, then receives an exact-match decision. This asks whether an attention-free generator retains enough context for the missing word. A full-attention decoder is the natural comparison.

[The v2 paper's zero-shot Table 1](https://arxiv.org/html/2312.00752v2) reports **69.2% LAMBADA accuracy and 4.23 perplexity for Mamba-2.8B**, versus **64.7% and 5.04 for Pythia-2.8B**. Accuracy measures correct final words; lower perplexity means less prediction surprise. No task demonstrations are supplied in zero-shot evaluation. Both models use the same Pile training budget and tokenizer in this comparison, with the EleutherAI evaluation harness. The separate **6.22** for Mamba is **Pile validation perplexity**, not LAMBADA perplexity or a test score. No customer deployment or business KPI was claimed.

**Notable vendor implementations/libraries:** The authors provide `mamba-ssm`, CUDA/Triton selective-scan code, and Hugging Face-hosted weights. A Mamba block is a reusable calculation; a checkpoint supplies learned weights; a serving wrapper exposes requests. They are three different objects.

**Architecture diagram description:** `embedding -> repeated RMSNorm/residual blocks: linear expansion -> {causal depthwise convolution + SiLU -> selective SSM scan} multiplied by {SiLU gate} -> output projection -> final norm -> vocabulary head`. The short convolution processes nearby positions separately within channels. Its state-update branch is multiplied by a gate, then projected back. The [original block implementation](https://github.com/state-spaces/mamba/blob/v1.0.1/mamba_ssm/modules/mamba_simple.py) has no query-key product or attention-softmax stage.

**Activation functions used and why:** SiLU/Swish transforms the convolution branch and output gate. Softplus makes the selected step size positive. Vocabulary softmax still supplies token probabilities even though **attention softmax is absent**. **Optional math:** The step size is $`\Delta_t`$, for step $`t`$. The real diagonal dynamics use $`A=-\exp(A_{\log})`$, where $`A_{\log}`$ holds learned log-magnitudes and exponentiation is applied to each entry. Negating the exponential gives negative diagonal values. These choices control nonlinear selection and decay.

**Loss function(s):** The Pile base model uses next-token cross-entropy. It learns selective updates through that prediction error, not through a separate reward or an attention-alignment target.

**Optimization algorithm(s):** The paper's improved recipe uses **AdamW**, linear warmup, and cosine decay. Peak rates follow its size-dependent, five-times-GPT-3 rule. **Optional math:** Gradient and squared-gradient averaging settings are **$`(0.9,0.95)`$**, and decay ends at **$`10^{-5}`$**. The cited materials do not list every 2.8B warmup override. No universal numerical warmup duration is invented.

**Regularization techniques:** The selected checkpoint uses reported weight decay **0.1**, **no dropout**, and RMSNorm. The implementation exempts the learned dynamics log-values and direct skip parameter from weight decay. **Technical detail (optional):** These exempt parameters are $`A_{\log}`$ and $`D`$, respectively. Keep the specialized step-size initialization: a generic rule that zeros all biases can overwrite it.

**Backpropagation considerations:** Gradients pass through input-dependent selectors and the scan. Reported gradient clipping is **1.0**. Fused kernels recompute intermediate states during the backward pass to avoid large memory transfers. The [checkpoint configuration](https://huggingface.co/state-spaces/mamba-2.8b/blob/main/config.json) keeps residuals in FP32, a 32-bit floating-point format. Reset recurrent state between independent sequences to prevent unintended sharing of context.

**Parameter count / scaling behavior:** The model is nominally **2.8B parameters**, width **2,560**, with **64 actual Mamba blocks**. The historical README's 32 Transformer-equivalent layers require doubling to count Mamba blocks. Original module defaults are **state order 16, convolution width four, expansion two**. These are not Mamba-2 defaults or attention-head counts.

**Training paradigm:** The reference uses **self-supervised Pile pretraining for 300B tokens at sequence length 2,048**, with no instruction or preference tuning. Separately, the paper fine-tunes DNA models to classify labeled great-ape species. That supervised variant is linked to [supervised neural learning](02-supervised-neural.md), not assigned to every Mamba checkpoint.

**Hardware/parallelism considerations:** Efficient accelerator training relies on fused convolution/scan kernels and parallel prefix calculations. Recurrent decoding avoids a growing attention cache. Output projection and moving weights from memory still take work. Plain Python scans, other batch sizes, and different devices cannot inherit the paper's GPU throughput claims.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Microsoft Phi | Text/code tasks on small local systems | Carefully selects data for useful abilities with small serving weights | Small serving size does not mean cheap training or complete knowledge | Original Phi-3-mini's GSM8K math test |
| NVIDIA Nemotron | Large text/code workloads and teacher-model studies | Publishes separate Base, Instruct, and Reward roles with supporting tools | Dense 340B reference needs very large compute and memory budgets | Nemotron-4-340B-Base's MMLU question test |
| IBM Granite | Enterprise text/code needing local adaptation | Publishes dense-model settings, scaling rules, and training documentation | Base is not a complete safety-aligned business application | Granite-3.0-8B-Base's HumanEval code test |
| StarCoder | Code completion and gaps with a known suffix | Combines code-focused training with fill-in-the-middle | Code still needs tests, security review, and attribution checks | BigCode's HumanEval code test |
| S4 | Long text, audio, and other ordered sequences | Uses structured fixed update rules; trains by convolution and generates with state | Internal sequence-mixing coefficients do not change with the token | S4's WikiText-103 test perplexity comparison |
| Mamba | Text, audio, and DNA in separately trained models | Chooses state updates from the input without attention or a growing KV cache | State can lose details; speed depends on implementation | Original Mamba-2.8B's LAMBADA final-word test |

## Coverage and continuation manifest

- **Covered: 22 neural family/formulation entries.** Section **3.11.1-3.11.4** covers BERT, RoBERTa, T5, and BART. Section **3.12.1-3.12.6** covers GPT, LLaMA/Llama, Mistral dense, Qwen dense, DeepSeek LLM dense Base, and BLOOM. Section **3.13.1-3.13.6** covers Claude, Gemini, Command R, ERNIE, Titan, and Nova. Section **3.14.1-3.14.6** covers Phi, Nemotron, Granite, StarCoder, S4, and Mamba. S4 uses a structured fixed-rule LTI operator; Mamba selects updates from the input. Neither mechanism is renamed attention.
- **Evidence boundary:** Twenty worked examples are research benchmarks. Two are sourced vendor reference demonstrations: Command R grounding and the Titan ABO workshop. None is presented as a measured customer production deployment. Scores were checked against public reports/cards, not independently rerun. Unknown proprietary architectures, counts, activations, optimizers, and datasets remain explicit. Model-card and configuration revisions are identified when they differ from historical papers.
- **Previous foundation:** [Unsupervised neural and self-supervised representation methods, sections 3.6-3.10](05-unsupervised-neural.md) explains related neural mechanisms. [Unsupervised classical methods](04-unsupervised-classical.md) covers clustering and other ways to find structure without neural networks. The original supervised Transformer and labeled neural tasks belong in [supervised neural methods](02-supervised-neural.md). Learning with mixed labeled/unlabelled data is discussed in [semi-supervised learning](03-semi-supervised.md).
- **Next and cross-cutting volumes:** [MoE model families, section 3.15](07-moe-models.md) separates Mixtral, sparse DeepSeek, and other expert-model releases from this chapter's dense references. [The dedicated MoE deep dive](08-moe-deep-dive.md) explains choosing experts, their capacity, balancing their use, training, and system costs. [The comparative guide](09-comparative-guide.md), [glossary](10-glossary.md), and [reading guide](00-reading-guide.md) support model selection, term lookup, and interpretation of evidence.
- **Further depth not included in this edition:** This is not a complete release history through September 2026. It does not cover every regional, quantized, distilled, instruction, vision, speech, code, embedding, or reasoning derivative. It does not fully catalog S4D/S5/S4ND, SaShiMi, Mamba-2/-3, or attention-hybrid designs. Other exclusions are complete tokenizer specifications and corpus lists; all licenses and changing endpoint quotas; full mathematical derivations of contrastive and multimodal losses; and reproducible end-to-end pretraining scripts. Comprehensive tutorials on parameter-efficient fine-tuning (PEFT), distillation, preference optimization, and standalone reinforcement learning are also outside scope. So are independent production cost/latency checks and audits of benchmark material in training data. These limits do not imply that the historical models discussed here are current leaders.
