# 6. Glossary

Definitions emphasize how terms are used in this book. A glossary mention is
not an additional fully cataloged algorithm entry.

## 6.1 Learning, objectives, and statistical reasoning

| Term | Definition |
|---|---|
| Algorithm | A specified procedure for fitting, searching, optimizing, or computing. It is not identical to a trained parameter set. |
| Architecture | The arrangement of operations, layers, states, and connections in a model. Different training signals can use the same architecture. |
| Model / checkpoint | A parameterized predictor or generator; a checkpoint is a saved state of its learned parameters and sometimes training state. |
| Supervised learning | Learning from examples with externally supplied targets. Targets can be scalar, categorical, structured, weak, or noisy. |
| Semi-supervised learning | Learning that explicitly combines labeled and unlabeled examples for a task. Its benefit depends on assumptions linking the two. |
| Unsupervised learning | Modeling observations or discovering structure without the task's target annotations. |
| Self-supervised learning | Training with targets derived from the observations, such as withheld words, future tokens, or another augmented view. |
| Weak supervision | Learning from imperfect supervision such as rules, distant labels, noisy captions, or approximate metadata. |
| Transductive learning | Learning or predicting with access to the particular unlabeled instances of interest during fitting. This differs from generalizing to arbitrary future inputs. |
| Inductive learning | Learning a rule intended to apply to new, previously unseen examples. |
| Loss function | A function measuring mismatch between a prediction and its target or another training criterion. It is minimized directly or through an estimator. |
| Objective | The complete quantity optimized, often prediction loss plus regularization and auxiliary terms. |
| Empirical risk | Average loss on an observed dataset; it estimates but does not equal expected population risk. |
| Cross-entropy | For a one-hot target, the negative log probability assigned to the correct class/token; more generally, $`-\sum_i q_i\log p_i`$. |
| Mean squared error | Average squared numerical prediction error. Large residuals receive disproportionately large penalties. |
| Hinge loss | A margin-based surrogate, typically $`\max(0,1-yf(x))`$ for binary labels $`y\in\{-1,1\}`$. |
| Likelihood | A probability model's probability or density for observed data, viewed as a function of model parameters. A density can exceed one without being invalid. |
| Maximum likelihood | Parameter estimation by maximizing the likelihood, equivalently minimizing negative log-likelihood. |
| Prior / posterior | A distribution before conditioning on specified observations, and the updated distribution after conditioning. |
| Maximum a posteriori estimation | Estimating a parameter mode under the posterior; common regularizers can correspond to particular priors. |
| ELBO | Evidence lower bound: a variational objective balancing expected data log-likelihood and divergence from a prior; a lower bound on log evidence under stated assumptions. |
| KL divergence | $`D_{\mathrm{KL}}(q\Vert p)=\mathbb{E}_q[\log q-\log p]`$. Nonnegative and generally asymmetric; not a metric distance. |
| Entropy | Expected information content of a distribution. Low predictive entropy means confidence, not necessarily correctness. |
| Regularization | Restrictions or penalties intended to control a model's effective flexibility or improve generalization. |
| Bias-variance tradeoff | The interaction between systematic approximation error and sensitivity to sampled training data; not a universal monotonic rule for every modern model. |
| Overfitting | Adapting to training-specific patterns that do not support the intended generalization. |
| Underfitting | Failing to capture useful structure because of inadequate flexibility, features, optimization, or data. |
| Inductive bias | Assumptions favoring some solutions over others, such as locality in a convolution or smoothness in a kernel. |
| Identifiability | Whether different parameter values imply distinguishable observable distributions. Good predictions do not guarantee unique latent factors. |
| Calibration | Agreement between predicted probabilities and empirical frequencies under a specified population and evaluation protocol. |
| Distribution shift | A difference between the data-generating conditions used for learning and those encountered later. |
| Out of distribution (OOD) | Outside a specified reference distribution; the boundary depends on what reference is chosen. |
| Causal effect | A change attributable to an intervention under a causal identification framework, not merely a predictive association. |
| Censoring | Partial observation of an event time, such as knowing only that failure had not occurred before monitoring ended. |

## 6.2 Optimization and neural computation

| Term | Definition |
|---|---|
| Gradient | The vector of partial derivatives of a scalar objective with respect to parameters or intermediate variables. |
| Gradient descent | Iteratively updating parameters opposite the gradient, for example $`\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}`$. |
| SGD | Stochastic gradient descent, which uses a sample or mini-batch estimate of the objective's gradient. |
| Momentum | Accumulating a moving direction of updates to smooth noisy gradients and traverse some geometries more effectively. |
| Adam | An adaptive optimizer using estimates of first and second gradient moments and bias correction. |
| AdamW | Adam with decoupled weight decay; not generally equivalent to adding an L2 penalty inside Adam's adaptive gradient update. |
| RMSProp | An adaptive optimizer that rescales updates using a moving estimate of squared gradients. |
| Adafactor | A memory-conscious adaptive optimizer using factored second-moment estimates for suitable parameter matrices. |
| LAMB | A layer-wise adaptive optimizer using a trust ratio, often studied for very large-batch training. |
| Learning rate | The scale of an optimizer's parameter updates. Its useful value depends on normalization, batch size, optimizer, and training stage. |
| Warmup | An initial period during which the learning rate rises toward its intended operating value. |
| Learning-rate schedule | A rule varying the learning rate over steps or tokens, such as cosine decay, inverse-square-root decay, or a staged schedule. |
| Epoch | One traversal of a defined training dataset. Streaming or repeated large corpora can make the definition less informative than token counts. |
| Mini-batch | A subset of examples processed together for an update or accumulated contribution to an update. |
| Backpropagation | Efficient application of the chain rule through a computation graph to calculate gradients. It is not itself an optimizer. |
| Backpropagation through time | Backpropagation through an unrolled recurrent computation; truncation restricts how far gradients flow. |
| Vanishing gradient | A gradient becoming too small across a computation chain to meaningfully train earlier operations. |
| Exploding gradient | A gradient becoming excessively large, destabilizing optimization or numerical representations. |
| Gradient clipping | Bounding a gradient norm or components to limit extreme updates; it does not automatically correct a poor objective. |
| Activation function | A transformation introducing nonlinearity or gating, such as ReLU, sigmoid, tanh, GELU, or SiLU. |
| ReLU | Rectified linear unit, $`\max(0,x)`$; simple and efficient, with a zero-gradient region on the negative side. |
| Sigmoid | $`\sigma(x)=1/(1+e^{-x})`$; useful for binary probabilities and gates, with saturation at large magnitudes. |
| Tanh | A bounded signed activation; saturation can attenuate gradients in recurrent chains. |
| GELU | A smooth gating-like activation $`x\Phi(x)`$, where $`\Phi`$ is the standard normal cumulative distribution function. |
| SiLU / Swish | The activation $`x\sigma(x)`$, used in many modern convolutional and gated feed-forward networks. |
| Softmax | A normalization from logits to positive weights summing to one. It is used in probability outputs, attention, and routers. |
| GLU / SwiGLU / GEGLU | Multiplicative feed-forward gates; variants use sigmoid, SiLU, or GELU transformations to modulate another projection. |
| Dropout | Randomly suppressing activations during training under a specified scaling rule; a form of stochastic regularization. |
| Weight decay | Shrinking weights during optimization; implementation and interaction with adaptive updates matter. |
| Early stopping | Choosing a checkpoint before continued training degrades the relevant validation criterion. |
| Batch normalization | Normalizing using mini-batch statistics with learned affine parameters; training/inference statistics can differ. |
| Layer normalization | Normalizing over a feature dimension within an example, without requiring cross-example batch statistics. |
| RMSNorm | Normalization using root-mean-square scale without subtracting the mean in its standard form. |
| Residual / skip connection | A path that adds or otherwise carries earlier representations around a transformation, assisting signal and gradient flow. |
| Data augmentation | Transformations intended to preserve useful task information while varying nuisance factors. Label-preservation is an assumption to validate. |
| Surrogate gradient | A differentiable approximation used during backward computation for a nondifferentiable forward operation, such as a spike threshold. |
| Straight-through estimator | A gradient estimator that substitutes a convenient derivative through a discrete operation; usually biased. |
| STDP | Spike-timing-dependent plasticity: local synaptic changes depend on relative presynaptic and postsynaptic spike timing. Unmodulated forms are commonly unsupervised; reward-modulated and other variants add signals. STDP is distinct from surrogate-gradient backpropagation through time. |

## 6.3 Representation, sequence, and generative modeling

| Term | Definition |
|---|---|
| Embedding | A learned or constructed vector representation of an item such as a token, image, entity, or graph node. |
| Latent variable | An unobserved variable introduced to model hidden structure or generative variation. |
| Feature map | A transformation into a representation space; in a CNN the term also refers to a spatial activation channel. |
| Kernel | A function representing similarities, often an implicit inner product in a feature space when positive semidefinite. |
| Margin | A measure of separation between a decision boundary and examples or between target and competing scores. |
| Attention | A content-dependent weighted aggregation of values using scores computed from queries and keys. |
| Self-attention | Attention whose queries, keys, and values come from the same sequence or representation set. |
| Cross-attention | Attention in which queries come from one representation and keys/values from another. |
| Causal mask | A restriction preventing a prediction from attending to future positions during autoregressive modeling. |
| Positional encoding | Information about order or coordinates supplied to a sequence model. |
| RoPE | Rotary positional embeddings, encoding relative-position structure by rotating query/key components. |
| GQA / MQA | Grouped-query/multi-query attention, sharing key/value heads across multiple query heads to reduce cache and bandwidth needs. |
| MLA | Multi-head Latent Attention, a compressed key/value representation introduced in DeepSeek-V2 for efficient attention caching. |
| Autoregressive model | A model factoring a joint distribution into conditional predictions, such as $`p(x)=\prod_t p(x_t\mid x_{<t})`$. |
| Teacher forcing | Training a sequence predictor with observed prior target tokens rather than its own earlier generated tokens. |
| Exposure bias | A mismatch between training on true previous outputs and inference on the model's own potentially erroneous outputs. |
| State-space model | A model evolving a state over inputs/time and mapping it to observations or outputs; neural sequence variants can use structured or input-dependent dynamics. |
| Contrastive learning | Learning representations by comparing matched examples with alternatives, often pulling positives together and separating negatives. |
| Positive / negative pair | A pair treated as semantically associated or contrasting under a particular representation objective; the choice defines the learned invariances. |
| InfoNCE | A contrastive softmax objective comparing a positive score against a set of alternatives, with mutual-information interpretations under assumptions. |
| Representation collapse | Different inputs receiving uninformatively identical or low-diversity representations. |
| Stop-gradient | Treating a value as constant during a backward pass while retaining its forward value. |
| Masked modeling | Predicting deliberately hidden components of an observation from the remaining context. |
| Denoising | Learning to reconstruct or characterize clean data from a corrupted observation. |
| Autoencoder | An encoder-decoder trained with a reconstruction-related objective; compression alone does not guarantee useful semantics. |
| Reparameterization trick | Expressing a random variable as a differentiable transform of parameters and parameter-independent noise, enabling pathwise gradients. |
| Posterior collapse | A latent-variable model effectively ignoring its latent code, often when a powerful decoder explains the observations without it. |
| GAN | A generative adversarial network trained through a generator/discriminator or critic game. The exact loss and divergence interpretation vary. |
| Mode collapse | A generator producing too few kinds of output relative to the target distribution. |
| Normalizing flow | An invertible transformation of a base distribution whose density can be evaluated using a tractable Jacobian determinant. |
| Diffusion model | A model learning to reverse a corruption/noising process, with sampling performed through learned denoising steps or a related dynamical system. |
| Score | The gradient of a log density with respect to the observation, $`\nabla_x\log p(x)`$, in score-based generative modeling. |
| Classifier-free guidance | Combining conditional and unconditional denoising predictions to adjust conditioning strength, with quality/diversity tradeoffs. |
| Latent diffusion | Diffusion performed in a learned latent space rather than directly at full pixel resolution. |
| Tokenizer | A procedure converting text or another modality into model symbols. Different tokenizers make raw token-count comparisons imperfect. |

## 6.4 Adaptation, alignment, and evaluation

| Term | Definition |
|---|---|
| Pretraining | An initial broad training stage whose parameters or representations are reused later. |
| Transfer learning | Reusing knowledge from a source task or distribution for another task or distribution. |
| Fine-tuning | Updating pretrained parameters using a later objective or dataset; not necessarily updating every parameter. |
| SFT | Supervised fine-tuning, commonly on instruction/response examples for assistant behavior. |
| Distillation | Training a student to imitate or learn from a teacher's outputs, probabilities, intermediate representations, or generated data. |
| RLHF | Reinforcement learning from human feedback; feedback often trains a reward model, followed by policy optimization. It is not a synonym for all preference training. |
| RLAIF | Reinforcement learning from AI feedback; AI-generated evaluations or preferences participate in the training signal. |
| DPO | Direct preference optimization, a method fitting preferred versus dispreferred responses without the same explicit rollout-based RL loop used in standard RLHF pipelines. |
| LoRA | Low-rank adaptation, representing selected weight updates through low-rank factors while typically freezing base weights. |
| RAG | Retrieval-augmented generation, providing retrieved information to a generator. Retrieval is a system component, not automatically a weight update. |
| In-context learning | Adapting behavior through examples or instructions in a prompt without necessarily updating model parameters. |
| Zero-shot / few-shot | Evaluation without task demonstrations or with a small specified number; exact prompts and source of examples matter. |
| Data leakage | Information from evaluation targets or future/unavailable conditions entering the learning or selection pipeline. |
| Benchmark contamination | Evaluation examples or closely related material occurring in training data, potentially inflating apparent generalization. |
| Accuracy | Fraction of correct predictions under the stated evaluation rule. Class imbalance can make it misleading. |
| Precision / recall | Fraction of predicted positives that are correct, and fraction of actual positives that are retrieved. |
| F1 | Harmonic mean of precision and recall; averaging across classes or examples changes its meaning. |
| AUROC / AUPRC | Areas under ROC or precision-recall curves; class prevalence and operating thresholds affect interpretation. |
| Perplexity | Exponentiated average negative log probability under a consistent log base. Comparisons require compatible tokenization and evaluation data. |
| Pass@k | Probability or estimate that at least one of $`k`$ generated candidates succeeds under a benchmark's test protocol. It is not the same as a single deterministic solution's accuracy. |
| Majority voting | Choosing the most common answer among multiple samples; extra sampling consumes extra inference compute. |
| FID | Frechet Inception Distance, comparing feature-distribution moments for generated and reference images; it is encoder-, sample-, and protocol-dependent. |
| BLEU | A machine-translation metric based on n-gram precision with a brevity penalty and specified aggregation; not a direct measure of human understanding. |
| Confidence interval | A frequentist interval procedure's coverage statement under assumptions, not a probability that a fixed parameter moves within the observed interval. |
| Ablation | A controlled removal or alteration of a component to study its contribution; changing several factors limits attribution. |
| Production KPI | A measured application/business outcome such as latency, cost per completed task, or downtime; not interchangeable with a laboratory accuracy score. |

## 6.5 Clustering, patterns, and sparse systems

| Term | Definition |
|---|---|
| Centroid / medoid | A cluster mean in a vector space, or a representative member selected from observed objects. |
| Density-based clustering | Forming groups through sufficiently dense, connected regions rather than assuming equal spherical clusters. |
| Neighborhood graph | A graph encoding local relations, such as nearest-neighbor or radius-based connections. Its construction is a substantive modeling choice. |
| Manifold assumption | The idea that observations concentrate near a lower-dimensional structure within their ambient feature space. |
| Support | Frequency or fraction of transactions containing an itemset, under a stated denominator. |
| Confidence of an association rule | Estimated $`P(B\mid A)`$ for a rule $`A\Rightarrow B`$; not a statistical confidence interval. |
| Lift | Rule confidence divided by the consequent's marginal frequency; a measure of association rather than causality. |
| Anomaly / novelty | A departure from a reference notion of normality. Anomaly detection and novelty detection can differ in whether contaminated training data are expected. |
| MoE | Mixture of Experts: a model combining expert functions through a gating mechanism, which can be dense or sparse. |
| Router / gate | A function choosing or weighting experts. Learned, fixed, token-choice, and expert-choice routers behave differently. |
| Top-k routing | Selecting a fixed number of highest-scoring experts per token or input, normally with a weighting rule. |
| Expert-choice routing | Each expert selects tokens, usually under a capacity constraint; a token may receive a variable number of experts. |
| Capacity factor | A multiplier governing permitted tokens/assignments per expert relative to a defined average load. Conventions differ across implementations. |
| Load-balancing loss | An auxiliary objective discouraging uneven expert usage; its scale can trade off with the task objective. |
| Router collapse | Concentration of traffic on too few experts, starving others of training or creating capacity bottlenecks. |
| Router z-loss | A penalty on router-logit normalization magnitude used to improve numerical stability; not the same objective as load balancing. |
| Token dropping | Skipping expert computation for assignments exceeding capacity; residual paths or alternative handling determine the actual effect. |
| Dropless MoE | Executing accepted routing assignments without capacity-based token dropping, using suitable kernels or dynamic packing; memory still has limits. |
| Total parameters | All counted model weights under a stated convention, whether selected for a particular token or not. |
| Active parameters | Weights used for one token's forward computation under a stated route and counting convention; not an exact FLOP or memory measure. |
| Expert parallelism | Placing different experts on different devices and routing token representations to them. |
| All-to-all communication | A distributed exchange in which devices send different data to multiple peer devices, commonly used for expert dispatch and combination. |
| Tensor parallelism | Partitioning individual tensor operations across devices, usually requiring collectives within layers. |
| Pipeline parallelism | Partitioning layer stages across devices and scheduling micro-batches through them. |
| Data parallelism | Running replicas on different data and synchronizing parameter updates. |
| FSDP / ZeRO | Families of distributed training techniques that shard parameters, gradients, and/or optimizer state, depending on configuration. |
| FLOPs / FLOP/s | A count of floating-point operations, and a rate of operations per second. Advertised peak rate is not achieved application throughput. |
| BF16 / FP16 / FP8 | Floating-point formats with different range and precision; safe usage depends on scaling, accumulation, hardware, and operation type. |
| Quantization | Representing values at reduced precision, often with scales or codebooks; storage savings do not always translate to speed. |
| KV cache | Stored attention keys and values for previous tokens during autoregressive decoding. |
| Prefill / decode | Processing a prompt to build model state, and generating subsequent tokens using that state. Their performance bottlenecks can differ. |
| Throughput / latency | Work completed per unit time, and elapsed time for a request or stage. A system can improve one while worsening the other. |
| Tail latency | A high percentile of request latency, often more revealing than the mean for interactive services. |
| Compute-bound / bandwidth-bound | Limited primarily by arithmetic throughput, or by movement of weights, activations, or distributed messages. |
| Open weights / reproducible training | Downloadable parameters, versus access to sufficient data, code, configuration, and procedures to reproduce learning; the former does not imply the latter. |

## Coverage and continuation manifest

This glossary defines the core statistical, neural, generative, evaluation, and
systems terms used by the first edition. Return to the
[complete table of contents](../CONTENTS.md) for mechanisms, examples, and
primary sources attached to individual entries.

It is not a full mathematical dictionary or a complete optimizer, causal, or
reinforcement-learning catalog. Those would require separate scoped volumes.
