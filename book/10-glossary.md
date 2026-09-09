# 6. Glossary

Use this chapter when a technical word interrupts your reading. You do not
need to memorize the terms before starting. Each definition explains how the
book uses the word.

A glossary mention is not another full algorithm entry. A few formulas are
included as optional details; the words explain the main idea.

## 6.1 Learning, objectives, and statistical reasoning

| Term | Definition |
|---|---|
| Algorithm | A set of steps for learning or solving a problem. It is the procedure, not the saved result of training. |
| Architecture | The layout of a model's parts and connections. The same layout can be trained for different goals. |
| Model / checkpoint | A model is a system that makes predictions or generates outputs. A checkpoint saves its learned settings and sometimes the state needed to resume training. |
| Dataset | A collection of examples, such as rows in a table, pictures, or text documents. |
| Feature | An input detail used by a model, such as age, temperature, or a pixel value. Some models learn new features from raw inputs. |
| Label / target | The answer used to guide a training example. It can be a class, number, sequence, or a value taken from the data itself. |
| Parameter / weight | A number the model learns. A simple line model learns its slope and intercept; a large network can learn billions of numbers. |
| Hyperparameter | A setting chosen for the learning process, such as a learning rate or tree depth. It is different from a weight learned by the model. |
| Training / inference | Training adjusts the model using examples. Inference uses the trained model to make a new prediction or output. |
| Probability distribution | A description of which outcomes are possible and how likely they are. For continuous values, a density describes how probability is spread over ranges. |
| Supervised learning | Learning from examples with supplied answers. The answers can still be noisy or incomplete. |
| Semi-supervised learning | Using some examples with answers and extra examples without them. The extra data help only when they fit the task. |
| Unsupervised learning | Finding patterns without the task's answer labels, such as grouping similar records. |
| Self-supervised learning | Making practice questions from the data itself. Hiding a word and asking the model to guess it is one example. |
| Weak supervision | Learning from imperfect clues, such as noisy captions or approximate labeling rules. The clues still supply information. |
| Transductive learning | Learning while already seeing the unlabeled inputs that will be predicted later. Their answers stay hidden, but their inputs are not fresh test data. |
| Inductive learning | Learning a rule that can be used on new examples not seen during training. |
| Loss function | A score that tells the model how far it is from its training goal. Learning usually tries to make this score smaller. |
| Objective | The complete score being optimized. It can combine prediction errors with extra penalties or training goals. |
| Empirical risk | The average loss on the available examples. It estimates performance on a wider population but does not prove it. |
| Cross-entropy | A loss that penalizes low probability on the correct answer, especially confident mistakes. Optional math: $`-\sum_i q_i\log p_i`$, where $`q_i`$ is the target probability and $`p_i`$ is the model's prediction. |
| Mean squared error | Find each prediction error, square it, then average the results. Large mistakes count much more than small ones. |
| Hinge loss | A loss that asks for the right class with a safety gap, called a margin. Optional math: $`\max(0,1-yf(x))`$, with class sign $`y\in\{-1,1\}`$ and prediction score $`f(x)`$. |
| Likelihood | How well a probability model makes the observed data fit its learned settings. For continuous data it uses density, which is not a probability and can exceed one. |
| Maximum likelihood | Choosing model settings that give the observed data the greatest likelihood. Minimizing negative log-likelihood is the same choice. |
| Prior / posterior | A prior describes uncertainty before using the stated evidence. A posterior is the updated description after using it. |
| Maximum a posteriori estimation | Choosing the setting at the highest point of the posterior. Some familiar weight penalties can be understood as adding a prior. |
| ELBO | "Evidence lower bound." A score used when a model's full likelihood is hard to compute. It balances explaining the data with keeping a learned hidden distribution close to a prior. It is a lower bound under the stated assumptions. |
| KL divergence | A measure of how one probability distribution differs from another. It is not negative, and reversing the two can change it. Optional math: $`D_{\mathrm{KL}}(q\Vert p)=\mathbb{E}_q[\log q-\log p]`$. The expectation means an average under $`q`$. |
| Entropy | A measure of uncertainty in a probability distribution. Confident predictions have low entropy, but they can still be wrong. |
| Regularization | Controls that help a model avoid fitting only the quirks of its training set. These include penalties, limits on model size, and other methods. |
| Bias-variance tradeoff | The balance between being too rigid to learn a pattern and changing too much with the training sample. It is a useful guide, not a simple law for every model. |
| Overfitting | Learning details of the training examples that do not help with new cases. |
| Underfitting | Missing useful patterns because the model, inputs, data, or training are not adequate. |
| Inductive bias | A model's built-in preferences about what patterns are likely. For example, a convolution looks for nearby image patterns. |
| Identifiability | Whether the data can distinguish between different model settings. Two different hidden explanations may make the same predictions. |
| Calibration | Predicted chances matching observed results. Among many similar predictions made with 80% confidence, about 80% should be right. This is a definition, not a result claimed for any model here. |
| Distribution shift | New data differing from the conditions used for training, such as a new camera or changed customer behavior. |
| Out of distribution (OOD) | Outside the data pattern chosen as the reference. What counts as outside depends on that reference. |
| Causal effect | A change caused by an action or intervention. A predictive link alone does not prove such an effect. |
| Censoring | Not knowing the full time until an event. For example, a machine may still work when a study ends; that does not mean it will never fail. |

## 6.2 Optimization and neural computation

| Term | Definition |
|---|---|
| Gradient | A list of local slopes. It tells how a small change in each model setting would change the loss. |
| Gradient descent | Repeatedly moving settings in a direction that locally lowers the loss. Optional math: $`\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}`$, where $`\theta`$ holds the settings, $`\eta`$ is step size, and the gradient gives the slopes. |
| SGD | Stochastic gradient descent. It estimates an update using a sample or small batch instead of all the data at once. |
| Momentum | Remembering part of the recent update direction. This can smooth noisy steps and help learning move through some difficult regions. |
| Adam | A weight-update rule that tracks recent gradient direction and size. It adjusts steps for each setting and corrects the early averages. |
| AdamW | Adam with a separate weight-shrinking step. This is not generally the same as putting an L2 penalty inside Adam's gradient calculation. |
| RMSProp | An update rule that uses recent squared gradients to adjust step sizes for different settings. |
| Adafactor | An adaptive update rule that saves memory by storing smaller summaries of gradient-size information for large weight arrays. |
| LAMB | An update rule that also adjusts the step for each layer using a size ratio. It is often studied with very large batches. |
| Learning rate | How large the learning steps are. A good value depends on the model, update rule, batch, and training stage. |
| Warmup | Starting with small learning steps and gradually increasing them. |
| Learning-rate schedule | A plan for changing step size during training. It may rise at first, then fall smoothly or in stages. |
| Epoch | One pass through a defined training dataset. For huge changing data streams, token counts may be more useful than epochs. |
| Mini-batch | A small set of examples processed together during learning. Several batches can also contribute to one update. |
| Backpropagation | Working backward through a network to calculate gradients. It tells an optimizer how to change weights; it is not the update rule itself. |
| Backpropagation through time | Applying that backward calculation through the steps of a recurrent model. A shortened version stops the calculation after a limited number of steps. |
| Vanishing gradient | The backward learning signal becoming too small to teach earlier layers or time steps effectively. |
| Exploding gradient | The backward signal becoming so large that updates or calculations become unstable. |
| Gradient clipping | Limiting an unusually large gradient. It helps control updates but does not fix a badly chosen training goal. |
| Activation function | A small rule that changes a layer's output. Such rules help a network learn more than straight-line relationships or control a gate. |
| ReLU | Keep positive values and replace negative values with zero: $`\max(0,x)`$. It is fast, but its negative side has no ordinary gradient. |
| Sigmoid | A smooth rule that puts a value between zero and one. It is useful for gates and two-class probabilities. Optional math: $`\sigma(x)=1/(1+e^{-x})`$. At the far ends it changes very slowly. |
| Tanh | A smooth rule with outputs between minus one and one. At the far ends, its small slope can weaken learning signals. |
| GELU | A smooth rule that reduces some signals more than others. Optional math: $`x\Phi(x)`$, where $`\Phi(x)`$ is the fraction of a standard bell curve below $`x`$. |
| SiLU / Swish | Multiply an input by its sigmoid gate: $`x\sigma(x)`$. This gives a smooth way to change its strength. |
| Softmax | Turns a list of scores, called logits, into positive shares that add to one. Those shares can weight answers, attention, or experts. |
| GLU / SwiGLU / GEGLU | Gate-based layers. One learned signal is multiplied by another that acts as a gate. The variants use sigmoid, SiLU, or GELU rules. |
| Dropout | Randomly switch off some signals during training, with suitable rescaling. This can reduce dependence on a few signals. |
| Weight decay | Shrinking learned weights during updates. Its effect depends on how the optimizer applies it. |
| Early stopping | Keeping a saved model from before more training makes the chosen validation result worse. |
| Batch normalization | Rescaling signals using statistics from a batch, then applying learned adjustments. Training and later use can rely on different statistics. |
| Layer normalization | Rescaling features within one example. It does not need averages across different examples in a batch. |
| RMSNorm | Rescaling by the root mean square: square the values, average them, and take the square root. The usual version does not subtract their mean first. |
| Residual / skip connection | A shortcut that carries earlier values around a layer. Adding the shortcut back can help signals and gradients travel through a deep network. |
| Data augmentation | Making changed versions of training examples, such as cropping pictures. The change should keep the answer valid; that must be checked. |
| Surrogate gradient | A stand-in slope used during the backward pass when the real operation has no useful slope. Spike thresholds are one example. |
| Straight-through estimator | Pretending a discrete step has a convenient slope for learning. The resulting gradient estimate is usually biased, not exact. |
| STDP | Spike-timing-dependent plasticity. A connection changes based on the timing of the spikes before and after it. Basic forms need no answer labels; other versions add rewards or signals. It is not surrogate-gradient backpropagation. |

## 6.3 Representation, sequence, and generative modeling

| Term | Definition |
|---|---|
| Embedding | A list of numbers representing something, such as a word or image. A useful embedding helps the model compare or process items. |
| Latent variable | An unobserved value a model uses to explain hidden variation, such as an image's style. It need not match a real-world cause. |
| Feature map | A new set of features made from the input. In a CNN, it can mean a grid showing where a learned image pattern responds. |
| Kernel | A function measuring a chosen kind of similarity. In many learning methods it acts like a dot product in a feature space without building all those features explicitly. |
| Margin | A gap between classes or prediction scores. A larger useful gap can make a decision less fragile. |
| Attention | Combining stored values using weights based on matching a query with keys. It is a calculation, not proof of human-like attention. |
| Self-attention | Queries, keys, and values come from the same sequence or set of features. |
| Cross-attention | Queries come from one set of features and keys and values from another, such as a decoder looking at an encoded source sentence. |
| Causal mask | A rule that hides future positions when predicting the next token. This prevents looking ahead at the answer. |
| Positional encoding | Extra information telling a model where an item sits in a sequence or grid. |
| RoPE | Rotary positional embeddings. They rotate parts of query and key vectors so attention can use relative position. |
| GQA / MQA | Grouped-query and multi-query attention. Several query heads share key/value heads, reducing stored data and memory traffic. |
| MLA | Multi-head Latent Attention. DeepSeek-V2 introduced it to store a compressed form of attention keys and values. |
| Autoregressive model | Predicts the next item using earlier items, then repeats. Optional math: $`p(x)=\prod_t p(x_t\mid x_{<t})`$: multiply each next-item probability given the earlier items. |
| Teacher forcing | During training, supply the true earlier tokens rather than the model's own earlier guesses. |
| Exposure bias | A mismatch between training with correct earlier answers and later using the model's own possibly wrong answers. |
| State-space model | Carries a running state forward through time or a sequence. Each step updates the state and produces an output. Some versions change their update rule based on the input. |
| Contrastive learning | Learns useful features by comparing matching examples with other examples. It often brings matches closer and pushes nonmatches apart. |
| Positive / negative pair | Two examples treated as a match or a contrast in training. Choosing these pairs helps decide what differences the model will ignore. |
| InfoNCE | A common contrastive loss. It asks a model to pick the matching example from alternatives using softmax scores. Its link to shared information requires extra assumptions. |
| Representation collapse | Different inputs receiving nearly the same unhelpful representation. The model then loses useful distinctions. |
| Stop-gradient | Keep a value in the forward calculation, but do not send a gradient back through that path. |
| Masked modeling | Hide part of an input and learn to predict it from what remains. |
| Denoising | Learn about clean data by removing or modeling added corruption. |
| Autoencoder | An encoder makes a code and a decoder tries to rebuild the input from it. Good rebuilding does not guarantee a useful meaning for the code. |
| Reparameterization trick | Separate random noise from learned settings when drawing a sample. This lets gradients pass through the learned part of the sampling calculation. |
| Posterior collapse | A model learns to ignore its hidden code, often because the decoder can explain the data without it. |
| GAN | Generative adversarial network. A generator makes samples while a discriminator or critic supplies feedback. The exact training game varies by design. |
| Mode collapse | A generator keeps producing too few kinds of output instead of the full variety in the data. |
| Normalizing flow | A chain of reversible transformations from a simple distribution. Tracking how the transformations stretch space lets the model calculate a density. |
| Diffusion model | Learns how to reverse added noise. It can generate a sample by following a learned sequence of denoising steps or a related continuous process. |
| Score | In score-based generation, a local direction toward higher log density. Optional math: $`\nabla_x\log p(x)`$. This is not a test accuracy score. |
| Classifier-free guidance | Mix predictions made with and without a condition, such as text, to change how strongly generation follows it. Stronger guidance can reduce variety or cause artifacts. |
| Latent diffusion | Run diffusion on a learned compact image code instead of the full set of pixels. The code is decoded back into an image. |
| Tokenizer | Splits text or another input into symbols the model uses. Tokens may be words, word parts, or other units; different tokenizers count differently. |

## 6.4 Adaptation, alignment, and evaluation

| Term | Definition |
|---|---|
| Pretraining | A broad first learning stage whose settings or features are reused later. |
| Transfer learning | Using learning from one task or dataset to help another. |
| Fine-tuning | Further training a pretrained model for a task. It may update some weights or all of them. |
| SFT | Supervised fine-tuning. An assistant often learns from supplied instruction-and-answer pairs. |
| Distillation | A student model learns from a teacher's answers, probabilities, features, or generated examples. It does not automatically keep every teacher ability. |
| RLHF | Reinforcement learning from human feedback. Human ratings often teach a reward model, which then guides the assistant's behavior. Not all preference training is RLHF. |
| RLAIF | Reinforcement learning from AI feedback. AI-generated ratings or preferences help supply the learning signal. |
| DPO | Direct preference optimization. It trains on preferred and less-preferred answers without the same generate-score-update RL loop used in standard RLHF. |
| LoRA | Low-rank adaptation. Usually freeze the base weights and learn a smaller set of factors that changes selected weight arrays. |
| RAG | Retrieval-augmented generation. Search for useful information and give it to a generator. This search does not itself retrain the model. |
| In-context learning | Give examples or instructions in a prompt to change how the model responds, usually without changing its weights. |
| Zero-shot / few-shot | Test with no task demonstrations, or a small stated number. The exact prompts and examples still matter. |
| Data leakage | Test answers, future information, or other unavailable clues getting into training or model selection. This can make a result look too good. |
| Benchmark contamination | Test examples or close copies appearing in training data. The test may then measure memory of those examples instead of fresh generalization. |
| Accuracy | The fraction of predictions counted as correct. It can hide poor results on rare classes. |
| Precision / recall | Precision asks, "Of the items flagged, how many were right?" Recall asks, "Of all the true items, how many were found?" |
| F1 | A combined precision-and-recall score using their harmonic mean. It is low if either is low. Different averaging rules can give different F1 results. |
| AUROC / AUPRC | Summaries of how a classifier's tradeoffs change as its decision threshold moves. AUROC uses true/false positive rates; AUPRC uses precision/recall. Class frequency affects interpretation. |
| Perplexity | A measure of how surprised a model is by the correct next tokens; lower is better. Technically it is the exponential of average negative log probability. Compare only compatible data and tokenizers. |
| Pass@k | How often at least one of $`k`$ generated attempts succeeds under a test's rules. It is not the same as getting one fixed answer right. |
| Majority voting | Generate several answers and choose the most common one. The extra attempts use extra compute. |
| FID | Frechet Inception Distance. It compares summary statistics of real and generated image features. The feature model, number of images, and test setup affect it. |
| BLEU | A translation score based on overlap of word groups with reference translations, with a penalty for overly short outputs. It is not percent correct or a direct measure of human understanding. |
| Confidence interval | A range built by a rule with a stated long-run coverage rate, under assumptions. The statement concerns repeated samples; it does not mean the fixed true value moves inside a chosen range. |
| Ablation | Remove or change a component to learn what it contributes. Changing many things at once makes the cause of a gain harder to identify. |
| Production KPI | A useful real-world measure, such as cost per finished task or machine downtime. A lab accuracy score is not automatically a business KPI. |

## 6.5 Clustering, patterns, and sparse systems

| Term | Definition |
|---|---|
| Graph / node / edge | A graph represents items and their links. Items are nodes; links are edges. An edge need not prove a real-world cause. |
| Centroid / medoid | A centroid is a group's average point. A medoid is a representative chosen from the actual observed items. |
| Density-based clustering | Forming groups from connected crowded regions, rather than assuming every group is a round ball of similar size. |
| Neighborhood graph | A graph linking nearby items under a chosen distance rule. Different distance rules can lead to different results. |
| Manifold assumption | The idea that data with many measurements lie near a simpler, lower-dimensional shape, like a curved sheet inside a room. |
| Support | How often a set of items appears together, as a count or a fraction of transactions. The denominator must be stated. |
| Confidence of an association rule | Among transactions containing A, the fraction also containing B. Optional math: $`P(B\mid A)`$ for $`A\Rightarrow B`$. This is not a confidence interval. |
| Lift | A rule's confidence divided by how often B appears overall. It compares co-occurrence with an independence baseline, not with a causal intervention. |
| Anomaly / novelty | Something unusual compared with a chosen pattern of normal data. Methods differ in whether unusual examples may already be mixed into training. |
| MoE | Mixture of Experts. A model mixes outputs from small networks called experts. It may use all experts or only a chosen few. |
| Router / gate | The part that chooses experts or sets their contribution weights. The rule may be learned or fixed. |
| Top-k routing | Choose the $`k`$ highest-scored experts for each token or input, then combine their outputs under a stated weighting rule. |
| Expert-choice routing | Each expert chooses inputs up to its capacity. Different inputs can receive different numbers of experts, including none. |
| Capacity factor | A setting that helps set how much work an expert may accept. Its exact definition differs between implementations. |
| Load-balancing loss | An extra training penalty that discourages sending most work to a few experts. Too much pressure to balance can hurt the main task. |
| Router collapse | The router keeps choosing too few experts. Other experts get little training, while the busy ones may become bottlenecks. |
| Router z-loss | A penalty that controls the size of a router's log-normalizing calculation. It helps numerical stability; it is not the same as balancing traffic. |
| Token dropping | Skipping an expert's work for an assignment that exceeds capacity. It usually does not mean deleting the token from the whole sequence. |
| Dropless MoE | Runs assigned expert work without the usual capacity-based dropping. Suitable packing and software make this possible, but memory is still limited. |
| Total parameters | All weights counted in the model, whether or not one token uses them. Different reports may count some shared parts differently. |
| Active parameters | Weights used for one token's forward calculation under the stated counting rule. This does not directly give memory use or exact arithmetic cost. |
| Expert parallelism | Store experts on different devices and send inputs to the devices that own the chosen experts. |
| All-to-all communication | Devices send different pieces of data to many other devices. MoE often uses it to send inputs out and bring expert results back. |
| Tensor parallelism | Devices divide a large array calculation within a layer. They must exchange results to finish it. |
| Pipeline parallelism | Different devices own successive layers. Batches move through those devices in stages. |
| Data parallelism | Model copies process different examples, then share the updates they learned. |
| FSDP / ZeRO | Training tools that divide weights, gradients, or saved optimizer state across devices, depending on their settings. |
| GPU / TPU | Processors often used to train or run neural networks. GPUs handle many calculations in parallel; Google's TPUs are built for machine-learning workloads. |
| FLOPs / FLOP/s | A count of floating-point calculations, and calculations per second. A processor's advertised peak rate is not a promise for a real model. |
| BF16 / FP16 / FP8 | Ways to store approximate numbers with different bit counts and ranges. Safe use depends on rounding, scaling, and which calculations keep more precision. |
| Quantization | Use fewer bits to store values, often with extra scale information. Saving memory does not always make a model faster or keep answers unchanged. |
| KV cache | Saved attention keys and values for earlier tokens. It avoids repeating some work while generating later tokens. |
| Prefill / decode | Prefill processes the prompt. Decode generates new tokens using that prepared state. Their main speed limits can differ. |
| Throughput / latency | Throughput is work finished per unit time. Latency is the wait for a request. Improving one can worsen the other. |
| Tail latency | The slow end of request times, such as a high percentile. It shows delays that an average can hide. |
| Compute-bound / bandwidth-bound | Limited mainly by calculation speed, or by how fast data can move between memory and devices. |
| Open weights / reproducible training | Open weights can be downloaded. Reproducible training also needs enough data, code, settings, and instructions to repeat the learning. One does not guarantee the other. |

## Coverage and continuation manifest

This glossary explains the main learning, neural-network, generation, testing,
and computing terms in the book. Use the
[complete table of contents](../CONTENTS.md) to find full methods and sources.

It is not a full math dictionary or a separate catalog of every optimizer,
cause-and-effect method, or reward-learning algorithm.
