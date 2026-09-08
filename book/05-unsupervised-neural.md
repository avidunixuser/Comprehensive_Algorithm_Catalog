# 3. Unsupervised Learning Algorithms: Neural Generation and Representation Pretraining

This volume covers sections **3.6-3.10**: reconstruction, adversarial generation, diffusion and text-to-image systems, flows and autoregression, and representation pretraining. It continues [classical unsupervised learning](04-unsupervised-classical.md) and precedes [foundation models](06-foundation-models.md).

**Evidence policy, checked 2026-09-08.** Historical papers and explicitly identified model versions are reference points, not claims about the newest available systems. A research benchmark, a selected demonstration, and a deployed service establish different things. Reported benchmark measurements below retain their evaluation conditions; none establishes commercial cost savings. Architectural explanations and comparisons are this book's technical synthesis unless attributed to a study.

**Supervision is a property of the objective and data, not the network name.** Reconstructing an input or predicting its missing components supplies self-supervision. Image-caption pairs supply natural-language supervision, even when collected without task-specific annotation. Class-conditioned generation uses class labels. Consequently, this chapter's editorial grouping is broader than strictly label-free learning: it includes conditional generators alongside their unconditional relatives and places CLIP with representation pretraining while explicitly cross-referencing [supervised contrastive and Siamese networks](02-supervised-neural.md). Evaluation with labeled linear probes does not retrospectively make the preceding image-only pretraining supervised.

**Cost and uncertainty conventions.** Let $`n`$ be training examples, $`B`$ batch size, $`E`$ epochs, $`p`$ trainable parameters, and $`C_f`$ the forward cost of a specified network for one example. An ordinary gradient update is approximately a constant multiple of $`BC_f`$, not universally $`O(Bp)`$: convolutions reuse weights at many positions. A convolution costs approximately $`O(HWk^2c_{\rm in}c_{\rm out})`$; a dense-attention Transformer with sequence length $`T`$, width $`d`$, and $`L`$ layers costs $`O(BL(Td^2+T^2d))`$ per forward pass. $`S`$ denotes sampling steps, not sequence length. Stored weights and optimizer state scale with $`p`$, while activations depend on batch, resolution, depth, and implementation. Sample diversity, reconstruction error, contrastive similarity, and likelihood are **not interchangeable with calibrated predictive uncertainty**.

## 3.6 Reconstruction and latent-variable models

An encoder-decoder bottleneck can learn a useful representation without defining a generative probability distribution. VAEs add an explicit latent-variable distribution; vector-quantized models add discrete codes and typically learn a separate prior. The distinction determines what sampling and likelihood statements are justified.

### 3.6.1 Autoencoders

**Name:** Autoencoder, with a deep undercomplete autoencoder as the representative neural realization.

**Category & sub-category:** Unsupervised/self-supervised learning; reconstruction and nonlinear dimensionality reduction.

**Originating paper/vendor/year:** Autoassociative networks predate modern deep learning. Hinton and Salakhutdinov's [*Reducing the Dimensionality of Data with Neural Networks*, Science, 2006](https://www.cs.toronto.edu/~hinton/absps/science.pdf) is an influential deep formulation, not the invention of every autoencoder.

**Core mechanism:** An encoder $`f_\phi`$ maps $`x`$ to a restricted code $`z`$; a decoder $`g_\theta`$ reconstructs $`x`$. Sharing a low-dimensional code forces the model to preserve recurring structure rather than independently storing each input coordinate. With linear layers, squared error, and suitable constraints, the learned subspace relates to PCA; nonlinear layers can represent curved structure. A sufficiently unconstrained network can instead learn a near-identity mapping.

**Inputs/outputs and typical data types:** Numerical vectors, images, spectra, or document-frequency vectors enter; latent coordinates and reconstructions leave. Feature scaling and the output observation model must match the data. A plain autoencoder does not automatically define a distribution from which arbitrary latent samples are meaningful.

**Strengths and limitations:** Useful for nonlinear compression and initialization when labels are scarce. Reconstruction can emphasize nuisance variation rather than the downstream task. Low residuals are not proof of normality, and high residuals are not calibrated anomaly probabilities; thresholding requires a separately validated reference population.

**Computational complexity / scalability notes:** For dense widths $`d_0,\ldots,d_L`$, one example costs $`O(\sum_l d_{l-1}d_l)`$ per forward pass; training adds decoder and backward work. Convolutional versions follow the spatial cost above. Encoding a new item needs only the encoder; reconstruction needs both networks.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** Hinton and Salakhutdinov studied news retrieval using 804,414 Reuters newswire stories, represented by probabilities over 2,000 word stems. Document vector -> encoder -> ten-dimensional code -> cosine-nearest documents gives a retrieval ranking. The study trained on half the stories and reported better category-based retrieval than latent semantic analysis. Nonlinear compression is the technical reason to prefer this model over a linear projection when semantic neighborhoods are not linearly organized. The [paper's retrieval experiment](https://www.cs.toronto.edu/~hinton/absps/science.pdf) is research evidence, not a Reuters production deployment or a measured analyst-productivity improvement.

**Notable vendor implementations/libraries:** PyTorch and Keras provide the necessary dense and convolutional layers; the authors released supporting code. A framework's ability to construct an autoencoder is not evidence that its vendor operates this retrieval system.

**Architecture diagram description:**

```text
word-probability vector (2000)
  -> dense 500 -> dense 250 -> dense 125 -> linear code (10)
  -> mirrored decoder -> normalized reconstructed word probabilities
```

**Activation functions used and why:** The cited document model uses logistic hidden units and a linear code, with a normalized output appropriate to word probabilities. Contemporary ReLU hidden layers are a different implementation choice, not the 2006 recipe.

**Loss function(s):** Reconstruction cross-entropy $`-\sum_j x_j\log \hat x_j`$ for the document experiment. Squared error is appropriate for a fixed-variance Gaussian observation model, not universally for counts or categorical values.

**Optimization algorithm(s):** The historical method first trains stacked restricted Boltzmann machines, then fine-tunes the unrolled network by backpropagation. Its staged optimization should not be relabeled Adam. For a new end-to-end implementation, Adam with validation-controlled learning-rate reduction is a reasonable explicitly proposed alternative, not a reported historical setting.

**Regularization techniques:** The narrow bottleneck is central. Weight penalties, sparsity, tied weights, and early stopping are optional variants. The original generative pretraining is an initialization strategy; it does not by itself guarantee generalization.

**Backpropagation considerations:** The chain rule traverses decoder and encoder. Deep sigmoid networks can saturate and lose gradient magnitude; this motivated the historical pretraining. A detached latent code would prevent ordinary joint encoder training.

**Parameter count / scaling behavior:** There is no family-wide count. Dense weights scale as adjacent-width products; shrinking the code does not eliminate the potentially expensive outer layers. Tied encoder-decoder weights reduce stored parameters.

**Training paradigm:** Input-as-target self-supervision, optionally followed by supervised downstream learning. The Reuters category labels evaluate neighborhoods rather than serve as reconstruction targets.

**Hardware/parallelism considerations:** Small dense models fit on CPUs; large image autoencoders benefit from GPU data parallelism. Decoder activations can dominate memory during training even when deployment stores only the encoder.

### 3.6.2 Denoising autoencoders

**Name:** Denoising autoencoder (DAE), including stacked denoising autoencoders.

**Category & sub-category:** Self-supervised learning; corruption-based reconstruction and representation pretraining.

**Originating paper/vendor/year:** Vincent, Larochelle, Bengio, and Manzagol, [*Extracting and Composing Robust Features with Denoising Autoencoders*, ICML 2008](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf); expanded by Vincent and colleagues in [JMLR, 2010](https://www.jmlr.org/papers/v11/vincent10a.html).

**Core mechanism:** Draw a corrupted input $`\tilde x\sim q_D(\tilde x\mid x)`$, then train $`g_\theta(f_\phi(\tilde x))`$ to recover clean $`x`$. The target is not the corrupted observation. In the original masking experiment, a selected fraction of coordinates is forced to zero. Learning to infer missing content discourages a trivial identity mapping and exploits dependencies among coordinates.

**Inputs/outputs and typical data types:** Clean training images or numerical vectors generate corrupted/clean pairs automatically. Outputs are repaired observations or hidden features. Applying this recipe to genuinely noisy measurements requires an appropriate corruption model; synthetically masking digits does not validate clinical-image denoising.

**Strengths and limitations:** Makes a useful learning task even for some overcomplete hidden representations. The corruption distribution determines robustness: random missing pixels, sensor noise, and adversarial perturbations are different problems. A deterministic reconstruction is generally a point estimate; variation across masks is not a calibrated posterior.

**Computational complexity / scalability notes:** One sampled corruption requires approximately one encoder-decoder forward/backward update, plus $`O(Bd)`$ masking for $`d`$ input coordinates. Averaging over $`m`$ independently corrupted views multiplies that work approximately by $`m`$. Stacked layerwise pretraining adds separate optimization stages.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The ICML study evaluated digit recognition on the Larochelle benchmark, including MNIST variants with rotation and image or random backgrounds. Images -> random coordinate removal -> stacked denoising encoders -> supervised classifier -> digit predictions separates pretraining from evaluation. The relevant variants used 10,000 training, 2,000 validation, and 50,000 test examples, not standard MNIST's usual split. The [study](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) found denoising-based initialization competitive with or better than ordinary autoencoder initialization on these tasks; the JMLR extension also reports lower classification errors and learned edge/stroke detectors. The technical fit is robustness-oriented initialization rather than PCA's variance preservation. No production OCR savings or business KPI was reported.

**Notable vendor implementations/libraries:** Keras and PyTorch support denoising reconstruction directly. Historical research code and the cited JMLR article document stacked training; a modern tutorial may use a different architecture.

**Architecture diagram description:**

```text
clean x -----> corruption sampler -----> corrupted x
   |                                      |
   |                               encoder -> hidden code
   |                                      |
   +-------- reconstruction loss <----- decoder -> repaired x
```

**Activation functions used and why:** The original basic formulation uses sigmoid encoder and decoder units for inputs in $`[0,1]`$. ReLU convolutional encoders and linear Gaussian-output decoders are later practical choices for different data ranges.

**Loss function(s):** $`\mathbb E_{x,\tilde x}[\ell(x,g_\theta(f_\phi(\tilde x)))]`$, with Bernoulli cross-entropy or squared reconstruction error as appropriate. It is conditional denoising, not adversarial discrimination or diffusion-chain likelihood.

**Optimization algorithm(s):** Original training uses stochastic gradient descent; layer sizes, corruption levels, pretraining duration, and supervised early stopping are selected experimentally. The paper does not prescribe a universal learning-rate schedule. A fixed initial rate followed by validation-triggered reduction is a reproducible design choice, not a claimed universal optimum.

**Regularization techniques:** Corruption is the defining regularizer. Undercomplete codes, weight decay, and early stopping can complement it. Corruptions must preserve enough task-relevant information to make reconstruction learnable.

**Backpropagation considerations:** Gradients traverse the reconstruction network, not the discrete mask draw. Monte Carlo corruption adds gradient variance. Sigmoid saturation and a decoder that over-smooths ambiguous inputs can reduce feature utility.

**Parameter count / scaling behavior:** Normally the same as the corresponding clean autoencoder; the random corruption operator adds no trainable weights. Depth and convolutional channel widths, not the name DAE, determine scale.

**Training paradigm:** Self-supervised pretraining on corrupted/clean views; supervised fine-tuning is a separate stage. If clean targets come from additional human-curated measurements, that extra supervision must also be recorded.

**Hardware/parallelism considerations:** Corruption can run on-device; data-parallel workers should use independent random masks. Large images require decoder activation memory, whereas frozen-feature inference can discard the decoder.

### 3.6.3 Variational autoencoders

**Name:** Variational autoencoder (VAE), using a diagonal-Gaussian approximate posterior.

**Category & sub-category:** Unsupervised generative modeling; amortized variational inference.

**Originating paper/vendor/year:** Kingma and Welling, [*Auto-Encoding Variational Bayes*, arXiv 2013, ICLR 2014](https://arxiv.org/abs/1312.6114). Related stochastic-backpropagation work appeared contemporaneously; the VAE is not simply a renamed deterministic autoencoder.

**Core mechanism:** Define a prior $`p(z)`$, decoder likelihood $`p_\theta(x\mid z)`$, and encoder $`q_\phi(z\mid x)`$. Optimize a variational bound rather than directly integrating every possible latent value. The encoder amortizes inference across examples; the decoder supports generation by sampling a prior latent and then an observation.

**Inputs/outputs and typical data types:** Images, numerical observations, or other data with a specified likelihood enter. Outputs include posterior mean/variance parameters, sampled latent codes, reconstructed distributions, and generated observations. A decoder's mean image is not the same object as a sample from its likelihood.

**Strengths and limitations:** Offers a principled reconstruction-versus-complexity objective and efficient approximate inference. Restrictive posteriors introduce an inference gap; factorized output likelihoods can produce blurry mean reconstructions. Latent posterior variance expresses uncertainty within the assumed model, not a Bayesian posterior over all network weights or automatically calibrated out-of-distribution uncertainty.

**Computational complexity / scalability notes:** With $`m`$ latent Monte Carlo samples, one minibatch costs an encoder pass plus approximately $`m`$ decoder passes and their gradients. Diagonal-Gaussian KL evaluation is linear in latent dimension $`r`$. Importance-sampled likelihood estimates add cost and must not be mislabeled exact likelihood.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [AEVB experiments](https://ar5iv.labs.arxiv.org/html/1312.6114) modeled MNIST and the Frey Face dataset. A face frame -> encoder's Gaussian parameters -> reparameterized latent -> Gaussian-output decoder yields a distribution over reconstructed frames; sampling prior latents yields new frames. The study compared variational bounds and estimated marginal likelihood against alternative estimators and demonstrated learned latent manifolds. The technical advantage over a deterministic autoencoder is an explicit latent prior and tractable inference objective. The original experiment used 200 hidden units for Frey faces and 500 for MNIST, with one Monte Carlo draw per example in minibatches of 100. These are research configurations, not a deployed facial-analysis system or a public business KPI.

**Notable vendor implementations/libraries:** Pyro, TensorFlow Probability, PyTorch, and Keras support VAE construction. Distribution libraries can implement the observation model and KL, but cannot determine whether those assumptions fit an application.

**Architecture diagram description:**

```text
x -> encoder -> mean mu(x), log-variance log sigma^2(x)
                       |
epsilon ~ N(0,I) -> z = mu + sigma * epsilon
                       |
                    decoder -> parameters of p(x | z)
prior N(0,I) ----------^       [generation bypasses encoder]
```

**Activation functions used and why:** The original MLP uses tanh hidden layers. Sigmoid outputs parameterize Bernoulli probabilities; Gaussian decoders need suitable mean and positive-variance parameterizations. Log-variance heads are unconstrained before exponentiation.

**Loss function(s):** Minimize negative ELBO: $`\mathbb E_q[-\log p_\theta(x\mid z)]+\mathrm{KL}(q_\phi(z\mid x)\|p(z))`$. For diagonal Gaussians against a standard normal, the KL is analytic; only reconstruction usually needs Monte Carlo estimation.

**Optimization algorithm(s):** The original experiments use Adagrad, selecting its global step size from 0.01, 0.02, and 0.1. Coordinatewise adaptation is the schedule mechanism reported there. Adam and warmup are common later recipes, not original AEVB requirements.

**Regularization techniques:** The prior-matching KL constrains information capacity. Weight regularization and early stopping remain optional. KL annealing or free bits can combat collapse but modify the training recipe and must be disclosed.

**Backpropagation considerations:** The reparameterization trick differentiates through $`z=\mu+\sigma\epsilon`$, avoiding a high-variance score-function estimator for continuous Gaussian latents. Monitor exponent overflow, posterior collapse, and decoder dominance.

**Parameter count / scaling behavior:** Encoder and decoder architecture determine $`p`$; mean and variance heads grow linearly with latent dimension times their input width. Training stores both networks; unconditional generation only needs the decoder and prior.

**Training paradigm:** Unsupervised density modeling with input-derived reconstruction targets. Conditional VAEs introduce labels or other conditioning data and are not strictly label-free.

**Hardware/parallelism considerations:** Minibatch data parallelism works naturally; independent noise draws are needed on each worker. Stable KL/log-variance arithmetic may require higher precision than the surrounding matrix multiplications.

### 3.6.4 Beta-VAE

**Name:** Beta-VAE, with the original weighted-KL objective distinguished from its capacity-controlled follow-up.

**Category & sub-category:** Unsupervised latent-variable modeling; information-constrained and disentangled representation learning.

**Originating paper/vendor/year:** Higgins and colleagues, [*beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, ICLR 2017](https://openreview.net/forum?id=Sy2fzU9gl). [Burgess and colleagues, 2018](https://arxiv.org/abs/1804.03599), analyze capacity control and provide the concrete convolutional recipe described below.

**Core mechanism:** Multiply the VAE KL penalty by $`\beta`$. Larger $`\beta`$ favors a more restricted representation, sometimes separating factors such as position, scale, and orientation at the expense of reconstruction. The follow-up instead penalizes deviation from a gradually increasing target KL capacity $`C`$. These objectives are related but not identical.

**Inputs/outputs and typical data types:** Images with repeated factors of variation enter; approximate latent distributions, reconstructions, and coordinate traversals leave. Ground-truth factors may be used to evaluate disentanglement without being provided to the encoder during the unsupervised experiment.

**Strengths and limitations:** A simple intervention on representation capacity can improve interpretability in controlled datasets. It does not guarantee that independent coordinates correspond to causal factors, particularly without suitable inductive biases. Reconstruction quality and disentanglement can conflict. Variational uncertainty remains conditional on the chosen prior, likelihood, and encoder family.

**Computational complexity / scalability notes:** Computational order matches a VAE: encoder, sampled decoder, and analytic diagonal KL per batch. Changing $`\beta`$ adds negligible arithmetic, but selecting it across seeds and datasets can require many complete training runs. Traversal plots sample latent coordinates rather than estimate deployment accuracy.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [dSprites dataset](https://github.com/google-deepmind/dsprites-dataset) contains 737,280 generated 64-by-64 binary images with controlled shape, scale, orientation, and position. Sprite pixels -> convolutional beta-VAE -> latent-coordinate traversal -> inspection against known factors tests whether representations separate transformations. The [Burgess study](https://ar5iv.labs.arxiv.org/html/1804.03599) demonstrates the reconstruction/disentanglement trade-off and capacity-controlled recovery of factors. Its separately described ground-truth-factor bottleneck experiment is supervised analysis, not evidence that every experiment was label-free. Compared with PCA, the technical attraction is nonlinear factorization; compared with beta=1, it is explicit information restriction. This is a scientific unit test, not an industrial deployment or business KPI.

**Notable vendor implementations/libraries:** Google's research disentanglement code and the DeepMind dSprites repository support evaluation; PyTorch and TensorFlow can implement both objective variants. The dataset explicitly is not a commercial Google product.

**Architecture diagram description:**

```text
64x64 sprite -> 4 strided convolutions -> 2 dense layers
  -> Gaussian mean/log-scale -> sampled latent
  -> transposed decoder -> Bernoulli pixel probabilities
                      KL to N(0,I) -> beta or capacity penalty
```

**Activation functions used and why:** The 2018 recipe uses ReLU hidden layers, four convolutional layers with 32 channels and 4-by-4 kernels, followed by two 256-unit dense layers. The decoder parameterizes Bernoulli pixels; its output probabilities must be bounded.

**Loss function(s):** Original: reconstruction negative log-likelihood plus $`\beta\mathrm{KL}(q\|p)`$. Capacity variant: reconstruction negative log-likelihood plus $`\gamma|\mathrm{KL}(q\|p)-C|`$. For $`\beta\ne1`$, the objective is not the ordinary VAE ELBO.

**Optimization algorithm(s):** The checked 2018 appendix uses Adam at $`5\times10^{-4}`$. For dSprites, the capacity target increases linearly from zero to 25 nats over 100,000 iterations; that is an information-capacity schedule, not learning-rate decay. These settings are not attributed to every original 2017 run.

**Regularization techniques:** KL weighting or capacity control is central. A factorized prior and diagonal approximate posterior impose additional structure. Selecting models using known factors introduces evaluation supervision even when the reconstruction stage does not.

**Backpropagation considerations:** Gaussian reparameterization applies. Excessive KL pressure can silence useful latents; monitor per-coordinate KL and reconstruction, not just total loss. A capacity penalty changes gradient direction when the target is crossed.

**Parameter count / scaling behavior:** The follow-up uses ten Gaussian latents for dSprites; its CelebA experiment uses a different latent size. Beta changes no weights by itself. Convolutional resolution and channel choices determine the actual parameter count.

**Training paradigm:** Primarily image-only self-supervised reconstruction, with an optional information-capacity curriculum. Ground-truth-factor interventions and downstream classifiers are distinct stages.

**Hardware/parallelism considerations:** Moderate-resolution experiments fit on ordinary GPUs. Data-parallel training must aggregate KL consistently; changing a sum to a mean changes the effective beta and invalidates recipe comparisons.

### 3.6.5 VQ-VAE

**Name:** Vector-quantized variational autoencoder (VQ-VAE), specifically the original single-level model.

**Category & sub-category:** Unsupervised generative modeling; discrete latent representations and learned tokenization.

**Originating paper/vendor/year:** Van den Oord, Vinyals, and Kavukcuoglu, DeepMind, [*Neural Discrete Representation Learning*, NeurIPS 2017](https://arxiv.org/abs/1711.00937). Hierarchical VQ-VAE-2 is a later extension, not the architecture assumed here.

**Core mechanism:** An encoder emits continuous vectors. Each is replaced by its nearest learned codebook vector; the decoder reconstructs from these discrete selections. A separately trained autoregressive prior models code indices for unconditional generation. Without this prior, the learned tokenizer is principally a reconstruction model, not a complete sampler of realistic code sequences.

**Inputs/outputs and typical data types:** Images, speech, or video enter. Outputs include discrete index grids/sequences, reconstructed observations, and samples generated through the learned prior. Codebook entries are vectors, not intrinsically interpretable phonemes or object labels.

**Strengths and limitations:** Makes representation learning compatible with discrete sequence models and compresses the prior's modeling problem. Dead codes, skewed code usage, and straight-through estimator bias require attention. Quantization error and codebook perplexity are diagnostics, not confidence intervals; a categorical prior's uncertainty is only as calibrated as its learned distribution.

**Computational complexity / scalability notes:** For $`N_z`$ latent positions, $`K`$ code vectors of width $`d_z`$, exhaustive nearest-code lookup costs $`O(BN_zKd_z)`$, in addition to encoder-decoder work. Codebook storage is $`O(Kd_z)`$. A PixelCNN prior samples positions sequentially but operates on a smaller grid than pixels.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original study](https://ar5iv.labs.arxiv.org/html/1711.00937) compresses 128-by-128 ImageNet images to a 32-by-32 index grid with 512 code choices. Image -> encoder -> nearest-code indices -> decoder produces recognizable reconstructions; a PixelCNN trained on those indices generates new image codes and decoded images. Modeling the smaller grid is the authors' rationale for allocating prior capacity to broader structure instead of pixel-level detail. The reported demonstrations establish reconstruction and sampling capability, not a standardized commercial compression saving. The nominal nine bits per index excludes model storage, prior coding overhead, and fidelity differences. The paper also demonstrates VCTK speech representations and speaker conversion; no public business KPI accompanies these research results.

**Notable vendor implementations/libraries:** DeepMind's Sonnet research examples and common PyTorch/JAX implementations support vector quantization. VQ tokenizers in later systems may use different objectives, codebook updates, or multiple levels.

**Architecture diagram description:**

```text
image -> convolutional encoder -> vectors z_e
  -> nearest codebook entries e[index] -> convolutional decoder -> reconstruction
encoded training indices -> PixelCNN prior
prior samples -----------> codebook lookup -> decoder -> generated image
```

**Activation functions used and why:** Convolutional/residual networks use ReLU hidden nonlinearities; decoder outputs match the reconstruction likelihood. Nearest-neighbor assignment is a discrete operation, not an activation with an ordinary useful derivative.

**Loss function(s):** Minimize $`-\log p_\theta(x\mid z_q)+\|\operatorname{sg}(z_e)-e\|^2+\beta\|z_e-\operatorname{sg}(e)\|^2`$. The terms train reconstruction, codebook placement, and encoder commitment. The index prior has its own categorical negative log-likelihood.

**Optimization algorithm(s):** The paper's image comparison uses Adam at $`2\times10^{-4}`$, batch 128, evaluated after 250,000 steps; no mandatory family-wide decay schedule follows. An EMA codebook update is an alternative to gradient-updated embeddings and must be identified as such.

**Regularization techniques:** Commitment prevents encoder outputs drifting away from codebook scale. The finite codebook constrains capacity. Usage monitoring and carefully justified code resets are implementation remedies, not proof against all representation collapse.

**Backpropagation considerations:** A straight-through estimator copies decoder gradients to encoder outputs across quantization. It is biased, not an exact derivative of argmin. Stop-gradient separates codebook and commitment updates; the prior is trained on discrete targets afterward.

**Parameter count / scaling behavior:** Total parameters include encoder, decoder, $`Kd_z`$ embeddings, and, when included, the prior. The prior can dominate storage even though the latent grid is compact.

**Training paradigm:** Input-derived reconstruction followed by unsupervised code-sequence modeling. Speaker-ID conditioning in the voice-conversion experiment adds explicit supervision and should not be confused with entirely label-free speech generation.

**Hardware/parallelism considerations:** Codebook distance matrices can be GPU-memory intensive. Distributed EMA codebooks require synchronized counts and sums; otherwise replicas learn incompatible vocabularies. Prior sampling latency remains a separate bottleneck.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Autoencoder | Numerical vectors, images, document frequencies | Nonlinear compact representation | Reconstruction need not preserve task semantics | Reuters news retrieval research |
| Denoising autoencoder | Corruptible images and sensor-like vectors | Learns dependencies by repairing corruption | Robustness depends on the chosen corruption | MNIST-variation recognition study |
| VAE | Data with an explicit observation likelihood | Amortized probabilistic latent inference | Approximation gap and posterior collapse | Frey Face and MNIST modeling |
| Beta-VAE | Images with repeatable generative factors | Explicit information-capacity control | Disentanglement is not guaranteed | dSprites factor-recovery experiments |
| VQ-VAE | Images, audio, video requiring discrete tokens | Discrete compression with a learned prior | Quantization bias and codebook collapse | ImageNet tokenization and VCTK research |

## 3.7 Adversarial generation

Adversarial training learns through a discriminator rather than requiring a tractable normalized data likelihood. It can produce sharp samples quickly at inference, but introduces a moving optimization target. A discriminator score is neither a general-purpose uncertainty estimate nor a reliable quality guarantee outside its training distribution.

### 3.7.1 Generative adversarial networks

**Name:** Generative adversarial network (GAN), generic framework instantiated by the original feed-forward image models.

**Category & sub-category:** Unsupervised generative modeling; adversarial distribution matching.

**Originating paper/vendor/year:** Goodfellow and colleagues, [*Generative Adversarial Nets*, NeurIPS 2014](https://arxiv.org/abs/1406.2661). Subsequent convolutional, Wasserstein, conditional, and style-based formulations change the architecture or objective.

**Core mechanism:** A generator $`G(z)`$ transforms noise into synthetic observations. A discriminator $`D(x)`$ learns to distinguish training observations from generated ones. Generator updates try to make generated observations harder to distinguish. The familiar equilibrium result assumes idealized optimization and sufficient model capacity; it does not guarantee convergence of practical simultaneous neural updates.

**Inputs/outputs and typical data types:** Real images or other observations train the system; random latent vectors generate samples. Conditional variants additionally consume labels, text, or another image. Standard GANs provide samples but generally no tractable normalized likelihood or native inverse encoder.

**Strengths and limitations:** Fast feed-forward synthesis and flexible differentiable generators are attractive when sampling matters more than likelihood. Mode collapse can omit substantial regions of the data distribution. A realistic-looking sample can still be memorized, biased, or structurally wrong. Latent randomness expresses diversity, not calibrated epistemic uncertainty.

**Computational complexity / scalability notes:** With $`k_D`$ discriminator updates per generator update, cost includes $`k_D`$ discriminator training passes plus generator training through the discriminator. Generator sampling requires one forward pass, unlike iterative diffusion. Training state includes both networks, their activations, and both optimizers.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [2014 paper](https://ar5iv.labs.arxiv.org/html/1406.2661) trained image generators on MNIST, the Toronto Face Database, and CIFAR-10. Real image batches and noise -> alternating discriminator/generator learning -> generated images -> sample inspection and distributional evaluation demonstrate learning without predefined pixel labels. The technical attraction over latent models with difficult inference is direct differentiable sampling without a training-time latent-inference procedure. The paper reports samples and Parzen-window likelihood estimates; those estimates are not exact GAN likelihoods and are not reused here as modern benchmark rankings. No verified commercial deployment or business KPI follows from this foundational experiment.

**Notable vendor implementations/libraries:** The authors' [research implementation](https://github.com/goodfeli/adversarial) documents the original system. PyTorch and TensorFlow support adversarial training, while research libraries provide distinct GAN variants; their results should not be attributed to generic GANs without naming the variant.

**Architecture diagram description:**

```text
noise z -> generator G -> fake image ----+
                                        +-> discriminator D -> real/fake logit
training real image --------------------+
generator update: differentiate through D into G, without updating D
```

**Activation functions used and why:** The original experiments used rectifier and sigmoid units in the generator and maxout units in the discriminator. The binary discriminator probability is produced by a sigmoid or equivalent stable logistic-loss implementation. Other activations belong to specific later architectures.

**Loss function(s):** Original game: $`\min_G\max_D \mathbb E_x\log D(x)+\mathbb E_z\log(1-D(G(z)))`$. The paper also motivates the non-saturating generator loss $`-\mathbb E_z\log D(G(z))`$, which improves early gradient signal without being the identical minimax generator update.

**Optimization algorithm(s):** Alternating minibatch stochastic gradient updates; the original algorithm used one discriminator step per generator step. Step sizes and their decay are implementation choices, not a universal GAN schedule. Adam is common in later GANs but is not necessary to define the framework.

**Regularization techniques:** The original discriminator uses dropout. Weight penalties, spectral normalization, gradient penalties, and data augmentation are distinct later choices; a Wasserstein gradient penalty should not be silently attached to the original logistic objective.

**Backpropagation considerations:** A discriminator that becomes too confident can starve the minimax generator of gradients. Non-saturating loss helps but does not solve all rotational game dynamics. Freeze discriminator parameter updates, not the computation graph connecting its input to the generator.

**Parameter count / scaling behavior:** $`p=p_G+p_D`$ during training; generation needs $`p_G`$. Greater discriminator capacity can improve feedback or overfit, so simply balancing parameter counts is not a stability theorem.

**Training paradigm:** Real-versus-generated labels are constructed by the training procedure, not externally supplied semantic classes. A conditional GAN with class labels or captions adds supervised information.

**Hardware/parallelism considerations:** GPUs accelerate both networks. Distributed training must synchronize the alternating updates and preserve consistent normalization statistics; generator-only inference is substantially simpler than the full training system.

### 3.7.2 DCGAN

**Name:** Deep convolutional generative adversarial network (DCGAN).

**Category & sub-category:** Unsupervised adversarial image generation; convolutional representation learning.

**Originating paper/vendor/year:** Radford, Metz, and Chintala, [*Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*, arXiv 2015, ICLR 2016](https://arxiv.org/abs/1511.06434).

**Core mechanism:** DCGAN constrains a GAN to an empirically effective convolutional design: learned strided downsampling in the discriminator, learned fractional-stride/transposed-convolution upsampling in the generator, and carefully placed batch normalization. It is an architecture-and-training recipe rather than a new likelihood. Discriminator features can subsequently support a supervised classifier.

**Inputs/outputs and typical data types:** Training uses normalized natural images; noise vectors produce images. The original illustrated generator maps a 100-dimensional latent to a 64-by-64 RGB image. Intermediate discriminator activations are another output when used as frozen representations.

**Strengths and limitations:** Provides a reproducible starting point for convolutional GANs and learns spatial feature hierarchies. Transposed convolutions can create uneven-overlap artifacts; batch normalization couples examples. Stable behavior on the published datasets is not a guarantee against collapse on smaller or more diverse datasets. Neither discriminator score nor interpolation smoothness establishes uncertainty calibration.

**Computational complexity / scalability notes:** Each layer's cost depends on spatial resolution, kernel, and channel products. Resolution growth increases activation memory even if weights are reused. The adversarial loop costs both generator and discriminator passes; extracting a representation after training needs only the discriminator backbone.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [DCGAN study](https://ar5iv.labs.arxiv.org/html/1511.06434) trained on the LSUN bedrooms collection, containing a little over three million training examples. Bedroom images scaled to $`[-1,1]`$ -> convolutional discriminator/generator training -> 64-by-64 synthetic rooms -> visual and representation evaluation formed the experiment. The authors demonstrated plausible room samples and investigated memorization rather than declaring photorealistic images alone sufficient evidence. Convolutional weight sharing is the technical advantage over a similarly sized dense GAN for spatial images. Their separate transfer experiments evaluate frozen features using labeled classifiers; those are not labels used by the bedroom generator. No hotel-design deployment or measured business saving was established.

**Notable vendor implementations/libraries:** The authors' [DCGAN Torch code](https://github.com/soumith/dcgan.torch) and the [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) provide implementations. Tutorial datasets, channel widths, and training budgets need not reproduce the paper.

**Architecture diagram description:**

```text
z (100) -> project/reshape -> transposed conv + BN + ReLU
        -> repeated spatial upsampling -> tanh -> 64x64 RGB
real/fake RGB -> strided conv + LeakyReLU
             -> conv + BN + LeakyReLU -> binary discriminator
```

**Activation functions used and why:** ReLU in the generator encourages usable gradients; tanh bounds generated pixels to the training range. Discriminator LeakyReLU retains a negative-side gradient; the paper uses slope 0.2. The binary output uses a logistic probability.

**Loss function(s):** Adversarial real/fake cross-entropy with a generator objective that encourages generated images to receive the real label. This is not a reconstruction loss; there is no target image paired with each noise vector.

**Optimization algorithm(s):** The paper uses Adam at 0.0002 with $`\beta_1=0.5`$, minibatch 128, rather than Adam's usual higher first-moment setting. No universal learning-rate decay is part of the architectural definition; this reported rate must not be generalized to every resolution.

**Regularization techniques:** Batch normalization excludes the generator output and discriminator input, where the paper found it destabilizing. Normal weight initialization with standard deviation 0.02 and the constrained architecture are important recipe components, although initialization is not itself a statistical regularizer.

**Backpropagation considerations:** Monitor discriminator domination and gradient oscillation. Batch statistics can leak real/fake batch composition; changing batch construction changes the game. Replacing transposed convolution with resize-convolution changes the architecture and artifacts.

**Parameter count / scaling behavior:** Channel multipliers determine count; the 100-dimensional latent does not specify total parameters. Generator and discriminator both contribute to training storage, but only generator weights are needed for sampling.

**Training paradigm:** Image-only adversarial learning in the representative model. A downstream linear SVM introduces labeled supervision at evaluation, and a class-conditioned DCGAN is a supervised conditional variant.

**Hardware/parallelism considerations:** Modest-resolution DCGANs fit on a single GPU. Large batches improve batch-statistic estimates but cost memory; distributed batch normalization choices can materially change reproduction results.

### 3.7.3 StyleGAN family

**Name:** StyleGAN family: StyleGAN, StyleGAN2, StyleGAN2-ADA, and StyleGAN3, with version boundaries retained.

**Category & sub-category:** Adversarial image generation; style-controlled synthesis and alias-aware generators.

**Originating paper/vendor/year:** NVIDIA researchers Karras and colleagues: [StyleGAN, arXiv 2018/CVPR 2019](https://arxiv.org/abs/1812.04948); [StyleGAN2, arXiv 2019/CVPR 2020](https://arxiv.org/abs/1912.04958); [adaptive discriminator augmentation, NeurIPS 2020](https://arxiv.org/abs/2006.06676); [StyleGAN3, NeurIPS 2021](https://arxiv.org/abs/2106.12423).

**Core mechanism:** A mapping network converts noise $`z`$ into styles $`w`$ that control synthesis at different scales.

- Original StyleGAN uses adaptive instance normalization, injected noise, and progressive growing.
- StyleGAN2 replaces problematic normalization with weight modulation/demodulation, changes generator structure, and introduces path-length regularization.
- StyleGAN2-ADA adjusts discriminator augmentation to combat limited-data overfitting; it is not a different semantic conditioning signal.
- StyleGAN3 redesigns signal processing to reduce texture sticking and aliasing. StyleGAN3-T targets translation equivariance; StyleGAN3-R additionally targets rotation equivariance. These are not interchangeable claims about every version.

**Inputs/outputs and typical data types:** Noise produces images; style mixing, latent inversion, or conditional variants provide additional controls. An inversion code is an optimization result, not necessarily the unique true cause of an input image.

**Strengths and limitations:** High-quality feed-forward generation and editable intermediate styles fit image synthesis research. Training distribution coverage limits identity, pose, and demographic diversity; inversion can miss unseen content. Truncation improves some visual-quality measures by sacrificing coverage. It is not confidence filtering.

**Computational complexity / scalability notes:** Mapping cost is usually smaller than high-resolution convolutional synthesis. Modulated convolutions still depend on channel products and spatial size; filtered nonlinearities in StyleGAN3 add work. Lazy regularization avoids computing expensive penalties every iteration but changes effective optimizer scheduling.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** NVIDIA introduced FFHQ, 70,000 aligned 1024-by-1024 face images, in the [original StyleGAN study](https://ar5iv.labs.arxiv.org/html/1812.04948). Faces -> adversarial training -> controlled styles and synthetic portraits -> evaluation of image quality and latent behavior demonstrate the method. StyleGAN2 investigates visible artifacts and reconstruction/inversion; [StyleGAN3](https://ar5iv.labs.arxiv.org/html/2106.12423) demonstrates more coherent transformations without texture sticking. These are documented research outcomes, not a claim that a specific film studio chose the model or reduced animation costs. Controllable scale-dependent synthesis is the technical fit relative to DCGAN; no audited production business KPI is asserted.

**Notable vendor implementations/libraries:** NVIDIA's official [StyleGAN2-ADA PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StyleGAN3](https://github.com/NVlabs/stylegan3) repositories. Check checkpoint, dataset, license, and configuration rather than relying on the family name.

**Architecture diagram description:**

```text
z -> mapping MLP -> w -> per-layer style controls
                         |
learned input / version-specific Fourier input
  -> modulated synthesis blocks -> RGB image -> discriminator
       StyleGAN1: AdaIN; StyleGAN2: demodulation
       StyleGAN3: alias-aware filtering and nonlinearities
```

**Activation functions used and why:** LeakyReLU is central to the mapping/synthesis implementations. StyleGAN3 filters nonlinear operations to suppress newly introduced high frequencies. Original AdaIN and StyleGAN2 demodulation must not be described as identical normalization.

**Loss function(s):** A representative StyleGAN2 recipe uses non-saturating logistic GAN loss, discriminator R1 input-gradient regularization, and generator path-length regularization. Penalty choices and strengths differ by version and configuration.

**Optimization algorithm(s):** The [official StyleGAN2-style configuration](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py) uses Adam with $`(\beta_1,\beta_2)=(0,0.99)`$ and base rate 0.002. Lazy-regularization scheduling adjusts effective optimizer settings; this is not a universal rate for all StyleGAN versions.

**Regularization techniques:** Style mixing, R1, path-length penalties, and EMA weights have distinct roles. ADA controls augmentation probability from discriminator behavior; it does not simply augment until every training image is unrecognizable.

**Backpropagation considerations:** R1 and path-length penalties require derivatives of derivatives and additional memory. Numerical conditioning of modulation/demodulation matters. StyleGAN3 removes per-pixel stochastic noise that conflicts with its equivariance goals.

**Parameter count / scaling behavior:** Resolution, channel caps, mapping depth, and version determine the count. Keep generator, discriminator, and optional inversion encoder counts separate; a resolution label alone is insufficient.

**Training paradigm:** Primarily unconditional image-distribution learning in the cited FFHQ work; labeled conditional variants are separate. Portrait identity must not be treated as ground truth supplied by an unconditional generator.

**Hardware/parallelism considerations:** The original paper reports training on eight V100 GPUs. Later implementations use fused/custom GPU kernels and mixed precision. Reproducible distributed runs must preserve augmentation, EMA, precision, and regularization intervals.

### 3.7.4 CycleGAN

**Name:** Cycle-consistent generative adversarial network (CycleGAN).

**Category & sub-category:** Unpaired image-to-image translation; adversarial learning with reconstruction consistency.

**Originating paper/vendor/year:** Zhu, Park, Isola, and Efros, [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017](https://arxiv.org/abs/1703.10593).

**Core mechanism:** Learn $`G:X\rightarrow Y`$ and $`F:Y\rightarrow X`$, each with a discriminator, while encouraging $`F(G(x))\approx x`$ and $`G(F(y))\approx y`$. Marginal adversarial matching encourages plausible target-domain images; cycle consistency discourages arbitrary mappings that discard all source content. It does not identify the unique semantically correct translation.

**Inputs/outputs and typical data types:** Separate image collections from two domains enter; translated images leave. Domain membership is known, but individual images need not be aligned pairs. Photographs, paintings, seasons, and semantic maps have different information-preservation requirements.

**Strengths and limitations:** Useful when aligned examples are scarce and a roughly content-preserving relationship exists. Cycle consistency can fail for many-to-one mappings, permit hidden information channels, or preserve the wrong semantics. A deterministic generator cannot express every plausible translation; cycle residual is not a calibrated uncertainty estimate.

**Computational complexity / scalability notes:** A cycle uses two generator passes, and training includes both directions and both discriminators. Cost is a constant multiple of the corresponding convolutional networks, with large activation memory for reconstructed cycles. One-direction deployment needs only one generator.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** In the [Cityscapes evaluation](https://ar5iv.labs.arxiv.org/html/1703.10593), the authors train translation between street photographs and semantic-label maps without using their available pair alignment. Label map -> generator -> synthetic street image -> pretrained FCN segmentation -> comparison with the input map measures whether generated imagery preserves recognizable scene categories. The reported evaluation and ablations support the combined adversarial/cycle objective over its incomplete variants. The technical reason to use CycleGAN rather than paired pix2pix is the absence of alignment during training; when reliable pairs exist, discarding them is not inherently advantageous. Domain labels and human-created segmentation maps still carry supervision. This is not an autonomous-driving deployment or a verified safety/business KPI.

**Notable vendor implementations/libraries:** The authors' [PyTorch CycleGAN and pix2pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) distinguishes the unpaired and paired methods. Third-party image filters may change the loss or generators.

**Architecture diagram description:**

```text
x in X -> G -> fake Y -> F -> reconstructed X
                 |                  |
                D_Y             cycle L1 to x
y in Y -> F -> fake X -> G -> reconstructed Y
                 |                  |
                D_X             cycle L1 to y
```

**Activation functions used and why:** The representative residual generators use ReLU internally and tanh outputs; PatchGAN discriminators use LeakyReLU. Instance normalization supports small-batch image translation without global batch statistics.

**Loss function(s):** Practical training uses least-squares adversarial losses, plus $`\lambda_{\rm cyc}(\|F(G(x))-x\|_1+\|G(F(y))-y\|_1)`$. Optional identity loss discourages unnecessary changes when a generator receives an image already in its output domain.

**Optimization algorithm(s):** The paper uses Adam, batch size one, and learning rate 0.0002. It holds the rate for 100 epochs, then linearly decays it to zero over 100 more. This schedule is distinct from a diffusion noise schedule.

**Regularization techniques:** Cycle consistency and optional identity penalties constrain the solution. A buffer of previously generated images reduces discriminator oscillation; resizing, cropping, and flipping provide data augmentation.

**Backpropagation considerations:** Cycle loss must differentiate through both generators. Separate updates must avoid accidentally backpropagating through a stale image buffer. Gradient competition between realism and content preservation requires inspecting both terms.

**Parameter count / scaling behavior:** Two generators and two discriminators contribute to training count. Residual-block number, input resolution, and channel widths matter; CycleGAN is not a fixed-parameter model.

**Training paradigm:** Unpaired conditional translation with known domains. It is often called unsupervised translation because pair correspondences are absent, not because domain partitioning or all source signals lack supervision.

**Hardware/parallelism considerations:** Batch-one training is feasible on a GPU but does not eliminate memory from cycle graphs. Multiple devices can parallelize examples or directions; retain consistent updates and avoid stale opposite-generator weights.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| GAN, generic | Data admitting a differentiable sampler | Direct feed-forward generation | Unstable game and missing modes | MNIST, Toronto Face Database, CIFAR-10 research |
| DCGAN | Moderate-resolution natural images | Convolutional synthesis and reusable features | Artifacts and batch-sensitive stability | LSUN bedroom-generation study |
| StyleGAN family | High-quality curated image domains | Style control; later alias-aware synthesis | Distribution bias and version-dependent behavior | FFHQ generation and transformation studies |
| CycleGAN | Two unpaired image domains | Translation without aligned examples | Cycle consistency does not ensure semantic truth | Cityscapes labels/photos research |

## 3.8 Diffusion and image-generation families

Diffusion learns to reverse a known corruption process. The denoiser can operate on pixels, compressed latents, or embeddings and can be a U-Net or a Transformer. The broader image-generation grouping also includes **DALL-E 1, which is autoregressive rather than a diffusion model**. DALL-E versions are separate entries because a product-family name does not establish architectural continuity.

Noise schedules determine the corruption process; learning-rate schedules determine optimization; inference schedulers determine numerical sampling. These are three different choices. Guidance and candidate reranking also change the sampled distribution and evaluation budget.

### 3.8.1 Denoising diffusion probabilistic models

**Name:** Denoising diffusion probabilistic model (DDPM), specifically the Ho, Jain, and Abbeel formulation.

**Category & sub-category:** Unsupervised generative modeling; discrete-time denoising diffusion.

**Originating paper/vendor/year:** Ho, Jain, and Abbeel, [*Denoising Diffusion Probabilistic Models*, NeurIPS 2020](https://arxiv.org/abs/2006.11239), building on earlier diffusion-based generative modeling rather than originating the entire diffusion idea.

**Core mechanism:** Repeated Gaussian perturbation transforms data toward noise. A time-conditioned network learns parameters of the reverse transitions. With $`\bar\alpha_t`$ the cumulative signal retention, training can draw $`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon`$ directly, without simulating all preceding forward steps.

**Inputs/outputs and typical data types:** Clean images and sampled noise levels train a noise predictor. At generation time, Gaussian noise -> reverse denoising chain -> image. The original CIFAR-10 model is unconditional; supplying class labels or captions changes the objective to conditional modeling.

**Strengths and limitations:** Stable regression-like training and broad mode coverage are advantages over adversarial games. Sampling requires many network evaluations in the original formulation. Denoising diversity is not a confidence interval about the truth of an image; generated anatomy, text, or events can be incorrect despite looking plausible.

**Computational complexity / scalability notes:** A training example normally samples one time index, so its update does not cost a full $`S`$-step sampling chain. Original sampling costs approximately $`SC_f`$. Resolution, attention placement, and U-Net channels determine $`C_f`$; faster samplers change the evaluation protocol.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** For unconditional 32-by-32 CIFAR-10 generation, clean training image -> analytically noised image/time -> U-Net noise estimate -> learned reverse sampler produces synthetic images for distributional evaluation. The [paper](https://ar5iv.labs.arxiv.org/html/2006.11239) reports **FID 3.17** and **Inception Score 9.46 +/- 0.11**, using 50,000 generated samples. FID uses training-set reference statistics and the minimum-FID checkpoint over training; the paper separately reports test-reference FID 5.24. These distinctions preclude treating 3.17 as a held-out classification accuracy. Noise regression is the technical alternative to GAN game optimization; the simplified objective improved sample quality relative to the paper's likelihood-bound objective. No public business KPI or deployed image service is established by this benchmark.

**Notable vendor implementations/libraries:** The authors' [TensorFlow implementation](https://github.com/hojonathanho/diffusion) and Hugging Face Diffusers implement DDPM components. A scheduler implementation alone does not identify the trained denoiser or checkpoint.

**Architecture diagram description:**

```text
clean image + sampled Gaussian noise -> x_t
x_t + time embedding -> residual U-Net with skip connections/attention
                    -> predicted noise epsilon_theta(x_t,t)
generation: Gaussian noise -> repeated reverse updates -> image
```

**Activation functions used and why:** The representative residual U-Net uses smooth swish nonlinearities and attention softmax; its noise-output head is unconstrained. Bounding predicted noise with tanh would alter the estimator.

**Loss function(s):** Simplified training minimizes $`\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2`$. The exact variational-bound terms have time-dependent weights; unweighted noise MSE is not numerically identical to optimizing that bound.

**Optimization algorithm(s):** The [official CIFAR configuration](https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py) uses Adam, rate $`2\times10^{-4}`$, 5,000-step warmup, and batch 128. It specifies 1,000 forward noise levels with linearly increasing beta from 0.0001 to 0.02. This beta schedule is distinct from learning-rate warmup.

**Regularization techniques:** Group normalization, dropout, random horizontal flips, and EMA weights support training and evaluation. The checked CIFAR configuration uses dropout 0.1.

**Backpropagation considerations:** Differentiate a single sampled denoising loss, not an unrolled full generation trajectory. Noise-level weighting changes gradient emphasis. Gradient clipping and careful variance arithmetic help numerical stability.

**Parameter count / scaling behavior:** The official CIFAR U-Net is identified as approximately 35.7 million parameters. This is not a universal DDPM count; higher-resolution conditional systems can be much larger.

**Training paradigm:** Self-supervised noise prediction on unlabeled images in the representative experiment. Conditional class/text signals remain supervision even though noise targets are synthetically constructed.

**Hardware/parallelism considerations:** The original implementation supports Cloud TPU training; GPU implementations are common. Data parallelism is straightforward, but sequential reverse steps constrain single-sample inference latency even with many devices.

### 3.8.2 Score-based SDE models

**Name:** Score-based generative modeling through stochastic differential equations (SDEs).

**Category & sub-category:** Unsupervised generative modeling; continuous-time score matching and stochastic/ODE sampling.

**Originating paper/vendor/year:** Song and colleagues, [*Score-Based Generative Modeling through Stochastic Differential Equations*, arXiv 2020, ICLR 2021](https://arxiv.org/abs/2011.13456).

**Core mechanism:** Define a forward SDE $`dx=f(x,t)dt+g(t)dw`$. Learn the time-dependent score $`\nabla_x\log p_t(x)`$, which determines the reverse-time drift. Variance-exploding, variance-preserving, and sub-variance-preserving SDEs organize different corruption families. A related probability-flow ODE has the same time marginals under an exact score, not the same individual stochastic trajectories.

**Inputs/outputs and typical data types:** Observations, continuous time, and Gaussian perturbations enter score training. A sampler returns images; an ODE-based density calculation can estimate log density. Conditional inverse problems additionally use observations and an appropriate conditioning procedure.

**Strengths and limitations:** Unifies discrete diffusion and noise-conditioned score methods and enables multiple numerical solvers. Discretization, score error, and conditioning approximations affect results. Posterior-looking samples in an inverse problem are not automatically calibrated; the measurement likelihood and score approximation must be validated.

**Computational complexity / scalability notes:** Training samples time and noise rather than integrating the whole SDE. Sampling cost is network evaluations times $`C_f`$, including predictor and corrector calls; adaptive ODE solvers have variable evaluation counts. Likelihood computation additionally estimates a divergence, often with Hutchinson trace probes and numerical integration.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [paper's CIFAR-10 experiments](https://ar5iv.labs.arxiv.org/html/2011.13456) train a continuous-time NCSN++ score network on perturbed images, then integrate a reverse sampler to produce images for evaluation. The **NCSN++ continuous deep, VE-SDE** model achieves **FID 2.20** under the authors' 50,000-sample protocol; the authors explicitly distinguish it from shallower and discrete-time models. The technical fit relative to a fixed discrete DDPM is access to continuous-time objectives and solver choices, not a universal guarantee of faster sampling. The paper also demonstrates inverse-problem applications, but the CIFAR result is a research benchmark, not an audited production KPI or a guaranteed calibrated reconstruction posterior.

**Notable vendor implementations/libraries:** The authors' [JAX/TensorFlow research code](https://github.com/yang-song/score_sde) and [PyTorch implementation](https://github.com/yang-song/score_sde_pytorch) expose SDE, model, and sampler choices separately.

**Architecture diagram description:**

```text
image x_0 + perturbation kernel at time t -> x_t
x_t + continuous-time embedding -> NCSN++ / DDPM++ score network
                               -> estimated score s_theta(x_t,t)
noise -> reverse-SDE predictor/corrector OR probability-flow ODE -> image
```

**Activation functions used and why:** The checked NCSN++ configuration uses swish, group normalization, and attention. Its score output is real-valued and can be scaled by the noise standard deviation; it is not a categorical softmax.

**Loss function(s):** Denoising score matching minimizes $`\mathbb E[\lambda(t)\|s_\theta(x_t,t)-\nabla_{x_t}\log p(x_t\mid x_0)\|^2]`$. The tractable conditional perturbation score provides the target. Weighting affects whether the emphasis is sample quality or a likelihood-related objective.

**Optimization algorithm(s):** The [official PyTorch CIFAR defaults](https://github.com/yang-song/score_sde_pytorch/blob/main/configs/default_cifar10_configs.py) use Adam at $`2\times10^{-4}`$, 5,000-step warmup, gradient clipping, and batch 128. These implementation defaults do not uniquely identify the best-paper deep checkpoint. SDE schedules and predictor/corrector settings are separately configurable.

**Regularization techniques:** EMA, dropout, input flips, residual rescaling, and noise-dependent targets stabilize the representative network. A continuous noise distribution is part of the objective, not proof against memorization.

**Backpropagation considerations:** Training normally avoids differentiating through the sampling solver. Low-noise targets can have large magnitude, requiring sensible scaling. ODE likelihood is exact only in the ideal mathematical model; finite tolerances and stochastic trace estimates introduce error.

**Parameter count / scaling behavior:** Network depth, channels, and attention set $`p`$. The cited improvement to 2.20 doubles residual blocks per resolution relative to the shallower continuous NCSN++; assigning that result to any NCSN++ is misleading.

**Training paradigm:** Unconditional self-supervised score estimation in the benchmark. Label conditioning, classifier guidance, or measurement conditioning creates different information regimes.

**Hardware/parallelism considerations:** Training parallelizes over images and times on GPUs/TPUs. Predictor-corrector sampling and divergence probes increase compute; batch-adaptive ODE evaluation can also waste work when examples require different tolerances.

### 3.8.3 Latent diffusion and Stable Diffusion

**Name:** Latent diffusion models (LDMs), with Stable Diffusion v1.4 and SDXL 1.0 distinguished explicitly.

**Category & sub-category:** Latent-space generative modeling; unconditional or text-conditioned diffusion.

**Originating paper/vendor/year:** Rombach and colleagues, [*High-Resolution Image Synthesis with Latent Diffusion Models*, arXiv 2021/CVPR 2022](https://arxiv.org/abs/2112.10752). Stable Diffusion v1 checkpoints are 2022 releases; [SDXL](https://arxiv.org/abs/2307.01952) and its 1.0 checkpoints are 2023 releases.

**Core mechanism:** First learn a perceptually useful image autoencoder; then train a diffusion model in its compressed latent space. For text conditioning, cross-attention injects text features into denoising. Stable Diffusion v1.4 uses a frozen text Transformer from the CLIP ViT-L/14 model package. The [SDXL comparison](https://ar5iv.labs.arxiv.org/html/2307.01952) identifies OpenCLIP ViT-H conditioning for SD 2.0/2.1 and two text encoders, CLIP ViT-L plus OpenCLIP ViT-bigG, for SDXL; these checkpoints are not text-embedding-compatible substitutes.

**Inputs/outputs and typical data types:** Training uses images and, for conditional checkpoints, paired text. Generation maps a prompt and random latent noise to an image; image-to-image tasks additionally encode an input image. SDXL base 1.0 can run alone or with its separately identified refiner.

**Strengths and limitations:** Spatial compression reduces denoising cost relative to pixel diffusion. The lossy autoencoder limits fine detail, and caption quality limits instruction following. Guidance trades diversity for conditioning strength, not truthfulness; seed variation is not calibrated uncertainty about a requested scene.

**Computational complexity / scalability notes:** With downsampling factor $`f`$, the denoiser processes approximately $`HW/f^2`$ latent positions instead of $`HW`$ pixels. This is not a guaranteed $`f^2`$ end-to-end speedup because channels, attention, decoder cost, and sampling calls also change. Classifier-free guidance commonly requires both conditional and unconditional predictions.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [Stable Diffusion v1.4 card](https://huggingface.co/CompVis/stable-diffusion-v1-4) describes training on filtered LAION image-text pairs and evaluates 10,000 COCO2017 validation prompts at 512-by-512 with 50 PLMS steps and several guidance scales. Image-caption pair -> fixed image/text encoders -> latent noise regression -> prompt-conditioned sampling produces the evaluated images. The card reports checkpoint quality/alignment comparisons and documents compositional, text-rendering, bias, and memorization limitations; no unverified curve value is transcribed here. Latent compression is the technical rationale versus pixel diffusion. Its astronaut-image example is a demonstration, not a production design-cost study; no business KPI is reported.

**Notable vendor implementations/libraries:** [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Stability AI's generative-models](https://github.com/Stability-AI/generative-models), and Hugging Face Diffusers. The [SDXL base 1.0 card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) distinguishes base-only and base-plus-refiner operation.

**Architecture diagram description:**

```text
training image -> fixed image encoder -> latent z -> add noise -> z_t
caption -> frozen text encoder(s) -> cross-attention conditioning
z_t + time + condition -> U-Net -> denoising prediction
sampling: latent noise -> iterative denoising [optional SDXL refiner]
                      -> image decoder -> RGB
```

**Activation functions used and why:** Representative Stable Diffusion U-Nets use SiLU/swish, group normalization, attention softmax, and gated feed-forward components. Text encoders have their own Transformer activations; identical activation choices should not be assumed across v1, v2, and SDXL.

**Loss function(s):** The image autoencoder combines reconstruction/perceptual objectives with appropriate latent regularization and adversarial components in the LDM formulation. The v1.4 denoiser uses latent noise-prediction MSE. Other checkpoints can change prediction parameterization; checkpoint-compatible schedulers are mandatory.

**Optimization algorithm(s):** The v1.4 card specifies AdamW, effective batch 2,048, and 10,000-step warmup to $`10^{-4}`$, then a constant rate. Its v1.4 continuation runs 225,000 steps at 512 resolution from v1.2. These are v1.4 provenance facts, not SDXL's recipe.

**Regularization techniques:** Autoencoder constraints, filtered data, EMA where configured, and dropped text conditioning support training. V1.4 drops conditioning on 10% of examples for classifier-free guidance. Safety filtering is a separate mitigation, not a guarantee from the loss.

**Backpropagation considerations:** Freeze pretrained components when specified; otherwise their gradients change the objective and memory budget. Guidance at inference is not backpropagation training. Mixed precision requires care in attention, autoencoder decoding, and variance arithmetic.

**Parameter count / scaling behavior:** The SDXL report lists approximately 860 million U-Net parameters for SD 1.4/1.5 and 2.6 billion for SDXL. These exclude text encoders, autoencoder, and optional refiner; they are not whole-pipeline counts.

**Training paradigm:** Text-conditioned checkpoints use natural-language paired supervision alongside self-supervised noise targets. They must not be called caption-free unsupervised models. Image-only LDM experiments occupy a different supervision regime.

**Hardware/parallelism considerations:** The v1.4 card reports 32-by-8 A100 GPUs and gradient accumulation. Consumer-device inference is not evidence that training required similar resources. SDXL's larger denoiser and optional second model increase inference memory and compute.

### 3.8.4 Diffusion Transformer

**Name:** Diffusion Transformer (DiT), specifically the original class-conditional latent DiT.

**Category & sub-category:** Diffusion generative modeling; Transformer denoising backbone.

**Originating paper/vendor/year:** Peebles and Xie, [*Scalable Diffusion Models with Transformers*, arXiv 2022, ICCV 2023](https://arxiv.org/abs/2212.09748). Later multimodal diffusion Transformers are related developments, not identical checkpoints.

**Core mechanism:** Patchify noisy autoencoder latents and process them with Transformer blocks instead of a U-Net. Time and class embeddings modulate adaptive layer normalization. The successful adaLN-Zero design initializes residual modulation so blocks initially behave near an identity mapping, supporting stable scaling.

**Inputs/outputs and typical data types:** A noisy image latent, noise step, and class label enter; the model predicts diffusion quantities that reconstruct the latent through sampling. A frozen image decoder maps the result to RGB. The original DiT is class-conditioned, not inherently a text-to-image model.

**Strengths and limitations:** A regular Transformer design allows systematic scaling of depth, width, and token count. Smaller patches improve granularity but increase attention cost. Class-conditional sample quality is not evidence of prompt-following ability, and generated variation is not calibrated uncertainty about real-world objects.

**Computational complexity / scalability notes:** For $`T`$ latent patches, cost per denoiser pass is $`O(L(Td^2+T^2d))`$, not just attention's quadratic term. Sampling adds $`S`$ passes and possibly additional guidance evaluations. Reducing patch width changes token count and FLOPs even at nearly fixed parameter count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** On class-conditional ImageNet at 256-by-256, image -> frozen VAE latent -> noised patches plus class -> DiT -> reverse diffusion -> decoded image tests whether a Transformer can replace a U-Net. The [paper](https://arxiv.org/html/2212.09748v2) reports **FID-50K 2.27** for **DiT-XL/2**, after seven million training steps, with classifier-free guidance scale 1.5 and 250 DDPM sampling steps, evaluated using ADM's TensorFlow suite. Unguided generation and the 400,000-step scaling experiments have different results. Architectural regularity and scaling are the technical fit versus U-Nets; the experiment is not an industrial image service or a public business KPI.

**Notable vendor implementations/libraries:** The authors' [official PyTorch DiT repository](https://github.com/facebookresearch/DiT) provides weights ported from JAX and distributed training code. It documents small numerical differences across frameworks/precision rather than promising bitwise reproduction.

**Architecture diagram description:**

```text
image latent + noise -> patch embedding + position embedding
class + timestep -> conditioning MLP -> adaLN-Zero controls
patch sequence -> repeated DiT attention/MLP blocks
               -> linear patch output -> unpatchify -> noise/variance prediction
denoised latent -> fixed VAE decoder -> image
```

**Activation functions used and why:** GELU in Transformer MLPs, softmax attention, and SiLU in conditioning embeddings provide nonlinear feature mixing. Adaptive layer normalization injects conditioning; the final prediction is real-valued.

**Loss function(s):** A diffusion noise-prediction objective with a learned-variance component following the referenced diffusion formulation. Classifier-free conditioning dropout trains conditional and null-condition predictions; this does not remove the supervision supplied by class labels.

**Optimization algorithm(s):** The original recipe uses AdamW at a constant $`10^{-4}`$, batch 256, **no weight decay and no learning-rate warmup**. EMA decay is 0.9999. AdamW's name must not be taken as evidence that nonzero decay was used.

**Regularization techniques:** Horizontal flips and EMA are reported; zero initialization and adaLN-Zero improve optimization rather than constituting proof of generalization. The paper did not require the usual strong ViT regularization recipe.

**Backpropagation considerations:** Random-time denoising avoids differentiating an entire sampling chain. Zero-initialized residual modulation controls early gradient flow. Long patch sequences increase attention activations; activation checkpointing trades recomputation for memory.

**Parameter count / scaling behavior:** The reported DiT-XL/2 adaLN-Zero denoiser has approximately 675 million parameters, excluding the VAE. At 256 resolution its forward work is about 118.6 GFLOPs in the paper's accounting; FLOPs and parameter counts are not interchangeable.

**Training paradigm:** Supervised class-conditioned generation with synthetic noise targets. This entry sits beside unconditional diffusion for architectural comparison, not because ImageNet class conditioning is unsupervised.

**Hardware/parallelism considerations:** The paper used JAX on TPU-v3 pods; the official implementation also supports GPU distributed data parallelism. A cited v3-256 configuration and later A100 reproductions are different hardware/protocol records, not generic requirements.

### 3.8.5 DALL-E 1

**Name:** DALL-E 1, the original discrete-VAE/autoregressive text-to-image system.

**Category & sub-category:** Conditional image generation; discrete image tokenization plus autoregressive sequence modeling. It is not a diffusion generator.

**Originating paper/vendor/year:** Ramesh and colleagues, OpenAI, [*Zero-Shot Text-to-Image Generation*, ICML 2021](https://proceedings.mlr.press/v139/ramesh21a.html).

**Core mechanism:** Train a discrete VAE to convert images into a grid of categorical tokens. Freeze the tokenizer and train a decoder-only Transformer on concatenated caption and image-token streams. At inference, a caption supplies the prefix and the Transformer generates image tokens, which the discrete-VAE decoder renders.

**Inputs/outputs and typical data types:** Internet image-caption pairs train the system; a text prompt yields image candidates. The [paper](https://ar5iv.labs.arxiv.org/html/2102.12092) uses up to 256 BPE text tokens and 1,024 image tokens for a 32-by-32 token grid, with an 8,192-entry image vocabulary.

**Strengths and limitations:** A shared token sequence enables broad natural-language conditioning without designing a separate task-specific generator for every class. The discrete bottleneck loses detail, and autoregressive sampling is sequential. Sampling temperature and contrastive reranking affect diversity; neither is a calibrated confidence estimate for compositional correctness.

**Computational complexity / scalability notes:** Tokenizer training is separate from Transformer training. The Transformer uses structured sparse image attention, so a universal dense $`T^2`$ estimate is inappropriate for its actual masks. Cached generation still proceeds one image token at a time. Generating and reranking many candidates multiplies inference cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original study](https://ar5iv.labs.arxiv.org/html/2102.12092) trained on approximately 250 million image-text pairs and evaluated text-to-image transfer on MS-COCO without training directly on its task split. Caption -> text tokens -> autoregressive image codes -> dVAE decoder -> contrastive candidate ranking gives the evaluated output. The paper reports favorable human comparisons and diverse compositional examples; its illustrated comparison chooses the best of 512 candidates using the contrastive model. That selection budget matters, so no unmatched single-sample FID or preference claim is substituted here. Its technical fit versus a class-specific GAN is broader language conditioning. Zero-shot evaluation does not mean absence of paired supervision or guaranteed absence of all source-image overlap. No public business KPI is established.

**Notable vendor implementations/libraries:** OpenAI released the [DALL-E discrete VAE](https://github.com/openai/DALL-E), not the complete original large autoregressive model. Community implementations and later similarly named systems are not the proprietary trained checkpoint.

**Architecture diagram description:**

```text
training image -> dVAE encoder -> 32x32 categorical codes
caption -> BPE text tokens ------+
image codes --------------------+-> causal/sparse Transformer -> next-token loss
generation: caption prefix -> sampled image codes -> dVAE decoder -> image
                                                      -> optional candidate reranking
```

**Activation functions used and why:** The [released dVAE encoder](https://github.com/openai/DALL-E/blob/master/dall_e/encoder.py) and decoder use ReLU hidden activations. Categorical softmax outputs represent image codes and next-token distributions. Gumbel-softmax supplies a differentiable tokenizer-training relaxation, distinct from nearest-code straight-through VQ-VAE training. These tokenizer details do not establish every hidden activation in the unreleased large Transformer.

**Loss function(s):** The tokenizer optimizes a relaxed variational objective with a log-Laplace image likelihood. Stage two uses autoregressive cross-entropy, with separately normalized text and image losses weighted $`1/8`$ and $`7/8`$, respectively, in the paper.

**Optimization algorithm(s):** Both stages use Adam with exponentially averaged iterates. The dVAE uses annealed relaxation temperature and step size; the paper identifies temperature annealing to $`1/16`$. These stage-specific schedules should not be confused with diffusion noise scheduling, which this generator does not use.

**Regularization techniques:** The discrete bottleneck, structured attention, and training-data processing constrain learning. The contrastive reranker changes output selection rather than regularizing the trained autoregressive likelihood.

**Backpropagation considerations:** Relaxed categorical gradients train the tokenizer; Transformer training uses fixed image tokens. The paper documents per-residual-block gradient scaling and higher-precision residual paths to address low-precision underflow/overflow at large scale.

**Parameter count / scaling behavior:** The autoregressive Transformer has 12 billion parameters and 64 self-attention layers in the reported system. Tokenizer and reranker are separate components; 12 billion is not an inventory of every serving component.

**Training paradigm:** Image-only tokenization followed by naturally paired image-text supervised modeling. Predicting the next token supplies a self-supervised objective over the paired sequence, but captions still provide semantic supervision.

**Hardware/parallelism considerations:** The paper describes parameter sharding across eight GPUs within each machine, all-gather before block computation, and reduce-scatter for gradients. Training scale and multi-candidate decoding require different resource accounting.

### 3.8.6 DALL-E 2

**Name:** DALL-E 2, represented by the published unCLIP generative stack.

**Category & sub-category:** Conditional image generation; CLIP-latent prior plus diffusion decoding and super-resolution.

**Originating paper/vendor/year:** Ramesh and colleagues, OpenAI, [*Hierarchical Text-Conditional Image Generation with CLIP Latents*, 2022](https://arxiv.org/abs/2204.06125).

**Core mechanism:** Learn a prior that predicts a CLIP image embedding from text, then a diffusion decoder that reconstructs an image conditioned on that embedding. The study compares autoregressive and diffusion priors; the decoder is diffusion-based in both cases. Two diffusion upsamplers increase image resolution. This is not DALL-E 1's discrete-image-token Transformer.

**Inputs/outputs and typical data types:** Paired captions and images train the components. Text -> image embedding -> base image -> upsampled image is the text-to-image route. Encoding an existing image supplies an alternative conditioning embedding for variations.

**Strengths and limitations:** CLIP embeddings provide a semantic intermediate representation and support image variations. They discard some spatial detail, and embeddings do not uniquely specify a scene. Prior and decoder errors accumulate; the spread of generated variations is not a calibrated posterior over the original image or a guarantee of textual correctness.

**Computational complexity / scalability notes:** Inference cost includes prior sampling, base decoder sampling, and both upsamplers; no single network count describes it. Spatial upsamplers can have high compute despite smaller parameter counts. Guidance and candidate selection alter both cost and sample distribution.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [unCLIP study](https://ar5iv.labs.arxiv.org/html/2204.06125) evaluates zero-shot MS-COCO caption-conditioned image generation and image variations. Caption -> prior-predicted CLIP image embedding -> 64-by-64 decoder -> 256 and 1,024 upsamplers yields an image for human/metric evaluation. The paper reports favorable zero-shot image-quality results and compares the two priors, choosing diffusion as the preferred cost/quality trade-off. That documented choice is stronger evidence than attributing a generic preference for diffusion to all commercial users. The generative stack uses the approximately 250-million-image DALL-E dataset; its CLIP encoder training uses a different data mixture. No audited public business KPI is reported, and the research recipe must not be assumed to enumerate every later production revision.

**Notable vendor implementations/libraries:** OpenAI operated the proprietary DALL-E 2 service. Open-source unCLIP-style implementations reproduce ideas, not the original production weights; a library interface does not establish checkpoint equivalence.

**Architecture diagram description:**

```text
caption -> text features -> prior [AR OR diffusion] -> CLIP image embedding
                                                      |
noise + embedding [+ text] -> diffusion decoder -> 64x64 image
                         -> diffusion upsampler -> 256x256
                         -> diffusion upsampler -> 1024x1024
```

**Activation functions used and why:** The published decoder inherits GLIDE's residual/attention architecture; its [U-Net implementation](https://github.com/openai/glide-text2im/blob/main/glide_text2im/unet.py) uses smooth SiLU nonlinearities and attention softmax. The diffusion prior has continuous embedding output; the AR prior requires discrete prediction heads. Undisclosed serving changes cannot be inferred from these components.

**Loss function(s):** The diffusion prior predicts the clean CLIP image embedding with a squared denoising objective. The base decoder uses the referenced diffusion noise/learned-variance training; upsamplers learn conditional denoising. A frozen CLIP similarity is conditioning information, not the entire generator loss.

**Optimization algorithm(s):** The [paper's hyperparameter table](https://ar5iv.labs.arxiv.org/html/2204.06125) specifies Adam settings, including diffusion-prior rate $`1.1\times10^{-4}`$ and base-decoder rate $`1.2\times10^{-4}`$, with stage-specific EMA. It uses cosine noising for the prior/base/first upsampler and linear noising for the final upsampler. The table's rates do not fully specify every learning-rate transition in a commercial training run.

**Regularization techniques:** Conditioning dropout supports guidance; image corruption improves upsampler robustness. The prior has weight decay, while dropout and EMA differ by stage. CLIP remains frozen during prior/decoder training.

**Backpropagation considerations:** Each stage has a distinct objective; gradients do not jointly flow through a frozen CLIP encoder and the complete sampled pipeline. Numerical scaling of continuous CLIP embeddings is important. Learned variance and multiple conditioning inputs complicate mixed-precision stability.

**Parameter count / scaling behavior:** The paper lists roughly 1 billion parameters for either prior, 3.5 billion for the base decoder, and 700/300 million for the upsamplers. These component counts are not a single complete DALL-E 2 service count.

**Training paradigm:** Naturally paired text-image supervision plus diffusion targets. CLIP pretraining and each generative stage are distinct; the larger CLIP data mixture must not be claimed as the decoder's training dataset.

**Hardware/parallelism considerations:** Large components motivate accelerator data/model parallelism and checkpointing. The checked paper does not provide a complete production hardware inventory; do not invent device counts or infer training cost from API pricing.

### 3.8.7 DALL-E 3

**Name:** DALL-E 3, a proprietary text-to-image system whose published research emphasizes caption quality.

**Category & sub-category:** Conditional image generation; descriptive recaptioning and improved prompt following. It is included in the image-generation family without asserting a disclosed low-level denoiser architecture.

**Originating paper/vendor/year:** Betker and colleagues, OpenAI with Microsoft collaborators, [*Improving Image Generation with Better Captions*, 2023](https://cdn.openai.com/papers/dall-e-3.pdf).

**Core mechanism:** The report trains an image captioner to create more descriptive image-text training pairs and investigates their effect on text-to-image learning. It discusses text-to-image diffusion models, but explicitly does not cover DALL-E 3's complete training or implementation details. Therefore a U-Net, DiT, latent codec, layer layout, or DALL-E 2-style prior cannot be supplied as a verified DALL-E 3 architecture.

**Inputs/outputs and typical data types:** Training uses images with original or generated descriptions; generation takes a text request and returns images. Prompt expansion is a separate possible language-model stage, not proof that the image generator itself is an autoregressive language model.

**Strengths and limitations:** Detailed captions can expose object attributes and relationships omitted by noisy alt-text. Captioners can also hallucinate, omit information, and impose stylistic biases. Better prompt following is not image factuality or correctness on every prompt. Candidate variation and any safety decision are not calibrated user-facing probabilities of correctness.

**Computational complexity / scalability notes:** Dataset recaptioning adds offline image-encoding and language-generation work. Actual generator inference/training FLOPs cannot be calculated from the published report because its full architecture and recipe are absent. Any numerical cost estimate would require additional disclosed assumptions, not a borrowed DALL-E 2 parameter count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [report](https://cdn.openai.com/papers/dall-e-3.pdf) studies instruction following using DrawBench, COCO-derived captions, and compositional evaluation. Image plus noisy caption -> descriptive captioning -> text-to-image training -> generated images -> human and automated evaluation tests whether richer pairing improves adherence. It reports improved prompt-following behavior and favorable comparisons while retaining limitations. The [official evaluation repository](https://github.com/openai/dalle3-eval-samples) releases four images per DrawBench prompt and distinguishes original from expanded prompts. Those are benchmark samples, not curated product demonstrations; the repository explicitly makes that distinction. The method fit is better supervision rather than an asserted undisclosed architectural advantage. No public audited business KPI is provided.

**Notable vendor implementations/libraries:** OpenAI's proprietary DALL-E 3 interfaces and [official evaluation artifacts](https://github.com/openai/dalle3-eval-samples). Evaluation samples are not model weights or a reproducible training implementation. Service availability and product routing can change independently of this historical model description.

**Architecture diagram description:**

```text
DISCLOSED TRAINING-DATA PATH:
image + existing caption -> image-captioning system -> richer paired description
paired images/descriptions -> text-to-image training [full generator undisclosed]

PRODUCT-LEVEL GENERATION PATH:
text request -> [optional prompt expansion] -> proprietary image generator -> image
No verified internal U-Net/DiT/codec/prior diagram is publicly supplied here.
```

**Activation functions used and why:** Full generator activation and normalization choices are not publicly disclosed in the cited report. The captioner's language objective uses token probabilities, but that does not establish hidden activations throughout DALL-E 3. Assigning GELU, SiLU, or a particular normalization to undisclosed blocks would be invention.

**Loss function(s):** The report describes image-conditioned language likelihood and joint contrastive/language pretraining for its captioner. Its exact final image-generator loss, noise-prediction parameterization, auxiliary penalties, and weights are not sufficiently disclosed to write a verified complete objective.

**Optimization algorithm(s):** DALL-E 3 generator optimizer, learning-rate schedule, gradient-clipping thresholds, and detailed noise/sampling schedules are not publicly specified by this report. DALL-E 2's Adam table is not evidence for them.

**Regularization techniques:** Mixing original and synthetic captions addresses captioner-induced distributional regularities. The paper's caption ablations include 95% synthetic-caption conditions; that experimental fraction should not be promoted into a fully disclosed final production recipe. Exact weight decay, dropout, and normalization remain undisclosed.

**Backpropagation considerations:** Offline generated captions are discrete training data, not evidence of end-to-end gradients through a sampled captioner into the image generator. The report does not establish final-generator gradient estimators, precision handling, or distributed synchronization.

**Parameter count / scaling behavior:** Not publicly disclosed for the full DALL-E 3 generator in this source. Resolution, output quality, or API latency cannot reliably identify parameter count, expert count, or active compute.

**Training paradigm:** Conditional, naturally paired and synthetically enriched language supervision. Caption generation does not erase supervision; the captioner itself has a training history. Public evaluation and prompt expansion are separate from the unknown full image-training recipe.

**Hardware/parallelism considerations:** Exact production training accelerators, fleet size, sharding, and utilization are undisclosed. Large-scale captioning/generation generally benefits from accelerators, but that engineering observation is not a sourced DALL-E 3 hardware specification.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| DDPM | Images and other continuously corrupted data | Simple denoising training | Original iterative sampling is slow | Unconditional CIFAR-10 generation |
| Score-based SDE models | Continuous data and modeled inverse problems | Unified stochastic/ODE formulation | Solver and score approximation errors | NCSN++ continuous deep on CIFAR-10 |
| Latent diffusion / Stable Diffusion | Images with optional text/image conditions | Denoising in a compressed spatial space | Lossy codec and checkpoint-specific behavior | Stable Diffusion v1.4 COCO2017 evaluation |
| DiT | Latent image patches with conditions | Systematic Transformer scaling | Token-count cost and supervised class conditioning | DiT-XL/2 ImageNet generation |
| DALL-E 1 | Paired images and natural-language captions | Autoregressive image-token generation | Sequential decoding and tokenizer loss | Zero-shot MS-COCO text-to-image research |
| DALL-E 2 | Paired images/text; semantic image embeddings | Prior plus diffusion decoding and variations | Multi-stage error and compute | unCLIP caption/image-variation evaluation |
| DALL-E 3 | Images with descriptive paired captions | Improved prompt following through caption quality | Proprietary internal recipe and residual errors | DrawBench and compositional evaluation |

## 3.9 Flows and autoregression

Normalizing flows use invertibility and change of variables to evaluate a continuous density. Autoregressive models factor a joint probability into conditionals. Both permit likelihood-based training, but have different sampling costs. A continuous density evaluated on dequantized pixels is not automatically an exact discrete image probability, and high likelihood alone is not a reliable out-of-distribution detector.

### 3.9.1 RealNVP

**Name:** Real-valued non-volume-preserving transformation model (RealNVP).

**Category & sub-category:** Unsupervised density estimation; neural normalizing flows with affine coupling.

**Originating paper/vendor/year:** Dinh, Sohl-Dickstein, and Bengio, [*Density Estimation using Real NVP*, arXiv 2016, ICLR 2017](https://arxiv.org/abs/1605.08803), extending earlier additive-coupling flow ideas.

**Core mechanism:** Split an input into two parts. Keep one unchanged and transform the other using scale and translation predicted from the first: $`y_a=x_a,\ y_b=x_b\odot\exp s(x_a)+t(x_a)`$. The Jacobian is triangular, so its log determinant is the sum of predicted log scales. Alternating masks and a multiscale design let all coordinates eventually interact.

**Inputs/outputs and typical data types:** Continuous vectors or suitably preprocessed/dequantized images enter. Outputs are latent values, continuous log densities, and generated observations obtained by reversing the transformations. The internal scale/translation networks need not themselves be invertible.

**Strengths and limitations:** Exact latent inversion and tractable continuous-density evaluation distinguish it from GANs and approximate-posterior VAEs. The coupling structure limits flexibility per layer, often requiring many layers. Likelihood can reward low-level statistics unrelated to semantic typicality; it is not an automatic calibrated anomaly probability.

**Computational complexity / scalability notes:** Coupling-network evaluations dominate cost; the determinant calculation is linear in transformed dimensions rather than cubic in full input dimension. Layers remain sequential, but each layer transforms many coordinates in parallel. Full dense Jacobian materialization is unnecessary.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [RealNVP paper](https://ar5iv.labs.arxiv.org/html/1605.08803) studies CIFAR-10, downsampled ImageNet, LSUN, and CelebA density modeling. Image -> preprocessing and invertible coupling layers -> Gaussian latent and accumulated log determinant -> likelihood evaluation or inverse sampling tests a model that supports both encoding and generation. It reports **3.49 bits/dimension on CIFAR-10's test set** under its image-likelihood protocol, worse than the listed PixelRNN 3.00 despite parallelizable synthesis. Thus its technical fit is tractability and fast inverse sampling, not winning every likelihood benchmark. Continuous preprocessing/dequantization conventions must be retained when interpreting the number. No deployed compression service or commercial saving is established.

**Notable vendor implementations/libraries:** TensorFlow Probability bijectors and Pyro coupling transforms provide RealNVP-style building blocks. A generic affine-coupling distribution is not necessarily the original paper's complete multiscale network.

**Architecture diagram description:**

```text
x -> split [x_a, x_b]
x_a -> neural s,t -> scale/shift x_b -> concatenate
  -> alternate mask / squeeze / additional coupling blocks
  -> latent z with simple prior
sampling: prior z -> inverse coupling blocks -> x
```

**Activation functions used and why:** A conventional neural instantiation uses ReLU residual subnetworks for scale/translation; those subnetworks need not be invertible. The coupling transform uses an exponential positive scale, often with controlled log-scale for stability, and an unbounded translation. Rectification occurs inside the parameter networks, not as a noninvertible replacement for the overall data transformation.

**Loss function(s):** Negative change-of-variables log likelihood: $`-\log p_Z(f_\theta(x))-\log|\det J_{f_\theta}(x)|`$, including any preprocessing Jacobian. Uniform dequantization supplies a bound-related objective for discrete pixels rather than making their probability equal to a point density.

**Optimization algorithm(s):** The paper uses Adam with its stated default hyperparameters; it does not prescribe a universal learning-rate-decay schedule. A reproduction must record the actual optimizer settings and stopping rule rather than infer them from the flow equations.

**Regularization techniques:** The original method uses residual subnetworks, normalization, and an L2 penalty on weight-scale parameters. Invertible normalization outside the subnetworks must contribute its own log determinant; otherwise the reported density is wrong.

**Backpropagation considerations:** Differentiate both prior density and log determinant. Excessive log scales cause overflow or ill-conditioned inversion. Running-statistic normalization must be used consistently at evaluation; batch-dependent transformations require careful density interpretation.

**Parameter count / scaling behavior:** Coupling subnetworks and multiscale depth determine $`p`$; invertibility does not mean few parameters. Alternating masks increase expressivity without requiring every subnetwork to be a square invertible matrix.

**Training paradigm:** Unsupervised maximum-likelihood learning for the representative image model. Conditioning scale/translation on labels yields a supervised conditional flow.

**Hardware/parallelism considerations:** GPUs parallelize coupling-network convolutions across positions and examples. Reversible computation can reduce activation storage, but numerical reversibility and recomputation cost must be checked rather than assuming memory is automatically constant.

### 3.9.2 Glow

**Name:** Glow, a generative flow with invertible one-by-one convolutions.

**Category & sub-category:** Unsupervised density estimation; multiscale neural normalizing flows.

**Originating paper/vendor/year:** Kingma and Dhariwal, OpenAI, [*Glow: Generative Flow with Invertible 1x1 Convolutions*, NeurIPS 2018](https://arxiv.org/abs/1807.03039).

**Core mechanism:** Each flow step combines activation normalization, learned invertible channel mixing, and affine coupling. The one-by-one convolution replaces a fixed channel permutation with a learned invertible matrix. Squeeze and split operations build a multiscale latent representation, retaining exact mathematical inversion of the modeled continuous transformation.

**Inputs/outputs and typical data types:** Images become multiscale latents and log densities; sampled latents become images. Latent interpolation or manipulation is possible because an image can be mapped directly to its corresponding latent under the flow.

**Strengths and limitations:** Learned channel mixing increases expressivity while retaining tractable density and parallel per-layer synthesis. Training can be memory-intensive, and excellent density need not mean excellent perceptual semantics. Low-temperature latent sampling favors visual quality but no longer samples the original fitted distribution; it is not uncertainty calibration.

**Computational complexity / scalability notes:** Applying an invertible channel matrix at $`HW`$ positions costs $`O(HWc^2)`$. Its naive determinant is $`O(c^3)`$, whereas an LU parameterization makes the log-determinant sum $`O(c)`$; it does **not** make the matrix application $`O(c)`$. Coupling convolutions and multiscale depth add their own cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [Glow paper](https://ar5iv.labs.arxiv.org/html/1807.03039) reports **3.35 bits/dimension on CIFAR-10**, compared with RealNVP's reported 3.49 in its benchmark table. Image -> ActNorm/channel mixing/coupling -> latent and density -> inverse sample evaluates likelihood and generation together. The technical rationale versus fixed permutations is learnable mixing; versus a GAN, it is a tractable density and direct inversion. Its separate CelebA-HQ experiment uses 256-resolution, **five-bit** images for visual quality and latent manipulation. Those faces must not be used to claim the same eight-bit likelihood/compression result as CIFAR-10. These are research results, not audited production benefits.

**Notable vendor implementations/libraries:** The [official OpenAI Glow repository](https://github.com/openai/glow) supplies the TensorFlow implementation. Flow libraries can reproduce individual operations, but different dequantization, bit depth, and coupling settings change the model.

**Architecture diagram description:**

```text
image -> squeeze -> [ActNorm -> invertible 1x1 conv -> affine coupling] repeated
      -> split off latent portion -> repeat at coarser scale -> remaining latent
all latent portions + prior log densities -> total image log density
sampling reverses every step
```

**Activation functions used and why:** The checked coupling subnetworks use ReLU; the implementation's affine scale uses a shifted sigmoid for controlled positive scale. ActNorm is a trainable affine transformation initialized from data, not batch normalization recomputed on every minibatch.

**Loss function(s):** Exact continuous change-of-variables negative log likelihood, including ActNorm, channel-mixing, and coupling determinants and the multiscale prior. Quantized images still require the paper/implementation's preprocessing and noise convention.

**Optimization algorithm(s):** The [official training defaults](https://github.com/openai/glow/blob/master/train.py) use **Adamax**, learning rate 0.001, with linear warmup over ten epochs. Adam is also an implementation option. These checked defaults should not be casually reported as mandatory Adam for every Glow run.

**Regularization techniques:** Data-dependent ActNorm initialization and zero-initialized final coupling convolutions improve early conditioning. Dequantization prevents fitting pathological point densities on a discrete lattice. Sampling-temperature reduction is an inference choice, not training regularization.

**Backpropagation considerations:** All determinant contributions must remain in the gradient graph. Invertible matrices can become poorly conditioned; LU parameterization helps determinant computation but does not guarantee numerical conditioning. Mixed-precision log determinants deserve care.

**Parameter count / scaling behavior:** Flow levels, steps per level, and coupling-network width determine size. A one-by-one mixing matrix contributes $`c^2`$ parameters, in addition to the usually larger coupling networks; there is no single family-wide Glow count.

**Training paradigm:** Unsupervised likelihood fitting in the cited benchmarks. Attribute-oriented latent manipulations may use labeled analyses or selected examples and should not be confused with an entirely unsupervised downstream task.

**Hardware/parallelism considerations:** The official implementation supports distributed accelerator training and gradient checkpointing. Invertibility enables recomputation strategies, but storing or reconstructing multiscale activations remains an engineering trade-off.

### 3.9.3 PixelCNN

**Name:** PixelCNN, distinguishing the original masked-convolution model from Gated PixelCNN.

**Category & sub-category:** Autoregressive generative modeling; categorical image likelihood.

**Originating paper/vendor/year:** Van den Oord, Kalchbrenner, and Kavukcuoglu introduced PixelCNN in [*Pixel Recurrent Neural Networks*, ICML 2016](https://arxiv.org/abs/1601.06759). Van den Oord and colleagues introduced the gated/conditional extension in [*Conditional Image Generation with PixelCNN Decoders*, NeurIPS 2016](https://arxiv.org/abs/1606.05328).

**Core mechanism:** Factorize the image probability into an ordered product of pixel/channel conditionals. Masks block future positions and disallowed same-pixel channels. During training the entire known image supplies context, permitting parallel convolutions. During generation each newly sampled value becomes context for later values. Gated PixelCNN separates vertical and horizontal streams to address the original receptive-field blind spot.

**Inputs/outputs and typical data types:** Discrete image intensities, optionally with class or embedding conditions, enter. Outputs are categorical distributions over the next intensity and, by iterative sampling, images. An image density can be evaluated directly by summing conditional log probabilities.

**Strengths and limitations:** Avoids latent-posterior approximations and models multimodal pixel conditionals. Sampling remains sequential even though training convolutions are parallel. Finite receptive fields can miss global structure. A confident next-pixel prediction is not confidence that the overall scene is semantically correct or in distribution.

**Computational complexity / scalability notes:** A teacher-forced batch costs one masked-network pass over the full grid, approximately the sum of convolutional costs. Naively rerunning the full grid for each sampled channel is expensive; cached implementations reduce redundant work but not the causal ordering. PixelCNN is not a recurrent network merely because decoding is sequential.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [Gated PixelCNN study](https://ar5iv.labs.arxiv.org/html/1606.05328) evaluates CIFAR-10 discrete image likelihood. Known pixels -> masked vertical/horizontal convolutions -> conditional 256-way probabilities -> accumulated test negative log likelihood or sequential samples is the worked pipeline. It reports **3.03 bits/dimension**, versus 3.14 for the earlier PixelCNN and 3.00 for PixelRNN in the comparison table. The technical fit is parallel convolutional training with near-PixelRNN likelihood, not faster fully parallel generation. Conditional ImageNet experiments additionally provide class labels. No commercial compression rate, deployment latency, or business KPI is asserted from these benchmark likelihoods.

**Notable vendor implementations/libraries:** PixelCNN-style research code and TensorFlow Probability distributions support autoregressive image models. PixelCNN++, which changes the likelihood and architecture, must not inherit the original/gated model's results merely because the names are similar.

**Architecture diagram description:**

```text
image -> mask-A convolution [exclude current/future target]
      -> mask-B residual layers [allowed earlier-channel context]
      -> pointwise heads -> 256-way distribution per RGB channel
gated variant: vertical stream -> horizontal stream -> gated residual heads
sampling: fill pixels/channels in causal order
```

**Activation functions used and why:** Original PixelCNN uses ReLU layers and categorical softmax outputs; Gated PixelCNN uses a tanh filter multiplied by a sigmoid gate. Binary MNIST can use Bernoulli/sigmoid outputs rather than 256 intensity classes.

**Loss function(s):** Exact autoregressive negative log likelihood $`-\sum_i\log p_\theta(x_i\mid x_{<i})`$, optionally conditioned on $`c`$. In the cited categorical model, this is discrete cross-entropy, not a Gaussian reconstruction proxy.

**Optimization algorithm(s):** The original PixelRNN/PixelCNN paper uses RMSProp, with manually selected dataset-specific learning-rate schedules. It does not give one universal schedule for all later gated/conditional implementations.

**Regularization techniques:** Causal masks constrain the probability factorization rather than randomly dropping units. The original experiments use small batches for smaller datasets and no image augmentation beyond input scaling/centering. Residual connections aid optimization.

**Backpropagation considerations:** Incorrect masks leak the target and invalidate likelihood, often yielding deceptively good training loss. Teacher forcing avoids differentiating through categorical sampling; test-time exposure to model-generated context can still reveal accumulated errors.

**Parameter count / scaling behavior:** Layer count, hidden channels, and output alphabet size determine parameters. The original architecture describes fifteen convolutional layers; the larger gated ImageNet configuration differs. Per-position output probabilities also create substantial activation storage.

**Training paradigm:** Self-supervised next-pixel prediction in unconditional models. Gated class-conditioned generation is supervised conditional modeling; image-embedding conditioning inherits the embedding model's supervision.

**Hardware/parallelism considerations:** Training is GPU-friendly and data parallel. The gated paper reports distributed GPU experiments, but sequence-dependent image sampling remains latency-bound; more GPUs mainly help generate more independent images.

### 3.9.4 WaveNet

**Name:** WaveNet, specifically the original autoregressive dilated-convolution waveform model; later production revisions are identified separately.

**Category & sub-category:** Autoregressive audio generation; raw-waveform density modeling and conditional speech synthesis.

**Originating paper/vendor/year:** Van den Oord and colleagues, DeepMind, [*WaveNet: A Generative Model for Raw Audio*, 2016](https://arxiv.org/abs/1609.03499).

**Core mechanism:** Dilated causal convolutions predict the distribution of the next audio sample from previous samples. Increasing dilation expands temporal receptive fields without equally deep stacks of adjacent-sample convolutions. Local linguistic/acoustic conditions or global speaker identity can control the waveform. Unconditional audio modeling and text-conditioned speech have different supervision.

**Inputs/outputs and typical data types:** Training takes waveform sequences and optional aligned linguistic/acoustic features. Outputs are next-sample probabilities and synthesized audio. The original commonly described head quantizes audio with eight-bit mu-law companding and predicts 256 categories.

**Strengths and limitations:** Direct waveform modeling avoids some assumptions of hand-designed vocoders and allows rich local structure. Autoregressive generation at audio sample rates is expensive. Quantization, receptive field, and conditioning quality matter; a likely waveform continuation is not calibrated confidence that spoken content is correct.

**Computational complexity / scalability notes:** Teacher-forced training costs approximately $`O(BT\sum_l k_lc_l^2)`$ for sequence length $`T`$ and same-width causal layers. Cached inference removes repeated history computation but still emits samples in sequence. Dilation expands receptive field without increasing kernel parameter count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Sourced application.** DeepMind's [October 4, 2017 announcement](https://deepmind.google/blog/wavenet-launches-in-the-google-assistant/) states that an **updated WaveNet** generated Google Assistant's US English and Japanese voices; it explicitly calls the original model too computationally intensive for consumer products. Requested response text -> linguistic/acoustic conditioning -> waveform generator -> audio delivered by Assistant is the application pipeline. Direct learned waveform synthesis addresses naturalness limitations of concatenative/parametric alternatives; the announcement documents the adoption, not a public cost-saving KPI. Separately, the [2016 research paper](https://ar5iv.labs.arxiv.org/html/1609.03499) reports English five-point listener MOS 4.21 for its linguistic-plus-F0 WaveNet versus 3.86 for its HMM-driven concatenative baseline. Those are original research-condition scores, not measurements of the updated 2017 production system.

**Notable vendor implementations/libraries:** DeepMind/Google's research and documented Assistant application; community WaveNet implementations. Distilled/parallel WaveNet and later neural vocoders are distinct architectures or training procedures, not automatically the original sampler.

**Architecture diagram description:**

```text
past mu-law audio -> causal convolution
  -> repeated dilated residual blocks:
       tanh(filter + condition) * sigmoid(gate + condition)
       -> residual path + skip path
summed skip outputs -> ReLU / pointwise layers -> 256-way next-sample softmax
```

**Activation functions used and why:** Tanh/sigmoid gating selects content and its transmission; residual and skip paths preserve gradients across dilation stacks. ReLU output processing and softmax produce categorical sample probabilities in the original formulation.

**Loss function(s):** Autoregressive cross-entropy $`-\sum_t\log p_\theta(x_t\mid x_{<t},c)`$. The quantized waveform target is not a spectrogram-MSE objective. Later continuous-output and distillation variants require their own losses.

**Optimization algorithm(s):** End-to-end stochastic gradient optimization trains the neural likelihood. A complete original optimizer/learning-rate schedule is not specified in the checked research report, and the production announcement does not disclose one. Adam with warmup/decay would be a proposed implementation choice, not a verified Google recipe.

**Regularization techniques:** Causality, finite receptive fields, and conditioning structure constrain learning; residual connections stabilize optimization. Do not infer a particular dropout or weight-decay value from the production product name.

**Backpropagation considerations:** Training parallelizes over known targets with teacher forcing; gradients do not traverse sampled categorical outputs. Clip unstable gradients where validated. Check conditioning alignment and receptive-field padding carefully to avoid future-audio leakage.

**Parameter count / scaling behavior:** Residual channels, dilation cycles, conditioning networks, and output heads determine count. Higher audio sample rates increase sequence work without necessarily changing weights. Production model counts are not disclosed by the cited announcement.

**Training paradigm:** Self-supervised next-sample prediction for unconditioned audio; paired linguistic/acoustic supervision for TTS. Its placement here reflects the autoregressive mechanism, not a claim that Assistant speech synthesis is unsupervised.

**Hardware/parallelism considerations:** GPUs efficiently train convolutional sequences; cached autoregressive inference still has strict latency constraints. The documented production revision was engineered to overcome prototype speed limits; its complete hardware and parallelization recipe is not established here.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| RealNVP | Continuous vectors and dequantized images | Tractable density and exact latent inversion | Coupling constraints and semantic likelihood failures | CIFAR-10 likelihood benchmark |
| Glow | Multiscale natural images | Learned invertible channel mixing | Memory cost and precision-sensitive inversion | CIFAR-10 and five-bit CelebA-HQ research |
| PixelCNN | Discrete image intensities | Explicit autoregressive pixel probability | Sequential sampling despite parallel training | Gated PixelCNN CIFAR-10 evaluation |
| WaveNet | Raw audio with optional aligned conditions | Direct learned waveform synthesis | High-rate sequential inference in original model | Updated WaveNet in Google Assistant, 2017 |

## 3.10 Representation pretraining

These methods train representations rather than necessarily producing new observations. The pretext objective, feature extractor, projection head, and downstream evaluator are separate objects. Linear probing freezes the encoder and trains a labeled linear classifier; fine-tuning updates the encoder; nearest-neighbor evaluation uses a labeled reference set. Their scores are not interchangeable.

Comparisons below retain the paper's backbone and training regime. A 200-epoch ResNet-50 contrastive model, a 1,000-epoch wider network, and a fine-tuned ViT-H are not controlled comparisons of losses alone. CLIP is included by an explicit cross-modal editorial convention and uses genuine paired language supervision.

The final two entries cover foundational static word embeddings. Word2Vec and GloVe are shallow log-linear/log-bilinear models: using the neural-field template for their trainable embedding realizations does not imply deep hidden layers. They provide a useful contrast with the contextual token representations in [foundation models](06-foundation-models.md).

### 3.10.1 SimCLR

**Name:** Simple Framework for Contrastive Learning of Visual Representations (SimCLR), original 2020 formulation.

**Category & sub-category:** Self-supervised visual representation learning; augmentation-based contrastive pretraining.

**Originating paper/vendor/year:** Chen, Kornblith, Norouzi, and Hinton, Google Research, [*A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020](https://arxiv.org/abs/2002.05709).

**Core mechanism:** Create two independently augmented views of each image. Encode each view and pass its representation through a nonlinear projection head. A contrastive loss pulls the two views together relative to other images in the batch. The representation before the projection head is normally retained for downstream tasks; the space best suited to the pretext loss is not necessarily the best downstream feature space.

**Inputs/outputs and typical data types:** Unlabeled images enter pretraining. Outputs are normalized projection vectors during training and encoder features for transfer. Downstream classifiers need labels or another explicitly specified decision mechanism.

**Strengths and limitations:** Conceptually simple and benefits from effective augmentations and scale. Large batches can be expensive, and different images of the same semantic category become false negatives. Aggressive crops may remove the very feature a downstream task requires. Contrastive softmax probability measures a within-batch matching task, not calibrated class uncertainty.

**Computational complexity / scalability notes:** Two views double encoder work relative to one-view training. Pairwise similarities over $`2B`$ projected vectors of width $`d_z`$ cost $`O(B^2d_z)`$; naive similarity storage is $`O(B^2)`$, in addition to network activations. Distributed implementations shard or gather embeddings.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** In the [ImageNet study](https://ar5iv.labs.arxiv.org/html/2002.05709), training images without class labels -> paired augmentations -> encoder/projector contrastive training -> frozen encoder -> labeled linear classifier produces validation predictions. The reported linear-evaluation top-1 accuracy is **69.3% for ResNet-50** and **76.5% for a four-times-width ResNet-50**, in the long-training results; the headline 76.5% is not the ordinary-width model. The paper distinguishes 100-epoch ablations from 1,000-epoch high-performance training. The technical reason to prefer this over reconstruction pretraining is learning augmentation-invariant semantic features without a pixel decoder. These are benchmark results, not production visual-inspection savings or a public business KPI.

**Notable vendor implementations/libraries:** Google's [official SimCLR repository](https://github.com/google-research/simclr). Later SimCLRv2 checkpoints and semi-supervised distillation stages are different recipes.

**Architecture diagram description:**

```text
image -> augmentation A -> shared encoder f -> h_A -> MLP g -> z_A
      -> augmentation B -> shared encoder f -> h_B -> MLP g -> z_B
z_A,z_B + other batch projections -> normalized contrastive loss
after pretraining: retain f; discard g; train/evaluate downstream head on h
```

**Activation functions used and why:** The representative ResNet uses ReLU; the two-layer projection MLP has a ReLU hidden layer. L2 normalization turns dot products into cosine similarity; temperature-scaled softmax defines the contrastive classification task.

**Loss function(s):** NT-Xent for paired views $`i,j`$: $`-\log[\exp(\operatorname{sim}(z_i,z_j)/\tau)/\sum_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)]`$, averaged across both directions and all images. The denominator includes the positive and excludes the anchor itself.

**Optimization algorithm(s):** The reported large-batch recipe uses LARS, base rate $`0.3B/256`$, ten-epoch warmup, cosine decay, and weight decay $`10^{-6}`$. At batch 4,096 this yields rate 4.8; it is not an Adam learning rate. Evaluation-head training uses a separate procedure.

**Regularization techniques:** Random resized crops, flipping, color distortion, grayscale, and Gaussian blur define useful invariances. Batch normalization and weight decay matter; projection-head removal after training is an architectural transfer choice.

**Backpropagation considerations:** Both positive branches receive gradients through shared weights; other views contribute negative terms. Distributed gathering must preserve intended gradients, and synchronized normalization avoids exploiting device-local statistics as shortcuts.

**Parameter count / scaling behavior:** The paper rounds the ResNet-50 feature extractor to 24 million parameters and the four-times-width extractor to 375 million, excluding the temporary projection head. Width multiplication increases convolutional parameters approximately quadratically.

**Training paradigm:** Image-only self-supervised contrastive pretraining, followed by supervised linear probing or fine-tuning. The labels used to report accuracy are not part of the pretraining loss.

**Hardware/parallelism considerations:** TPU/GPU distributed training accommodates large effective batches. Global embedding communication, similarity matrices, augmentation throughput, and synchronized normalization can become bottlenecks independent of encoder FLOPs.

### 3.10.2 MoCo

**Name:** Momentum Contrast (MoCo), with the original ResNet-based MoCo v1 recipe as reference.

**Category & sub-category:** Self-supervised contrastive representation learning; momentum encoder and queued dictionary.

**Originating paper/vendor/year:** He, Fan, Wu, Xie, and Girshick, [*Momentum Contrast for Unsupervised Visual Representation Learning*, arXiv 2019, CVPR 2020](https://arxiv.org/abs/1911.05722). Later MoCo v2/v3 recipes change important components and should not inherit the v1 benchmark.

**Core mechanism:** Train a query encoder against keys produced by a slowly updated momentum encoder. Store previous key embeddings in a queue, allowing a large negative dictionary without requiring an equivalently large current minibatch. EMA updates keep old and new keys reasonably consistent as the representation evolves.

**Inputs/outputs and typical data types:** Augmented images enter the query and key encoders. A contrastive embedding supports pretraining; the query backbone supplies features for downstream classification, detection, or segmentation.

**Strengths and limitations:** Decouples dictionary size from current batch size and supports strong transfer. Old keys are stale, and a large queue can contain false negatives. Momentum smoothing is a representation-stability device, not a statistical ensemble that automatically provides uncertainty estimates.

**Computational complexity / scalability notes:** For queue length $`K`$ and embedding width $`d_z`$, comparisons cost $`O(BKd_z)`$ and stored keys cost $`O(Kd_z)`$. Only the online encoder needs ordinary gradient/optimizer state; the key encoder still needs weights and forward computation. The queue avoids storing full image activations from past batches.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original MoCo study](https://ar5iv.labs.arxiv.org/html/1911.05722) pretrains on ImageNet-1M without labels, then evaluates frozen features with a supervised linear classifier. Image -> two augmentations -> query/current positive key plus queued negatives -> trained encoder -> classifier -> ImageNet validation label is the pipeline. The ResNet-50 model reports **60.6% top-1** under this protocol after 200 pretraining epochs. That should not be compared causally with SimCLR's 1,000-epoch figures as if only the loss differed. The paper also evaluates transfer to detection tasks. The documented design motivation is a large, consistent dictionary without huge current batches; no public business KPI follows from the benchmarks.

**Notable vendor implementations/libraries:** Meta/Facebook Research's [official MoCo repository](https://github.com/facebookresearch/moco), which distinguishes recipes and checkpoints. Library defaults may implement v2 rather than v1.

**Architecture diagram description:**

```text
view A -> online query encoder -> q -----------------+
                                                    +-> InfoNCE logits/loss
view B -> momentum key encoder -> positive k --------+
previous key embeddings -> FIFO queue -> negatives --+
online weights -- EMA update, no backprop --> key encoder
```

**Activation functions used and why:** The original ResNet backbone uses ReLU; projected features are L2-normalized before dot products. A temperature-scaled softmax selects the positive key among alternatives. Later nonlinear projection heads are not inherent to v1.

**Loss function(s):** $`-\log[\exp(q\cdot k^+/\tau)/(\exp(q\cdot k^+/\tau)+\sum_{k^-}\exp(q\cdot k^-/\tau))]`$. Queue entries are treated as fixed keys during an update rather than differentiated through their entire historical computation.

**Optimization algorithm(s):** Original ImageNet training uses SGD with momentum 0.9, batch 256, initial learning rate 0.03, and tenfold reductions at epochs 120 and 160 of 200. The key encoder uses an EMA update, not a second gradient optimizer.

**Regularization techniques:** Image augmentations and weight decay regularize learning. The original shuffled-batch-normalization procedure reduces shortcuts from shared batch statistics. Queue consistency and representation invariance serve different purposes.

**Backpropagation considerations:** Stop gradients to keys and queued features. Accidentally retaining their graphs defeats the memory benefit and changes the algorithm. The momentum coefficient trades responsiveness against compatibility with stale keys.

**Parameter count / scaling behavior:** The paper reports roughly 24 million parameters for the ResNet-50 feature extractor. Two weight copies are needed during training, plus the queue and projection parameters; the deployment encoder is not doubled.

**Training paradigm:** Image-only contrastive pretraining. Linear classification and object-detection fine-tuning add labels afterward. An Instagram-pretrained model and an ImageNet-pretrained model have different provenance even with the same architecture.

**Hardware/parallelism considerations:** The original ImageNet recipe uses eight GPUs. Queue updates, positive-key ordering, and normalization shuffling must stay consistent across ranks; distributed throughput is not only a matter of increasing queue size.

### 3.10.3 BYOL

**Name:** Bootstrap Your Own Latent (BYOL).

**Category & sub-category:** Self-supervised representation learning; negative-free bootstrap prediction.

**Originating paper/vendor/year:** Grill and colleagues, DeepMind, [*Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*, NeurIPS 2020](https://arxiv.org/abs/2006.07733).

**Core mechanism:** An online network predicts the projected representation of another augmented view, supplied by an EMA target network. A predictor exists only on the online branch; gradients do not update the target directly. The two roles are swapped to construct a symmetric objective. No explicit negative examples are needed.

**Inputs/outputs and typical data types:** Two augmentations of an unlabeled image enter. Projection/prediction vectors train the system; a backbone representation is retained for downstream tasks. Target features are learned signals, not externally supplied semantic labels.

**Strengths and limitations:** Avoids maintaining a negative dictionary and can be more robust to some augmentation changes than contrastive baselines. Constant-output collapse remains a possible solution to naive feature regression; success depends on the interaction of asymmetry, target dynamics, normalization, and optimization. "Bootstrap" here does not mean statistical resampling and supplies no confidence interval.

**Computational complexity / scalability notes:** There are two online views with gradients and target forward passes, plus projector/predictor overhead. There is no all-pairs $`B^2`$ negative-similarity matrix. EMA updates cost $`O(p)`$; target weights consume memory even though they do not require optimizer moments.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [BYOL ImageNet experiments](https://ar5iv.labs.arxiv.org/html/2006.07733) use image-only pretraining followed by supervised linear evaluation. Images -> augmented views -> online predictions and moving targets -> frozen encoder -> linear classifier yields **74.3% top-1 for ResNet-50** in the 1,000-epoch setting. The headline 79.6% uses a larger ResNet and is not assigned to ResNet-50 here. The technical fit relative to SimCLR/MoCo is avoiding explicit negatives rather than assuming all other images are dissimilar. Transfer experiments support feature usefulness, but no public business KPI or production deployment is established by these scores.

**Notable vendor implementations/libraries:** DeepMind's [official BYOL research implementation](https://github.com/google-deepmind/deepmind-research/tree/master/byol) provides models and recipes. Negative-free methods sharing an EMA teacher are not automatically BYOL.

**Architecture diagram description:**

```text
view A -> online encoder -> projector -> predictor -> normalized prediction
view B -> target encoder -> projector -------------> normalized target [stop-grad]
prediction/target -> squared-distance loss; repeat with views swapped
online encoder/projector weights -- EMA --> target weights
```

**Activation functions used and why:** The ResNet uses ReLU; MLP projector and predictor use hidden nonlinearities and batch normalization. Output vectors are L2-normalized for comparison. The loss does not require semantic-class softmax probabilities.

**Loss function(s):** Squared distance between normalized prediction and stop-gradient normalized target, equivalent to $`2-2`$ cosine similarity for each direction. There is no explicit contrastive denominator over negatives.

**Optimization algorithm(s):** The original large-batch recipe uses LARS with base rate $`0.2B/256`$, ten-epoch warmup, cosine decay over 1,000 epochs, and weight decay $`1.5\times10^{-6}`$. Target EMA momentum rises from 0.996 toward one with a cosine schedule.

**Regularization techniques:** Crops, flips, color transformations, blur/solarization, normalization, and small weight decay work together. A slow teacher and predictor asymmetry are algorithmic components, not independently proven universal anti-collapse guarantees.

**Backpropagation considerations:** Stop gradients on the target branch but retain online encoder/projector/predictor gradients. Monitor representation variance and downstream probes; declining feature-regression loss alone could indicate collapse rather than learning.

**Parameter count / scaling behavior:** The backbone determines deployment size; projectors, predictor, and target copies add training parameters/state. A larger teacher is not required: the target ordinarily matches the online encoder/projector architecture.

**Training paradigm:** Image-only self-supervised feature prediction, with labeled downstream evaluation. The EMA teacher is derived from the learner, not a separately human-labeled pretrained teacher.

**Hardware/parallelism considerations:** The paper's large configuration uses batch 4,096 across 512 TPU-v3 cores. Smaller-batch configurations are also described; the large reported resource budget is not a minimal requirement. Synchronizing normalization/EMA matters for reproduction.

### 3.10.4 DINO

**Name:** DINO, the original self-distillation-with-no-labels method, not an unspecified later DINO-branded model.

**Category & sub-category:** Self-supervised vision representation learning; momentum-teacher distribution matching.

**Originating paper/vendor/year:** Caron and colleagues, [*Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021](https://arxiv.org/abs/2104.14294).

**Core mechanism:** A student matches the probability-like output distribution of an EMA teacher across different image crops. Centering and sharpening the teacher's output help avoid two opposing collapse modes: one dominant dimension and a uniform output for everything. The method supports convolutional backbones and Vision Transformers; the cited visual properties particularly concern ViTs.

**Inputs/outputs and typical data types:** Unlabeled images produce large global and smaller local views. Training outputs are distributions over a learned projection space, not human semantic classes. Encoder vectors and patch features support classification, retrieval, and exploratory object-discovery analyses.

**Strengths and limitations:** Learns useful patch/global structure without negative pairs or human labels during pretraining. Attention visualizations can expose foreground structure but are not guaranteed object masks or explanations. Teacher entropy and attention weights are not calibrated semantic uncertainty.

**Computational complexity / scalability notes:** Cost includes student encoding of multiple crops and teacher encoding of global crops. ViT computation follows $`O(L(Td^2+T^2d))`$ per crop. Halving patch width roughly quadruples $`T`$, making the dense-attention term roughly sixteen times larger at fixed resolution, not implying a sixteenfold total runtime.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [DINO study](https://ar5iv.labs.arxiv.org/html/2104.14294) trains on ImageNet images without labels. Image -> multicrop teacher/student learning -> frozen ViT-S/16 features -> labeled linear classifier or labeled nearest-neighbor reference set yields validation predictions. For ViT-S/16, the table reports **77.0% linear top-1** and **74.5% k-nearest-neighbor top-1**; these are different evaluation procedures. Retrieval and patch-attention results further examine representation structure. The technical fit versus pixel reconstruction is learning cross-view agreement at feature level. The paper does not establish a production segmentation system or business KPI, and nearest-neighbor evaluation is not label-free merely because it avoids training a classifier.

**Notable vendor implementations/libraries:** Meta/Facebook Research's [official DINO implementation](https://github.com/facebookresearch/dino). DINOv2, detection models also called DINO, and third-party checkpoints require separate provenance and should not inherit these results.

**Architecture diagram description:**

```text
global crop -> EMA teacher backbone -> projection head -> centered/sharpened target
global/local crops -> student backbone -> projection head -> student distributions
different-view distributions -> cross-entropy
student weights -- EMA --> teacher; teacher targets use stop-gradient
```

**Activation functions used and why:** The ViT backbone uses GELU and attention softmax with layer normalization. Temperature-controlled softmax normalizes projected outputs; centering subtracts running teacher-logit statistics before sharpening. These outputs are learned codes, not named class probabilities.

**Loss function(s):** Cross-entropy $`-\sum_k P_{\rm teacher}^{(k)}\log P_{\rm student}^{(k)}`$ between different views, with a stopped teacher target. Centering and teacher/student temperatures affect the target distribution and gradient sharpness.

**Optimization algorithm(s):** The paper uses AdamW, base rate $`0.0005B/256`$, ten-epoch warmup, and cosine decay. Weight decay rises from 0.04 to 0.4; teacher EMA momentum rises from 0.996 toward one. Teacher-temperature warmup is yet another distinct schedule.

**Regularization techniques:** Multi-crop augmentation, color transforms, blur/solarization, and weight decay work with teacher centering/sharpening. EMA is not a substitute for all regularization, and collapse prevention depends on the full recipe.

**Backpropagation considerations:** Teacher outputs are detached; only the student receives ordinary gradient updates. Synchronize center statistics across devices. Excessive sharpening or an incorrectly updated center can create degenerate targets even when loss decreases.

**Parameter count / scaling behavior:** The paper rounds the ViT-S feature extractor to 21 million parameters, excluding its training projection head. Patch size strongly changes compute at nearly unchanged backbone weight count; the teacher adds another stored copy.

**Training paradigm:** Image-only self-distillation with no human labels in pretraining. Labeled linear/k-NN evaluation and downstream fine-tuning remain separate, explicitly supervised procedures.

**Hardware/parallelism considerations:** The reported ViT-S/16 recipe uses batch 1,024 over 16 GPUs. Crop-size heterogeneity complicates batching; distributed center and teacher updates must preserve their intended global statistics.

### 3.10.5 Masked autoencoders

**Name:** Masked autoencoder (MAE), specifically the asymmetric Vision Transformer formulation.

**Category & sub-category:** Self-supervised vision representation learning; high-ratio masked reconstruction.

**Originating paper/vendor/year:** He, Chen, Xie, Li, Dollar, and Girshick, [*Masked Autoencoders Are Scalable Vision Learners*, arXiv 2021, CVPR 2022](https://arxiv.org/abs/2111.06377).

**Core mechanism:** Remove a large random subset of image patches before the expensive encoder. A smaller decoder receives encoded visible patches plus mask tokens and reconstructs pixels. Placing mask tokens only in the decoder avoids wasting full encoder computation on invisible input. The encoder is retained for downstream recognition; the decoder is normally discarded.

**Inputs/outputs and typical data types:** Images become patch sequences with random masks. Training outputs are reconstructed patches; deployment outputs are encoder features or predictions from a task-specific head. Standard inference for recognition uses the full image rather than perpetually masking 75%.

**Strengths and limitations:** A simple pixel target supports high-capacity vision pretraining without negative pairs or paired captions. Reconstruction can emphasize low-level details, and good reconstructions alone do not validate semantic features. Squared-error reconstructions are point predictions; plausible alternative completions are not represented by a calibrated posterior.

**Computational complexity / scalability notes:** If visible fraction is $`v`$, encoder projections/MLPs scale with $`vT`$ and dense attention with $`(vT)^2`$; the lightweight decoder still processes the full patch sequence. Thus 75% masking does not imply a universal sixteenfold training speedup. Larger downstream resolutions restore full-token encoder cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [MAE study](https://ar5iv.labs.arxiv.org/html/2111.06377) pretrains on ImageNet-1K images, typically masking 75% of patches. Image -> visible-only ViT encoder -> decoder reconstruction -> discard decoder -> labeled end-to-end fine-tuning produces classification predictions. The paper reports **87.8% top-1 for ViT-H at 448-pixel evaluation resolution**; its ViT-H 224-resolution result is 86.9%. These are **fine-tuned results**, not frozen linear probes or accuracy of the reconstruction task. The technical fit versus a conventional full-token denoising autoencoder is asymmetric compute allocation. The study also evaluates transfer tasks, but no deployed recognition service or public business KPI is established.

**Notable vendor implementations/libraries:** Meta/Facebook Research's [official MAE repository](https://github.com/facebookresearch/mae), plus compatible Vision Transformer libraries. Masked image methods predicting discrete tokenizer codes have different targets and are not automatically MAE.

**Architecture diagram description:**

```text
image -> patches -> random keep/remove -> visible patches only -> large ViT encoder
encoded visible patches + mask tokens + positions -> small decoder
                                                   -> reconstructed pixel patches
after pretraining: full image -> encoder -> downstream head; decoder discarded
```

**Activation functions used and why:** Standard ViT GELU MLPs, attention softmax, and layer normalization appear in encoder/decoder blocks. A linear reconstruction head emits real-valued pixels; it need not use categorical softmax.

**Loss function(s):** Mean squared error on **masked patches only**, with the paper also studying normalized per-patch pixel targets. This is not the same loss as reconstructing all patches equally or predicting a discrete codebook.

**Optimization algorithm(s):** The paper's pretraining table uses AdamW, base rate $`1.5\times10^{-4}`$ scaled by batch/256, batch 4,096, 40-epoch warmup, cosine decay, weight decay 0.05, and betas (0.9, 0.95). Supervised fine-tuning has different schedules and regularization.

**Regularization techniques:** High-ratio random masking is central; pretraining augmentation is comparatively simple. Mixup, CutMix, label smoothing, and stronger stochastic-depth choices belong to specified fine-tuning recipes, not automatically to the masked reconstruction stage.

**Backpropagation considerations:** Gradients pass through the decoder into visible-patch encoder features. There are no encoder activations for removed tokens. Target normalization, patch order restoration, and masked-loss indexing are correctness-critical.

**Parameter count / scaling behavior:** ViT-B/L/H choices set encoder size; the lightweight decoder adds training-only parameters. Increasing resolution expands tokens and positional representations far more than the main shared Transformer weight matrices.

**Training paradigm:** Image-only self-supervised reconstruction, followed by explicitly labeled probing or fine-tuning. The headline recognition number therefore measures a two-stage learning pipeline.

**Hardware/parallelism considerations:** Accelerator data parallelism and activation checkpointing support large encoders. Visible-only encoding reduces pretraining memory; high-resolution full-image fine-tuning may nevertheless become the memory bottleneck.

### 3.10.6 CLIP

**Name:** Contrastive Language-Image Pretraining (CLIP), specifically OpenAI's original 2021 dual-encoder family.

**Category & sub-category:** Cross-modal representation pretraining with **natural language supervision**, preserving the original paper's characterization. Editorial placement here highlights representation transfer, not an absence of supervision. Noisy web pairing can justify a weak-supervision description, but captions still provide semantic training information. See [supervised contrastive learning and Siamese networks](02-supervised-neural.md) for related objectives and architectures.

**Originating paper/vendor/year:** Radford and colleagues, OpenAI, [*Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021](https://arxiv.org/abs/2103.00020).

**Core mechanism:** Train an image encoder and text encoder to match corresponding image-caption pairs against mismatched pairs in a batch. After pretraining, encode candidate class descriptions or search queries and compare them with image embeddings. The text encoder makes the classification vocabulary changeable without training a new fixed class head.

**Inputs/outputs and typical data types:** Paired images and natural-language text train the model. Outputs are aligned embeddings, similarity scores, retrieval rankings, or zero-shot class selections. The original family uses modified ResNets or ViTs for images and a text Transformer; image and text encoders are not weight-tied Siamese twins.

**Strengths and limitations:** Enables flexible cross-modal retrieval and prompt-defined classification. Caption noise, cultural bias, shortcut text, duplicate pairs, and class-description wording affect predictions. A contrastive score or softmax over a chosen candidate list is not calibrated open-world confidence; changing the candidate list changes the apparent probability.

**Computational complexity / scalability notes:** Encoder cost depends on image backbone and text length. With batch $`B`$, pairwise cross-modal similarity costs $`O(B^2d_z)`$, although it can be sharded. At retrieval time, text/image embeddings can be precomputed and indexed; scoring millions of candidates is a separate retrieval-system cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [CLIP paper](https://ar5iv.labs.arxiv.org/html/2103.00020) trains on 400 million internet image-text pairs and tests transfer to numerous datasets. Image-caption pairs -> contrastive dual encoders -> ImageNet class-name prompt templates -> text/image similarities -> selected class yields zero-shot validation predictions. The best **ViT-L/14@336px** model reports **76.2% ImageNet top-1**, using the paper's prompt-based zero-shot evaluation, not a supervised ImageNet linear head. "Zero-shot" means no task-specific labeled training for that classifier, not no image-text supervision or provably no web-data overlap. The appendix also evaluates Flickr30k/MS-COCO retrieval with prompted descriptions; retrieval direction and candidate set differ from classification. The technical fit versus a fixed supervised classifier is a language-defined vocabulary. No enterprise-search cost saving or public business KPI is asserted.

**Notable vendor implementations/libraries:** OpenAI's [official CLIP code and weights](https://github.com/openai/CLIP); Hugging Face Transformers provides compatible model classes. OpenCLIP and other reproductions use different data, architectures, licenses, or recipes and require their own checkpoint provenance.

**Architecture diagram description:**

```text
image -> ResNet or ViT -> projection -> normalized image embedding
caption -> text Transformer -> projection -> normalized text embedding
all image/text dot products in batch -> learned logit scale -> symmetric CE
zero-shot: image embedding vs embeddings of candidate-class descriptions
```

**Activation functions used and why:** Modified ResNet encoders use rectified convolutional nonlinearities; ViT/text Transformer implementations use GELU-family MLP activations and attention softmax. Final L2 normalization and a learned inverse temperature control cross-modal similarity, not semantic truth probabilities.

**Loss function(s):** Average image-to-text and text-to-image cross-entropy over the batch's matching-pair indices. This is contrastive pairing supervision, not caption-generation likelihood and not a supervised class-label loss. False negatives arise when more than one batch caption accurately describes an image.

**Optimization algorithm(s):** The paper uses Adam with decoupled weight decay, cosine learning-rate decay, and architecture-specific hyperparameters over 32 epochs. The strongest ViT-L/14 gets an additional epoch at 336 resolution. The paper initializes the learned temperature at an equivalent 0.07 and reports bounding logit scaling to avoid instability.

**Regularization techniques:** Weight decay excludes certain gains/biases; image preprocessing and broad paired data support transfer. Prompt ensembling is an evaluation technique, not a replacement for training regularization or a universally optimal user prompt.

**Backpropagation considerations:** Both encoders receive contrastive gradients. Distributed similarity calculations must preserve intended gradients to both modalities. Large logit scales can make softmax numerically sharp; mixed-precision statistics and gradient checkpointing require care.

**Parameter count / scaling behavior:** Count depends on the chosen image and text encoders plus projections, not the CLIP name. ViT-L/14@336px is a resolution-specific model, not equivalent to every ViT-L/14 benchmark. Batch similarity cost can grow quadratically without any increase in model parameters.

**Training paradigm:** **Natural language supervision** in the original paper's terminology. Matching targets are constructed from existing image-text pairs, so a broad use of "self-supervised" can describe the pair-prediction objective. That does not make the images semantically unsupervised: human-authored captions and their pairing supply information. "Weakly supervised" emphasizes noisy web associations and the lack of curated task-specific class labels, not zero supervision. Zero-shot prompting, labeled linear probing, and fine-tuning are distinct downstream regimes.

**Hardware/parallelism considerations:** The paper uses batch 32,768, mixed precision, checkpointing, and sharded similarities. It reports the largest ViT training on 256 V100 GPUs and the largest ResNet on 592; these are historical training configurations, not deployment requirements.

### 3.10.7 Word2Vec

**Name:** Word2Vec family: continuous bag-of-words (CBOW) and continuous skip-gram, with output-training variants distinguished.

**Category & sub-category:** Self-supervised language representation pretraining; shallow predictive, log-linear word embeddings.

**Originating paper/vendor/year:** Mikolov, Chen, Corrado, and Dean, Google, [*Efficient Estimation of Word Representations in Vector Space*, 2013](https://ar5iv.labs.arxiv.org/html/1301.3781). Mikolov and colleagues' [*Distributed Representations of Words and Phrases and their Compositionality*, NeurIPS 2013](https://ar5iv.labs.arxiv.org/html/1310.4546) develops negative sampling, subsampling, and phrase modeling. These works did not invent all distributed word representations.

**Core mechanism:** Learn input and output embedding tables from nearby words in running text.

- **CBOW:** Aggregate surrounding-word vectors without preserving their order and predict the center word. One prediction shares evidence from several context positions.
- **Skip-gram:** Use the center word to predict nearby context words, creating multiple prediction pairs per center.

Both avoid the expensive nonlinear hidden layer of earlier neural language models. Hierarchical softmax and negative sampling change output training; they are not synonyms for CBOW and skip-gram.

**Inputs/outputs and typical data types:** Tokenized text produces context/target examples; learned static vectors support nearest-neighbor search or downstream NLP features. A word receives the same vector across contexts. Unseen tokens and polysemy need additional handling; subword models and contextual Transformers are different methods.

**Strengths and limitations:** Efficient semantic sharing replaces unrelated one-hot coordinates with reusable dense features. Window size, frequency, corpus bias, and tokenization determine what is learned. Analogy offsets need not encode universal relationships. Cosine similarity and negative-sampling probabilities are not calibrated semantic confidence or word-sense uncertainty.

**Computational complexity / scalability notes:** With vocabulary $`V`$, dimension $`d`$, and $`K`$ negatives, skip-gram negative sampling costs $`O((K+1)d)`$ per observed pair, multiplied by the number of context pairs. Full softmax costs $`O(Vd)`$; hierarchical softmax costs $`O(hd)`$ for tree-path length $`h`$, typically logarithmic on average. CBOW additionally aggregates its context vectors.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [2013 architecture comparison](https://ar5iv.labs.arxiv.org/html/1301.3781) trains 640-dimensional vectors on 320 million words from LDC corpora with an 82,000-word vocabulary. Text windows -> CBOW or skip-gram -> vectors -> cosine search for $`v_b-v_a+v_c`$ -> predicted analogy answer evaluates linguistic relationships. Under its Semantic-Syntactic Word Relationship protocol, skip-gram achieves **55% semantic accuracy versus CBOW's 24%**, while CBOW achieves **64% syntactic accuracy versus skip-gram's 59%**. These are matched-corpus research comparisons, not downstream classification accuracy or universal model rankings. Their technical fit relative to a full neural language model is cheaper representation learning without an expensive hidden network. No public business KPI is reported.

**Notable vendor implementations/libraries:** The [author-hosted Word2Vec C implementation](https://github.com/tmikolov/word2vec) and [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html). Library defaults, negative counts, and released Google News vectors must not be assumed to reproduce the 640-dimensional comparison.

**Architecture diagram description:**

```text
CBOW: context token IDs -> shared embedding lookups -> sum/mean -> predict center
skip-gram: center ID -> embedding lookup -> predict each nearby context token
output choice: full softmax OR binary tree decisions OR positive/negative logits
after training: retain word vectors -> similarity search / downstream model
```

**Activation functions used and why:** The projection is linear; no hidden ReLU/tanh is required. Full softmax normalizes vocabulary scores. Hierarchical softmax uses sigmoid tree decisions; negative sampling uses sigmoid positive-versus-noise scores, not a normalized language-model distribution.

**Loss function(s):** CBOW minimizes center-word conditional cross-entropy; skip-gram sums context-word prediction losses. Skip-gram negative sampling instead minimizes $`-\log\sigma(u_c^\top v_w)-\sum_{j=1}^{K}\log\sigma(-u_{n_j}^\top v_w)`$. Sampling negatives from smoothed unigram frequencies, raised to $`3/4`$ in the paper, estimates this binary objective; it is not an unbiased estimator of full-softmax cross-entropy.

**Optimization algorithm(s):** The first paper reports SGD/backpropagation with initial rate 0.025 and linear decay toward zero for its serial experiments; its DistBelief parallel setup uses asynchronous minibatch updates with Adagrad. These are separate recipes, not a universal optimizer shared by every Word2Vec implementation.

**Regularization techniques:** Frequent-word subsampling, limited/random context windows, vocabulary thresholds, and finite embedding dimension limit dominance and capacity. Negative sampling also changes the objective; it should not be described merely as dropout.

**Backpropagation considerations:** Update selected embedding rows rather than dense one-hot matrices. CBOW distributes gradients across aggregated context rows; skip-gram accumulates pairwise updates. Sigmoid saturation and frequent-token update collisions can affect optimization; gradients do not differentiate through sampled token identities.

**Parameter count / scaling behavior:** Standard negative sampling stores two $`V\times d`$ tables, approximately $`2Vd`$ parameters. Exporting only input vectors halves that embedding storage; vocabulary metadata and sampler structures remain additional costs.

**Training paradigm:** Self-supervised prediction from observed text neighborhoods, without manually labeled semantic relationships. Analogy answers evaluate the representations; supervised downstream fine-tuning adds a separate label signal.

**Hardware/parallelism considerations:** Sparse updates are CPU-friendly and often memory-bandwidth-bound. Asynchronous threads improve throughput but reduce determinism; distributed tables introduce hot-word contention and communication. A large dense-matrix GPU workload is not required by the architecture.

### 3.10.8 GloVe

**Name:** Global Vectors for Word Representation (GloVe), the original 2014 log-bilinear embedding model.

**Category & sub-category:** Unsupervised/self-supervised language representation pretraining; global co-occurrence fitting. The neural fields describe differentiable embedding lookups and a bilinear score, not an invented deep network.

**Originating paper/vendor/year:** Pennington, Socher, and Manning, Stanford, [*GloVe: Global Vectors for Word Representation*, EMNLP 2014](https://aclanthology.org/D14-1162/). The original code release dates to August 2014; later vector releases listed on the [project page](https://nlp.stanford.edu/projects/glove/) have separate corpora and provenance.

**Core mechanism:** Aggregate word-context co-occurrence counts $`X_{ij}`$, then fit their logarithms with target/context embedding dot products and biases. Weighting limits the influence of very rare counts without allowing frequent pairs to grow unbounded in importance. The paper motivates the representation through ratios of co-occurrence probabilities, which can distinguish properties such as those associated with ice versus steam.

**Inputs/outputs and typical data types:** A tokenized corpus becomes weighted nonzero word-context counts. Outputs are static target/context vectors and biases; the original paper combines the two vector tables by summation for evaluation. This is not a context-sensitive sentence encoder or a normalized next-word distribution.

**Strengths and limitations:** Reusing aggregated statistics can make repeated optimization efficient and exposes the learning target clearly. Co-occurrence storage and preprocessing can be expensive; rare words, ambiguous senses, and corpus biases remain limitations. Similarity is not calibrated confidence, and a fitted log-count residual is not a semantic uncertainty interval.

**Computational complexity / scalability notes:** For $`n`$ corpus tokens and window radius $`c`$, constructing counts takes approximately $`O(nc)`$ pair-processing work. With $`M`$ nonzero pairs and dimension $`d`$, one optimization pass costs $`O(Md)`$; $`E`$ passes cost $`O(EMd)`$. Sparse count storage is $`O(M)`$, not inevitably $`O(V^2)`$, but can still dominate memory.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original paper](https://nlp.stanford.edu/pubs/glove.pdf) evaluates entity extraction from Reuters newswire in CoNLL-2003. Unlabeled-corpus counts -> GloVe vectors -> 50-dimensional features for words in a five-word window -> a CRF trained on labeled CoNLL data -> person/location/organization/miscellaneous tags forms the pipeline. Its NER table reports **88.3 test F1 with GloVe versus 85.4 for discrete features alone**; HPCA scores 88.7 on that test, so GloVe is not best there. The technical rationale is sharing lexical information beyond unrelated discrete features. This is supervised downstream evaluation of unsupervised embeddings, not a label-free NER system, Reuters deployment, or public business KPI.

**Notable vendor implementations/libraries:** Stanford's [official GloVe C implementation and vector releases](https://github.com/stanfordnlp/GloVe). The historical Wikipedia 2014 plus Gigaword 5 vectors differ from later releases. Loading vectors into PyTorch/Keras embedding layers does not reproduce their original training.

**Architecture diagram description:**

```text
corpus -> window counting -> sparse nonzero X_ij
target ID i -> embedding w_i ------+
context ID j -> embedding u_j -----+-> dot product + b_i + b_j
log X_ij + count weight f(X_ij) ----+-> weighted squared-error loss
after training: combine target/context vectors -> NLP features
```

**Activation functions used and why:** No hidden nonlinear activation is required: embedding lookups feed a bilinear dot product plus biases. Logarithms transform count targets and a fractional power weights examples; these are data/objective transformations, not sigmoid class probabilities.

**Loss function(s):** $`J=\sum_{X_{ij}>0}f(X_{ij})(w_i^\top u_j+b_i+\tilde b_j-\log X_{ij})^2`$. The paper uses $`f(x)=\min((x/x_{\max})^\alpha,1)`$, with $`x_{\max}=100`$ and $`\alpha=3/4`$. Omitting zero entries avoids $`\log0`$; this is weighted least squares rather than negative-sampling logistic loss.

**Optimization algorithm(s):** The original recipe uses Adagrad with initial rate 0.05, sampling nonzero count entries. It runs 50 iterations below 300 dimensions and 100 otherwise. Coordinatewise accumulated-gradient scaling supplies adaptation; no cosine or Transformer warmup schedule is implied. The current small-corpus demo has different settings.

**Regularization techniques:** Low-dimensional factors constrain rank; weighting, vocabulary truncation, and inverse-distance context weighting control statistical influence. The weighting function is not L2 parameter regularization. The original result does not require an assumed dropout layer.

**Backpropagation considerations:** Each pair updates two vector rows and two biases. Counts are fixed targets, not differentiable text tokens. The jointly bilinear problem is nonconvex; stable log-count computation, initialization, and accumulated-gradient precision matter.

**Parameter count / scaling behavior:** For vocabulary $`V`$ and dimension $`d`$, separate target/context tables and biases contain $`2Vd+2V`$ parameters. Summed exported vectors need $`Vd`$ values; optimizer accumulators and the sparse count corpus add training storage.

**Training paradigm:** Corpus-derived co-occurrence supervision without manually labeled lexical meanings. The CRF example adds annotated BIO entity targets afterward; its F1 must not be attributed to embeddings alone or to wholly unsupervised training.

**Hardware/parallelism considerations:** The reference implementation uses multithreaded CPU training over shuffled sparse records. Preprocessing can be disk- and memory-intensive; distributing embedding rows adds communication and contention. GPUs are optional implementation choices, not evidence of a hidden deep architecture.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| SimCLR | Images with useful invariance-preserving augmentations | Simple scalable contrastive features | Large-batch cost and false negatives | ImageNet frozen-feature linear evaluation |
| MoCo | Large image collections with moderate current batches | Large queued dictionary with momentum consistency | Stale/false-negative keys | ImageNet linear probing and detection transfer |
| BYOL | Images with meaningful alternate views | No explicit negatives required | Collapse prevention depends on the full recipe | ImageNet ResNet-50 linear evaluation |
| DINO | Image patches/global views, especially ViTs | Useful cross-view global and patch features | Teacher/centering sensitivity | ImageNet ViT-S/16 linear and k-NN evaluation |
| MAE | Patch-structured images | Visible-only encoder lowers pretraining work | Pixel reconstruction is not calibrated semantics | Fine-tuned ImageNet ViT-H evaluation |
| CLIP | Naturally paired images and language | Language-defined classification and retrieval | Paired supervision, bias, and uncalibrated scores | Zero-shot ImageNet and cross-modal retrieval |
| Word2Vec: CBOW / skip-gram | Tokenized text and local context windows | Efficient static predictive embeddings | Polysemy, vocabulary gaps, and corpus bias | Matched-corpus semantic/syntactic analogy evaluation |
| GloVe | Sparse global word-context counts | Reusable weighted co-occurrence statistics | Count storage and context-independent vectors | CoNLL-2003 NER with embedding features |

## Evidence reading notes

- **FID and Inception Score:** DDPM's 3.17 FID and 9.46 +/- 0.11 IS are checked against the original CIFAR-10 table and 50,000-sample protocol, including training-reference versus test-reference statistics. Score-SDE's 2.20 belongs to the continuous deep NCSN++ VE model, not the generic framework. DiT's 2.27 belongs to the guided, long-trained 256-resolution XL/2 configuration, not its unguided 400,000-step ablations. These numbers are sourced, not locally reproduced.
- **Likelihood:** RealNVP's 3.49, Glow's 3.35, and Gated PixelCNN's 3.03 bits/dimension are paper-specific CIFAR-10 results. Continuous/dequantized likelihood and discrete autoregressive likelihood have different qualifications. Five-bit Glow face demonstrations must not be mixed into an eight-bit comparison.
- **Representation evaluation:** SimCLR's wider-backbone headline, MoCo's shorter training budget, BYOL's backbone size, DINO's linear versus k-NN evaluator, MAE's supervised fine-tuning and resolution, and CLIP's prompted zero-shot protocol all matter. The chapter is not a leaderboard computed under a common budget.
- **Static word embeddings:** Word2Vec's CBOW/skip-gram semantic and syntactic percentages are checked against the original 640-dimensional, matched-corpus comparison, not mixed with later Google News exports. GloVe's 88.3 versus 85.4 CoNLL test F1 comes from a supervised CRF with 50-dimensional embedding features versus its discrete-only baseline; the paper's HPCA result is higher on that test. These are neither embedding-only accuracy nor business gains. CLIP retains the original paper's natural language supervision characterization despite the chapter's broad self-/weak-supervision editorial grouping.
- **Production versus research:** The updated WaveNet deployment is supported by Google's dated announcement; the original WaveNet listener scores are a separate research result. Other worked examples primarily establish research capabilities. Model cards and demo code are not evidence of audited customer savings.
- **Disclosure limits:** The original beta-VAE OpenReview PDF was unavailable to this evidence pass; its original attribution and dataset are supported by the authors' dSprites documentation, while detailed architecture/optimizer statements are explicitly tied to the checked 2018 follow-up. DALL-E 3's report and evaluation artifacts were checked, but they intentionally do not disclose a complete generator recipe. Stable Diffusion version distinctions here use the v1.4/SDXL cards and SDXL paper; unverified per-checkpoint details are not inferred from inaccessible cards.

## Coverage and continuation manifest

**Covered: 28 entries, each with the nine common and nine neural fields, an architecture diagram, a worked example, and category-table coverage.**

| Section range | Coverage | Entry count |
|---|---|---|
| 3.6.1-3.6.5 | Autoencoder; denoising autoencoder; VAE; beta-VAE; VQ-VAE | 5 |
| 3.7.1-3.7.4 | GAN; DCGAN; versioned StyleGAN family; CycleGAN | 4 |
| 3.8.1-3.8.7 | DDPM; score-SDE; latent diffusion/Stable Diffusion; DiT; DALL-E 1; DALL-E 2; DALL-E 3 | 7 |
| 3.9.1-3.9.4 | RealNVP; Glow; PixelCNN; WaveNet | 4 |
| 3.10.1-3.10.8 | SimCLR; MoCo; BYOL; DINO; MAE; CLIP; Word2Vec (CBOW/skip-gram); GloVe | 8 |

**Connections to the rest of the book:** [Reading guide and evidence policy](00-reading-guide.md); [supervised neural, contrastive, and Siamese methods](02-supervised-neural.md); [semi-supervised learning and pseudo-labeling](03-semi-supervised.md); [classical unsupervised methods](04-unsupervised-classical.md); [foundation-model pretraining](06-foundation-models.md); [MoE models](07-moe-models.md); [dedicated MoE deep dive](08-moe-deep-dive.md); [comparative selection guide](09-comparative-guide.md); [glossary](10-glossary.md).

**Bounded continuation, not a claim of exhaustiveness:** Additional depth could cover hierarchical VQ-VAE-2 and residual quantization; Wasserstein and energy-based models; consistency/distilled diffusion; flow matching and rectified flows; newer Stable Diffusion/DiT-derived checkpoints; video, audio, and 3D diffusion; spline and continuous normalizing flows; PixelCNN++ and parallel WaveNet; DINOv2 and later variants; SimCLRv2/MoCo v2-v3; fastText/subword and multilingual static embeddings; multimodal masked modeling; domain-specific scientific applications; and rigorous calibration, memorization, licensing, and privacy audits. These are non-required extensions, not silently covered by the 28 entries. No standalone reinforcement-learning taxonomy or undisclosed proprietary training recipe is implied.
