# 3. Unsupervised Learning Algorithms: Neural Generation and Representation Pretraining

This chapter covers sections **3.6-3.10**. Some methods learn to rebuild data or create new examples. Others learn useful ways to describe images, sounds, or words with numbers. It continues [classical unsupervised learning](04-unsupervised-classical.md) and comes before [foundation models](06-foundation-models.md).

An **encoder** turns an input into a list of numbers called a code or representation. A **decoder** turns that code back into an output, such as an image. A compact code uses less space than the original input. A **probability distribution** describes possible outcomes and their relative chances. It can describe a list of choices or a range of possible values.

A **vector** is a list of numbers. To take two vectors' **dot product**, multiply matching entries, then add those products. A **dense layer** connects each of its input values to each output using learned weights.

During training, a model changes its **weights**, the numbers that control its calculations. A **loss** measures the error it tries to reduce. **Backpropagation** works backward through the calculations to find how weights affect that error. An **optimizer** uses this feedback to update them. Training often processes a batch of examples at once; an **epoch** is one pass through the training set.

**Evidence policy, checked 2026-09-08.** The cited papers and named versions are historical reference points, not a list of today's newest systems. A research test, selected demonstration, and working service prove different things. Results below keep their original test conditions. None establishes commercial cost savings. Unless a study is named, explanations and comparisons are this book's interpretation.

**The data and training task determine the supervision.** Rebuilding an input or predicting hidden parts uses the input itself as a teaching signal. This is self-supervision. Captions supply language supervision, even without specially written class labels. Class-controlled generators use class labels. This chapter therefore includes more than strictly label-free methods. It also includes CLIP and links to [supervised contrastive and Siamese networks](02-supervised-neural.md). Training a labeled classifier later does not change how the earlier image-only model learned.

**Cost and uncertainty conventions.** Running a network once is a forward pass. Training also works backward, so each update costs more. Image filters reuse weights across many positions; counting weights alone does not measure the work. Stored weights and optimizer records grow with model size. Intermediate results also take memory, depending on batch size, image size, depth, and implementation.

**Optional math:** Let $`n`$ be the number of training examples, $`B`$ the batch size, $`E`$ the epochs, $`p`$ the trainable weight count, and $`C_f`$ one example's forward-pass cost. An ordinary update costs roughly a constant times $`BC_f`$, not always $`O(Bp)`$. Here Big-O describes how work grows, rather than exact running time. A convolution costs about $`O(HWk^2c_{\rm in}c_{\rm out})`$: $`H,W`$ are image height and width, $`k`$ is filter width, and $`c_{\rm in},c_{\rm out}`$ count input and output channels. Channels hold different learned image details. A dense-attention Transformer costs $`O(BL(Td^2+T^2d))`$ per forward pass. Here $`T`$ counts sequence items, $`d`$ is their vector width, and $`L`$ counts layers. $`S`$ will mean sampling steps, not sequence length.

Varied samples, small rebuilding errors, close matching scores, and high likelihood are different measurements. None alone gives a trustworthy chance that an answer is correct. **Calibration** means predicted chances match observed frequencies under the conditions tested.

## 3.6 Reconstruction and latent-variable models

These models squeeze an input through a restricted code, then try to rebuild it. That can teach useful patterns without providing a way to create new examples. A VAE adds rules for drawing random codes. A vector-quantized model chooses codes from a learned list and usually learns a separate rule for generating them. These differences matter when claiming that a model can generate data or measure its probability.

### 3.6.1 Autoencoders

**In plain English:** An autoencoder learns a compact code by trying to rebuild its input. The code can help compress data or find similar items.

**Name:** Autoencoder. This entry uses a deep model with a code smaller than its input, called an undercomplete autoencoder.

**Category & sub-category:** Unsupervised/self-supervised learning. It learns to rebuild inputs and describe them with fewer numbers.

**Originating paper/vendor/year:** Networks that learn to copy inputs existed before modern deep learning. Hinton and Salakhutdinov's [*Reducing the Dimensionality of Data with Neural Networks*, Science, 2006](https://www.cs.toronto.edu/~hinton/absps/science.pdf) presented an influential deep version. They did not invent every form of autoencoder.

**Core mechanism:** The encoder turns an input into a short code. The decoder uses that code to rebuild the input. Limited code space encourages the model to keep recurring patterns rather than every input detail.

With linear layers, squared error, and suitable restrictions, this relates to PCA's straight-line compression. Nonlinear layers can follow curved patterns. Without enough restrictions, however, the network may simply learn to copy its input.

**Inputs/outputs and typical data types:** Inputs include number lists, images, spectra, and document word frequencies. Outputs are codes and rebuilt inputs. Input scaling and output rules must suit the data. Randomly chosen codes do not automatically produce meaningful new examples.

**Strengths and limitations:** It can compress complex patterns and provide useful starting weights when labels are scarce. But rebuilding inputs may preserve irrelevant details instead of useful ones. A small rebuilding error does not prove an item is normal. A large error is not a trustworthy chance of an anomaly. Any cutoff needs testing on a separate, suitable reference population.

**Computational complexity / scalability notes:** Wider neighboring layers need more connections and more work. Encoding a new item needs only the encoder; rebuilding it needs both networks. Training also adds backward calculations. Image-filter versions follow the convolution cost described above.

**Optional math:** For layer widths $`d_0,\ldots,d_L`$, a forward pass costs $`O(\sum_{l=1}^{L}d_{l-1}d_l)`$ per example. Here $`L`$ counts the layers being evaluated. Each product counts connections between two neighboring layers.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** Hinton and Salakhutdinov studied search across 804,414 Reuters newswire stories. Each story became probabilities over 2,000 word stems. The encoder reduced these to a ten-dimensional code. Search ranked other stories by cosine similarity, which compares the directions of their codes.

The study trained on half the stories. It reported better retrieval by news category than latent semantic analysis, a linear method. Nonlinear compression can help when related documents do not follow straight-line patterns. The [retrieval experiment](https://www.cs.toronto.edu/~hinton/absps/science.pdf) was research, not a Reuters production system or a measured improvement in analyst productivity.

**Notable vendor implementations/libraries:** PyTorch and Keras supply the needed dense and image-filter layers. The authors also released supporting code. These tools do not prove that their vendors run the news-search system.

**Architecture diagram description:**

```text
word-probability vector (2000)
  -> dense 500 -> dense 250 -> dense 125 -> linear code (10)
  -> mirrored decoder -> normalized reconstructed word probabilities
```

**Activation functions used and why:** The document model uses logistic units, which squeeze values between zero and one. Its code is linear, and its final probabilities are normalized to sum to one. ReLU, which removes negative values, is a modern alternative, not the 2006 recipe.

**Loss function(s):** The document experiment penalizes rebuilt word probabilities that disagree with the input. It uses cross-entropy. Squared error instead suits a model with bell-shaped errors of fixed spread; it is not right for every count or category.

**Optional math:** The document loss is $`-\sum_j x_j\log \hat x_j`$. Here $`j`$ indexes word stems, $`x_j`$ is an input probability, and $`\hat x_j`$ is its rebuilt probability. The sum measures disagreement across all stems; lower is better.

**Optimization algorithm(s):** The historical method first trains stacked restricted Boltzmann machines to obtain starting weights. It then joins the layers into an encoder-decoder and improves them by backpropagation. This was not Adam training. Adam with a rate reduced when validation stops improving is a possible new recipe, not a historical claim.

**Regularization techniques:** The narrow code is the main restriction. Other options penalize large weights, favor mostly inactive units, share encoder-decoder weights, or stop training early. Learning starting weights beforehand does not guarantee success on new data.

**Backpropagation considerations:** Error feedback must pass through both decoder and encoder. Logistic units near their limits can pass back very weak feedback; this helped motivate staged pretraining. Cutting the feedback path at the code prevents normal joint training.

**Parameter count / scaling behavior:** There is no fixed autoencoder size. Dense layers store a weight for each connection between neighboring layers. A smaller code does not remove expensive outer layers. Sharing encoder-decoder weights reduces storage.

**Training paradigm:** The input supplies its own target, making reconstruction self-supervised. A later task may add labels. Reuters category labels checked whether nearby codes represented related stories; they were not reconstruction targets.

**Hardware/parallelism considerations:** Small models can run on CPUs. Large image models benefit from GPUs processing different batches together. Decoder intermediate results may use most training memory, even if later use needs only the encoder.

### 3.6.2 Denoising autoencoders

**In plain English:** A denoising autoencoder learns by repairing deliberately damaged inputs. This helps it notice relationships, rather than merely copy what it sees.

**Name:** Denoising autoencoder (DAE). Several can also be stacked to learn in stages.

**Category & sub-category:** Self-supervised learning. It rebuilds damaged inputs to learn useful features before a later task.

**Originating paper/vendor/year:** Vincent, Larochelle, Bengio, and Manzagol introduced the method in [*Extracting and Composing Robust Features with Denoising Autoencoders*, ICML 2008](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf). Vincent and colleagues expanded it in [JMLR, 2010](https://www.jmlr.org/papers/v11/vincent10a.html).

**Core mechanism:** Start with a clean example and randomly damage part of it. Send the damaged version through an encoder and decoder. Train the output to match the clean original, not the damaged input. The original masking experiment set selected input values to zero. To fill these gaps, the network must use relationships among the remaining values.

**Inputs/outputs and typical data types:** Clean images or number lists provide both damaged inputs and clean targets automatically. Outputs are repaired data or learned features. Real noisy measurements need a suitable damage model. Repairing masked digits does not validate a system for clinical images.

**Strengths and limitations:** The repair task can be useful even when the hidden code is larger than the input. But practice with one kind of damage does not ensure resistance to another. Missing pixels, sensor noise, and deliberately misleading changes differ. One repaired output is a single estimate. Trying different masks does not produce verified probabilities for alternative originals.

**Computational complexity / scalability notes:** Each damaged version needs an encoder-decoder training pass, plus the work of creating damage. Using several independently damaged versions adds roughly that many passes. Training stacked layers separately adds more stages.

**Optional math:** Masking costs $`O(Bd)`$ for $`B`$ examples with $`d`$ input values each. Using $`m`$ damaged views multiplies the corresponding work by roughly $`m`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The ICML study tested digit recognition on the Larochelle benchmark. Its MNIST variants included rotation and image or random backgrounds. It removed random input values, trained stacked repair networks, then trained a labeled classifier on their features. The relevant variants used 10,000 training, 2,000 validation, and 50,000 test examples. This is not standard MNIST's usual split.

The [study](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) found these starting weights competitive with or better than ordinary autoencoder starting weights. The JMLR extension also reports lower classification errors and units that detect edges or strokes. Unlike PCA, the aim is useful resistance to damage, not just preserving the largest variation. No production text-recognition savings or business performance measure was reported.

**Notable vendor implementations/libraries:** Keras and PyTorch support this repair task. Historical code and the JMLR article describe stacked training. Modern tutorials may use different networks.

**Architecture diagram description:**

```text
clean x -----> corruption sampler -----> corrupted x
   |                                      |
   |                               encoder -> hidden code
   |                                      |
   +-------- reconstruction loss <----- decoder -> repaired x
```

**Activation functions used and why:** The original basic model uses sigmoid units for inputs in $`[0,1]`$, or zero to one. Outputs also stay in that range. Later models may use ReLU image filters or linear outputs with bell-shaped error assumptions for other data ranges.

**Loss function(s):** Compare the repaired output with the clean input, averaging over examples and random damage. Use binary cross-entropy or squared error as the data require. This trains repair from an input. It is not a GAN's real/fake test or the probability calculation for a diffusion chain.

**Optional math:** The loss is $`\mathbb E_{x,\tilde x}[\ell(x,g_\theta(f_\phi(\tilde x)))]`$. Here $`x`$ is clean data, $`\tilde x`$ is its damaged version, and $`\mathbb E`$ means average. The encoder $`f_\phi`$ and decoder $`g_\theta`$ have learned weights $`\phi,\theta`$. The function $`\ell`$ measures rebuilding error.

**Optimization algorithm(s):** Original training uses stochastic gradient descent, which updates weights from small batches of errors. Experiments choose layer sizes, damage levels, pretraining length, and when to stop labeled training. No learning-rate schedule is universal. Starting with a fixed rate and reducing it when validation stalls is a possible reproducible choice.

**Regularization techniques:** Deliberate damage is the main safeguard against simple copying. Small codes, penalties on large weights, and early stopping can help too. Enough useful information must survive the damage for repair to be learnable.

**Backpropagation considerations:** Error feedback passes through the repair network, not through the random mask choice. Random damage makes updates noisier. Nearly saturated sigmoid units weaken feedback. Averaging several plausible repairs into one smooth output can also hurt useful features.

**Parameter count / scaling behavior:** A DAE normally has as many weights as its clean-input counterpart. Random masking adds none. Depth and image-filter channel widths determine model size.

**Training paradigm:** Damaged/clean pairs provide self-supervision. Labeled fine-tuning comes afterward. If people supply extra clean measurements as targets, that additional supervision must also be stated.

**Hardware/parallelism considerations:** Damage can be created directly on the training device. Different workers should use independent random masks. Large-image decoders need substantial training memory; feature extraction afterward can omit the decoder.

### 3.6.3 Variational autoencoders

**In plain English:** A VAE learns ranges of possible compact codes, rather than one fixed code per input. It can draw a code at random and decode it to create a new example.

**Name:** Variational autoencoder (VAE). This version uses a bell-shaped estimate for each code coordinate, without estimated correlations between coordinates.

**Category & sub-category:** Unsupervised generation. It learns a shared encoder that quickly estimates plausible hidden codes for each input.

**Originating paper/vendor/year:** Kingma and Welling, [*Auto-Encoding Variational Bayes*, arXiv 2013, ICLR 2014](https://arxiv.org/abs/1312.6114). Related ways to train through random choices appeared around the same time. A VAE is not just a renamed fixed-code autoencoder.

**Core mechanism:** A **prior** describes possible codes before seeing an input. The encoder estimates which codes could explain a particular input. The decoder describes possible outputs for a code. Training balances rebuilding the input against keeping its codes close to the prior. It uses a manageable bound instead of checking every possible code. The same encoder handles new examples. To generate data, draw a code from the prior, then an output from the decoder.

**Inputs/outputs and typical data types:** Images or numerical observations enter, with an explicit rule for modeling their probabilities. Outputs include code averages and spreads, sampled codes, and distributions over rebuilt or new observations. A decoder's average image is not the same as a randomly drawn image.

**Strengths and limitations:** It gives a clear way to balance rebuilding accuracy and code complexity. But simple code distributions may miss important possibilities. Treating output pixels independently can give blurry average images. Code spread only describes uncertainty within these assumptions. It does not describe uncertainty over all network weights or guarantee reliable warnings about unfamiliar data.

**Computational complexity / scalability notes:** Each batch needs one encoder pass. Drawing more codes per input adds decoder passes and their backward work. The code-distribution penalty grows directly with code length. More elaborate probability estimates add work and remain estimates, not exact probabilities.

**Optional math:** With $`m`$ random code draws, the batch needs about $`m`$ decoder passes. The diagonal-Gaussian KL penalty costs $`O(r)`$ per example, where $`r`$ is code length.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [AEVB experiments](https://ar5iv.labs.arxiv.org/html/1312.6114) modeled MNIST digits and Frey Face images. For a face frame, the encoder predicts bell-shaped code distributions. A random code goes to a decoder that describes possible rebuilt frames. Drawing from the prior instead creates new frames.

The study compared training bounds and estimated overall data likelihood with other estimators. It also showed how learned codes organized examples. Unlike a fixed-code autoencoder, the VAE has an explicit prior and a manageable probability-based training goal. The experiment used 200 hidden units for Frey faces and 500 for MNIST. It drew one random code per example in minibatches of 100. This was research, not a deployed face-analysis system or a reported business benefit.

**Notable vendor implementations/libraries:** Pyro, TensorFlow Probability, PyTorch, and Keras support VAEs. Probability libraries implement output distributions and their comparison penalties. They cannot decide whether those assumptions suit an application.

**Architecture diagram description:**

```text
x -> encoder -> mean mu(x), log-variance log sigma^2(x)
                       |
epsilon ~ N(0,I) -> z = mu + sigma * epsilon
                       |
                    decoder -> parameters of p(x | z)
prior N(0,I) ----------^       [generation bypasses encoder]
```

**Activation functions used and why:** The original dense network uses tanh, which bounds hidden values between minus one and one. Sigmoid outputs give probabilities for binary data. Bell-shaped output distributions need an average and a positive variance, which measures spread. Predicting log variance and exponentiating it ensures a positive value.

**Loss function(s):** The loss combines rebuilding error with a penalty for moving code distributions away from the prior. It is called the negative evidence lower bound, or negative ELBO.

**Optional math:** Minimize $`\mathbb E_q[-\log p_\theta(x\mid z)]+\mathrm{KL}(q_\phi(z\mid x)\|p(z))`$. Here $`x`$ is the input and $`z`$ the code. The encoder distribution is $`q_\phi`$, the decoder distribution is $`p_\theta`$, and their weights are $`\phi,\theta`$. The prior is $`p(z)`$. The first term averages rebuilding loss over encoder-drawn codes. KL measures disagreement between the code distribution and prior. For independent Gaussian coordinates and a standard normal prior, KL has a direct formula; rebuilding usually still uses random draws.

**Optimization algorithm(s):** The original experiments use Adagrad, which adjusts step sizes separately for each weight. They choose a global step size from 0.01, 0.02, and 0.1. Adam and gradual learning-rate warmup are later options, not original AEVB requirements.

**Regularization techniques:** The KL penalty limits how much information the codes carry. Weight penalties and early stopping are optional. Gradually increasing KL pressure, or allowing some unpenalized information through "free bits," can prevent unused codes. These changes must be reported.

**Backpropagation considerations:** Keep the random draw separate from the learned average and spread. This lets error feedback pass through the code calculation with less noise than an alternative probability-score estimator. Check for numerical overflow and for a decoder that ignores the code, called posterior collapse.

**Optional math:** Draw $`z=\mu+\sigma\epsilon`$. Here $`\mu`$ is the encoder's average code, $`\sigma`$ its coordinate-by-coordinate spread, and $`\epsilon`$ independent standard-normal noise. This is the reparameterization trick: learn $`\mu,\sigma`$ while holding the sampled noise fixed during the backward calculation.

**Parameter count / scaling behavior:** Encoder and decoder sizes determine the weight count. Output layers for code averages and variances grow with code length times their input width. Training stores both networks. Unconditional generation needs only the decoder and prior.

**Training paradigm:** The representative model learns data probabilities without external labels, using inputs as reconstruction targets. Conditional VAEs add labels or other control information, so they are not strictly label-free.

**Hardware/parallelism considerations:** Different workers can train on different minibatches, with independent noise draws. KL and log-variance calculations may need more numerical precision than surrounding matrix multiplications.

### 3.6.4 Beta-VAE

**In plain English:** Beta-VAE puts a tighter limit on what an image's compact code can remember. This can help separate properties such as position and shape, but may lose detail.

**Name:** Beta-VAE. The original method weights a code penalty; a later version controls a changing information limit.

**Category & sub-category:** Unsupervised learning of hidden codes. It limits information to encourage separate code coordinates for different image properties.

**Originating paper/vendor/year:** Higgins and colleagues introduced [*beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, ICLR 2017](https://openreview.net/forum?id=Sy2fzU9gl). [Burgess and colleagues, 2018](https://arxiv.org/abs/1804.03599), studied a changing information limit. The detailed image-filter recipe below comes from that follow-up.

**Core mechanism:** Start with a VAE's encoder, random code, and decoder. Increase the penalty for letting each input's code distribution differ from the prior. The multiplier is beta. More pressure sometimes separates position, size, and rotation, but harms rebuilding. The follow-up instead penalizes distance from an information target that grows during training. These are related, not identical, training goals.

**Inputs/outputs and typical data types:** Inputs are images with repeated, changing properties. Outputs include code distributions, rebuilt images, and images made by changing one code coordinate. Known image properties can check whether coordinates separate them, without supplying those properties during encoder training.

**Strengths and limitations:** Restricting information can make codes easier to interpret on controlled datasets. Separate coordinates do not necessarily reveal the real causes of an image. The network's built-in assumptions matter. Better separation may mean worse reconstruction. Its uncertainty still depends on the chosen prior, output distribution, and encoder.

**Computational complexity / scalability notes:** Each batch needs the same main work as a VAE: encoding, decoding sampled codes, and a code-distribution penalty. Changing beta adds little arithmetic. Choosing beta across datasets and random starting conditions can require many full runs. Images made by moving code coordinates are inspections, not deployment-accuracy measurements.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** [dSprites](https://github.com/google-deepmind/dsprites-dataset) contains 737,280 generated 64-by-64 binary images. Shape, scale, orientation, and position are controlled. A beta-VAE encodes each image. Researchers change one code coordinate and inspect which known image property changes.

The [Burgess study](https://ar5iv.labs.arxiv.org/html/1804.03599) shows the trade-off between accurate rebuilding and separate properties. It also shows recovery of properties with the changing information limit. A separate experiment directly used known factors in its bottleneck; that analysis was supervised. Compared with PCA, the appeal is separating nonlinear patterns. Compared with beta=1, it is a stronger information restriction. This was a controlled scientific test, not an industrial deployment or business performance measure.

**Notable vendor implementations/libraries:** Google's research code and DeepMind's dSprites repository support these tests. PyTorch and TensorFlow can implement both loss variants. The dataset explicitly is not a commercial Google product.

**Architecture diagram description:**

```text
64x64 sprite -> 4 strided convolutions -> 2 dense layers
  -> Gaussian mean/log-scale -> sampled latent
  -> transposed decoder -> Bernoulli pixel probabilities
                      KL to N(0,I) -> beta or capacity penalty
```

**Activation functions used and why:** The 2018 recipe uses ReLU to keep positive hidden values and remove negative ones. Four image-filter layers have 32 channels and 4-by-4 filters. Two dense layers then have 256 units each. The decoder gives probabilities for binary pixels, so outputs must stay between zero and one.

**Loss function(s):** Both variants reward successful rebuilding. The original adds a weighted penalty for code distributions differing from the prior. The follow-up instead rewards staying near a chosen amount of code information.

**Optional math:** The original extra term is $`\beta\mathrm{KL}(q\|p)`$; the follow-up uses $`\gamma|\mathrm{KL}(q\|p)-C|`$. Here $`q`$ is the encoder's code distribution, $`p`$ the prior, and KL their difference measure. $`\beta`$ and $`\gamma`$ set penalty strength; $`C`$ is the target information amount. With $`\beta\ne1`$, this is not the ordinary VAE evidence lower bound.

**Optimization algorithm(s):** The checked 2018 appendix uses Adam at $`5\times10^{-4}`$ (0.0005). For dSprites, the information target grows evenly from zero to 25 nats over 100,000 iterations. A nat is an information unit based on natural logarithms. This changes allowed information, not the learning rate. These are not claimed settings for every 2017 run.

**Regularization techniques:** The information penalty is the central restriction. Treating prior coordinates and estimated code coordinates independently adds further assumptions. Choosing a model using known image properties adds supervision to model selection, even if reconstruction training uses no labels.

**Backpropagation considerations:** As in a VAE, random codes are calculated from learned averages and spreads. Too much penalty pressure can make useful coordinates inactive. Track each coordinate's penalty and rebuilding error. Crossing the information target reverses the direction of the capacity penalty's feedback.

**Parameter count / scaling behavior:** The follow-up uses ten Gaussian code coordinates for dSprites. Its CelebA experiment uses a different code size. Changing beta alone adds no weights. Image resolution and filter channels determine the actual network size.

**Training paradigm:** Mainly image-only self-supervised rebuilding, sometimes with a gradually changing information limit. Experiments using known factors, and later labeled classifiers, are separate stages.

**Hardware/parallelism considerations:** Moderate-resolution experiments fit on ordinary GPUs. Workers must combine KL penalties consistently. Replacing a sum with an average changes beta's effective strength, so the recipes no longer match.

### 3.6.5 VQ-VAE

**In plain English:** VQ-VAE describes an input using entries from a learned code list. A separate model can learn to choose new code sequences, which the decoder turns into new images or sounds.

**Name:** Vector-quantized variational autoencoder (VQ-VAE). This entry covers the original single-level model.

**Category & sub-category:** Unsupervised generation and discrete coding. It turns inputs into selections from a learned list, called a codebook.

**Originating paper/vendor/year:** Van den Oord, Vinyals, and Kavukcuoglu, DeepMind, [*Neural Discrete Representation Learning*, NeurIPS 2017](https://arxiv.org/abs/1711.00937). The later VQ-VAE-2 uses several code levels. It is not the model described here.

**Core mechanism:** The encoder first outputs number vectors. Replace each vector with the nearest entry in the codebook. The decoder rebuilds the input from these choices. A separate model then learns which code choices follow others. This is an autoregressive prior: it generates codes in order. Without it, the tokenizer mainly rebuilds inputs; it is not a complete generator of realistic code sequences.

**Inputs/outputs and typical data types:** Images, speech, or video enter. Outputs include grids or sequences of code numbers, rebuilt inputs, and new samples from the prior. Codebook entries are learned vectors. They are not automatically named speech sounds or objects.

**Strengths and limitations:** Short discrete codes make the separate sequence model's job smaller. Some codes may never get used, while a few dominate. Training also uses approximate feedback through code selection. Code-matching error and codebook perplexity, which summarizes how broadly codes are used, are diagnostics, not confidence intervals. Prior probabilities need calibration checks.

**Computational complexity / scalability notes:** Checking every codebook entry takes more work as either the codebook or input grid grows. This adds to encoder-decoder work. A PixelCNN prior still generates positions in order, but the code grid is smaller than the pixel grid.

**Optional math:** Full lookup costs $`O(BN_zKd_z)`$. Here $`B`$ is batch size, $`N_z`$ positions per input, $`K`$ codebook entries, and $`d_z`$ numbers per entry. Codebook storage is $`O(Kd_z)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original study](https://ar5iv.labs.arxiv.org/html/1711.00937) turns 128-by-128 ImageNet images into a 32-by-32 grid with 512 code choices. The encoder selects nearby codes; the decoder produces recognizable rebuilt images. A PixelCNN trained on the grids then creates new codes and images.

The authors use the smaller grid to focus the prior on broader structure rather than pixel detail. These demonstrations show rebuilding and generation, not standardized commercial compression savings. The nominal nine bits per index excludes model storage, prior coding overhead, and differences in fidelity. The paper also demonstrates VCTK speech codes and speaker conversion. No public business performance measure accompanies these research results.

**Notable vendor implementations/libraries:** DeepMind's Sonnet examples and common PyTorch/JAX implementations support codebook lookup. Later tokenizers may change losses, codebook updates, or the number of code levels.

**Architecture diagram description:**

```text
image -> convolutional encoder -> vectors z_e
  -> nearest codebook entries e[index] -> convolutional decoder -> reconstruction
encoded training indices -> PixelCNN prior
prior samples -----------> codebook lookup -> decoder -> generated image
```

**Activation functions used and why:** The image-filter networks use ReLU, which keeps positive values, and shortcut connections between layers. Decoder outputs must match the chosen data-probability model. Choosing the nearest code is a discrete selection, not a smooth activation that ordinary backpropagation can follow.

**Loss function(s):** Three terms reward accurate rebuilding, move codebook entries toward encoder outputs, and keep encoder outputs near chosen entries. The separate prior learns to predict code numbers with its own probability loss.

**Optional math:** Minimize $`-\log p_\theta(x\mid z_q)+\|\operatorname{sg}(z_e)-e\|^2+\beta\|z_e-\operatorname{sg}(e)\|^2`$. Here $`x`$ is the input, $`z_e`$ the encoder output, $`e`$ the chosen entry, and $`z_q`$ the code sent to the decoder. The decoder distribution $`p_\theta`$ has weights $`\theta`$. Squared lengths measure mismatch. $`\beta`$ weights the last term, and $`\operatorname{sg}`$ means "stop error feedback here."

**Optimization algorithm(s):** The image comparison uses Adam at $`2\times10^{-4}`$ (0.0002), batch 128, and evaluation after 250,000 steps. No rate-decay schedule is required for the whole family. Updating codes with an exponential moving average, or EMA, is an alternative. EMA blends recent values with older ones rather than using gradient updates.

**Regularization techniques:** The commitment term keeps encoder outputs close to codebook values. The finite list limits information capacity. Monitoring unused entries and carefully resetting codes may help. These fixes do not guarantee that codes avoid collapsing to the same few choices.

**Backpropagation considerations:** A straight-through estimator copies decoder feedback across the discrete lookup to the encoder. This is a biased approximation, not the exact derivative of choosing the nearest entry. Stopping feedback in selected loss terms separates codebook updates from encoder commitment. The prior is trained afterward on fixed code targets.

**Parameter count / scaling behavior:** Count the encoder, decoder, codebook, and any prior separately. The prior may take most storage despite the compact grid. **Optional math:** The codebook alone has $`Kd_z`$ values, for $`K`$ entries of width $`d_z`$.

**Training paradigm:** Input-based reconstruction comes first, followed by unsupervised code-sequence learning. The voice-conversion experiment also uses speaker identity as a condition. That is explicit supervision, not entirely label-free speech generation.

**Hardware/parallelism considerations:** Comparing many vectors with all codes can use substantial GPU memory. Workers using EMA codebooks must share counts and sums, or their code lists diverge. Waiting for the prior to generate codes remains a separate speed limit.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Autoencoder | Number lists, images, word frequencies | Learns a small code for complex patterns | Rebuilding can preserve the wrong details | Research on searching Reuters news |
| Denoising autoencoder | Images or sensor-like values that can be damaged | Learns relationships by repairing inputs | Practice with one damage type may not transfer | Recognition tests on MNIST variants |
| VAE | Data with a chosen probability model | Quickly estimates possible codes and generates samples | Code estimates can miss possibilities or go unused | Tests on Frey Face and MNIST |
| Beta-VAE | Images with repeatedly changing properties | Controls how much codes can remember | Separate properties are not guaranteed | Tests of known dSprites properties |
| VQ-VAE | Images, sound, or video needing discrete codes | Learns codes and a rule for generating them | Approximate training and unused codes | ImageNet coding and VCTK speech research |

## 3.7 Adversarial generation

A GAN learns through feedback between a creator and a checker. The **generator** creates examples. The **discriminator** checks how generated examples differ from training examples. The generator learns from that checker's feedback. It is not simply a pair of classifiers: one network makes data.

This approach can create sharp-looking samples quickly without calculating a complete probability model for the data. But both networks keep changing, making training difficult. A checker's score does not guarantee quality or reliable uncertainty, especially on unfamiliar data.

### 3.7.1 Generative adversarial networks

**In plain English:** A GAN learns to make examples that a trained checker has trouble distinguishing from real ones. Once trained, its generator can create an image in one forward pass.

**Name:** Generative adversarial network (GAN). This entry describes the general framework through the original image models.

**Category & sub-category:** Unsupervised generation. Competing training goals help generated data resemble the training data's overall patterns.

**Originating paper/vendor/year:** Goodfellow and colleagues, [*Generative Adversarial Nets*, NeurIPS 2014](https://arxiv.org/abs/1406.2661). Later convolutional, Wasserstein, conditional, and style-based GANs change the network design or training goal.

**Core mechanism:** Start the generator with random numbers and let it make an example. Train the discriminator to distinguish generated examples from real training examples. Then update the generator so its outputs are harder to distinguish. Repeat. A mathematical balance point exists under ideal training and sufficient capacity. That does not guarantee that real networks reach it.

**Inputs/outputs and typical data types:** Real images or other observations train the system. Random number vectors produce new samples. Conditional versions also take labels, text, or another image. A standard GAN usually lacks both an easy exact data-probability calculation and an encoder that reverses generation.

**Strengths and limitations:** Generation is fast, and many network designs can serve as generators. But **mode collapse** can make outputs cover only a small part of the training data's variety. Realistic images may still be copied, biased, or structurally wrong. Different random inputs create variety, not a trustworthy measure of what the model does not know.

**Computational complexity / scalability notes:** Training pays for both networks, including their intermediate results and optimizer records. More discriminator updates mean more work. A generator update also sends feedback through the discriminator. Sampling afterward needs one generator pass, unlike diffusion's repeated passes.

**Optional math:** If $`k_D`$ is the number of discriminator updates per generator update, each round includes $`k_D`$ discriminator training passes, plus generator training through the discriminator.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [2014 paper](https://ar5iv.labs.arxiv.org/html/1406.2661) trained on MNIST, the Toronto Face Database, and CIFAR-10. It alternated checker and generator updates using real-image batches and random inputs. Researchers then inspected generated samples and compared data distributions. No predefined pixel-class labels were needed.

The method directly trains a sampler without first solving for a hidden code for each training image. This avoids a difficult step in some hidden-variable models. The paper's Parzen-window likelihood estimates are approximate measurements built around samples, not exact GAN likelihoods. They are not presented here as modern rankings. The experiment establishes neither commercial deployment nor a business performance measure.

**Notable vendor implementations/libraries:** The authors' [research implementation](https://github.com/goodfeli/adversarial) documents the original system. PyTorch and TensorFlow support GAN training. Libraries also supply later variants, whose results must retain their specific model names.

**Architecture diagram description:**

```text
noise z -> generator G -> fake image ----+
                                        +-> discriminator D -> real/fake logit
training real image --------------------+
generator update: differentiate through D into G, without updating D
```

**Activation functions used and why:** The original generator uses rectifiers, which remove negative values, and sigmoids, which bound values. The discriminator uses maxout units, which choose the largest of several learned values. A sigmoid or equivalent stable loss calculation gives its real/fake probability. Later GANs may use other activations.

**Loss function(s):** The discriminator is rewarded for telling real from generated examples. The generator is rewarded for making that harder. The paper also proposes a generator loss that gives stronger early feedback when the checker easily wins.

**Optional math:** The original game is $`\min_G\max_D \mathbb E_x\log D(x)+\mathbb E_z\log(1-D(G(z)))`$. Here $`G`$ generates from random input $`z`$, $`D`$ estimates "real," $`x`$ is real data, and $`\mathbb E`$ means average. The discriminator maximizes this score while the generator minimizes it. The alternative, non-saturating generator loss is $`-\mathbb E_z\log D(G(z))`$. It strengthens early feedback but is not the identical generator update.

**Optimization algorithm(s):** The original algorithm alternates small-batch gradient updates, with one discriminator step per generator step. Learning rates and their decay are implementation choices, not a universal GAN schedule. Later GANs often use Adam, but GANs do not require it.

**Regularization techniques:** The original discriminator uses dropout, which temporarily removes selected units during training. Later options constrain weights, limit how strongly layers amplify signals, penalize gradients, or modify training images. A Wasserstein gradient penalty belongs to a different recipe; do not silently add it to the original loss.

**Backpropagation considerations:** An overly confident checker can leave the original generator loss with almost no useful feedback. The non-saturating loss helps, but updates can still circle rather than settle. During generator updates, hold discriminator weights fixed while keeping the feedback path through its calculations.

**Parameter count / scaling behavior:** Training stores generator and discriminator weights; generation stores only the generator. A larger checker may provide better feedback or simply memorize training data. Equal weight counts do not guarantee stability.

**Optional math:** Training count is $`p=p_G+p_D`$, where $`p_G`$ and $`p_D`$ count generator and discriminator weights. Sampling requires $`p_G`$.

**Training paradigm:** The training process supplies real/generated labels itself. These are not externally supplied object categories. A conditional GAN adds supervision when it uses class labels or captions.

**Hardware/parallelism considerations:** GPUs speed up both networks. Multiple workers must coordinate alternating updates and normalization statistics, which rescale intermediate values. Running only the trained generator is much simpler than training both networks.

### 3.7.2 DCGAN

**In plain English:** DCGAN uses learned image filters in both a GAN's creator and checker. This makes it a practical starting point for generating moderate-sized images.

**Name:** Deep convolutional generative adversarial network, usually shortened to DCGAN.

**Category & sub-category:** Unsupervised GAN image generation. It also learns image-filter features that can be reused for recognition.

**Originating paper/vendor/year:** Radford, Metz, and Chintala described it in [*Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*, arXiv 2015, ICLR 2016](https://arxiv.org/abs/1511.06434).

**Core mechanism:** The discriminator shrinks image grids using learned filters that move several positions at a time. The generator enlarges grids using learned transposed-convolution filters. Carefully placed batch normalization rescales values using batch statistics. This is a tested design and training recipe, not a new data-likelihood formula. Afterward, a labeled classifier can use the checker's learned features.

**Inputs/outputs and typical data types:** Training uses natural images scaled to a suitable numeric range. The illustrated generator turns a 100-dimensional random code into a 64-by-64 RGB image. The discriminator's intermediate features can also be kept fixed and reused.

**Strengths and limitations:** DCGAN learns small-to-large image patterns with a repeatable design. Enlarging grids with overlapping filters can create uneven artifacts. Batch normalization makes each example depend partly on its batchmates. Published stability does not guarantee diverse outputs on every dataset. Smooth changes between generated images, or high checker scores, do not establish calibrated uncertainty.

**Computational complexity / scalability notes:** Layer work grows with grid size, filter size, and channel counts. Larger images also need more intermediate memory, even when filters reuse the same weights. Training evaluates both networks. Extracting learned features afterward needs only the discriminator's feature layers.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [DCGAN study](https://ar5iv.labs.arxiv.org/html/1511.06434) used LSUN bedrooms, with a little over three million training examples. It scaled pixels to $`[-1,1]`$, meaning minus one to one. GAN training then produced 64-by-64 synthetic rooms for visual and feature tests.

The authors showed plausible rooms and checked memorization; realistic appearance alone was not treated as enough evidence. Reusing filters across an image is the advantage over a similarly sized fully connected GAN. Separate transfer tests trained labeled classifiers on frozen features. Those labels did not train the bedroom generator. No hotel-design deployment or measured business saving was established.

**Notable vendor implementations/libraries:** The authors' [DCGAN Torch code](https://github.com/soumith/dcgan.torch) and [PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) provide implementations. Tutorials can change datasets, channel widths, and training budgets, so they need not reproduce the paper.

**Architecture diagram description:**

```text
z (100) -> project/reshape -> transposed conv + BN + ReLU
        -> repeated spatial upsampling -> tanh -> 64x64 RGB
real/fake RGB -> strided conv + LeakyReLU
             -> conv + BN + LeakyReLU -> binary discriminator
```

**Activation functions used and why:** Generator ReLU keeps positive signals; tanh bounds final pixels to the training range. The discriminator's LeakyReLU also passes some negative-side feedback, with slope 0.2 in the paper. A logistic output supplies the real/fake probability.

**Loss function(s):** Real/fake cross-entropy trains the checker to distinguish images. The generator learns to make its images receive the real label. There is no target image paired with each random input, so this is not a reconstruction loss.

**Optimization algorithm(s):** The paper uses Adam at 0.0002 and minibatch 128. There is no universal rate-decay rule, and this rate need not suit every resolution.

**Technical detail (optional):** Its first-moment setting is $`\beta_1=0.5`$. This controls smoothing of recent update directions and is lower than Adam's usual value.

**Regularization techniques:** The paper leaves batch normalization out of the generator output and discriminator input because it destabilized training there. It starts weights with a bell-shaped distribution of standard deviation 0.02. That controls their initial spread. Starting weights and design restrictions matter, but initialization is not itself statistical regularization.

**Backpropagation considerations:** Watch for a checker that wins too easily or updates that swing back and forth. Batch statistics can reveal whether a batch contains real or generated images. Changing batch construction therefore changes training. Replacing transposed filters with resizing followed by filters changes the design and its artifacts.

**Parameter count / scaling behavior:** Channel widths determine much of the weight count. A 100-dimensional code does not specify total model size. Training stores both networks; sampling needs only generator weights.

**Training paradigm:** The representative GAN learns from images alone. A later linear support vector machine uses labels to classify fixed features. A DCGAN that takes class labels during generation is a supervised conditional variant.

**Hardware/parallelism considerations:** Moderate-resolution models fit on one GPU. Larger batches improve estimates of batch statistics but use more memory. Whether workers share normalization statistics can substantially change reproduced results.

### 3.7.3 StyleGAN family

**In plain English:** StyleGAN creates images using controls that affect details at different sizes. Later versions change how those controls work, reduce artifacts, or help training with limited data.

**Name:** StyleGAN family. StyleGAN, StyleGAN2, StyleGAN2-ADA, and StyleGAN3 are distinct versions, not interchangeable names.

**Category & sub-category:** GAN image generation with style controls. Later versions also address artifacts caused by sampling image signals on a grid.

**Originating paper/vendor/year:** Karras and colleagues at NVIDIA developed [StyleGAN, arXiv 2018/CVPR 2019](https://arxiv.org/abs/1812.04948), [StyleGAN2, arXiv 2019/CVPR 2020](https://arxiv.org/abs/1912.04958), [adaptive discriminator augmentation, NeurIPS 2020](https://arxiv.org/abs/2006.06676), and [StyleGAN3, NeurIPS 2021](https://arxiv.org/abs/2106.12423).

**Core mechanism:** A mapping network turns random input into style controls. Different generator layers use them to affect image structure at different sizes.

- Original StyleGAN uses adaptive instance normalization, or AdaIN, to adjust feature averages and spreads. It adds noise and grows the network progressively.
- StyleGAN2 instead adjusts and rescales filter weights, called modulation/demodulation. It changes the generator and adds a penalty for uneven responses to style changes.
- StyleGAN2-ADA changes how strongly training images are modified for the discriminator. This helps with limited-data overfitting; it does not add a new meaning-based input condition.
- StyleGAN3 changes signal processing so textures move with image content rather than sticking to the grid. It reduces aliasing, or sampling artifacts. StyleGAN3-T targets consistent shifts, called translation equivariance. StyleGAN3-R also targets consistent rotations. These claims do not apply equally to every version.

**Inputs/outputs and typical data types:** Random inputs produce images. Mixing styles, searching for a code that rebuilds an image, or adding conditions provides more control. A code found by this search, called inversion, is not necessarily the image's unique true cause.

**Strengths and limitations:** Fast generation and editable styles are useful in image research. Training data limits the range of identities, poses, and demographic groups. Inversion may miss unfamiliar content. Truncation favors a narrower range of codes to improve some visual-quality measures. It sacrifices variety; it does not filter by trustworthy confidence.

**Computational complexity / scalability notes:** Mapping random inputs to styles usually costs less than building the high-resolution image. Generator work grows with image size and channel counts. StyleGAN3's filtering adds work. Computing expensive penalties only occasionally saves time, but this "lazy regularization" changes effective optimizer settings.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** NVIDIA introduced FFHQ in the [original StyleGAN study](https://ar5iv.labs.arxiv.org/html/1812.04948). It has 70,000 aligned 1024-by-1024 face images, with faces placed in similar positions. GAN training learns styles and creates portraits. Researchers evaluate image quality and how code changes affect outputs.

StyleGAN2 studies visible artifacts and rebuilding images through inversion. [StyleGAN3](https://ar5iv.labs.arxiv.org/html/2106.12423) shows more consistent transformations without texture sticking. Compared with DCGAN, the useful difference is control at different image scales. These are research findings, not evidence of a film studio's adoption or reduced animation costs. No audited production business measure is asserted.

**Notable vendor implementations/libraries:** NVIDIA publishes [StyleGAN2-ADA PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StyleGAN3](https://github.com/NVlabs/stylegan3). Check the saved model, dataset, license, and settings rather than relying on the family name.

**Architecture diagram description:**

```text
z -> mapping MLP -> w -> per-layer style controls
                         |
learned input / version-specific Fourier input
  -> modulated synthesis blocks -> RGB image -> discriminator
       StyleGAN1: AdaIN; StyleGAN2: demodulation
       StyleGAN3: alias-aware filtering and nonlinearities
```

**Activation functions used and why:** LeakyReLU allows some feedback through negative values in mapping and image-building layers. StyleGAN3 filters around nonlinear operations to suppress newly created high-frequency detail that would cause artifacts. Original AdaIN and StyleGAN2 demodulation rescale different things; they are not identical normalization.

**Loss function(s):** A representative StyleGAN2 recipe uses the non-saturating real/fake GAN loss. Its R1 penalty limits the checker's sensitivity to changes in real inputs. Its path-length penalty encourages more even image changes as styles change. Versions and settings differ in which penalties they use and how strongly.

**Optimization algorithm(s):** The [official StyleGAN2-style configuration](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py) uses Adam at base rate 0.002. Occasional penalty calculations adjust effective optimizer settings. This rate is not universal across StyleGAN versions.

**Technical detail (optional):** Its smoothing settings are $`(\beta_1,\beta_2)=(0,0.99)`$. These betas control averages of updates and squared updates.

**Regularization techniques:** Mixing styles, limiting checker sensitivity, smoothing style responses, and averaging weights over time serve different purposes. ADA uses discriminator behavior to set the chance of modifying images. It does not keep increasing damage until every image becomes unrecognizable.

**Backpropagation considerations:** R1 and path-length penalties require tracking how gradients themselves change, which uses extra memory. Filter scaling and rescaling must stay numerically stable. StyleGAN3 removes random per-pixel noise because it conflicts with consistently shifting or rotating image content.

**Parameter count / scaling behavior:** Resolution, maximum channel widths, mapping depth, and version determine size. Count generator, discriminator, and any inversion encoder separately. An image-resolution label alone does not tell you the weight count.

**Training paradigm:** The cited FFHQ work mainly learns to generate images without labels as conditions. Labeled conditional versions are separate. A generated portrait's apparent identity is not supplied ground truth.

**Hardware/parallelism considerations:** The original paper reports eight V100 GPUs for training. Later code combines operations in specialized GPU routines and mixes numeric precision levels. Reproducing runs requires matching image modifications, weight averaging, precision, and how often penalties are calculated.

### 3.7.4 CycleGAN

**In plain English:** CycleGAN learns to change images from one collection's style to another without matched before-and-after pairs. It checks that changing an image back can recover the original.

**Name:** Cycle-consistent generative adversarial network, or CycleGAN.

**Category & sub-category:** Unpaired image translation. It combines GAN feedback with a round-trip rebuilding check.

**Originating paper/vendor/year:** Zhu, Park, Isola, and Efros presented [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017](https://arxiv.org/abs/1703.10593).

**Core mechanism:** Train one generator to change collection X into collection Y, and another to go back. Each direction has a checker for realistic-looking output. A round-trip penalty encourages the two changes together to recover the starting image. This discourages throwing away all input content. It does not guarantee the uniquely correct meaning-preserving translation.

**Inputs/outputs and typical data types:** Inputs are two separate image collections; outputs are translated images. The system knows each image's collection, but does not need matched pairs. Photos, paintings, seasons, and maps of scene categories require different kinds of information to survive.

**Strengths and limitations:** It helps when matched examples are scarce and content should mostly survive translation. Round-trip checks struggle when many inputs should share one output. They can also allow hidden information or preserve the wrong meaning. A fixed output cannot represent every plausible translation. Small round-trip error is not a verified uncertainty estimate.

**Computational complexity / scalability notes:** A round trip uses two generator passes. Training includes both directions and both discriminators, with extra memory for round-trip intermediate results. Work is a constant multiple of the corresponding image-filter networks. Using only one translation direction afterward requires one generator.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** In the [Cityscapes evaluation](https://ar5iv.labs.arxiv.org/html/1703.10593), researchers translated between street photos and maps of scene-category labels. They ignored the available matching between individual maps and photos. A generator turned a label map into a street image. A pretrained fully convolutional network, or FCN, then labeled its pixels. Comparing those labels with the input map checked whether scene categories remained recognizable.

Tests that removed parts of the method supported combining GAN and round-trip losses. CycleGAN suits missing pair alignment; paired pix2pix uses aligned examples. Discarding reliable pairs is not inherently better. Collection labels and human-created scene maps still supply supervision. This was not an autonomous-driving deployment or a verified safety or business measure.

**Notable vendor implementations/libraries:** The authors' [PyTorch repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) distinguishes unpaired CycleGAN from paired pix2pix. Other image filters may change the generators or losses.

**Architecture diagram description:**

```text
x in X -> G -> fake Y -> F -> reconstructed X
                 |                  |
                D_Y             cycle L1 to x
y in Y -> F -> fake X -> G -> reconstructed Y
                 |                  |
                D_X             cycle L1 to y
```

**Activation functions used and why:** The generators use shortcut connections, ReLU hidden units, and bounded tanh outputs. PatchGAN checkers judge image regions and use LeakyReLU, preserving some negative-side feedback. Instance normalization rescales each image's features without needing statistics from a large batch.

**Loss function(s):** Practical training uses squared real/fake errors plus round-trip rebuilding error. An optional identity loss discourages changes when an input already belongs to the desired collection.

**Optional math:** The round-trip penalty is $`\lambda_{\rm cyc}(\|F(G(x))-x\|_1+\|G(F(y))-y\|_1)`$. Here $`x,y`$ are images from the two collections. $`G`$ translates X to Y, and $`F`$ translates Y to X. Each $`\|\cdot\|_1`$ adds absolute pixel errors; $`\lambda_{\rm cyc}`$ sets the penalty's strength.

**Optimization algorithm(s):** The paper uses Adam, batch size one, and learning rate 0.0002. The rate stays fixed for 100 epochs, then decreases evenly to zero over 100 more. This controls weight updates, not the noise process used in diffusion.

**Regularization techniques:** Round-trip and optional identity penalties restrict possible translations. Keeping some earlier generated images helps stop checker updates swinging back and forth. Resizing, cropping, and flipping provide varied training views.

**Backpropagation considerations:** Round-trip feedback must pass through both generators. Stored old images should not accidentally retain old training graphs. Realism and content preservation can pull updates in different directions, so inspect both losses.

**Parameter count / scaling behavior:** Training includes two generators and two discriminators. The number of shortcut blocks, image resolution, and channel widths affect size. CycleGAN has no fixed weight count.

**Training paradigm:** It learns conditional translation between known collections without matched image pairs. "Unsupervised translation" refers to missing pair matches, not the absence of every supervision signal.

**Hardware/parallelism considerations:** Batch-one training fits on a GPU, but round-trip graphs still need memory. Multiple devices can process examples or directions. Their updates must stay consistent, without using outdated weights from the opposite generator.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| GAN, generic | Data a network can generate and learn through | Creates samples in one forward pass | Unstable feedback and missing variety | Tests on MNIST, Toronto faces, and CIFAR-10 |
| DCGAN | Moderate-resolution natural images | Uses image filters and reusable features | Visible artifacts and sensitivity to batches | Research on generating LSUN bedrooms |
| StyleGAN family | Carefully selected image collections | Style controls; later versions reduce grid artifacts | Training-data bias and differences between versions | FFHQ portrait and transformation research |
| CycleGAN | Two collections without matched images | Translates without before-and-after pairs | A successful round trip can still change meaning | Cityscapes photo and label-map tests |

## 3.8 Diffusion and image-generation families

Diffusion training adds known random noise to examples and learns how to reverse that damage. Generation starts with fresh noise and repeatedly uses learned steps to build a sample. This is **not ordinary sharpening of a blurry photograph**. The model must learn patterns from data to generate missing structure.

The network can work on pixels, compact image codes, or other feature vectors. It may use a U-Net, which shrinks and expands image grids with shortcuts, or a Transformer, which mixes information across sequence items. This section also includes **DALL-E 1, an autoregressive model, not diffusion**. It generates image codes one after another. DALL-E versions get separate entries because a shared product name does not imply the same design.

A noise schedule controls training-image damage. A learning-rate schedule controls weight updates. A sampling scheduler controls the numerical steps used to create an output. These are separate choices. Guidance steers generation toward a condition, such as text. Ranking several candidates selects among outputs. Both change the result distribution and the work needed for evaluation.

### 3.8.1 Denoising diffusion probabilistic models

**In plain English:** A DDPM learns to reverse different amounts of random noise added to images. It then starts from noise and builds a new image through many small steps.

**Name:** Denoising diffusion probabilistic model (DDPM). This entry uses Ho, Jain, and Abbeel's version.

**Category & sub-category:** Unsupervised generation through a fixed sequence of noise-removal steps, called discrete-time diffusion.

**Originating paper/vendor/year:** Ho, Jain, and Abbeel presented [*Denoising Diffusion Probabilistic Models*, NeurIPS 2020](https://arxiv.org/abs/2006.11239). Their work builds on earlier diffusion generation; it did not originate the whole idea.

**Core mechanism:** Repeatedly adding random values from a bell-shaped, or Gaussian, distribution gradually overwhelms an image. A network sees a noisy image and its noise level, then learns the reverse-step calculation. Training can create an example at any noise level directly, without running all earlier damage steps. Generation runs the learned reverse steps in sequence.

**Optional math:** A noisy training input is $`x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon`$. Here $`x_0`$ is a clean image, $`x_t`$ its version at step $`t`$, and $`\epsilon`$ standard Gaussian noise. $`\bar\alpha_t`$ describes how much original signal remains. The square-root factors set the signal and noise mixture.

**Inputs/outputs and typical data types:** Training takes clean images and randomly chosen noise levels. The network predicts noise. Generation starts from Gaussian noise and ends with an image. The original CIFAR-10 model uses no class condition. Adding labels or captions makes a different, conditional training task.

**Strengths and limitations:** Predicting noise is often more stable than training two competing GAN networks. It can cover a broad range of data patterns. The original sampler, however, needs many network calls. Varied generated anatomy, text, or events can still be wrong. Their variation is not a confidence interval about truth.

**Computational complexity / scalability notes:** Training usually chooses just one noise step per example. It does not run a whole generation chain for each update. Original sampling repeats the network many times; more steps mean roughly proportionally more work. Image size, channel widths, and attention placement affect each pass.

**Optional math:** Sampling costs approximately $`SC_f`$, where $`S`$ is the number of steps and $`C_f`$ one network pass's cost. Faster samplers change the evaluation procedure and must be identified.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** On unconditional 32-by-32 CIFAR-10, a clean image is mixed with known noise. A U-Net predicts that noise at the given step. The learned reverse sampler then creates images for comparison with the dataset.

The [paper](https://ar5iv.labs.arxiv.org/html/2006.11239) reports **FID 3.17** and **Inception Score 9.46 +/- 0.11** from 50,000 generated samples. FID compares real and generated image-feature distributions; lower is better. Inception Score rewards confident class predictions and class variety using a pretrained classifier. Higher is better, but it is not percent-correct accuracy.

The 3.17 FID uses training-set reference statistics and the lowest-FID checkpoint selected during training. Test-reference FID is separately 5.24. Thus 3.17 is not held-out classification accuracy. Noise prediction replaces GAN competition, and the simpler loss improved sample quality over the paper's likelihood-bound loss. This benchmark establishes no deployed image service or public business benefit.

**Notable vendor implementations/libraries:** The authors' [TensorFlow code](https://github.com/hojonathanho/diffusion) and Hugging Face Diffusers supply DDPM components. A scheduler alone does not identify which trained network or saved checkpoint it uses.

**Architecture diagram description:**

```text
clean image + sampled Gaussian noise -> x_t
x_t + time embedding -> residual U-Net with skip connections/attention
                    -> predicted noise epsilon_theta(x_t,t)
generation: Gaussian noise -> repeated reverse updates -> image
```

**Activation functions used and why:** The U-Net uses smooth swish units and shortcut connections. Attention softmax turns comparison scores into mixing weights. The final noise prediction is unbounded. Using tanh to restrict it would change what noise values it can predict.

**Loss function(s):** The simple loss compares added noise with predicted noise. Mean squared error, or MSE, averages squared prediction errors; lower is better. The full probability-bound loss weights noise levels differently, so it is not numerically the same objective.

**Optional math:** Minimize $`\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2`$. Here $`\epsilon`$ is the known added noise and $`\epsilon_\theta`$ the network prediction with weights $`\theta`$. It sees noisy input $`x_t`$ at step $`t`$. $`\mathbb E`$ means average; the squared length adds squared errors across values.

**Optimization algorithm(s):** The [official CIFAR configuration](https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py) uses Adam, rate $`2\times10^{-4}`$ (0.0002), 5,000-step warmup, and batch 128. Its 1,000 forward noise levels increase beta evenly from 0.0001 to 0.02. Beta controls added noise. Warmup instead gradually raises the learning rate.

**Regularization techniques:** The recipe rescales groups of features, drops some units, randomly flips images horizontally, and averages weights over time. These support training and evaluation. The checked CIFAR dropout rate is 0.1.

**Backpropagation considerations:** Feedback comes from one sampled noise-prediction task, not the full generation chain. Weighting noise levels changes which errors matter most. Limiting oversized gradients and carefully calculating noise variance help avoid numerical failures.

**Parameter count / scaling behavior:** The official CIFAR U-Net has approximately 35.7 million parameters. This is not a standard DDPM size. Higher-resolution conditional models can be much larger.

**Training paradigm:** The representative model predicts automatically added noise on unlabeled images, so it is self-supervised. Class labels or text remain supervision when added, even though the noise targets are synthetic.

**Hardware/parallelism considerations:** The original code supports Cloud TPU training; GPU versions are common. Batches split easily across devices. Yet each reverse step waits for the previous one, limiting how quickly one image can be generated.

### 3.8.2 Score-based SDE models

**In plain English:** These models learn which direction a noisy example should move to become more like the training data. They use that direction repeatedly to turn noise into a new sample.

**Name:** Score-based generation through stochastic differential equations, or SDEs. An SDE describes continuous change that includes random motion.

**Category & sub-category:** Unsupervised generation with continuously varying noise levels. Sampling can use random-motion equations or deterministic ordinary differential equations, called ODEs.

**Originating paper/vendor/year:** Song and colleagues presented [*Score-Based Generative Modeling through Stochastic Differential Equations*, arXiv 2020, ICLR 2021](https://arxiv.org/abs/2011.13456).

**Core mechanism:** Specify how an example changes as continuous time adds noise. Train a network to estimate the **score** at each noise level. Here "score" means a local direction toward higher data density, not an image-quality grade. It determines how the reverse process moves. Variance-exploding, variance-preserving, and sub-variance-preserving SDEs are different rules for how signal and noise spread evolve.

**Optional math:** The forward rule is $`dx=f(x,t)dt+g(t)dw`$. Here $`x`$ is the current data, $`t`$ time, $`dt`$ a small time change, and $`dx`$ the resulting data change. $`f`$ gives directed motion; $`g`$ sets the strength of random motion $`dw`$. The score is $`\nabla_x\log p_t(x)`$: the direction of change in log density $`p_t`$ at time $`t`$. A related probability-flow ODE removes random motion. With an exact score, it gives the same distribution at each time, not the same individual sample paths.

**Inputs/outputs and typical data types:** Training takes observations, a time value, and Gaussian noise. Sampling produces images. An ODE calculation can also estimate log density. Reconstructing an image from partial measurements needs those measurements and a suitable rule for using them as conditions.

**Strengths and limitations:** One framework connects discrete diffusion with noise-dependent direction learning and several numerical solvers. Approximate steps, imperfect scores, and approximate conditions all cause error. Plausible alternatives to a measured image do not automatically have reliable probabilities. Both the measurement model and learned score need testing.

**Computational complexity / scalability notes:** Training chooses a time and noise draw without running the full process. Sampling work depends on all network calls, including both predictor and corrector steps. Adaptive ODE solvers can need different numbers of calls. Computing likelihood also tracks how the transformation expands or contracts space, called divergence. Hutchinson trace probes estimate this quantity using random vectors.

**Optional math:** Multiply the number of network evaluations by $`C_f`$, the work for one network pass. Density calculations add divergence estimation and numerical integration.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [CIFAR-10 experiments](https://ar5iv.labs.arxiv.org/html/2011.13456) train an NCSN++ network to predict scores at continuous noise levels. A reverse sampler follows its predictions to create images. The **NCSN++ continuous deep, VE-SDE** model achieves **FID 2.20** with the authors' 50,000-sample protocol. FID compares real and generated feature distributions; lower is better. This result belongs to that deep version, not shallower or discrete-time versions.

Compared with a fixed-step DDPM, the benefit is continuous-time training and a choice of solvers, not guaranteed faster generation. The paper also demonstrates reconstruction from measurements. The CIFAR score is neither an audited production measure nor proof that reconstruction probabilities are calibrated.

**Notable vendor implementations/libraries:** The authors' [JAX/TensorFlow code](https://github.com/yang-song/score_sde) and [PyTorch version](https://github.com/yang-song/score_sde_pytorch) keep the noise process, network, and sampler choices separate.

**Architecture diagram description:**

```text
image x_0 + perturbation kernel at time t -> x_t
x_t + continuous-time embedding -> NCSN++ / DDPM++ score network
                               -> estimated score s_theta(x_t,t)
noise -> reverse-SDE predictor/corrector OR probability-flow ODE -> image
```

**Activation functions used and why:** The checked NCSN++ uses smooth swish activations, rescaling within feature groups, and attention. Its output is a real-valued direction, not probabilities over categories. It can be rescaled by the noise standard deviation, which measures noise spread.

**Loss function(s):** Training compares the predicted direction with a target calculated from the known noise process. Weighting different noise levels changes the emphasis between sample quality and a likelihood-related goal.

**Optional math:** Minimize $`\mathbb E[\lambda(t)\|s_\theta(x_t,t)-\nabla_{x_t}\log p(x_t\mid x_0)\|^2]`$. Here $`x_0`$ is clean data and $`x_t`$ its noisy version at time $`t`$. The network $`s_\theta`$ has weights $`\theta`$. The gradient of the known conditional noise log density supplies the target direction. $`\lambda(t)`$ weights each time; $`\mathbb E`$ averages squared mismatches.

**Optimization algorithm(s):** The [official PyTorch CIFAR defaults](https://github.com/yang-song/score_sde_pytorch/blob/main/configs/default_cifar10_configs.py) use Adam at $`2\times10^{-4}`$ (0.0002), 5,000-step warmup, gradient clipping, and batch 128. Clipping limits oversized updates. These defaults do not uniquely identify the best deep checkpoint. Noise schedules and predictor/corrector settings are separate choices.

**Regularization techniques:** The recipe averages weights over time, drops units, flips inputs, rescales shortcut paths, and adjusts targets for noise level. These support stability. A continuous range of noise levels does not prevent memorization by itself.

**Backpropagation considerations:** Training normally does not send feedback through the sampling solver. Near-clean inputs can have large score targets, so scaling matters. ODE likelihood is exact only in the ideal mathematical model. Finite solver tolerances and random trace estimates introduce calculation error.

**Parameter count / scaling behavior:** Depth, channels, and attention determine weight count. The version achieving FID 2.20 doubles shortcut blocks per resolution compared with shallower continuous NCSN++. The improvement must not be assigned to every NCSN++ model.

**Training paradigm:** The benchmark learns scores without external conditions, using self-supervised noise targets. Adding labels, classifier guidance, or measurements changes the information supplied to the model.

**Hardware/parallelism considerations:** GPUs or TPUs can process different images and times together. Predictor/corrector calls and divergence probes add work. Batch ODE solving may waste work when some examples require finer tolerances than others.

### 3.8.3 Latent diffusion and Stable Diffusion

**In plain English:** Latent diffusion creates images by removing noise from compact image codes rather than full pixel grids. A decoder turns the finished code into an image, often guided by text.

**Name:** Latent diffusion models (LDMs). Stable Diffusion v1.4 and SDXL 1.0 are treated as different versions here.

**Category & sub-category:** Generation in compact-code space. Diffusion may run without conditions or use text to guide the output.

**Originating paper/vendor/year:** Rombach and colleagues, [*High-Resolution Image Synthesis with Latent Diffusion Models*, arXiv 2021/CVPR 2022](https://arxiv.org/abs/2112.10752). Stable Diffusion v1 saved models were released in 2022. [SDXL](https://arxiv.org/abs/2307.01952) and its 1.0 models were released in 2023.

**Core mechanism:** First train an autoencoder to keep visually useful information in a smaller image code. Then train diffusion on those codes. For text guidance, cross-attention lets each image-code part use relevant text features while predicting noise.

Stable Diffusion v1.4 uses the frozen text Transformer from the CLIP ViT-L/14 package. The [SDXL comparison](https://ar5iv.labs.arxiv.org/html/2307.01952) identifies OpenCLIP ViT-H conditioning for SD 2.0/2.1. SDXL uses two text encoders: CLIP ViT-L plus OpenCLIP ViT-bigG. Their text-vector formats differ, so these saved models cannot simply be swapped.

**Inputs/outputs and typical data types:** Training uses images and, for conditional versions, matching text. A prompt and random code noise produce an image. Image-to-image use also encodes an existing image. SDXL base 1.0 can run alone or use a separate refiner for further processing.

**Strengths and limitations:** A smaller grid reduces noise-prediction work compared with pixels. But the lossy code can discard fine detail, and poor captions limit prompt following. Stronger guidance trades variety for stronger conditioning, not truth. Changing the random seed does not measure reliable uncertainty about the requested scene.

**Computational complexity / scalability notes:** Shrinking both image dimensions greatly reduces the number of positions processed. The whole system does not speed up by exactly that amount: channels, attention, decoding, and step counts also matter. Classifier-free guidance commonly runs predictions both with and without the condition.

**Optional math:** With height $`H`$, width $`W`$, and downsampling factor $`f`$ per side, diffusion uses about $`HW/f^2`$ positions instead of $`HW`$. This does not guarantee an $`f^2`$ total speedup.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [Stable Diffusion v1.4 card](https://huggingface.co/CompVis/stable-diffusion-v1-4) describes filtered LAION image-text training pairs. Fixed image and text encoders prepare codes and text features. The diffusion model learns noise prediction on those codes, then generates from prompts.

Evaluation uses 10,000 COCO2017 validation prompts at 512-by-512, 50 PLMS sampling steps, and several guidance scales. The card compares saved models for image quality and text matching. It also records problems with combining objects correctly, drawing text, bias, and memorization. No unverified chart value is copied here. Compact codes are the advantage over pixel diffusion. The astronaut image is a demonstration, not a study of production design costs. No business performance measure is reported.

**Notable vendor implementations/libraries:** Implementations include [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Stability AI's generative-models](https://github.com/Stability-AI/generative-models), and Hugging Face Diffusers. The [SDXL base 1.0 card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) explains base-only use and the optional refiner.

**Architecture diagram description:**

```text
training image -> fixed image encoder -> latent z -> add noise -> z_t
caption -> frozen text encoder(s) -> cross-attention conditioning
z_t + time + condition -> U-Net -> denoising prediction
sampling: latent noise -> iterative denoising [optional SDXL refiner]
                      -> image decoder -> RGB
```

**Activation functions used and why:** Representative U-Nets use smooth SiLU/swish units, rescale feature groups, and use attention softmax to mix information. Gates control which values pass through some dense layers. Text encoders have their own Transformer activations. Choices need not match across v1, v2, and SDXL.

**Loss function(s):** The LDM autoencoder combines rebuilding and visual-feature comparison losses with code restrictions and GAN feedback. The v1.4 denoiser uses mean squared error on predicted code noise. MSE averages squared errors; lower is better. Other saved models may predict different quantities, so the sampling scheduler must match the checkpoint.

**Optimization algorithm(s):** The v1.4 card specifies AdamW, effective batch 2,048, and 10,000-step warmup to $`10^{-4}`$ (0.0001), then a constant rate. Its v1.4 continuation starts from v1.2 and runs 225,000 steps at 512 resolution. These facts describe v1.4's training history, not SDXL's recipe.

**Regularization techniques:** Training uses code restrictions, filtered data, weight averaging where configured, and omitted text conditions. V1.4 omits text on 10% of examples to learn classifier-free guidance. Safety filtering is a separate safeguard. The loss does not guarantee safe output.

**Backpropagation considerations:** Keep pretrained components fixed when the recipe requires it. Updating them changes both the learning problem and memory use. Guidance during generation is not backpropagation training. Mixed numeric precision needs care in attention, image decoding, and noise-variance calculations.

**Parameter count / scaling behavior:** The SDXL report lists approximately 860 million U-Net parameters for SD 1.4/1.5, compared with 2.6 billion for SDXL. These exclude text encoders, the autoencoder, and any refiner. They are not full-system counts.

**Training paradigm:** Text-conditioned models use paired language supervision as well as automatically created noise targets. They are not caption-free unsupervised systems. Image-only LDM experiments use a different set of teaching signals.

**Hardware/parallelism considerations:** The v1.4 card reports 32-by-8 A100 GPUs and accumulated gradients across smaller batches. Running a trained model on a consumer device does not show it was trained there. SDXL's larger denoiser and optional second model increase generation work and memory.

### 3.8.4 Diffusion Transformer

**In plain English:** DiT uses a Transformer to remove noise from image-code patches. The original model takes a class label, such as an image category, rather than a written prompt.

**Name:** Diffusion Transformer (DiT). This entry covers the original class-conditioned model operating on compact image codes.

**Category & sub-category:** Diffusion generation with a Transformer as the main noise-prediction network.

**Originating paper/vendor/year:** Peebles and Xie, [*Scalable Diffusion Models with Transformers*, arXiv 2022, ICCV 2023](https://arxiv.org/abs/2212.09748). Later Transformers that combine more input types are related designs, not these same saved models.

**Core mechanism:** Divide a noisy autoencoder code into patches. A Transformer mixes information across patches instead of using a U-Net. Time and class vectors control how layers rescale their values. The successful adaLN-Zero design starts its shortcut-path controls at zero. Blocks initially change their inputs very little, which helps larger models train stably.

**Inputs/outputs and typical data types:** The inputs are a noisy image code, a noise step, and a class label. Predictions guide the reverse process toward a completed code. A fixed decoder turns it into RGB pixels. Original DiT is class-conditioned, not inherently a text-to-image system.

**Strengths and limitations:** Its regular blocks make it easier to study larger depth, width, and patch counts. Smaller patches provide finer detail but make attention more expensive. Good class-conditioned images do not demonstrate written-prompt following. Output variety does not give calibrated uncertainty about real objects.

**Computational complexity / scalability notes:** More patches mean more comparisons between patches, as well as more dense-layer work. Smaller patch width therefore increases arithmetic even if weight count barely changes. Generation repeats the network and may add guidance calls.

**Optional math:** One pass costs $`O(L(Td^2+T^2d))`$. Here $`T`$ counts code patches, $`d`$ is vector width, and $`L`$ counts layers. The first term includes dense-layer work; the second includes all-pairs attention. Sampling adds $`S`$ passes, where $`S`$ counts steps, plus any extra guidance evaluations.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The ImageNet experiment tests class-conditioned generation at 256-by-256. A fixed VAE encodes an image. Noisy code patches and the class label train DiT. Repeated reverse steps produce a code that the decoder turns into an image.

The [paper](https://arxiv.org/html/2212.09748v2) reports **FID-50K 2.27** for **DiT-XL/2** after seven million training steps. FID compares real and generated feature distributions; lower is better. "50K" identifies a 50,000-sample evaluation. The result uses guidance scale 1.5, 250 DDPM sampling steps, and ADM's TensorFlow evaluation suite. Unguided runs and 400,000-step scaling tests have different results. Regular blocks and predictable scaling motivate replacing the U-Net. This was not an industrial image service or a reported business benefit.

**Notable vendor implementations/libraries:** The authors' [official PyTorch repository](https://github.com/facebookresearch/DiT) provides weights converted from JAX and multi-worker training code. It notes small differences across frameworks and numeric precision, rather than promising identical output bits.

**Architecture diagram description:**

```text
image latent + noise -> patch embedding + position embedding
class + timestep -> conditioning MLP -> adaLN-Zero controls
patch sequence -> repeated DiT attention/MLP blocks
               -> linear patch output -> unpatchify -> noise/variance prediction
denoised latent -> fixed VAE decoder -> image
```

**Activation functions used and why:** Dense Transformer layers use smooth GELU units; attention softmax weights information from other patches. Condition vectors use SiLU. Adaptive layer normalization lets time and class information adjust feature scaling. Final predictions are real-valued, not category probabilities.

**Loss function(s):** The loss trains noise prediction and a component that learns reverse-step variance, or spread. Sometimes dropping the class condition trains both conditioned and unconditioned predictions for guidance. This does not erase the supervision supplied by class labels.

**Optimization algorithm(s):** The original recipe uses AdamW at constant $`10^{-4}`$ (0.0001), batch 256, **no weight decay and no learning-rate warmup**. Weight averaging uses EMA decay 0.9999. The name AdamW does not prove that its weight-decay option was turned on.

**Regularization techniques:** The paper reports horizontal flips and moving-average weights. Zero-started controls and adaLN-Zero help training, but do not prove good behavior on unseen data. The paper did not require the usual strong set of Vision Transformer training restrictions.

**Backpropagation considerations:** Random-time noise training avoids tracking feedback through the entire sampling chain. Zero-started shortcut controls regulate early feedback. Long patch lists require more attention memory. Activation checkpointing saves memory by recomputing some intermediate results later.

**Parameter count / scaling behavior:** The reported DiT-XL/2 adaLN-Zero denoiser has approximately 675 million parameters, excluding the VAE. At 256 resolution, one forward pass uses about 118.6 GFLOPs in the paper's accounting. GFLOPs means billions of floating-point arithmetic operations. Operations and stored weights measure different things.

**Training paradigm:** This is supervised class-conditioned generation with synthetic noise targets. It appears beside unconditional diffusion to compare network designs. ImageNet class conditioning is not unsupervised.

**Hardware/parallelism considerations:** The paper used JAX on TPU-v3 pods. Official code also splits training batches across GPUs. The cited v3-256 setup and later A100 reproductions describe different hardware and procedures. Neither is a universal requirement.

### 3.8.5 DALL-E 1

**In plain English:** DALL-E 1 reads a caption, then chooses image codes one after another. A decoder turns those codes into pixels; this version does not use diffusion.

**Name:** DALL-E 1. The original system combines a discrete VAE image tokenizer with a next-token Transformer.

**Category & sub-category:** Text-conditioned image generation using discrete codes. It predicts each new code from earlier ones, called autoregression, rather than diffusion.

**Originating paper/vendor/year:** Ramesh and colleagues at OpenAI presented [*Zero-Shot Text-to-Image Generation*, ICML 2021](https://proceedings.mlr.press/v139/ramesh21a.html).

**Core mechanism:** First train a discrete VAE to turn images into grids of code choices, or tokens. Keep that tokenizer fixed. Put each caption's tokens before its image tokens and train a decoder-only Transformer to predict the next token. During generation, supply the caption and let it choose image tokens in order. The image decoder renders the completed grid.

**Inputs/outputs and typical data types:** Internet image-caption pairs train the system. A text prompt produces image candidates. The [paper](https://ar5iv.labs.arxiv.org/html/2102.12092) uses up to 256 BPE text tokens, which are word pieces. It uses 1,024 image tokens in a 32-by-32 grid, with 8,192 possible image-code choices.

**Strengths and limitations:** A single token sequence supports many text descriptions without a separate generator for each class. But image codes lose detail, and generation proceeds sequentially. Temperature changes how adventurous token choices are. Matching-based reranking chooses preferred candidates. Neither gives a calibrated chance that the image combines requested objects correctly.

**Computational complexity / scalability notes:** Tokenizer training is separate from Transformer training. Image attention only checks selected positions using structured masks, so it is not ordinary all-pairs attention. Saving earlier computations helps, but image tokens still arrive one at a time. Producing and ranking many candidates adds substantial work.

**Optional math:** Dense attention's $`T^2`$ pair count, where $`T`$ counts sequence tokens, does not describe this model's actual sparse image masks.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original study](https://ar5iv.labs.arxiv.org/html/2102.12092) trained on approximately 250 million image-text pairs. It tested MS-COCO text-to-image transfer without directly training on that task's split. Captions became tokens; the Transformer generated image codes; the discrete VAE decoded them. A contrastive model then ranked candidates by image-text matching.

The paper reports favorable human comparisons and varied combinations of concepts. Its illustrated comparison picks the best of 512 candidates. That budget matters: it cannot be replaced by a claim about one unselected sample. No unmatched FID, a feature-distribution distance where lower is better, or preference score is substituted here. The advantage over a class-specific GAN is broader language control. "Zero-shot" does not mean no paired supervision or proven absence of source-image overlap. No public business performance measure is established.

**Notable vendor implementations/libraries:** OpenAI released the [DALL-E discrete VAE](https://github.com/openai/DALL-E), not the complete large next-token model. Community implementations and later similarly named systems are not the original proprietary checkpoint.

**Architecture diagram description:**

```text
training image -> dVAE encoder -> 32x32 categorical codes
caption -> BPE text tokens ------+
image codes --------------------+-> causal/sparse Transformer -> next-token loss
generation: caption prefix -> sampled image codes -> dVAE decoder -> image
                                                      -> optional candidate reranking
```

**Activation functions used and why:** The [released dVAE encoder](https://github.com/openai/DALL-E/blob/master/dall_e/encoder.py) and decoder use ReLU hidden units. Softmax turns code or next-token scores into probabilities. Gumbel-softmax makes categorical choices temporarily smooth enough for error feedback during tokenizer training. That differs from VQ-VAE's nearest-code straight-through approximation. These tokenizer facts do not reveal every activation in the unreleased Transformer.

**Loss function(s):** The tokenizer learns to explain images through discrete codes, using a smooth approximation during training. Its pixel-probability rule is called a log-Laplace likelihood. The Transformer then uses cross-entropy to penalize incorrect next-token predictions.

**Technical detail (optional):** The tokenizer uses a relaxed variational objective. Stage two separately normalizes the text and image losses, then weights them $`1/8`$ and $`7/8`$, respectively. These fractions control each part's contribution.

**Optimization algorithm(s):** Both stages use Adam and exponentially averaged weights. The discrete VAE gradually lowers both its code-choice relaxation temperature and step size. The paper lowers that temperature to $`1/16`$, making choices less soft. These are stage-specific learning schedules, not diffusion noise schedules.

**Regularization techniques:** A finite code grid, restricted attention, and training-data processing constrain learning. The contrastive reranker only changes which generated output is selected. It does not regularize the Transformer's trained probability model.

**Backpropagation considerations:** Smooth approximate code choices let feedback train the tokenizer. Transformer training uses fixed image tokens instead. The paper rescales gradients per shortcut block and uses higher-precision shortcut paths. This addresses numbers becoming too tiny or too large during low-precision training.

**Parameter count / scaling behavior:** The reported next-token Transformer has 12 billion parameters and 64 self-attention layers. Tokenizer and reranker are separate. The 12 billion figure does not count every component needed to serve outputs.

**Training paradigm:** Image-only tokenizer training comes first. Learning from natural image-text pairs comes next. Predicting the next token is self-supervised within the paired sequence, but captions still supply meaning-based supervision.

**Hardware/parallelism considerations:** The paper spreads weights across eight GPUs within each machine. All-gather collects needed weights before a block runs; reduce-scatter combines and redistributes gradients afterward. Training resources and the cost of generating many candidates must be counted separately.

### 3.8.6 DALL-E 2

**In plain English:** DALL-E 2 first predicts a compact description of visual content from text. Diffusion models use that description to create an image, then enlarge it.

**Name:** DALL-E 2. This entry describes its published unCLIP generation pipeline.

**Category & sub-category:** Conditional image generation. It predicts a CLIP image-feature vector, decodes it with diffusion, and uses further diffusion models to increase resolution.

**Originating paper/vendor/year:** Ramesh and colleagues at OpenAI described [*Hierarchical Text-Conditional Image Generation with CLIP Latents*, 2022](https://arxiv.org/abs/2204.06125).

**Core mechanism:** CLIP learns image-text matching and can describe an image with a number vector, or embedding. DALL-E 2 trains a prior to predict such an image vector from text. A diffusion decoder generates an image using that vector. Two further diffusion models enlarge it. The study compares autoregressive and diffusion priors, but both use a diffusion image decoder. This differs from DALL-E 1's sequential image-token Transformer.

**Inputs/outputs and typical data types:** Matching captions and images train the components. Text becomes an image vector, then a small image, then a larger image. Encoding an existing image instead supplies a vector for generating variations.

**Strengths and limitations:** CLIP vectors describe broad content and support image variations. They lose some spatial detail, so a vector does not uniquely identify a scene. Errors from the prior and decoder can add up. Variation among outputs is not a calibrated probability distribution over the original image or a guarantee of prompt correctness.

**Computational complexity / scalability notes:** Count prior generation, base-image generation, and both enlargement stages. A smaller enlargement network may still do heavy work on a large pixel grid. Guidance and candidate selection change both computation and which outputs are likely.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [unCLIP study](https://ar5iv.labs.arxiv.org/html/2204.06125) tests zero-shot MS-COCO caption-to-image generation and image variations. A caption drives the prior's CLIP image vector. The decoder makes a 64-by-64 image; upsamplers enlarge it to 256 and 1,024 resolution. People and metrics then evaluate the results.

The paper reports favorable zero-shot image quality and compares the two priors. It chooses diffusion for its cost/quality trade-off. That is a documented research choice, not proof of a preference among all commercial users. The generation stages use the approximately 250-million-image DALL-E dataset. CLIP encoder training uses a different mixture. No audited public business measure is reported, and this recipe does not describe every later production revision.

**Notable vendor implementations/libraries:** OpenAI operated the proprietary DALL-E 2 service. Open-source unCLIP-style systems reproduce ideas, not the original production weights. Similar library interfaces do not make checkpoints equivalent.

**Architecture diagram description:**

```text
caption -> text features -> prior [AR OR diffusion] -> CLIP image embedding
                                                      |
noise + embedding [+ text] -> diffusion decoder -> 64x64 image
                         -> diffusion upsampler -> 256x256
                         -> diffusion upsampler -> 1024x1024
```

**Activation functions used and why:** The published decoder follows GLIDE's shortcut-and-attention design. Its [U-Net code](https://github.com/openai/glide-text2im/blob/main/glide_text2im/unet.py) uses smooth SiLU units and softmax attention weights. The diffusion prior outputs continuous vector values. The autoregressive prior needs discrete-choice output layers. These components do not reveal undisclosed service changes.

**Loss function(s):** The diffusion prior predicts the clean CLIP image vector, penalizing squared errors. The base decoder learns noise and reverse-step variance, or spread. Upsamplers learn to reverse noise while using a smaller image as a condition. Frozen CLIP features guide generation; CLIP similarity is not the whole generator loss.

**Optimization algorithm(s):** The [paper's settings table](https://ar5iv.labs.arxiv.org/html/2204.06125) specifies Adam, including diffusion-prior rate $`1.1\times10^{-4}`$ and base-decoder rate $`1.2\times10^{-4}`$. Weight averaging differs by stage. The prior, base decoder, and first upsampler use cosine-shaped noise schedules. The last upsampler uses a linear noise schedule. These describe damage, not every learning-rate change in a commercial run.

**Regularization techniques:** Occasionally removing conditions supports guidance. Damaging input images helps upsamplers handle imperfect inputs. The prior uses weight decay; dropout and moving-average settings differ by stage. CLIP weights stay fixed while the prior and decoder train.

**Backpropagation considerations:** Each stage has its own training goal. Feedback does not run jointly through fixed CLIP weights and the full sampled pipeline. Scaling CLIP vector values matters. Predicting variance and using several conditions require care when mixing numeric precision levels.

**Parameter count / scaling behavior:** The paper lists roughly 1 billion parameters for either prior and 3.5 billion for the base decoder. The upsamplers have 700/300 million. These are component counts, not a complete inventory of the DALL-E 2 service.

**Training paradigm:** Natural image-text pairs supply supervision alongside diffusion targets. CLIP pretraining and the generation stages are separate. The larger CLIP data mixture must not be described as the decoder's training dataset.

**Hardware/parallelism considerations:** Large components benefit from splitting examples or model weights across accelerators. Recomputing saved intermediate results can trade work for memory. The checked paper does not give a full production hardware inventory. Device counts and training cost cannot be inferred from API prices.

### 3.8.7 DALL-E 3

**In plain English:** DALL-E 3's published research shows how richer image descriptions can improve prompt following. Its complete image-generator design is not publicly disclosed in that report.

**Name:** DALL-E 3. It is a proprietary text-to-image system; the published study focuses on better training captions.

**Category & sub-category:** Conditional image generation using more descriptive replacement captions. Its placement here does not imply that the report discloses a detailed noise-prediction network.

**Originating paper/vendor/year:** Betker and colleagues at OpenAI, with Microsoft collaborators, published [*Improving Image Generation with Better Captions*, 2023](https://cdn.openai.com/papers/dall-e-3.pdf).

**Core mechanism:** Train a captioning system to write richer descriptions of images. Use those descriptions as training pairs and test whether generation follows text better. The report discusses text-to-image diffusion, but explicitly leaves out DALL-E 3's full training and implementation. It does not verify a U-Net, DiT, image-code system, layer layout, or DALL-E 2-style prior for the full generator.

**Inputs/outputs and typical data types:** Images have original or generated descriptions during training. A text request produces images. A language model may separately expand the request before generation. That does not prove the image generator itself predicts language tokens in sequence.

**Strengths and limitations:** Detailed captions can reveal object properties and relationships missing from messy web descriptions. Captioners can also invent details, omit information, or favor certain styles. Better prompt following does not guarantee factual images or success on every request. Output variety and safety decisions are not calibrated probabilities of correctness.

**Computational complexity / scalability notes:** Writing new dataset captions adds work before image-generator training: images must be encoded and descriptions generated. The report lacks enough architecture and recipe details to calculate generator training or sampling operations, often measured as FLOPs. Numerical estimates need further disclosed assumptions. Borrowing DALL-E 2's weight count would not supply them.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [report](https://cdn.openai.com/papers/dall-e-3.pdf) studies DrawBench, COCO-derived captions, and tests of combining requested concepts correctly. It replaces noisy captions with richer descriptions, trains image generation, and evaluates outputs with people and automated measures. It reports improved prompt following and favorable comparisons, while retaining limitations.

The [official evaluation repository](https://github.com/openai/dalle3-eval-samples) releases four images per DrawBench prompt. It distinguishes original from expanded prompts and explicitly calls these benchmark samples, not curated product demonstrations. Better supervision is the supported explanation, not an invented advantage from unknown network internals. No public audited business performance measure is provided.

**Notable vendor implementations/libraries:** OpenAI supplies proprietary DALL-E 3 interfaces and [official evaluation artifacts](https://github.com/openai/dalle3-eval-samples). Samples are not model weights or enough code to reproduce training. Service availability and which model a product uses can change independently of this historical description.

**Architecture diagram description:**

```text
DISCLOSED TRAINING-DATA PATH:
image + existing caption -> image-captioning system -> richer paired description
paired images/descriptions -> text-to-image training [full generator undisclosed]

PRODUCT-LEVEL GENERATION PATH:
text request -> [optional prompt expansion] -> proprietary image generator -> image
No verified internal U-Net/DiT/codec/prior diagram is publicly supplied here.
```

**Activation functions used and why:** The full generator's activation and normalization choices are not publicly disclosed in the cited report. The captioner predicts word-piece probabilities. This does not reveal the image generator's hidden units. Naming GELU, SiLU, or a particular rescaling method for unknown blocks would invent details.

**Loss function(s):** The captioner learns to predict language from images. Its pretraining combines image-text matching with language prediction: matching rewards correct pairs over wrong ones. The exact final image-generator loss is not sufficiently disclosed. Neither are its noise target, extra penalties, or their weights, so a complete verified formula cannot be given.

**Optimization algorithm(s):** The report does not publicly specify the generator's optimizer, learning-rate schedule, gradient limits, or detailed noise and sampling schedules. DALL-E 2's Adam settings do not establish DALL-E 3's settings.

**Regularization techniques:** Mixing original and generated captions helps address repeated patterns introduced by the captioner. Tests of caption mixtures include **95% synthetic-caption conditions**. That experimental fraction is not a fully disclosed final production recipe. Exact weight decay, dropout, and normalization remain undisclosed.

**Backpropagation considerations:** Captions generated beforehand are fixed text training data. They do not show that error feedback runs through caption generation and image generation together. The report does not establish the final generator's gradient method, numeric precision handling, or coordination across workers.

**Parameter count / scaling behavior:** The full DALL-E 3 generator's parameter count is **not publicly disclosed** in this source. Resolution, output quality, and API response time do not reliably reveal weight count, number of expert subnetworks, or computation actually used.

**Training paradigm:** Natural image-text pairs and generated descriptions supply language supervision. Generated captions do not erase that supervision; the captioner also learned from earlier data. Public tests and prompt expansion are separate from the unknown complete image-training recipe.

**Hardware/parallelism considerations:** Exact training accelerators, fleet size, division of weights across devices, and utilization are undisclosed. Large captioning and generation jobs generally benefit from accelerators. That general observation is not a sourced DALL-E 3 hardware specification.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| DDPM | Images and data with continuously valued noise | Learns by predicting known added noise | Original generation takes many steps | CIFAR-10 generation without class conditions |
| Score-based SDE models | Continuous data; reconstruction from measurements | Offers several continuous-time sampling rules | Imperfect directions and numerical steps cause error | Deep continuous NCSN++ tests on CIFAR-10 |
| Latent diffusion / Stable Diffusion | Images, optionally paired with text or image conditions | Predicts noise on smaller code grids | Lost code detail and version differences | COCO2017 tests of Stable Diffusion v1.4 |
| DiT | Conditioned patches of image codes | Regular blocks make scaling easier to study | Many patches cost more; original model uses class labels | ImageNet tests of DiT-XL/2 |
| DALL-E 1 | Matching images and captions | Generates image codes from a text prefix | Sequential code choices and lost detail | MS-COCO zero-shot generation tests |
| DALL-E 2 | Matching images/text and image-feature vectors | Generates through CLIP vectors and supports variations | Several stages add errors and work | unCLIP caption and image-variation tests |
| DALL-E 3 | Images paired with detailed descriptions | Richer captions improve prompt following | Unknown full recipe and remaining mistakes | DrawBench and concept-combination tests |

## 3.9 Flows and autoregression

Flows transform data through steps that can each be undone. By tracking how these steps stretch or shrink values, they can calculate a continuous probability density. Density describes how probability is spread across ranges; it is not itself the probability of one exact value.

Autoregressive models instead predict each new value using earlier values. Both approaches train by making observed data more likely, but generation costs differ. Image pixels often receive small added noise, called dequantization, before continuous modeling. A density at those noisy values is not automatically an exact probability for the original discrete image. High likelihood alone also does not reliably identify whether data is familiar or appropriate.

### 3.9.1 RealNVP

**In plain English:** RealNVP learns reversible steps between data and a simpler random code. It can both encode an example and run the steps backward to generate a new one.

**Name:** Real-valued non-volume-preserving transformation model, or RealNVP. Its transformations can stretch or shrink the space of values.

**Category & sub-category:** Unsupervised density estimation using normalizing flows. Its coupling steps scale and shift one part of the data using another part.

**Originating paper/vendor/year:** Dinh, Sohl-Dickstein, and Bengio presented [*Density Estimation using Real NVP*, arXiv 2016, ICLR 2017](https://arxiv.org/abs/1605.08803). It extends earlier flows that used additive coupling.

**Core mechanism:** Split the input into two parts. Keep one part unchanged. Use it to predict how to scale and shift the other. Because the first part remains available, the second change can be undone. Repeated layers swap which values they hold fixed and work at several scales. This lets all values eventually influence one another.

**Optional math:** A layer uses $`y_a=x_a,\ y_b=x_b\odot\exp s(x_a)+t(x_a)`$. Here $`x_a,x_b`$ are input parts, $`y_a,y_b`$ output parts, and $`s,t`$ learned log-scale and shift functions. The symbol $`\odot`$ means coordinate-by-coordinate multiplication. The Jacobian records how outputs change with inputs. Its triangular structure makes its log determinant, the total log stretch, a sum of predicted log scales.

**Inputs/outputs and typical data types:** Inputs are continuous number lists or images with suitable preprocessing and added noise. Outputs include codes, continuous log densities, and generated data from reverse steps. The internal networks predicting scale and shift need not be reversible themselves.

**Strengths and limitations:** The model provides exact mathematical inversion and manageable continuous-density calculations, unlike ordinary GANs or approximate-code VAEs. Each coupling layer is restricted, so many layers may be needed. High likelihood can reward superficial statistics rather than meaningful familiarity. It is not automatically a calibrated chance of an anomaly.

**Computational complexity / scalability notes:** Most work is in the scale-and-shift networks. The stretch calculation grows directly with transformed values, rather than with the cube of total input size. There is no need to build the whole Jacobian matrix. Layers run in order, but each handles many values in parallel.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [paper](https://ar5iv.labs.arxiv.org/html/1605.08803) models CIFAR-10, downsampled ImageNet, LSUN, and CelebA. Preprocessing and reversible layers map an image to Gaussian codes while tracking total stretch. This supports both likelihood evaluation and reverse generation.

It reports **3.49 bits/dimension on CIFAR-10's test set**, compared with PixelRNN's listed 3.00. Bits/dimension is a likelihood-based coding measure per pixel-channel value; lower is better. The number depends on the paper's continuous preprocessing and dequantization rules. It must not be read as an exact discrete-image probability. RealNVP offers manageable density calculations and fast reverse sampling, not the best result on every likelihood test. This is not a measured compression-service rate or a commercial saving.

**Notable vendor implementations/libraries:** TensorFlow Probability's reversible transformations, called bijectors, and Pyro coupling layers provide RealNVP-style parts. A generic scale-and-shift flow need not match the paper's full multiscale network.

**Architecture diagram description:**

```text
x -> split [x_a, x_b]
x_a -> neural s,t -> scale/shift x_b -> concatenate
  -> alternate mask / squeeze / additional coupling blocks
  -> latent z with simple prior
sampling: prior z -> inverse coupling blocks -> x
```

**Activation functions used and why:** A typical scale-and-shift network uses ReLU and shortcut paths. These internal networks need not be reversible. The full coupling step uses an exponential to keep scales positive and leaves shifts unbounded. Limiting log-scale values helps stability. ReLU belongs inside the prediction network, not as a replacement for the reversible data step.

**Loss function(s):** Make observed data likely by combining the code's density with the stretch of all transformations. Include preprocessing changes too. Adding uniform noise to discrete pixels gives a bound-related training objective. It does not turn point density into the exact pixel probability.

**Optional math:** Minimize $`-\log p_Z(f_\theta(x))-\log|\det J_{f_\theta}(x)|`$. Here $`x`$ is the input and $`f_\theta`$ the flow with weights $`\theta`$. The code prior has density $`p_Z`$. The Jacobian $`J`$ describes local stretching; the absolute determinant measures its size. Its log corrects the code density for that change.

**Optimization algorithm(s):** The paper uses Adam with its stated default settings. It does not require one rate-decay schedule. Reproductions must record optimizer settings and when training stops; these cannot be read from the flow equation alone.

**Regularization techniques:** The original method uses shortcut subnetworks, normalization, and an L2 penalty, which penalizes squared weight-scale values. Reversible normalization outside those subnetworks also stretches data. Its log determinant must be included or the calculated density is wrong.

**Backpropagation considerations:** Feedback must include both prior density and stretch terms. Extreme log scales can overflow or make reversal numerically unstable. Running normalization statistics must be used consistently during evaluation. Transformations depending on batchmates require special care when interpreting density.

**Parameter count / scaling behavior:** Coupling-network size and the number of scales and layers determine weight count. Reversibility does not imply a small model. Alternating which values are held fixed adds flexibility without requiring every internal network to be an invertible square matrix.

**Training paradigm:** The representative image model learns without labels by increasing training-data likelihood. Using labels to choose scales and shifts creates a supervised conditional flow.

**Hardware/parallelism considerations:** GPUs process coupling filters across image positions and examples. Reversing calculations can save intermediate memory, but adds work and numerical risk. Memory does not automatically stay constant just because equations are reversible.

### 3.9.2 Glow

**In plain English:** Glow learns reversible image transformations and how to mix image channels at each step. It can map an image to a code, then reverse that route to generate or modify images.

**Name:** Glow. It adds learned, reversible one-by-one image-channel mixing to a generative flow.

**Category & sub-category:** Unsupervised density estimation with reversible neural transformations at several image scales.

**Originating paper/vendor/year:** Kingma and Dhariwal at OpenAI presented [*Glow: Generative Flow with Invertible 1x1 Convolutions*, NeurIPS 2018](https://arxiv.org/abs/1807.03039).

**Core mechanism:** Each step first rescales features, then reversibly mixes channels at each position, then applies a scale-and-shift coupling step. Instead of always swapping channels in a fixed order, Glow learns a mixing matrix it can undo. Rearranging grids and splitting off some code values builds several scales. The modeled continuous transformation remains mathematically reversible.

**Inputs/outputs and typical data types:** An image becomes code values at several scales, plus a log density. Sampled codes become images. Direct encoding also allows gradual code changes or movement between image codes, followed by decoding.

**Strengths and limitations:** Learned mixing makes each layer more flexible while preserving manageable density calculations. Values within a layer can still be processed together. Training may need much memory, and strong likelihood need not mean meaningful-looking images. Low-temperature sampling narrows code choices to favor visual quality. It changes the fitted distribution rather than calibrating uncertainty.

**Computational complexity / scalability notes:** Channel mixing work grows with image area and roughly the square of channel count. A special matrix form makes the stretch calculation cheap, but not the mixing itself. Coupling filters and additional scales also add work.

**Optional math:** With height $`H`$, width $`W`$, and $`c`$ channels, mixing costs $`O(HWc^2)`$. A naive determinant costs $`O(c^3)`$. Storing the matrix as lower and upper triangular factors, called LU, reduces its log-determinant sum to $`O(c)`$. It does **not** reduce matrix application to $`O(c)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [Glow paper](https://ar5iv.labs.arxiv.org/html/1807.03039) reports **3.35 bits/dimension on CIFAR-10**, versus RealNVP's listed 3.49. This is a likelihood-based coding measure per pixel-channel value; lower is better. Its interpretation depends on the specified pixel-noise and preprocessing rules. Images pass through normalization, channel mixing, and coupling to produce codes and densities. Reversing those steps generates images.

Learned mixing improves on fixed channel swaps. Compared with a GAN, the appeal is manageable density calculation and direct reversal. A separate CelebA-HQ visual-quality and code-editing experiment uses 256-resolution, **five-bit** images. Those faces do not establish the same eight-bit likelihood or compression result as CIFAR-10. These are research results, not audited production benefits.

**Notable vendor implementations/libraries:** OpenAI's [official Glow repository](https://github.com/openai/glow) supplies TensorFlow code. Other flow libraries may reproduce individual steps. Different added pixel noise, bit depth, or coupling settings change the model.

**Architecture diagram description:**

```text
image -> squeeze -> [ActNorm -> invertible 1x1 conv -> affine coupling] repeated
      -> split off latent portion -> repeat at coarser scale -> remaining latent
all latent portions + prior log densities -> total image log density
sampling reverses every step
```

**Activation functions used and why:** The checked coupling networks use ReLU. A shifted sigmoid keeps the scale positive and controlled. ActNorm learns a scale and shift for each channel, starting from data statistics. Unlike batch normalization, it does not recalculate those statistics for every minibatch.

**Loss function(s):** The loss is negative continuous log likelihood; it rewards higher density for observed data. Count the stretch from ActNorm, channel mixing, and coupling, plus the prior at every scale. The continuous formula is exact, but discrete images still require the specified preprocessing and added-noise rules.

**Optimization algorithm(s):** The [official training defaults](https://github.com/openai/glow/blob/master/train.py) use **Adamax** at learning rate 0.001. Warmup raises the rate evenly over ten epochs. Adam is also an option in the code. Neither the defaults nor every Glow run should be casually described as mandatory Adam.

**Regularization techniques:** Data-based ActNorm starting values and zero-started final coupling filters make early training easier. Adding noise to discrete pixels prevents the model from fitting extreme density spikes only at allowed pixel values. Lower sampling temperature changes generation, not training regularization.

**Backpropagation considerations:** Keep all stretch terms in the feedback path. A reversible matrix can still amplify tiny numeric errors badly. LU makes determinants easier to calculate but does not prevent that instability. Log determinants require care when numeric precision varies.

**Parameter count / scaling behavior:** The number of scales, steps per scale, and coupling-network width determine size. Coupling networks usually hold more weights than channel mixing. **Optional math:** A mixing matrix for $`c`$ channels adds $`c^2`$ parameters. There is no fixed count for all Glow models.

**Training paradigm:** The cited likelihood tests train without labels. Studying edits to specific image attributes may use labeled analysis or selected examples. Such later uses are not necessarily unsupervised.

**Hardware/parallelism considerations:** Official code supports several accelerators and checkpointing that recomputes intermediate results to save memory. Reversibility offers further recomputation options. Storing versus rebuilding results across scales still involves trade-offs in time, memory, and accuracy.

### 3.9.3 PixelCNN

**In plain English:** PixelCNN creates an image by predicting pixel values in order. It uses learned filters that cannot peek at values the model has not generated yet.

**Name:** PixelCNN. The original masked-filter model and Gated PixelCNN are different versions.

**Category & sub-category:** Autoregressive generation. It assigns probabilities to discrete image intensities using earlier pixel and color values.

**Originating paper/vendor/year:** Van den Oord, Kalchbrenner, and Kavukcuoglu introduced PixelCNN in [*Pixel Recurrent Neural Networks*, ICML 2016](https://arxiv.org/abs/1601.06759). Van den Oord and colleagues added the gated/conditional version in [*Conditional Image Generation with PixelCNN Decoders*, NeurIPS 2016](https://arxiv.org/abs/1606.05328).

**Core mechanism:** Choose an order for pixel and color values. Predict each value from the earlier ones. A mask is a rule blocking forbidden connections, not random damage to the image. It prevents future pixels and disallowed same-pixel colors from revealing the answer. Training can process a known image in parallel while obeying these rules. Generation must choose each new value before proceeding. Gated PixelCNN uses separate vertical and horizontal paths to reach earlier pixels missed by the original filters.

**Inputs/outputs and typical data types:** Inputs are discrete image intensities, sometimes with a class label or feature-vector condition. Outputs give probabilities for the next intensity. Repeating the choice builds an image. Adding all conditional log probabilities gives the image's log probability directly.

**Strengths and limitations:** The model can assign probability to several possible next values without approximating a hidden-code distribution. Generation remains sequential despite parallel training. Limited context can miss the overall scene. A confident next-pixel choice does not prove that the whole image makes sense or resembles familiar data.

**Computational complexity / scalability notes:** Training supplies known earlier values, called teacher forcing. One masked-network pass covers the whole grid, with work equal to its filter calculations. Repeating a full-grid pass for every generated value is costly. Caching saves repeated work but cannot remove the required order. Sequential decoding alone does not make PixelCNN a recurrent network.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [Gated PixelCNN study](https://ar5iv.labs.arxiv.org/html/1606.05328) tests CIFAR-10 image probabilities. Known pixels pass through masked vertical and horizontal filters. Each predicted intensity has 256 choices. Adding test-image negative log probabilities gives the likelihood measure; choosing values in sequence instead generates an image.

The paper reports **3.03 bits/dimension**, versus 3.14 for earlier PixelCNN and 3.00 for PixelRNN. This is average negative log probability per image value, measured in bits; lower is better. The advantage is parallel filter training with likelihood near PixelRNN, not fully parallel generation. Conditional ImageNet experiments also use class labels. These numbers do not establish a commercial compression rate, deployed response time, or business benefit.

**Notable vendor implementations/libraries:** Research code and TensorFlow Probability support PixelCNN-style image models. PixelCNN++ changes both network design and likelihood. Similar names do not justify assigning it the original or gated model's results.

**Architecture diagram description:**

```text
image -> mask-A convolution [exclude current/future target]
      -> mask-B residual layers [allowed earlier-channel context]
      -> pointwise heads -> 256-way distribution per RGB channel
gated variant: vertical stream -> horizontal stream -> gated residual heads
sampling: fill pixels/channels in causal order
```

**Activation functions used and why:** Original PixelCNN uses ReLU hidden layers and softmax probabilities over intensities. Gated PixelCNN multiplies a bounded tanh signal by a sigmoid gate that controls how much passes. Binary MNIST can use a single probability for each binary pixel instead of 256 intensity categories.

**Loss function(s):** Penalize assigning low probability to each observed next value. Summing these penalties gives exact autoregressive negative log likelihood. In this categorical model, it is discrete cross-entropy, not an approximate squared rebuilding error.

**Optional math:** The loss is $`-\sum_i\log p_\theta(x_i\mid x_{<i})`$. Here $`i`$ is position in the chosen order, $`x_i`$ its value, and $`x_{<i}`$ all allowed earlier values. $`p_\theta`$ is the probability model with weights $`\theta`$. A supplied condition $`c`$ can be added to every prediction.

**Optimization algorithm(s):** The original PixelRNN/PixelCNN paper uses RMSProp, which adapts update sizes using recent squared gradients. It manually selects learning-rate schedules for each dataset. These are not one universal schedule for all later gated or conditional implementations.

**Regularization techniques:** Masks enforce prediction order; they do not randomly drop units. Original experiments use small batches on smaller datasets. They do not modify images beyond scaling and centering values. Shortcut connections make the network easier to train.

**Backpropagation considerations:** A wrong mask can expose the answer and produce misleadingly low loss. Teacher forcing uses known context, avoiding feedback through sampled categories. At generation time, the model sees its own earlier choices instead. Their mistakes can accumulate.

**Parameter count / scaling behavior:** Layer count, channel width, and the number of output choices determine weight count. The original model describes fifteen image-filter layers. The larger gated ImageNet setup differs. Keeping probabilities at every position also takes substantial intermediate memory.

**Training paradigm:** Unconditional models use self-supervised next-pixel prediction. Gated class-conditioned models use labels. When a feature vector supplies the condition, its own training history determines what supervision it carries.

**Hardware/parallelism considerations:** GPUs train the image filters efficiently and can split batches across devices. The gated paper reports multi-GPU experiments. Sequential generation still limits the speed of one image; more GPUs mainly create more independent images at once.

### 3.9.4 WaveNet

**In plain English:** WaveNet builds sound by predicting the next tiny audio sample from earlier ones. The original research model and the faster version used in Google Assistant are not the same system.

**Name:** WaveNet. This entry explains the original sequential waveform model and identifies later production revisions separately.

**Category & sub-category:** Autoregressive raw-audio generation. It can model sound alone or use conditions to synthesize speech.

**Originating paper/vendor/year:** Van den Oord and colleagues at DeepMind published [*WaveNet: A Generative Model for Raw Audio*, 2016](https://arxiv.org/abs/1609.03499).

**Core mechanism:** Predict a probability distribution for the next sound sample using only earlier samples. Causal filters block future sound. Dilated filters leave gaps between the earlier positions they inspect, reaching farther back without equally many extra layers. Language features, acoustic features, or speaker identity can guide predictions. Sound-only learning and text-conditioned speech use different supervision.

**Inputs/outputs and typical data types:** Training uses waveform sequences, optionally with aligned language or acoustic features. Outputs are next-sample probabilities and generated sound. The original commonly described output uses eight-bit mu-law companding, which rescales sound amplitude before dividing it into 256 categories.

**Strengths and limitations:** Learning the waveform directly avoids some restrictions of hand-designed sound-synthesis components, called vocoders. It can capture rich short-range structure. But audio requires many sequential samples, making generation expensive. Amplitude categories, available history, and condition quality all matter. A likely sound continuation does not give calibrated confidence that its spoken claim is correct.

**Computational complexity / scalability notes:** Training supplies known earlier samples, called teacher forcing, and processes many positions together. More audio positions and wider filters add work. Cached generation avoids recalculating old history, but still emits samples in order. Increasing filter gaps expands history without adding filter weights.

**Optional math:** Training costs about $`O(BT\sum_l k_lc_l^2)`$ for same-width causal layers. Here $`B`$ is batch size, $`T`$ sequence length, $`k_l`$ filter length in layer $`l`$, and $`c_l`$ its channel width.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Sourced application.** DeepMind's [October 4, 2017 announcement](https://deepmind.google/blog/wavenet-launches-in-the-google-assistant/) says an **updated WaveNet** generated Google Assistant's US English and Japanese voices. It explicitly describes the original model as too expensive computationally for consumer products. Requested response text supplied language and acoustic conditions; a waveform generator produced Assistant's audio. Direct learned sound generation addresses naturalness limits in methods that join recorded speech or use hand-designed speech parameters. The announcement documents adoption, not measured cost savings.

Separately, the [2016 paper](https://ar5iv.labs.arxiv.org/html/1609.03499) reports English five-point listener **MOS 4.21**, versus **3.86** for its baseline. MOS is mean opinion score: an average listener rating, with higher better on this scale. The research WaveNet used language features plus F0, the fundamental frequency associated with pitch. The HMM-driven concatenative baseline used a hidden Markov model to choose and join recorded speech. These are original research-condition scores, not measurements of the updated 2017 production system.

**Notable vendor implementations/libraries:** DeepMind/Google provide research and the documented Assistant application; community implementations also exist. Distilled or parallel WaveNet and later learned vocoders change the design or training. They are not automatically the original sequential sampler.

**Architecture diagram description:**

```text
past mu-law audio -> causal convolution
  -> repeated dilated residual blocks:
       tanh(filter + condition) * sigmoid(gate + condition)
       -> residual path + skip path
summed skip outputs -> ReLU / pointwise layers -> 256-way next-sample softmax
```

**Activation functions used and why:** A tanh signal carries content; a sigmoid gate controls how much passes through. Residual and skip paths carry information and error feedback around layers. ReLU processing and final softmax convert combined outputs into probabilities for the original sample categories.

**Loss function(s):** Cross-entropy penalizes assigning low probability to each observed next sound category. This predicts waveform values, not a sound-frequency chart using mean squared error. Later continuous-output or teacher-student distillation versions need their own losses.

**Optional math:** The loss is $`-\sum_t\log p_\theta(x_t\mid x_{<t},c)`$. Here $`x_t`$ is the sample at time $`t`$, $`x_{<t}`$ earlier samples, and $`c`$ any supplied condition. The probability model has weights $`\theta`$. The sum adds next-sample penalties across the sequence.

**Optimization algorithm(s):** Gradient updates train the whole probability model. The checked research report does not specify a complete original optimizer and learning-rate schedule. The production announcement does not disclose one either. Adam with warmup and decay would be a proposed recipe, not verified Google settings.

**Regularization techniques:** Past-only access, limited history, and specified conditions restrict what the model can learn. Shortcut connections improve training stability. The product name does not establish a particular dropout or weight-decay value.

**Backpropagation considerations:** Training processes known target samples in parallel; feedback does not pass through random categorical choices. Limit unstable gradients where tests justify it. Carefully align conditions and pad the available history so future sound cannot leak into a prediction.

**Parameter count / scaling behavior:** Channel widths, repeated dilation patterns, condition networks, and output layers determine size. Higher audio sample rates add more sequence work without necessarily adding weights. The production announcement does not disclose model counts.

**Training paradigm:** Sound-only next-sample prediction is self-supervised. Text-to-speech, or TTS, uses paired language/acoustic supervision. WaveNet appears here because of its sequential prediction mechanism, not because Assistant speech synthesis is unsupervised.

**Hardware/parallelism considerations:** GPUs train these filter sequences efficiently. Even cached original generation must meet strict sample-by-sample timing limits. The documented production update addressed prototype speed limits, but its full hardware and parallelization recipe is not established here.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| RealNVP | Continuous values; images with added small noise | Calculates density and reverses codes directly | Restricted layers; likelihood can miss meaning | CIFAR-10 test-likelihood research |
| Glow | Natural images at several scales | Learns reversible channel mixing | High memory use and numerical reversal errors | CIFAR-10 and five-bit face-image tests |
| PixelCNN | Discrete pixel intensities | Gives probabilities for each next pixel value | Generation stays sequential | CIFAR-10 tests of Gated PixelCNN |
| WaveNet | Raw sound, optionally with aligned conditions | Learns sound samples directly | Original model must produce samples in sequence | 2017 Assistant voices used an updated WaveNet |

## 3.10 Representation pretraining

These methods learn useful number-based descriptions, or representations, rather than necessarily creating new data. A practice task teaches a feature extractor. A projection head is a small extra network that prepares features for that task. A later evaluator measures how useful the retained features are. These parts do different jobs.

**Contrastive matching** rewards related views or pairs for receiving similar vectors, compared with mismatched pairs. Some methods match two modified views of the same image. CLIP matches an image with its caption. Other methods below learn without explicit mismatches, or rebuild hidden image patches.

Evaluation procedures also differ. **Linear probing** keeps the encoder fixed and trains a labeled classifier using weighted sums of features. **Fine-tuning** updates the encoder too. **Nearest-neighbor evaluation** compares features with a reference set that has labels. Their scores cannot be treated as the same test.

The comparisons retain each paper's main network and training budget. A 200-epoch ResNet-50, a wider 1,000-epoch network, and a fine-tuned ViT-H do not isolate the effect of the loss alone. CLIP appears here because it learns image-and-language representations. It uses genuine paired language supervision.

The last two entries cover **static word vectors**: each known word has one learned number list, regardless of its sentence. Word2Vec and GloVe are shallow models with linear or two-vector scoring rules, not deep hidden networks. They use the neural field template because their vectors are trainable. Compare them with the sentence-dependent token representations in [foundation models](06-foundation-models.md).

### 3.10.1 SimCLR

**In plain English:** SimCLR learns that two changed views of the same image should still match. It keeps the resulting image features for later tasks such as classification.

**Name:** Simple Framework for Contrastive Learning of Visual Representations (SimCLR). This entry covers the original 2020 version.

**Category & sub-category:** Self-supervised image-feature learning. It practices matching modified image views before learning a later task.

**Originating paper/vendor/year:** Chen, Kornblith, Norouzi, and Hinton at Google Research presented [*A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020](https://arxiv.org/abs/2002.05709).

**Core mechanism:** Make two independently modified views of each image, for example through cropping and color changes. One shared encoder describes both views. A small projection network then prepares their vectors for comparison. Training brings matching views closer relative to other images in the batch. Later tasks normally use the encoder features before projection. Features best for the matching exercise are not necessarily best for a new task.

**Inputs/outputs and typical data types:** Unlabeled images enter pretraining. Training outputs are projection vectors scaled to equal length. Later outputs are encoder features. A classifier still needs labels or another stated rule for making decisions.

**Strengths and limitations:** The idea is simple and benefits from suitable image changes and larger training scale. Big batches cost memory. Different images of the same category are treated as mismatches, called false negatives. Severe crops may remove important details. Matching probabilities describe the batch exercise, not calibrated uncertainty over real-world classes.

**Computational complexity / scalability notes:** Two views double encoder work compared with one view. Comparing every projection with the others grows roughly with the square of batch size. Holding all comparison scores also takes memory. Workers may gather vectors together or divide the comparisons among devices.

**Optional math:** For batch size $`B`$, there are $`2B`$ view vectors. If each has width $`d_z`$, pair comparisons cost $`O(B^2d_z)`$. A full similarity table uses $`O(B^2)`$ memory, in addition to network intermediate results.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [ImageNet study](https://ar5iv.labs.arxiv.org/html/2002.05709) pretrains on images without class labels. Modified pairs train the encoder and projector. The encoder is then frozen, and a labeled linear classifier predicts ImageNet validation classes.

In the long-training results, top-1 accuracy is **69.3% for ResNet-50** and **76.5% for a four-times-width ResNet-50**. Top-1 means the percentage whose first-choice class is correct. The headline 76.5% is not the ordinary-width network. The paper separates 100-epoch component tests from 1,000-epoch high-performance training. Unlike reconstruction, matching can teach features that ignore chosen image changes without a pixel decoder. These benchmarks do not establish production inspection savings or a public business measure.

**Notable vendor implementations/libraries:** Google provides the [official SimCLR repository](https://github.com/google-research/simclr). Later SimCLRv2 models and semi-supervised teacher-student training stages use different recipes.

**Architecture diagram description:**

```text
image -> augmentation A -> shared encoder f -> h_A -> MLP g -> z_A
      -> augmentation B -> shared encoder f -> h_B -> MLP g -> z_B
z_A,z_B + other batch projections -> normalized contrastive loss
after pretraining: retain f; discard g; train/evaluate downstream head on h
```

**Activation functions used and why:** The ResNet uses ReLU, as does the hidden layer of the two-layer projection network. L2 normalization gives vectors unit length, so their dot product measures directional, or cosine, similarity. A temperature setting controls how sharply softmax favors the best matching view.

**Loss function(s):** The loss rewards the other view of the same image over competing views. Each view takes a turn as the query. The method averages over both directions and all images. This loss is called NT-Xent.

**Optional math:** For matched views $`i,j`$, use $`-\log[\exp(\operatorname{sim}(z_i,z_j)/\tau)/\sum_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)]`$. Here $`z_i,z_j,z_k`$ are projected view vectors, $`\operatorname{sim}`$ is cosine similarity, and $`\tau`$ is temperature. The denominator includes the correct match but excludes query $`i`$ itself. The ratio is the matching probability the loss tries to increase.

**Optimization algorithm(s):** The large-batch recipe uses LARS, which adapts update sizes by layer. It uses ten-epoch warmup, cosine-shaped decay, and weight decay $`10^{-6}`$. **Optional math:** Base rate is $`0.3B/256`$, where $`B`$ is batch size. Batch 4,096 gives rate 4.8; this is not an Adam rate. The evaluation classifier has its own training procedure.

**Regularization techniques:** Resized crops, flips, color changes, grayscale, and Gaussian blur teach which differences to ignore. Batch normalization and weight decay also matter. Removing the projector afterward changes which features a later task receives; it is not another image modification.

**Backpropagation considerations:** Both matching views send feedback through the same encoder weights. Other views contribute mismatch penalties. Gathering vectors across devices must preserve that feedback. Shared normalization statistics help avoid shortcuts based on which device handled an image.

**Parameter count / scaling behavior:** The paper rounds the ResNet-50 feature extractor to 24 million parameters. The four-times-width extractor is 375 million. Both exclude the temporary projector. Image-filter weights grow approximately with the square of a width multiplier.

**Training paradigm:** Image-only contrastive pretraining is self-supervised. Labeled linear probing or fine-tuning comes later. Labels used to report accuracy do not enter the pretraining loss.

**Hardware/parallelism considerations:** Several TPUs or GPUs can support large batches. Exchanging vectors, storing comparisons, preparing modified images, and sharing normalization statistics may limit speed separately from encoder arithmetic.

### 3.10.2 MoCo

**In plain English:** MoCo learns image matching while keeping a queue of earlier image descriptions to compare against. This gives it many comparison examples without requiring one enormous current batch.

**Name:** Momentum Contrast (MoCo). The reference here is the original ResNet-based MoCo v1.

**Category & sub-category:** Self-supervised matching of image features. It uses a slowly changing encoder and a queue of earlier comparison vectors.

**Originating paper/vendor/year:** He, Fan, Wu, Xie, and Girshick presented [*Momentum Contrast for Unsupervised Visual Representation Learning*, arXiv 2019, CVPR 2020](https://arxiv.org/abs/1911.05722). MoCo v2/v3 change important components. They should not inherit v1's benchmark result.

**Core mechanism:** An actively trained encoder produces a query vector. A second encoder produces comparison vectors called keys. It changes slowly by averaging the active encoder's weights over time, an EMA update. A queue stores older keys as mismatches, removing the oldest as new keys arrive. Slow changes keep older and newer keys reasonably comparable without storing a huge batch of images.

**Inputs/outputs and typical data types:** Modified views enter the query and key encoders. Matching vectors train the system. The query encoder's main network then supplies features for classification, object detection, or pixel labeling.

**Strengths and limitations:** The queue can be large even when the current batch is modest. Learned features can transfer well to other tasks. But old keys become outdated, and some supposed mismatches may depict similar things. Slow weight averaging stabilizes features; it is not an uncertainty-estimating ensemble by itself.

**Computational complexity / scalability notes:** Comparisons grow with batch size, queue length, and vector width. Only the active encoder needs ordinary gradient and optimizer records. The key encoder still uses weight storage and forward computation. The queue stores vectors, not all earlier images' intermediate training results.

**Optional math:** Comparisons cost $`O(BKd_z)`$, with batch size $`B`$, queue length $`K`$, and vector width $`d_z`$. Key storage is $`O(Kd_z)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original study](https://ar5iv.labs.arxiv.org/html/1911.05722) pretrains on ImageNet-1M without labels. Two views produce a query and its matching current key. Queued keys supply mismatches. After training, a labeled linear classifier uses the frozen encoder to predict ImageNet validation labels.

The ResNet-50 model reaches **60.6% top-1** after 200 pretraining epochs. Top-1 is the percentage of correct first-choice classes. This is not a controlled comparison with SimCLR's 1,000-epoch results where only the loss changed. The paper also tests transfer to detection. The stated design goal is many consistent comparisons without huge current batches. These benchmarks report no public business benefit.

**Notable vendor implementations/libraries:** Meta/Facebook Research's [official MoCo repository](https://github.com/facebookresearch/moco) distinguishes versions and saved models. Library defaults may implement v2 rather than v1.

**Architecture diagram description:**

```text
view A -> online query encoder -> q -----------------+
                                                    +-> InfoNCE logits/loss
view B -> momentum key encoder -> positive k --------+
previous key embeddings -> FIFO queue -> negatives --+
online weights -- EMA update, no backprop --> key encoder
```

**Activation functions used and why:** The original ResNet uses ReLU. Projected vectors are scaled to unit length before dot-product comparison. Temperature-scaled softmax rewards the matching key among alternatives. Nonlinear projectors added in later recipes are not automatically part of v1.

**Loss function(s):** The loss rewards matching the query with its companion view rather than queued alternatives. Old keys stay fixed during this update. Training does not trace feedback through their entire history. This comparison loss is called InfoNCE.

**Optional math:** Use $`-\log[\exp(q\cdot k^+/\tau)/(\exp(q\cdot k^+/\tau)+\sum_{k^-}\exp(q\cdot k^-/\tau))]`$. Here $`q`$ is the query, $`k^+`$ its matching key, and $`k^-`$ queued alternatives. Dot products compare directions. Temperature $`\tau`$ controls how sharply the ratio favors the closest key.

**Optimization algorithm(s):** Original ImageNet training uses SGD with momentum 0.9, batch 256, and initial rate 0.03. It divides the rate by ten at epochs 120 and 160 of 200. Momentum smooths update directions. The key encoder uses weight averaging, not a second gradient optimizer.

**Regularization techniques:** Image changes and weight decay help limit overfitting. Shuffling examples before batch normalization reduces shortcuts from shared batch statistics. Keeping keys comparable and learning to ignore selected image changes solve different problems.

**Backpropagation considerations:** Stop error feedback into keys and queued vectors. Keeping their old computation graphs wastes memory and changes the method. The key encoder's averaging coefficient trades quick adaptation against consistency with older keys.

**Parameter count / scaling behavior:** The paper reports roughly 24 million parameters for the ResNet-50 feature extractor. Training needs two weight copies, projection weights, and the queue. The encoder used afterward is not doubled.

**Training paradigm:** Pretraining uses images without labels. Linear classification and object-detection fine-tuning add labels later. An Instagram-trained encoder and an ImageNet-trained encoder have different data histories even with identical network designs.

**Hardware/parallelism considerations:** The original ImageNet recipe uses eight GPUs. Workers must agree on queue updates, matching-key order, and normalization shuffling. Increasing queue size alone does not ensure faster distributed training.

### 3.10.3 BYOL

**In plain English:** BYOL learns by predicting how a slowly changing copy of itself describes another view of the same image. It does not need a list of deliberately mismatched images.

**Name:** Bootstrap Your Own Latent, usually shortened to BYOL.

**Category & sub-category:** Self-supervised image-feature prediction without explicit negative, or mismatched, examples.

**Originating paper/vendor/year:** Grill and colleagues at DeepMind presented [*Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*, NeurIPS 2020](https://arxiv.org/abs/2006.07733).

**Core mechanism:** Make two views of one image. The active network encodes one view, projects its features, and predicts the other view's projected features. A target network supplies those features. It follows a moving average of active-network weights rather than ordinary gradient updates. Only the active side has a predictor. Swap the views and repeat the comparison. No explicit mismatches are required.

**Inputs/outputs and typical data types:** Inputs are two modified views of an unlabeled image. Projection and prediction vectors train the system. The main encoder's features are retained for later tasks. Target vectors are learned teaching signals, not human-supplied image-category labels.

**Strengths and limitations:** BYOL avoids a negative-example queue and can tolerate some image-modification changes better than contrastive baselines. But a naive predictor could output the same vector for every image. Avoiding this collapse depends on unequal branch designs, target updates, normalization, and optimization together. "Bootstrap" here does not mean statistical resampling or provide a confidence interval.

**Computational complexity / scalability notes:** Both views need active-network forward and backward work, plus target forward passes and small projector/predictor networks. There is no all-pairs negative-comparison table. Target weights still use memory, although they do not need optimizer moment records.

**Optional math:** BYOL avoids a $`B^2`$ negative-similarity matrix for batch size $`B`$. Its weight-averaging update costs $`O(p)`$, where $`p`$ counts averaged weights.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [ImageNet experiments](https://ar5iv.labs.arxiv.org/html/2006.07733) pretrain from images alone. Modified views train active predictions against moving targets. A labeled linear classifier then uses the frozen encoder.

The 1,000-epoch setting reaches **74.3% top-1 for ResNet-50**. Top-1 measures the percentage of correct first-choice classes. The headline 79.6% uses a larger ResNet, not ResNet-50. Unlike SimCLR/MoCo, BYOL avoids explicitly treating other images as dissimilar. Transfer tests support useful features, but these scores establish no production deployment or public business measure.

**Notable vendor implementations/libraries:** DeepMind's [official BYOL implementation](https://github.com/google-deepmind/deepmind-research/tree/master/byol) supplies models and recipes. Not every negative-free method with a moving-average teacher is BYOL.

**Architecture diagram description:**

```text
view A -> online encoder -> projector -> predictor -> normalized prediction
view B -> target encoder -> projector -------------> normalized target [stop-grad]
prediction/target -> squared-distance loss; repeat with views swapped
online encoder/projector weights -- EMA --> target weights
```

**Activation functions used and why:** The ResNet uses ReLU. Dense projector and predictor networks use nonlinear hidden units and batch normalization. Final vectors are scaled to unit length for comparison. The loss does not need softmax probabilities for named image classes.

**Loss function(s):** Minimize squared distance between equal-length prediction and target vectors in both view directions. The target does not receive gradient updates. There is no denominator comparing against negative examples.

**Optional math:** For unit-length vectors, squared distance equals $`2-2s`$, where $`s`$ is their cosine similarity. Reducing this distance increases their directional agreement.

**Optimization algorithm(s):** The large-batch recipe uses LARS with ten-epoch warmup, cosine decay over 1,000 epochs, and weight decay $`1.5\times10^{-6}`$. Target averaging momentum rises from 0.996 toward one along a cosine schedule, making updates slower. **Optional math:** Base rate is $`0.2B/256`$, with batch size $`B`$.

**Regularization techniques:** Cropping, flips, color changes, blur, and solarization, which reverses some bright pixel values, work with normalization and small weight decay. The slow target and one-sided predictor are parts of the method. Neither alone is a universal guarantee against collapse.

**Backpropagation considerations:** Stop gradients on the target side, while retaining them through the active encoder, projector, and predictor. Check whether features still vary across images and help later tasks. Falling matching loss alone could mean that all outputs collapsed to the same vector.

**Parameter count / scaling behavior:** The main encoder determines size after training. Projectors, predictor, and target copies add training storage. The target usually matches the active encoder/projector design; a larger teacher is not required.

**Training paradigm:** Image-only feature prediction is self-supervised. Later evaluation uses labels. The moving-average teacher comes from the learner itself, not a separate teacher pretrained on human labels.

**Hardware/parallelism considerations:** The paper's large setup uses batch 4,096 across 512 TPU-v3 cores. It also describes smaller batches. The largest reported setup is not a minimum requirement. Reproduction needs consistent normalization and weight averaging across workers.

### 3.10.4 DINO

**In plain English:** DINO teaches a network to describe different crops of an image consistently. A slowly updated teacher supplies the targets, without human image labels during pretraining.

**Name:** DINO, the original "self-distillation with no labels" method. Later models sharing the DINO name are not assumed here.

**Category & sub-category:** Self-supervised image-feature learning. A student matches the output distribution of a teacher updated by weight averaging.

**Originating paper/vendor/year:** Caron and colleagues presented [*Emerging Properties in Self-Supervised Vision Transformers*, ICCV 2021](https://arxiv.org/abs/2104.14294).

**Core mechanism:** Make several crops of an image. The student learns to match the teacher's distribution over learned code positions when viewing a different crop. The teacher slowly follows a moving average of student weights. Centering removes running average output scores; sharpening makes preferred choices stand out. Together they help avoid always choosing one position or giving everything equal weight. DINO supports image-filter networks and Vision Transformers, or ViTs. The cited visual findings especially concern ViTs.

**Inputs/outputs and typical data types:** Unlabeled images produce large global crops and smaller local crops. Training distributions describe learned codes, not named human classes. Whole-image vectors and patch features support classification, search, and exploratory studies of object discovery.

**Strengths and limitations:** DINO can learn useful image and patch patterns without negative pairs or pretraining labels. Attention plots may highlight foreground structure. They are not guaranteed object boundaries or explanations. Neither attention weights nor the spread of teacher outputs gives calibrated uncertainty about meaning.

**Computational complexity / scalability notes:** The student processes several crops; the teacher processes global crops. Smaller patches create longer sequences and much more all-pairs attention work. Halving patch width roughly quadruples patch count and makes the dense-attention term roughly sixteen times larger at fixed resolution. Total runtime does not necessarily rise sixteenfold.

**Optional math:** ViT work per crop is $`O(L(Td^2+T^2d))`$, with $`L`$ layers, $`T`$ patches, and vector width $`d`$. Only part of the cost grows with $`T^2`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [DINO study](https://ar5iv.labs.arxiv.org/html/2104.14294) trains on ImageNet images without labels. Multiple crops drive teacher-student learning. A frozen ViT-S/16 then supplies features for two labeled tests: a linear classifier and a nearest-neighbor reference set.

ViT-S/16 reaches **77.0% linear top-1** and **74.5% k-nearest-neighbor top-1**. Both report the percentage of correct first-choice classes, but use different decision procedures. Nearest-neighbor testing still uses reference labels despite not training a classifier. Search and patch-attention tests also study the features. The aim is agreement between views, rather than pixel rebuilding. These findings establish no production pixel-labeling system or public business measure.

**Notable vendor implementations/libraries:** Meta/Facebook Research provides the [official DINO code](https://github.com/facebookresearch/dino). DINOv2, object-detection models called DINO, and other saved models have separate histories. They must not inherit these results.

**Architecture diagram description:**

```text
global crop -> EMA teacher backbone -> projection head -> centered/sharpened target
global/local crops -> student backbone -> projection head -> student distributions
different-view distributions -> cross-entropy
student weights -- EMA --> teacher; teacher targets use stop-gradient
```

**Activation functions used and why:** ViT uses smooth GELU units, softmax attention weights, and layer normalization to rescale values. Another temperature-controlled softmax converts projection scores into distributions. Centering subtracts running teacher-score averages before sharpening. The outputs are probabilities over learned code positions, not named classes.

**Loss function(s):** Cross-entropy rewards the student's agreement with the teacher across different views. Teacher targets stay fixed during the student's update. Centering and the two temperature settings affect what target is learned and how strongly errors drive updates.

**Optional math:** The loss is $`-\sum_k P_{\rm teacher}^{(k)}\log P_{\rm student}^{(k)}`$. Here $`k`$ indexes learned output positions. Each $`P`$ gives the teacher's or student's probability at that position for its image view. The sum penalizes student probabilities that disagree with the teacher.

**Optimization algorithm(s):** The paper uses AdamW with ten-epoch warmup and cosine learning-rate decay. Weight decay rises from 0.04 to 0.4. Teacher averaging momentum rises from 0.996 toward one. Teacher-temperature warmup is a separate schedule. **Optional math:** Base rate is $`0.0005B/256`$, where $`B`$ is batch size.

**Regularization techniques:** Several crops, color changes, blur, solarization, and weight decay work alongside centering and sharpening. Solarization reverses some bright pixel values. Weight averaging does not replace these other choices. Avoiding constant or useless outputs depends on the full recipe.

**Backpropagation considerations:** Only the student receives ordinary gradient updates; teacher outputs are detached from that feedback. Devices must share center statistics consistently. Excessive sharpening or a wrongly updated center can create useless targets even while the loss falls.

**Parameter count / scaling behavior:** The paper rounds the ViT-S feature extractor to 21 million parameters, excluding the training projector. Changing patch size can greatly change computation with nearly unchanged main-network weights. The teacher requires another stored copy.

**Training paradigm:** Pretraining uses images alone, with no human labels. Labeled linear or k-nearest-neighbor tests and later fine-tuning are separate supervised procedures.

**Hardware/parallelism considerations:** The reported ViT-S/16 recipe uses batch 1,024 across 16 GPUs. Differently sized crops complicate batching. Center and teacher updates across workers must preserve the intended global averages.

### 3.10.5 Masked autoencoders

**In plain English:** MAE hides many image patches and learns to rebuild them from the visible pieces. Its expensive encoder sees only visible patches, saving pretraining work.

**Name:** Masked autoencoder (MAE). This entry uses the Vision Transformer version with a large encoder and smaller decoder.

**Category & sub-category:** Self-supervised image-feature learning by rebuilding a high fraction of hidden patches.

**Originating paper/vendor/year:** He, Chen, Xie, Li, Dollar, and Girshick presented [*Masked Autoencoders Are Scalable Vision Learners*, arXiv 2021, CVPR 2022](https://arxiv.org/abs/2111.06377).

**Core mechanism:** Divide an image into patches and randomly remove many before encoding. The encoder processes only visible patches. A smaller decoder gets their encoded features plus placeholders, called mask tokens, showing where pieces are missing. It predicts pixel values to rebuild the image. Putting placeholders only in the decoder avoids expensive encoder work on invisible patches. Later recognition normally keeps the encoder and discards the decoder.

**Inputs/outputs and typical data types:** Images become randomly masked patch sequences. Training predicts rebuilt patches. Later use produces encoder features or task-specific predictions. Standard recognition sees the full image, rather than continuing to hide 75% of it.

**Strengths and limitations:** Simple pixel targets can train large vision encoders without mismatched pairs or captions. But rebuilding may focus on small visual details rather than meaning. Good reconstruction alone does not validate useful recognition features. Squared-error outputs are single estimates, not calibrated probabilities for every plausible completion.

**Computational complexity / scalability notes:** Removing patches reduces both encoder dense-layer work and the more expensive all-pairs attention work. The smaller decoder still processes the full patch list. Thus 75% masking does not guarantee sixteenfold faster training. Later full-image tasks restore the encoder's full patch cost, especially at larger resolutions.

**Optional math:** Let $`T`$ count all patches and $`v`$ be the visible fraction. Encoder projections and dense layers scale with $`vT`$, while dense attention scales with $`(vT)^2`$. These savings do not cover every part of training.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [MAE study](https://ar5iv.labs.arxiv.org/html/2111.06377) pretrains on ImageNet-1K, typically hiding 75% of patches. Visible-only encoding and decoder rebuilding teach image features. Researchers discard the decoder, then use labels to fine-tune the encoder and classifier together.

The paper reports **87.8% top-1 for ViT-H at 448-pixel evaluation resolution**. Its ViT-H result at 224 resolution is 86.9%. Top-1 is the percentage of correct first-choice image classes. These are **fine-tuned results**, not frozen-feature tests or reconstruction accuracy. The advantage over a full-patch denoising autoencoder is spending most work on visible content. Transfer tasks are also studied, but no deployed recognition service or public business measure is established.

**Notable vendor implementations/libraries:** Meta/Facebook Research supplies the [official MAE repository](https://github.com/facebookresearch/mae); compatible ViT libraries also exist. Methods that predict discrete image codes instead of pixels use different targets. They are not automatically MAE.

**Architecture diagram description:**

```text
image -> patches -> random keep/remove -> visible patches only -> large ViT encoder
encoded visible patches + mask tokens + positions -> small decoder
                                                   -> reconstructed pixel patches
after pretraining: full image -> encoder -> downstream head; decoder discarded
```

**Activation functions used and why:** Encoder and decoder blocks use ViT's smooth GELU units, softmax attention weights, and layer normalization. A linear final layer predicts real-valued pixels. It does not need softmax probabilities over categories.

**Loss function(s):** Mean squared error averages squared pixel mistakes on **masked patches only**; lower is better. The paper also tests targets rescaled separately within each patch. This differs from scoring every patch equally or predicting entries in a discrete codebook.

**Optimization algorithm(s):** The pretraining table uses AdamW, batch 4,096, 40-epoch warmup, cosine decay, weight decay 0.05, and betas (0.9, 0.95). Betas control smoothing of update directions and squared updates. **Optional math:** Base rate $`1.5\times10^{-4}`$ is multiplied by batch size divided by 256. Labeled fine-tuning uses different schedules and restrictions.

**Regularization techniques:** Hiding many random patches is central; other pretraining image changes are fairly simple. Some fine-tuning recipes use Mixup to blend examples or CutMix to swap image regions. Label smoothing softens targets; stochastic depth randomly skips blocks, with stronger settings in some fine-tuning recipes. These choices are not automatically part of reconstruction pretraining.

**Backpropagation considerations:** Feedback passes through the decoder to encoded visible patches. Removed patches have no encoder intermediate results. Correct training depends on target rescaling, restoring patch order, and applying loss only at the right masked positions.

**Parameter count / scaling behavior:** Choosing ViT-B, ViT-L, or ViT-H sets encoder size. The small decoder adds training-only weights. Larger image resolution increases patch and position information far more than the main shared Transformer weight matrices.

**Training paradigm:** Image-only reconstruction is self-supervised. Labeled probing or fine-tuning follows. The headline recognition score therefore measures a two-stage pipeline, not label-free classification.

**Hardware/parallelism considerations:** Splitting batches across accelerators and recomputing intermediate results can support large encoders. Visible-only encoding saves pretraining memory. Full-image fine-tuning at high resolution may still be the most memory-demanding stage.

### 3.10.6 CLIP

**In plain English:** CLIP learns which images and written descriptions match. It can then search images with text or choose among class descriptions, without training a new fixed classifier for each list.

**Name:** Contrastive Language-Image Pretraining (CLIP). This entry covers OpenAI's original 2021 family with separate image and text encoders.

**Category & sub-category:** Image-and-language representation learning with **natural language supervision**, as the original paper states. CLIP is here because its features transfer to new tasks, not because it lacks supervision. Noisy web pairs may be called weak supervision, but captions still teach meaning. See [supervised contrastive learning and Siamese networks](02-supervised-neural.md) for related methods.

**Originating paper/vendor/year:** Radford and colleagues at OpenAI presented [*Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021](https://arxiv.org/abs/2103.00020).

**Core mechanism:** Encode every image and caption in a batch as number vectors. Reward correct pairs for matching more closely than incorrect pairs. After training, encode search text or candidate class descriptions and compare them with image vectors. Changing the descriptions changes the available class choices without training another fixed output classifier.

**Inputs/outputs and typical data types:** Matching images and natural-language text train the model. Outputs include comparable vectors, match scores, search rankings, and zero-shot class choices. Original image encoders are modified ResNets or ViTs; text uses a Transformer. **There is no weight tying between the image and text encoders:** each learns its own weights. They are not identical-weight Siamese twins.

**Strengths and limitations:** Text can define flexible search queries and classes. Caption errors, cultural bias, text shortcuts, duplicate pairs, and prompt wording all affect predictions. A match score or softmax over selected candidates is not calibrated confidence across every possible real-world answer. Changing the candidate list changes the apparent probabilities.

**Computational complexity / scalability notes:** Encoder work depends on image network and text length. Comparing every image with every caption grows with the square of batch size. Workers can divide that work. For later search, vectors can be computed and indexed beforehand. Searching millions of candidates still has its own system costs.

**Optional math:** Pair comparisons cost $`O(B^2d_z)`$, where $`B`$ is batch size and $`d_z`$ the shared vector width. This does not include encoder computation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [paper](https://ar5iv.labs.arxiv.org/html/2103.00020) trains CLIP on 400 million internet image-text pairs, then tests transfer to many datasets. ImageNet class names are placed in prompt templates. Their text vectors are compared with each validation image vector to choose a class.

The best **ViT-L/14@336px** reports **76.2% ImageNet top-1**, the percentage of correct first-choice classes. This uses prompted zero-shot testing, not a supervised ImageNet linear classifier. Zero-shot means no task-specific labeled training for that classifier. It does not mean no image-text supervision or proven absence of web-data overlap. The appendix also tests Flickr30k/MS-COCO search with prompted descriptions. Search direction and candidate sets differ from classification. Language-defined choices are the advantage over a fixed supervised classifier. No enterprise-search savings or public business measure is asserted.

**Notable vendor implementations/libraries:** OpenAI releases [official CLIP code and weights](https://github.com/openai/CLIP). Hugging Face Transformers offers compatible model classes. OpenCLIP and other reproductions may change data, design, licenses, or recipes, so each saved model needs its own history.

**Architecture diagram description:**

```text
image -> ResNet or ViT -> projection -> normalized image embedding
caption -> text Transformer -> projection -> normalized text embedding
all image/text dot products in batch -> learned logit scale -> symmetric CE
zero-shot: image embedding vs embeddings of candidate-class descriptions
```

**Activation functions used and why:** Modified ResNets use rectifying image-filter activations. ViT and text Transformers use smooth GELU-family units and softmax attention. Final vectors are scaled to unit length. A learned inverse temperature controls how sharply match scores differ, not the probability that an interpretation is true.

**Loss function(s):** For each image, reward selecting its paired caption. For each caption, reward selecting its paired image. Average cross-entropy across both directions. This learns pair matching, not caption generation or named-class prediction. Other captions may also describe an image correctly, creating false negatives.

**Optimization algorithm(s):** The paper uses Adam with weight decay applied separately from gradient adaptation. It uses cosine learning-rate decay and network-specific settings over 32 epochs. The strongest ViT-L/14 receives an additional epoch at 336 resolution. Temperature starts at an equivalent 0.07. The paper reports bounding score scaling to prevent unstable, overly sharp choices.

**Regularization techniques:** Weight decay excludes selected scale and bias values. Image preprocessing and broad paired data support transfer. Combining predictions from several prompt templates is an evaluation technique. It is neither training regularization nor a universally best user prompt.

**Backpropagation considerations:** Both encoders receive feedback from matching errors. Distributed comparisons must preserve that feedback to both images and text. Very large raw-score scales make softmax concentrate sharply and can cause numeric problems. Mixed-precision statistics and recomputed intermediates require care.

**Parameter count / scaling behavior:** Count the chosen image encoder, text encoder, and projection layers. The CLIP name alone does not specify size. ViT-L/14@336px is resolution-specific, not equivalent to every ViT-L/14 result. Batch comparisons can grow quadratically without adding any model weights.

**Training paradigm:** The original paper calls this **natural language supervision**. Existing pairs automatically supply matching targets, so some broad definitions also call the practice task self-supervised. Yet human-written captions and their pairing teach image meaning. "Weakly supervised" highlights noisy web pairings and missing curated task-specific class labels; it does not mean zero supervision. Zero-shot prompting, labeled linear probing, and fine-tuning are separate later procedures.

**Hardware/parallelism considerations:** The paper uses batch 32,768, mixed precision, recomputation checkpoints, and comparison work divided across devices. The largest ViT trains on 256 V100 GPUs; the largest ResNet uses 592. These are historical training setups, not requirements for using a trained model.

### 3.10.7 Word2Vec

**In plain English:** Word2Vec learns a number list for each word by predicting words found nearby. Related words can get useful shared patterns, but each word keeps the same list in every sentence.

**Name:** Word2Vec includes continuous bag-of-words (CBOW) and continuous skip-gram. Different ways to train their output predictions are separate choices.

**Category & sub-category:** Self-supervised word-feature learning. These shallow models predict from learned word vectors without a deep hidden network.

**Originating paper/vendor/year:** Mikolov, Chen, Corrado, and Dean at Google wrote [*Efficient Estimation of Word Representations in Vector Space*, 2013](https://ar5iv.labs.arxiv.org/html/1301.3781). Mikolov and colleagues' [*Distributed Representations of Words and Phrases and their Compositionality*, NeurIPS 2013](https://ar5iv.labs.arxiv.org/html/1310.4546) adds negative sampling, frequent-word subsampling, and phrase modeling. These papers did not invent all number-based word representations.

**Core mechanism:** Read nearby words in running text and learn two lookup tables: input vectors and output vectors.

- **CBOW:** Add or average surrounding-word vectors, ignoring their order, to predict the center word. Several neighbors support one prediction.
- **Skip-gram:** Use a center word to predict its neighbors, creating several training pairs.

Neither needs the costly nonlinear hidden layer used in earlier neural language models. Output training can check the whole vocabulary, use a tree of choices called hierarchical softmax, or distinguish observed pairs from sampled noise words. That last method is negative sampling. These choices are not synonyms for CBOW and skip-gram.

**Inputs/outputs and typical data types:** Text split into tokens supplies neighboring-word prediction examples. Outputs are static word vectors for similarity search or later language tasks. A word has the same vector in every context, even with several meanings. Unknown words need extra handling. Word-piece models and sentence-aware Transformers are different methods.

**Strengths and limitations:** Learned vectors share information between words instead of treating each as an unrelated slot. Context-window size, word frequency, text bias, and token splitting affect the result. Vector differences do not represent universal relationships. Cosine similarity and negative-sampling scores are not calibrated confidence about meaning or word sense.

**Computational complexity / scalability notes:** Full-vocabulary prediction checks every word. Negative sampling checks the observed pair and a limited number of sampled alternatives, making each pair cheaper. A tree follows only the decisions leading to a word. Skip-gram repeats work for each neighbor; CBOW also combines context vectors.

**Optional math:** Let $`V`$ be vocabulary size, $`d`$ vector length, and $`K`$ sampled negatives. Skip-gram negative sampling costs $`O((K+1)d)`$ per observed pair. Full softmax costs $`O(Vd)`$. Hierarchical softmax costs $`O(hd)`$, where $`h`$ is tree-path length, typically growing logarithmically on average with vocabulary size.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [2013 comparison](https://ar5iv.labs.arxiv.org/html/1301.3781) trains 640-dimensional vectors on 320 million words from LDC corpora, with an 82,000-word vocabulary. Text windows train CBOW or skip-gram. Analogy tests apply the vector change between two words to a third, then search for the nearest answer by direction.

Under the Semantic-Syntactic Word Relationship protocol, skip-gram gets **55% semantic accuracy versus CBOW's 24%**. CBOW gets **64% syntactic accuracy versus skip-gram's 59%**. Semantic questions test meaning relationships; syntactic questions test grammar relationships. These percentages score analogy answers on matched training text, not a later classifier or a universal ranking. Avoiding an expensive hidden network makes representation learning cheaper than a full neural language model. No public business performance measure is reported.

**Optional math:** The analogy query is $`v_b-v_a+v_c`$, where $`v_a,v_b,v_c`$ are vectors for three given words. It applies the change from word $`a`$ to word $`b`$ to word $`c`$.

**Notable vendor implementations/libraries:** The [author-hosted C implementation](https://github.com/tmikolov/word2vec) and [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) provide implementations. Defaults, negative counts, and released Google News vectors need not reproduce the 640-dimensional comparison.

**Architecture diagram description:**

```text
CBOW: context token IDs -> shared embedding lookups -> sum/mean -> predict center
skip-gram: center ID -> embedding lookup -> predict each nearby context token
output choice: full softmax OR binary tree decisions OR positive/negative logits
after training: retain word vectors -> similarity search / downstream model
```

**Activation functions used and why:** Projection is linear; it needs no hidden ReLU or tanh layer. Full softmax turns vocabulary scores into probabilities that sum to one. Tree prediction uses sigmoid decisions. Negative sampling uses sigmoid observed-versus-noise scores, not a full normalized word-probability distribution.

**Loss function(s):** CBOW penalizes wrong center-word predictions. Skip-gram adds prediction losses for neighboring words. Negative sampling instead rewards observed word pairs over noise pairs. It uses a different binary training goal, not an unbiased shortcut for full-softmax cross-entropy.

**Optional math:** Skip-gram negative sampling minimizes $`-\log\sigma(u_c^\top v_w)-\sum_{j=1}^{K}\log\sigma(-u_{n_j}^\top v_w)`$. Here $`v_w`$ is the center word's input vector, $`u_c`$ the observed neighbor's output vector, and $`u_{n_j}`$ a sampled noise word's output vector. There are $`K`$ noise words. Each dot product gives a pair score; sigmoid $`\sigma`$ turns it into a binary score. The paper samples noise words from individual-word frequencies raised to $`3/4`$, smoothing their frequency differences.

**Optimization algorithm(s):** The first paper's serial experiments use SGD/backpropagation, starting at 0.025 and reducing the rate evenly toward zero. Its parallel DistBelief setup uses asynchronous minibatch Adagrad updates. These are distinct recipes, not one optimizer required by all Word2Vec implementations.

**Regularization techniques:** Subsampling removes some frequent-word examples. Limited or random context windows, minimum vocabulary frequencies, and finite vector length also restrict learning. Negative sampling changes the loss itself; it is not simply dropout.

**Backpropagation considerations:** Update the relevant lookup-table rows rather than large one-hot matrices with a separate slot for every word. CBOW shares feedback across context rows; skip-gram adds pairwise updates. Nearly saturated sigmoids weaken feedback. Simultaneous updates to common words can collide. No gradients pass through the random choice of token identity.

**Parameter count / scaling behavior:** Standard negative sampling keeps separate input and output vector tables. Exporting only input vectors halves their storage, but vocabulary information and sampling structures still cost space.

**Optional math:** For vocabulary size $`V`$ and vector length $`d`$, the two $`V\times d`$ tables store approximately $`2Vd`$ parameters.

**Training paradigm:** Nearby words supply self-supervised targets, without manually labeled meaning relationships. Analogy answers evaluate the resulting vectors. A later supervised task adds its own label signal.

**Hardware/parallelism considerations:** Sparse row updates suit CPUs and often wait on memory transfers rather than arithmetic. Asynchronous threads improve throughput but make results less repeatable. Distributed tables need communication and may compete over frequent words. Large dense GPU calculations are not required.

### 3.10.8 GloVe

**In plain English:** GloVe learns word vectors from counts of which words appear near each other. Each word keeps one fixed vector, which can help a later language model share information across words.

**Name:** Global Vectors for Word Representation (GloVe). The original 2014 model fits log counts using pairs of learned vectors.

**Category & sub-category:** Unsupervised/self-supervised word-feature learning from global neighboring-word counts. The neural fields describe trainable lookups and two-vector scores, not a deep hidden network.

**Originating paper/vendor/year:** Pennington, Socher, and Manning at Stanford presented [*GloVe: Global Vectors for Word Representation*, EMNLP 2014](https://aclanthology.org/D14-1162/). Original code was released in August 2014. Later vector releases on the [project page](https://nlp.stanford.edu/projects/glove/) have different training-text sources and histories.

**Core mechanism:** Count how often each word appears near each context word across the text collection. Learn target and context vectors whose dot product, plus two offsets, fits the logarithm of that count. Weight rare pairs less, while capping the importance of very frequent pairs. The paper motivates this through ratios of nearby-word probabilities, which can distinguish properties associated with ice versus steam.

**Inputs/outputs and typical data types:** Tokenized text becomes weighted counts for word pairs that occur together. Outputs are static target/context vectors and offsets, called biases. The original paper adds the two vector tables for evaluation. This is neither a sentence-aware encoder nor a normalized next-word probability distribution.

**Strengths and limitations:** Reusing combined counts makes repeated training efficient and the target clear. Preparing and storing counts can be expensive. Rare words, multiple meanings, and text biases remain problems. Vector similarity is not calibrated confidence. Log-count prediction error is not an uncertainty interval about meaning.

**Computational complexity / scalability notes:** Counting work grows with text length and the number of nearby positions checked. Training then visits stored pairs, with more work for longer vectors or more passes. Store only observed pairs, not every possible word pair. Even sparse counts can use most memory.

**Optional math:** For $`n`$ text tokens and window radius $`c`$, counting costs about $`O(nc)`$. With $`M`$ nonzero pairs and vector length $`d`$, one training pass costs $`O(Md)`$; $`E`$ passes cost $`O(EMd)`$. Count storage is $`O(M)`$, not necessarily $`O(V^2)`$, where $`V`$ is vocabulary size.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:** **Evidence status: Research benchmark.** The [original paper](https://nlp.stanford.edu/pubs/glove.pdf) tests named-entity recognition, or NER, on CoNLL-2003 Reuters newswire. Counts from unlabeled text train GloVe vectors. A conditional random field, or CRF, then learns labeled word sequences using 50-dimensional features in a five-word window. It assigns person, location, organization, and miscellaneous entity tags.

The NER table reports **88.3 test F1 with GloVe versus 85.4 for discrete features alone**. F1 balances how many predicted entities are correct with how many true entities are found; higher is better. It is not ordinary percent-correct accuracy. Another compared method, HPCA, scores 88.7, so GloVe is not best on that test. Vectors help share word information beyond separate discrete features. This evaluates unsupervised vectors inside a supervised system. It is not label-free NER, a Reuters deployment, or a public business benefit.

**Notable vendor implementations/libraries:** Stanford supplies [official GloVe C code and vectors](https://github.com/stanfordnlp/GloVe). Wikipedia 2014 plus Gigaword 5 vectors differ from later releases. Loading them into PyTorch or Keras does not reproduce the training that created them.

**Architecture diagram description:**

```text
corpus -> window counting -> sparse nonzero X_ij
target ID i -> embedding w_i ------+
context ID j -> embedding u_j -----+-> dot product + b_i + b_j
log X_ij + count weight f(X_ij) ----+-> weighted squared-error loss
after training: combine target/context vectors -> NLP features
```

**Activation functions used and why:** No hidden nonlinear activation is needed. Two lookup vectors feed a dot product plus biases. Logarithms rescale target counts, while a fractional power sets example weights. These transform the data and loss; they are not sigmoid class probabilities.

**Loss function(s):** Compare each predicted log count with its observed log count. Square the error and weight it by how common the pair is. Skip zero-count pairs because their logarithm is undefined. This is weighted least squares, not negative-sampling logistic loss.

**Optional math:** The loss is $`J=\sum_{X_{ij}>0}f(X_{ij})(w_i^\top u_j+b_i+\tilde b_j-\log X_{ij})^2`$. Here $`X_{ij}`$ counts target word $`i`$ with context word $`j`$. Their vectors are $`w_i,u_j`$, and their biases are $`b_i,\tilde b_j`$. The weight is $`f(x)=\min((x/x_{\max})^\alpha,1)`$, with count $`x`$, threshold $`x_{\max}=100`$, and exponent $`\alpha=3/4`$. It raises pair weight up to a cap of one. Excluding zero entries avoids $`\log0`$.

**Optimization algorithm(s):** The original recipe uses Adagrad, starting at 0.05 and sampling nonzero counts. It runs 50 iterations below 300 vector dimensions and 100 otherwise. Adagrad adapts each weight's step using accumulated squared gradients. No cosine decay or Transformer warmup is implied. The current small-text demo has different settings.

**Regularization techniques:** Short vectors limit the patterns the model can fit. Pair weights, vocabulary cutoffs, and stronger weights for closer context words control influence. Weighting training pairs is not the same as an L2 penalty on model weights. No assumed dropout layer is required by the original result.

**Backpropagation considerations:** Each pair updates two vector rows and two biases. Counts are fixed targets; text tokens are not differentiable. Training both vector tables together can have several local solutions rather than one guaranteed best solution. Stable logarithms, starting values, and accurate accumulated-gradient records matter.

**Parameter count / scaling behavior:** Training stores target vectors, context vectors, and both sets of biases. Adding the vector tables for export reduces storage. Optimizer records and sparse pair counts are additional training costs.

**Optional math:** With vocabulary size $`V`$ and vector length $`d`$, training parameters total $`2Vd+2V`$. Summed exported vectors need $`Vd`$ values.

**Training paradigm:** Observed neighboring words supply targets without manually labeled meanings. The CRF example later adds BIO labels: beginning, inside, or outside an entity. Its F1 belongs to that full supervised system, not the vectors alone or wholly unsupervised training.

**Hardware/parallelism considerations:** Reference code uses several CPU threads on shuffled sparse records. Preparing counts can demand much disk access and memory. Distributing vector rows adds communication and competing updates. GPUs are optional; their use would not imply a hidden deep network.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| SimCLR | Images with useful modified views | Learns features by matching paired views | Big batches cost more; related images may be treated as mismatches | ImageNet tests of frozen features |
| MoCo | Large image collections with moderate batches | Reuses many slowly changing comparison vectors | Old or wrongly mismatched queue entries | ImageNet classification and detection-transfer tests |
| BYOL | Images with meaningful alternate views | Does not need explicit mismatches | Full recipe must prevent constant outputs | ResNet-50 ImageNet linear-classifier tests |
| DINO | Whole images and patches, especially with ViTs | Learns consistent whole-image and patch features | Sensitive to teacher and centering settings | ViT-S/16 linear and nearest-neighbor ImageNet tests |
| MAE | Images divided into patches | Saves encoder work by hiding patches | Good pixel repairs do not prove reliable meaning | Fine-tuned ViT-H ImageNet tests |
| CLIP | Naturally matching images and language | Text defines search queries and classes | Needs paired supervision; scores can be biased and uncalibrated | Zero-shot ImageNet and image-text search tests |
| Word2Vec: CBOW / skip-gram | Tokenized text and neighboring words | Learns useful static vectors cheaply | Multiple meanings, unknown words, and text bias | Meaning and grammar analogy tests on matched text |
| GloVe | Stored counts of neighboring word pairs | Reuses weighted counts across training passes | Large count storage; one vector per word | CoNLL-2003 entity recognition using learned vectors |

## Evidence reading notes

- **FID and Inception Score:** FID compares image-feature distributions; lower is better. Inception Score, or IS, rewards confident class predictions and class variety using a pretrained classifier; higher is better. Neither is percent-correct image accuracy. DDPM's 3.17 FID and 9.46 +/- 0.11 IS retain the original CIFAR-10 table and 50,000-sample procedure. Training-reference and test-reference statistics remain distinct. Score-SDE's 2.20 belongs to continuous deep NCSN++ VE, not every score model. DiT's 2.27 belongs to guided, long-trained XL/2 at 256 resolution, not its unguided 400,000-step component tests. These numbers come from sources, not local reruns.
- **Likelihood:** RealNVP's 3.49, Glow's 3.35, and Gated PixelCNN's 3.03 bits/dimension are paper-specific CIFAR-10 results. This likelihood-based measure describes coding cost per image value; lower is better. Continuous models with added pixel noise and discrete next-pixel models require different qualifications. Do not mix Glow's five-bit face demonstrations into an eight-bit comparison.
- **Representation evaluation:** Keep SimCLR's wider headline model, MoCo's shorter training, and BYOL's network size attached to their scores. DINO's linear classifier and k-nearest-neighbor reference test differ. MAE's results include labeled fine-tuning and a specified resolution. CLIP uses prompted zero-shot evaluation. This chapter is not a ranking under one shared training budget.
- **Static word embeddings:** Word2Vec's meaning and grammar percentages use the original 640-dimensional comparison on matched text. They are not mixed with later Google News vectors. GloVe's 88.3 versus 85.4 CoNLL test F1 compares a labeled CRF with 50-dimensional vector features against discrete features alone. F1 balances correct entity predictions and coverage of true entities; higher is better. HPCA scored higher on that test. These are not vector-only accuracy or business gains. CLIP also retains the paper's description of natural language supervision despite this chapter's broad grouping.
- **Production versus research:** Google's dated announcement supports deployment of an updated WaveNet. Original WaveNet listener ratings are a separate research result. Most other examples demonstrate research capabilities. Model cards and demo code do not prove audited customer savings.
- **Disclosure limits:** The original beta-VAE OpenReview PDF was unavailable during the evidence check. Attribution and dataset facts use the authors' dSprites documentation; detailed network and optimizer settings come from the checked 2018 follow-up. DALL-E 3's report and evaluation artifacts were checked, but do not reveal a complete generator recipe. Stable Diffusion version differences use the v1.4/SDXL cards and SDXL paper. Details from inaccessible cards were not guessed.

## Coverage and continuation manifest

**This chapter covers 28 entries. Each keeps all nine common fields and nine neural fields, a network diagram, a worked example, and a row in its category table.**

| Section range | Coverage | Entry count |
|---|---|---|
| 3.6.1-3.6.5 | Rebuilding and codes: autoencoder, denoising autoencoder, VAE, beta-VAE, VQ-VAE | 5 |
| 3.7.1-3.7.4 | Generator/checker feedback: GAN, DCGAN, distinct StyleGAN versions, CycleGAN | 4 |
| 3.8.1-3.8.7 | Diffusion and image systems: DDPM, score-SDE, latent diffusion/Stable Diffusion, DiT, DALL-E 1, DALL-E 2, DALL-E 3 | 7 |
| 3.9.1-3.9.4 | Reversible or sequential generation: RealNVP, Glow, PixelCNN, WaveNet | 4 |
| 3.10.1-3.10.8 | Learned features: SimCLR, MoCo, BYOL, DINO, MAE, CLIP, Word2Vec (CBOW/skip-gram), GloVe | 8 |

**Read next or review earlier material:** Start with the [reading guide and evidence policy](00-reading-guide.md). Related chapters cover [supervised neural and matching methods](02-supervised-neural.md), [semi-supervised learning and machine-generated labels](03-semi-supervised.md), and [classical unsupervised methods](04-unsupervised-classical.md). Continue to [foundation-model pretraining](06-foundation-models.md), [mixture-of-experts models](07-moe-models.md), or the [deeper MoE explanation](08-moe-deep-dive.md). The [selection guide](09-comparative-guide.md) compares choices, and the [glossary](10-glossary.md) explains terms.

**Where this chapter stops:** The 28 entries are not an exhaustive survey. Possible extensions, not covered here, include:

- Hierarchical VQ-VAE-2 and residual quantization, which use several coding stages.
- Wasserstein GAN variants and energy-based models, which use other distribution-comparison or scoring rules.
- Consistency and distilled diffusion, which aim to shorten generation; flow matching and rectified flows, which learn paths from noise to data.
- Newer Stable Diffusion/DiT-derived models, plus diffusion for video, audio, and 3D data.
- Spline flows using flexible curved transformations, and continuous normalizing flows using continuous-time transformations.
- PixelCNN++, parallel WaveNet, DINOv2 and later variants, and SimCLRv2/MoCo v2-v3.
- fastText, word-piece and multilingual static vectors, and masked learning across several data types.
- Scientific applications and rigorous checks of calibration, memorization, licensing, and privacy.

These are optional extensions, not topics silently included in the 28 entries. This chapter also does not provide a separate reinforcement-learning classification or reveal undisclosed proprietary training recipes.
