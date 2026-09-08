# 2. Semi-Supervised Learning Algorithms

This volume covers 22 methods and explicitly scoped method families in six categories. It is a bounded reference, not a claim to enumerate all semi-supervised learning. **Evidence policy date: 2026-09-08.** The cited papers are historical sources, not assertions about the latest available models. Conference years and earlier arXiv uploads are distinguished where relevant. Public reference code establishes an implementation, not commercial deployment.

Semi-supervised learning uses labeled observations and unlabeled observations to learn a task for which some labels are available. An unlabeled image is not a negative example. Nor is an automatically generated label equivalent to a human annotation. The additional data help only through assumptions connecting the distribution of inputs to the desired labels.

Use the [reading guide](00-reading-guide.md) for the book's taxonomy and evidence conventions. Neural architectures are explained further in [supervised neural learning](02-supervised-neural.md); related representation-learning objectives appear in [unsupervised neural learning](05-unsupervised-neural.md). A VAE can have an unsupervised stage inside a semi-supervised system, and a Transformer can have self-supervised pretraining before semi-supervised fine-tuning. Architecture alone does not determine supervision.

### Reading the objectives and costs

Let $`D_L=\{(x_i,y_i)\}_{i=1}^{n_l}`$, $`D_U=\{u_i\}_{i=1}^{n_u}`$, and $`n=n_l+n_u`$. Let $`C`$ be the number of target classes, $`p_\theta(y\mid x)`$ a classifier, and $`H(q)=-\sum_c q_c\log q_c`$. Most neural recipes minimize


$$
\mathcal L=\mathcal L_s(D_L)+\lambda(t)\mathcal L_u(D_U).
$$


This notation does not mean that their unlabeled objectives are interchangeable. They may encourage confidence, agreement under perturbation, agreement with another model, or successful reconstruction. The relative weighting depends on whether each loss is a sum, a batch mean, or a mean over selected examples.

Here $`d`$ denotes feature dimension or a specifically identified hidden width, $`p`$ denotes learned parameters, $`B`$ denotes batch size, and $`E`$ denotes epochs. For neural costs, $`F`$ denotes the forward cost of the stated backbone on one example; a backward pass is another architecture-dependent constant multiple of this cost. For graphs, $`m`$ denotes stored edges and $`s`$ solver iterations. These qualifications matter: neither a convolutional model nor an iterative kernel solver has a universal cost determined by its algorithm name.

### Assumptions, failure modes, and evaluation

- **Cluster assumption:** examples in the same high-density region tend to share a label. A class may occupy several clusters. Unlabeled clusters do not reveal their semantic names; with no labeled representative, identifying a new class is generally impossible without further information.
- **Low-density separation:** a useful decision boundary avoids regions containing much probability mass. This complements the cluster assumption, but is wrong when genuine classes overlap or the target cuts across a dense population.
- **Manifold assumption:** observations concentrate near a lower-dimensional structure, and the target varies smoothly along its relevant directions. Smoothness in raw pixel distance, a graph chosen for convenience, or a pretrained embedding is not automatically task-relevant.
- **View and augmentation assumptions:** co-training needs genuinely informative alternative views; consistency methods need perturbations that preserve the target. A transformation useful for photographs may corrupt a digit label. Back-translation can alter negation or sentiment.
- **Confirmation bias:** an incorrect prediction becomes a training target, making the model increasingly confident in its mistake. High confidence is not calibrated correctness. Thresholds, warm-up, teacher averaging, and disagreement checks mitigate different parts of this feedback loop; none proves that a pseudo-label is correct.
- **Out-of-distribution unlabeled data:** an unknown class can receive an extremely confident known-class prediction. Graph edges can also connect unrelated populations. Filtering, unknown-class handling, subgroup audits, and a matched-domain supervised baseline are necessary design choices, not automatic properties of the recipes below. [Oliver et al.'s realistic evaluation, full text v4](https://arxiv.org/html/1804.09170v4) directly documents degradation from out-of-class unlabeled examples.

**Transductive versus inductive evaluation.** A transductive experiment exposes the inputs of the particular evaluation pool during training, while hiding their labels. This is legitimate when declared, but it is not an estimate of performance on untouched future inputs. Inductive evaluation reserves a separate pool whose features and labels never participate in fitting or model selection. Graph inference commonly predicts only existing vertices; an out-of-sample rule or graph rebuild must be specified. Conversely, a transductively trained SVM still has a decision function: its existence does not retroactively make a transductive evaluation inductive.

**Label-budget protocol.** Report training labels, validation labels, teacher-training labels, and labels involved in augmentation or checkpoint selection separately. A claim of "100 labels" may mean 100 labels in the gradient objective plus thousands used to tune hyperparameters. Declare whether the draw is class-balanced, naturally imbalanced, grouped by patient or document source, or stratified in another way. Repeat both label draws and model initializations; report the statistic and its uncertainty. Record unlabeled-pool size, provenance, duplicate filtering, class overlap, pretraining, augmentations, backbone, training steps, and checkpoint rule. Never select the best test checkpoint for a deployment estimate. Some historical tables below do report best-checkpoint results; they are identified rather than silently compared with test-locked experiments.

In the worked examples, an inference "decision" means how a predicted class would be used in the named research task. It does not imply that a hospital, search engine, or other organization deployed the model. Technical reasons for suitability are this volume's analysis unless explicitly attributed to a source.

## 2.1 Self-labeling and multiple-view approaches

These methods turn model predictions into additional supervision. Their main differences are who supplies a target, what makes that target admissible, and how diversity is maintained.

### 2.1.1 Self-training and Pseudo-Label

**Name:** Self-training / pseudo-labeling; the neural reference instantiation here is Lee's Pseudo-Label.

**Category & sub-category:** Semi-supervised learning; self-labeling with a single classifier.

**Originating paper/vendor/year:** Self-training predates deep learning. Dong-Hyun Lee's [2013 ICML workshop paper, *Pseudo-Label*](https://www.kaggle.com/blobs/download/forum-message-attachment-files/746/pseudo_label_final.pdf) popularized a particularly simple neural implementation; it did not invent the entire self-training family. The link is a public copy of the original workshop paper.

**Core mechanism:** Fit a classifier using labeled data, predict labels for unlabeled inputs, then train against some of those predictions. A common modern variant retains $`\hat y=\arg\max_c p_\theta(c\mid u)`$ only when confidence exceeds a threshold. Lee's original formulation instead recomputes hard targets during training and gradually increases their loss weight; a confidence cutoff is not part of its defining procedure. Offline rounds with a frozen teacher and online regeneration of targets are different implementations of the same broad idea.

**Inputs/outputs and typical data types:** Labeled and unlabeled examples in a shared feature space; outputs are pseudo-labels and a classifier for new examples. Applications include images, sparse text vectors, and tabular records, provided the base learner can handle them.

**Strengths and limitations:** The wrapper is simple, architecture-agnostic, and easy to compare with a supervised baseline. It can exploit abundant in-domain data without constructing a similarity graph. Its central weakness is self-reinforcing error; confidence filtering can additionally starve rare or difficult classes. Predicting a label does not supply independent evidence for that label.

**Computational complexity / scalability notes:** Each offline labeling round costs approximately $`O(n_uF)`$, followed by the chosen learner's retraining cost. Online neural training is roughly $`O(EnF)`$ for fixed passes per example, with potentially different labeled/unlabeled sampling rates. Hard-label caches require $`O(n_u)`$ storage; full probability vectors require $`O(n_uC)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Lee studies handwritten-digit recognition on MNIST. In the 600-labeled-example experiment, pixel vectors enter a one-hidden-layer network; its current digit predictions become targets for unlabeled training images; the resulting classifier assigns a digit to each test image. Table 2 reports **8.57% test error for the dropout network and 5.03% after adding Pseudo-Label**, without the additional denoising-autoencoder pretraining used in another row. This comparison makes self-labeling's contribution easier to interpret than mixing pretraining and pseudo-labeling results. Hyperparameters were selected with a validation set, so 600 is a training-label count, not an all-inclusive annotation budget. No production OCR deployment or business KPI is reported. [Original experiment](https://www.kaggle.com/blobs/download/forum-message-attachment-files/746/pseudo_label_final.pdf).

**Notable vendor implementations/libraries:** [scikit-learn's `SelfTrainingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html) wraps estimators and supports selection policies. It is not an implementation of every detail of Lee's neural training schedule.

**Architecture diagram description:** Lee's actual MNIST backbone is `784 pixels -> dense 5,000 -> dense 10 outputs`. Optional denoising-autoencoder initialization is a separate stage, not a mandatory part of self-training.

**Activation functions used and why:** ReLU supplies piecewise-linear, potentially sparse hidden responses. Importantly, the original experiment uses **independent sigmoid outputs**, not a softmax; the paper explicitly favors their saturation regions even though MNIST classes are mutually exclusive. Many later implementations instead use a categorical softmax.

**Loss function(s):** The original network uses summed binary cross-entropies against one-hot true or pseudo-labels, with an epoch-dependent multiplier on the unlabeled term. A modern categorical implementation normally uses softmax cross-entropy. These losses should not be silently substituted when reproducing the original result.

**Optimization algorithm(s):** Mini-batch SGD with momentum. The paper reports initial learning rate 1.5, exponential multiplication by 0.998 per epoch, and momentum increasing from 0.5 to 0.99 over 500 epochs. Its update also scales the new gradient by $`1-\text{momentum}`$, so copying only the learning-rate scalar into another optimizer is not faithful. These settings belong to that sigmoid/dropout implementation, not to all pseudo-labeling.

**Regularization techniques:** Dropout and a delayed unlabeled-loss ramp; without pretraining, the original ramp starts at epoch 100 and reaches weight 3 at epoch 600. Denoising pretraining changes the schedule and is reported separately.

**Backpropagation considerations:** Treat the hard argmax target as fixed for its update; do not differentiate through label selection. Incorrect saturated targets can produce misleading or weak corrective gradients. Stable logit-based cross-entropy avoids numerical overflow.

**Parameter count / scaling behavior:** The stated affine layers have $`784(5000)+5000+5000(10)+10=3,975,010`$ parameters, an arithmetic count excluding an optional pretraining decoder. The wrapper itself adds no learned parameters.

**Training paradigm:** Supervised anchoring followed by joint supervised and self-labeled training; optional unsupervised pretraining must be counted as an additional stage.

**Hardware/parallelism considerations:** A single GPU can handle the reference MLP. At larger scale, prediction and retraining can be data-parallel, but pseudo-label versioning and refresh cadence become part of reproducibility.

### 2.1.2 Co-training

**Name:** Co-training.

**Category & sub-category:** Semi-supervised learning; multiple-view self-labeling.

**Originating paper/vendor/year:** Avrim Blum and Tom Mitchell, [*Combining Labeled and Unlabeled Data with Co-Training*, COLT 1998](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf).

**Core mechanism:** Represent each example through two views, train a classifier on each, and let confident predictions from one view expand the labeled information available to the other. The idealized analysis assumes that both views suffice for prediction and are conditionally independent given the class, with additional learnability conditions. Merely splitting a feature vector in half does not establish these assumptions. In practice, complementary errors can still make the procedure useful without exact independence.

**Inputs/outputs and typical data types:** Paired views of the same objects, a small labeled seed, and a larger unlabeled pool. The original example uses webpage text and incoming hyperlink anchor text. Outputs are two classifiers, optionally combined, and labels for previously unlabeled examples.

**Strengths and limitations:** One view can provide evidence unavailable to the other, making this less circular than a classifier teaching only itself. However, correlated mistakes can still be exchanged and amplified. Missing views, weak views, class imbalance, and confidence scores that are not comparable across classes complicate selection. A multi-view dataset is not automatically a co-training-compatible dataset.

**Computational complexity / scalability notes:** For $`R`$ rounds, the cost is the sum of two learners' repeated training costs plus prediction over candidate pools. Sparse multinomial naive Bayes scales with processed nonzero feature counts rather than a dense $`nd`$ matrix. The two fits can run concurrently, but label exchange synchronizes rounds; there is no universal polynomial bound for arbitrary wrapped learners.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The original web-page study uses **1,051 pages from four universities**, targeting course homepages versus other pages. Each run holds out 263 pages, starts from **3 positive and 9 negative labels**, and uses the remaining 776 pages as unlabeled data. Page words and incoming-link words enter separate naive Bayes classifiers; exchanged predictions improve a combined course-page classifier, whose decision identifies pages for a course-oriented index. Across five random splits, Table 2 reports combined-classifier error falling from **11.1% with supervised training to 5.0% with co-training**. The technical fit is that link text supplies evidence beyond page content; that is not a claim about a deployed university search product. No business KPI is reported. [Study and protocol](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf).

**Notable vendor implementations/libraries:** The paper supplies the original algorithm and naive Bayes instantiation. [scikit-learn's `MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) can supply view classifiers, but is not a complete co-training wrapper. This entry describes the non-neural original implementation; using neural view encoders changes the training and hardware requirements.

### 2.1.3 Tri-training

**Name:** Tri-training.

**Category & sub-category:** Semi-supervised learning; agreement-based ensemble self-labeling without prescribed views.

**Originating paper/vendor/year:** Zhi-Hua Zhou and Ming Li, [*Tri-Training: Exploiting Unlabeled Data Using Three Classifiers*, IEEE TKDE, 2005](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde05.pdf).

**Core mechanism:** Bootstrap the labeled data to initialize three classifiers. When two agree on an unlabeled example, their agreed label can supervise the third. The original procedure also estimates the error of the agreeing pair on labeled data and controls the number of added examples; indiscriminately adding every agreement is not the full algorithm. The third classifier need not disagree with the pair. Repeated updates seek useful diversity without requiring two naturally separate feature views.

**Inputs/outputs and typical data types:** One labeled set and one unlabeled set with the same schema. Outputs are three updated classifiers and a majority-vote prediction. The reference implementation described here uses decision trees on tabular features, not neural networks.

**Strengths and limitations:** Unlike co-training, tri-training does not require sufficient and redundant views or calibrated probability outputs. Agreement offers a practical selection signal. It is not equivalent to independent corroboration: bootstrap models can share a systematic blind spot, and very small labeled sets provide unreliable error estimates. The updating and sample-size safeguards matter precisely because agreement alone can be wrong.

**Computational complexity / scalability notes:** Each round entails three candidate-labeling operations and up to three retrainings. With tree learners, cost depends on tree depth, split search, and training-set growth. For a fixed fitted tree, a prediction follows a path of approximately its depth. Storing several evolving pseudo-labeled sets and repeatedly fitting the ensemble can dominate any savings in manual labeling.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Zhou and Li evaluate the Wisconsin Diagnostic Breast Cancer dataset, listed as **569 records with 30 attributes**. About one quarter is held out; their 80%-unlabeled protocol hides labels for 80% of the remaining training pool. Features enter three bootstrapped J4.8 trees; agreement-selected records are added to the third tree's training data; majority voting produces the benchmark's diagnostic class. Table III reports error changing from **0.094 to 0.075** for the initial versus final tri-training ensemble, averaged over three random partitions. The methodological fit relative to co-training is that this dataset does not supply the required redundant views. This is retrospective classification research, not evidence of clinical deployment, patient benefit, or a verified medical business KPI. Exact per-split label counts depend on partition rounding. [Original tables and evaluation](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde05.pdf).

**Notable vendor implementations/libraries:** The original study uses J4.8 trees, alongside separately evaluated base-learner alternatives. [Weka's J48](https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/J48.html) supplies the tree learner; this does not mean its base classifier automatically performs tri-training. Neural base learners are possible extensions, not the instantiation documented here.

### 2.1.4 Noisy Student

**Name:** Noisy Student training.

**Category & sub-category:** Semi-supervised learning; large-scale teacher-student self-training with student noise.

**Originating paper/vendor/year:** Qizhe Xie and colleagues, [*Self-training with Noisy Student improves ImageNet classification*, CVPR 2020; arXiv submission 2019](https://arxiv.org/html/1911.04252v4). The work is associated with Google Research.

**Core mechanism:** Train a teacher on labeled images, generate pseudo-labels without student-style noise, and train an equal-sized or larger student on labeled and pseudo-labeled data. Noise is applied to the student, forcing it to predict stable targets under more difficult conditions. Promote the student to teacher and repeat. Teacher capacity, data filtering, class rebalancing, and noise are substantive parts of the procedure, not merely implementation decorations.

**Inputs/outputs and typical data types:** Labeled images, a very large candidate image pool, and teacher predictions; outputs are a stronger image classifier and optionally another generation of pseudo-labels. Soft or hard teacher labels are possible; their storage and loss behavior differ.

**Strengths and limitations:** The method can improve an already strong classifier, so its usefulness is not restricted to tiny label budgets. A larger student need not simply copy teacher errors. Nevertheless, the method requires substantial compute and a sufficiently relevant image pool. Class balancing can reproduce incorrect teacher assignments, and a private corpus limits exact external reproduction.

**Computational complexity / scalability notes:** Each round combines teacher inference over candidates with multiple student-training epochs. Teacher and student forward costs need not be equal. Offline pseudo-labeling is a major independent workload; soft targets require space proportional to the number of stored class probabilities.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** For ImageNet recognition, the study combines the full labeled ImageNet training set with **300 million candidate JFT images used without their labels**. Filtering and rebalancing produce **130 million sampled training items representing 81 million unique images**; those numbers are not interchangeable. A teacher labels candidates, an EfficientNet student learns from the filtered mixture, and the final classifier selects an ImageNet category. The reported EfficientNet-L2 result is **88.4% top-1 accuracy**, with 480 million parameters. Student noise offers a technical advantage over straightforward imitation, but the result also depends on capacity, data, iterations, and resolution handling. This is a benchmark result, not a demonstrated commercial image-search KPI. [Data preparation and Tables 2 and 8](https://arxiv.org/html/1911.04252v4).

**Notable vendor implementations/libraries:** Google's [Noisy Student research repository](https://github.com/google-research/noisystudent) and released EfficientNet models. Availability of those artifacts does not establish use in a specific Google product.

**Architecture diagram description:** The cited example is **EfficientNet-L2**: `image -> convolutional stem -> repeated expanded mobile inverted-bottleneck blocks with depthwise convolution and squeeze-excitation -> pooling -> classification head`. The teacher and student are separate networks; the recipe is not itself a new block architecture.

**Activation functions used and why:** EfficientNet uses smooth, self-gating Swish-type nonlinearities in its feature extractor, bounded sigmoid channel gates in squeeze-excitation, and softmax for categorical output probabilities. The projection inside an inverted bottleneck is linear; not every convolution is followed by the same nonlinearity.

**Loss function(s):** Cross-entropy on genuine and teacher-provided labels. With soft targets this is distribution matching, equivalent to forward KL up to the fixed teacher entropy. The paper concatenates labeled and unlabeled samples when forming the average training loss.

**Optimization algorithm(s):** The [reference optimizer](https://github.com/google-research/noisystudent/blob/master/utils.py) is RMSProp with momentum. The paper's large-model schedule starts at learning rate 0.128 for labeled batch size 2,048 and decays by 0.97 every 2.4 epochs in a 350-epoch run. Warm-up and resolution fine-tuning belong to the concrete implementation.

**Regularization techniques:** RandAugment, dropout, and stochastic depth noise the student. The paper reports final-layer dropout 0.5 and final-block survival probability 0.8. Candidate filtering and class balancing are data-selection controls, not guarantees of clean labels.

**Backpropagation considerations:** Teacher-generated targets are fixed during a student update; gradients do not flow into an offline teacher. Large-batch normalization, different image resolutions, and strong augmentation affect optimization independently of the pseudo-label rule.

**Parameter count / scaling behavior:** EfficientNet-L2 has **480M parameters** in this study. Its reported training/test resolutions are 475/800. An additional teacher increases training storage or offline preprocessing, but only the final student is required for ordinary inference.

**Training paradigm:** Supervised teacher training, pseudo-label generation and filtering, noisy student training, repeated teacher replacement, and final resolution adjustment.

**Hardware/parallelism considerations:** This is a distributed accelerator-scale example. Teacher inference can be sharded; student gradients and normalization statistics require suitable synchronization. It is not a representative cost estimate for applying self-training to a small tabular dataset.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Self-training / Pseudo-Label | Images, text, or tabular data supported by the base learner | Simple way to reuse confident predictions | Confirmation bias and class starvation | Lee's MNIST digit benchmark |
| Co-training | Objects with genuinely complementary paired views | Cross-view supervision adds distinct evidence | Strong view assumptions can fail | Course-page classification using page and link text |
| Tri-training | Single-view tabular or vector data | Agreement without natural feature views | Correlated ensemble errors | Wisconsin Diagnostic Breast Cancer benchmark |
| Noisy Student | Large labeled and unlabeled image collections | Scales teacher-student learning beyond scarce labels | High compute and corpus-reproduction costs | ImageNet with unlabeled JFT candidates |

## 2.2 Low-density boundaries and manifold regularization

The following methods use unlabeled geometry directly rather than relying on a neural augmentation pipeline. Low-density separation and graph smoothness are related inductive biases, but they produce different objectives and failure modes.

### 2.2.1 Transductive and semi-supervised SVM

**Name:** Transductive SVM (TSVM) / semi-supervised SVM (S3VM), scoped to margin-based classification with unknown labels optimized jointly.

**Category & sub-category:** Semi-supervised learning; low-density decision boundaries.

**Originating paper/vendor/year:** Transductive inference is associated with Vapnik's statistical learning framework. Thorsten Joachims's [*Transductive Inference for Text Classification using Support Vector Machines*, ICML 1999](https://www.cs.cornell.edu/people/tj/publications/joachims_99c.pdf) provides the influential text-classification formulation and scalable training procedure used here.

**Core mechanism:** Jointly choose a large-margin classifier and labels for unlabeled examples. A representative binary objective is

$$
\tfrac12\lVert w\rVert^2+
C_l\sum_{i\in L}[1-y_if(x_i)]_+
+C_u\sum_{i\in U}[1-|f(x_i)|]_+.
$$

Here $`f(x)=w^\top\phi(x)+b`$, and $`C_l,C_u`$ weight the two hinge penalties. The last term penalizes placing unlabeled examples inside the margin. A class-balance constraint or prior is important: otherwise assigning almost everything to one class can be attractive. Alternating label changes and SVM fitting, often increasing $`C_u`$ gradually, yields a practical approximation, not a globally solved convex SVM.

**Inputs/outputs and typical data types:** Labeled feature vectors plus a particular unlabeled pool; outputs are pool labels and an SVM decision function. Sparse, high-dimensional text is the original motivating case. A semi-supervised use may evaluate the resulting function on a separate future sample.

**Strengths and limitations:** The learner can place a boundary using otherwise unobserved population structure. It is unsuitable when low-density gaps do not align with classes or the assumed class ratio is wrong. The unknown-label optimization introduces local minima and initialization sensitivity absent from the ordinary convex binary SVM problem.

**Computational complexity / scalability notes:** Complexity depends on the inner SVM solver, kernel, cache, number of label switches, and continuation schedule. A dense kernel matrix needs $`O(n^2)`$ memory; linear text implementations exploit sparsity. Reporting the ordinary supervised SVM cost alone omits the outer nonconvex search.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Joachims uses Reuters-21578 with the ModApte split: **9,603 training documents and 3,299 test documents**, considering the ten most frequent categories while retaining documents. In one experiment only **17 labeled training documents** are supplied and the 3,299 target document vectors are exposed without labels. Stemmed, weighted text vectors enter the joint margin/label optimization; category decisions organize the target news collection. The paper reports improved precision/recall-breakeven performance over an inductive SVM in this scarce-label Reuters setting; no numerical value is inferred here from a plot. The technical fit is access to the particular collection to be organized. This is explicitly **transductive**, not an untouched-test or production news-service result. No business KPI is reported. [Sections 5.1-5.3](https://www.cs.cornell.edu/people/tj/publications/joachims_99c.pdf).

**Notable vendor implementations/libraries:** Joachims's [SVMlight](https://www.cs.cornell.edu/people/tj/svm_light/) supports transductive learning. An ordinary `SVC` trained only on known labels is not a TSVM, even if predictions on unlabeled examples are subsequently inspected.

### 2.2.2 Laplacian SVM and manifold regularization

**Name:** Laplacian SVM (LapSVM), within the manifold-regularization framework; Laplacian regularized least squares is a related loss variant, not an alias.

**Category & sub-category:** Semi-supervised learning; kernel methods with graph-based manifold smoothness.

**Originating paper/vendor/year:** Mikhail Belkin, Partha Niyogi, and Vikas Sindhwani, [*Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples*, JMLR 2006](https://www.jmlr.org/papers/v7/belkin06a.html).

**Core mechanism:** Build a similarity graph over labeled and unlabeled examples and regularize a kernel function in two ways:

$$
\frac1{n_l}\sum_{i\in L}\ell(y_i,f(x_i))
+\gamma_A\lVert f\rVert_{\mathcal H}^2
+\frac{\gamma_I}{n^2}\mathbf f^\top L_G\mathbf f.
$$

The RKHS term controls the ambient function; the graph-Laplacian term penalizes variation along nearby observations. With hinge loss the method is LapSVM; squared loss produces Laplacian RLS. A representer theorem yields a finite kernel expansion over the observed inputs, providing an explicit function for new inputs rather than only a table of vertex labels.

**Inputs/outputs and typical data types:** Feature vectors, labels on a subset, a kernel, and a similarity graph. Outputs include a fitted kernel classifier and predictions for new examples. Images, speech features, and document representations are common research inputs.

**Strengths and limitations:** Combines a genuine out-of-sample function with unlabeled geometry. For a fixed graph and suitable loss, fitting avoids the TSVM's combinatorial label assignment. Bad neighbors, inappropriate distance scaling, and smoothness across a real class boundary can nonetheless harm classification. Graph construction and regularization strengths require validation.

**Computational complexity / scalability notes:** Exact dense similarities and kernels require $`O(n^2)`$ storage; dense linear-algebra approaches can reach cubic time. Sparse graph multiplication is $`O(m)`$ for a scalar prediction vector, but does not by itself remove a dense kernel matrix. Iterative primal solvers and approximations change these costs; convergence and stopping rules must be reported.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The paper's one-versus-rest USPS digit experiment uses **50 labeled and 1,957 unlabeled examples**, with ten random splits. Image-derived features enter the kernel and neighborhood graph; fitting combines labeled classification with graph smoothness; the largest class score becomes the digit decision. Table 3 reports mean error **12.7% for LapSVM versus 23.6% for SVM**. This particular comparison uses the USPS test-set pool as a semi-supervised/transductive benchmark; it is not the paper's separate out-of-sample experiment. The latter separately examines generalization to new images. Graph regularization fits the hypothesis that nearby handwriting representations carry useful class structure; it is not proof that this hypothesis holds for every recognition dataset. No production OCR KPI is reported. [Original experimental section](https://www.jmlr.org/papers/volume7/belkin06a/belkin06a.pdf).

**Notable vendor implementations/libraries:** The [Melacci Laplacian SVM library](https://www3.diism.unisi.it/~melacci/lapsvmp/) implements primal training and related classifiers. Its documentation explicitly notes that the implementation still stores the whole kernel matrix. It is research software, not a claim of a hosted vendor service.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Transductive / semi-supervised SVM | Sparse text and vector data with meaningful density gaps | Places margins using the target input distribution | Nonconvex label search and class-balance assumptions | Reuters-21578 topic assignment |
| Laplacian SVM / manifold regularization | Kernel-compatible features with useful neighborhoods | Combines graph smoothness with an out-of-sample function | Graph quality and dense kernel costs | USPS digit classification |

## 2.3 Graph label inference

Here labels are inferred on a graph whose vertices include the unlabeled observations. These entries concern classical fixed-graph inference, not trainable graph-convolution or graph-attention networks.

### 2.3.1 Label propagation through Gaussian fields and harmonic functions

**Name:** Label propagation, scoped here to the hard-clamped Gaussian-field / harmonic-function formulation.

**Category & sub-category:** Semi-supervised learning; transductive graph label inference.

**Originating paper/vendor/year:** Xiaojin Zhu, Zoubin Ghahramani, and John Lafferty, [*Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions*, ICML 2003](https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf). "Label propagation" is a broader term; this entry does not equate every algorithm bearing that name with this objective.

**Core mechanism:** Construct symmetric nonnegative affinities $`W`$, degrees $`D`$, and the unnormalized Laplacian $`L_G=D-W`$. Fix the labeled rows $`F_L=Y_L`$, and minimize graph Dirichlet energy $`\frac12\sum_{ij}W_{ij}\lVert F_i-F_j\rVert^2`$. Interior vertices become weighted averages of their neighbors. Partitioning the Laplacian gives

$$
F_U=-L_{UU}^{-1}L_{UL}Y_L,
$$

implemented by solving a linear system, not explicitly computing an inverse. Repeated neighbor averaging with the labeled rows reclamped is an iterative route to this harmonic solution under appropriate connectivity conditions.

**Inputs/outputs and typical data types:** A weighted graph or features from which to construct it, together with class labels for some vertices. Output is a class-score vector at each unlabeled vertex. A connected component without any labeled boundary has no uniquely anchored class solution without an additional convention or prior.

**Strengths and limitations:** The formulation has an interpretable smoothness objective and, with suitable anchoring, a unique solution. Hard clamping respects trusted annotations exactly. It also preserves incorrect annotations exactly. Graph homophily, connectivity, and affinity scale are decisive; scores should not automatically be treated as calibrated deployment probabilities.

**Computational complexity / scalability notes:** Dense construction and storage are quadratic in $`n`$. A direct dense solve over $`n_u`$ unknown vertices is cubic in $`n_u`$, apart from multiple-class right-hand sides. Sparse iterations cost approximately $`O(smC)`$, with convergence dependent on graph conditioning; exact nearest-neighbor construction can itself be expensive.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Zhu and colleagues study handwritten digits from the **CEDAR/Buffalo digit database**, represented after preprocessing by 256-dimensional image vectors. Labeled digits provide boundary values, image similarities produce graph edges, harmonic inference supplies scores for remaining vertices, and a score-based rule assigns digit classes. Section 7 reports label-budget curves comparing harmonic methods with nearest-neighbor and RBF classifiers. It also evaluates **class mass normalization**, which adds class-prior information and must not be mistaken for the unmodified hard-clamped solution. This source supports a concrete recognition experiment, but no exact error value is inferred from its curves here and no production OCR KPI is reported. The technical rationale is borrowing labels along handwriting neighborhoods rather than relying only on distance to an isolated labeled prototype. [Experiment and formulation](https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf).

**Notable vendor implementations/libraries:** [scikit-learn's `LabelPropagation`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html) supplies a hard-clamped propagation implementation with supported graph kernels. Kernel construction and optional class-prior postprocessing differ from choices in the original study. Its out-of-sample prediction rule should be distinguished from inference on the training graph.

### 2.3.2 Label spreading and local-and-global consistency

**Name:** Label spreading / learning with local and global consistency.

**Category & sub-category:** Semi-supervised learning; normalized graph diffusion with soft label retention.

**Originating paper/vendor/year:** Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Scholkopf, [*Learning with Local and Global Consistency*, NIPS 2003, proceedings volume published 2004](https://papers.nips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf).

**Core mechanism:** Form normalized similarity $`S=D^{-1/2}WD^{-1/2}`$ and iterate

$$
F^{(t+1)}=\alpha SF^{(t)}+(1-\alpha)Y,\qquad 0<\alpha<1.
$$

The fixed point is $`(1-\alpha)(I-\alpha S)^{-1}Y`$. It balances graph smoothness and fidelity to the initial label indicators. Unlike hard-clamped propagation, labeled vertices can move away from their initial one-hot scores. Degree normalization also changes what counts as a smooth function. Multiplying all final scores by the same positive constant does not change their argmax, but can matter when interpreting them as probabilities.

**Inputs/outputs and typical data types:** Labeled and unlabeled vertices in a nonnegative affinity graph. Output is a class-score field, usually converted to labels by argmax. Feature normalization, graph construction, and the treatment of zero-degree vertices must be specified.

**Strengths and limitations:** Soft retention can reduce the damage caused by a single erroneous label, and normalized diffusion moderates degree effects. It does not diagnose which label is wrong. Too much diffusion oversmooths class structure; too little ignores useful unlabeled geometry. The mathematical smoothness assumption remains inappropriate for heterophilous graphs.

**Computational complexity / scalability notes:** Sparse diffusion costs $`O(smC)`$, with $`O(m+nC)`$ storage after graph construction. A dense direct solve can require $`O(n^3)`$ time and $`O(n^2)`$ memory. Iteration count depends on $`\alpha`$, spectrum, and tolerance; taking $`\alpha`$ close to one can slow convergence.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The original digit experiment uses **3,874 USPS images from digits 1-4**, with class sizes 1,269, 929, 824, and 852. Pixels define affinities; a few labeled vertices seed diffusion; the resulting score maximum assigns a digit to each unlabeled image. The main curves average 100 trials and favor consistency-based inference over the tested supervised baselines. A separately described **one-pixel-jittered affinity variant** reaches approximately **1% error with 30 labeled points**; that is not the result of every plain RBF label-spreading model. The paper also acknowledges using optimal parameters for comparison methods, a limitation for realistic label-budget claims. Neighborhood smoothing fits digit-shape variation better than relying only on a few prototypes, but this is a transductive research result with no business KPI. [Section 4.2 and model-selection discussion](https://papers.nips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf).

**Notable vendor implementations/libraries:** [scikit-learn's `LabelSpreading`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html) implements normalized graph propagation with a clamping factor. It is not a graph neural network and requires no backpropagation through trainable layers.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Harmonic label propagation | Homophilous similarity graphs | Interpretable, hard-anchored smooth solution | Unanchored components and incorrect fixed labels | CEDAR/Buffalo handwritten-digit study |
| Label spreading | Similarity graphs where soft label retention is useful | Degree-normalized diffusion and soft clamping | Graph mistakes and oversmoothing | USPS digits 1-4 benchmark |

## 2.4 Entropy and consistency

Entropy minimization encourages decisive predictions. Consistency regularization asks related observations or models to agree. A classifier can be consistently uncertain, confidently wrong, or consistently wrong; these are distinct conditions. The following objectives therefore need supervised anchoring and careful evaluation.

### 2.4.1 Entropy minimization

**Name:** Minimum-entropy regularization / entropy minimization.

**Category & sub-category:** Semi-supervised learning; confidence-based regularization of a predictive distribution.

**Originating paper/vendor/year:** Yves Grandvalet and Yoshua Bengio, [*Semi-supervised Learning by Entropy Minimization*, NIPS 2004](https://papers.nips.cc/paper_files/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf). Their original experiments include logistic and kernel-logistic models; the neural instantiation below is the explicitly evaluated entropy term in [Miyato et al.'s VAT study](https://arxiv.org/html/1704.03976v2).

**Core mechanism:** Add $`\lambda\,\mathbb E_{u\in D_U}H(p_\theta(\cdot\mid u))`$ to supervised risk. A uniform $`C`$-class prediction has entropy $`\log C`$; a point mass has entropy zero. Under a suitable cluster assumption, discouraging uncertainty at observed inputs encourages boundaries away from them. It does not identify which confident class is correct. Unlike hard pseudo-labeling, entropy minimization differentiates a smooth function of the entire current probability vector.

**Inputs/outputs and typical data types:** A probabilistic classifier, labeled examples, and unlabeled examples. Output is the same type of classifier; the regularizer neither reconstructs inputs nor creates a new architecture. It applies to differentiable tabular, image, and language models.

**Strengths and limitations:** Inexpensive and easy to combine with other objectives. By itself it can favor overconfidence or class collapse, particularly with a large unlabeled weight. Low entropy is not calibration, uncertainty estimation, or OOD detection. The original paper explicitly analyzes the assumptions needed for unlabeled data to be informative.

**Computational complexity / scalability notes:** Entropy evaluation costs $`O(BC)`$ after a batch's predictions exist. Processing extra unlabeled inputs still incurs backbone forward/backward costs. No graph, additional teacher, or all-pairs distance calculation is inherently required.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Miyato and colleagues examine natural-image classification on CIFAR-10 with **4,000 labeled training examples**. Their Conv-Large classifier processes augmented images; VAT promotes local stability, and the extra entropy penalty makes predictions at unlabeled observations more decisive; a ten-class score vector becomes an object-category decision. The augmented-data comparison reports **11.36% error for VAT and 10.55% for VAT+EntMin**, with reported standard deviations 0.34 and 0.05 percentage points. This is evidence for adding entropy to that particular recipe, **not** a result for entropy minimization alone. Confidence regularization complements smoothness, but the paired experiment does not establish a general superiority to pseudo-labeling. No deployed recognition system or business KPI is reported. [Original ablation table](https://arxiv.org/html/1704.03976v2).

**Notable vendor implementations/libraries:** Entropy can be expressed directly using stable log-softmax operations in PyTorch or TensorFlow. The [VAT TensorFlow reference](https://github.com/takerum/vat_tf) supplies a concrete neural context. A generic entropy function does not implement a complete SSL protocol.

**Architecture diagram description:** The neural example uses the VAT paper's **Conv-Large**, not an unspecified modern backbone: `RGB image -> three 128-channel convolutions -> pool/dropout -> three 256-channel convolutions -> pool/dropout -> 512-channel convolution -> 256- and 128-channel 1x1 convolutions -> global pooling -> 10-way head`.

**Activation functions used and why:** Leaky ReLU with slope 0.1 in this CNN, followed by softmax class probabilities. Leakage retains a gradient for negative feature responses; softmax provides the normalized distribution whose entropy is penalized.

**Loss function(s):** The standalone principle is supervised cross-entropy plus positive-weight predictive entropy. The worked neural example additionally includes the VAT KL penalty. Minimizing the entropy of individual predictions is different from maximizing the entropy of the aggregate class distribution.

**Optimization algorithm(s):** For the cited neural study, Adam starts at 0.001. Appendix D specifies a 48,000-update validation schedule with linear decay over its final 16,000 updates and separately reports extending final CIFAR-10 runs to 200,000 updates. That experimental budget is not intrinsic to entropy minimization.

**Regularization techniques:** The example also uses batch normalization, dropout, image augmentation, and VAT. Their benefits must not be attributed solely to the entropy term. The entropy coefficient itself controls a bias toward confidence rather than conventional weight shrinkage.

**Backpropagation considerations:** Differentiate through both appearances of $`p`$ in $`-\sum p\log p`$. Replacing this with cross-entropy against a detached copy of the same probabilities is not equivalent and can give a zero logit gradient. Stable log-softmax avoids evaluating $`\log 0`$.

**Parameter count / scaling behavior:** No additional learned parameters beyond the classifier. The described Conv-Large has approximately 3.1M convolution/head weights by arithmetic from its widths; normalization adds small additional state.

**Training paradigm:** Joint supervised and unlabeled regularization; in the worked example, jointly optimized with VAT rather than as a separate pretraining stage.

**Hardware/parallelism considerations:** The entropy arithmetic is small compared with CNN computation and is easily data-parallel. Numerical stability and unlabeled-batch composition matter more than dedicated hardware for the regularizer.

### 2.4.2 Pi Model

**Name:** Pi Model, conventionally written $`\Pi`$-Model.

**Category & sub-category:** Semi-supervised learning; same-model stochastic consistency.

**Originating paper/vendor/year:** The named formulation is described by Samuli Laine and Timo Aila in [*Temporal Ensembling for Semi-Supervised Learning*, ICLR 2017; arXiv submission 2016](https://arxiv.org/pdf/1610.02242).

**Core mechanism:** Evaluate the same example twice with independent stochastic perturbations, obtaining $`p_\theta(y\mid a_1(x),\xi_1)`$ and $`p_\theta(y\mid a_2(x),\xi_2)`$. Penalize their squared difference while applying supervised cross-entropy where labels exist. Here $`a`$ can change the input and $`\xi`$ can represent dropout or other model noise. Agreement encourages predictions that are stable over the perturbation neighborhood, not just at labeled points.

**Inputs/outputs and typical data types:** A shared labeled/unlabeled input pool and a stochastic classifier. The output is one ordinary classifier. Images are the source experiment, but the construction transfers only where defensible perturbations exist.

**Strengths and limitations:** Avoids an explicit pseudo-label cache or separate teacher model. However, two simultaneous predictions may agree on an incorrect answer, and excessive invariance can erase task information. The method requires careful augmentation design and an initially restrained consistency weight.

**Computational complexity / scalability notes:** Approximately two stochastic evaluations per processed example, with gradients through both branches. For fixed augmentation cost, training remains $`O(EnF)`$ with a larger constant than supervised training. Only one set of learned parameters is stored, but both branches' activations contribute to memory use.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Laine and Aila use CIFAR-10 with **4,000 labeled images** and the remaining training images available without labels. Cropped or translated image views pass through the same convolutional network with different noise; the consistency penalty keeps the output class distribution stable; the final argmax supplies a ten-category recognition decision. Their augmented Pi Model reports **12.36% test error, with standard deviation 0.31 percentage points**. The technical advantage over labeled-only augmentation is that unlabeled examples also constrain invariance. This is a controlled recognition benchmark, not a commercial vision deployment; no business KPI is reported. The paper's validation and augmentation protocol is part of the result. [Table 1 and implementation appendix](https://arxiv.org/pdf/1610.02242).

**Notable vendor implementations/libraries:** Laine's [temporal-ensembling research code](https://github.com/smlaine2/tempens) includes the Pi Model comparison. Later unified SSL libraries may use different backbones and should not be assumed to reproduce the original numbers automatically.

**Architecture diagram description:** The actual reference CNN is `32x32 RGB -> 128/128/128 convolutions -> max-pool/dropout -> 256/256/256 convolutions -> max-pool/dropout -> 512 convolution -> 256 then 128 pointwise convolutions -> global average pool -> dense 10`. The network is evaluated twice with shared weights.

**Activation functions used and why:** Leaky ReLU, slope 0.1, retains negative-side feature gradients; softmax supplies normalized probabilities for consistency comparison. This implementation uses weight normalization and mean-only batch normalization, not an assumption that every consistency CNN uses ordinary full batch normalization.

**Loss function(s):** Labeled cross-entropy plus a ramped mean-squared difference between the two probability vectors, evaluated on the appropriate labeled and unlabeled inputs. Normalization across class dimensions and examples affects the coefficient's meaning.

**Optimization algorithm(s):** Adam with maximum learning rate 0.003. The source trains for 300 epochs, ramps learning rate and consistency weight during the first 80 epochs, and reduces the learning rate during the final 50 epochs. These are reference settings, not part of the abstract Pi objective.

**Regularization techniques:** Gaussian input noise, dropout, weight normalization, mean-only batch normalization, and dataset-appropriate geometric augmentation. Horizontal image flips should not be inherited indiscriminately by digit tasks.

**Backpropagation considerations:** In the original Pi Model, **both predictions receive gradients**. Treating one branch as a detached teacher is a different implementation. Independent noise is essential; identical deterministic branches would yield no useful consistency signal.

**Parameter count / scaling behavior:** Approximately 3.1M convolution/head weights from the listed widths, plus normalization parameters. Two evaluations do not mean two independently learned parameter sets.

**Training paradigm:** End-to-end supervised anchoring and stochastic consistency, with a warm-up that delays strong reliance on initially unreliable predictions.

**Hardware/parallelism considerations:** A GPU handles the reference CNN. Branches can be batched together, but normalization, augmentation independence, and memory accounting must remain equivalent to the intended two-view computation.

### 2.4.3 Temporal ensembling

**Name:** Temporal ensembling.

**Category & sub-category:** Semi-supervised learning; per-example prediction averaging across training epochs.

**Originating paper/vendor/year:** Samuli Laine and Timo Aila, [*Temporal Ensembling for Semi-Supervised Learning*, ICLR 2017](https://arxiv.org/pdf/1610.02242), first posted in 2016.

**Core mechanism:** Maintain an exponential moving average of each training example's predictions. With $`Z_i^{(t)}=\beta Z_i^{(t-1)}+(1-\beta)p_i^{(t)}`$, use the bias-corrected target $`\tilde Z_i^{(t)}=Z_i^{(t)}/(1-\beta^t)`$ after initialization at zero. Train the current noisy prediction to match a target accumulated from earlier epochs. This ensembles historical outputs, not model weights, and avoids requiring two fresh stochastic predictions for every update.

**Inputs/outputs and typical data types:** A persistently indexed dataset of labeled and unlabeled examples. Output is one classifier; the auxiliary state is a class-probability history for each training example. Stable example identity is essential when shuffling or distributing batches.

**Strengths and limitations:** Historical averaging can provide less noisy targets and reduces repeated forward/backward work relative to the Pi Model. Targets are stale within an epoch, and the method does not naturally accommodate an ever-growing stream of previously unseen examples without cache management. A biased historical prediction may remain influential for several epochs.

**Computational complexity / scalability notes:** Current-network training is approximately one stochastic evaluation per example, plus $`O(nC)`$ target-cache storage and refresh work. For many examples or classes this cache can dominate the wrapper's memory, even though it is inexpensive on small benchmarks. It is not an inference-time ensemble.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On SVHN cropped-house-number recognition, the paper uses **500 labels within the official 73,257-image training set**, without the supplied extra-image split. An image's historical predictions are averaged across epochs, its current noisy prediction is trained toward that average, and inference for the benchmark remains a single digit classifier. Table 2 reports **5.12% error with standard deviation 0.13 percentage points**, versus **6.65% with standard deviation 0.53** for the augmented Pi Model in that experiment. The technical fit is obtaining a smoother target without doubling current model evaluation; this is not a measured saving in a production address-reading service. No business KPI is reported. [Original SVHN table and appendix](https://arxiv.org/pdf/1610.02242).

**Notable vendor implementations/libraries:** The [authors' implementation](https://github.com/smlaine2/tempens) supplies the original method. A generic exponential-moving-average parameter utility does **not** implement temporal ensembling's per-example target cache.

**Architecture diagram description:** The reference backbone is the same 128/256/512-channel convolutional classifier specified for the Pi Model: `noisy image -> CNN -> current probability vector -> consistency against stored historical target`. Cache storage sits outside the neural architecture.

**Activation functions used and why:** Leaky ReLU with slope 0.1 preserves a gradient for negative feature values; the softmax head produces probabilities that can be averaged across epochs. Averaging logits instead changes the target.

**Loss function(s):** Supervised cross-entropy plus ramped squared distance from current probabilities to the bias-corrected historical probability target. The first epoch must not penalize the model for disagreement with an all-zero uninitialized cache.

**Optimization algorithm(s):** Adam; the source uses maximum learning rate **0.001 for temporal ensembling on SVHN**, rather than its usual 0.003 setting. Training lasts 300 epochs with the 80-epoch ramp-up and 50-epoch ramp-down. Prediction-ensemble decay is 0.6 in the source experiments.

**Regularization techniques:** Dropout, input noise, weight normalization, mean-only batch normalization, and dataset-specific augmentation. Historical averaging is a target regularizer, not a substitute for these mechanisms.

**Backpropagation considerations:** Targets from previous epochs are detached stored values; gradients do not traverse the entire training history. Incorrect bias correction or mismatched example indices silently changes the objective.

**Parameter count / scaling behavior:** The CNN remains approximately 3.1M convolution/head weights. Auxiliary memory grows as $`nC`$, unlike Mean Teacher's dataset-size-independent teacher weights.

**Training paradigm:** Joint semi-supervised training with epoch-level target refresh. The final classifier can predict new examples inductively without maintaining their histories.

**Hardware/parallelism considerations:** Distributed training must merge or consistently shard prediction histories. Cache communication and dataset identity can be more important bottlenecks than the averaging arithmetic.

### 2.4.4 Mean Teacher

**Name:** Mean Teacher.

**Category & sub-category:** Semi-supervised learning; consistency with an exponentially averaged parameter teacher.

**Originating paper/vendor/year:** Antti Tarvainen and Harri Valpola, [*Mean teachers are better role models*, NeurIPS 2017](https://arxiv.org/html/1703.01780v6). The cited expanded arXiv version is from 2018.

**Core mechanism:** Train a student with labeled classification loss and agreement with a teacher under perturbed inputs or model noise. After each student update, set $`\theta'_t=\alpha\theta'_{t-1}+(1-\alpha)\theta_t`$. The teacher is an average of weights, rather than a cache of predictions for each example. Consequently, it can supply a target immediately for a newly sampled unlabeled input and refreshes every step instead of every epoch.

**Inputs/outputs and typical data types:** Labeled and unlabeled inputs, independently perturbed student/teacher views, and two parameter states. Output is a single classifier, commonly the final EMA teacher. The paper evaluates both small CNNs and residual architectures; their results must be distinguished.

**Strengths and limitations:** Avoids the dataset-sized temporal-ensembling cache and often stabilizes targets. The teacher remains dependent on the student's history, so it is not an independent source of truth. Excessive averaging slows adaptation; inappropriate augmentation and unreliable early predictions can still cause failure.

**Computational complexity / scalability notes:** One teacher forward pass accompanies student training. EMA updates cost $`O(p)`$ per step and teacher storage is $`O(p)`$, independent of $`n`$. Only the student needs ordinary training gradients; final inference does not require evaluating all historical models.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On SVHN house-number recognition with **250 labels**, the paper's non-residual convolutional Mean Teacher reports **4.35% error**. Normalized digit images receive independent perturbations; the EMA teacher supplies a probability target; student learning combines that target with the few known labels; inference chooses one of ten digit classes. The methodological advantage over temporal ensembling is step-level target updates without keeping a prediction record for every image. This number belongs to the CNN experiment, not the paper's separate, substantially different residual-network results. Standard SVHN training data and the authors' validation-label protocol apply; no production address-processing KPI is reported. [Paper abstract and experimental appendix](https://arxiv.org/html/1703.01780v6).

**Notable vendor implementations/libraries:** [Curious AI's Mean Teacher repository](https://github.com/CuriousAI/mean-teacher) includes TensorFlow and PyTorch research implementations. EMA utilities in other frameworks implement only one component of the method.

**Architecture diagram description:** For this example, use the paper's Table 6 CNN with convolutional groups of 128, 256, and then 512/256/128 channels and a pooled ten-class head. `Student view -> CNN(theta)` and `teacher view -> identical CNN(theta_EMA)` meet at a consistency loss.

**Activation functions used and why:** Leaky ReLU, slope 0.1, keeps negative-side feature gradients; softmax provides teacher/student distributions on a common probability scale. The small reference CNN uses weight normalization and mean-only batch normalization. The paper's residual-network experiments have separate architectural and regularization choices.

**Loss function(s):** Labeled student cross-entropy plus probability-vector mean-squared consistency in the small-CNN experiment. The paper also studies KL-based consistency; choosing it changes the scale and behavior of the coefficient.

**Optimization algorithm(s):** Adam with maximum learning rate 0.003 for the small CNN. In semi-supervised SVHN, learning rate and consistency strength ramp over 40,000 steps, with no final ramp-down; the no-extra-data run lasts 180,000 steps. Teacher decay and Adam's second-moment coefficient change from 0.99 during ramp-up to 0.999 afterward.

**Regularization techniques:** Input noise, translations, dropout, normalization, and teacher averaging. The reference SVHN sampler uses one labeled and 99 unlabeled examples per minibatch; other datasets use different ratios.

**Backpropagation considerations:** Stop gradients through teacher targets. EMA is an explicit update, not gradient descent on teacher loss and not backpropagation through past optimizer steps. Handle normalization state deliberately when copying or averaging models.

**Parameter count / scaling behavior:** Each reference CNN has approximately 3.1M convolution/head weights. Training stores roughly two model states plus student optimizer state; inference needs one selected model.

**Training paradigm:** Joint semi-supervised consistency learning from a student and its continually averaged teacher; no separate pretrained teacher is required by the recipe.

**Hardware/parallelism considerations:** Teacher inference adds compute but less activation memory than a second trainable network. Distributed implementations must keep EMA and normalization state consistent across replicas.

### 2.4.5 Virtual adversarial training

**Name:** Virtual adversarial training (VAT).

**Category & sub-category:** Semi-supervised learning; label-independent local adversarial consistency.

**Originating paper/vendor/year:** Takeru Miyato and colleagues introduced a [distributional-smoothing formulation](https://arxiv.org/abs/1507.00677) in 2015, followed by the expanded [*Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning*](https://arxiv.org/html/1704.03976v2), posted in 2017 and revised in 2018 for the journal work.

**Core mechanism:** Find a small perturbation that most changes the model's current predictive distribution:

$$
r_{\rm vadv}\approx\arg\max_{\lVert r\rVert\le\epsilon}
D_{\rm KL}\!\left(\operatorname{sg}[p_\theta(\cdot\mid x)]
\parallel p_\theta(\cdot\mid x+r)\right).
$$

Then penalize that divergence during training. A finite-difference power-iteration approximation finds a sensitive direction without constructing a Hessian. "Virtual" means the target distribution is the model's prediction, not the unknown true class. Compared with random noise, the perturbation deliberately searches for a locally fragile direction.

**Inputs/outputs and typical data types:** Differentiable inputs or embeddings, labeled and unlabeled examples, and a perturbation norm and radius. Outputs are a regularized classifier; perturbations are training auxiliaries, not generated ground-truth examples.

**Strengths and limitations:** Does not require labels to construct local adversarial directions and need not rely entirely on handcrafted augmentations. The radius is meaningful only relative to preprocessing and feature scale. Local smoothness does not establish global adversarial robustness, and perturbations may leave the natural-data manifold.

**Computational complexity / scalability notes:** Each power iteration adds gradient-based perturbation work, followed by evaluation of the perturbed input. A fixed small iteration count gives a constant-factor increase over ordinary neural training, not a dense Hessian cost. More iterations, longer sequences, or larger backbones increase actual cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Miyato and colleagues classify CIFAR-10 images with **4,000 labeled examples**. The Conv-Large network predicts a distribution, a locally sensitive input direction is estimated, and the classifier learns to preserve its prediction under that perturbation while respecting genuine labels. The augmented-data table reports **11.36% test error with standard deviation 0.34 percentage points for VAT**. The separate 10.55% VAT+EntMin row includes an additional regularizer, as explained above. VAT fits a task where local robustness around unlabeled images is useful even beyond available geometric augmentations. This is an image-classification experiment, not evidence of a deployed adversarially secure system or a business KPI. [Original comparison and Appendix D](https://arxiv.org/html/1704.03976v2).

**Notable vendor implementations/libraries:** [The authors' VAT TensorFlow code](https://github.com/takerum/vat_tf). Generic adversarial-attack libraries do not necessarily implement the same stopped target, norm, or training objective.

**Architecture diagram description:** The cited example is **Conv-Large**: the 128/256-channel convolutional groups, 512/256/128-channel final group, pooling, and ten-class head described in the entropy entry. `Input -> prediction -> perturbation search -> perturbed input -> same CNN` is a training computation graph, not an extra inference architecture.

**Activation functions used and why:** Leaky ReLU with slope 0.1 retains negative-side gradients; softmax supplies the distribution compared by KL in the source CNN. Batch normalization is used; perturbation search must not inadvertently change the normalization behavior being tested.

**Loss function(s):** Labeled cross-entropy plus expected KL divergence to the stopped clean prediction in the estimated adversarial direction. Entropy minimization is an optional **additional** term, not part of VAT's definition.

**Optimization algorithm(s):** Adam starts at 0.001 in Appendix D. Its validation schedule uses 48,000 updates with linear decay over the final 16,000; the source separately extends final CIFAR-10 training to 200,000 updates. Radius selection uses validation data and therefore consumes a tuning budget.

**Regularization techniques:** VAT itself, together with the experiment's dropout, batch normalization, and optional input augmentation. Neither its norm nor its radius is a universal image-independent constant.

**Backpropagation considerations:** Stop the reference probabilities and ordinarily detach the constructed perturbation before the outer update. This avoids differentiating through the inner search. Small finite-difference steps need adequate numerical precision; storing a full Hessian is unnecessary.

**Parameter count / scaling behavior:** No additional learned parameters over the approximately 3.1M-weight Conv-Large backbone. Additional memory holds perturbed activations and input gradients rather than a second independently trained classifier.

**Training paradigm:** End-to-end supervised learning plus label-independent local smoothness on unlabeled, and potentially labeled, observations.

**Hardware/parallelism considerations:** GPU/autodiff support is important for efficient input gradients. Multi-device training must preserve the intended perturbation norm per example rather than accidentally normalizing across a whole distributed batch.

### 2.4.6 Unsupervised Data Augmentation

**Name:** Unsupervised Data Augmentation for Consistency Training (UDA).

**Category & sub-category:** Semi-supervised learning; consistency under strong, domain-appropriate augmentation.

**Originating paper/vendor/year:** Qizhe Xie and colleagues, [*Unsupervised Data Augmentation for Consistency Training*, NeurIPS 2020; arXiv submission 2019](https://arxiv.org/html/1904.12848v6). Despite its name, the classifier training discussed here uses labeled data.

**Core mechanism:** Make a prediction on an original or weakly changed unlabeled input, then train the model to agree on a substantially augmented version. UDA emphasizes the quality of the augmentation: image transformations and text back-translation provide different kinds of semantic invariance. The framework also discusses confidence masking, sharpening, and training-signal annealing, which limits already-easy labeled examples early in training. Those options and their coefficients must be identified per experiment.

**Inputs/outputs and typical data types:** Labeled and unlabeled images or documents, plus an augmentation mechanism. Outputs are task probabilities and a classifier. A text back-translation system can have its own training data and cost; those are not "free" merely because downstream labels are scarce.

**Strengths and limitations:** Makes consistency meaningful over richer variations than small Gaussian noise. It combines with pretrained representations. However, a fluent paraphrase may change a label, and a large pretrained model already embodies substantial upstream data and compute. A twenty-label fine-tuning result is not a twenty-example-from-scratch learning result.

**Computational complexity / scalability notes:** Training includes target-view inference and augmented-view learning, plus augmentation generation. For the text example, a dense Transformer layer costs $`O(T^2d+Td^2)`$ for sequence length $`T`$, when feed-forward width scales with hidden width $`d`$; attention alone is not the full layer cost. Back-translation may be performed offline but still counts toward resource use.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On IMDb sentiment classification, Table 4 uses **20 labeled reviews**. The **BERT_FINETUNE** configuration is BERT-Large additionally pretrained on in-domain unlabeled text, then trained with UDA. Reviews and back-translated variants become token sequences; the consistency-trained classifier outputs positive/negative probabilities; the benchmark decision is the review's sentiment label. The paper reports **4.20% error with UDA versus 6.50% without UDA for that initialization**. It is not the result for randomly initialized Transformers or plain BERT-Large, and it does not establish that only twenty annotations were used across every development stage. Additional unlabeled-corpus counts are not inferred from the headline table. No deployed review-analysis service or business KPI is reported. [Table 4 and Appendix E.1](https://arxiv.org/html/1904.12848v6).

**Notable vendor implementations/libraries:** Google's [UDA research repository](https://github.com/google-research/uda) includes text and image implementations. A generic augmentation library alone does not implement its consistency, sampling, or model-selection protocol.

**Architecture diagram description:** The worked text backbone is **BERT-Large**: `WordPiece and position embeddings -> 24 Transformer encoder layers, hidden width 1,024 and 16 attention heads -> pooled [CLS] representation -> two-class head`. The [BERT paper](https://arxiv.org/html/1810.04805v2) establishes the architecture; UDA's image experiments instead use their own CNN backbones.

**Activation functions used and why:** GELU supplies smooth gating in BERT feed-forward sublayers; softmax normalizes attention weights and class probabilities; the original tanh pooler bounds its transformed representation. LayerNorm and residual connections support optimization. These are backbone choices, not requirements of UDA.

**Loss function(s):** Supervised cross-entropy plus weighted KL consistency from a stopped original-view distribution to the augmented-view prediction. The text setup reports unlabeled weight 1. Image-specific temperature and confidence settings should not be copied into the IMDb result without verification.

**Optimization algorithm(s):** The [text reference optimizer](https://github.com/google-research/uda/blob/master/text/bert/optimization.py) uses an Adam-family update with weight decay, linear warm-up, and linear decay. Appendix E.1 explores fine-tuning rates $`10^{-5},2\times10^{-5},5\times10^{-5}`$; this is a search range, not a claimed single setting for all datasets.

**Regularization techniques:** Back-translation, BERT dropout of 0.1 in the cited fine-tuning setup, and consistency. Input noising must be checked for sentiment-preserving behavior rather than assumed correct because it is linguistically plausible.

**Backpropagation considerations:** Teacher-view predictions are stopped targets; gradients pass through the augmented classifier. Discrete back-translation is not differentiated through as part of ordinary UDA training. Sequence truncation and tokenization must be consistent across views.

**Parameter count / scaling behavior:** BERT-Large has approximately **340M parameters**, plus its small task head. UDA itself adds no mandatory trainable backbone parameters; its augmentation generator, if used, is a separate model and resource.

**Training paradigm:** Self-supervised BERT pretraining, additional in-domain pretraining for BERT_FINETUNE, then semi-supervised task fine-tuning. The stages should not be collapsed into a single supervision label.

**Hardware/parallelism considerations:** The paper's text experiments use a v3-32 Cloud TPU Pod and length-512 sequences. This reports the source setup, not a minimum hardware requirement for all UDA applications. Augmentation throughput and sequence memory can dominate practical cost.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Entropy minimization | Probabilistic models with meaningful low-density separation | Cheap confidence regularizer | Can amplify overconfidence and collapse | Entropy addition in the CIFAR-10 VAT ablation |
| Pi Model | Images or other data with defensible stochastic perturbations | No historical cache or separate teacher weights | Two gradient-bearing views and correlated errors | CIFAR-10 with 4,000 labels |
| Temporal ensembling | Persistently indexed finite datasets | Smooth historical targets with fewer fresh evaluations | Per-example cache and epoch-stale targets | SVHN with 500 labels |
| Mean Teacher | Large or changing unlabeled pools | Step-level EMA targets without a dataset-sized cache | Teacher inherits student biases | SVHN with 250 labels |
| Virtual adversarial training | Differentiable vectors, images, or embeddings | Finds locally sensitive directions without labels | Radius sensitivity and extra input-gradient work | CIFAR-10 with 4,000 labels |
| UDA | Images and text with strong label-preserving augmentation | Effective consistency beyond weak noise | Augmentation errors and upstream-model costs | IMDb sentiment with BERT_FINETUNE and 20 labels |

## 2.5 Combined modern recipes

These recipes combine several mechanisms rather than introducing a unique neural architecture. Their small-image examples commonly use **Wide ResNet-28-2 (WRN-28-2)**, approximately 1.5M parameters: a convolutional stem, three groups of four residual units, widening feature channels through roughly 32/64/128, global pooling, and a class head. The cited SSL implementations use preactivation normalization and leaky ReLU; that is a concrete implementation choice, not a definition of every Wide ResNet. See the [Google reference backbone](https://github.com/google-research/fixmatch/blob/master/libml/models.py) and [TorchSSL backbone](https://github.com/TorchSSL/TorchSSL/blob/main/models/nets/wrn.py).

Comparisons across the original papers are not a leaderboard under one controlled protocol. In particular, later TorchSSL tables reimplement earlier methods and use different checkpoint reporting. The worked examples identify the relevant comparison rather than attributing every numerical difference to a new threshold rule.

### 2.5.1 MixMatch

**Name:** MixMatch.

**Category & sub-category:** Semi-supervised learning; combined label guessing, entropy reduction, and interpolation consistency.

**Originating paper/vendor/year:** David Berthelot and colleagues, [*MixMatch: A Holistic Approach to Semi-Supervised Learning*, NeurIPS 2019](https://arxiv.org/html/1905.02249v2).

**Core mechanism:** Average predictions over several augmentations of an unlabeled image, sharpen the average into a lower-entropy soft target, and mix both labeled and unlabeled examples and their targets. Sharpening with temperature $`\tau_s`$ gives $`q_c\propto\bar p_c^{1/\tau_s}`$. MixUp-style interpolation forms $`x'=\lambda x_a+(1-\lambda)x_b`$ and the corresponding interpolated target. The objective asks the classifier to behave sensibly between examples as well as under augmentation. A guessed label remains a soft distribution rather than necessarily becoming an argmax class.

**Inputs/outputs and typical data types:** Labeled/unlabeled images and label-preserving augmentation. Output is one inductive classifier. Interpolating raw pixels is a modeling bias, not a literal assertion that a blended cat and truck image has an independently observed fractional label.

**Strengths and limitations:** Integrates complementary regularizers and can use low-confidence examples through soft targets. It is more complex than plain pseudo-labeling, and errors can enter through guessing, sharpening, and interpolation simultaneously. Sharpening can make an incorrect target harder to correct; interpolation may be inappropriate in some discrete or structured domains.

**Computational complexity / scalability notes:** Label guessing requires multiple unlabeled forward passes, followed by training on mixed examples. For a fixed augmentation count, the cost remains linear in examples and epochs up to a backbone-dependent multiplier. It has no mandatory dataset-sized prediction cache or graph.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On CIFAR-10, the paper exposes only **250 training labels** from the 50,000-image training pool. Augmented images produce averaged and sharpened class guesses; mixed image/target pairs train WRN-28-2; the resulting argmax supplies an object-category decision on the held-out test set. The paper reports **11.08% error with standard deviation 0.87 percentage points** for this setting. It reports median error over the last twenty checkpoints and uses a separate **5,000-example validation resource for hyperparameter selection**. Thus the training-label count is not the entire development-label budget. The technical fit versus plain pseudo-labeling is combining uncertainty-aware targets with interpolation regularization. No commercial image-recognition KPI is reported. [Implementation details and CIFAR-10 results](https://arxiv.org/html/1905.02249v2).

**Notable vendor implementations/libraries:** Google's [MixMatch research code](https://github.com/google-research/mixmatch), including its [training implementation](https://github.com/google-research/mixmatch/blob/master/mixmatch.py). Its runtime flags should be recorded rather than assumed to match all published experiments.

**Architecture diagram description:** The example is **WRN-28-2**: `augmented image views -> shared residual CNN -> averaged/sharpened targets; mixed images -> same CNN -> supervised and unlabeled losses`. Target construction adds operations, not a new feature-extractor architecture.

**Activation functions used and why:** The reference residual implementation uses leaky ReLU with slope 0.1 to retain negative-side gradients, batch normalization, and softmax to make probability averaging well-defined. Sharpening changes target probabilities; it is not a hidden-layer activation.

**Loss function(s):** Cross-entropy for the mixed examples originating in the labeled partition, plus squared probability error for the mixed unlabeled partition. Both use interpolated targets. The distinction between these two losses is a defining difference from recipes that use cross-entropy everywhere.

**Optimization algorithm(s):** The public implementation uses Adam with default learning rate 0.002. The paper describes a non-decaying learning-rate approach with EMA evaluation and a 16,000-step linear ramp of the unlabeled coefficient. This does not mean that every later reimplementation uses Adam.

**Regularization techniques:** Augmentation, low-temperature sharpening, MixUp interpolation, weight shrinkage, and parameter EMA. Record the effective shrinkage update: the code multiplies its weight-decay flag by learning rate, so the flag alone is not a per-step fractional shrinkage.

**Backpropagation considerations:** Detach guessed targets before their use in the training loss; do not optimize the classifier by moving its own target toward an easier answer in the same computation. Mixing and minibatch interleaving must preserve target alignment and the intended normalization statistics.

**Parameter count / scaling behavior:** Approximately **1.5M parameters** for the cited WRN-28-2. An EMA copy adds training/evaluation state, not another learned architecture. Widening the backbone increases many convolutional parameter counts approximately quadratically in width.

**Training paradigm:** Joint supervised and pseudo-target learning with two augmented predictions per unlabeled example in the standard reference setting; the paper commonly uses sharpening temperature 0.5.

**Hardware/parallelism considerations:** GPU batch processing is straightforward, but label guessing and mixed-batch learning increase image throughput requirements. Distributed shuffles must move labels with their corresponding images.

### 2.5.2 ReMixMatch

**Name:** ReMixMatch.

**Category & sub-category:** Semi-supervised learning; distribution-aligned, augmentation-anchored combination recipe.

**Originating paper/vendor/year:** David Berthelot and colleagues, [*ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring*, ICLR 2020; arXiv submission 2019](https://arxiv.org/html/1911.09785v2).

**Core mechanism:** Adjust an unlabeled prediction by a ratio of estimated target class frequency to running model-prediction frequency, then renormalize and sharpen it. This **distribution alignment** counters an aggregate class bias. Use a weakly augmented prediction as an anchor for several strongly augmented views rather than averaging potentially destructive strong views into the target. The recipe also retains interpolation and adds a rotation-prediction objective. Its online CTAugment policy adapts augmentation choices during training.

**Inputs/outputs and typical data types:** Labeled and unlabeled images, an estimated target class marginal, and weak/strong augmentation mechanisms. Outputs are a task classifier and auxiliary training state; an optional rotation head is not needed to make ordinary class predictions.

**Strengths and limitations:** Improves target quality and exploits stronger perturbations. Distribution alignment is useful only when its reference marginal is appropriate. A balanced seed set does not prove a balanced deployment population, and out-of-class images can be forced into known categories. More components create additional interactions to validate.

**Computational complexity / scalability notes:** Multiple strong views substantially increase backbone evaluations; the paper uses eight augmentations in its standard configuration. Running class statistics cost $`O(C)`$ storage, while mixed examples and the auxiliary rotation computation add batch-level work. This is not cost-equivalent to a one-strong-view recipe.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The CIFAR-10 study uses **250 labeled training images**. A weak view supplies a class guess, running class statistics adjust it, strong views train against the adjusted target, and the learned classifier assigns a natural-image category. Table 1 reports **6.27% error with standard deviation 0.34 percentage points**, compared with **11.08% and 0.87** for the MixMatch comparison reported in that table. The technical rationale is that an anchored weak prediction may be more trustworthy than a guess averaged over difficult augmentations. The paper cautions that some externally reported comparison methods use different implementations, so not every table row supports a controlled component-level claim. No production computer-vision or business KPI is reported. [Original Table 1 and comparison notes](https://arxiv.org/html/1911.09785v2).

**Notable vendor implementations/libraries:** Google's [ReMixMatch repository](https://github.com/google-research/remixmatch). CTAugment is part of that research recipe; its presence is not evidence that a vendor product uses the trained classifier.

**Architecture diagram description:** The reference backbone is **WRN-28-2** with approximately 1.5M parameters. `Weak image -> shared CNN -> distribution-aligned target; several strong images and mixed images -> shared CNN -> class losses`; a small rotation head branches from the representation.

**Activation functions used and why:** Leaky ReLU retains negative-side gradients in the reference residual CNN; batch normalization conditions its features. Softmax supplies normalized class and rotation probabilities. Temperature sharpening operates on targets rather than replacing the network's hidden nonlinearity.

**Loss function(s):** The recipe uses cross-entropy on mixed labeled and unlabeled targets, additional unmixed strong-view consistency, and an auxiliary rotation-classification cross-entropy. In particular, it changes MixMatch's unlabeled squared-error choice. Loss coefficients and the distribution-alignment normalization are part of a faithful reproduction.

**Optimization algorithm(s):** The paper uses Adam with fixed learning rate 0.002 and evaluates an EMA of parameters with decay 0.999. Its reported coefficient choices belong to this particular implementation; a weight-decay flag should not be interpreted without its update convention.

**Regularization techniques:** Distribution alignment, weak-to-strong anchoring, CTAugment, MixUp, target sharpening, weight decay, and rotation prediction. The reference sharpening temperature is 0.5 and its Beta interpolation parameter is 0.75.

**Backpropagation considerations:** Guessed targets and running marginal estimates are treated as target-building state. Gradients train the classifier and auxiliary head, not an argmax policy through discrete augmentation choices. A mistaken alignment ratio can distort all examples of a class.

**Parameter count / scaling behavior:** The principal network remains approximately **1.5M parameters**, plus a small auxiliary head and EMA state. Increased cost chiefly comes from extra views, not from a much larger classifier.

**Training paradigm:** Joint semi-supervised classification with a self-supervised rotation auxiliary. This auxiliary training signal does not make the whole recipe unsupervised.

**Hardware/parallelism considerations:** Multiple strong views increase activation memory and augmentation throughput. Class-marginal estimates should represent the intended global data distribution when batches are spread across devices.

### 2.5.3 FixMatch

**Name:** FixMatch.

**Category & sub-category:** Semi-supervised learning; confidence-filtered weak-to-strong pseudo-label consistency.

**Originating paper/vendor/year:** Kihyuk Sohn and colleagues, [*FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*, NeurIPS 2020](https://arxiv.org/pdf/2001.07685v2). The cited full-paper snapshot is **arXiv v2, dated 2020-11-25**, not a claim about a current model release.

**Core mechanism:** Predict on a weakly augmented unlabeled image. If the largest class probability exceeds a threshold, use its argmax as a hard target for a strongly augmented version of the same image. Low-confidence examples contribute zero unlabeled loss at that step. Unlike MixMatch, the basic recipe does not require soft-target averaging and MixUp; unlike Mean Teacher, it does not require a separately averaged teacher to generate targets.

**Inputs/outputs and typical data types:** Labeled images, an unlabeled image pool, weak and strong augmentation, and a confidence threshold. Output is a conventional inductive classifier. The target classes and augmentation semantics must match the unlabeled population.

**Strengths and limitations:** Compact, effective, and comparatively easy to ablate. It can initially use very few unlabeled examples and then admit more as confidence grows. A fixed threshold can exclude hard classes, include confidently wrong OOD images, or make performance depend heavily on the choice of a few labeled seeds.

**Computational complexity / scalability notes:** With labeled batch size $`B`$ and unlabeled ratio $`\mu`$, each step processes $`B`$ labeled views and approximately $`2\mu B`$ weak/strong unlabeled views. Target confidence computation is cheap relative to the CNN, but the extra image processing is not free.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On CIFAR-10 with **250 labels**, Table 2 of the original paper reports **5.07% error with standard deviation 0.65 percentage points for FixMatch (RA)** over five folds, equivalent to **94.93% accuracy**. This is the RandAugment variant; the separately reported CTAugment variant has a different standard deviation. A weakly transformed image produces a candidate category; only a sufficiently confident guess supervises its strongly transformed counterpart; a trained WRN-28-2 supplies the final object-category decision. This fits a setting where augmentations are reasonably label-preserving and a small labeled seed can initialize useful predictions. The number is a research benchmark result, not a production success rate or a controlled comparison with later TorchSSL reproductions. No business KPI is reported. [Full paper v2, Table 2 and Section 4.1](https://arxiv.org/pdf/2001.07685v2).

**Notable vendor implementations/libraries:** Google's [FixMatch repository](https://github.com/google-research/fixmatch), including the [training recipe](https://github.com/google-research/fixmatch/blob/master/fixmatch.py). Framework ports can change normalization, augmentation, and checkpoint averaging.

**Architecture diagram description:** The example uses **WRN-28-2**. `Weak unlabeled view -> CNN -> detached argmax and confidence mask; strong view -> same CNN -> masked cross-entropy`. A separate labeled branch supplies supervised learning.

**Activation functions used and why:** Leaky ReLU with slope 0.1 retains negative-side gradients in the cited SSL residual implementation; batch normalization conditions features, and softmax supplies class probabilities for confidence filtering. A hard argmax is a target-selection operation, not a differentiable hidden activation.

**Loss function(s):** Labeled cross-entropy plus

$$
\frac{\lambda_u}{\mu B}\sum_b
\mathbf1[\max q_b\ge\tau]\,
\operatorname{CE}(\arg\max q_b,p_\theta(\cdot\mid a_s(u_b))).
$$

The denominator is the whole unlabeled batch, not merely the accepted subset. Changing it changes the effective early-training weight.

**Optimization algorithm(s):** The reference uses SGD with Nesterov momentum 0.9, initial learning rate 0.03, and $`\eta_s=\eta_0\cos(7\pi s/(16S))`$ over the configured training budget $`S`$. The standard small-image configuration runs $`2^{20}`$ updates.

**Regularization techniques:** Weak augmentation, strong RandAugment or CTAugment-based policies as separately configured, weight decay, and confidence masking. Typical reference values are $`\tau=0.95`$, $`\mu=7`$, and unlabeled weight 1. They are settings, not universally optimal constants.

**Backpropagation considerations:** Stop gradients through pseudo-labels and their confidence mask. Maintain correspondence between weak and strong views. Normalization statistics can couple examples, so changing batch interleaving may change results even with the same written loss.

**Parameter count / scaling behavior:** Approximately **1.5M learned parameters** in WRN-28-2. The threshold adds no learned parameters; an optional EMA for evaluation adds state but is not a Mean Teacher target generator.

**Training paradigm:** Joint supervised and online pseudo-labeled training. No separate generative model, graph solve, or mandatory pretraining is required.

**Hardware/parallelism considerations:** GPU/TPU data parallelism is natural. Strong-augmentation throughput, the unlabeled batch ratio, and normalization synchronization are practical bottlenecks; changing GPU count can change effective training if those are not controlled.

### 2.5.4 FlexMatch

**Name:** FlexMatch, applying Curriculum Pseudo Labeling (CPL) to FixMatch.

**Category & sub-category:** Semi-supervised learning; class-adaptive confidence thresholds.

**Originating paper/vendor/year:** Bowen Zhang and colleagues, [*FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling*, NeurIPS 2021](https://arxiv.org/html/2110.08263v3).

**Core mechanism:** Estimate a class's learning progress from the number of unlabeled examples confidently assigned to it above a base threshold. Normalize these counts to obtain relative progress $`\beta_t(c)`$, then lower the admission threshold for classes with lower estimated progress. CPL adds threshold warm-up and can apply a nonlinear mapping to progress. The estimate is not measured per-class accuracy; interpreting it that way requires assumptions about class balance and confidence quality.

**Inputs/outputs and typical data types:** FixMatch-style labeled/unlabeled images, class statistics, and persistent state recording sufficiently confident sample assignments. Output remains a normal classifier; class thresholds and selection state are training auxiliaries.

**Strengths and limitations:** Admits useful samples from classes that a single fixed threshold might neglect. The source does not need repeated validation-set inference to estimate progress. However, few confident predictions can mean rarity, ambiguity, distribution shift, or poor learning; counts do not distinguish these explanations. The original experiments do not show uniform improvement on every dataset.

**Computational complexity / scalability notes:** Reuses predictions already needed for weak-to-strong training, without another backbone forward or backward pass. Cached per-example assignments require approximately $`O(n_u)`$ auxiliary storage, plus class counts. Efficient updates avoid rescoring the entire unlabeled pool at each step.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In the paper's CIFAR-10 experiment with **40 labels, four per class**, WRN-28-2 predicts weak-view classes, CPL lowers thresholds for underrepresented confident assignments, and accepted labels train strong views. The final decision is an object category. Table 1 reports **4.97% error with standard deviation 0.06 percentage points**, versus **7.47% and 0.28** for the same study's FixMatch implementation. These are **best-checkpoint results over three runs**; the paper separately supplies last-checkpoint-window statistics. They are not directly interchangeable with FixMatch's original-paper score or a deployment estimate selected without test feedback. The technical fit is uneven class-learning progress under scarce labels; no business KPI is reported. [Methods, evaluation rule, and Table 1](https://arxiv.org/html/2110.08263v3).

**Notable vendor implementations/libraries:** The authors' [TorchSSL framework](https://github.com/TorchSSL/TorchSSL) supplies the comparison setting. Its unified implementations are research baselines, not proof of a vendor's production method selection.

**Architecture diagram description:** The cited CIFAR-10 backbone is **WRN-28-2**, with the same weak/strong/labeled branches as FixMatch. `Weak-view probabilities -> cached confidence counts -> class threshold -> strong-view loss mask` is the additional control path.

**Activation functions used and why:** TorchSSL's reference WRN uses leaky ReLU with slope 0.1 for nonzero negative-side gradients, batch normalization, and softmax for the probabilities used by class-progress estimation. Threshold adaptation changes example selection, not the residual units' activations.

**Loss function(s):** Supervised cross-entropy plus masked hard pseudo-label cross-entropy. Replace the fixed threshold in FixMatch with $`\tau_t(c)`$, based on normalized progress; the threshold used for admission is distinct from the base threshold used to estimate progress.

**Optimization algorithm(s):** The source uses SGD with momentum 0.9, initial learning rate 0.03, the cosine schedule $`\eta_0\cos(7\pi s/(16S))`$, and $`S=2^{20}`$ updates. It evaluates an EMA model with decay 0.999. These are the comparison's backbone-training settings, not learned curriculum parameters.

**Regularization techniques:** RandAugment, weight decay, weak-to-strong consistency, threshold warm-up, and class-dependent selection. The usual base threshold is 0.95. Thresholds need not increase monotonically if assignments change.

**Backpropagation considerations:** Count updates and admission decisions are not differentiated. Gradients pass through the classifier's selected losses. Confusing the confident-assignment counter with all current pseudo-labels changes the curriculum.

**Parameter count / scaling behavior:** Approximately **1.5M learned parameters** for WRN-28-2; CPL adds no learned layers. Its auxiliary memory is not identical to FreeMatch's compact running class statistics.

**Training paradigm:** Online semi-supervised learning with a confidence-based curriculum, rather than a separate curriculum trained using held-out class accuracies.

**Hardware/parallelism considerations:** Similar accelerator work to FixMatch, plus bookkeeping. Distributed implementations must keep sample identities and class statistics consistent; an independently evolving curriculum on each worker is a different procedure.

### 2.5.5 FreeMatch

**Name:** FreeMatch.

**Category & sub-category:** Semi-supervised learning; self-adaptive global/local thresholds and marginal-diversity regularization.

**Originating paper/vendor/year:** Yidong Wang and colleagues, [*FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning*, ICLR 2023; arXiv submission 2022](https://arxiv.org/html/2205.07246v3).

**Core mechanism:** Estimate global model confidence by an EMA of the largest predicted probability in each unlabeled example. Separately average each class's predicted probability. The class threshold is

$$
\tau_t(c)=\tau_t^{\rm global}
\frac{\bar p_t(c)}{\max_j\bar p_t(j)}.
$$

Both statistics start from uniform-class values and evolve during training. A self-adaptive fairness term encourages diversity in aggregate predictions using probability and hard-label histograms. It is not the same as asserting that every deployment population has a uniform class prior.

**Inputs/outputs and typical data types:** Labeled/unlabeled images, weak and strong views, and running confidence, probability, and class-histogram statistics. Outputs are a classifier and adaptive thresholds used during training.

**Strengths and limitations:** Removes the need to fix one admission-confidence value across the entire training trajectory and can help at extremely low label counts. It does not remove all hyperparameters: EMA rates, objective weights, augmentations, and optimizer settings remain. Biased confidence, missing classes, and OOD inputs can corrupt both the thresholds and the fairness statistics. "Class fairness" here is not a guarantee of demographic fairness.

**Computational complexity / scalability notes:** Backbone work is comparable to weak/strong-view pseudo-labeling. Running statistics require $`O(C)`$ state, plus ordinary batch probabilities, rather than a historical target for every image. Computing and communicating class marginals is generally small compared with CNN training.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The paper studies CIFAR-10 with **ten labels, one per class**. Weak-view predictions determine an evolving confidence threshold and class corrections; accepted strong-view predictions train the classifier, while the fairness term discourages degenerate aggregate class use; inference assigns one object category. Table 1 reports **8.07% error with standard deviation 4.24 percentage points**, compared with **13.85% and 12.04** for its FlexMatch comparison. The large variability matters. These results use three random seeds and the paper's **best-error-over-checkpoints reporting rule**, not a test-locked deployment protocol. The technical fit is an initially unreliable fixed confidence level; the result is not evidence that one label per class reliably suffices in a new domain. No business KPI is reported. [Setup and Table 1](https://arxiv.org/html/2205.07246v3).

**Notable vendor implementations/libraries:** The paper links the [TorchSSL ecosystem](https://github.com/TorchSSL/TorchSSL); [Microsoft's Semi-supervised-learning/SemiLearn repository](https://github.com/microsoft/Semi-supervised-learning) includes modern recipe implementations. Repository ownership does not identify a deployed product using the method.

**Architecture diagram description:** The CIFAR-10 example uses **WRN-28-2**: `weak view -> CNN -> running confidence and class statistics -> adaptive mask; strong view -> shared CNN -> class and marginal-diversity losses`.

**Activation functions used and why:** Leaky ReLU retains negative-side feature gradients and batch normalization conditions the reference WRN. Softmax produces normalized predictions, which also make its running class-marginal estimates meaningful.

**Loss function(s):** $`\mathcal L_s+w_u\mathcal L_u+w_f\mathcal L_f`$, where $`\mathcal L_u`$ is adaptively masked pseudo-label cross-entropy. The paper's Eq. 11 defines $`\mathcal L_f=-\operatorname{CE}(a,b)`$: $`a`$ and $`b`$ are normalized running and accepted-batch probability marginals, each divided by its corresponding hard-prediction histogram. This is the source's histogram-corrected marginal-entropy/diversity proxy, not ordinary positive cross-entropy to a fixed uniform target or per-example entropy minimization.

**Optimization algorithm(s):** The source comparison uses SGD with momentum 0.9, initial learning rate 0.03, $`\eta_0\cos(7\pi s/(16S))`$, and $`S=2^{20}`$ updates. EMA model evaluation uses decay 0.999. Adaptation-statistic EMA and model-weight EMA are distinct mechanisms.

**Regularization techniques:** Strong augmentation, weight decay, adaptive pseudo-label selection, and the self-adaptive fairness penalty. The fairness weight and statistic smoothing remain choices to validate, despite the method's name.

**Backpropagation considerations:** Treat running estimates and hard selection as auxiliary state, while differentiating the classifier's current probability losses. Histogram divisions need safeguards for empty classes; skipping that detail can produce undefined values precisely in scarce-label cases.

**Parameter count / scaling behavior:** Approximately **1.5M learned parameters** in WRN-28-2, with only class-sized adaptation state beyond normal training buffers. No new expert network or learned threshold-prediction head is required.

**Training paradigm:** End-to-end semi-supervised classification with continuously estimated confidence and class statistics; no separate validation-driven curriculum.

**Hardware/parallelism considerations:** Similar image-processing load to FixMatch, with inexpensive class-statistic reductions. If different devices see different class mixtures, synchronize the intended statistics rather than silently producing worker-specific thresholds.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| MixMatch | Images where interpolation and weak augmentation are useful | Soft label guessing plus interpolation regularization | Several coupled target-construction choices | CIFAR-10 with 250 labels |
| ReMixMatch | Images with reliable weak anchors and strong augmentations | Distribution alignment and richer consistency | Prior mismatch and many augmented views | CIFAR-10 with 250 labels |
| FixMatch | Images with trustworthy weak/strong view semantics | Simple confidence-filtered consistency | Fixed threshold excludes some useful classes | CIFAR-10 with 250 labels |
| FlexMatch | Closed-set images with uneven class learning progress | Class-specific admission curriculum | Confidence counts confound difficulty and rarity | CIFAR-10 with 40 labels |
| FreeMatch | Very sparsely labeled closed-set image tasks | Adapts overall and class-specific confidence levels | Sensitive running statistics and high small-label variance | CIFAR-10 with ten labels |

## 2.6 Generative and reconstruction approaches

These approaches obtain additional learning signals from explaining or reconstructing the inputs. Modeling input structure can help classification, but likelihood, reconstruction quality, image realism, and classification accuracy are different objectives. None should be used as an unqualified proxy for the others.

### 2.6.1 Semi-supervised variational autoencoders: M1 and M2

**Name:** Semi-supervised deep generative models M1, M2, and the explicitly stacked M1+M2 system.

**Category & sub-category:** Semi-supervised learning; variational generative modeling and latent-feature classification.

**Originating paper/vendor/year:** Diederik Kingma, Shakir Mohamed, Danilo Rezende, and Max Welling, [*Semi-supervised Learning with Deep Generative Models*, NIPS 2014](https://arxiv.org/html/1406.5298v2).

**Core mechanism:** **M1** learns a VAE representation from all inputs, then trains a classifier on latent features; the paper evaluates a TSVM in that role. **M2** introduces a class variable: $`p_\theta(x,y,z)=p(y)p(z)p_\theta(x\mid y,z)`$, with inference networks $`q_\phi(y\mid x)`$ and $`q_\phi(z\mid x,y)`$. Known labels condition the labeled objective; unknown labels are summed over, weighted by the inferred class posterior. **M1+M2** trains the latter model on the former's learned representation. These are related but distinct systems, and their benchmark scores differ substantially.

**Inputs/outputs and typical data types:** Labeled and unlabeled vectors or images. Outputs include latent features, a class posterior, and a conditional generative model. M1 alone has no classification labels in its VAE objective; its downstream classifier makes the complete workflow supervised or semi-supervised.

**Strengths and limitations:** Gives an explicit probabilistic connection between classes and input structure, potentially separating category from style. However, a good input-density model need not emphasize discriminative features. Inference approximations, likelihood choice, and latent collapse can limit performance, and summing over many classes becomes expensive.

**Computational complexity / scalability notes:** For fixed Monte Carlo samples, a VAE minibatch costs encoder/decoder forward and backward passes. M2's exact unlabeled-class sum scales approximately as $`O(BCF)`$, where $`F`$ now denotes one class-conditioned inference/generation evaluation, plus classifier cost. This class factor is not present in an ordinary single-head classifier.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In permutation-invariant MNIST recognition with **100 labeled training examples**, images first produce M1 latent features; M2 uses those features to infer digit class and style; the class posterior supplies the digit decision. Table 1 reports **3.33% error with standard deviation 0.14 percentage points for M1+M2**, whereas **M2 alone reports 11.97% with standard deviation 1.71**. The headline combined-model result must not be assigned to M2 alone. The technical fit versus direct classification is that unlabeled images can teach reusable variation before scarce labels identify classes. This is a research comparison, not proof of a deployed document-processing system, and no business KPI is reported. [Original model definitions and MNIST table](https://arxiv.org/html/1406.5298v2).

**Notable vendor implementations/libraries:** The authors' [NIPS 2014 semi-supervised code](https://github.com/dpkingma/nips14-ssl). Modern probabilistic-programming or tensor libraries can express these objectives, but a generic VAE example does not automatically implement M2's class marginalization and additional classification term.

**Architecture diagram description:** The source's M1 uses `input -> two 600-unit hidden layers -> mean/log-variance of 50-dimensional z`, with decoder `z -> 600 -> 600 -> input likelihood parameters`. M2 uses 50-dimensional $`z`$ and one 500-unit hidden layer in its component MLPs: `x -> q(y|x)`; `(x,y) -> q(z|x,y)`; `(y,z) -> p(x|y,z)`. These are dense networks, not an unspecified convolutional backbone.

**Activation functions used and why:** Softplus provides smooth hidden nonlinearities in the reported MLPs. Softmax normalizes the class posterior, sigmoid constrains MNIST Bernoulli likelihood parameters to valid probabilities, and Gaussian inference uses separate mean/variance parameterizations. Positive variance must be enforced; it is not a categorical activation.

**Loss function(s):** M1 minimizes the ordinary negative VAE bound: expected negative reconstruction log-likelihood plus latent KL, before fitting its downstream classifier. For M2, let $`\operatorname{ELBO}_l(x,y)`$ be the labeled variational bound. The unlabeled bound is

$$
\operatorname{ELBO}_u(x)=
\sum_y q_\phi(y\mid x)\operatorname{ELBO}_l(x,y)
+H(q_\phi(y\mid x)).
$$

Minimize negative labeled/unlabeled bounds plus a weighted labeled cross-entropy for $`q_\phi(y\mid x)`$. The **positive entropy in this ELBO** must not be confused with adding positive entropy to a minimized entropy-minimization loss.

**Optimization algorithm(s):** Stochastic variational gradient updates. The cited paper contains an optimizer inconsistency: Section 3.2 says experimental results used AdaGrad, while its detailed implementation discussion describes a momentum/bias-corrected RMSProp variant with **constant learning rate 0.0003**. This volume preserves that source discrepancy rather than inventing a uniquely specified Adam recipe.

**Regularization techniques:** Latent KL penalties, priors, stochastic latent sampling, and the reported M2 parameter prior/weight penalty. The MNIST procedure also samples binary inputs from normalized pixel intensities. Batch normalization and dropout are not assumed merely because later VAE implementations use them.

**Backpropagation considerations:** Reparameterize $`z=\mu+\sigma\odot\epsilon`$ to obtain low-variance pathwise gradients. For M2's small class vocabulary, enumerate labels and differentiate the weighted sum rather than sampling an unobserved class with an unnecessary score-function estimator.

**Parameter count / scaling behavior:** No single count applies to M1, raw-input M2, and stacked M1+M2. A dense layer from $`a`$ inputs to $`b`$ outputs contributes $`(a+1)b`$ parameters, and a diagonal Gaussian head has separate mean and variance outputs. Stacking changes M2's input dimension, so adding two raw-input model counts would be misleading.

**Training paradigm:** M1 representation pretraining followed by a classifier, joint generative/discriminative M2 training, or the explicitly staged M1+M2 combination. A downstream TSVM also changes the transductive/inductive evaluation question.

**Hardware/parallelism considerations:** MLP training is accelerator-friendly. Enumerating class-conditioned branches can be vectorized, but memory and compute grow with class count; large-label-vocabulary applications require additional approximations not part of this entry.

### 2.6.2 Semi-supervised GAN with a K+1 classifier

**Name:** Semi-supervised GAN using $`K+1`$ categories: $`K=C`$ real classes plus a generated/fake category.

**Category & sub-category:** Semi-supervised learning; generative adversarial feature learning with a classification discriminator.

**Originating paper/vendor/year:** The representative formulation is Tim Salimans and colleagues' [*Improved Techniques for Training GANs*, NeurIPS 2016](https://arxiv.org/html/1606.03498v1). A related independent proposal is Augustus Odena's [*Semi-Supervised Learning with Generative Adversarial Networks*, 2016](https://arxiv.org/abs/1606.01583).

**Core mechanism:** Train the discriminator to classify labeled real examples, recognize unlabeled observations as belonging to some real class, and recognize generated examples as fake. The generator supplies additional structure around the real-data distribution. In the successful reference SSL setup, **feature matching** trains the generator to match mean intermediate discriminator features of real data. Photorealistic generation and good semi-supervised classification are not identical goals; the paper explicitly finds that a technique improving sample appearance need not give the best classifier.

**Inputs/outputs and typical data types:** Labeled real images, unlabeled real images, and random generator inputs. Outputs are real-class predictions and generated samples. The fake category is trained against this generator's outputs; it is not a generally validated unknown-class or OOD detector.

**Strengths and limitations:** Can learn useful features without hand-specifying every invariance and offers a generative output in addition to classification. Alternating optimization is harder to stabilize than a single classification objective. Mode collapse, discriminator dominance, and generator artifacts can affect the learning signal.

**Computational complexity / scalability notes:** Training requires real and generated discriminator passes, generator passes, and separate gradient updates. Per-step cost depends on both networks and the update ratio. The generator can be discarded for class inference; classification cost is therefore not the full training-system cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** For permutation-invariant MNIST recognition with **100 labeled examples**, the discriminator receives digit vectors while a generator supplies synthetic vectors. Real/fake discrimination shapes its features, labeled cross-entropy names the digit classes, and the real-class argmax supplies a recognition decision. The original Table 1 reports **93 incorrectly classified test images on average, standard deviation 6.5, out of the 10,000-image test set**, averaged over ten seeds. This is the single-model row, not the distinct ten-model ensemble row. Feature matching fits the goal of learning class-useful representations even when generated samples are not maximally realistic. No production OCR performance or business KPI is reported. [MNIST experiment and feature-matching discussion](https://arxiv.org/html/1606.03498v1).

**Notable vendor implementations/libraries:** OpenAI's historical [improved-gan repository](https://github.com/openai/improved-gan). The detailed neural fields here describe its public [MNIST feature-matching script](https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_mnist_feature_matching.py), not an undocumented commercial checkpoint.

**Architecture diagram description:** The inspected reference code uses discriminator `784 -> 1000 -> 500 -> 250 -> 250 -> 250 -> 10 real logits`; the fake logit is implicitly fixed to zero, giving an equivalent $`K+1`$ distribution. Its generator is `100-dimensional noise -> 500 -> 500 -> 784`. These are concrete script settings, not a claim that every experiment in the paper used this exact generator.

**Activation functions used and why:** ReLU supplies piecewise-linear discriminator features; softplus supplies smooth generator nonlinearities, and sigmoid bounds image-output intensities. The real-class decision uses normalized logits; the implicit fake category can be computed through a log-sum-exp normalization.

**Loss function(s):** Labeled cross-entropy over real classes, real-versus-fake terms on real and generated observations, and generator feature matching $`\lVert\mathbb E_x h(x)-\mathbb E_z h(G(z))\rVert_2^2`$. A generic minimax GAN generator loss is not a substitute for the feature-matching configuration associated with this example.

**Optimization algorithm(s):** The public MNIST script alternates Adam-family updates, uses learning rate **0.003** without an explicit decay schedule in that script, and sets first-moment coefficient 0.5. This reports the inspected implementation, not a universal GAN optimizer prescription.

**Regularization techniques:** Weight normalization and Gaussian noise in the discriminator, normalization in the generator, and feature matching. Do not assume that every technique discussed in *Improved Techniques*, such as minibatch discrimination, was simultaneously used for its best SSL result.

**Backpropagation considerations:** For a discriminator update, generated inputs should not update generator parameters. For a generator update, gradients pass through discriminator features to generator outputs while discriminator parameters remain fixed. Detach the real feature target as appropriate to the alternating objective.

**Parameter count / scaling behavior:** Training stores $`p_D+p_G`$; classification uses $`p_D`$. The displayed discriminator has approximately 1.54M affine parameters by arithmetic from its layer sizes, with normalization state in addition. The classifier and generator may scale independently.

**Training paradigm:** Joint labeled classification, unlabeled real/fake learning, and generator training. This is not an unsupervised GAN merely evaluated with an external classifier afterward.

**Hardware/parallelism considerations:** A GPU supports the reference MLP setup; image-scale CNN variants need more memory and throughput. Distributed feature-mean estimates and the discriminator/generator update ratio must be controlled for comparable training.

### 2.6.3 Ladder Networks

**Name:** Semi-supervised Ladder Network.

**Category & sub-category:** Semi-supervised learning; joint classification and layerwise denoising reconstruction.

**Originating paper/vendor/year:** Antti Rasmus and colleagues, [*Semi-Supervised Learning with Ladder Networks*, NIPS 2015](https://arxiv.org/html/1507.02672v2), extending the earlier Ladder architecture associated with Harri Valpola.

**Core mechanism:** Run clean and corrupted encoders with shared weights. A top-down decoder receives lateral connections from corrupted representations at each layer and reconstructs the corresponding clean representations. Labeled examples contribute classification loss at the noisy encoder's output; all examples contribute denoising losses. Lateral information allows each layer to reconstruct local detail without forcing the topmost representation to preserve everything about the input.

**Inputs/outputs and typical data types:** Labeled and unlabeled feature vectors or images. During training, outputs include class probabilities and layerwise reconstructions; ordinary classification uses only the clean encoder. This is not a latent-variable likelihood model in the same sense as M2.

**Strengths and limitations:** Provides learning signals throughout the network when label gradients are scarce, without requiring separate layerwise pretraining. Its decoder, normalization, corruption levels, and per-layer cost weights make implementation more delicate than a simple classification wrapper. Reconstruction can preserve irrelevant variation unless the supervised objective shapes the representation appropriately.

**Computational complexity / scalability notes:** Training includes clean and corrupted encoder computation plus the denoising decoder. For fixed architecture and corruption, cost remains linear in examples per epoch with a larger constant. Storing activations for several paths can dominate memory; classifier inference discards the decoder.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In permutation-invariant MNIST recognition, the fully connected Ladder model uses **100 labels in the supervised training objective** and reports **1.06% test error with standard deviation 0.37 percentage points**. Pixel vectors enter clean and noisy encoders, the decoder learns to denoise intermediate representations, and the clean encoder's final class maximum assigns a digit. Crucially, the authors used **10,000 labeled validation images for model and hyperparameter development**; final runs used all 60,000 training images with training labels restricted as specified. They increased the 100-label evaluation to forty runs because occasional failures affected the average. The technical fit is obtaining lower-layer learning signals from unlabeled handwriting, not merely improving a pixel reconstruction score. No deployed OCR system or business KPI is reported. [Section 4.1 and Table 1](https://arxiv.org/html/1507.02672v2).

**Notable vendor implementations/libraries:** The authors' [Ladder research implementation](https://github.com/CuriousAI/ladder). A stacked denoising autoencoder without lateral/top-down denoising at each level is not the same architecture.

**Architecture diagram description:** The reference encoder is `784 -> 1000 -> 500 -> 250 -> 250 -> 250 -> 10`. Clean and noise-corrupted copies share weights. At each level, `top-down decoder signal + lateral corrupted activation -> denoising function -> reconstructed clean activation`; training compares these paired representations.

**Activation functions used and why:** ReLU supplies nonlinear encoder features, and softmax supplies a normalized class distribution for cross-entropy. The denoising functions use sigmoid and affine components to condition estimated mean and scale on top-down signals. Their role is reconstruction, not another categorical output at every layer.

**Loss function(s):** Noisy-encoder supervised cross-entropy plus weighted mean-squared denoising errors across layers, with the source's normalization of clean targets and reconstructed quantities. Cost multipliers differ by layer; replacing the normalized reconstruction objective with an arbitrary raw-activation MSE changes the method.

**Optimization algorithm(s):** Adam with learning rate **0.002 for 100 epochs**, followed by **50 epochs of linear decay to zero** in the MNIST experiments. Minibatch size is 100. The model is trained jointly, not by a mandatory sequence of isolated autoencoder fits.

**Regularization techniques:** Gaussian corruption, batch normalization, and layerwise denoising. For the reported 100-label configuration, the paper gives noise standard deviation 0.3 and cost weights 1000 at the input level, 10 at the first hidden level, and 0.1 at higher levels.

**Backpropagation considerations:** Shared encoder weights receive supervised and denoising learning signals. Clean/noisy alignment and consistent variance normalization are essential; incorrect normalization can create trivial scaling solutions or unstable gradients. Lateral and layerwise losses provide shorter learning paths than relying only on the final classifier.

**Parameter count / scaling behavior:** The listed encoder has approximately **1.54M affine parameters**, calculated from its widths, plus normalization parameters. Training adds decoder matrices and per-unit denoising parameters; only the encoder is needed for normal inference.

**Training paradigm:** Joint semi-supervised classification and self-supervised denoising from random initialization. The reconstruction component does not remove the labeled classification signal from the overall method.

**Hardware/parallelism considerations:** GPU training benefits from batching all paths, but retains more intermediate state than a classifier alone. Distributed normalization and per-layer losses should use consistent batch definitions; inference is substantially simpler than the training graph.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Semi-supervised VAE M1 / M2 | Vectors or images with useful latent generative structure | Explicit class-conditioned probabilistic learning | Likelihood mismatch, inference approximation, and class-sum cost | MNIST M1+M2 with 100 training labels |
| K+1 semi-supervised GAN | Images where generated examples can aid feature learning | Couples classification with generative feature constraints | Unstable alternating optimization and misleading realism proxies | MNIST feature-matching GAN with 100 labels |
| Ladder Network | Images or vectors benefiting from hierarchical denoising | Unlabeled learning signals at multiple layers | Decoder and normalization complexity | MNIST with 100 training labels and a separate large validation budget |

## Evidence and reproduction boundaries

All 22 worked examples in this volume are **research benchmarks**, not verified production deployments. Their datasets address concrete recognition, document-classification, or scientific classification tasks, but no unreported business outcome has been supplied. "No business KPI reported" does not mean an algorithm has never been deployed; it means the cited evidence does not establish such an outcome.

Exact metrics above were checked against the linked primary papers or their original-paper copies. Qualitative graph-based results are described without inventing point estimates from plots. Parameter counts explicitly identified as arithmetic are calculations from stated layer sizes, not claims that a vendor published a checkpoint with exactly that count. These checks do not constitute independent retraining of the experiments.

Reproduction needs particular care with historical code: branches can change, older framework dependencies may be unavailable, and a public script need not identify every setting behind a published result. Record a commit and complete configuration before claiming numerical reproduction. The M1/M2 paper's conflicting optimizer descriptions, the GAN script-versus-paper architectural distinction, validation-label costs, private JFT data, and best-checkpoint reporting are disclosed rather than filled in with guessed details.

## Coverage and continuation manifest

- **2.1.1-2.1.4:** Self-training/Pseudo-Label, co-training, tri-training, and Noisy Student.
- **2.2.1-2.2.2:** Transductive/semi-supervised SVM and Laplacian SVM/manifold regularization.
- **2.3.1-2.3.2:** Hard-clamped harmonic label propagation and normalized, soft-clamped label spreading.
- **2.4.1-2.4.6:** Entropy minimization, Pi Model, temporal ensembling, Mean Teacher, VAT, and UDA.
- **2.5.1-2.5.5:** MixMatch, ReMixMatch, FixMatch, FlexMatch, and FreeMatch.
- **2.6.1-2.6.3:** Semi-supervised VAE M1/M2, K+1 semi-supervised GAN, and Ladder Networks.
- **Total:** 22 scoped entries, six category comparison tables. Neural deep dives describe concrete reference instantiations, not universal architectures for the wrapper names.
- **Related volumes:** [Supervised classical methods](01-supervised-classical.md), [supervised neural architectures](02-supervised-neural.md), [unsupervised classical methods](04-unsupervised-classical.md), [unsupervised neural and self-supervised learning](05-unsupervised-neural.md), [foundation models and their training stages](06-foundation-models.md), [cross-cutting comparisons](09-comparative-guide.md), and [glossary](10-glossary.md).
- **Further non-required depth not included:** semi-supervised regression and structured prediction; formal generalization and graph-convergence proofs; missing-view and missing-not-at-random label theory; open-set and long-tailed evaluation suites beyond the failure modes discussed here; federated SSL; continuous-stream cache/teacher management; active annotation policies; and full augmentation-policy search algorithms. Trainable graph representation architectures are distinct from the two fixed-graph inference entries and should be read alongside the related neural volumes.
