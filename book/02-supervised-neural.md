# 1. Supervised Learning Algorithms: Neural Architectures

This chapter continues [supervised learning](01-supervised-classical.md) with neural architectures, from a single trainable decision boundary to distributed mixtures of translation experts. Its evidence policy is dated **2026-09-08**. Historical papers and explicitly identified implementations are reference points, not assertions about the latest available model.

An architecture is not a supervision category. Here the representative training signal is an externally supplied class, numerical target, annotation, translation, or same/different pair. A recurrent network can also learn from unlabelled text; a graph network can propagate through unlabelled nodes; a vision Transformer can learn by reconstructing masked patches. Those uses belong with the corresponding training objectives in [semi-supervised learning](03-semi-supervised.md), [unsupervised neural learning](05-unsupervised-neural.md), and [foundation-model pretraining](06-foundation-models.md).

The worked examples distinguish research and documentation benchmarks from deployed applications. A published accuracy is not a business outcome. Results from different training sets, checkpoint recipes, crops, ensembles, or evaluation scripts are not automatically comparable. Benchmark numbers are attributed to the cited sources; models were not retrained for this chapter. Library availability establishes implementability, not use inside a vendor's commercial products.

Unless redefined locally, $`n`$ is the number of training examples, $`B`$ the batch size, $`d`$ a feature or hidden width, $`p`$ trainable parameters, $`T`$ sequence length, $`L`$ layers, and $`E`$ training epochs. Image height and width are $`H,W`$; $`q`$ denotes a convolutional kernel side. Graph entries use $`V`$ vertices and $`m`$ edges, avoiding confusion between edges and training epochs. Arithmetic counts describe a stated pass, not elapsed time or energy.

## 1.5 Neural foundations

The perceptron learns a boundary in the supplied representation. An MLP also learns intermediate representations. That distinction explains both the extra expressive power and the more difficult optimization of multilayer networks.

### 1.5.1 Perceptron

**Name:** Perceptron; the representative implementation is a linear, supervised classifier, extended to multiple classes by one-versus-rest classifiers.

**Category & sub-category:** Supervised learning; neural foundations; single-layer threshold networks and online linear classification.

**Originating paper/vendor/year:** Frank Rosenblatt's [1958 perceptron paper](https://doi.org/10.1037/h0042519) is the canonical origin. The modern numerical implementation below is documented by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html), not a reconstruction of the historical hardware.

**Core mechanism:** A binary unit computes $`s=w^\top x+b`$ and predicts its sign. For labels $`y\in\{-1,+1\}`$, a mistake or nonpositive signed margin triggers $`w\leftarrow w+\eta yx`$, $`b\leftarrow b+\eta y`$. Correctly classified examples normally cause no update. This is error correction, not probability estimation. Under bounded inputs and a positive separating margin, the classical mistake bound is proportional to the squared radius-to-margin ratio. Without separability, cycling and persistent mistakes are possible.

**Inputs/outputs and typical data types:** Numeric vectors, including sparse text features or flattened image pixels, become class labels and decision scores. A score is not a calibrated probability; multiclass scores come from separate binary problems.

**Architecture diagram description:**

```text
feature vector x -> weighted sum w.x + b -> threshold -> binary label
                          |
                    labelled error -> weight update
multiclass: parallel one-versus-rest units -> largest score
```

**Activation functions used and why:** A hard threshold implements the decision. Its discontinuity is intentional: the model is defined by a separating hyperplane rather than a smooth probability surface.

**Loss function(s):** The perceptron criterion can be written $`\max(0,-ys)`$, with a chosen update convention at zero margin. Unlike the SVM hinge loss, it does not demand a unit margin.

**Optimization algorithm(s):** The classical algorithm uses a fixed positive step size. Scikit-learn's wrapper fixes `loss="perceptron"` and a constant learning-rate schedule; the documented default multiplier is `eta0=1`. Epoch limits and tolerance stopping are implementation controls, not a proof of convergence on noisy data.

**Regularization techniques:** None is required by the original rule. The library optionally supports L1, L2, or elastic-net penalties. Feature scaling, validation-based stopping, and limiting passes are practical controls, but change the fitted result.

**Backpropagation considerations:** There is no hidden-layer backpropagation and no useful ordinary derivative through the hard decision. The mistake update directly adjusts the linear weights.

**Parameter count / scaling behavior:** A binary model has $`d+1`$ parameters. A $`c`$-class one-versus-rest implementation has approximately $`c(d+1)`$, independent of training-set size.

**Training paradigm:** Explicit class labels supervise online or repeated-pass learning. Learning a representation without labels is not part of this formulation.

**Hardware/parallelism considerations:** CPUs usually suffice. Sparse updates exploit nonzero features; independent one-versus-rest models can run in parallel. Sequential online updates are not identical to arbitrary parallel batch updates.

**Strengths and limitations:** It is an inexpensive diagnostic baseline with inspectable coefficients. It cannot solve XOR in its original two-dimensional representation and lacks calibrated uncertainty. Hand-engineered nonlinear features can help, but then the representation, not the threshold alone, supplies the nonlinearity.

**Computational complexity / scalability notes:** A dense binary pass costs $`O(nd)`$; $`E`$ passes cost $`O(End)`$, with $`O(d)`$ model memory. Multiclass one-versus-rest adds a factor $`c`$. Sparse costs depend on nonzeros, not merely nominal feature dimension.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Illustrative (not a claimed deployment).** For handwritten-character recognition, scikit-learn's [documented digits example](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) fits the public 8-by-8 digit vectors and reports `score(X, y) = 0.939...`. This is **training accuracy on the same data**, not held-out performance. Pixels enter ten linear scorers; their largest score selects a digit for a recognition pipeline. My rationale for starting here rather than with an MLP is to test whether the supplied features already separate classes cheaply. As a separate toy update, $`x=(1,2), y=+1, w=(0,0), b=0,\eta=1`$ gives $`w=(1,2),b=1`$. Neither the tutorial score nor this calculation establishes production OCR quality.

**Notable vendor implementations/libraries:** Scikit-learn `Perceptron` and `SGDClassifier` expose the same underlying linear machinery. They are software implementations, not evidence of a commercial deployment.

### 1.5.2 Multilayer perceptron (MLP)

**Name:** Multilayer perceptron; a fully connected feed-forward network. The worked instantiation has one 40-unit hidden layer.

**Category & sub-category:** Supervised learning; neural foundations; dense nonlinear classification and regression.

**Originating paper/vendor/year:** Multilayer networks predate their widespread practical training. Rumelhart, Hinton, and Williams' [1986 backpropagation paper](https://doi.org/10.1038/323533a0) popularized learning internal representations by reverse differentiation; it should not be credited as the invention of every component of the MLP.

**Core mechanism:** Each layer forms $`h_\ell=\phi(W_\ell h_{\ell-1}+b_\ell)`$. Hidden nonlinearities let successive layers carve and recombine regions of input space; the last layer maps that representation to a target. Without nonlinearities, a stack of affine layers collapses to one affine transformation. Training adjusts early features according to their eventual contribution to output error, unlike a perceptron with fixed input features.

**Inputs/outputs and typical data types:** Fixed-length numeric vectors become class probabilities, multilabel scores, or continuous predictions. Flattening an image permits MLP processing but discards explicit spatial adjacency.

**Architecture diagram description:**

```text
MNIST 28x28 pixels -> flatten/scale -> 784 inputs
                   -> Dense(40), ReLU -> Dense(10), softmax
                   -> digit probabilities -> selected digit
```

**Activation functions used and why:** The [scikit-learn example](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html) uses the classifier's default ReLU hidden activation and multiclass softmax output. ReLU avoids positive-side saturation; softmax makes competing digit scores a normalized distribution. A regression variant normally uses a linear output.

**Loss function(s):** Multiclass cross-entropy with an L2 penalty for this example. Squared error is a different supervised regression instantiation, not its classification loss.

**Optimization algorithm(s):** The example explicitly selects SGD, initial learning rate 0.2, eight epochs, and the implementation's constant schedule. Default SGD momentum is 0.9 with Nesterov acceleration. The short run is deliberately stopped before convergence for documentation resource limits; it is not an optimized MNIST recipe.

**Regularization techniques:** `alpha=1e-4` specifies the L2 penalty. Pixel intensities are divided by 255. The example does not add dropout or batch normalization. Its held-out partition measures generalization but does not make early termination equivalent to validation-tuned early stopping.

**Backpropagation considerations:** Reverse-mode differentiation applies the chain rule through every dense layer. ReLU units can become inactive; saturating alternatives can produce small gradients. Initialization and input scaling matter, and gradient descent has no general guarantee of finding the globally best nonlinear classifier.

**Parameter count / scaling behavior:** The displayed 784-40-10 network has $`784(40)+40+40(10)+10=31,810`$ parameters, calculated including biases. In general $`p=\sum_\ell(d_{\ell-1}+1)d_\ell`$; doubling adjacent widths approximately quadruples their matrix size.

**Training paradigm:** Label-supervised digit classification from random initialization. Autoencoders and self-supervised MLP-based objectives are separate uses, discussed in [unsupervised neural learning](05-unsupervised-neural.md).

**Hardware/parallelism considerations:** Small models run comfortably on a CPU. Larger batches and widths map well to GPU matrix multiplication; weights, activations, gradients, and optimizer buffers all consume memory.

**Strengths and limitations:** MLPs are flexible baselines for engineered features and modest nonlinear problems. They do not automatically exploit image translation, sequence order, or graph symmetry, and can be less sample-efficient than architectures that encode those structures.

**Computational complexity / scalability notes:** Dense inference is $`O(p)`$ per example; a training epoch is approximately $`O(np)`$, with an implementation-dependent backpropagation constant. Training stores batch activations in addition to $`O(p)`$ parameters and optimizer state.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Scikit-learn's [MNIST weight-visualization study](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html) addresses digit recognition using OpenML `mnist_784`, version 1. Its random-state-0 split uses **30% training and 70% testing**, not MNIST's conventional 60,000/10,000 split. The displayed test accuracy is **0.953061**, versus training accuracy **0.986429**, after eight epochs. A normalized 784-pixel vector activates learned stroke-like combinations and produces a ten-way prediction that could feed a document-recognition stage. My rationale relative to the perceptron is learning nonlinear feature combinations; this does not establish superiority under a controlled matched-split comparison. No business KPI or deployed document workflow is reported.

**Notable vendor implementations/libraries:** Scikit-learn `MLPClassifier`/`MLPRegressor`, PyTorch `Linear`, and Keras `Dense`. Optimizer and normalization defaults differ across these libraries.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Perceptron | Fixed or sparse feature vectors | Cheap online linear baseline | Requires suitable feature-space separation | Scikit-learn digits demonstration; reported score is in-sample |
| MLP | Fixed-length numeric vectors | Learns nonlinear feature combinations | No built-in spatial or relational structure | MNIST documentation benchmark with a nonstandard 30/70 split |

## 1.6 Convolutional families

Convolution shares a local detector across positions. The families below change receptive fields, connectivity, channel mixing, or scaling strategy. None guarantees perfect translation invariance: padding, stride, pooling, and boundary handling all affect equivariance.

### 1.6.1 Generic convolutional neural network (CNN)

**Name:** Convolutional neural network; instantiated here as the small supervised MNIST classifier in the Keras example.

**Category & sub-category:** Supervised learning; convolutional families; local, weight-shared visual feature extraction.

**Originating paper/vendor/year:** CNNs have multiple historical precursors; LeCun and colleagues' [1998 document-recognition paper](https://leon.bottou.org/papers/lecun-98h) is a foundational trainable formulation. The executable representative is Francois Chollet's [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/), originally published as an example in 2015 and presented in a Keras 3-compatible form at the evidence date.

**Core mechanism:** A kernel slides across a feature map, detecting the same local pattern at each valid location. Multiple kernels learn different patterns; nonlinearities and further layers combine them into more contextual features. Pooling reduces spatial resolution. Shared weights make this more constrained than connecting every pixel to every hidden unit, an advantageous bias when nearby pixels form reusable strokes or textures.

**Inputs/outputs and typical data types:** Regular grids, here 28-by-28 single-channel images, become image-level class probabilities. One-dimensional and three-dimensional convolutions adapt the mechanism to signals or volumes, but have different shapes and costs.

**Architecture diagram description:**

```text
28x28x1 -> Conv3x3(32), ReLU -> MaxPool2
         -> Conv3x3(64), ReLU -> MaxPool2 -> 5x5x64
         -> Flatten(1600) -> Dropout(0.5) -> Dense(10), softmax
```

**Activation functions used and why:** ReLU makes the local feature hierarchy nonlinear and inexpensive to evaluate. Softmax makes the final outputs competing digit probabilities. Pooling is a reduction operation, not an additional learned activation.

**Loss function(s):** Categorical cross-entropy on one-hot digit labels in the cited implementation. It evaluates the probability assigned to the annotated digit, rather than merely counting correct argmax predictions.

**Optimization algorithm(s):** Adam, batch size 128, and 15 epochs. The example passes `optimizer="adam"`, using Keras's default initial rate of 0.001 rather than specifying a custom schedule. No learning-rate decay schedule is introduced by the example.

**Regularization techniques:** Dropout 0.5 before the final dense layer, intensity scaling to [0,1], and a 10% validation reservation from the training data. This particular network has neither batch normalization nor synthetic geometric augmentation.

**Backpropagation considerations:** Shared kernels accumulate gradients from all their spatial uses. Max-pooling sends gradients through winning positions; ReLU blocks gradients for negative preactivations. Layer geometry and padding must remain consistent between training and inference.

**Parameter count / scaling behavior:** The example reports **34,826 trainable parameters**. A standard convolution has $`q^2C_{\rm in}C_{\rm out}`$ kernel weights plus optional biases, independent of image area; a flattened dense head does depend on the final area.

**Training paradigm:** Supervised learning from digit labels, initialized and trained on the designated training subset. This is not an autoencoder or a self-supervised visual representation objective.

**Hardware/parallelism considerations:** Convolutions parallelize across images, locations, and channels. Optimized GPU kernels help large workloads; for this small network, transfer and launch overhead can matter more than theoretical peak arithmetic throughput.

**Strengths and limitations:** The model efficiently learns local image features. Downsampling can discard small details, and a fixed flattened head constrains input geometry. Robustness to handwriting styles outside MNIST must be measured, not inferred from parameter sharing.

**Computational complexity / scalability notes:** A dense convolution costs $`O(BH_{\rm out}W_{\rm out}q^2C_{\rm in}C_{\rm out})`$ per forward pass. Summing over layers, then accounting for backward operations and $`E`$ epochs, is necessary; "CNNs are linear-time" without these dimensions is misleading.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Keras's [documented MNIST run](https://keras.io/examples/vision/mnist_convnet/) reports **0.9919000268 test accuracy**, about **99.19%**, on the conventional **10,000-image test set**. Of the 60,000 supplied training images, 6,000 are reserved by `validation_split=0.1`. A scanned digit is normalized, transformed into local stroke features, and assigned its highest-probability numeral; a document system would still need segmentation and confidence handling. My rationale relative to flattening directly into an MLP is reusable local structure. This is an educational benchmark, not measured automation of a bank or postal operation, and its split differs from the MLP example above.

**Notable vendor implementations/libraries:** Keras/TensorFlow convolutional layers, PyTorch `Conv1d/2d/3d`, and accelerator convolution libraries such as NVIDIA cuDNN. These supply operators, not one universal CNN recipe.

### 1.6.2 LeNet

**Name:** LeNet, focusing on the historical LeNet-5 architecture and distinguishing it from simplified teaching reproductions.

**Category & sub-category:** Supervised learning; convolutional families; small document and handwritten-character recognizers.

**Originating paper/vendor/year:** Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner, [Gradient-Based Learning Applied to Document Recognition](https://leon.bottou.org/papers/lecun-98h), Proceedings of the IEEE, 1998. Earlier LeNet-family systems preceded this paper; 1998 identifies the detailed reference, not the first convolutional network.

**Core mechanism:** LeNet alternates learned local feature detectors and subsampling, progressively combining strokes into character-level evidence. LeNet-5's C3 layer connects selected, rather than all, earlier feature maps. Its learned subsampling stages are not identical to a modern parameter-free average-pooling layer. The final historical classifier measures distances from a learned feature vector to class prototypes, illustrating that a CNN need not end in today's usual softmax linear head.

**Inputs/outputs and typical data types:** Centered, normalized grayscale character images, represented in a 32-by-32 input field, become ten digit-associated scores. Recognition of an entire cheque additionally requires locating fields and resolving character segmentation.

**Architecture diagram description:**

```text
32x32 -> C1: 6 maps, 5x5 -> S2: subsample
      -> C3: 16 selectively connected maps -> S4: subsample
      -> C5: 120 -> F6: 84 -> 10 Euclidean-RBF scores
      -> lowest-energy digit
```

**Activation functions used and why:** The historical feature units use scaled hyperbolic tangents; the output uses Euclidean radial-basis-function distances. Bounded feature responses suit the prototype coding. Modern ReLU/softmax "LeNet" examples are useful adaptations, not evidence that the original used those operations.

**Loss function(s):** The original discussion includes squared-error and discriminative, likelihood-related criteria for the distance-based outputs. A competitive criterion lowers the correct class energy relative to alternatives. Replacing that head with logits normally entails supervised cross-entropy; the head and loss must be changed together.

**Optimization algorithm(s):** The historical training uses backpropagation with stochastic, curvature-aware learning procedures discussed in the paper; there is no family-wide learning-rate schedule. For an explicitly reproducible **different** recipe, [Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html) trains its sigmoid/average-pooling LeNet adaptation with SGD at 0.1 for ten epochs on Fashion-MNIST. That schedule is not attributed to the historical cheque system.

**Regularization techniques:** Local connectivity and weight sharing constrain the hypothesis space. Normalization and suitable image transformations address handwriting variability. The original architecture does not contain batch normalization or modern dropout.

**Backpropagation considerations:** Saturated tanh units weaken gradients, making input scaling and initialization important. Gradients must respect C3's actual connection pattern and the trainable subsampling parameters; silently replacing them changes the model.

**Parameter count / scaling behavior:** LeNet-5 is approximately a **60,000-parameter** model. Fully connected C3 replacements and modern heads produce different totals. Weight sharing keeps the convolutional portion small, whereas the final feature-to-class mapping depends on the chosen character vocabulary.

**Training paradigm:** Supervised character learning. The larger document system can additionally optimize multiple modules against labelled strings; it is not unsupervised pretraining.

**Hardware/parallelism considerations:** The small model is easily accommodated by contemporary CPUs and GPUs. Historical document throughput depended on the complete recognition pipeline, not just neural arithmetic.

**Strengths and limitations:** LeNet explains how inductive bias makes small-data image learning practical. Its small receptive fields and low-resolution design do not make it a drop-in recognizer for arbitrary photographs or complex page layouts.

**Computational complexity / scalability notes:** Sum convolution costs $`HWq^2C_{\rm in}C_{\rm out}`$, adjusted for sparse channel connectivity, plus dense-layer matrix products. Running a character network separately on many candidate crops can make segmentation search a larger expense than one network evaluation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** The [publisher's account of the 1998 study](https://proceedingsoftheieee.ieee.org/gradient-based-learning-applied-to-document-recognition/) describes a commercially deployed **graph-transformer cheque-reading system containing CNN character recognizers**. A cheque image enters field extraction and segmentation hypotheses; recognizer scores inform candidate digit strings; globally trained modules select a reading or support rejection. The documented design combines recognition with segmentation rather than assuming characters arrive perfectly cropped. It establishes a real document-processing application, **not that every deployed module was an unchanged LeNet-5**, nor an isolated LeNet accuracy or financial saving. The source reports deployment at substantial daily volume, but this chapter does not convert system throughput into an architecture-specific business KPI.

**Notable vendor implementations/libraries:** Historical author implementations and modern PyTorch, Keras, and D2L reproductions. Check activation, pooling, padding, C3 connectivity, and output head before treating two implementations as equivalent.

### 1.6.3 AlexNet

**Name:** AlexNet; the 2012 ImageNet convolutional architecture, not every later pretrained object carrying that name.

**Category & sub-category:** Supervised learning; convolutional families; large-scale image classification with GPU-trained deep CNNs.

**Originating paper/vendor/year:** Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), NeurIPS, then called NIPS, 2012; University of Toronto.

**Core mechanism:** Five convolutional layers build visual features before three fully connected layers classify them. ReLU permits faster optimization than the saturating alternatives investigated in the paper. Overlapping max-pooling reduces resolution; local response normalization operates across nearby feature channels. Some convolutions are split into groups because the original model is partitioned between two GPUs. This historical engineering constraint is part of the published architecture.

**Inputs/outputs and typical data types:** RGB crops from natural photographs become probabilities for 1,000 ImageNet classes. The paper describes 224-by-224 crops; reproductions using 227 pixels or different padding must not be assumed shape-identical.

**Architecture diagram description:**

```text
RGB crop -> Conv11/stride4 -> ReLU/LRN/pool
         -> Conv5 -> ReLU/LRN/pool
         -> Conv3 -> Conv3 -> Conv3 -> pool
         -> FC4096 -> FC4096 -> FC1000 -> softmax
            [selected convolutional groups split across two GPUs]
```

**Activation functions used and why:** ReLU throughout the hidden network; softmax at the output. LRN rescales activations but is not batch normalization, which was not part of this model.

**Loss function(s):** Multinomial negative log-likelihood, equivalently cross-entropy for the annotated image class, plus weight decay during optimization.

**Optimization algorithm(s):** SGD with momentum 0.9, batch size 128, and weight decay 0.0005. The initial learning rate is 0.01, manually divided by ten when validation improvement stalls; the paper reports three such reductions.

**Regularization techniques:** Dropout 0.5 in the first two fully connected layers, random crops and horizontal reflections, and PCA-based RGB intensity perturbations. Test-time crop averaging is a separate inference procedure, not another training regularizer.

**Backpropagation considerations:** ReLU reduces saturation but does not eliminate dead units or optimization instability. Weight sharing aggregates spatial gradients; cross-GPU communication is necessary at selected connections. The original normalization and group structure must be included when reproducing the result.

**Parameter count / scaling behavior:** Approximately **60 million parameters**, much of the storage in dense layers. Shrinking the classifier can reduce parameters more than it reduces convolutional feature-extraction work.

**Training paradigm:** Supervised ImageNet classification. Additional ImageNet Fall 2011 **labelled** pretraining was used for two members of the best 2012 ensemble; that is not self-supervised image learning.

**Hardware/parallelism considerations:** The published implementation used two NVIDIA GPUs with explicit model partitioning. A modern single accelerator may hold the network, but that does not reproduce historical timing.

**Strengths and limitations:** AlexNet demonstrated learnable large-scale visual representations and practical GPU training. Its large dense head, early aggressive downsampling, and historical normalization make it an inefficient default for many contemporary tasks.

**Computational complexity / scalability notes:** Add the convolutional costs at each stage and the dense matrix products. Grouped layers reduce channel interactions relative to fully connected convolutions. Ten test crops multiply inference work even for one trained model; ensembles multiply it further.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** For natural-image recognition, the [paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) reports **18.2% top-5 validation error** for its single ILSVRC-2012 CNN and **15.3% top-5 test error** for an ensemble of **seven networks**, including two pretrained on the approximately 15-million-image, 22,000-category Fall 2011 release and then fine-tuned. Five ordinary models gave 16.4%, **not a single model**. The [challenge results](https://image-net.org/challenges/LSVRC/2012/results.html) independently distinguish supplied-only from additional-data entries. RGB crops become averaged class scores and a five-label candidate list; top-5 counts success if the reference label appears anywhere in it. My technical rationale relative to fixed feature pipelines is end-to-end feature learning. No commercial photo-tagging KPI is established.

**Notable vendor implementations/libraries:** The authors' CUDA-convnet lineage and torchvision's `alexnet`. Torchvision's implementation is a later architectural adaptation, so its weights and evaluation recipe must be identified separately.

### 1.6.4 VGG

**Name:** VGG, with VGG-16 configuration D as the main reference and VGG-19 configuration E as a deeper variant.

**Category & sub-category:** Supervised learning; convolutional families; deep, homogeneous stacks of small convolutions.

**Originating paper/vendor/year:** Karen Simonyan and Andrew Zisserman, Oxford Visual Geometry Group, [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/html/1409.1556v6). The preprint appeared in 2014; the conference publication is ICLR 2015.

**Core mechanism:** Repeated 3-by-3 convolutions, interleaved with nonlinearities, enlarge effective receptive fields without using a large spatial kernel at every step. Two stride-one 3-by-3 layers have a 5-by-5 receptive field, but also an intervening nonlinearity. Channels increase as pooling reduces image area. A large dense classifier aggregates the final feature maps. The design's uniformity made depth a comparatively easy experimental variable.

**Inputs/outputs and typical data types:** RGB images become 1,000-way class scores; intermediate features can be reused for other visual tasks. Training uses 224-pixel crops, while dense evaluation can operate on larger resized images.

**Architecture diagram description:**

```text
RGB -> [Conv3x3(64)]x2 -> pool -> [Conv3x3(128)]x2 -> pool
    -> [Conv3x3(256)]x3 -> pool -> [Conv3x3(512)]x3 -> pool
    -> [Conv3x3(512)]x3 -> pool -> FC4096 -> FC4096 -> FC1000
    -> softmax                          [VGG-16 configuration D]
```

**Activation functions used and why:** ReLU follows hidden weight layers; softmax closes the classifier. The main VGG-16/19 configurations do not use batch normalization. "VGG16-BN" denotes a subsequent variant, not the same historical model.

**Loss function(s):** Multinomial logistic loss, or class cross-entropy, with weight decay.

**Optimization algorithm(s):** Minibatch SGD with momentum 0.9, batch size 256, and initial learning rate 0.01. The rate drops by ten when validation accuracy stalls; the paper describes three drops and 74 training epochs. Some deeper runs initialize layers from a shallower trained configuration.

**Regularization techniques:** Weight decay $`5\times10^{-4}`$, dropout 0.5 in the first two dense layers, random crops and reflections, and scale jittering in designated runs. These are part of the measured recipe, not effects attributable solely to depth.

**Backpropagation considerations:** Deep plain stacks are sensitive to initialization and gradient conditioning. The paper's staged initialization is historically relevant; modern initialization improvements should not be silently substituted into a purported reproduction.

**Parameter count / scaling behavior:** The source reports approximately **138 million parameters for VGG-16** and **144 million for VGG-19**. Dense layers dominate storage, while early high-resolution convolutions contribute substantial computation.

**Training paradigm:** Supervised ImageNet learning. Feature extraction and supervised fine-tuning reuse that learned representation; using VGG inside an unsupervised objective does not retrospectively change its pretraining signal.

**Hardware/parallelism considerations:** Regular 3-by-3 layers map well to optimized accelerators. Nevertheless, weight storage, dense-layer bandwidth, and large activation tensors can make VGG expensive relative to compact backbones.

**Strengths and limitations:** The architecture is straightforward to inspect and adapt, with multiscale hierarchical features. Parameter-heavy dense heads and the absence of residual shortcuts make large-scale expansion less attractive than in later families.

**Computational complexity / scalability notes:** Each 3-by-3 layer costs approximately $`O(9BHW C_{\rm in}C_{\rm out})`$. More layers can increase useful receptive field without proportionately increasing kernel size, but feature-map storage and repeated high-channel convolutions remain material costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In the [paper's Table 3](https://arxiv.org/html/1409.1556v6), configuration D trained with image-side jittering over [256,512] obtains **25.6% top-1 and 8.1% top-5 error** on the **ImageNet validation set**, evaluated at image-side scale 384. This is a **single model at one test scale with dense evaluation**, not a single 224-pixel crop and not an ensemble. A resized photograph yields a spatial class-score map; spatial and reflected-image scores are aggregated before selecting labels. My rationale relative to AlexNet is a simpler, deeper small-kernel feature hierarchy, not a claim that depth alone explains this result. The study reports recognition accuracy, not commercial image-search value.

**Notable vendor implementations/libraries:** Oxford's released models, torchvision VGG, and Keras VGG16/VGG19. Check whether weights use batch normalization and whether preprocessing follows the original RGB/BGR and mean-subtraction conventions.

### 1.6.5 ResNet

**Name:** Residual network, or ResNet; the original post-activation family, including bottleneck ResNet-50/101/152.

**Category & sub-category:** Supervised learning; convolutional families; residual feature learning.

**Originating paper/vendor/year:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Microsoft Research, [Deep Residual Learning for Image Recognition](https://arxiv.org/html/1512.03385v1), 2015 preprint and CVPR 2016 paper.

**Core mechanism:** A block learns a correction $`F(x)`$ and forms $`y=\operatorname{ReLU}(x+F(x))`$ in the original post-activation design. When dimensions differ, a projection or specified padding aligns the shortcut. A bottleneck reduces channels with 1-by-1 convolution, processes them with 3-by-3 convolution, and expands again. Learning a small correction can be easier than forcing every additional block to relearn a useful transformation from scratch.

**Inputs/outputs and typical data types:** Images become class probabilities or spatial feature maps for downstream detectors and segmenters. The representative classification model ends with global average pooling.

**Architecture diagram description:**

```text
image -> convolutional stem -> residual stages -> global average -> classifier
                              |
block: x -> 1x1/BN/ReLU -> 3x3/BN/ReLU -> 1x1/BN --+
       x ---------------- identity/projection ----------------+ -> add -> ReLU
```

**Activation functions used and why:** ReLU and batch normalization in the original blocks, plus softmax for classification. The later pre-activation ResNet moves normalization and activations relative to the addition; it is not identical.

**Loss function(s):** Supervised softmax cross-entropy for ImageNet. Detector or segmentation uses attach different task-specific losses.

**Optimization algorithm(s):** SGD with momentum 0.9, batch size 256, initial learning rate 0.1, and reductions by ten when error plateaus. The paper permits up to 600,000 iterations; this is not necessarily the later standardized "90-epoch ResNet recipe."

**Regularization techniques:** Weight decay $`10^{-4}`$, training crops and flips, and batch normalization. The original classification experiments do not use dropout. Batch normalization requires appropriate training/inference statistics.

**Backpropagation considerations:** Shortcut paths improve signal and gradient propagation, but do not guarantee every deep model trains. The paper's degradation argument is not simply "all gradients vanished": normalized plain networks could still have worse training error when made deeper.

**Parameter count / scaling behavior:** ResNet-152 is roughly a **60-million-parameter** model; count varies with head and implementation. Bottlenecks permit deeper models without the parameter growth of equally wide, full 3-by-3 stacks.

**Training paradigm:** Supervised class learning; labelled transfer tasks reuse the backbone. Contrastive ResNet pretraining is a different objective and belongs in the representation-learning chapters.

**Hardware/parallelism considerations:** Convolutions use accelerator-friendly dense operations. Shortcuts add relatively little arithmetic but require keeping tensors live; normalization and small batches can become system-level constraints.

**Strengths and limitations:** Residual learning makes deep feature hierarchies practical and widely reusable. Greater depth still increases latency and memory, and residual connections do not eliminate distribution shift, shortcut learning, or overfitting.

**Computational complexity / scalability notes:** A bottleneck's cost is the sum of two channel projections and one reduced-width spatial convolution, approximately $`BHW(Cr+9r^2+rC')`$. It is incorrect to price every layer as a full $`C`$ by $`C`$ 3-by-3 convolution.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original paper](https://arxiv.org/html/1512.03385v1) reports **4.49% top-5 ImageNet validation error** for a **single ResNet-152**, using its best fully convolutional, multiscale evaluation rather than a single crop. The often-quoted **3.57%** is instead the **six-model ensemble's ILSVRC-2015 test error**. A photograph passes through residual corrections, global pooling, and class scoring; evaluation asks whether its labelled class appears in the top five. The paper's documented motivation is the degradation of deeper plain networks, not deployment in a named Microsoft product. No business outcome is inferred from challenge rank.

**Notable vendor implementations/libraries:** Torchvision, Keras applications, and numerous detection frameworks. ResNet-v1, pre-activation variants, and torchvision's bottleneck stride placement differ; select a specific graph and checkpoint.

### 1.6.6 Inception

**Name:** Inception family, anchored to **GoogLeNet/Inception-v1**. Later Inception versions add different factorization and normalization choices.

**Category & sub-category:** Supervised learning; convolutional families; multibranch, multiscale feature extraction.

**Originating paper/vendor/year:** Christian Szegedy and colleagues, [Going Deeper with Convolutions](https://arxiv.org/html/1409.4842v1), 2014 preprint and CVPR 2015 paper, involving Google and collaborating researchers.

**Core mechanism:** An Inception module processes the same input through several branches and concatenates their channels. One branch uses 1-by-1 convolution, others use reduced-width 3-by-3 or 5-by-5 convolution, and another pools before projection. Bottleneck projections control the cost of large spatial filters. The network can combine local and broader-scale evidence without choosing one kernel size for every feature.

**Inputs/outputs and typical data types:** RGB images yield class probabilities or reusable convolutional feature maps. GoogLeNet's main path has 22 layers when counting parameterized layers along its depth.

**Architecture diagram description:**

```text
                         +-> 1x1 -----------------+
input feature map -------+-> 1x1 -> 3x3 -----------+
                         +-> 1x1 -> 5x5 -----------+-> channel concat
                         +-> pool -> 1x1 ---------+
stem -> repeated modules -> global average -> dropout -> classifier
                         \-> auxiliary heads during training only
```

**Activation functions used and why:** ReLU in the convolutional branches and softmax in classification heads. Inception-v1 is not the later batch-normalized Inception-v2/v3 family.

**Loss function(s):** Main supervised cross-entropy plus auxiliary classifier losses, each weighted 0.3 during the original training. Auxiliary heads are removed for inference.

**Optimization algorithm(s):** Asynchronous SGD with momentum 0.9; the documented schedule decreases the learning rate by 4% every eight epochs. The authors explicitly describe changes to training procedures across ensemble members, so a universal initial learning rate for every final member is not reconstructed here. Polyak averaging forms inference parameters.

**Regularization techniques:** Crop and scale augmentation, dropout, and auxiliary supervision. The main classifier uses 40% dropout in the architecture table. Global average pooling substantially reduces the need for a huge fully connected head.

**Backpropagation considerations:** Gradients from the main and auxiliary objectives reach earlier layers through different paths. Concatenated branches must preserve compatible spatial dimensions. Additional supervision assists optimization but also changes the objective compared with a plain sequential CNN.

**Parameter count / scaling behavior:** The main network has roughly **seven million parameters**, as indicated by its layerwise counts; auxiliary training heads add parameters. Channel-reduction widths are crucial: removing bottlenecks can make a multibranch design much more expensive.

**Training paradigm:** Supervised ImageNet classification. "Sparse" architectural motivation in this paper does not mean token-routed mixture-of-experts training.

**Hardware/parallelism considerations:** Branches expose parallel work, but concatenation, small kernels, and synchronization can limit actual utilization. The paper's distributed training setup is not a universal production implementation recipe.

**Strengths and limitations:** Inception provides multiscale processing with a compact main classifier. Branch widths and topology are more complicated to tune than a homogeneous VGG stack; FLOP savings do not guarantee corresponding latency savings.

**Computational complexity / scalability notes:** Add each branch's convolution and pooling cost. A 1-by-1 reduction from $`C`$ to $`r`$ changes the following 5-by-5 cost from $`25HWC C'`$ to $`25HWrC'`$, while adding the reduction cost $`HWCr`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [2014 preprint's ILSVRC-2014 result](https://arxiv.org/html/1409.4842v1) is **6.67% top-5 classification error** on the validation and test evaluations for its final submission. That submission averages **seven models** using **144 crops per image**, not one pass through one GoogLeNet. Each crop produces a class distribution; averaging distributions yields the final ranked labels. Multiscale evidence is the authors' architectural motivation, while the extensive evaluation aggregation is separately responsible for part of the final score. The result establishes a recognition benchmark, not a measured Google-product deployment.

**Notable vendor implementations/libraries:** TensorFlow-Slim and torchvision `googlenet` implement this lineage. Inception-v3 factories expose a materially different architecture and must not inherit v1 benchmark claims.

### 1.6.7 EfficientNet

**Name:** EfficientNet, focusing on the original B0-B7 family rather than EfficientNetV2 or later self-trained checkpoints.

**Category & sub-category:** Supervised learning; convolutional families; compound-scaled mobile inverted-bottleneck networks.

**Originating paper/vendor/year:** Mingxing Tan and Quoc Le, Google, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html), ICML 2019. Quantitative details below use the explicitly identified [arXiv v5](https://arxiv.org/html/1905.11946v5) tables.

**Core mechanism:** A searched B0 baseline uses mobile inverted bottlenecks: expand channels, apply depthwise spatial filtering, recalibrate channels with squeeze-and-excitation, and project back. Compound scaling increases depth, width, and input resolution together. With coefficients $`\alpha,\beta,\gamma`$, depth, width, and resolution scale as $`\alpha^\phi,\beta^\phi,\gamma^\phi`$, subject approximately to $`\alpha\beta^2\gamma^2\approx2`$. This balances competing capacity dimensions instead of scaling only layer count.

**Inputs/outputs and typical data types:** RGB images at a model-specific resolution become class probabilities. B0's reference resolution is 224; larger variants require larger inputs and more activation memory.

**Architecture diagram description:**

```text
image -> stem -> repeated MBConv stages -> 1x1 head -> global average -> classifier
MBConv: expand -> SiLU -> depthwise -> SiLU -> squeeze/excite -> linear project
         input ---------------- compatible residual shortcut ----------------+
```

**Activation functions used and why:** SiLU, also called Swish-1, provides a smooth gated nonlinearity. Squeeze-and-excitation uses sigmoid channel gates; the projection bottleneck is linear. Softmax supplies image-class probabilities.

**Loss function(s):** Supervised classification cross-entropy. Loss modifications in later distillation or self-training recipes are not part of the architecture definition.

**Optimization algorithm(s):** The v5 recipe specifies RMSProp with decay 0.9 and momentum 0.9; initial learning rate 0.256 decays by 0.97 every 2.4 epochs. These values describe the paper's training configuration, not a portable rate for arbitrary batch sizes.

**Regularization techniques:** Batch normalization, weight decay $`10^{-5}`$, AutoAugment, stochastic depth with reported survival probability 0.8, and dropout increasing from 0.2 at B0 to 0.5 at B7. A 25,000-image training minival subset supports checkpoint stopping.

**Backpropagation considerations:** Residual paths and normalization assist optimization. Channel gates can suppress gradients, and increasing resolution changes batch-memory constraints. Changing batch size may require revisiting normalization and optimization settings.

**Parameter count / scaling behavior:** V5 reports **5.3 million parameters for B0** and **66 million for B7**. Scaling width increases many pointwise-convolution weights quadratically; resolution increases activations and computation without directly enlarging most kernel tensors.

**Training paradigm:** The cited B-family experiment is supervised ImageNet training, followed where relevant by supervised transfer. Noisy Student and other additional-data training stages are separate recipes.

**Hardware/parallelism considerations:** Pointwise matrix products and depthwise kernels have different utilization characteristics. Larger resolution can erase apparent parameter-efficiency advantages for memory-limited devices; measured latency must specify runtime and accelerator.

**Strengths and limitations:** Compound scaling offers a principled accuracy/resource trade-off. The optimal balance is hardware- and workload-dependent, and architecture-search cost is additional to training an already selected B0.

**Computational complexity / scalability notes:** A depthwise-plus-pointwise block costs terms proportional to $`HWq^2C`$ and $`HWC C'`$, plus expansion and excitation. The approximate compound-cost relation is a design model, not a claim of exactly doubled wall-clock time per scaling step.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In [v5 Table 2](https://arxiv.org/html/1905.11946v5), **EfficientNet-B0** achieves **77.1% top-1 and 93.3% top-5 ImageNet validation accuracy**, using the paper's **single-model, single-crop** evaluation. A 224-pixel photograph passes through scaled bottlenecks and channel gates to produce an object label, suitable as one component of a visual cataloguing workflow. My rationale relative to a uniformly widened CNN is resource-aware scaling; no deployed cataloguing KPI is reported. Version provenance matters: v5 gives B7 **84.3% top-1**, whereas the ICML landing-page abstract says **84.4%**. Those values are not silently treated as one interchangeable checkpoint.

**Notable vendor implementations/libraries:** Google's EfficientNet implementations, Keras applications, torchvision, and timm. Record the specific weights, preprocessing resolution, and whether additional-data training was used.

### 1.6.8 ConvNeXt

**Name:** ConvNeXt, focusing on the original **ConvNeXt-Tiny**, not ConvNeXt V2.

**Category & sub-category:** Supervised learning; convolutional families; modernized pure convolutional backbones.

**Originating paper/vendor/year:** Zhuang Liu and colleagues, [A ConvNet for the 2020s](https://arxiv.org/html/2201.03545v2), CVPR 2022, associated with Meta AI and academic collaborators.

**Core mechanism:** ConvNeXt revisits a ResNet-style hierarchy using design choices also associated with vision Transformers: patch-like downsampling, large depthwise kernels, inverted bottlenecks, layer normalization, and fewer activations. It remains convolutional rather than computing content-dependent self-attention. The paper also modernizes the training recipe before making architectural comparisons, separating part of the recipe effect from the architectural effect.

**Inputs/outputs and typical data types:** RGB images become class predictions or hierarchical feature maps. Tiny uses four stages whose reference widths are 96, 192, 384, and 768.

**Architecture diagram description:**

```text
RGB -> 4x4/stride4 stem -> stages [3,3,9,3] -> global average -> linear head
block: x -> depthwise 7x7 -> LayerNorm -> pointwise expand 4x
         -> GELU -> pointwise project -> LayerScale -> stochastic depth -> +x
```

**Activation functions used and why:** GELU provides the block's principal smooth nonlinearity. Layer normalization replaces the batch-dependent normalization used in many older CNNs. The classification output is trained as logits with cross-entropy.

**Loss function(s):** Supervised cross-entropy with label smoothing and the soft targets induced by Mixup/CutMix. These targets differ from a single hard one-hot class but remain label-derived supervision.

**Optimization algorithm(s):** For ImageNet-1K, AdamW with learning rate 0.004, batch size 4096, and 300 epochs; 20 epochs of linear warmup precede cosine decay. Weight decay is 0.05. Separate ImageNet-22K pretraining/fine-tuning recipes are not the Tiny 1K-only result below.

**Regularization techniques:** Mixup, CutMix, RandAugment, random erasing, label smoothing, and stochastic depth. LayerScale starts at $`10^{-6}`$; exponential moving average is used in the reported training setup.

**Backpropagation considerations:** Residual connections and small initial residual scales stabilize a deeper network. Normalization layout, stochastic-depth behavior, and mixed-precision handling can affect implementation equivalence; adopting the name without the recipe is insufficient.

**Parameter count / scaling behavior:** The paper reports **29 million parameters for ConvNeXt-T**. Larger family members increase width and stage depth. Large depthwise spatial kernels add only $`q^2C`$ weights, while pointwise expansions dominate much of channel-mixing capacity.

**Training paradigm:** The reference Tiny result is supervised ImageNet-1K learning. Masked-image pretraining and ConvNeXt V2's changes are separate methods, not implied by the 2022 name.

**Hardware/parallelism considerations:** Channels-last layouts and efficient depthwise kernels can be important. Layer normalization and data layout conversion introduce costs not captured by counting multiplications. A FLOP-matched Transformer is not guaranteed to have equal latency.

**Strengths and limitations:** ConvNeXt provides an uncomplicated convolutional alternative to attention backbones with competitive scaling. Its strong results depend partly on substantial augmentation and long training, so architecture-only comparisons can mislead.

**Computational complexity / scalability notes:** A block of width $`C`$ costs approximately $`HW(q^2C+8C^2)`$ for the depthwise and two 4x pointwise projections, ignoring small elementwise terms. Fixed-resolution image inference remains a sequence of dense spatial operations.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper's ImageNet-1K table](https://arxiv.org/html/2201.03545v2) reports **82.1% top-1 validation accuracy** for **ConvNeXt-T at 224-by-224**, trained on ImageNet-1K, versus **81.3%** for the listed Swin-T configuration. This is a single-model classification comparison, not a claim about the later 22K-pretrained models. A photograph becomes multiresolution convolutional features, then a pooled class decision. The authors' documented question is whether modernized convolutional design and training can compete with hierarchical Transformers. No commercial inspection, search, or sales impact is established by the accuracy difference.

**Notable vendor implementations/libraries:** Meta's released ConvNeXt repository, torchvision, and timm. ConvNeXt V2 and differently trained checkpoints should carry their own provenance.

### 1.6.9 DenseNet

**Name:** Densely connected convolutional network; the representative is **DenseNet-121 with bottleneck and compression**.

**Category & sub-category:** Supervised learning; convolutional families; dense feature reuse through concatenation.

**Originating paper/vendor/year:** Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Weinberger, [Densely Connected Convolutional Networks](https://arxiv.org/html/1608.06993v5), 2016 preprint and CVPR 2017 paper. The cited v5 revision is dated 2018.

**Core mechanism:** Inside a dense block, layer $`\ell`$ receives the concatenation of every preceding layer's feature maps and adds $`k`$ new maps, where $`k`$ is the growth rate. Unlike ResNet, it does not sum old and new representations into the same channels. Bottleneck 1-by-1 convolutions limit the input width of 3-by-3 kernels; transition layers compress channels and downsample between blocks.

**Inputs/outputs and typical data types:** Images become pooled class scores or multistage spatial features. DenseNet-121 uses four dense blocks with 6, 12, 24, and 16 composite layers, and growth rate 32.

**Architecture diagram description:**

```text
stem -> dense block -> transition -> dense block -> ... -> global average -> head
dense block:
x0 ------------+--------------------+
x1 = H1(x0) ---+--> x2=H2([x0,x1]) -+--> x3=H3([x0,x1,x2])
H: BN/ReLU/1x1 bottleneck -> BN/ReLU/3x3 -> k new maps
```

**Activation functions used and why:** Batch normalization and ReLU precede the bottleneck/spatial transformations. The classifier uses softmax cross-entropy. Concatenation preserves feature identities rather than acting as a nonlinear activation.

**Loss function(s):** Supervised class cross-entropy. "Implicit deep supervision" describes shorter gradient paths, not separate labelled auxiliary heads at every layer.

**Optimization algorithm(s):** ImageNet training uses SGD with Nesterov momentum 0.9, batch size 256, and 90 epochs. The initial rate 0.1 is divided by ten at epochs 30 and 60.

**Regularization techniques:** Weight decay $`10^{-4}`$, image augmentation, and batch normalization. The paper's dropout 0.2 applies to specified small datasets without augmentation; it should not automatically be inserted into the ImageNet recipe.

**Backpropagation considerations:** Direct access to earlier features gives many short gradient routes. Naively materializing concatenations and their normalized copies can consume much more memory than parameter count suggests; memory-efficient implementations recompute selected intermediates.

**Parameter count / scaling behavior:** DenseNet-121 is approximately an **eight-million-parameter** model. Within a block, the input width grows as $`C_0+\ell k`$. Bottlenecks and transition compression are essential to keeping this growth manageable.

**Training paradigm:** Supervised visual classification for the cited result. Dense connectivity can also appear in generative or self-supervised models without changing its connectivity definition.

**Hardware/parallelism considerations:** Feature concatenation increases memory traffic and requires access to multiple earlier tensors. An implementation with fewer weights can still use more training activation memory than a residual network.

**Strengths and limitations:** DenseNet encourages feature reuse and parameter efficiency. Its accumulating feature history complicates memory planning and can make naive implementations slower than their arithmetic count suggests.

**Computational complexity / scalability notes:** At fixed block resolution and growth rate, summing transformations over increasing input widths can introduce a quadratic-in-block-depth term. Unique produced feature maps grow linearly, whereas repeatedly materialized concatenation buffers may grow quadratically. These are different memory quantities, not a universal quadratic-memory requirement.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [V5 Table 3](https://arxiv.org/html/1608.06993v5) reports **25.02% top-1 and 7.71% top-5 error** for DenseNet-121 on the **ImageNet validation set with a single 224-pixel crop**. Its **10-crop** errors are instead **23.61% and 6.66%**. An object photograph produces a retained history of low- and high-level features; pooling and the classifier turn that history into a ranked label list. The authors' rationale relative to residual summation is preserving and reusing feature maps. These results do not establish deployed radiology or industrial-inspection performance merely because DenseNet can be adapted to those domains.

**Notable vendor implementations/libraries:** The authors' implementations, torchvision DenseNet, and Keras DenseNet. Memory-efficient switches can change speed/memory trade-offs without changing the mathematical prediction.

### 1.6.10 MobileNet

**Name:** MobileNet, anchored to **MobileNet-v1, width multiplier 1.0, 224-pixel input**. The worked example identifies the 2018-08-02 TensorFlow-Slim checkpoint bundle; v2's inverted residuals and v3's searched blocks are distinct extensions.

**Category & sub-category:** Supervised learning; convolutional families; efficient depthwise-separable visual networks.

**Originating paper/vendor/year:** Andrew Howard and colleagues, Google, [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/html/1704.04861v1), 2017 technical report.

**Core mechanism:** A standard convolution mixes spatial neighborhoods and channels simultaneously. MobileNet-v1 separates these jobs: a depthwise kernel filters each input channel, and a 1-by-1 pointwise convolution mixes channels. Width multiplier $`\alpha`$ reduces channel counts; resolution multiplier reduces spatial work. These controls offer a family of trade-offs rather than one network guaranteed to fit every device.

**Inputs/outputs and typical data types:** RGB images become class probabilities or compact backbone features. The ImageNet task has 1,000 classes; the released Slim evaluator uses a 1,001-logit indexing convention, so its label mapping must be preserved.

**Architecture diagram description:**

```text
RGB -> standard Conv3x3 -> depthwise/pointwise blocks -> global average -> head
block: depthwise 3x3 -> BN -> ReLU6 -> pointwise 1x1 -> BN -> ReLU6
       [released TensorFlow-Slim activation convention]
selected depthwise strides downsample the feature map
```

**Activation functions used and why:** The v1 paper describes ReLU, while the [released TensorFlow-Slim implementation](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) specifies clipped ReLU6 after normalized depthwise and pointwise operations. Softmax performs classification. These are identified conventions, not an assumption that every release is identical.

**Loss function(s):** Supervised image-class cross-entropy. Detection adaptations use additional localization and classification objectives.

**Optimization algorithm(s):** The original paper specifies RMSProp and asynchronous gradient descent, but does **not enumerate a complete independent learning-rate schedule**. This is the paper-level recipe, not a complete training manifest for the 2018 inference artifact. A subsequently published trainer's defaults do not, by themselves, establish how that particular checkpoint was trained.

**Regularization techniques:** Batch normalization and reduced image augmentation. The authors specifically use little or no weight decay on depthwise filters, and do not use auxiliary heads or label smoothing in this setup.

**Backpropagation considerations:** Gradients pass through the separate spatial and channel transformations. The very small depthwise parameter sets can be over-regularized. Aggressively reduced channel widths create representational bottlenecks that optimization alone cannot fix.

**Parameter count / scaling behavior:** The paper reports **4.2 million parameters** and **569 million multiply-adds**; the released checkpoint table gives **4.24 million parameters**. Pointwise costs scale approximately with $`\alpha^2`$, whereas depthwise costs scale approximately with $`\alpha`$; the first and last layers need separate accounting.

**Training paradigm:** Supervised ImageNet classification in the reference experiment. Transfer or distillation can produce differently trained MobileNet checkpoints and should be described as separate stages.

**Hardware/parallelism considerations:** Depthwise convolutions reduce arithmetic but may be memory-bound. Runtime kernels, operator fusion, quantization, batch size, and target hardware determine real latency. "Mobile" in the name does not establish battery savings on an unspecified phone.

**Strengths and limitations:** MobileNet supplies clear size and arithmetic controls. Lower width and resolution reduce accuracy, especially when fine details or many distinct channel features are required.

**Computational complexity / scalability notes:** A depthwise-separable layer costs $`O(BHW(q^2C_{\rm in}+C_{\rm in}C_{\rm out}))`$, versus $`O(BHWq^2C_{\rm in}C_{\rm out})`$ for standard convolution. The arithmetic ratio is approximately $`1/C_{\rm out}+1/q^2`$, not a guaranteed hardware speedup.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The official [2018-08-02 checkpoint bundle](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) lists **70.9% top-1 and 89.9% top-5 accuracy** for floating-point `MobileNet_v1_1.0_224`. Its [evaluator](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_eval.py) targets all **50,000 ImageNet validation images**, using a [single central crop resized to 224](https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py). This is not the paper's separate 70.6% record. Images become inexpensive spatial/channel features and a class decision for a possible on-device recognition interface. The authors motivate resource-constrained vision, but neither checkpoint scores nor arithmetic counts establish a named phone deployment or measured energy savings.

**Notable vendor implementations/libraries:** TensorFlow-Slim, Keras MobileNet, and mobile inference runtimes. Verify v1 versus v2/v3, activation conventions, resolution, width multiplier, and checkpoint evaluation protocol.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Generic CNN | Regular image grids | Shared local feature learning | Sensitive to resolution and distribution shift | Keras MNIST classification benchmark |
| LeNet | Small grayscale characters | Compact document-recognition features | Full documents require additional modules | CNN recognizers inside the published commercial cheque-reading system |
| AlexNet | Natural RGB photographs | Early successful large-scale learned features | Heavy dense head; historical implementation constraints | ILSVRC-2012, with single-model and ensemble results separated |
| VGG | Images requiring reusable hierarchical features | Simple repeated small-kernel design | Large parameter and compute footprint | ImageNet dense, single-scale evaluation |
| ResNet | Images and multistage visual backbones | Residual learning supports depth | Depth still increases latency and memory | ResNet-152 ImageNet validation benchmark |
| Inception | Images with multiscale structure | Parallel receptive fields with bottlenecks | Branch complexity and inference aggregation costs | GoogLeNet ILSVRC-2014 ensemble benchmark |
| EfficientNet | Images under a stated compute budget | Joint depth/width/resolution scaling | Hardware-dependent efficiency | B0 ImageNet single-crop benchmark |
| ConvNeXt | Images and dense-prediction backbones | Strong modern pure-convolutional design | Results depend on substantial training recipes | ConvNeXt-T ImageNet-1K benchmark |
| DenseNet | Images benefiting from feature reuse | Parameter-efficient concatenated features | Activation traffic and naive memory overhead | DenseNet-121 single-crop ImageNet benchmark |
| MobileNet | Resource-constrained visual inputs | Depthwise/pointwise factorization | Arithmetic savings need not equal latency savings | Explicitly versioned MobileNet-v1 ImageNet validation checkpoint |

## 1.7 Recurrent and sequence architectures

Sequence models differ in how they retain context, align inputs with targets, and expose parallelism. Predicting a translated sentence or an annotated phoneme sequence is supervised, even when the decoder predicts its output one token at a time.

### 1.7.1 Vanilla recurrent neural network (RNN)

**Name:** Vanilla RNN, with tanh recurrent units. The measured representative is the three-level bidirectional tanh network in a supervised speech-recognition comparison.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures; recurrent state without explicit memory gates.

**Originating paper/vendor/year:** Jeffrey Elman's [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1), 1990, is a canonical simple-recurrent-network reference, not the origin of every recurrent architecture. The supervised experiment here comes from Graves, Mohamed, and Hinton's [2013 speech-recognition study](https://arxiv.org/html/1303.5778v1).

**Core mechanism:** At each step, $`h_t=\tanh(W_xx_t+W_hh_{t-1}+b)`$. Reusing $`W_h`$ makes the state depend on the ordered history without adding new parameters for every time step. A bidirectional model also processes the sequence backwards and combines both representations; stacking recurrent layers adds depth across representations, distinct from recurrence across time.

**Inputs/outputs and typical data types:** Ordered acoustic feature vectors, sensor observations, or symbol embeddings become sequence labels or one final label. The speech instance uses 123-component acoustic vectors and phoneme-plus-blank output scores.

**Architecture diagram description:**

```text
x1 -> tanh state h1 -> tanh state h2 -> ... -> tanh state hT
                       ^ x2                         ^ xT
speech instance: forward + backward streams, stacked three levels
                -> frame softmax -> CTC decoding -> phoneme sequence
```

**Activation functions used and why:** Tanh bounds the recurrent state and supplies nonlinearity; softmax normalizes competing phoneme/blank scores. There are no sigmoid input, forget, or output gates.

**Loss function(s):** Connectionist temporal classification, or CTC, sums the probabilities of valid frame-level paths that collapse to the labelled phoneme sequence. It avoids requiring a fixed ground-truth alignment for every acoustic frame.

**Optimization algorithm(s):** In the cited comparison, SGD uses learning rate $`10^{-4}`$ and momentum 0.9, with weights initialized uniformly in [-0.1,0.1]. The reported recipe does not introduce a decay schedule for this rate.

**Regularization techniques:** Inputs are normalized using training-set statistics and a separate development set controls model selection. Crucially, this tanh model is trained **without** the Gaussian weight-noise phase used by most LSTM comparators, because it failed to learn with that noise.

**Backpropagation considerations:** Backpropagation through time multiplies recurrent Jacobians over many steps, leading to vanishing or exploding gradients. Clipping can limit an explosion but cannot restore forgotten information. Truncating the backward window further limits credit assignment; the full-sequence speech experiment should not be confused with a truncated streaming recipe.

**Parameter count / scaling behavior:** One simple recurrent layer has roughly $`dh+h^2+h`$ parameters before its output head. The paper's **CTC-3l-500h-tanh** model has **3.7 million weights**, with 500 hidden units per direction at each level.

**Training paradigm:** Supervised acoustic-to-phoneme sequence learning. The architecture can also predict text from text under a self-supervised objective; that is not this experiment.

**Hardware/parallelism considerations:** Batch examples and matrix operations parallelize, but successive recurrent states remain dependent. Bidirectionality requires future input and therefore prevents immediate causal output for the whole model.

**Strengths and limitations:** The recurrence is simple and parameter sharing supports variable lengths. Long-range dependencies, optimization sensitivity, and sequential execution are substantial limitations relative to gated or attention-based alternatives.

**Computational complexity / scalability notes:** A dense layer costs $`O(T(dh+h^2))`$ per sequence before the head. Full BPTT stores approximately $`O(BTLh)`$ states; CTC adds dynamic programming over input and target lengths, commonly $`O(TU)`$ for target length $`U`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On [TIMIT in the 2013 study](https://arxiv.org/html/1303.5778v1), this tanh network obtains **37.6% phoneme error rate** on the **24-speaker core test set**. Training uses the standard 462-speaker set without SA recordings; 50 speakers form the development set. Training/decoding uses 61 phoneme labels, mapped to 39 for scoring, with beam width 100. Acoustic frames become recurrent states, CTC paths, and a phoneme transcription. The roughly parameter-matched three-level LSTM obtains 18.6%, illustrating a concrete limitation rather than an assumed RNN success. Results are single runs with unknown initialization variance; no deployed speech-product KPI is claimed.

**Notable vendor implementations/libraries:** PyTorch `RNN`, Keras `SimpleRNN`, and accelerator recurrent kernels. Bidirectional wrappers and CTC losses are additional components, not automatic properties of every RNN.

### 1.7.2 Long short-term memory (LSTM)

**Name:** Long short-term memory network. The supervised representative is a stacked, bidirectional LSTM with CTC; its modern gated cell is distinguished from the first 1997 formulation.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures; gated additive memory.

**Originating paper/vendor/year:** Sepp Hochreiter and Jurgen Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), Neural Computation, 1997. Forget gates and other cell refinements followed. The worked instance is from [Graves, Mohamed, and Hinton, 2013](https://arxiv.org/html/1303.5778v1).

**Core mechanism:** A typical modern cell updates $`c_t=f_t\odot c_{t-1}+i_t\odot g_t`$ and exposes $`h_t=o_t\odot\tanh(c_t)`$. Input, forget, and output gates respectively control writing, retaining, and exposing memory. The additive cell path can preserve information and gradients when the forget gate remains near one. The speech paper additionally describes peephole connections from cell state to gates.

**Inputs/outputs and typical data types:** Acoustic frames, multivariate time series, or embeddings become a sequence of states and task outputs. The reference uses 123-component speech features and a CTC phoneme head.

**Architecture diagram description:**

```text
[x_t,h_(t-1)] -> sigmoid gates i,f,o; tanh candidate g
c_(t-1) -- multiply f --+-- add i*g --> c_t -- tanh -- multiply o --> h_t
speech: three bidirectional LSTM levels -> softmax -> CTC -> phonemes
```

**Activation functions used and why:** Sigmoid gates lie between zero and one and act multiplicatively; tanh supplies bounded candidate and exposed state values. The cell state itself has an additive update, not simply another tanh recurrence.

**Loss function(s):** CTC negative log-likelihood for the labelled phoneme sequence. An LSTM trained with frame-aligned cross-entropy or a recurrent transducer has a different objective.

**Optimization algorithm(s):** The 2013 CTC setup uses SGD at $`10^{-4}`$ with momentum 0.9. The reported rate is not replaced here by the Adam recipe common in later tutorials.

**Regularization techniques:** Training-set feature normalization, development-set stopping, and a second training phase with Gaussian **weight noise of standard deviation 0.075**. The noise phase starts from the best development log-probability point of the initial noiseless training.

**Backpropagation considerations:** BPTT differentiates through gates and the cell path. The latter improves long-range credit assignment, but gate saturation, exploding states, and truncated histories remain possible. Bidirectional gradients depend on both preceding and following input.

**Parameter count / scaling behavior:** A basic one-direction cell has roughly $`4h(d+h+1)`$ parameters, with additional terms for peepholes or separate biases. The paper's **CTC-3l-250h** has **3.8 million weights**, not the parameter count of every LSTM.

**Training paradigm:** Supervised sequence transcription from labelled utterances. No unlabelled-text language-model pretraining is required for this CTC result.

**Hardware/parallelism considerations:** Gate matrix operations can be fused, but time-step dependencies remain. Multiple layers and bidirectionality increase activation storage. Offline bidirectional recognition should not be advertised as an equivalent low-latency streaming system.

**Strengths and limitations:** Gated memory is effective when relevant information must survive many steps. It costs more operations per step than a simple RNN and remains less parallel across time than teacher-forced Transformer training.

**Computational complexity / scalability notes:** Per sequence, matrix work is $`O(T\,4h(d+h))`$ for one layer, plus output and CTC costs. The factor four is an approximate gate-operation account; actual kernels, projections, directions, and cell variants alter constants.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 1 of the TIMIT study](https://arxiv.org/html/1303.5778v1) reports **18.6% core-test phoneme error rate** for **CTC-3l-250h**, compared with the 3.7-million-weight tanh model's 37.6%. The split, 61-to-39 label mapping, and beam-100 protocol match the preceding entry, but weight-noise treatment differs. Audio frames enter gated bidirectional layers and produce a decoded phoneme sequence for an acoustic transcription system. The often-quoted **17.7%** belongs to **PreTrans-3l-250h**, a pretrained recurrent-transducer system, not this CTC network. The paper reports single runs, so the observed gap does not provide a confidence interval or establish production speech accuracy.

**Notable vendor implementations/libraries:** PyTorch `LSTM`, Keras `LSTM`, and cuDNN-compatible kernels. Common library cells omit the historical peepholes, so operator names alone do not reproduce the speech architecture.

### 1.7.3 Gated recurrent unit (GRU)

**Name:** Gated recurrent unit; the original encoder-decoder formulation introduced by Cho and colleagues.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures; compact gated recurrence.

**Originating paper/vendor/year:** Kyunghyun Cho and colleagues, [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/html/1406.1078v3), EMNLP 2014. The paper calls its component a new gated hidden unit; GRU became the standard name.

**Core mechanism:** A reset gate controls which previous-state components contribute to a candidate, and an update gate blends candidate and old state. Under one convention, $`\tilde h_t=\tanh(Wx_t+U(r_t\odot h_{t-1}))`$, $`h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t`$. Unlike an LSTM, it has no separate exposed hidden state and persistent cell state. Encoder and decoder reuse this mechanism to score a target phrase conditioned on a source phrase.

**Inputs/outputs and typical data types:** Token sequences or numerical time series become states, sequence predictions, or conditional phrase scores. The original translation system uses learned word embeddings and a probability over target words.

**Architecture diagram description:**

```text
source phrase -> embedding -> GRU encoder -> phrase summary
                                                |
target prefix -> embedding -> GRU decoder -> maxout/readout -> next-word softmax
GRU cell: reset-gated candidate + update-gated carry -> new hidden state
```

**Activation functions used and why:** Sigmoid reset/update gates, a tanh candidate, and softmax output. The paper's decoder readout contains 500 maxout units pooling pairs of inputs; that readout is not an intrinsic part of every GRU cell.

**Loss function(s):** Conditional sequence negative log-likelihood for labelled source/target phrase pairs. The learned conditional score becomes an additional feature in a phrase-based translation system.

**Optimization algorithm(s):** Minibatch stochastic updates use Adadelta with $`\rho=0.95`$, $`\epsilon=10^{-6}`$, and 64 phrase pairs per update. Adadelta supplies an adaptive step mechanism; the paper does not prescribe a conventional fixed-rate decay schedule for this run.

**Regularization techniques:** Low-rank word input/output mappings constrain capacity; recurrent matrices are initialized using orthogonal directions obtained from singular vectors. The cited recipe should not acquire undocumented dropout merely because modern GRU libraries offer it.

**Backpropagation considerations:** BPTT passes through both gate-controlled paths. Libraries differ over whether reset multiplication occurs before or after a recurrent affine transformation; biases make these genuinely different parameterizations. Gate conventions can also swap the meanings of $`z`$ and $`1-z`$.

**Parameter count / scaling behavior:** The source uses 1,000 hidden units in each encoder and decoder and rank-100 embedding mappings. A simple one-direction GRU has approximately $`3h(d+h+1)`$ cell parameters, before vocabulary embeddings and readout; therefore hidden width alone is not a full checkpoint count.

**Training paradigm:** Supervised parallel-phrase learning. Additional monolingual language-model features reported in other rows of the translation experiment are separate components.

**Hardware/parallelism considerations:** Fewer gate matrices than a standard LSTM can reduce work and memory, but wall-clock gains depend on fused kernels. Recurrence still imposes a sequential critical path.

**Strengths and limitations:** GRUs offer useful memory control with a comparatively compact cell. They retain the fixed-summary bottleneck when used in an encoder-decoder without attention and do not universally outperform LSTMs.

**Computational complexity / scalability notes:** A dense cell costs $`O(T\,3h(d+h))`$ per sequence, plus vocabulary/readout computation. Full-softmax output can be significant for large vocabularies even when the recurrent cell is compact.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On the [paper's WMT-2014 English-to-French **newstest2014** evaluation](https://arxiv.org/html/1406.1078v3), adding the gated encoder-decoder phrase score raises the phrase-based baseline's **test BLEU from 33.30 to 33.87**; development scores are separately reported as 30.64 and 31.20. A source phrase and candidate French phrase enter the model, which supplies a conditional score to the larger decoder's decision among translations. This is **not 33.87 BLEU from a standalone GRU-only end-to-end translator**, nor the higher-scoring row with an additional language model. The documented goal is better phrase representations; business localization quality and translator productivity were not measured.

**Notable vendor implementations/libraries:** PyTorch `GRU`, Keras `GRU`, and recurrent translation toolkits. Record reset placement, bias layout, bidirectionality, and readout structure when porting weights.

### 1.7.4 Sequence-to-sequence with Bahdanau attention

**Name:** Attentional sequence-to-sequence, specifically Bahdanau additive attention in the original **RNNsearch** translation model.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures; learned source-target alignment.

**Originating paper/vendor/year:** Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/html/1409.0473v7), 2014 preprint and ICLR 2015 paper.

**Core mechanism:** A bidirectional encoder supplies one annotation $`h_j`$ per source position. At target step $`i`$, an additive scorer evaluates compatibility, for example $`e_{ij}=v^\top\tanh(W_s s_{i-1}+W_hh_j)`$. Softmax across source positions gives $`\alpha_{ij}`$, and $`c_i=\sum_j\alpha_{ij}h_j`$ becomes the decoder's context. Instead of compressing the entire source into one fixed vector, the decoder repeatedly retrieves different weighted source information.

**Inputs/outputs and typical data types:** A source-language token sequence becomes a target-language sequence, with soft alignment weights as intermediate quantities. Alignment weights are model computations, not guaranteed human explanations.

**Architecture diagram description:**

```text
source tokens -> forward/backward GRU encoder -> h1,h2,...,hS
                                                   |
previous decoder state -> additive scores -> softmax weights -> context c_i
target prefix + c_i -> GRU decoder -> maxout/readout -> target-word distribution
```

**Activation functions used and why:** Sigmoid/tanh gated recurrent units, tanh in the additive alignment network, softmax over source positions, and a maxout-based output network followed by vocabulary softmax.

**Loss function(s):** Conditional target-sequence cross-entropy, summing the negative log-probability of the reference translation's tokens. No ground-truth word alignment is needed.

**Optimization algorithm(s):** The source uses minibatch SGD with Adadelta, $`\rho=0.95,\epsilon=10^{-6}`$, and 80 sentences per update. The longer RNNsearch-50 run continues until development performance stops improving; it is a separately identified result rather than a universal fixed-epoch schedule.

**Regularization techniques:** Vocabulary restrictions, initialization, and development monitoring constrain the reported training. The paper's sentence-length restrictions define training variants; they should not be mistaken for a general regularizer or an evaluation exclusion.

**Backpropagation considerations:** Both recurrent BPTT and gradients through the soft alignment weights are required; the training appendix clips global gradient norm at one. Attention opens shorter routes to source annotations, but recurrence remains. A hard argmax alignment would remove ordinary gradients through position selection.

**Parameter count / scaling behavior:** RNNsearch uses 1,000 units in each encoder direction and 1,000 decoder units. The complete parameter count additionally depends strongly on vocabulary embeddings, the alignment network, and maxout readout; it is not inferred from those widths alone.

**Training paradigm:** Supervised learning from bilingual parallel sentences, with teacher-forced target prefixes during training and beam search during inference. This is not masked-token or next-token pretraining on an unpaired corpus.

**Hardware/parallelism considerations:** Encoder annotations can be cached, and their alignment projections reused. Decoder steps remain sequential; beam search multiplies state and scoring work.

**Strengths and limitations:** Content-dependent context alleviates the fixed-vector bottleneck, especially on longer sentences. Word-level vocabulary limits still create unknown-token failures; attention does not guarantee factual or semantically faithful translation.

**Computational complexity / scalability notes:** For source length $`S`$, target length $`U`$, and alignment width $`a`$, computing all alignment scores costs approximately $`O(SUa)`$ after reusable projections, in addition to recurrent and vocabulary-output costs. It is incorrect to charge only this alignment term as the entire translator.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 1](https://arxiv.org/html/1409.0473v7) reports **26.75 BLEU** for **RNNsearch-50** versus **17.82** for **RNNencdec-50** on **all 3,003 WMT-2014 English-to-French news-test-2014 sentences**. The longer-trained starred model reaches 28.45; the separate "No UNK" subset must not replace the full-set result. A source sentence supplies positionwise annotations; while choosing a French word, the decoder weights relevant source positions, predicts a distribution, and extends a beam of candidate translations. Addressing long-sentence compression is the authors' documented motivation. BLEU is a reference-overlap metric, not a demonstrated improvement in commercial translation acceptance or human productivity.

**Notable vendor implementations/libraries:** Additive-attention layers in Keras and historical recurrent NMT toolkits implement the mechanism. A generic attention wrapper does not automatically reproduce RNNsearch's encoder, gates, vocabulary, or maxout head.

### 1.7.5 Original encoder-decoder Transformer

**Name:** The original Transformer for supervised sequence transduction, including base and big translation configurations.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures; non-recurrent attention-based encoder-decoder. Placement in this category is about sequence tasks, not an assertion that the Transformer is recurrent.

**Originating paper/vendor/year:** Ashish Vaswani and colleagues, [Attention Is All You Need](https://arxiv.org/html/1706.03762v7), NeurIPS 2017. The cited arXiv revision retains the historical translation experiments.

**Core mechanism:** Multi-head attention forms $`\operatorname{softmax}(QK^\top/\sqrt{d_k})V`$, allowing each position to aggregate other positions' representations. The encoder uses unrestricted source self-attention; the decoder uses causal target self-attention and cross-attention to the encoder. Positionwise feed-forward networks transform each token, while residual connections and layer normalization stabilize the stack. Sinusoidal positional encodings supply order that attention alone would not know.

**Inputs/outputs and typical data types:** Tokenized parallel sentences become conditional target-token distributions and decoded translations. Padding and causal masks serve different purposes.

**Architecture diagram description:**

```text
source -> embedding + position -> [self-attention -> FFN] x6 -> encoder states
target prefix -> embedding + position
              -> [masked self-attention -> cross-attention to source -> FFN] x6
              -> vocabulary projection -> softmax -> next target token
each sublayer: residual addition and original post-layer-normalization
```

**Activation functions used and why:** Attention softmax, ReLU in the two-layer feed-forward network, and output softmax. The original does not use the GELU/SwiGLU choices common in later pretrained families.

**Loss function(s):** Supervised target-token cross-entropy with label smoothing 0.1. The target tokens are supplied translations conditioned on a source sentence.

**Optimization algorithm(s):** Adam with $`\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}`$. The rate is $`d^{-1/2}\min(t^{-1/2},t\,4000^{-3/2})`$: 4,000 warmup steps, then inverse-square-root decay.

**Regularization techniques:** Residual/embedding dropout, layer normalization, and label smoothing. The base dropout rate is 0.1; the big English-German configuration uses 0.3. Final translation models average recent checkpoints, not predictions from independently trained ensembles.

**Backpropagation considerations:** Teacher forcing permits parallel computation over target positions during training despite causal masking. Residual paths reduce depth-related optimization difficulties. Autoregressive inference still depends on previously generated tokens.

**Parameter count / scaling behavior:** Base uses width 512, feed-forward width 2,048, eight heads, and about **65 million parameters**. Big uses width 1,024, feed-forward width 4,096, sixteen heads, and about **213 million**. Both have six encoder and six decoder layers.

**Training paradigm:** **Supervised parallel-corpus translation.** BERT, GPT, and T5's base pretraining objectives are treated separately in [foundation models](06-foundation-models.md); a causal decoder does not by itself make the original Transformer experiment unsupervised.

**Hardware/parallelism considerations:** Training attention and feed-forward operations use large parallel matrix products; the reported experiments used eight P100 GPUs. Cached keys/values reduce repeated decoding work but consume memory and do not eliminate sequential token generation.

**Strengths and limitations:** Direct interactions shorten information paths and training parallelizes well. Attention, vocabulary projections, and feed-forward blocks remain expensive, and translation quality is not guaranteed outside the measured domain.

**Computational complexity / scalability notes:** At length $`T`$, attention interactions cost $`O(T^2d)`$, **but projections cost $`O(Td^2)`$ and the FFN costs $`O(Tdf)`$** for FFN width $`f`$. Encoder-decoder cross-attention adds $`O(SUd)`$. Naively stored attention probabilities use quadratic memory; fused implementations can reduce stored intermediates without removing the dense interaction arithmetic.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original translation table](https://arxiv.org/html/1706.03762v7) reports **27.3 BLEU for base and 28.4 for big** on **WMT-2014 English-to-German newstest2014**. Evaluation uses beam width four and length penalty 0.6; base averages five checkpoints and big twenty. A German candidate token attends to its English source and earlier German tokens, then beam search selects the translation sequence. The authors' rationale is parallelizable sequence modelling without recurrence. These are supervised research results, not evidence of a particular commercial translation service's architecture or customer impact.

**Notable vendor implementations/libraries:** Tensor2Tensor historically, PyTorch `Transformer`, and encoder-decoder translation toolkits. Default normalization order, tokenization, weight tying, and positional representations must be checked.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Vanilla RNN | Ordered features and short-context sequences | Simple recurrent state sharing | Difficult long-range optimization | TIMIT tanh-RNN comparator, with a reported high phoneme error rate |
| LSTM | Speech and longer-context time series | Controlled additive memory | Sequential time dependency and gate cost | Bidirectional LSTM-CTC on TIMIT |
| GRU | Token sequences and compact recurrent models | Gated state with fewer cell components | Recurrence and fixed-summary bottlenecks | Phrase scoring inside the WMT-2014 translation system |
| Seq2Seq with Bahdanau attention | Paired variable-length sequences | Learns soft source-target alignment | Recurrent decoding and vocabulary limits | RNNsearch English-to-French benchmark |
| Original Transformer | Parallel text and other sequence transduction | Parallel training with direct attention paths | Attention plus projections/FFNs are costly | Supervised WMT-2014 English-to-German translation |

## 1.8 Vision transformers and prediction architectures

The next six entries distinguish a backbone from a prediction system. ViT and Swin primarily specify feature encoders. U-Net predicts a spatial label field; Faster R-CNN, YOLO, and DETR predict objects and their locations. Their scores measure different tasks and must not be placed on one undifferentiated "accuracy" scale.

### 1.8.1 Vision Transformer (ViT)

**Name:** Vision Transformer; the worked checkpoint family is **ViT-L/16 with supervised ImageNet-21k pretraining** and ImageNet-1K fine-tuning.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures; global patch-token image encoders.

**Originating paper/vendor/year:** Alexey Dosovitskiy and colleagues, Google Research, [An Image is Worth 16x16 Words](https://arxiv.org/html/2010.11929v2), 2020 preprint and ICLR 2021 paper.

**Core mechanism:** The image is divided into fixed-size patches, each flattened and linearly embedded. A learned classification token and positional embeddings join the sequence. A Transformer encoder repeatedly mixes information globally through self-attention and transforms features through MLPs. The class-token representation feeds a classifier. Unlike convolution, attention weights depend on the image content; unlike a hierarchical backbone, original ViT largely keeps one patch resolution.

**Inputs/outputs and typical data types:** RGB images become class predictions or patch representations. "L/16" specifies the Large configuration and 16-by-16 patches, not a 16-layer network.

**Architecture diagram description:**

```text
image -> 16x16 patches -> linear patch embeddings + learned positions
      -> prepend class token -> 24 encoder blocks -> class-token readout -> label
block: LayerNorm -> multi-head attention -> residual
       LayerNorm -> MLP/GELU -> residual
```

**Activation functions used and why:** GELU in encoder MLPs and softmax within attention. The fine-tuned classifier produces class logits. Its new classification head must match the downstream label set.

**Loss function(s):** Cross-entropy against supplied image-category targets; the [released training code](https://github.com/google-research/vision_transformer/blob/main/vit_jax/train.py) explicitly computes target-weighted negative log-softmax. The downstream ImageNet-1K task is single-label classification. Masked-patch reconstruction is not the quoted checkpoint's objective.

**Optimization algorithm(s):** For ImageNet-21k, the paper specifies Adam, learning rate $`10^{-3}`$, batch size 4096, 10,000-step warmup, and linear decay. Fine-tuning uses SGD with momentum 0.9, batch size 512, and cosine decay; a learning-rate grid is selected for the downstream task rather than a universal fine-tuning rate.

**Regularization techniques:** The appendix's **ImageNet-21k** recipe specifies weight decay **0.03** and dropout **0.1**; its JFT recipe uses different values. Fine-tuning uses no weight decay and clips global gradient norm at one. The high-resolution ImageNet runs also use parameter averaging.

**Backpropagation considerations:** Pre-normalized residual blocks assist optimization. Fine-tuning at a different resolution requires interpolating patch positional embeddings while treating the class token separately. That is a model adaptation, not simply reshaping an arbitrary learned sequence.

**Parameter count / scaling behavior:** The paper lists **307 million parameters for ViT-L**, with 24 layers, width 1,024, MLP width 4,096, and sixteen heads. Base and Huge are approximately 86 million and 632 million, respectively.

**Training paradigm:** Supervised large-label-set pretraining followed by supervised task adaptation. See [unsupervised neural models](05-unsupervised-neural.md) for masked-image and other self-supervised Transformer objectives.

**Hardware/parallelism considerations:** Dense attention and MLPs exploit accelerators, but high-resolution fine-tuning greatly increases token count. Reported TPU training resources are specific to the experiments and not general inference requirements.

**Strengths and limitations:** ViT learns global interactions with few image-specific architectural assumptions. That flexibility can require more data than a convolutional prior, and the original uniform-resolution representation is inconvenient for some dense tasks.

**Computational complexity / scalability notes:** With patch side $`P`$, token count is approximately $`N=HW/P^2`$. A block costs $`O(N^2d+Nd^2+Ndf)`$. Halving $`P`$ quadruples tokens and increases the quadratic attention term sixteenfold, while linear-in-token terms grow fourfold.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 2](https://arxiv.org/html/2010.11929v2) reports **85.30% +/- 0.02 top-1 ImageNet accuracy** for **ViT-L/16 pretrained on ImageNet-21k**, then fine-tuned for ImageNet-1K; the stated deviation is across three fine-tuning runs, not an ensemble gain. The ImageNet evaluation uses the held-out validation benchmark and the paper's higher-resolution **512-pixel** ViT-L fine-tuning setting. Patch tokens from a photograph exchange information, and the class token selects an object label. The authors investigate whether scale can compensate for weaker convolutional bias. The separate JFT-pretrained Huge result of 88.55% is not this checkpoint, and neither result establishes a deployed visual-search KPI.

**Notable vendor implementations/libraries:** Google's `vision_transformer` release, timm, and torchvision. A self-supervised or differently pretrained ViT should be identified by weights and objective, not just architecture size.

### 1.8.2 Swin Transformer

**Name:** Swin Transformer, specifically **Swin-Tiny from the original 2021 family**.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures; hierarchical shifted-window attention.

**Originating paper/vendor/year:** Ze Liu and colleagues, Microsoft Research and collaborators, [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/html/2103.14030v2), ICCV 2021.

**Core mechanism:** Attention is restricted to small nonoverlapping windows. The next block shifts the partition so that tokens can exchange information across former window boundaries. Patch-merging layers reduce spatial resolution and increase channel width, producing a multiscale hierarchy suitable for both classification and dense prediction. A mask prevents cyclic shifting from creating spurious wraparound connections.

**Inputs/outputs and typical data types:** Images become hierarchical feature maps and, with a classification head, class probabilities. Detection and segmentation frameworks add their own heads and losses.

**Architecture diagram description:**

```text
image -> 4x4 patch embedding -> stage1 -> merge -> stage2 -> merge -> stage3
      -> merge -> stage4 -> global pooling -> classifier
within stages: window attention -> MLP -> shifted-window attention -> MLP
               [LayerNorm and residual paths around each sublayer]
```

**Activation functions used and why:** GELU in the feed-forward MLP, softmax inside each attention window, and learned relative-position biases in attention scores. Biases express geometry; they are not another activation.

**Loss function(s):** Supervised class cross-entropy with the classification training recipe's label smoothing and augmented targets. Detector or segmenter results require separate task objectives.

**Optimization algorithm(s):** For ImageNet-1K, AdamW with initial rate 0.001, weight decay 0.05, batch size 1024, and 300 epochs. Twenty warmup epochs precede cosine decay.

**Regularization techniques:** Layer normalization, stochastic depth, image augmentation, and the source's DeiT-style training controls. The paper specifically excludes repeated augmentation and EMA from this 1K setup because they did not improve it.

**Backpropagation considerations:** Gradients traverse residual blocks and masked window interactions. Correct padding and cyclic-shift masking are necessary for both forward semantics and gradients; omitting masks creates a different model.

**Parameter count / scaling behavior:** Swin-T has approximately **28 million parameters**, stage depths [2,2,6,2], and initial width 96. Patch merging roughly doubles channels while quartering token count between stages.

**Training paradigm:** The reference result is supervised ImageNet-1K training. Other published Swin results use supervised ImageNet-22K pretraining; self-supervised Swin-based encoders form a separate training category.

**Hardware/parallelism considerations:** Shared key sets within windows improve locality, but partitioning, shifting, masking, and padding add overhead. Different window sizes and resolutions can alter throughput even with similar arithmetic counts.

**Strengths and limitations:** Swin supplies a scalable multiscale backbone without global attention at every layer. Long-range communication requires successive shifted blocks and hierarchy; a single local window does not see the whole image.

**Computational complexity / scalability notes:** For $`N`$ tokens and window side $`M`$, attention interactions cost $`O(NM^2d)`$, not $`O(N^2d)`$. Projections and MLPs still cost $`O(Nd^2+Ndf)`$. "Linear in image size" assumes fixed window size and channel widths; it is not independence from resolution or model width.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original paper](https://arxiv.org/html/2103.14030v2) reports **81.3% top-1 ImageNet-1K validation accuracy** for **Swin-T at 224-by-224**, trained on ImageNet-1K, compared with 79.8% for the listed DeiT-S configuration. An image is progressively partitioned, locally mixed, shifted, and merged before a pooled label decision. The documented motivation is an efficient hierarchical backbone that can also support high-resolution tasks. The paper's stronger detection and 22K-pretrained results involve additional systems or training data and are not attributes of this Tiny classification run. No Microsoft-product deployment is inferred.

**Notable vendor implementations/libraries:** Microsoft's released Swin code, timm, torchvision, and OpenMMLab integrations. Swin V2, detector backbones, and classification checkpoints have different configuration requirements.

### 1.8.3 U-Net

**Name:** U-Net; the original two-dimensional, valid-convolution biomedical segmentation architecture.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures; convolutional encoder-decoder dense segmentation.

**Originating paper/vendor/year:** Olaf Ronneberger, Philipp Fischer, and Thomas Brox, University of Freiburg, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/html/1505.04597v1), MICCAI 2015.

**Core mechanism:** A contracting path extracts contextual features while reducing resolution. An expanding path upsamples those features and concatenates appropriately cropped, high-resolution encoder features. This combines context with localization rather than classifying every pixel using an independently evaluated sliding window. The original uses valid convolutions, so the predicted tile is smaller than its input; overlap-tile inference and reflected border context cover large images.

**Inputs/outputs and typical data types:** Microscopy images become per-pixel class probabilities and segmentation masks. Three-dimensional medical-volume variants are important extensions but are not the original measured model.

**Architecture diagram description:**

```text
572x572 image -> [64 -> 128 -> 256 -> 512 -> 1024] encoder channels
                  |      |      |      |        |
                  +------ cropped skip features-+--> expanding decoder
decoder: up-convolution -> concatenate skip -> two 3x3/ReLU convolutions
       -> 1x1 class projection -> 388x388 probability map
```

**Activation functions used and why:** ReLU follows the feature convolutions. A pixelwise softmax represents competing segmentation classes. Upsampling and concatenation recover spatial detail but do not themselves classify pixels.

**Loss function(s):** Weighted pixelwise cross-entropy. The weighting emphasizes class balance and separating borders between touching objects, using distances to the nearest two object boundaries. This is not the Dice loss often used in later U-Net implementations.

**Optimization algorithm(s):** The source uses SGD with **momentum 0.99** and a batch of one large image tile. The main paper does not give a complete numerical learning-rate schedule; reproducing a particular run requires its solver configuration rather than assuming modern Adam defaults.

**Regularization techniques:** Strong elastic deformations and image transformations compensate for limited annotated data. Initialization is chosen for rectified layers. The original architecture should not automatically acquire modern batch normalization or a padded-convolution geometry.

**Backpropagation considerations:** Pixelwise gradients flow through both context and skip paths. Valid-convolution cropping must align spatial locations precisely. Boundary weighting increases gradients at narrow separation regions, making label quality and class weighting consequential.

**Parameter count / scaling behavior:** A standard reconstruction of the displayed 64-base-channel, two-class graph is about **31 million parameters**, calculated from its layer dimensions. Different base widths, dimensionality, padding choices, and class heads alter the count.

**Training paradigm:** Supervised learning from manually annotated segmentation masks, not unsupervised discovery of cells. Pseudo-label or consistency-based U-Nets are separate semi-supervised formulations.

**Hardware/parallelism considerations:** High-resolution feature maps and retained skips dominate training memory. Overlap-tile inference trades redundant border computation for processing images larger than accelerator memory.

**Strengths and limitations:** U-Net combines local detail and wider context effectively with limited labels. Domain shift between microscopes or tissues, imperfect boundaries, and class imbalance can undermine downstream scientific or clinical use.

**Computational complexity / scalability notes:** Sum convolutional work across encoder and decoder resolutions. Skip storage scales with $`B\sum_\ell H_\ell W_\ell C_\ell`$; moving from 2D to 3D adds a depth dimension and substantially increases memory and compute.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On the [ISBI electron-microscopy segmentation challenge](https://arxiv.org/html/1505.04597v1), the Freiburg study uses 30 annotated 512-by-512 training images of Drosophila neuronal tissue and submits predictions for the hidden-label test set. Averaging **seven rotated inputs**, U-Net reports **warping error 0.0003529**, compared with 0.000420 for the cited sliding-window CNN submission. This is a topology-sensitive challenge metric, **not pixel accuracy, Dice, or IoU**. Images become membrane probability maps; thresholded boundaries support separating neuronal structures for scientific tracing. The authors motivate shared dense computation and boundary-sensitive learning. The experiment does not establish clinical diagnostic safety or laboratory labor savings.

**Notable vendor implementations/libraries:** Freiburg's original release and U-Net implementations in MONAI, nnU-Net, Keras, and PyTorch ecosystems. nnU-Net is a broader self-configuring pipeline, not merely the original 2015 network.

### 1.8.4 Faster R-CNN

**Name:** Faster R-CNN; the original region-proposal-network detector, with a VGG-16 backbone in the worked example.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures; two-stage object detection with shared visual features.

**Originating paper/vendor/year:** Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun, [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/html/1506.01497v3), NeurIPS 2015, with a later expanded manuscript.

**Core mechanism:** A convolutional backbone computes one shared feature map. A region proposal network, or RPN, predicts objectness and coordinate adjustments for anchors at each location. Selected proposals feed a Fast R-CNN head through region-of-interest pooling, which classifies and refines each region. Shared features avoid recomputing a full CNN for every candidate crop.

**Inputs/outputs and typical data types:** Images and supervised object boxes/classes become a variable-length list of detections. Background is represented explicitly during training.

**Architecture diagram description:**

```text
image -> shared VGG feature map --+-> RPN: anchors -> objectness/box offsets
                                |                    -> proposal selection/NMS
                                +-> RoI pooling <-------------+
                                    -> region head -> class + refined box
```

**Activation functions used and why:** ReLU in backbone and region layers; softmax for objectness/class decisions; linear box-regression outputs. Modern feature-pyramid and normalized backbones are separate variants.

**Loss function(s):** RPN binary objectness loss plus positive-anchor smooth-L1 regression; detector multiclass cross-entropy plus class-conditioned box regression. Matching rules determine which anchors and regions receive positive or background supervision.

**Optimization algorithm(s):** The paper's principal experiments use alternating RPN/detector training. Its PASCAL RPN schedule uses SGD at 0.001 for 60,000 minibatches and 0.0001 for 20,000 more, momentum 0.9, and weight decay 0.0005. Detector fine-tuning has its own Fast R-CNN settings.

**Regularization techniques:** Labelled ImageNet initialization, weight decay, image transformations, and sampled positive/background anchors and regions. Balancing sampled examples is important because background candidates vastly outnumber objects.

**Backpropagation considerations:** Losses update shared visual features, but proposal selection, NMS, and original RoI pooling are not a fully smooth geometry pipeline. The paper distinguishes alternating training from approximate joint training, which treats proposal coordinates as fixed for the detector's backward pass.

**Parameter count / scaling behavior:** Total parameters depend on backbone and region head. For a 512-channel feature map, the shared 3-by-3 RPN layer and nine-anchor prediction heads add approximately **2.4 million parameters**, calculated from those dimensions; that is not the whole detector.

**Training paradigm:** Supervised bounding-box and category learning after supervised backbone pretraining. Automatically proposed regions are not unlabelled pseudo-targets.

**Hardware/parallelism considerations:** Backbone computation is shared, while the region head scales with the retained proposal count. NMS, resizing, and proposal transfers can become latency bottlenecks even when convolution kernels are fast.

**Strengths and limitations:** Two stages permit focused region classification and localization. Anchors, matching thresholds, proposal budgets, and postprocessing introduce complexity, and crowded or small objects remain difficult.

**Computational complexity / scalability notes:** Cost includes the backbone, dense RPN over feature locations and anchors, and $`R`$ region-head evaluations. NMS can be $`O(R^2)`$ in a straightforward implementation. A small RPN does not make the full system constant-time in image resolution or proposal count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [VGG-based experiment](https://arxiv.org/html/1506.01497v3) reports **73.2% mAP on PASCAL VOC 2007 test**, trained on **VOC 2007 plus 2012 trainval**, using **300 RPN proposals** at test time. This is VOC's IoU-0.5 detection evaluation, not COCO's averaged-IoU AP. An image generates shared features and object proposals; region classification and box refinement produce labelled locations for objects such as people, vehicles, or animals. The authors' documented rationale is replacing external proposal computation while sharing image features. It is not evidence that a particular surveillance or vehicle product deploys this detector.

**Notable vendor implementations/libraries:** Original Caffe releases, Detectron2, torchvision detection, and MMDetection. FPN, RoIAlign, and newer backbones change both the graph and its benchmark identity.

### 1.8.5 YOLO, original version

**Name:** **YOLOv1**, the original "You Only Look Once" detector; later YOLO versions are not interchangeable with this entry.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures; single-stage grid-based object detection.

**Originating paper/vendor/year:** Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi, [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/html/1506.02640v5), 2015 preprint and CVPR 2016 paper.

**Core mechanism:** One network predicts a grid of object hypotheses directly from the full image. In the VOC configuration, each of 7-by-7 cells predicts two boxes and a shared distribution over 20 classes. A cell is responsible for an object whose center falls within it. Each box has coordinates and confidence, with confidence intended to combine object presence and overlap quality. There is no learned region-proposal stage.

**Inputs/outputs and typical data types:** A fixed-size RGB image becomes class-scored bounding boxes. The original detector trains at 448-by-448 resolution.

**Architecture diagram description:**

```text
448x448 RGB -> 24 convolutional layers with pooling -> two dense layers
           -> 7x7x(2*5 + 20) predictions
           -> class-confidence scores -> threshold/NMS -> labelled boxes
```

**Activation functions used and why:** Hidden layers use leaky ReLU with negative slope 0.1; the final prediction layer is linear. Later sigmoid-based, anchor-based, or anchor-free YOLO heads must not be substituted into the v1 description.

**Loss function(s):** Weighted sum-squared errors for coordinates, confidence, and classes. Coordinate weight is five; no-object confidence weight is 0.5. Width and height are compared through square roots, reducing large boxes' dominance. This objective is not identical to optimizing mAP.

**Optimization algorithm(s):** SGD with momentum 0.9, weight decay 0.0005, and batch size 64. The paper warms the rate from $`10^{-3}`$ toward $`10^{-2}`$, then describes 75 epochs at $`10^{-2}`$, 30 at $`10^{-3}`$, and 30 at $`10^{-4}`$.

**Regularization techniques:** Dropout 0.5 after the first dense layer, random scaling/translations, and exposure/saturation changes. Classification pretraining initializes convolutional features before detection fine-tuning.

**Backpropagation considerations:** High early learning rates can destabilize coordinate predictions, motivating warmup. Assignment to the highest-overlap responsible box and NMS involve discrete decisions; ordinary gradients train the selected prediction losses, not an idealized smooth detector.

**Parameter count / scaling behavior:** Parameters are the sum of convolutional kernels and dense matrices. A dense mapping from flattened width $`F`$ to hidden width $`h`$ alone has $`Fh+h`$ parameters, so the fixed-grid dense head can be expensive. No unchecked "YOLO-family parameter count" is assigned to this historical graph.

**Training paradigm:** Supervised image classification pretraining followed by supervised box/category learning. There is no self-supervised language-model objective in the detector.

**Hardware/parallelism considerations:** One dense network pass exposes parallel work across the image. Preprocessing, NMS, camera acquisition, and hardware still determine end-to-end latency; historical paper FPS is not a guarantee on another device.

**Strengths and limitations:** The detector reasons with full-image context and avoids a separate proposal pipeline. Shared cell-level class predictions and limited boxes per cell make nearby small objects and crowded scenes difficult.

**Computational complexity / scalability notes:** Network work is the sum of convolution and dense-head costs at the configured resolution. Grid size controls the number of raw predictions; postprocessing adds candidate-dependent work. Changing input geometry may require changing the original dense head.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [The original paper](https://arxiv.org/html/1506.02640v5) reports **63.4% mAP on VOC 2007 test** for YOLOv1 trained on **VOC 2007+2012 trainval**. The smaller **Fast YOLO** is a different model with 52.7% mAP. A scene image produces 98 box hypotheses and cell-level class scores; confidence combination and NMS select labelled locations. The authors' motivation is unified, fast detection rather than separate region processing. My explanation of its trade-off is that a fixed coarse grid purchases simplicity at the expense of crowded-object representation. These are IoU-0.5 VOC results, not COCO AP or demonstrated road-safety performance.

**Notable vendor implementations/libraries:** The original Darknet implementation is the historical reference. Current Darknet and Ultralytics releases often implement different YOLO generations with different heads, objectives, training data, and parameter counts.

### 1.8.6 DETR

**Name:** Detection Transformer, or **DETR**, specifically the original ResNet-50-based 2020 model.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures; direct set prediction for object detection.

**Originating paper/vendor/year:** Nicolas Carion and colleagues, Facebook AI Research, [End-to-End Object Detection with Transformers](https://arxiv.org/html/2005.12872v3), ECCV 2020.

**Core mechanism:** A CNN supplies spatial features to a Transformer encoder. A decoder processes a fixed set of learned object queries, each producing a class and box. During training, Hungarian matching assigns ground-truth objects to predictions; unmatched predictions learn a no-object class. One-to-one matching discourages duplicate predictions, removing the need for the original pipeline to use anchors and NMS.

**Inputs/outputs and typical data types:** Images and annotated object sets become fixed-size prediction sets, with foreground confidence determining useful detections.

**Architecture diagram description:**

```text
image -> ResNet-50 -> projected spatial features + positions -> 6-layer encoder
100 learned object queries --------------------------------> 6-layer decoder
                                                              |
                                                  shared class/box heads
training: Hungarian assignment -> matched set loss + no-object supervision
```

**Activation functions used and why:** ReLU in feed-forward blocks, attention softmax, class softmax, and sigmoid-normalized box coordinates. Positional information distinguishes spatial locations and object queries.

**Loss function(s):** Matched classification loss plus L1 and generalized-IoU box losses; unmatched slots receive the downweighted no-object classification loss. Auxiliary losses at intermediate decoder layers aid training.

**Optimization algorithm(s):** AdamW, Transformer learning rate $`10^{-4}`$, backbone rate $`10^{-5}`$, and weight decay $`10^{-4}`$. The main comparison uses **500 epochs with a tenfold rate drop after epoch 400**; the 300-epoch/200-drop ablation recipe is different.

**Regularization techniques:** ImageNet-pretrained ResNet with frozen batch-normalization statistics, Transformer dropout, random image resizing/cropping, and gradient control in the released recipe. Longer training is a real cost, not free architectural capacity.

**Backpropagation considerations:** Matching is a discrete assignment; gradients flow through the selected differentiable classification and box losses, not through a continuously differentiable Hungarian solver. Intermediate decoder supervision helps align object queries earlier in training.

**Parameter count / scaling behavior:** The reference ResNet-50 DETR has **41 million parameters**, Transformer width 256, and 100 output queries. Increasing query count raises decoder and matching costs; increasing image-feature resolution raises encoder attention costs.

**Training paradigm:** Supervised detection after supervised backbone pretraining. Object queries are learned parameters, not unlabeled training examples or retrieval requests.

**Hardware/parallelism considerations:** Encoder and decoder matrix operations parallelize. Original DETR trains for a long schedule and attends globally over image features; dilating the backbone to increase feature resolution increases memory and computation.

**Strengths and limitations:** Set prediction simplifies postprocessing and handles large objects well in the original comparisons. Small-object performance and slow convergence are weaknesses; later deformable variants address different trade-offs and are separate architectures.

**Computational complexity / scalability notes:** For $`N`$ image tokens and $`Q`$ queries, attention interactions include $`O(N^2d)`$ encoder, $`O(QNd)`$ cross-attention, and $`O(Q^2d)`$ query self-attention. Projections and FFNs add width-dependent costs; straightforward assignment algorithms can add $`O(Q^3)`$ work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [The paper's Table 1](https://arxiv.org/html/2005.12872v3) reports **42.0 box AP on COCO 2017 validation** for **ResNet-50 DETR**, trained on COCO train2017 with the 500-epoch schedule. This AP averages IoU thresholds **0.50 through 0.95**, unlike VOC mAP at 0.5. It matches the strengthened Faster R-CNN-FPN+ baseline's 42.0 overall AP while trading weaker small-object AP for stronger large-object AP. A photograph becomes image tokens; queries jointly propose a labelled object set scored without NMS. This is a detection benchmark, not evidence of a named retail, autonomous-driving, or surveillance deployment.

**Notable vendor implementations/libraries:** Meta's original DETR release and implementations in Hugging Face and detection frameworks. Deformable DETR, DINO detectors, and real-time DETR variants need separate provenance.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| ViT | Images with substantial labelled pretraining data | Global content-dependent patch interactions | High-resolution attention and data requirements | ImageNet-21k-pretrained ViT-L/16 transfer benchmark |
| Swin | Images requiring multiscale features | Windowed attention with a hierarchy | Locality and window-management overhead | Swin-T ImageNet-1K classification |
| U-Net | Annotated image grids and microscopy | Context plus precise skip-connected localization | Memory, boundary labels, and domain shift | ISBI neuronal-membrane segmentation |
| Faster R-CNN | Images with object boxes | Shared proposals and focused region classification | Proposal, matching, and postprocessing complexity | VGG-based VOC 2007 detection |
| YOLOv1 | Fixed-resolution object-detection images | Unified single-stage prediction | Coarse-grid failures on crowded small objects | Original VOC 2007 benchmark, not a later YOLO version |
| DETR | Images with annotated object sets | One-to-one set prediction without NMS | Original model's convergence and small-object limitations | COCO 2017 validation box AP |

## 1.9 Graph networks

Graph supervision has two separate axes: which targets are labelled, and whether evaluation nodes or graphs are visible during training. Inductive does not automatically mean supervised, and transductive does not automatically mean unsupervised. The representatives below use labelled training targets; their original semi-supervised or unsupervised counterparts are explicitly distinguished.

### 1.9.1 Graph convolutional network (GCN)

**Name:** Graph convolutional network; specifically the Kipf-Welling normalized neighborhood operator, instantiated here in a supervised whole-graph classifier.

**Category & sub-category:** Supervised learning; graph networks; normalized message passing followed by graph-level pooling.

**Originating paper/vendor/year:** Thomas Kipf and Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/html/1609.02907v4), 2016 preprint and ICLR 2017 paper. The title correctly describes their original citation-network experiment. The supervised instantiation below is from the [DGL graph-classification tutorial](https://www.dgl.ai/dgl_docs/en/1.1.x/tutorials/blitz/5_graph_classification.html).

**Core mechanism:** Add self-loops and normalize adjacency to $`\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}`$. A layer computes $`H'=\phi(\hat AHW)`$, combining a node's own features with degree-normalized neighboring features. Shared weights permit evaluation on new graphs. A permutation-invariant mean readout turns node representations into one graph prediction; pooling is an additional architectural choice, not part of the original node classifier.

**Inputs/outputs and typical data types:** Node-feature matrices and sparse adjacency lists become node features; the representative outputs one class distribution per protein graph.

**Architecture diagram description:**

```text
protein graph: node features + edges + self-loops
 -> normalized GraphConv(3 -> 16) -> ReLU
 -> normalized GraphConv(16 -> 2) -> mean over nodes
 -> two graph logits -> predicted graph class
```

**Activation functions used and why:** ReLU between the two graph convolutions. The second convolution produces linear logits, pooled before cross-entropy applies its class normalization. Normalized adjacency is a fixed propagation operator, not an activation.

**Loss function(s):** Cross-entropy over **labelled training graphs**. There is no loss on unlabelled test nodes and no masked-label citation-network objective in this instantiation.

**Optimization algorithm(s):** The DGL example uses Adam at 0.01, batch size five graphs, and 20 epochs, without an explicit learning-rate decay schedule.

**Regularization techniques:** The displayed tutorial does not introduce dropout or weight decay. Its small hidden layer and normalized aggregation constrain capacity. A rigorous application would separately validate regularization and splitting rather than assuming this minimal demonstration is optimized.

**Backpropagation considerations:** Gradients flow through node transforms, sparse aggregation, and graph pooling, not through a learned choice of graph edges. Repeated propagation can oversmooth representations; averaging may lose distinctions needed for the graph label.

**Parameter count / scaling behavior:** For the displayed 3-16-2 network with biases, the count is $`3(16)+16+16(2)+2=\mathbf{98}`$, calculated from the tutorial. The same weights are reused at all vertices and across all graphs.

**Training paradigm:** Fully supervised, inductive graph classification on disjoint training/test graphs. The original paper's Cora results instead use a small labelled node subset within a graph whose other features and edges are visible; see [semi-supervised learning](03-semi-supervised.md).

**Hardware/parallelism considerations:** Disjoint graphs can be batched as a disconnected union while preserving separate readouts. Sparse aggregation often has less favorable memory locality than dense matrix multiplication.

**Strengths and limitations:** GCNs encode relational structure without imposing a node ordering. Simple smoothing can be unsuitable for heterophilous edges, and a small mean-pooled model cannot distinguish every graph structure or capture arbitrary biochemical interactions.

**Computational complexity / scalability notes:** A transform-first sparse layer costs $`O(Vd_{\rm in}d_{\rm out}+(m+V)d_{\rm out})`$, with self-loops included. Sparse adjacency needs $`O(V+m)`$ storage; a dense $`V`$ by $`V`$ implementation loses this advantage.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** DGL's [PROTEINS demonstration](https://www.dgl.ai/dgl_docs/en/1.1.x/tutorials/blitz/5_graph_classification.html) addresses structural-bioinformatics graph classification using three-component node features and two graph labels. It places the first 80% of dataset indices in training and the remainder in testing; random samplers shuffle **within**, not between, those partitions. The viewed documentation prints **0.273542600896861 test accuracy**, approximately **27.35%**, for one illustrative run. This poor result is retained rather than replaced with an invented improvement. Protein-graph neighborhoods produce pooled logits and a graph class that would require validation before scientific use. My rationale relative to flattened vectors is permutation-aware relational processing. The tutorial is not a robust, seed-averaged biological benchmark or a production discovery claim.

**Notable vendor implementations/libraries:** DGL `GraphConv` and PyTorch Geometric `GCNConv`. Normalization, self-loop handling, cached adjacency, and graph readout need explicit configuration.

### 1.9.2 Graph attention network (GAT)

**Name:** Graph attention network; the original multi-head **GAT**, instantiated for supervised inductive protein-function prediction.

**Category & sub-category:** Supervised learning; graph networks; learned neighborhood attention.

**Originating paper/vendor/year:** Petar Velickovic and colleagues, [Graph Attention Networks](https://arxiv.org/html/1710.10903v3), 2017 preprint and ICLR 2018 paper. The study includes both semi-supervised citation-network tasks and a supervised inductive PPI task.

**Core mechanism:** Transform node features, score connected pairs with an additive attention function, and normalize scores over each node's neighbors. The new state is a weighted sum of transformed neighbor features. Multiple heads learn different neighborhood weighting patterns; hidden heads can be concatenated and output heads averaged. Attention is restricted by graph connectivity rather than computed between every pair of vertices.

**Inputs/outputs and typical data types:** A protein-interaction graph with 50 features per node becomes 121 independent function-label scores per node in the PPI experiment.

**Architecture diagram description:**

```text
PPI node features + adjacency
 -> GAT: 4 heads x 256 features -> concatenate -> ELU
 -> GAT: 4 heads x 256 features -> concatenate -> ELU, skip connection
 -> GAT: 6 heads x 121 scores -> average -> sigmoid -> function labels
```

**Activation functions used and why:** Leaky ReLU in the attention scorer, neighborhood softmax for normalized weights, ELU after hidden layers, and logistic sigmoid at the multilabel output. An output softmax would incorrectly force the 121 biological labels to be mutually exclusive.

**Loss function(s):** Multilabel binary cross-entropy on labelled protein nodes in the training graphs. This differs from single-label cross-entropy in the citation-network experiments.

**Optimization algorithm(s):** Adam at 0.005 for PPI, with batches of two graphs. The paper uses validation-based early stopping with patience 100 epochs; it does not prescribe a separate decaying rate for this experiment.

**Regularization techniques:** For PPI, the authors explicitly report **no dropout and no L2 penalty**, finding the labelled training set sufficiently large. They use a skip connection across the intermediate attentional layer. The strong dropout used on small citation datasets must not be copied into this result.

**Backpropagation considerations:** Gradients affect both transformed features and normalized neighbor weights. Attention coefficients can become concentrated, and high-degree neighborhoods create large activation workloads. A large coefficient is not, by itself, evidence of a causal biological interaction.

**Parameter count / scaling behavior:** For $`a`$ heads of width $`f`$, a simple layer has roughly $`a(df+2f)`$ transform/scoring parameters before biases and skip projections. Parameter count is not proportional to the number of edges, although intermediate attention storage is.

**Training paradigm:** Supervised **inductive** node classification: 20 labelled PPI graphs train the model, two graphs validate it, and two unseen graphs test it. The paper's Cora/Citeseer/Pubmed experiments are instead transductive, semi-supervised settings; they are not the example here.

**Hardware/parallelism considerations:** Graph batching and sparse edge kernels are useful, but variable degrees create irregular work. More heads increase both message traffic and feature storage.

**Strengths and limitations:** GAT learns which connected neighbors contribute most for its objective, instead of fixing every weight by degree normalization. It remains limited by available edges, original attention expressivity, oversmoothing, and possible graph-distribution shift.

**Computational complexity / scalability notes:** Sparse per-layer work is approximately $`O(Vadf+maf)`$, plus softmax reductions. Attention coefficient storage is $`O(ma)`$; dense all-pairs attention would require a different $`O(V^2)`$ edge assumption.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [PPI experiment](https://arxiv.org/html/1710.10903v3) reports **micro-F1 0.973 +/- 0.002** on the **two held-out tissue graphs**, compared with **0.934 +/- 0.006** for the constant-attention control using the same broad architecture. The full dataset contains 24 graphs, with 44,906 training, 6,514 validation, and 5,524 test nodes. Protein attributes and interaction edges enter attention layers; sigmoid scores produce predicted function annotations, evaluated jointly over label decisions. The authors use the constant-attention control to examine the value of learned neighbor weighting. This is not experimentally verified protein function, clinical validation, or a deployed drug-discovery system.

**Notable vendor implementations/libraries:** The authors' GAT repository, DGL `GATConv`, and PyTorch Geometric `GATConv`. GATv2 changes the scoring architecture and should not inherit the original PPI result without a new evaluation.

### 1.9.3 GraphSAGE

**Name:** GraphSAGE, emphasizing its **supervised mean-aggregator** variant and distinguishing unsupervised random-walk training.

**Category & sub-category:** Supervised learning; graph networks; inductive neighborhood sampling and aggregation.

**Originating paper/vendor/year:** William Hamilton, Rex Ying, and Jure Leskovec, Stanford, [Inductive Representation Learning on Large Graphs](https://arxiv.org/html/1706.02216v4), NeurIPS 2017. The cited revision includes corrections and clarifications relative to earlier preprints.

**Core mechanism:** Sample a fixed number of neighbors at each aggregation depth. Aggregate their representations, combine that summary with the node's own representation, transform it, and normalize. Learning the aggregation function rather than a free embedding for each node allows new nodes to be encoded from their features and neighborhoods. Mean, pooling, LSTM, and GCN-style aggregators are distinct variants; the GCN-style variant does not use the same self/neighbor concatenation as ordinary GraphSAGE-mean.

**Inputs/outputs and typical data types:** Node attributes and graph neighborhoods become embeddings and node-class predictions. Text-derived post features and interaction-derived edges are used in the Reddit benchmark.

**Architecture diagram description:**

```text
root node -> sample neighbors -> sample their neighbors
           -> aggregate outer-hop features -> aggregate nearer-hop features
           -> concatenate self + neighbor mean -> learned transform/ReLU
           -> normalize embedding -> supervised classification head
```

**Activation functions used and why:** The reported GraphSAGE variants use ReLU, with normalized output embeddings. Classification uses softmax for the single-label Reddit task. Pooling or LSTM aggregators add their own internal transformations.

**Loss function(s):** Supervised class cross-entropy for this instance. The paper's **unsupervised** alternative instead uses random-walk positive pairs and negative sampling; those results occupy separate columns in its table.

**Optimization algorithm(s):** Adam; the supervised learning-rate search is over 0.01, 0.001, and 0.0001, selected using validation data. The paper specifies two aggregation depths with sample sizes $`S_1=25,S_2=10`$; it does not establish one universal decay schedule for all variants.

**Regularization techniques:** Stochastic neighborhood sampling, normalized representations, and validation-selected model size. These do not justify importing the later, specially tuned PPI dropout/normalization recipe into the original Reddit result.

**Backpropagation considerations:** Gradients propagate through sampled message computations and trainable aggregators, not through discrete neighbor selection. Repeatedly sampled vertices may share cached work or appear multiple times, depending on implementation.

**Parameter count / scaling behavior:** A mean-aggregator transform on concatenated width $`2d`$ to width $`h`$ has roughly $`2dh`$ weights, plus optional biases and the head. Weights do not grow with the number of indexed vertices; stored features and adjacency still do.

**Training paradigm:** Supervised inductive node classification in the chosen experiment. Pre-existing GloVe features come from another training process, but the GraphSAGE objective here is label-supervised. Its self-supervised variant is cross-referenced to [representation learning](05-unsupervised-neural.md).

**Hardware/parallelism considerations:** Minibatch neighborhood construction can bottleneck host memory or graph-store access. GPU kernels accelerate transformed features, but sampling and data transfers remain part of end-to-end cost.

**Strengths and limitations:** Sampling controls per-batch neighborhood work and supports unseen nodes. Sampling introduces variance and can miss informative neighbors; deeper receptive fields can expand rapidly despite bounded fanout.

**Computational complexity / scalability notes:** For $`B`$ roots and hop fanouts $`s_1,\ldots,s_L`$, the computation graph can contain $`O(B[1+s_1+s_1s_2+\cdots+\prod_\ell s_\ell])`$ node occurrences before deduplication. Each needs feature transformations and aggregation. Calling sampling "constant-time" is only meaningful with fixed depth, fanouts, widths, and data-access assumptions.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In the [Stanford Reddit experiment](https://arxiv.org/html/1706.02216v4), researchers construct a graph of 232,965 September-2014 posts from 50 communities, connecting posts when the same user comments on both. Features combine title/comment GloVe averages with post statistics. The first 20 days train the model; remaining days supply held-out data, with 30% used for validation. **Supervised GraphSAGE-mean reaches test micro-F1 0.950**, versus the listed raw-feature baseline's 0.585. A new post and its available neighborhood become an embedding and predicted community. The documented aim is inductive representation learning, **not a claim that Reddit selected or deployed this model for ranking**.

**Notable vendor implementations/libraries:** Stanford's GraphSAGE release, DGL `SAGEConv`, and PyTorch Geometric `SAGEConv`. Sampler ordering, replacement, layer fanouts, normalization, and aggregator type are part of reproducibility.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| GCN | Labelled graphs with useful neighborhood structure | Simple normalized, permutation-aware aggregation | Smoothing and pooling can discard distinctions | DGL PROTEINS graph-classification demonstration; poor recorded result disclosed |
| GAT | Graphs where neighbor contributions vary | Learned sparse neighborhood weighting | Irregular edge work and limited attention expressivity | Fully supervised inductive PPI function-label benchmark |
| GraphSAGE | Large evolving attributed graphs | Encodes unseen nodes with sampled neighborhoods | Fanout expansion, sampling variance, and data movement | Stanford's supervised Reddit community-classification benchmark |

## 1.10 Metric and event-based networks

These families are connected by neither topology nor hardware. They are grouped here because their distinctive training or representation choices cut across ordinary classification backbones: Siamese networks learn from relationships between examples, while SNNs communicate through time-dependent spikes.

### 1.10.1 Supervised Siamese networks

**Name:** Supervised Siamese networks; instantiated by Koch, Zemel, and Salakhutdinov's convolutional same/different classifier for one-shot character recognition.

**Category & sub-category:** Supervised learning; metric and event-based networks; shared-encoder pair comparison.

**Originating paper/vendor/year:** Bromley and colleagues' [1993 signature-verification work](https://doi.org/10.1142/S0218001493000339) is a foundational Siamese-network reference. The specific architecture and benchmark here come from [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), the ICML Deep Learning Workshop, 2015, not a main-conference ICML paper.

**Core mechanism:** Two inputs pass through identical encoders with shared parameters. A comparison head operates on their representations and predicts whether they belong to the same class. In the chosen paper it computes componentwise absolute differences and learns a weighted combination followed by a sigmoid. At test time, a query is compared with one labelled support example from each candidate class; the largest similarity selects the class without retraining a new output layer.

**Inputs/outputs and typical data types:** Pairs of grayscale character images plus same/different labels become similarity probabilities. The one-shot decision uses support-set class labels in addition to the trained comparator.

**Architecture diagram description:**

```text
image A -> shared Conv[64,128,128,256] -> shared Dense4096 --+
image B -> shared Conv[64,128,128,256] -> shared Dense4096 --+-> abs difference
                                                        -> weighted sum/sigmoid
query versus each labelled support image -> highest similarity -> class
```

**Activation functions used and why:** ReLU in the convolutional feature extractor, sigmoid in the final feature layer and comparison output. The learned output is a similarity score; a general sigmoid comparator need not satisfy metric axioms such as the triangle inequality.

**Loss function(s):** Binary cross-entropy for same/different labels plus layerwise L2 penalties. Contrastive-margin and triplet losses are important Siamese-family alternatives, **not the loss used for this quoted result**.

**Optimization algorithm(s):** Momentum SGD with minibatches of 128. Layerwise initial rates are selected from a $`10^{-4}`$ to $`10^{-1}`$ search range; all decay by a factor 0.99 per epoch. Momentum starts at 0.5 and ramps to selected layerwise values. Training stops on one-shot validation performance, with a maximum of 200 epochs.

**Regularization techniques:** Shared weights, layerwise L2 penalties, affine image distortions, and validation-based stopping. Pair construction and class/writer separation matter as much as ordinary weight regularization.

**Backpropagation considerations:** Both branches contribute gradients to the **same** encoder weights. Duplicating parameters instead of tying them changes the model. Pair sampling determines which similarities receive training signal and can bias the embedding toward easy comparisons.

**Parameter count / scaling behavior:** Weight sharing means two branches do not double stored encoder parameters. The displayed 4,096-unit feature layer is large: with the paper's 6-by-6-by-256 final map, its weight matrix alone contains $`9,216\times4,096`$, about **37.7 million weights**, calculated from the architecture.

**Training paradigm:** Supervised pair learning, followed by support-based one-shot classification on unseen classes. It is not label-free contrastive pretraining; support labels remain explicit supervision at evaluation.

**Hardware/parallelism considerations:** Pair branches parallelize, and support embeddings can be cached for repeated queries. Training stores activations for both inputs even though their weights are shared.

**Strengths and limitations:** The comparator can generalize to new categories without a fixed class-specific output head. Success depends on the quality and diversity of training pairs; high benchmark similarity does not establish reliable biometric identification.

**Computational complexity / scalability notes:** A pair requires approximately two encoder forward passes plus a linear-in-embedding comparison. With cached support embeddings, a query requires one encoder pass and $`O(kd)`$ comparison work for $`k`$ support classes. Enumerating every pair in $`n`$ examples would be quadratic; sampled pair training need not be.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [2015 study's Table 2](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) reports **92.0% accuracy** over **400 trials of 20-way within-alphabet Omniglot one-shot classification**. Its study reserves 30 alphabets for training, ten for validation, and ten for testing, with separated writers; these details should not be replaced by another widely used Omniglot split. A novel character is compared with twenty labelled examples, and the most similar supplies the output class. The authors aim to transfer learned verification features to new classes. This is character recognition, not a measured signature-fraud reduction or deployed identity-verification outcome.

**Notable vendor implementations/libraries:** Shared Keras models and PyTorch modules can implement Siamese branches; metric-learning libraries support contrastive and triplet alternatives. A wrapper does not specify the encoder or training objective.

### 1.10.2 Spiking neural networks (SNNs)

**Name:** Spiking neural network; the representative is a **supervised surrogate-gradient leaky-integrate-and-fire network** from snnTorch Tutorial 5.

**Category & sub-category:** Supervised learning; metric and event-based networks; temporally evolving spike-based neural computation.

**Originating paper/vendor/year:** SNNs are a broad historical family, not one invention. [Neftci, Mostafa, and Zenke's 2019 surrogate-gradient review](https://arxiv.org/abs/1901.09948) explains the training approach. The concrete recipe is [Eshraghian's snnTorch tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html), based on the project's published training guidance.

**Core mechanism:** Each neuron accumulates current in a leaky membrane state and emits a spike when a threshold is crossed. A reset changes subsequent state. The forward spike is discontinuous, so supervised training replaces its unusable ordinary derivative with a smooth surrogate during the backward pass. The example repeats static image-derived current over 25 simulation steps; it does **not** require Poisson encoding or an event camera.

**Inputs/outputs and typical data types:** Continuous image-derived currents or genuine event streams can drive SNNs. Here, 784 normalized MNIST pixel values yield output spike trains; the class with the greatest output spike count is selected.

**Architecture diagram description:**

```text
static MNIST vector, repeated for 25 steps
 -> Linear(784,1000) -> LIF membrane/spikes
 -> Linear(1000,10) -> LIF membrane/spikes -> spike counts -> digit
each LIF: previous membrane -> leak + input current -> threshold spike -> reset
```

**Activation functions used and why:** A Heaviside threshold generates spikes in the forward pass. The tutorial uses the documented arctangent-based surrogate convention for gradients; it does not make the actual emitted spikes continuous. Membrane leak is 0.95 in this example.

**Loss function(s):** Sum of class cross-entropies applied to the **output membrane potentials at every time step**. Evaluation uses output spike counts. It would be incorrect to describe this specific implementation as cross-entropy on spike counts.

**Optimization algorithm(s):** Adam at $`5\times10^{-4}`$, $`\beta_1=0.9,\beta_2=0.999`$, with no explicit rate decay in the displayed one-epoch training loop.

**Regularization techniques:** The minimal tutorial does not specify dropout or weight decay. Leak, reset, and a finite simulation horizon constrain dynamics but are not substitutes for measuring generalization.

**Backpropagation considerations:** BPTT unrolls membrane dynamics and uses surrogate derivatives at threshold crossings. The resulting gradient is an approximation, not the exact derivative of the discontinuous spike map. Reset differentiation and surrogate scale influence credit assignment and must be stated when changing implementations.

**Parameter count / scaling behavior:** With the two biased linear layers and fixed neuron constants, the tutorial has $`784(1000)+1000+1000(10)+10=\mathbf{795,010}`$ trainable parameters, calculated. Membrane states add runtime memory, not necessarily trainable parameters.

**Training paradigm:** Supervised learning from digit labels. Unmodulated spike-timing-dependent plasticity, or **STDP**, is a different local learning rule, defined in the [optimization glossary](10-glossary.md#62-optimization-and-neural-computation) but not separately cataloged in this edition. Neither all SNNs nor all STDP variants share one supervision category.

**Hardware/parallelism considerations:** Sparse spikes can benefit suitable neuromorphic hardware, but a dense GPU simulation still performs time-step operations and stores states. **No energy saving is claimed here:** measured energy depends on hardware, event rates, encoding, precision, memory traffic, simulation horizon, and whether training or inference is measured.

**Strengths and limitations:** Explicit temporal state is attractive for event-driven data. Surrogate choice, simulation length, conversion overhead, and limited hardware portability complicate fair comparisons with ordinary neural networks.

**Computational complexity / scalability notes:** This dense two-layer simulation performs roughly $`O(Tp)`$ arithmetic per example and stores $`O(BT H_{\rm neurons})`$ temporal activations for straightforward BPTT. An event-driven implementation instead depends on actual synaptic events; the two cost models are not interchangeable.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) reports **9,387 correct classifications out of all 10,000 MNIST test images: 93.87% accuracy** in its displayed run. The final test loader explicitly retains the last partial batch. A static digit drives repeated currents, hidden spikes influence ten output membranes, and accumulated output spikes select the digit. This documents supervised trainability of the chosen spiking instance, not superiority over the CNN above; the recipes are different. It reports no neuromorphic-device energy measurement, sensor deployment, or commercial efficiency KPI.

**Notable vendor implementations/libraries:** snnTorch, SpikingJelly, and hardware-oriented neuromorphic software stacks. A model simulated in one library is not automatically executable or energy-efficient on a particular chip.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Supervised Siamese networks | Labelled pairs and few-shot support sets | Shared comparison features can transfer to unseen classes | Pair-selection bias and uncalibrated similarities | Omniglot 20-way one-shot character recognition |
| Supervised surrogate-gradient SNN | Event streams or explicitly encoded temporal inputs | Trainable spike-based temporal dynamics | Surrogate approximation and hardware-specific efficiency | snnTorch static-MNIST simulation; no energy claim |

## 1.11 Supervised mixtures

Mixture-of-experts models are organized by their objective, not by the presence of a router. The 1991 adaptive mixture learns from labelled examples. GShard's model learns from parallel translations. Later sparse language models trained on unpaired text belong in [the model catalog](07-moe-models.md), while [the MoE deep dive](08-moe-deep-dive.md) develops shared routing and systems concepts.

### 1.11.1 Adaptive mixture of local experts

**Name:** Adaptive mixture of local experts, the **Jacobs/Jordan/Nowlan/Hinton 1991** model.

**Category & sub-category:** Supervised learning; supervised mixtures; learned input-dependent specialization.

**Originating paper/vendor/year:** Robert Jacobs, Michael Jordan, Steven Nowlan, and Geoffrey Hinton, [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jacobs.pdf), Neural Computation 3:79-87, 1991. MIT and University of Toronto researchers developed the method; it is not a later language-model vendor architecture.

**Core mechanism:** Several experts receive the same input, while a gating network outputs mixing probabilities $`g_i(x)`$. Training encourages experts to specialize on cases where their predictions are relatively successful. The paper contrasts error on an averaged prediction with objectives based on individual expert errors and a probabilistic mixture. Under the mixture likelihood, target-conditioned responsibilities determine how strongly each expert learns from a case; the gate learns which experts tend to succeed in that input region.

**Inputs/outputs and typical data types:** Numeric features and explicit target vectors become a prediction or conditional output distribution. The original application uses two acoustic formants to discriminate four vowel classes.

**Architecture diagram description:**

```text
                       +-> expert 1 -> output distribution --+
input x ---------------+-> expert 2 -> output distribution --+-> weighted mixture
                       +-> ... expert k --------------------+
        \-> gating network -> softmax probabilities ---------+
label y -> expert errors/responsibilities -> update experts and gate
```

**Activation functions used and why:** Softmax normalizes gate outputs. Expert activation depends on the supervised output distribution; the vowel experiment restricts experts to simple linear decision boundaries. The paper does not establish one universal hidden-layer activation recipe, and no modern ReLU stack is implied.

**Loss function(s):** The paper develops expected expert squared error, $`\sum_i g_i\|y-f_i(x)\|^2`$, and a Gaussian-mixture-style negative log-likelihood, $`-\log\sum_i g_i\exp[-\|y-f_i(x)\|^2/2]`$, with fixed-scale constants suppressed. Neither equals squared error of the mixture's average prediction. The experiment uses average squared error 0.08 as its stopping criterion.

**Optimization algorithm(s):** The reported vowel experiments use **full-batch gradient descent with a fixed step size**, chosen by limited convergence exploration for each system. They use **no momentum**. A numerical schedule not supplied for every configuration is not reconstructed from later MoE practice.

**Regularization techniques:** Small experts with restricted decision surfaces control capacity. The original encourages specialization and can leave an expert effectively unused; it does not contain today's auxiliary load-balancing penalty.

**Backpropagation considerations:** The gate and experts receive coupled but distinct gradients. For a likelihood mixture, responsibilities are proportional to $`g_i p_i(y\mid x)`$, so a good expert receives stronger credit. Expert permutation symmetry and poor initialization can produce nonunique or unhelpful decompositions.

**Parameter count / scaling behavior:** With $`k`$ equal-size experts, total parameters are approximately $`kp_{\rm expert}+p_{\rm gate}`$. The experiment compares four/eight experts with roughly parameter-matched six/twelve-hidden-unit conventional networks; no modern billion-parameter count is applicable.

**Training paradigm:** Explicitly supervised learning from labelled vowel cases. A probabilistic latent expert identity does not make the training unsupervised.

**Hardware/parallelism considerations:** Experts can evaluate in parallel, but the classical training objective generally evaluates all experts. This is **not** automatically the sparse per-token execution of later top-k Transformer MoEs.

**Strengths and limitations:** Input-dependent specialists can reduce interference between subtasks. Specialization is not guaranteed to be interpretable, load-balanced, or beneficial; extra experts can remain unused, and dense mixture evaluation can cost more than one expert.

**Computational complexity / scalability notes:** Per-example evaluation costs $`O(kC_{\rm expert}+C_{\rm gate})`$ when all experts run. Training epochs add backward costs for the expert and gate computations. Fewer epochs to a stopping criterion is not by itself an equal-factor reduction in wall-clock cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original study](https://www.cs.toronto.edu/~hinton/absps/jacobs.pdf) uses Peterson-Barney vowel formants from **75 speakers**, training on the first 50 and testing on the remaining 25. Across **25 simulations per configuration**, both mixtures and conventional networks report **90% test classification accuracy**. Four experts require an average **1,124 epochs** to the stated error criterion, versus **2,209** for the six-hidden-unit backpropagation network. Thus the reported advantage is convergence in epochs, **not improved test accuracy**. Two formants enter specialists and a gate; weighted class distributions select a vowel. The documented rationale is reduced interference between naturally different decision regions. No speech-service deployment, runtime speedup, or commercial KPI is established.

**Notable vendor implementations/libraries:** The architecture can be implemented with ordinary neural-network layers and mixture-distribution utilities. Such a custom implementation must choose its expert likelihood and gate explicitly; no modern MoE package is automatically an exact 1991 reproduction.

### 1.11.2 GShard multilingual translation

**Name:** **GShard's sparsely gated multilingual translation Transformer**, as reported in 2020. GShard is also the name of the automatic-sharding system supporting the model.

**Category & sub-category:** Supervised learning; supervised mixtures; distributed sparse expert encoder-decoder translation.

**Originating paper/vendor/year:** Dmitry Lepikhin and colleagues, Google, [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/html/2006.16668v1), June 2020 technical report, subsequently presented at ICLR 2021.

**Core mechanism:** Replace every other positionwise feed-forward layer in a Transformer encoder and decoder with an MoE layer. A softmax router selects up to two experts per token under the paper's capacity-limited, partly stochastic dispatch procedure. Tokens are grouped for dispatch; overloaded experts cannot accept unlimited tokens. Expert outputs are weighted and combined back into the residual stream. Automatic SPMD sharding distributes the computation and parameters instead of manually rewriting the model for each device configuration.

**Inputs/outputs and typical data types:** Source-language subword sequences paired with English translations become conditional English-token distributions. Source and target tokenizers use distinct multilingual-source and English-target vocabularies.

**Architecture diagram description:**

```text
source tokens -> encoder: attention + alternating dense FFN / routed MoE
                                                        |
target prefix -> decoder: masked attention + cross-attention + dense FFN / MoE
                                                        -> English-token scores
MoE: softmax router -> capacity-limited top-2 dispatch -> expert FFNs -> combine
                         [experts sharded across TPU cores]
```

**Activation functions used and why:** ReLU in the two-layer expert feed-forward transformations, softmax for attention/routing and output classification, and Transformer residual/normalization operations. This is not a later SwiGLU-based sparse language-model recipe.

**Loss function(s):** Conditional translation negative log-likelihood plus the paper's auxiliary load-balancing term. The auxiliary objective discourages routing concentration that would waste expert capacity; it does not replace translation supervision.

**Optimization algorithm(s):** Appendix A.2 specifies Adafactor with factored second moments, no first moment, a second-moment schedule described as $`1-t^{-0.8}`$, update clipping threshold one, and learning rate one followed by inverse-square-root decay after 10,000 steps. These are the report's adaptive-optimizer conventions, not an interchangeable Adam learning rate.

**Regularization techniques:** Input, residual, and attention dropout 0.1, plus auxiliary routing balance. Capacity constraints and stochastic second-expert routing affect which expert computations occur; they are systems/training choices, not guarantees of uniform expert use.

**Backpropagation considerations:** Selected experts receive task gradients, while router probabilities and the balancing objective provide routing-related gradients. Discrete selection/capacity decisions do not have an ordinary continuous derivative. Dispatch and combine operations must preserve correct token-to-expert correspondence in both passes.

**Parameter count / scaling behavior:** The principal large model is reported as approximately **600 billion total weights**, dominated by expert parameters, with **2,048 experts per MoE layer** and **36 combined encoder-plus-decoder layers**. The appendix specifies model width 1,024, expert/FFN hidden width 8,192, and sixteen attention heads with 128-dimensional keys/values. **The original 2020 GShard report does not give a precise whole-model active-parameters-per-token inventory.** Multiplying 600B by $`2/2048`$ incorrectly ignores shared layers, embeddings, attention, depth, and actual capacity-limited routing.

A later cross-paper comparison, [Du et al.'s GLaM Table 2, v2 dated 2022-08-01](https://arxiv.org/html/2112.06905v2#S2.T2), lists **GShard-M4: 600B total / 1.5B activated parameters per input token**. This is the GLaM authors' rounded count under that table's counting convention for the representative encoder-decoder, not an independent recomputation or a universal count for every GShard configuration. It does not supply the missing precise inventory in the original report or change GShard's supervised translation signal.

**Training paradigm:** **Supervised translation, not next-token unsupervised pretraining.** The mined parallel corpus contains about 25 billion examples across directions; the reported many-to-English training uses approximately **13 billion examples**. Web origin and noisy alignment do not remove the supervision supplied by paired translations.

**Hardware/parallelism considerations:** Expert parallelism requires substantial cross-device dispatch/combination traffic. The 600B experiment uses **2,048 TPU v3 cores** and about **four days**. These hardware-specific research figures establish neither portable GPU timing nor commercial energy or cost savings.

**Strengths and limitations:** Sparse expert activation grows total capacity without evaluating every expert for each token. Routing balance, communication, deployment memory, low-resource transfer, and sequential autoregressive inference remain constraints; more stored parameters do not guarantee universally better sharing.

**Computational complexity / scalability notes:** With $`k`$ experts, routing scores can require $`O(BTdk)`$ work. Selected two-layer experts add approximately $`O(BT r d f)`$ for at most $`r=2`$ accepted routes, **in addition to** dense/shared Transformer, attention, and communication costs. Total expert storage grows roughly as the number of MoE layers times $`kdf`$. Grouped capacity and sharding assumptions determine actual scaling.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In [Table 3](https://arxiv.org/html/2006.16668v1), **MoE(2048E,36L)** reports **44.3 average BLEU** on the study's held-out **100-language-to-English evaluation**, versus **36.9** for its listed dense **T(96L)** baseline. These are the authors' multilingual test sets and averaging protocol, not WMT newstest2014 or a universally comparable BLEU aggregate. The underlying mined corpus and exact test reconstruction are not fully released in this report. A non-English sentence is encoded; successive English tokens route through experts while attending to the source, and beam search selects a translation. This demonstrates supervised translation scaling, not evidence that Google Translate commercially deployed this exact model. Later sparse language models are covered in [07](07-moe-models.md) and [08](08-moe-deep-dive.md).

**Notable vendor implementations/libraries:** The paper's TensorFlow/XLA SPMD sharding approach and related research code illustrate the systems contribution. Modern expert-parallel libraries implement related ideas, but do not establish release of every GShard training artifact or an exact public 600B checkpoint.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Adaptive mixture of local experts, 1991 | Labelled data with distinct input regimes | Learned specialization can reduce interference | All-expert cost and unused or unstable specialists | Speaker-disjoint vowel experiment: equal accuracy, fewer training epochs |
| GShard translation, 2020 | Large multilingual parallel corpora | Sparse expert capacity with automatic sharding | Communication, routing capacity, and incomplete public reproduction artifacts | Supervised 100-language-to-English research benchmark |

## Coverage and continuation manifest

This bounded first-edition chapter contains **30 entries**. Every entry identifies a representative supervision signal, includes the nine common fields and nine neural-specific fields, supplies a text architecture diagram, and distinguishes its example's evidence status.

| Covered range | Sub-category | Entries |
|---|---|---:|
| 1.5.1-1.5.2 | Perceptron and MLP foundations | 2 |
| 1.6.1-1.6.10 | Generic CNN, LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet, ConvNeXt, DenseNet, MobileNet | 10 |
| 1.7.1-1.7.5 | RNN, LSTM, GRU, Bahdanau-attention Seq2Seq, original Transformer | 5 |
| 1.8.1-1.8.6 | ViT, Swin, U-Net, Faster R-CNN, YOLOv1, DETR | 6 |
| 1.9.1-1.9.3 | GCN, GAT, GraphSAGE | 3 |
| 1.10.1-1.10.2 | Supervised Siamese networks and surrogate-gradient SNNs | 2 |
| 1.11.1-1.11.2 | Original adaptive local experts and GShard translation | 2 |

**Reading connections.** Begin with the [reading guide and evidence policy](00-reading-guide.md) and [supervised classical models](01-supervised-classical.md). Continue to [semi-supervised learning](03-semi-supervised.md) for masked-label graph settings, consistency, and pseudo-label variants; [unsupervised classical learning](04-unsupervised-classical.md) for non-neural representation and clustering methods; and [unsupervised neural learning](05-unsupervised-neural.md) for autoencoders, contrastive/masked objectives, and the STDP cross-reference. [Foundation models](06-foundation-models.md) distinguish BERT/GPT/T5 pretraining from the original supervised Transformer. [MoE model families](07-moe-models.md) and the [dedicated MoE deep dive](08-moe-deep-dive.md) extend the routing discussion. The [comparative guide](09-comparative-guide.md) and [glossary](10-glossary.md) connect terminology across the book.

**Evidence boundaries.** Documentation runs are labelled as such, including the perceptron's in-sample score and the weak recorded PROTEINS result. Historical crop, ensemble, checkpoint, dataset, and loss distinctions are retained. Missing complete schedules or unpublished artifacts are not filled with another implementation's defaults. No neural family is assigned a proprietary product recipe without evidence. SNN energy claims remain hardware-qualified; the GShard-M4 active count is attributed to the later GLaM comparison rather than inferred from routing sparsity.

**Further non-required depth not included.** Separate full entries for Inception-v2/v3, EfficientNetV2, ConvNeXt V2, MobileNet-v2/v3, later YOLO generations, Mask R-CNN, RetinaNet, Deformable DETR, volumetric U-Nets, and nnU-Net's configuration algorithm would extend the vision coverage. Graph isomorphism networks, relational/heterogeneous and temporal GNNs, graph Transformers, advanced samplers, continuous-time recurrent models, neural differential equations, and state-space sequence models are further extensions. Detailed quantization recipes, compiler/kernel implementations, calibrated deployment studies, and controlled neuromorphic energy comparisons require their own evidence and experiments. Their omission bounds this edition; it is not a claim that the architectures above exhaust neural learning.
