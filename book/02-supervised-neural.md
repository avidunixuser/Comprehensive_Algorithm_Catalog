# 1. Supervised Learning Algorithms: Neural Architectures

This chapter continues [supervised learning](01-supervised-classical.md) with neural networks. It moves from a single learned decision rule to huge translation networks. The evidence policy is dated **2026-09-08**. Papers and named software versions are reference points, not claims about the newest models.

A neural network passes information through **layers**, or stages of calculation. Features are input details, such as pixel brightness or a sound measurement. Each layer uses **weights**, numbers learned during training, to combine those details. Weights are often stored in matrices, rectangular tables of numbers. A **bias** is another learned number that shifts the result. An **activation function** changes a layer's result so the network can learn more than straight-line relationships.

Training starts with examples and known answers, called labels or targets. A **loss** measures how far the network's output is from those answers. Training sends an error signal backward through the layers. This process, **backpropagation**, calculates gradients: slopes showing how small weight changes affect the loss. An **optimizer** uses those slopes to choose weight updates that aim to reduce the loss. Its learning rate controls the step size. A batch is a group of examples processed together; an epoch is one pass through the training data.

Deeper networks sometimes use **residual shortcuts**. These carry a block's input around its calculations and add it to the block's output. The block can then learn a correction rather than rebuild everything. Shortcuts also give backward error signals a more direct path. A **GPU**, or graphics processing unit, can perform many similar calculations at once. This helps with large networks, but memory use and data transfers still matter.

The network's design does not determine its kind of supervision. Here, training uses supplied classes, numbers, annotations, translations, or same/different labels. The same designs can learn differently. An RNN can learn from unlabelled text. A graph network can use both labelled and unlabelled nodes. An image Transformer can learn to rebuild hidden image patches. Those objectives belong in [semi-supervised learning](03-semi-supervised.md), [unsupervised neural learning](05-unsupervised-neural.md), and [foundation-model pretraining](06-foundation-models.md).

Worked examples separate research tests, documentation demonstrations, and real applications. Accuracy on training examples is not accuracy on new examples. Different datasets, saved weight sets (**checkpoints**), crops, model combinations (**ensembles**), and scoring scripts can produce different scores. A higher research score is not proof of business value. This chapter reports cited results; it does not retrain the models. Available software is not evidence that a vendor uses it in a product.

The math is optional and its symbols are explained nearby. In general, n counts training examples, B is batch size, d is a feature or hidden width, and p counts learned parameters. T is sequence length, L counts layers, and E counts epochs. Image dimensions are H and W; q is a filter's side length. Graph sections use V for nodes and m for edges. Work estimates count calculations for the stated task, not seconds or energy use.

## 1.5 Neural foundations

A perceptron learns one decision boundary using the features it receives. An MLP also learns new combinations of those features inside the network. This makes an MLP more flexible, but harder to train.

### 1.5.1 Perceptron
**In plain English:** A perceptron combines input numbers and chooses a class. It is a cheap first check of whether a simple decision rule is enough.

**Name:** Perceptron. This entry uses a linear classifier. For several classes, it trains one rule per class against all the others.

**Category & sub-category:** Supervised learning; neural foundations. It learns a straight decision boundary and can update after each example.

**Originating paper/vendor/year:** Frank Rosenblatt's [1958 perceptron paper](https://doi.org/10.1037/h0042519) is the standard historical reference. The example uses modern [scikit-learn software](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html), not a rebuild of the original hardware.

**Core mechanism:** The model multiplies each input by its weight, adds the results and a bias, then checks the sign. A wrong answer, or an answer exactly on the boundary, triggers a weight update. An ordinary correct answer does not. This corrects mistakes; it does not estimate reliable probabilities.

**Optional math:** The score is $`s=w^\top x+b`$. Here x is the input vector, w the weights, and b the bias. The product $`w^\top x`$ multiplies matching components and adds them. With label $`y\in\{-1,+1\}`$ and positive step size $`\eta`$, update when $`ys\leq0`$: $`w\leftarrow w+\eta yx`$, $`b\leftarrow b+\eta y`$. If bounded inputs can be separated with a positive gap, a classical mistake bound scales with the squared input-radius-to-gap ratio. If no such boundary exists, mistakes and repeated cycling can continue.

**Inputs/outputs and typical data types:** Numeric lists, sparse text features, or flattened pixels go in. A class and a decision score come out. Sparse means most entries are zero. Separate binary rules supply the multiclass scores. A score is not a calibrated probability: it does not promise that predicted chances match observed frequencies.

**Architecture diagram description:** The main path makes a decision. The error path changes the weights after a labelled example.

```text
feature vector x -> weighted sum w.x + b -> threshold -> binary label
                          |
                    labelled error -> weight update
multiclass: parallel one-versus-rest units -> largest score
```

**Activation functions used and why:** A hard threshold chooses one side of the boundary. It jumps between answers rather than changing smoothly. That jump is part of this model's definition.

**Loss function(s):** The loss penalizes scores on the wrong side, without requiring a large safety gap. **Optional math:** $`\max(0,-ys)`$ uses label y and score s; the update rule must specify what happens at zero. Unlike an SVM's hinge loss, this loss does not require a unit margin.

**Optimization algorithm(s):** The original rule uses a fixed positive step size. Scikit-learn fixes `loss="perceptron"` and a constant learning rate, with default multiplier `eta0=1`. Limits on epochs and small-improvement stopping control runtime. They do not prove the model will settle on noisy data.

**Regularization techniques:** The original rule needs no penalty on weights. The library offers L1, L2, and elastic-net penalties, which discourage large weights in different ways. Scaling features, stopping based on validation results, and limiting passes can also help. These choices change the learned model.

**Backpropagation considerations:** There are no hidden layers to send error signals through. The hard threshold has no useful ordinary derivative for training. The mistake rule directly changes the weights instead.

**Parameter count / scaling behavior:** Model size depends on input width, not the number of examples. **Optional math:** With d input features, a binary model has $`d+1`$ parameters. With c one-versus-rest class rules, it has about $`c(d+1)`$.

**Training paradigm:** Known class labels guide updates, either one example at a time or over repeated passes. This version does not learn features without labels.

**Hardware/parallelism considerations:** A CPU, the computer's general-purpose processor, is usually enough. Sparse inputs let updates skip zero features. Class-specific rules can run in parallel, but arbitrary batch updates do not reproduce the same sequence of online updates.

**Strengths and limitations:** The model is inexpensive, and its weights are easy to inspect. It cannot solve XOR using the original two input coordinates. XOR means choosing "yes" when exactly one of two inputs is on, but not both. Added nonlinear features can help, but those features provide the extra flexibility, not the threshold alone. Its scores also lack calibrated uncertainty.

**Computational complexity / scalability notes:** Doubling examples or input features roughly doubles a dense pass's work. **Optional math:** With n examples, d features, and E passes, costs are $`O(nd)`$ per pass and $`O(End)`$ overall, with $`O(d)`$ model memory. For c class rules, multiply work by c. Sparse costs instead depend on how many entries are nonzero.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Illustrative (not a claimed deployment).** Scikit-learn's [digits example](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) recognizes handwritten digits from 8-by-8 pixel images. It reports `score(X, y) = 0.939...`. This is **training accuracy on the same data**, not a test on unseen images. Ten linear scorers read the pixels; the largest score chooses the digit. The reason to try this before an MLP is to check whether simple, cheap boundaries suffice.

**Optional math:** A toy update uses $`x=(1,2), y=+1, w=(0,0), b=0,\eta=1`$. Here x is the input, y its label, w the initial weights, b the bias, and eta the step size. The update gives $`w=(1,2),b=1`$. Neither this calculation nor the tutorial score proves production-quality optical character recognition (OCR).

**Notable vendor implementations/libraries:** Scikit-learn's `Perceptron` and `SGDClassifier` use the same underlying linear tools. Their availability shows how to implement the method, not where it is commercially deployed.

### 1.5.2 Multilayer perceptron (MLP)
**In plain English:** An MLP learns useful combinations of input details before choosing an answer. Hidden layers let it handle patterns that one straight boundary cannot separate.

**Name:** Multilayer perceptron, or fully connected feed-forward network. Information moves forward through connected layers. The example has one hidden layer of 40 units.

**Category & sub-category:** Supervised learning; neural foundations. Dense layers learn nonlinear rules for classes or numerical predictions.

**Originating paper/vendor/year:** Rumelhart, Hinton, and Williams' [1986 backpropagation paper](https://doi.org/10.1038/323533a0) helped make hidden-layer learning widely used. Multilayer networks already existed. The paper did not invent every part of an MLP.

**Core mechanism:** Each layer combines the previous layer's outputs, then applies an activation. Hidden layers learn intermediate features; the last layer turns them into an answer. Backward error signals show earlier layers how their features affected that answer. Without nonlinear activations, all the layers together would still make just one linear-plus-bias transformation.

**Optional math:** $`h_\ell=\phi(W_\ell h_{\ell-1}+b_\ell)`$. Here h is a layer's output, the index ell identifies its layer, W contains weights, b contains biases, and phi is the activation. The previous layer supplies the input.

**Inputs/outputs and typical data types:** Fixed-length numeric lists become class probabilities, several label scores, or numerical predictions. An image can be flattened into such a list, but this removes its explicit row-and-column arrangement.

**Architecture diagram description:** The hidden layer learns 40 combinations of the 784 pixels. The final layer scores ten digits.

```text
MNIST 28x28 pixels -> flatten/scale -> 784 inputs
                   -> Dense(40), ReLU -> Dense(10), softmax
                   -> digit probabilities -> selected digit
```

**Activation functions used and why:** The [scikit-learn example](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html) uses default ReLU hidden units. ReLU keeps positive inputs and replaces negative ones with zero, so positive responses do not flatten out. Softmax turns the ten output scores into probabilities that sum to one. This alone does not make them calibrated. A numerical-regression version usually leaves the output linear.

**Loss function(s):** This classifier uses cross-entropy: it penalizes giving the correct digit too little probability. An L2 penalty also discourages large weights. Squared prediction error belongs to a different, regression version of the model.

**Optimization algorithm(s):** The example uses stochastic gradient descent (SGD), an initial rate of 0.2, a constant schedule, and eight epochs. SGD updates from batches rather than the entire dataset. Its default momentum is 0.9, with Nesterov acceleration; these use recent update directions to guide steps. Documentation resource limits end the run before it settles. This is not a tuned MNIST recipe.

**Regularization techniques:** `alpha=1e-4` sets the L2 penalty. Dividing pixels by 255 scales the inputs. There is no dropout, which would randomly disable units during training, or batch normalization, which would rescale intermediate values using batch statistics. A held-out test partition measures new-example performance; it does not turn the short run into validation-tuned early stopping.

**Backpropagation considerations:** The chain rule links each layer's weight changes to the final loss. Some ReLU units can stay inactive. Other activations can flatten out and send very small error signals backward. Starting weights and input scaling matter. Gradient descent does not guarantee the best possible nonlinear classifier.

**Parameter count / scaling behavior:** Wider connected layers need many more weights. **Optional math:** Including biases, the 784-40-10 network has $`784(40)+40+40(10)+10=31,810`$ parameters. In general, $`p=\sum_\ell(d_{\ell-1}+1)d_\ell`$: p is total parameters and d is each layer's width. Doubling both adjacent widths roughly quadruples their weight matrix.

**Training paradigm:** This model starts with random weights and learns from digit labels. Autoencoders and label-free MLP objectives are different uses; see [unsupervised neural learning](05-unsupervised-neural.md).

**Hardware/parallelism considerations:** Small MLPs run well on a CPU. Larger layers and batches suit GPU matrix multiplication, which combines many weighted sums at once. Memory must hold weights, layer outputs, backward signals, and optimizer records.

**Strengths and limitations:** MLPs are useful starting points for prepared features and modest nonlinear problems. They do not build in image location, sequence order, or graph structure. Networks designed for those structures may learn from fewer examples.

**Computational complexity / scalability notes:** More weights or examples increase work roughly in proportion. **Optional math:** For p parameters and n examples, one prediction costs $`O(p)`$ and a training epoch about $`O(np)`$. Backpropagation adds an implementation-dependent multiplier. Training also stores batch outputs beyond the $`O(p)`$ weights and optimizer state.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Scikit-learn's [MNIST study](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html) uses OpenML `mnist_784`, version 1. Its random-state-0 split is **30% training and 70% testing**, not the usual MNIST 60,000/10,000 split. After eight epochs, it reports **0.953061 test accuracy** and **0.986429 training accuracy**.

A scaled 784-pixel image activates learned stroke combinations and produces one of ten digit choices. This could be a stage in document recognition. The reason to try an MLP over a perceptron is its learned nonlinear combinations. These differently split examples are not a controlled comparison proving one model better. No deployed document workflow or business result is reported.

**Notable vendor implementations/libraries:** Scikit-learn provides `MLPClassifier` and `MLPRegressor`; PyTorch provides `Linear`; Keras provides `Dense`. Their default optimizers and normalization choices differ.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Perceptron | Fixed-length or mostly zero feature lists | Cheap rule that updates after mistakes | Needs features a straight boundary can separate | Scikit-learn digits demo; score uses the training data |
| MLP | Fixed-length numeric lists | Learns new combinations of input details | Does not build in location or relationships | MNIST documentation test with an unusual 30/70 split |

## 1.6 Convolutional families

A convolution is a small learned pattern detector that moves across an image. It might respond to a stroke or texture. The same weights check every position. Each detector produces a **feature map**, a grid of responses; a stack of these maps forms channels. Later layers combine small patterns into larger ones.

Pooling shrinks a map by summarizing nearby values. Stride is how far a detector moves each step; padding adds border values. These choices affect what happens when an object moves in the image. Shared detectors do not guarantee identical predictions at every position. The families below change detector sizes, connections, channel mixing, or overall network size.

### 1.6.1 Generic convolutional neural network (CNN)
**In plain English:** A CNN searches images for small patterns, then combines them to recognize larger shapes. Sharing each detector across the image keeps the model relatively small.

**Name:** Convolutional neural network. The example is Keras's small, supervised MNIST digit classifier.

**Category & sub-category:** Supervised learning; convolutional families. Local image detectors share their learned weights across positions.

**Originating paper/vendor/year:** CNNs have several historical roots. LeCun and colleagues' [1998 document-recognition paper](https://leon.bottou.org/papers/lecun-98h) is a foundational reference. This example uses Francois Chollet's [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/), first published in 2015 and shown in Keras 3-compatible form at the evidence date.

**Core mechanism:** Small filters check each image region for learned patterns. ReLU changes their responses, further filters combine them, and pooling reduces map size. The final layer uses these features to choose a digit. This uses fewer separate connections than linking every pixel to every hidden unit. It is useful when nearby pixels form reusable strokes or textures.

**Inputs/outputs and typical data types:** Here, 28-by-28 grayscale images enter and digit probabilities leave. Other CNNs use one-dimensional signals or three-dimensional volumes. Those versions have different shapes and costs.

**Architecture diagram description:** Two convolution-and-pooling stages produce 1,600 values. The final classifier turns them into ten digit scores.

```text
28x28x1 -> Conv3x3(32), ReLU -> MaxPool2
         -> Conv3x3(64), ReLU -> MaxPool2 -> 5x5x64
         -> Flatten(1600) -> Dropout(0.5) -> Dense(10), softmax
```

**Activation functions used and why:** ReLU supplies a cheap nonlinear change after each convolution. Softmax turns final scores into competing digit probabilities. Max-pooling keeps the largest response in a region; it is a size-reduction step, not another learned activation.

**Loss function(s):** Categorical cross-entropy penalizes low probability for the labelled digit. Labels are one-hot lists: the correct class is marked and the others are not. The loss measures more than whether the largest score happens to be correct.

**Optimization algorithm(s):** Adam adapts weight-update sizes using recent gradients. This example uses batches of 128 and 15 epochs. `optimizer="adam"` selects Keras's default initial rate of 0.001. The example adds no learning-rate decay schedule.

**Regularization techniques:** Dropout 0.5 randomly disables half the inputs to the final dense layer during training. Pixels are scaled to [0,1], and 10% of training images are reserved for validation. There is no batch normalization or synthetic geometric image augmentation here.

**Backpropagation considerations:** Every place that uses a shared filter contributes to that filter's update. Max-pooling sends error signals through the positions it selected. ReLU blocks signals where its input was negative. Padding and layer shapes must match between training and later predictions.

**Parameter count / scaling behavior:** The example reports **34,826 trainable parameters**. **Optional math:** A square convolution has $`q^2C_{\rm in}C_{\rm out}`$ weights, plus any biases. Here q is filter side, and the C terms count input and output channels. Larger images do not directly add filter weights, but they can enlarge a flattened dense head.

**Training paradigm:** Digit labels supervise learning on the chosen training subset. This model neither rebuilds its inputs as an autoencoder nor learns from a label-free image objective.

**Hardware/parallelism considerations:** GPUs can process different images, positions, and channels together. For this small model, moving data and starting GPU operations can matter more than peak calculation speed.

**Strengths and limitations:** The network learns local image structure efficiently. Pooling can discard small details, and its flattened head expects a fixed shape. It still needs tests on handwriting styles beyond MNIST.

**Computational complexity / scalability notes:** Larger maps, filters, and channel counts all add work. **Optional math:** One dense convolution costs $`O(BH_{\rm out}W_{\rm out}q^2C_{\rm in}C_{\rm out})`$. B is batch size; H and W are output dimensions; q is filter side; C counts channels. Total training adds every layer, backward operations, and E epochs. Calling all CNNs "linear-time" hides these choices.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Keras's [MNIST run](https://keras.io/examples/vision/mnist_convnet/) reports **0.9919000268 test accuracy**, about **99.19%**, on the conventional **10,000-image test set**. From 60,000 training images, `validation_split=0.1` reserves 6,000 for validation.

The model scales a digit image, detects strokes, and selects the highest-probability digit. A document system would still need to locate characters and handle uncertain answers. Reusable local structure is the reason to try this instead of a flattened MLP. This educational test does not measure bank or postal automation. Its split also differs from the MLP example.

**Notable vendor implementations/libraries:** Keras/TensorFlow and PyTorch `Conv1d/2d/3d` provide convolution layers. NVIDIA cuDNN supplies optimized GPU operations. These are building blocks, not one standard CNN training recipe.

### 1.6.2 LeNet
**In plain English:** LeNet combines small stroke detectors to recognize handwritten characters. It shows how a compact image network can become part of a larger document-reading system.

**Name:** LeNet, focusing on historical LeNet-5. Simplified teaching versions are not necessarily the same network.

**Category & sub-category:** Supervised learning; convolutional families. Small CNNs recognize document characters and handwriting.

**Originating paper/vendor/year:** Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner describe it in [Gradient-Based Learning Applied to Document Recognition](https://leon.bottou.org/papers/lecun-98h), Proceedings of the IEEE, 1998. Earlier LeNet systems already existed. This date identifies the detailed reference, not the first CNN.

**Core mechanism:** Learned filters find local strokes. Smaller maps then help later layers combine them into character features. The C3 layer connects only selected earlier maps, not every map. Its shrinking stages also have learned values, unlike modern parameter-free average pooling. At the end, the historical network compares features with class prototypes and picks a low-distance answer. It does not use today's usual softmax head.

**Inputs/outputs and typical data types:** Centered, scaled grayscale characters occupy a 32-by-32 input field. The model outputs ten digit-related scores. Reading a whole cheque also requires finding fields and deciding where characters begin and end.

**Architecture diagram description:** C names convolution layers; S names subsampling stages; F6 is a dense feature layer. The final RBF scores measure distance to prototypes.

```text
32x32 -> C1: 6 maps, 5x5 -> S2: subsample
      -> C3: 16 selectively connected maps -> S4: subsample
      -> C5: 120 -> F6: 84 -> 10 Euclidean-RBF scores
      -> lowest-energy digit
```

**Activation functions used and why:** Historical feature units use scaled tanh, a smooth function whose output stays within bounds. This suits the prototype representation. The output uses Euclidean radial-basis-function distances. ReLU/softmax teaching versions are useful adaptations, not the original operations.

**Loss function(s):** The paper discusses squared-error and likelihood-related losses for the distance scores. A competitive loss pushes the correct class's energy below the others. If a modern version replaces distances with ordinary class scores, it usually also changes the loss to cross-entropy.

**Optimization algorithm(s):** Historical training uses stochastic backpropagation methods that also consider how sharply the loss changes. The family has no single learning-rate schedule. A clearly **different** recipe appears in [Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html): its sigmoid/average-pooling adaptation uses SGD at 0.1 for ten Fashion-MNIST epochs. That is not the cheque system's schedule.

**Regularization techniques:** Local connections and shared weights limit what the network can learn, helping avoid memorization. Scaling and suitable image changes address handwriting variation. Original LeNet has neither batch normalization nor modern dropout.

**Backpropagation considerations:** Tanh flattens near its limits, weakening backward signals. Input scaling and starting weights therefore matter. Updates must follow C3's real connections and include the learned subsampling values. Replacing either changes the model.

**Parameter count / scaling behavior:** LeNet-5 has approximately **60,000 parameters**. Connecting all C3 maps or changing the output head changes this count. Shared filters keep the early network small; the final class mapping depends on the character vocabulary.

**Training paradigm:** Known character labels guide learning. A larger document system can also train several connected modules using labelled strings. This is not unsupervised pretraining.

**Hardware/parallelism considerations:** Modern CPUs and GPUs can easily hold the small model. Historical document throughput measured the whole recognition process, not only its neural calculations.

**Strengths and limitations:** LeNet builds useful image assumptions into a small model. Its low-resolution design and limited visible regions do not make it a ready-made solution for arbitrary photographs or complicated pages.

**Computational complexity / scalability notes:** Reading many possible character crops may cost more than one network pass. **Optional math:** Convolution work includes terms $`HWq^2C_{\rm in}C_{\rm out}`$, adjusted for missing channel connections, plus dense matrix products. H and W are map dimensions, q is filter side, and C counts channels.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** The [publisher's account](https://proceedingsoftheieee.ieee.org/gradient-based-learning-applied-to-document-recognition/) describes a commercially deployed **graph-transformer cheque-reading system containing CNN character recognizers**. It extracts fields and tries possible character divisions. Recognizer scores guide candidate digit strings; jointly trained modules choose a reading or support rejecting it.

The system tackles character recognition and segmentation together, rather than assuming perfect crops. This documents a real application, **not that every deployed module was unchanged LeNet-5**. It supplies neither isolated LeNet accuracy nor a financial saving. The reported substantial daily volume concerns the full system, not an architecture-specific business gain.

**Notable vendor implementations/libraries:** Historical author code and modern PyTorch, Keras, and D2L versions exist. Compare activations, pooling, padding, C3 connections, and the output head before treating versions as equivalent.

### 1.6.3 AlexNet
**In plain English:** AlexNet learns several levels of image patterns to recognize objects in photographs. Its large network showed how GPUs could make this kind of training practical.

**Name:** AlexNet, the 2012 ImageNet network. Later models or saved weights with this name may differ.

**Category & sub-category:** Supervised learning; convolutional families. Deep image classification trained at large scale on GPUs.

**Originating paper/vendor/year:** Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, University of Toronto, [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), NeurIPS, then called NIPS, 2012.

**Core mechanism:** Five convolution layers build visual features, then three dense layers classify them. ReLU trained faster than the flattening activations tested in the paper. Overlapping max-pooling shrinks maps. Local response normalization (LRN) rescales responses using nearby channels. Some convolutions connect channels in groups because the original network was split across two GPUs. This hardware choice is part of the published design.

**Inputs/outputs and typical data types:** RGB photograph crops become probabilities for 1,000 ImageNet classes. The paper describes 224-by-224 crops. Versions using 227 pixels or different padding do not necessarily have identical shapes.

**Architecture diagram description:** FC means fully connected, or dense. Selected convolution groups run on separate GPUs in the original.

```text
RGB crop -> Conv11/stride4 -> ReLU/LRN/pool
         -> Conv5 -> ReLU/LRN/pool
         -> Conv3 -> Conv3 -> Conv3 -> pool
         -> FC4096 -> FC4096 -> FC1000 -> softmax
            [selected convolutional groups split across two GPUs]
```

**Activation functions used and why:** Hidden layers use ReLU; the output uses softmax. LRN rescales activations across channels. It is not batch normalization, which this model did not use.

**Loss function(s):** Class cross-entropy, also called multinomial negative log-likelihood, penalizes low probability for the image label. Training also uses weight decay to discourage large weights.

**Optimization algorithm(s):** SGD uses momentum 0.9, batch size 128, and weight decay 0.0005. The initial rate is 0.01. When validation improvement stalls, it is manually divided by ten; the paper reports three reductions.

**Regularization techniques:** The first two dense layers use dropout 0.5. Training changes crops, mirrors images horizontally, and varies RGB intensity using principal-component-based changes. Averaging test crops is a separate prediction procedure, not a training regularizer.

**Backpropagation considerations:** ReLU reduces flattened responses but can still leave units inactive or training unstable. Each shared filter gathers errors from all positions. Some connections need cross-GPU communication. Reproducing the paper requires its normalization and group layout.

**Parameter count / scaling behavior:** AlexNet has about **60 million parameters**, many in its dense layers. A smaller classifier can greatly reduce storage without removing much of the convolution work.

**Training paradigm:** ImageNet labels supervise training. Two members of the best 2012 ensemble also used **labelled** ImageNet Fall 2011 pretraining. That extra stage was not self-supervised learning.

**Hardware/parallelism considerations:** The original used two NVIDIA GPUs with explicitly divided work. A modern single accelerator may fit the model, but does not reproduce its historical timing.

**Strengths and limitations:** AlexNet demonstrated large-scale learned image features and useful GPU training. Its heavy dense head, early sharp downsampling, and older normalization often make it an inefficient starting point today.

**Computational complexity / scalability notes:** Add every convolution and dense layer's work. Grouped convolutions use fewer channel connections. Ten crops mean about ten image passes even for one trained model; averaging several models adds further work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) reports **18.2% top-5 validation error** for one ILSVRC-2012 CNN. Top-5 error means the correct label is missing from the five highest-ranked answers. Its **seven-network ensemble** reports **15.3% top-5 test error**. Two members first trained on the approximately 15-million-image, 22,000-category Fall 2011 release, then were fine-tuned. Five ordinary models gave 16.4%; **that was not a single-model result**. The [challenge results](https://image-net.org/challenges/LSVRC/2012/results.html) also separate supplied-data-only and extra-data entries.

Each crop produces class scores. Averaging them yields five candidate labels. Learning features from end to end, rather than fixing them by hand, motivates this approach. These scores establish no commercial photo-tagging benefit.

**Notable vendor implementations/libraries:** The authors' CUDA-convnet code and torchvision's `alexnet` belong to this family. Torchvision uses a later adaptation; identify its weights and test procedure separately.

### 1.6.4 VGG
**In plain English:** VGG builds an image recognizer by repeating small pattern detectors. Its simple layout makes deeper image features easier to study, but uses substantial memory.

**Name:** VGG. The main reference is VGG-16 configuration D; VGG-19 configuration E is a deeper version.

**Category & sub-category:** Supervised learning; convolutional families. Deep stacks repeatedly apply small convolutions.

**Originating paper/vendor/year:** Karen Simonyan and Andrew Zisserman of Oxford's Visual Geometry Group wrote [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/html/1409.1556v6). The preprint appeared in 2014; the ICLR conference paper appeared in 2015.

**Core mechanism:** Repeated 3-by-3 filters combine information from larger regions without using a large filter at each layer. Two such layers moving one pixel at a time see a 5-by-5 region together. They also add an activation between them, unlike one large filter. Pooling shrinks maps as channel counts grow. A large dense head combines the final features. Repeating the same design made depth easier to compare.

**Inputs/outputs and typical data types:** RGB images become scores for 1,000 classes. Earlier features can support other image tasks. Training uses 224-pixel crops. Dense evaluation applies the network across a larger resized image rather than only one crop.

**Architecture diagram description:** The repeated counts specify VGG-16 configuration D. Five pooling stages lead to three dense layers.

```text
RGB -> [Conv3x3(64)]x2 -> pool -> [Conv3x3(128)]x2 -> pool
    -> [Conv3x3(256)]x3 -> pool -> [Conv3x3(512)]x3 -> pool
    -> [Conv3x3(512)]x3 -> pool -> FC4096 -> FC4096 -> FC1000
    -> softmax                          [VGG-16 configuration D]
```

**Activation functions used and why:** ReLU follows the hidden weight layers; softmax produces class probabilities. Main VGG-16/19 versions do not use batch normalization. "VGG16-BN" is a later variant, not the same historical model.

**Loss function(s):** Class cross-entropy, also called multinomial logistic loss, rewards probability on the correct class. Weight decay discourages large weights.

**Optimization algorithm(s):** Minibatch SGD uses momentum 0.9, batch size 256, and initial rate 0.01. The rate falls by ten when validation accuracy stalls. The paper describes three drops and 74 epochs. Some deeper runs start certain layers from a trained shallower network.

**Regularization techniques:** The first two dense layers use dropout 0.5. Training also uses random crops, reflections, and, in selected runs, changes in image scale. **Optional math:** Weight decay is $`5\times10^{-4}`$, meaning 0.0005. These training choices contribute to the score; depth alone does not explain it.

**Backpropagation considerations:** A long stack without shortcuts can be sensitive to starting weights and backward signal size. The paper's use of shallower trained layers is part of its method. Substituting modern starting-weight rules would not reproduce the historical recipe exactly.

**Parameter count / scaling behavior:** The source reports about **138 million parameters for VGG-16** and **144 million for VGG-19**. Dense layers take much of the weight storage. Early convolutions process large maps and account for substantial calculation.

**Training paradigm:** ImageNet class labels supervise learning. Other labelled tasks can reuse or fine-tune these features. Later using VGG in an unsupervised system does not change how its original weights were learned.

**Hardware/parallelism considerations:** Repeated 3-by-3 convolutions suit optimized accelerators. Large weight arrays, dense-layer data transfers, and intermediate maps can still make VGG expensive compared with compact networks.

**Strengths and limitations:** The regular design is easy to inspect and adapt, and learns features at several scales. Its large dense head and lack of residual shortcuts make further expansion less appealing than in later designs.

**Computational complexity / scalability notes:** More layers broaden the visible image region, but also repeat costly calculations. **Optional math:** A 3-by-3 layer costs about $`O(9BHW C_{\rm in}C_{\rm out})`$. B counts images in a batch; H and W are map dimensions; C counts input and output channels. Storing maps and repeating wide layers remain important costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 3](https://arxiv.org/html/1409.1556v6) reports **25.6% top-1 and 8.1% top-5 error** for configuration D on **ImageNet validation**. Top-1 checks the first label; top-5 checks whether the correct label appears among five choices. Training varies image side over [256,512]; evaluation uses side 384. This is **one model, one test scale, and dense evaluation**, not one 224-pixel crop or an ensemble.

A resized photograph produces a map of class scores. Scores across positions and reflected images are combined to rank labels. Compared with AlexNet, the design offers a simpler stack of small filters. The result does not show that depth alone caused the gain, or establish commercial image-search value.

**Notable vendor implementations/libraries:** Oxford's released models, torchvision VGG, and Keras VGG16/VGG19 provide versions. Check batch normalization and input preparation, including RGB/BGR channel order and mean subtraction.

### 1.6.5 ResNet
**In plain English:** ResNet lets each block improve an existing representation instead of replacing it. Shortcut paths carry earlier information forward, making very deep image networks easier to train.

**Name:** Residual network, or ResNet. This entry covers the original post-activation family, including bottleneck ResNet-50/101/152.

**Category & sub-category:** Supervised learning; convolutional families. Residual blocks learn corrections to image features.

**Originating paper/vendor/year:** Microsoft Research's Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun wrote [Deep Residual Learning for Image Recognition](https://arxiv.org/html/1512.03385v1). It appeared as a preprint in 2015 and a CVPR paper in 2016.

**Core mechanism:** A block calculates a correction, adds the unchanged input through a shortcut, then applies ReLU. If the two paths have different sizes, a learned projection or specified padding aligns them. A bottleneck first reduces channels with a 1-by-1 filter, applies a 3-by-3 filter, then expands channels again. This lets a new block make a small useful change rather than relearn the entire transformation.

**Optional math:** $`y=\operatorname{ReLU}(x+F(x))`$. Here x is the block input, F is its learned correction, and y is the output. ReLU comes after the sum in the original post-activation design.

**Inputs/outputs and typical data types:** Images become class probabilities or feature maps for object finding and pixel labelling. The classifier averages each final feature map across its spatial positions.

**Architecture diagram description:** BN means batch normalization, which rescales layer values using training-batch statistics. The shortcut and correction join by addition.

```text
image -> convolutional stem -> residual stages -> global average -> classifier
                              |
block: x -> 1x1/BN/ReLU -> 3x3/BN/ReLU -> 1x1/BN --+
       x ---------------- identity/projection ----------------+ -> add -> ReLU
```

**Activation functions used and why:** Original blocks use ReLU and batch normalization; classification uses softmax. Later pre-activation ResNets move normalization and activations relative to the sum. They are different designs.

**Loss function(s):** The ImageNet classifier uses softmax cross-entropy. Object detection or pixel segmentation adds losses suited to those different tasks.

**Optimization algorithm(s):** SGD uses momentum 0.9, batch size 256, and initial rate 0.1. The rate falls by ten when error levels off. The paper allows up to 600,000 iterations. This should not automatically be replaced by a later "90-epoch ResNet recipe."

**Regularization techniques:** Training uses crops, flips, and batch normalization, but no dropout in the original classification experiments. **Optional math:** Weight decay is $`10^{-4}`$, or 0.0001. Batch normalization must use the correct statistics during training and prediction.

**Backpropagation considerations:** Shortcuts help information and backward error signals travel through deep networks. They do not guarantee successful training. The paper's problem was not simply that every gradient vanished: deeper plain networks could have worse training error even with normalization.

**Parameter count / scaling behavior:** ResNet-152 has roughly **60 million parameters**, depending on the head and implementation. Bottlenecks make depth less costly than equally wide stacks using full-width 3-by-3 layers throughout.

**Training paradigm:** Class labels supervise training; other labelled tasks can reuse the main feature network, or backbone. Contrastive ResNet pretraining uses a different objective and belongs in the representation-learning chapters.

**Hardware/parallelism considerations:** Dense convolution work suits GPUs and other accelerators. Shortcuts add little arithmetic but require keeping earlier values in memory. Normalization and small batches can also limit efficient execution.

**Strengths and limitations:** Residual blocks make deep, reusable feature networks practical. More depth still adds delay and memory use. Shortcuts do not prevent overfitting, reliance on misleading clues, or failure when new data differ from training data.

**Computational complexity / scalability notes:** Narrowing the middle of a block reduces the expensive spatial filtering. **Optional math:** Its work is about $`BHW(Cr+9r^2+rC')`$. B is batch size, H and W are map dimensions, C and C' are input/output widths, and r is reduced width. This counts two projections and one narrow 3-by-3 filter, not three full-width spatial filters.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original paper](https://arxiv.org/html/1512.03385v1) reports **4.49% top-5 ImageNet validation error** for a **single ResNet-152**. This uses its best fully convolutional, multiscale evaluation, not one crop. The famous **3.57%** instead belongs to the **six-model ensemble on the ILSVRC-2015 test set**.

The image passes through residual corrections, spatial averaging, and class scoring. Evaluation checks whether the true class is among the five highest scores. The authors sought to fix worsening training error in deeper plain networks. These results do not identify a Microsoft product deployment or a business gain from challenge rank.

**Notable vendor implementations/libraries:** Torchvision, Keras applications, and detection frameworks offer ResNets. ResNet-v1, pre-activation models, and torchvision's placement of bottleneck stride differ. Choose a specific network and checkpoint.

### 1.6.6 Inception
**In plain English:** Inception checks an image with several sizes of pattern detector in parallel. It combines their results while using narrow intermediate layers to control the cost.

**Name:** Inception family, using **GoogLeNet/Inception-v1** as the reference. Later versions change how filters are split and values are normalized.

**Category & sub-category:** Supervised learning; convolutional families. Several branches gather image patterns at different scales.

**Originating paper/vendor/year:** Christian Szegedy and colleagues, Google and collaborating researchers, [Going Deeper with Convolutions](https://arxiv.org/html/1409.4842v1), 2014 preprint and CVPR 2015 paper.

**Core mechanism:** Each module sends the same input down four paths. One uses a 1-by-1 filter, two use narrowed 3-by-3 or 5-by-5 filters, and one pools before a projection. The outputs are placed side by side as channels, called concatenation. Narrowing channels before large filters controls cost. The next stage can use both local details and broader patterns.

**Inputs/outputs and typical data types:** RGB images become class probabilities or reusable feature maps. GoogLeNet's main path is 22 layers deep when counting layers with parameters.

**Architecture diagram description:** Four parallel branches join by channel concatenation. Extra classification heads assist training but are absent during prediction.

```text
                         +-> 1x1 -----------------+
input feature map -------+-> 1x1 -> 3x3 -----------+
                         +-> 1x1 -> 5x5 -----------+-> channel concat
                         +-> pool -> 1x1 ---------+
stem -> repeated modules -> global average -> dropout -> classifier
                         \-> auxiliary heads during training only
```

**Activation functions used and why:** Branches use ReLU; classification heads use softmax. Inception-v1 is not the later batch-normalized Inception-v2/v3 family.

**Loss function(s):** Training adds the main cross-entropy loss and extra classifier losses, each weighted 0.3. Those auxiliary heads are removed for inference, meaning ordinary predictions after training.

**Optimization algorithm(s):** The paper uses asynchronous SGD, where workers update without waiting for every other worker, with momentum 0.9. The learning rate falls by 4% every eight epochs. Procedures changed across ensemble members, so no universal starting rate is reconstructed here. Polyak averaging averages learned parameter values for prediction.

**Regularization techniques:** Training changes crop and image scale, uses dropout, and adds the extra supervised heads. The architecture table lists 40% dropout at the main classifier. Averaging maps across positions avoids a huge fully connected head.

**Backpropagation considerations:** Main and auxiliary losses send signals to earlier layers along different routes. Branch outputs need matching spatial dimensions before they can join. The extra heads change the training objective, not only the route for its gradients.

**Parameter count / scaling behavior:** Layerwise counts give roughly **seven million parameters** for the main network. Training-only heads add more. Without channel-narrowing bottlenecks, the multiple branches would be much more expensive.

**Training paradigm:** ImageNet labels supervise classification. The paper's motivation for a "sparse" architecture does not mean that a router selects different experts for each token.

**Hardware/parallelism considerations:** Branches can run in parallel. Joining their outputs, waiting for branches, and executing small operations can reduce the gain. The research setup for distributed training is not a universal production recipe.

**Strengths and limitations:** Inception combines several image scales with a compact main classifier. Branch sizes and connections are harder to tune than VGG's repeated pattern. Fewer arithmetic operations, often counted as FLOPs, do not guarantee equally faster predictions.

**Computational complexity / scalability notes:** Add the cost of every branch, including pooling. **Optional math:** A 1-by-1 reduction from C to r channels changes a later 5-by-5 filter from $`25HWC C'`$ work to $`25HWrC'`$, plus $`HWCr`$ for the reduction. H and W are map dimensions; C' is output width. The saving depends on choosing r smaller than C.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [2014 preprint](https://arxiv.org/html/1409.4842v1) reports **6.67% top-5 error** on validation and test evaluations for its final ILSVRC-2014 submission. It averages **seven models** and **144 crops per image**. This is not one pass through one GoogLeNet.

Each crop supplies class probabilities; averaging them produces ranked labels. Missing the correct label from the first five counts as an error. Multiscale features motivated the design, but extensive score averaging also contributes to the result. The benchmark does not establish a measured deployment inside a Google product.

**Notable vendor implementations/libraries:** TensorFlow-Slim and torchvision `googlenet` implement this family. Inception-v3 builders create a substantially different network and cannot inherit v1's benchmark claims.

### 1.6.7 EfficientNet
**In plain English:** EfficientNet grows an image network in several balanced ways instead of only adding layers. It aims to get useful recognition accuracy for a stated amount of computing work.

**Name:** EfficientNet, specifically the original B0-B7 family. EfficientNetV2 and later self-trained checkpoints are separate.

**Category & sub-category:** Supervised learning; convolutional families. These networks jointly scale depth, channel width, and image resolution.

**Originating paper/vendor/year:** Mingxing Tan and Quoc Le, Google, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html), ICML 2019. Numerical details here follow [arXiv v5](https://arxiv.org/html/1905.11946v5).

**Core mechanism:** A search process chooses the small B0 starting network. Its blocks expand channels, filter each channel separately, adjust each channel's importance, then narrow the output. These are mobile inverted bottlenecks, or MBConv blocks. "Squeeze-and-excitation" means summarizing channels and using those summaries to control their strengths. Larger models add layers, widen channels, and enlarge inputs together.

**Optional math:** Depth, width, and resolution grow by $`\alpha^\phi,\beta^\phi,\gamma^\phi`$. Alpha, beta, and gamma are chosen growth factors; phi sets the model's scale. The approximate rule $`\alpha\beta^2\gamma^2\approx2`$ balances their estimated computation. It is not a measured timing law.

**Inputs/outputs and typical data types:** RGB images at a variant's chosen resolution become class probabilities. B0 uses reference resolution 224. Larger variants need larger images and more memory for intermediate maps.

**Architecture diagram description:** Each channel is filtered separately in the depthwise step. A compatible input can also travel along the residual shortcut.

```text
image -> stem -> repeated MBConv stages -> 1x1 head -> global average -> classifier
MBConv: expand -> SiLU -> depthwise -> SiLU -> squeeze/excite -> linear project
         input ---------------- compatible residual shortcut ----------------+
```

**Activation functions used and why:** SiLU, also called Swish-1, smoothly gates a value rather than sharply cutting it off. Sigmoid gates assign channel strengths between zero and one. The narrowing projection stays linear. Softmax produces class probabilities.

**Loss function(s):** Image labels guide cross-entropy training. Later teacher-student training or self-training may change the loss; those changes are not part of this architecture's definition.

**Optimization algorithm(s):** V5 uses RMSProp, which adjusts steps using recent gradient sizes, with decay 0.9 and momentum 0.9. Its initial learning rate 0.256 is multiplied by 0.97 every 2.4 epochs. These settings belong to the paper's training setup, not arbitrary batch sizes.

**Regularization techniques:** Training uses batch normalization and AutoAugment's selected image changes. Stochastic depth sometimes skips blocks; the reported survival probability is 0.8. Dropout rises from 0.2 in B0 to 0.5 in B7. A 25,000-image training minival subset guides when to stop and keep a checkpoint. **Optional math:** Weight decay is $`10^{-5}`$, or 0.00001.

**Backpropagation considerations:** Shortcuts and normalization help backward signals. Channel gates can weaken them. Larger images also restrict batch size through memory use; changing batch size may require new normalization and optimizer settings.

**Parameter count / scaling behavior:** V5 lists **5.3 million parameters for B0** and **66 million for B7**. Widening both sides of pointwise filters can roughly square their weight growth. Larger images mostly increase calculations and stored responses, not filter weights.

**Training paradigm:** The quoted B-family study uses supervised ImageNet training and, where relevant, supervised transfer to other tasks. Noisy Student and other extra-data stages are different recipes.

**Hardware/parallelism considerations:** Channel-mixing matrix products and per-channel filters use hardware differently. Larger images can erase a small-parameter model's memory advantage. Any speed comparison must name the runtime and accelerator.

**Strengths and limitations:** Balanced scaling offers a way to trade accuracy against resources. The best balance depends on the device and task. Searching for B0 costs extra work beyond training a design already chosen.

**Computational complexity / scalability notes:** Filtering each channel and mixing channels have different costs. **Optional math:** Their main terms are $`HWq^2C`$ and $`HWC C'`$, plus expansion and channel gating. H and W are map dimensions, q is filter side, and C and C' are channel widths. A scaling step's approximate doubled work does not promise doubled elapsed time.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [V5 Table 2](https://arxiv.org/html/1905.11946v5) gives **EfficientNet-B0 77.1% top-1 and 93.3% top-5 ImageNet validation accuracy**, using **one model and one crop**. Top-1 checks the first choice; top-5 allows any of five. A 224-pixel image passes through bottlenecks and channel gates to choose an object label. This could support visual cataloguing, but no deployed business result is reported.

Resource-aware growth, rather than simply widening everything, motivates the comparison. Version differences matter: v5 lists B7 at **84.3% top-1**, while the ICML landing-page abstract says **84.4%**. These are not silently combined into one checkpoint result.

**Notable vendor implementations/libraries:** Google releases, Keras applications, torchvision, and timm offer EfficientNet versions. Record the exact weights, input resolution, and any additional-data training.

### 1.6.8 ConvNeXt
**In plain English:** ConvNeXt updates the design and training of a convolutional image network. It tests whether small pattern detectors can remain competitive without using attention layers.

**Name:** ConvNeXt, using original **ConvNeXt-Tiny**, not ConvNeXt V2.

**Category & sub-category:** Supervised learning; convolutional families. A modern convolution-only network supplies image features.

**Originating paper/vendor/year:** Zhuang Liu and colleagues, Meta AI and academic collaborators, [A ConvNet for the 2020s](https://arxiv.org/html/2201.03545v2), CVPR 2022.

**Core mechanism:** ConvNeXt revises a ResNet-style hierarchy. It starts with patch-like downsampling, uses large per-channel filters, expands and narrows channels, and uses fewer activations. Layer normalization rescales each example's internal features rather than relying on other examples in a batch. The model remains convolutional. It does not use attention's input-dependent weighted lookups. The paper improves training before comparing designs, helping separate training effects from architecture effects.

**Inputs/outputs and typical data types:** RGB images become class predictions or feature maps at several scales. Tiny's four stages have widths 96, 192, 384, and 768.

**Architecture diagram description:** The stage counts are [3,3,9,3]. LayerScale controls the correction's size before it joins the shortcut.

```text
RGB -> 4x4/stride4 stem -> stages [3,3,9,3] -> global average -> linear head
block: x -> depthwise 7x7 -> LayerNorm -> pointwise expand 4x
         -> GELU -> pointwise project -> LayerScale -> stochastic depth -> +x
```

**Activation functions used and why:** GELU smoothly reduces less useful responses instead of using ReLU's sharp cutoff. Layer normalization controls feature scale. The head produces raw class scores, called logits, for cross-entropy training.

**Loss function(s):** Supervised cross-entropy uses softened labels. Label smoothing avoids assigning all target weight to one class. Mixup blends images and labels; CutMix combines image regions and their labels. The targets still come from supplied labels.

**Optimization algorithm(s):** ImageNet-1K training uses AdamW, which separates weight decay from adaptive updates. Settings are learning rate 0.004, batch size 4096, 300 epochs, and weight decay 0.05. The rate rises for 20 warmup epochs, then follows a smooth cosine decline. ImageNet-22K pretraining/fine-tuning is a separate recipe from the Tiny 1K-only result.

**Regularization techniques:** Training uses Mixup, CutMix, RandAugment image changes, random erasing, label smoothing, and stochastic depth. An exponential moving average keeps a running weighted average of model weights. **Optional math:** LayerScale starts at $`10^{-6}`$, a very small initial multiplier for residual corrections.

**Backpropagation considerations:** Shortcuts and initially small corrections help deep training stay stable. Normalization placement, block skipping, and numerical precision can change behavior. A matching model name is not enough to reproduce the setup.

**Parameter count / scaling behavior:** ConvNeXt-T has **29 million parameters** in the paper. Larger versions increase width and stage depth. **Optional math:** A depthwise filter needs only $`q^2C`$ weights, where q is filter side and C counts channels. Channel-expanding pointwise layers hold much of the remaining capacity.

**Training paradigm:** This Tiny result uses supervised ImageNet-1K labels. Masked-image learning and ConvNeXt V2 are different methods, not implied by the original 2022 name.

**Hardware/parallelism considerations:** Efficient depthwise operations and channels-last memory layout can help. Normalization and data rearrangement take time beyond multiplication counts. Equal FLOPs do not mean equal latency for ConvNeXt and a Transformer.

**Strengths and limitations:** ConvNeXt offers a convolution-based alternative to attention networks. Its results also depend on extensive image changes and long training. Comparing only network diagrams can therefore mislead.

**Computational complexity / scalability notes:** Wider blocks make channel mixing much more expensive. **Optional math:** A block costs about $`HW(q^2C+8C^2)`$, with map dimensions H and W, filter side q, and channel width C. This counts depthwise filtering and two 4x pointwise projections, excluding small elementwise steps. Fixed-resolution prediction remains a sequence of spatial calculations.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ImageNet-1K table](https://arxiv.org/html/2201.03545v2) reports **82.1% top-1 validation accuracy** for **ConvNeXt-T at 224-by-224**, versus **81.3%** for the listed Swin-T. Top-1 checks the highest-scored class. This compares single classifiers trained on ImageNet-1K, not later 22K-pretrained versions.

An image becomes features at several resolutions, then an averaged class decision. The authors ask whether updated convolutional design and training can compete with hierarchical Transformers. The difference does not measure commercial inspection, search, or sales gains.

**Notable vendor implementations/libraries:** Meta's ConvNeXt release, torchvision, and timm offer models. ConvNeXt V2 and differently trained weights need their own source and recipe information.

### 1.6.9 DenseNet
**In plain English:** DenseNet keeps earlier image features available to later layers. Instead of repeatedly replacing them, it adds new features to a growing collection.

**Name:** Densely connected convolutional network. The reference is **DenseNet-121 with bottleneck and compression**.

**Category & sub-category:** Supervised learning; convolutional families. Concatenation lets later layers reuse earlier features.

**Originating paper/vendor/year:** Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Weinberger wrote [Densely Connected Convolutional Networks](https://arxiv.org/html/1608.06993v5). Its preprint appeared in 2016 and its CVPR paper in 2017. The cited v5 revision is dated 2018.

**Core mechanism:** Every layer in a dense block reads all earlier feature maps in that block. It adds a small number of new maps to the collection. Unlike ResNet, it keeps old and new channels separate rather than adding them together. A 1-by-1 bottleneck narrows inputs before 3-by-3 filtering. Transition layers shrink maps and compress channel counts between blocks.

**Inputs/outputs and typical data types:** Images become class scores or feature maps from several stages. DenseNet-121 has four dense blocks containing 6, 12, 24, and 16 composite layers. Its growth rate is 32 new maps per layer.

**Architecture diagram description:** Brackets collect earlier maps side by side. Each H transformation adds k maps, where k is the growth rate.

```text
stem -> dense block -> transition -> dense block -> ... -> global average -> head
dense block:
x0 ------------+--------------------+
x1 = H1(x0) ---+--> x2=H2([x0,x1]) -+--> x3=H3([x0,x1,x2])
H: BN/ReLU/1x1 bottleneck -> BN/ReLU/3x3 -> k new maps
```

**Activation functions used and why:** Batch normalization and ReLU come before the bottleneck and spatial filters. Classification uses softmax cross-entropy. Concatenation preserves separate features; it is not itself an activation.

**Loss function(s):** Supplied class labels guide cross-entropy. The phrase "implicit deep supervision" refers to short backward paths. It does not mean every layer has a separate labelled output head.

**Optimization algorithm(s):** ImageNet training uses SGD with Nesterov momentum 0.9, batch size 256, and 90 epochs. Initial rate 0.1 falls by ten at epochs 30 and 60.

**Regularization techniques:** Training uses image changes and batch normalization. **Optional math:** Weight decay is $`10^{-4}`$, or 0.0001. The paper's dropout 0.2 applies to specified small datasets without augmentation, not automatically to its ImageNet recipe.

**Backpropagation considerations:** Direct access to earlier maps creates many short routes for error signals. Saving repeated concatenations and normalized copies can use far more memory than the weights. Memory-efficient versions recalculate selected intermediate values instead of keeping them all.

**Parameter count / scaling behavior:** DenseNet-121 has about **eight million parameters**. **Optional math:** After ell additions, block width grows as $`C_0+\ell k`$, where C0 is starting width and k is growth rate. Bottlenecks and transition compression keep that growth manageable.

**Training paradigm:** The quoted result uses supervised image classification. The same connection pattern can appear in generative or self-supervised models with different objectives.

**Hardware/parallelism considerations:** Keeping and joining earlier maps increases memory traffic. Fewer weights than a residual network do not necessarily mean less training memory.

**Strengths and limitations:** DenseNet reuses features with relatively few parameters. Its growing feature history complicates memory planning. A simple implementation may run slowly despite a modest arithmetic count.

**Computational complexity / scalability notes:** At fixed resolution and growth rate, doubling block depth can roughly quadruple part of the work because each layer reads more maps. Unique new maps grow only linearly. Repeatedly copied concatenation buffers can grow quadratically. These are different quantities; quadratic memory is not unavoidable in every implementation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [V5 Table 3](https://arxiv.org/html/1608.06993v5) lists **25.02% top-1 and 7.71% top-5 error** for DenseNet-121 on **ImageNet validation with one 224-pixel crop**. With **10 crops**, errors are **23.61% and 6.66%**. These measure whether the first choice, or any of five choices, contains the correct label.

The photograph builds a retained collection of simple and complex features. Pooling and a classifier turn it into ranked labels. Preserving features, rather than summing them as ResNet does, motivates the design. These ImageNet results do not establish performance in deployed radiology or industrial inspection.

**Notable vendor implementations/libraries:** Author releases, torchvision DenseNet, and Keras DenseNet offer versions. Memory-efficient settings can exchange extra calculation for less storage without changing the mathematical prediction.

### 1.6.10 MobileNet
**In plain English:** MobileNet splits image filtering into two cheaper jobs: finding patterns within channels and mixing channels. It offers smaller networks for devices with limited computing resources.

**Name:** MobileNet, specifically **MobileNet-v1, width multiplier 1.0, 224-pixel input**. The example uses the 2018-08-02 TensorFlow-Slim checkpoint bundle. V2's inverted residuals and v3's searched blocks are separate extensions.

**Category & sub-category:** Supervised learning; convolutional families. Depthwise-separable filters reduce image-processing work.

**Originating paper/vendor/year:** Andrew Howard and colleagues at Google published the technical report [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/html/1704.04861v1) in 2017.

**Core mechanism:** A standard convolution finds spatial patterns and mixes channels in one operation. V1 first filters each channel separately, then uses 1-by-1 filters to mix them. A width multiplier, alpha, reduces channel counts; a resolution multiplier reduces spatial work. These are adjustable trade-offs, not guarantees that every variant fits every device.

**Inputs/outputs and typical data types:** RGB images become class probabilities or compact reusable features. ImageNet has 1,000 classes here, but the released Slim evaluator uses 1,001 output scores for its indexing convention. Preserve that label mapping.

**Architecture diagram description:** The main block separates spatial filtering from channel mixing. Selected strides shrink the image maps.

```text
RGB -> standard Conv3x3 -> depthwise/pointwise blocks -> global average -> head
block: depthwise 3x3 -> BN -> ReLU6 -> pointwise 1x1 -> BN -> ReLU6
       [released TensorFlow-Slim activation convention]
selected depthwise strides downsample the feature map
```

**Activation functions used and why:** The paper describes ReLU. The [released TensorFlow-Slim code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) instead specifies ReLU6, which clips positive responses at six, after normalized depthwise and pointwise steps. Softmax supplies class probabilities. These are distinct, identified conventions.

**Loss function(s):** Image-class cross-entropy uses supplied labels. Detection versions need additional losses for object locations as well as classes.

**Optimization algorithm(s):** The paper specifies RMSProp and asynchronous gradient descent, but **does not provide a complete independent learning-rate schedule**. This is not a complete training record for the 2018 saved model. Later trainer defaults do not prove how that checkpoint was trained.

**Regularization techniques:** The setup uses batch normalization and reduced image augmentation. Depthwise filters receive little or no weight decay. The authors do not use auxiliary heads or label smoothing here.

**Backpropagation considerations:** Error signals pass through the separate filtering and mixing steps. Strong penalties can overwhelm the very small depthwise weight sets. Extremely narrow channels also limit what the network can represent; a better optimizer cannot recover missing capacity.

**Parameter count / scaling behavior:** The paper lists **4.2 million parameters** and **569 million multiply-adds**; the released checkpoint table lists **4.24 million parameters**. **Optional math:** With width multiplier $`\alpha`$, pointwise costs scale roughly as $`\alpha^2`$, while depthwise costs scale as $`\alpha`$. First and last layers require separate counts.

**Training paradigm:** The reference uses supervised ImageNet labels. Transfer learning or learning from a teacher model creates a separately trained checkpoint and needs a separate description.

**Hardware/parallelism considerations:** Depthwise filters reduce calculations but may spend much of their time moving data. Runtime support, combining operations, lower-precision numbers, batch size, and device all affect speed. The name "Mobile" alone proves no battery saving.

**Strengths and limitations:** Width and resolution give clear controls over size and work. Reducing them can lower accuracy, especially when the task needs fine detail or many different features.

**Computational complexity / scalability notes:** Splitting the two jobs reduces arithmetic, not necessarily elapsed time. **Optional math:** A separated layer costs $`O(BHW(q^2C_{\rm in}+C_{\rm in}C_{\rm out}))`$, versus $`O(BHWq^2C_{\rm in}C_{\rm out})`$ for a standard filter. B is batch size, H and W are map dimensions, q is filter side, and C counts channels. Their approximate work ratio is $`1/C_{\rm out}+1/q^2`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The official [2018-08-02 checkpoint bundle](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) lists **70.9% top-1 and 89.9% top-5 accuracy** for floating-point `MobileNet_v1_1.0_224`. The [evaluator](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_eval.py) targets all **50,000 ImageNet validation images**, using a [single central crop resized to 224](https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py). Top-1 checks one label; top-5 checks five. This is not the paper's separate 70.6% record.

The image becomes low-cost spatial and channel features, then a class choice. That could serve an on-device recognition interface. Limited-resource vision motivates the design, but the score and work count prove neither a named phone deployment nor measured energy savings.

**Notable vendor implementations/libraries:** TensorFlow-Slim, Keras MobileNet, and mobile runtimes support this family. Check generation, activation, resolution, width multiplier, and the saved model's evaluation procedure.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Generic CNN | Images arranged as pixel grids | Reuses small learned pattern detectors | Can fail at new resolutions or on unfamiliar images | Keras MNIST digit-recognition test |
| LeNet | Small grayscale character images | Compact learned stroke features | Whole documents need extra processing stages | CNNs within the published commercial cheque reader |
| AlexNet | Natural RGB photographs | Learns large-scale image features | Heavy dense head and older hardware-driven choices | ILSVRC-2012; single-model and ensemble scores kept separate |
| VGG | Images needing reusable features at several scales | Simple repeated small filters | Many weights and calculations | ImageNet test using dense evaluation at one scale |
| ResNet | Images and reusable multistage features | Shortcuts help train deeper networks | Depth still costs time and memory | ResNet-152 ImageNet validation test |
| Inception | Images with both small and large patterns | Several filter sizes with narrowed channels | Complex branches and costly score averaging | GoogLeNet ILSVRC-2014 ensemble test |
| EfficientNet | Images with a stated computing budget | Balances layers, widths, and resolution | Efficiency depends on the device | B0 ImageNet single-crop test |
| ConvNeXt | Images and tasks needing spatial feature maps | Updated convolution-only design | Strong results also need substantial training | ConvNeXt-T ImageNet-1K test |
| DenseNet | Images where earlier features remain useful | Reuses features with relatively few weights | Retained maps can cost memory and data movement | DenseNet-121 ImageNet single-crop test |
| MobileNet | Images processed with limited resources | Splits filtering into cheaper steps | Fewer calculations do not guarantee lower delay | Versioned MobileNet-v1 ImageNet validation checkpoint |

## 1.7 Recurrent and sequence architectures

A sequence has an order, such as sound frames or words in a sentence. A recurrent network carries a **state**, a list of numbers, from one step to the next. This lets later calculations use earlier information. The state is not human memory. Other sequence models retrieve weighted information from earlier representations instead.

These models differ in what context they keep, how they match inputs to outputs, and which calculations can run together. Supplied translations or phoneme labels still provide supervision, even when a model generates its answer one token at a time. A token is a word or smaller text unit. A phoneme is a speech-sound category.

### 1.7.1 Vanilla recurrent neural network (RNN)
**In plain English:** An RNN reads a sequence in order and carries a changing numerical state. It can use earlier input when labelling later input, but may struggle to keep distant information.

**Name:** Vanilla RNN with tanh units. The measured speech model has three stacked levels, each reading both forward and backward.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures. It carries state without separate memory-control gates.

**Originating paper/vendor/year:** Jeffrey Elman's [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1), 1990, is a standard simple-RNN reference, not the origin of every recurrent design. This supervised example comes from [Graves, Mohamed, and Hinton's 2013 speech study](https://arxiv.org/html/1303.5778v1).

**Core mechanism:** At each step, the same layer combines the new input with its previous state. Tanh transforms that combination into the next state. Sharing weights allows different sequence lengths without adding weights per step. A bidirectional network also reads backward and combines both streams. Stacking levels adds another kind of depth: each level processes features from the previous one.

**Optional math:** $`h_t=\tanh(W_xx_t+W_hh_{t-1}+b)`$. Here t is the step, x is input, h is carried state, Wx and Wh are learned matrices, and b is the bias. Tanh bounds the new values.

**Inputs/outputs and typical data types:** Ordered sound features, sensor readings, or numeric symbol representations become labels at multiple steps or one final label. This speech model uses 123-component sound vectors and scores for phonemes plus a blank symbol.

**Architecture diagram description:** The first line shows carried state. The speech version stacks three forward/backward levels before turning frame scores into phonemes.

```text
x1 -> tanh state h1 -> tanh state h2 -> ... -> tanh state hT
                       ^ x2                         ^ xT
speech instance: forward + backward streams, stacked three levels
                -> frame softmax -> CTC decoding -> phoneme sequence
```

**Activation functions used and why:** Tanh keeps state values bounded and makes their update nonlinear. Softmax gives competing phoneme/blank probabilities. This simple cell has no sigmoid input, forget, or output gates.

**Loss function(s):** Connectionist temporal classification (CTC) learns from the final phoneme sequence without needing a label for every sound frame. It adds the probabilities of valid frame-by-frame paths that reduce to that sequence after repeats and blanks are handled.

**Optimization algorithm(s):** The comparison uses SGD with momentum 0.9 and uniformly chosen initial weights in [-0.1,0.1]. **Optional math:** Learning rate $`10^{-4}`$ means 0.0001. The reported recipe adds no decay schedule.

**Regularization techniques:** Inputs use training-set scaling statistics. A separate development set guides model selection. This tanh model uses **no Gaussian weight-noise phase**: it failed to learn with the noise used for most LSTM comparisons.

**Backpropagation considerations:** Training traces error signals backward through the carried states. This is backpropagation through time (BPTT). Repeated transformations can shrink or enlarge signals too much. Clipping caps large signals but cannot restore forgotten information. Cutting the backward history also limits which earlier steps get credit. This full-sequence experiment is not a shortened-history streaming recipe.

**Parameter count / scaling behavior:** Weights are reused at every step. **Optional math:** A simple layer has roughly $`dh+h^2+h`$ parameters before its head, with input width d and state width h. The paper's **CTC-3l-500h-tanh** has **3.7 million weights**, with 500 hidden units per direction at each level.

**Training paradigm:** Labelled speech guides sound-to-phoneme learning. Predicting text from unlabelled text would use a different, self-supervised objective with the same general architecture.

**Hardware/parallelism considerations:** Examples in a batch and calculations within a step can run together. Successive states still depend on one another. A backward-reading stream needs future input, so the whole bidirectional model cannot provide an equivalent immediate streaming answer.

**Strengths and limitations:** The simple repeated rule handles varying sequence lengths. Long-distance information, unstable training, and step-by-step execution are major limits compared with gated or attention-based models.

**Computational complexity / scalability notes:** Longer sequences need more state updates and stored training history. **Optional math:** One layer costs $`O(T(dh+h^2))`$ before the head; T is length, d input width, and h state width. Full BPTT stores about $`O(BTLh)`$ values for batch size B and L layers. CTC's path calculation commonly adds $`O(TU)`$ work for target length U.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On [TIMIT](https://arxiv.org/html/1303.5778v1), the tanh model has **37.6% phoneme error rate** on the **24-speaker core test set**; lower is better. Training uses the standard 462-speaker set without SA recordings, and 50 speakers form the development set. Training and decoding use 61 phoneme labels, reduced to 39 for scoring. Beam width 100 means decoding keeps a limited set of candidate paths.

Sound frames produce carried states, CTC paths, and a phoneme transcription. The roughly size-matched three-level LSTM gets 18.6%, exposing a real weakness of this RNN. These are single runs, with unknown variation from starting weights. They establish no deployed speech-product benefit.

**Notable vendor implementations/libraries:** PyTorch `RNN`, Keras `SimpleRNN`, and accelerator recurrent operations supply components. Bidirectional processing and CTC require additional configuration; not every RNN includes them.

### 1.7.2 Long short-term memory (LSTM)
**In plain English:** An LSTM carries information through a sequence using learned controls for keeping, adding, and revealing it. These controls can help useful details survive for longer.

**Name:** Long short-term memory network. The example stacks bidirectional LSTM levels with CTC. Its refined cell is not identical to the first 1997 version.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures. Gates control a carried memory state that updates by addition.

**Originating paper/vendor/year:** Sepp Hochreiter and Jurgen Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), Neural Computation, 1997. Forget gates and other refinements came later. The worked model is from [Graves, Mohamed, and Hinton, 2013](https://arxiv.org/html/1303.5778v1).

**Core mechanism:** A forget gate chooses how much old state to keep. An input gate controls new information added to it. An output gate controls what other layers can see. These gates are learned numbers, not switches chosen by a person. Adding to the cell state gives information and error signals a more direct route, especially when the forget gate stays near one. The speech model also uses peephole connections, letting gates inspect the cell state.

**Optional math:** $`c_t=f_t\odot c_{t-1}+i_t\odot g_t`$ and $`h_t=o_t\odot\tanh(c_t)`$. At step t, c is cell state, h is exposed state, f/i/o are forget/input/output gates, and g is new candidate information. The symbol $`\odot`$ means multiplying matching components.

**Inputs/outputs and typical data types:** Sound frames, several time-varying measurements, or token vectors become states and task predictions. This model uses 123-component speech features and a CTC phoneme head.

**Architecture diagram description:** The cell carries c separately from the exposed state h. The speech network stacks three bidirectional levels.

```text
[x_t,h_(t-1)] -> sigmoid gates i,f,o; tanh candidate g
c_(t-1) -- multiply f --+-- add i*g --> c_t -- tanh -- multiply o --> h_t
speech: three bidirectional LSTM levels -> softmax -> CTC -> phonemes
```

**Activation functions used and why:** Sigmoid sets gate values between zero and one. Tanh bounds candidate and exposed-state values. The cell itself updates by addition; it is not simply another tanh recurrence.

**Loss function(s):** CTC negative log-likelihood penalizes low probability for the labelled phoneme sequence while allowing different frame alignments. Frame-aligned cross-entropy and recurrent-transducer training are different objectives.

**Optimization algorithm(s):** The 2013 CTC recipe uses SGD with momentum 0.9. **Optional math:** Its rate is $`10^{-4}`$, or 0.0001. Later tutorials' common Adam settings are not substituted here.

**Regularization techniques:** Training uses feature scaling and development-set stopping. A second phase adds Gaussian **weight noise with standard deviation 0.075**. It starts from the initial noiseless run's best development log-probability point, not an arbitrary checkpoint.

**Backpropagation considerations:** BPTT traces errors through gates and the additive cell path. This helps connect distant inputs to later errors, but gates can still flatten out, states can grow too large, and shortened histories can lose training signals. Bidirectional updates use both earlier and later input.

**Parameter count / scaling behavior:** Gates add weight matrices. **Optional math:** A basic one-direction cell has about $`4h(d+h+1)`$ parameters, with input width d and state width h. Peepholes or separate biases add terms. The paper's **CTC-3l-250h** has **3.8 million weights**; this is not a universal LSTM count.

**Training paradigm:** Labelled utterances supervise sequence transcription. This CTC result requires no unlabelled-text language-model pretraining.

**Hardware/parallelism considerations:** Implementations can combine gate calculations, but one time step still depends on the last. More levels and both reading directions store more intermediate states. Offline bidirectional recognition is not the same as low-delay streaming.

**Strengths and limitations:** Controlled memory helps retain important details over many steps. Each step costs more than a simple RNN. Across time, it remains less parallel than Transformer training with known target prefixes.

**Computational complexity / scalability notes:** Sequence length multiplies the gate work. **Optional math:** One layer costs about $`O(T\,4h(d+h))`$, plus output and CTC work. T is sequence length, d input width, and h state width. The four roughly counts the main gate/candidate calculations; cell variants, projections, directions, and software change the constants.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [TIMIT Table 1](https://arxiv.org/html/1303.5778v1) gives **18.6% core-test phoneme error rate** for **CTC-3l-250h**, versus 37.6% for the 3.7-million-weight tanh model. Lower error is better. The split, 61-to-39 label mapping, and beam-100 decoding match the preceding entry, but the weight-noise treatment differs.

Audio frames pass through gated layers in both directions and become a phoneme sequence. The often-quoted **17.7%** belongs to **PreTrans-3l-250h**, a pretrained recurrent-transducer system, not this CTC model. Single runs do not supply a confidence interval or establish production speech accuracy.

**Notable vendor implementations/libraries:** PyTorch `LSTM`, Keras `LSTM`, and cuDNN-compatible operations offer cells. Common versions omit the historical peepholes; the operator name alone does not reproduce this speech model.

### 1.7.3 Gated recurrent unit (GRU)
**In plain English:** A GRU carries a state and learns how much of it to keep or replace. It offers memory control with fewer cell parts than a standard LSTM.

**Name:** Gated recurrent unit, using Cho and colleagues' original encoder-decoder form.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures. A compact recurrent cell uses gates to control state updates.

**Originating paper/vendor/year:** Kyunghyun Cho and colleagues, [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/html/1406.1078v3), EMNLP 2014. The paper calls it a new gated hidden unit; GRU later became the standard name.

**Core mechanism:** A reset gate controls which old-state values help form a new candidate. An update gate blends that candidate with the old state. Unlike an LSTM, there is no separate persistent cell state and exposed hidden state. In the original task, an encoder reads a source phrase and a decoder scores a possible translated phrase.

**Optional math:** One convention is $`\tilde h_t=\tanh(Wx_t+U(r_t\odot h_{t-1}))`$ and $`h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t`$. Here t is the step, x input, h state, the tilde marks a candidate, W/U are learned matrices, r is reset gate, and z is update gate. The symbol $`\odot`$ multiplies matching components.

**Inputs/outputs and typical data types:** Token sequences or numeric time series become states, predictions, or phrase scores. The translation model learns word vectors, called embeddings, and predicts probabilities over target words.

**Architecture diagram description:** One encoder summarizes the source phrase. A decoder uses that summary and the target prefix to score the next word.

```text
source phrase -> embedding -> GRU encoder -> phrase summary
                                                |
target prefix -> embedding -> GRU decoder -> maxout/readout -> next-word softmax
GRU cell: reset-gated candidate + update-gated carry -> new hidden state
```

**Activation functions used and why:** Reset and update gates use sigmoid; candidates use tanh; word outputs use softmax. The paper's readout has 500 maxout units, each keeping the larger of two inputs. That readout is not part of every GRU cell.

**Loss function(s):** The loss penalizes low probability for labelled source/target phrase pairs. This conditional sequence negative log-likelihood teaches a score that becomes one feature in a larger phrase-based translation system.

**Optimization algorithm(s):** Adadelta adapts step sizes from recent updates and gradients, using batches of 64 phrase pairs. **Optional math:** Its controls are $`\rho=0.95`$ for averaging and $`\epsilon=10^{-6}`$ for numerical stability. The paper gives no conventional fixed-rate decay schedule for this run.

**Regularization techniques:** Low-rank word mappings send inputs and outputs through narrower spaces, limiting capacity. Recurrent weights start with orthogonal directions, meaning directions at right angles in that space. These come from a matrix calculation using singular vectors. The cited recipe does not gain undocumented dropout just because modern libraries offer it.

**Backpropagation considerations:** BPTT follows both gated paths. Libraries differ in whether the reset multiplication happens before or after a recurrent linear-plus-bias transformation. Those are genuinely different models. Some definitions also swap the roles of z and one minus z.

**Parameter count / scaling behavior:** Encoder and decoder each have 1,000 hidden units and use rank-100 embedding mappings. **Optional math:** A one-direction cell has about $`3h(d+h+1)`$ parameters, for input width d and state width h. Embeddings and the readout add more, so hidden width alone cannot establish checkpoint size.

**Training paradigm:** Supplied parallel phrases supervise this model. Additional monolingual language-model features in other experiment rows are separate components.

**Hardware/parallelism considerations:** Fewer gate matrices can reduce work and memory compared with a standard LSTM. Actual speed depends on how well software combines operations. The sequence still has to advance step by step.

**Strengths and limitations:** GRUs provide useful memory control with a compact cell. A translation encoder without attention still compresses the whole source into one fixed summary. GRUs do not always outperform LSTMs.

**Computational complexity / scalability notes:** Longer sequences add repeated cell work; a large vocabulary can also make output scoring expensive. **Optional math:** One dense cell costs $`O(T\,3h(d+h))`$ per sequence, with length T, input width d, and state width h. This excludes vocabulary and readout calculations.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On [WMT-2014 English-to-French newstest2014](https://arxiv.org/html/1406.1078v3), adding this phrase score raises **test BLEU from 33.30 to 33.87**. Development scores are separately 30.64 and 31.20. BLEU compares wording with reference translations; it is not a percent-correct score.

The model scores a source phrase and a candidate French phrase. The larger translation system uses that score when choosing an answer. This is **not 33.87 BLEU from a standalone GRU-only translator**, nor the higher-scoring row that adds a language model. Better phrase representations were the goal; business localization quality and translator productivity were not measured.

**Notable vendor implementations/libraries:** PyTorch `GRU`, Keras `GRU`, and recurrent translation toolkits provide versions. When moving weights, check reset placement, biases, reading directions, and readout structure.

### 1.7.4 Sequence-to-sequence with Bahdanau attention
**In plain English:** This translator looks back at different parts of the source sentence while writing each output word. Learned weights decide how much information to retrieve from each source position.

**Name:** Sequence-to-sequence with Bahdanau additive attention, specifically the original **RNNsearch** translator.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures. The model learns which source positions contribute to each target step.

**Originating paper/vendor/year:** Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio wrote [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/html/1409.0473v7). It appeared as a preprint in 2014 and an ICLR paper in 2015.

**Core mechanism:** A forward/backward encoder stores a feature vector for each source position. Before choosing an output word, the decoder scores how well each vector matches its current needs. Softmax turns those scores into weights. The weighted source vectors form a context vector for that step. This **attention** is a numerical lookup, not human attention or proof of understanding. It avoids forcing the whole sentence through one fixed summary.

**Optional math:** One scorer is $`e_{ij}=v^\top\tanh(W_s s_{i-1}+W_hh_j)`$. Here i is target step, j source position, s previous decoder state, h source features, and v/Ws/Wh learned weights. Softmax over j gives weights $`\alpha_{ij}`$. The context is $`c_i=\sum_j\alpha_{ij}h_j`$, a weighted sum of source vectors.

**Inputs/outputs and typical data types:** A source-language token sequence becomes a translated sequence. The intermediate alignment weights describe the model's calculation. They are not guaranteed explanations of a human translator's reasoning.

**Architecture diagram description:** Source features remain available for repeated weighted lookups. The decoder uses each new context alongside the target words already supplied.

```text
source tokens -> forward/backward GRU encoder -> h1,h2,...,hS
                                                   |
previous decoder state -> additive scores -> softmax weights -> context c_i
target prefix + c_i -> GRU decoder -> maxout/readout -> target-word distribution
```

**Activation functions used and why:** The GRUs use sigmoid gates and tanh candidates. Tanh also shapes alignment scores; softmax makes source-position weights sum to one. A maxout readout keeps selected larger responses before vocabulary softmax produces word probabilities.

**Loss function(s):** Target-sequence cross-entropy penalizes low probability for each reference-translation token, then adds those penalties. Training needs paired sentences but no supplied word-to-word alignment.

**Optimization algorithm(s):** The source uses minibatch SGD with Adadelta and 80 sentences per update. **Optional math:** Adadelta uses $`\rho=0.95,\epsilon=10^{-6}`$ for its averaging and numerical-stability controls. The longer RNNsearch-50 run stops when development performance stops improving. It is a separate result, not a universal fixed-epoch schedule.

**Regularization techniques:** Vocabulary limits, starting weights, and development checks constrain training. Sentence-length limits define different training versions. They are not general regularizers or permission to exclude hard sentences from evaluation.

**Backpropagation considerations:** Error signals pass through both the recurrent history and the soft lookup weights. The appendix clips the global gradient norm at one, capping the overall signal size. Attention creates shorter routes to source features, but recurrent steps remain. Choosing only the largest alignment score as a hard position would remove ordinary gradients through that choice.

**Parameter count / scaling behavior:** RNNsearch has 1,000 units in each encoder direction and 1,000 decoder units. Vocabulary embeddings, the alignment network, and maxout readout also need weights. These widths alone do not give the complete parameter count.

**Training paradigm:** Bilingual sentence pairs supervise training. Teacher forcing supplies the correct earlier target words during training. At prediction time, beam search instead keeps a limited group of candidate translations. This is not masked-token or next-token pretraining on unpaired text.

**Hardware/parallelism considerations:** Source vectors and their alignment projections can be calculated once and reused. Output steps remain sequential. Keeping multiple beam candidates multiplies state and scoring work.

**Strengths and limitations:** Repeated lookups reduce the single-summary bottleneck, especially for long sentences. Word-level vocabulary limits can still produce unknown tokens. Attention alone guarantees neither accurate meaning nor faithful translation.

**Computational complexity / scalability notes:** Longer source and target sentences create more source-target comparisons. **Optional math:** With source length S, target length U, and scorer width a, alignment costs about $`O(SUa)`$ after reusable projections. Recurrent steps and vocabulary scoring add further work; alignment is not the entire translator's cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 1](https://arxiv.org/html/1409.0473v7) gives **26.75 BLEU for RNNsearch-50**, versus **17.82 for RNNencdec-50**, on **all 3,003 WMT-2014 English-to-French news-test-2014 sentences**. The longer-trained starred model reaches 28.45. The separate "No UNK" subset cannot replace the full-set result. BLEU measures wording overlap with references, not percent correct.

For each French word, the decoder weights stored source positions, predicts word probabilities, and extends candidate translations. The authors aim to reduce the damage from compressing long sentences into one vector. The benchmark measures neither commercial translation acceptance nor human productivity.

**Notable vendor implementations/libraries:** Keras additive-attention layers and historical recurrent translation toolkits implement the lookup mechanism. A generic wrapper does not recreate RNNsearch's encoder, gates, vocabulary, and maxout head.

### 1.7.5 Original encoder-decoder Transformer
**In plain English:** The original Transformer learns which source words are useful when producing a translation. During training, it can process many positions at once instead of following a strictly word-by-word chain.

**Name:** Original Transformer for converting one sequence into another. This entry includes base and big translation versions.

**Category & sub-category:** Supervised learning; recurrent and sequence architectures. It is a non-recurrent encoder-decoder placed here because it handles sequences, not because it is an RNN.

**Originating paper/vendor/year:** Ashish Vaswani and colleagues, [Attention Is All You Need](https://arxiv.org/html/1706.03762v7), NeurIPS 2017. The cited revision retains the historical translation experiments.

**Core mechanism:** Each position creates a query describing what information to retrieve. Other positions supply keys for matching and values to retrieve. Match scores become weights for averaging values. Several attention heads perform different learned lookups. These are numerical operations, not human attention.

The encoder looks across the source sentence. The decoder looks at earlier target positions and retrieves source information. A small feed-forward network then transforms each position separately. Residual shortcuts and layer normalization help the stack train. Sine-wave positional encodings add order information that these lookups alone would lack.

**Optional math:** Attention computes $`\operatorname{softmax}(QK^\top/\sqrt{d_k})V`$. Q, K, and V collect queries, keys, and values. The transpose makes query-key dot products; dk is key width. Dividing by its square root controls score size. Softmax supplies weights, and multiplying by V retrieves the weighted values.

**Inputs/outputs and typical data types:** Tokenized parallel sentences become target-token probabilities and a decoded translation. A padding mask excludes filler positions. A causal mask hides future target tokens; the two masks serve different purposes.

**Architecture diagram description:** FFN means the small feed-forward network used at each position. Both stacks have six layers, with different lookup types in the decoder.

```text
source -> embedding + position -> [self-attention -> FFN] x6 -> encoder states
target prefix -> embedding + position
              -> [masked self-attention -> cross-attention to source -> FFN] x6
              -> vocabulary projection -> softmax -> next target token
each sublayer: residual addition and original post-layer-normalization
```

**Activation functions used and why:** Softmax supplies attention weights and final word probabilities. The two-layer feed-forward block uses ReLU. The original does not use later families' GELU or SwiGLU activations.

**Loss function(s):** Supplied translation tokens guide cross-entropy, with label smoothing 0.1. The target is the correct translation given the source, not text predicted without a paired source.

**Optimization algorithm(s):** Adam adapts steps using recent gradients. **Optional math:** Its controls are $`\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}`$: two averaging settings and a numerical-stability constant. The rate is $`d^{-1/2}\min(t^{-1/2},t\,4000^{-3/2})`$, with model width d and training step t. It rises for 4,000 warmup steps, then falls in proportion to one over the square root of the step.

**Regularization techniques:** Dropout applies to residual and embedding paths; training also uses layer normalization and label smoothing. Base dropout is 0.1; big English-German uses 0.3. Final models average recent saved weights. This is not an ensemble averaging predictions from independently trained models.

**Backpropagation considerations:** During training, correct target prefixes are available. This teacher forcing lets target positions be processed together while the causal mask still blocks future answers. Shortcuts help error signals cross layers. During prediction, generated tokens must still arrive one after another.

**Parameter count / scaling behavior:** Base uses width 512, feed-forward width 2,048, eight heads, and about **65 million parameters**. Big uses width 1,024, feed-forward width 4,096, sixteen heads, and about **213 million**. Both contain six encoder and six decoder layers.

**Training paradigm:** **Supervised translation from parallel sentences.** [Foundation models](06-foundation-models.md) separately discuss BERT, GPT, and T5 pretraining. A decoder that hides future tokens does not make this original translation experiment unsupervised.

**Hardware/parallelism considerations:** Large matrix products let many training calculations run together. The experiments used eight P100 GPUs. Saving prior keys and values reduces repeated decoding work, but uses memory and does not remove sequential token generation.

**Strengths and limitations:** Direct lookups shorten routes between distant positions and support parallel training. Attention, vocabulary scoring, and feed-forward layers remain costly. Translation quality outside the tested domain is not guaranteed.

**Computational complexity / scalability notes:** Doubling sequence length roughly quadruples pairwise attention work. Other major costs grow differently. **Optional math:** For length T, model width d, and feed-forward width f, interactions cost $`O(T^2d)`$, **projections $`O(Td^2)`$, and FFNs $`O(Tdf)`$**. Source-target attention adds $`O(SUd)`$ for source length S and target length U. Saving all attention weights takes quadratic memory. Fused implementations can store less without eliminating dense pairwise calculations.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [translation table](https://arxiv.org/html/1706.03762v7) reports **27.3 BLEU for base and 28.4 for big** on **WMT-2014 English-to-German newstest2014**. BLEU compares wording with reference translations; it is not percent correct. Evaluation uses beam width four and length penalty 0.6. Base averages five checkpoints; big averages twenty.

A German candidate token retrieves information from the English sentence and earlier German tokens. Beam search then selects a sequence. The goal was parallelizable sequence learning without recurrence. These results establish neither a commercial service's architecture nor customer impact.

**Notable vendor implementations/libraries:** Historical Tensor2Tensor, PyTorch `Transformer`, and translation toolkits offer implementations. Check normalization order, tokenization, shared weights, and positional encodings rather than assuming defaults match.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Vanilla RNN | Ordered features with mostly nearby context | Simple shared rule carries state | Distant information is hard to retain and train | TIMIT tanh-RNN comparison with high phoneme error |
| LSTM | Speech and time series needing longer context | Gates control what the state keeps | Sequential steps and extra gate work | Bidirectional LSTM with CTC on TIMIT |
| GRU | Token sequences and smaller recurrent models | Controls memory with fewer cell parts | Sequential steps and fixed summaries can limit it | Extra phrase score in the WMT-2014 translation system |
| Seq2Seq with Bahdanau attention | Paired sequences of different lengths | Retrieves weighted source information at each step | Recurrent decoding and limited vocabulary | RNNsearch English-to-French translation test |
| Original Transformer | Paired text and other sequence-conversion tasks | Parallel training and direct weighted lookups | Attention, projections, and feed-forward layers cost work | Supervised WMT-2014 English-to-German translation |

## 1.8 Vision transformers and prediction architectures

A **backbone** extracts reusable image features. A full prediction system also needs a head that turns those features into task answers. ViT and Swin mainly describe backbones. U-Net assigns labels to pixels, called segmentation. Faster R-CNN, YOLO, and DETR find objects and their boxes, called detection. Their scores measure different tasks, so they cannot share one undivided "accuracy" ranking.

### 1.8.1 Vision Transformer (ViT)
**In plain English:** ViT splits a picture into patches and lets their features exchange information through weighted lookups. It uses the combined information to choose an image label.

**Name:** Vision Transformer. The worked model is **ViT-L/16, pretrained with ImageNet-21k labels**, then fine-tuned on ImageNet-1K.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures. Global attention connects image-patch representations.

**Originating paper/vendor/year:** Alexey Dosovitskiy and colleagues at Google Research wrote [An Image is Worth 16x16 Words](https://arxiv.org/html/2010.11929v2). Its preprint appeared in 2020 and its ICLR paper in 2021.

**Core mechanism:** The image is cut into equal patches. A learned linear mapping turns each flattened patch into a vector. Position vectors tell the model where patches belong. A special learned class token joins them. Encoder layers make weighted lookups across patch vectors, then MLPs transform the results. The final class-token vector supplies the image decision. Unlike fixed convolution filters, lookup weights depend on the image. Original ViT mostly keeps one patch resolution.

**Inputs/outputs and typical data types:** RGB images become class predictions or patch features. "L/16" means the Large configuration with 16-by-16 patches, not a 16-layer network.

**Architecture diagram description:** The 24 encoder blocks update patch information and the class token. Each block has attention and MLP branches with shortcuts.

```text
image -> 16x16 patches -> linear patch embeddings + learned positions
      -> prepend class token -> 24 encoder blocks -> class-token readout -> label
block: LayerNorm -> multi-head attention -> residual
       LayerNorm -> MLP/GELU -> residual
```

**Activation functions used and why:** MLPs use the smooth GELU activation; attention uses softmax to form lookup weights. Fine-tuning produces class logits, or raw scores. The replacement head must match the new task's labels.

**Loss function(s):** Cross-entropy uses supplied image categories. The [released training code](https://github.com/google-research/vision_transformer/blob/main/vit_jax/train.py) computes target-weighted negative log-softmax: it penalizes low probability on target classes. ImageNet-1K assigns one label per image. Rebuilding hidden patches is not this checkpoint's objective.

**Optimization algorithm(s):** ImageNet-21k training uses Adam, batch size 4096, 10,000 warmup steps, then linear rate decay. **Optional math:** The learning rate is $`10^{-3}`$, or 0.001. Fine-tuning uses SGD with momentum 0.9, batch size 512, and cosine decay. Its starting rate is selected from a grid for the target task, not fixed universally.

**Regularization techniques:** The appendix's **ImageNet-21k** recipe uses weight decay **0.03** and dropout **0.1**. JFT training uses different settings. Fine-tuning uses no weight decay and clips the overall gradient norm at one. High-resolution ImageNet runs also average parameter values.

**Backpropagation considerations:** Each branch is normalized before its main calculation, helping the residual stack train. Changing image resolution changes the number of patches. Fine-tuning must interpolate patch-position vectors while handling the class token separately. Simply reshaping all positions would not make the same adaptation.

**Parameter count / scaling behavior:** ViT-L has **307 million parameters**, 24 layers, width 1,024, MLP width 4,096, and sixteen heads. Base and Huge have about 86 million and 632 million parameters respectively.

**Training paradigm:** Labelled large-dataset pretraining is followed by labelled task adaptation. [Unsupervised neural models](05-unsupervised-neural.md) cover masked-image and other self-supervised Transformer objectives separately.

**Hardware/parallelism considerations:** Dense attention and MLPs suit accelerators. Higher image resolution creates many more tokens. The paper's TPU training resources describe those experiments, not a universal requirement for making predictions.

**Strengths and limitations:** ViT learns whole-image interactions with fewer built-in image assumptions than a CNN. It may therefore require more data. Keeping one patch resolution also makes some pixel-level tasks less convenient.

**Computational complexity / scalability notes:** Smaller patches sharply raise lookup costs. Halving patch side creates four times as many tokens and sixteen times the pairwise attention work. Other token-dependent terms grow fourfold. **Optional math:** $`N=HW/P^2`$ approximately counts patches for image dimensions H/W and patch side P. A block costs $`O(N^2d+Nd^2+Ndf)`$, with feature width d and MLP width f.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 2](https://arxiv.org/html/2010.11929v2) reports **85.30% +/- 0.02 top-1 ImageNet accuracy** for **ViT-L/16 pretrained on ImageNet-21k** and fine-tuned on ImageNet-1K. Top-1 checks the highest-scored label. The stated variation covers three fine-tuning runs, not an ensemble gain. Evaluation uses the held-out validation benchmark and the paper's higher-resolution **512-pixel** ViT-L fine-tuning setting.

Patch features exchange weighted information; the class token supplies the object label. The authors test whether larger-scale learning can make up for fewer built-in convolutional assumptions. The JFT-pretrained Huge score of 88.55% belongs to a different model. Neither result proves a deployed visual-search benefit.

**Notable vendor implementations/libraries:** Google's `vision_transformer`, timm, and torchvision offer models. Record the weights and learning objective, not just size, especially for self-supervised or differently pretrained versions.

### 1.8.2 Swin Transformer
**In plain English:** Swin lets nearby image patches share information, then shifts the groups so information can cross boundaries. It builds features at several image scales without comparing every patch at every layer.

**Name:** Swin Transformer, using **Swin-Tiny from the original 2021 family**.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures. A hierarchy uses attention inside shifting local windows.

**Originating paper/vendor/year:** Ze Liu and colleagues at Microsoft Research and collaborating groups published [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/html/2103.14030v2) at ICCV 2021.

**Core mechanism:** Each small, nonoverlapping window makes weighted lookups only among its own tokens. The next block shifts the grouping, allowing exchanges across the old boundaries. Patch merging reduces spatial detail while widening feature vectors. A mask prevents the software's cyclic shift from wrongly linking opposite image edges.

**Inputs/outputs and typical data types:** Images become feature maps at several scales. A classification head turns them into class probabilities. Detection and segmentation systems need separate heads and losses.

**Architecture diagram description:** Three merge steps connect four stages. Within a stage, ordinary and shifted-window blocks alternate.

```text
image -> 4x4 patch embedding -> stage1 -> merge -> stage2 -> merge -> stage3
      -> merge -> stage4 -> global pooling -> classifier
within stages: window attention -> MLP -> shifted-window attention -> MLP
               [LayerNorm and residual paths around each sublayer]
```

**Activation functions used and why:** MLPs use GELU. Softmax normalizes lookup weights within each window. Learned relative-position biases adjust scores based on patch locations; these added geometry values are not separate activations.

**Loss function(s):** Supervised cross-entropy uses label smoothing and targets from augmented images. Detection and segmentation results depend on additional task-specific objectives.

**Optimization algorithm(s):** ImageNet-1K training uses AdamW, initial rate 0.001, weight decay 0.05, batch size 1024, and 300 epochs. The rate warms up for twenty epochs, then follows cosine decay.

**Regularization techniques:** The recipe uses layer normalization, stochastic depth, image changes, and the source's DeiT-style training controls. It specifically excludes repeated augmentation and exponential moving average (EMA) for this 1K setup because they did not improve it. EMA would keep a running weighted average of model weights.

**Backpropagation considerations:** Error signals cross residual branches and masked window lookups. Padding and shift masks must be correct. Missing masks change both the forward connections and their backward signals.

**Parameter count / scaling behavior:** Swin-T has about **28 million parameters**, stage depths [2,2,6,2], and initial width 96. Between stages, patch merging roughly doubles channel width while reducing token count to one quarter.

**Training paradigm:** The reference uses supervised ImageNet-1K labels. Other Swin results add supervised ImageNet-22K pretraining. Self-supervised Swin encoders are another training category.

**Hardware/parallelism considerations:** Tokens in one window share key sets, helping nearby memory access. Creating windows, shifting them, masking, and padding all take work. Window size and image resolution can change speed even with similar arithmetic counts.

**Strengths and limitations:** Swin supplies multiscale features without global attention in every layer. Distant regions communicate through repeated shifts and merging. One local window does not see the entire image.

**Computational complexity / scalability notes:** With fixed windows and widths, adding image tokens adds roughly proportional attention work. **Optional math:** For N tokens, window side M, feature width d, and MLP width f, interactions cost $`O(NM^2d)`$, not $`O(N^2d)`$. Projections and MLPs add $`O(Nd^2+Ndf)`$. "Linear in image size" assumes those fixed settings; resolution and width still matter.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper](https://arxiv.org/html/2103.14030v2) reports **81.3% top-1 ImageNet-1K validation accuracy** for **Swin-T at 224-by-224**, trained on ImageNet-1K. The listed DeiT-S gets 79.8%. Top-1 checks whether the first class choice is correct.

Windows mix nearby features, shift, and merge before the model chooses an image label. The goal is an efficient hierarchy that can also support high-resolution tasks. Stronger detection and 22K-pretrained results use extra systems or data; they do not describe this Tiny run. No Microsoft-product deployment is implied.

**Notable vendor implementations/libraries:** Microsoft's Swin release, timm, torchvision, and OpenMMLab integrations offer implementations. Swin V2, detection backbones, and classification checkpoints need different settings.

### 1.8.3 U-Net
**In plain English:** U-Net gives each image pixel a class, such as membrane or background. It combines broad context with saved fine detail to locate boundaries more precisely.

**Name:** U-Net, the original two-dimensional biomedical segmentation network using valid convolutions, which do not pad the image borders.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures. A convolutional encoder-decoder labels an entire image region.

**Originating paper/vendor/year:** Olaf Ronneberger, Philipp Fischer, and Thomas Brox of the University of Freiburg published [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/html/1505.04597v1) at MICCAI 2015.

**Core mechanism:** The downward path shrinks maps to gather wider context. The upward path enlarges them again and joins saved, cropped features from earlier high-resolution layers. These skip features help locate details lost during shrinking. This shares work across pixels instead of running a separate window classifier at every position. Valid convolutions make the output tile smaller than the input. Overlapping tiles and reflected border context cover larger images.

**Inputs/outputs and typical data types:** Microscopy images become class probabilities at each pixel, then masks marking regions. Three-dimensional medical-volume versions are extensions, not the original measured network.

**Architecture diagram description:** Saved encoder features join the expanding decoder by concatenation. The original geometry changes a 572-by-572 input into a 388-by-388 output.

```text
572x572 image -> [64 -> 128 -> 256 -> 512 -> 1024] encoder channels
                  |      |      |      |        |
                  +------ cropped skip features-+--> expanding decoder
decoder: up-convolution -> concatenate skip -> two 3x3/ReLU convolutions
       -> 1x1 class projection -> 388x388 probability map
```

**Activation functions used and why:** Feature convolutions use ReLU. Pixelwise softmax supplies competing class probabilities at each location. Enlarging maps and joining skip features restores access to detail, but those steps alone do not classify pixels.

**Loss function(s):** Weighted pixelwise cross-entropy gives extra importance to class balance and borders between touching objects. Its border weights use distance to the nearest two object boundaries. This is not the Dice loss common in later U-Nets.

**Optimization algorithm(s):** The original uses SGD with **momentum 0.99**, processing one large image tile per batch. The main paper lacks a complete numerical learning-rate schedule. A reproduction needs the run's solver settings, not assumed modern Adam defaults.

**Regularization techniques:** Strong elastic distortions and other image changes help with few annotated examples. Starting weights are chosen for rectified activations such as ReLU. Original U-Net does not automatically include modern batch normalization or padded convolutions.

**Backpropagation considerations:** Every pixel's error can update both context and skip paths. Cropping must align the same image locations correctly. Border weighting strengthens signals at narrow separations, making annotation quality and chosen weights important.

**Parameter count / scaling behavior:** Reconstructing the displayed 64-base-channel, two-class network gives about **31 million parameters**, calculated from its dimensions. Changing width, dimensionality, padding, or the output classes can change the count.

**Training paradigm:** Human-drawn masks supply supervision. The model is not discovering cells without labels. Pseudo-label and consistency-based U-Nets use different, semi-supervised formulations.

**Hardware/parallelism considerations:** High-resolution maps and saved skips use much of the training memory. Overlap-tile prediction repeats some border calculations in exchange for handling images larger than accelerator memory.

**Strengths and limitations:** U-Net combines wider context with local detail using limited labels. Different microscopes or tissues, inaccurate boundaries, and uneven class counts can undermine later scientific or clinical use.

**Computational complexity / scalability notes:** Both contracting and expanding paths require convolution work. Moving from two-dimensional (2D) images to three-dimensional (3D) data adds another spatial dimension. This greatly increases work and memory. **Optional math:** Skip storage scales with $`B\sum_\ell H_\ell W_\ell C_\ell`$. B is batch size; each layer ell has map height H, width W, and C channels.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** For the [ISBI electron-microscopy challenge](https://arxiv.org/html/1505.04597v1), the study uses 30 annotated 512-by-512 images of Drosophila neuronal tissue for training. Test labels are hidden. Averaging predictions from **seven rotated inputs**, U-Net reports **warping error 0.0003529**, versus 0.000420 for the cited sliding-window CNN.

Warping error is sensitive to how structures connect; lower is better. It is **not pixel accuracy, Dice, or intersection-over-union (IoU)**. Dice and IoU instead measure region overlap. Images become membrane probability maps, then thresholded boundaries that support tracing neuronal structures. Shared pixel computation and stronger boundary learning motivate the design. This is a research test, not evidence of clinical safety or saved laboratory labor.

**Notable vendor implementations/libraries:** Freiburg's release and MONAI, nnU-Net, Keras, and PyTorch ecosystems provide U-Nets. nnU-Net is a broader system that configures a pipeline, not merely the original 2015 network.

### 1.8.4 Faster R-CNN
**In plain English:** Faster R-CNN first suggests regions that may contain objects, then checks and refines them. Both stages reuse the same image features to avoid repeatedly processing each crop.

**Name:** Faster R-CNN, the original region-proposal-network detector. The example uses a VGG-16 backbone.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures. Two-stage object detection shares a visual feature network.

**Originating paper/vendor/year:** Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun published [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/html/1506.01497v3) at NeurIPS 2015. A later manuscript expands the account.

**Core mechanism:** A CNN creates one shared feature map. The region proposal network (RPN) checks preset candidate boxes, called anchors, at each location. It predicts whether they contain objects and how to adjust their coordinates. Selected proposals then enter region-of-interest (RoI) pooling, which produces features of a standard size. A second head classifies each region and refines its box. The whole CNN need not run again for every crop.

**Inputs/outputs and typical data types:** Images, with object boxes and class labels during training, become a variable-length list of labelled boxes. Training also includes an explicit background class.

**Architecture diagram description:** The RPN and region classifier share VGG features. NMS, explained below, removes overlapping duplicate candidates.

```text
image -> shared VGG feature map --+-> RPN: anchors -> objectness/box offsets
                                |                    -> proposal selection/NMS
                                +-> RoI pooling <-------------+
                                    -> region head -> class + refined box
```

**Activation functions used and why:** Backbone and region layers use ReLU. Softmax supplies object/background and class scores. Box adjustments use linear outputs. Newer normalized backbones and feature-pyramid versions are separate variants.

**Loss function(s):** The RPN learns object/background decisions and box corrections for positive anchors. The detector learns classes and class-specific corrections. Classification uses cross-entropy; box fitting uses smooth-L1, which limits the effect of large coordinate errors. Matching rules choose which candidates receive object or background labels.

**Optimization algorithm(s):** Main experiments alternate training the RPN and detector. For the PASCAL RPN, SGD uses 0.001 for 60,000 minibatches, then 0.0001 for 20,000 more. Momentum is 0.9 and weight decay 0.0005. Detector fine-tuning follows its own Fast R-CNN settings.

**Regularization techniques:** The model starts with labelled ImageNet training, then uses weight decay and image changes. It samples positive and background anchors and regions. This balance matters because background candidates greatly outnumber objects.

**Backpropagation considerations:** Both losses update shared features, but not every step changes smoothly. Proposal selection and non-maximum suppression (NMS) make discrete choices; NMS removes lower-scored boxes that overlap stronger ones. Original RoI pooling also is not fully smooth in box geometry. Approximate joint training treats proposal coordinates as fixed during the detector's backward pass, unlike alternating training.

**Parameter count / scaling behavior:** Total size depends on the backbone and head. With a 512-channel map, the shared 3-by-3 RPN layer and nine-anchor prediction heads add about **2.4 million parameters**, calculated from those dimensions. That is not the full detector's count.

**Training paradigm:** Supplied object boxes and categories supervise detection after supervised backbone pretraining. Automatically suggested regions are candidates, not unlabelled pseudo-targets.

**Hardware/parallelism considerations:** Sharing the backbone saves repeated image processing. Region-head work still rises with retained proposals. Resizing, NMS, and moving proposals can limit speed even when convolution is fast.

**Strengths and limitations:** Two stages focus classification and location refinement on promising regions. Anchors, matching thresholds, proposal limits, and postprocessing add complexity. Small objects and crowded scenes remain difficult.

**Computational complexity / scalability notes:** Work includes the backbone, RPN at every feature location and anchor, and a head evaluation per retained region. **Optional math:** With R proposals, straightforward NMS can cost $`O(R^2)`$. Doubling proposals can therefore quadruple that part. A small RPN does not make image resolution or proposal count irrelevant.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [VGG-based experiment](https://arxiv.org/html/1506.01497v3) reports **73.2% mAP on PASCAL VOC 2007 test**, trained on **VOC 2007 plus 2012 trainval**, with **300 RPN proposals** during testing. Mean average precision (mAP) combines detection quality across classes and score thresholds. VOC here uses **IoU 0.5**: box overlap divided by the area covered by either box must meet that threshold. It is not COCO's average across several overlap thresholds.

An image generates shared features and proposals, then class labels and refined boxes for people, vehicles, or animals. The authors sought to replace external proposal generation while sharing image features. This benchmark does not establish use in a specific surveillance or vehicle product.

**Notable vendor implementations/libraries:** Original Caffe releases, Detectron2, torchvision detection, and MMDetection offer versions. Feature pyramids (FPN), RoIAlign, and newer backbones change the network and the identity of its benchmark.

### 1.8.5 YOLO, original version
**In plain English:** Original YOLO predicts object boxes and classes in one network pass. Its fixed grid keeps the design simple, but can struggle with small objects close together.

**Name:** **YOLOv1**, the original "You Only Look Once" detector. Later YOLO generations are not interchangeable with this entry.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures. Single-stage detection predicts objects from a grid.

**Originating paper/vendor/year:** Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi wrote [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/html/1506.02640v5). Its preprint appeared in 2015 and its CVPR paper in 2016.

**Core mechanism:** One network reads the full image and predicts boxes directly. For VOC, each cell in a 7-by-7 grid predicts two boxes and one shared distribution over 20 classes. The cell containing an object's center is responsible for it. Each box includes coordinates and confidence intended to combine object presence with overlap quality. There is no learned proposal stage.

**Inputs/outputs and typical data types:** Fixed-size RGB images become boxes with class scores. Original detection training uses 448-by-448 images.

**Architecture diagram description:** Twenty-four convolution layers and two dense layers produce the grid. Each box has five outputs; the cell also supplies twenty class values.

```text
448x448 RGB -> 24 convolutional layers with pooling -> two dense layers
           -> 7x7x(2*5 + 20) predictions
           -> class-confidence scores -> threshold/NMS -> labelled boxes
```

**Activation functions used and why:** Hidden layers use leaky ReLU with negative slope 0.1, keeping a small response for negative input. The output layer is linear. Later sigmoid heads and anchor-based or anchor-free YOLO designs are not substitutions for v1.

**Loss function(s):** The loss adds weighted squared errors for coordinates, confidence, and classes. Coordinate weight is five; no-object confidence weight is 0.5. Comparing square roots of widths and heights reduces large boxes' influence. This loss does not directly optimize mAP, the final detection score.

**Optimization algorithm(s):** SGD uses momentum 0.9, weight decay 0.0005, and batches of 64. **Optional math:** The paper warms the rate from $`10^{-3}`$ toward $`10^{-2}`$, then lists 75 epochs at $`10^{-2}`$, 30 at $`10^{-3}`$, and 30 at $`10^{-4}`$. These rates mean 0.001, 0.01, and 0.0001 respectively.

**Regularization techniques:** Dropout 0.5 follows the first dense layer. Training randomly changes image size, position, exposure, and color saturation. Classification pretraining supplies starting convolution features before supervised detection fine-tuning.

**Backpropagation considerations:** Large starting steps can destabilize coordinates, which motivates warmup. Training chooses the responsible box by highest overlap. This choice and NMS are discrete decisions. Gradients update selected prediction losses, not a fully smooth version of every detector step.

**Parameter count / scaling behavior:** Total size adds convolution weights and dense matrices. **Optional math:** Mapping F flattened features into h hidden units needs $`Fh+h`$ parameters, including biases. This can make the fixed-grid head expensive. No unchecked family-wide YOLO count is assigned to this historical network.

**Training paradigm:** Labelled image classification comes first, followed by learning from object boxes and categories. The detector has no self-supervised language-model objective.

**Hardware/parallelism considerations:** One network pass supports parallel image calculations. Input preparation, NMS, camera capture, and hardware still affect end-to-end delay. Historical frames-per-second (FPS) measurements do not guarantee speed on another device.

**Strengths and limitations:** The network uses full-image context without a separate proposal system. Sharing class predictions within cells, and limiting boxes per cell, makes nearby small objects and crowded scenes harder.

**Computational complexity / scalability notes:** Add all convolution and dense-head work at the configured resolution. A larger grid gives more raw predictions and more possible postprocessing. Changing input shape may also require changing the original dense head.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper](https://arxiv.org/html/1506.02640v5) reports **63.4% mAP on VOC 2007 test** for YOLOv1 trained on **VOC 2007+2012 trainval**. **Fast YOLO** is a different, smaller model with 52.7% mAP. Mean average precision combines detection results across classes and confidence thresholds. These VOC results use IoU 0.5, an overlap-area measure, not COCO's multi-threshold AP.

The image yields 98 candidate boxes and cell-level class scores. Combined confidence and NMS choose the labelled locations. The goal is unified, fast detection. The coarse grid trades crowded-object detail for simplicity; these scores do not demonstrate road-safety performance.

**Notable vendor implementations/libraries:** Original Darknet is the historical reference. Current Darknet and Ultralytics often supply other YOLO generations with different heads, losses, training data, and parameter counts.

### 1.8.6 DETR
**In plain English:** DETR predicts a set of objects using learned output slots. During training, each real object is matched to one slot, reducing the need to remove duplicate boxes afterward.

**Name:** Detection Transformer, shortened to **DETR**. This entry uses the original 2020 model with a ResNet-50 backbone.

**Category & sub-category:** Supervised learning; vision transformers and prediction architectures. A network predicts an object set directly.

**Originating paper/vendor/year:** Nicolas Carion and colleagues at Facebook AI Research published [End-to-End Object Detection with Transformers](https://arxiv.org/html/2005.12872v3) at ECCV 2020.

**Core mechanism:** A CNN supplies spatial image features to a Transformer encoder. A decoder uses learned object queries, or output slots, to retrieve weighted image information. Each slot predicts a class and box. During training, Hungarian matching chooses a one-to-one pairing of predictions with labelled objects. Unmatched slots learn "no object." This discourages duplicates, so the original system needs neither anchors nor NMS.

**Inputs/outputs and typical data types:** Images and annotated object sets train the system. It produces a fixed-size prediction set; foreground scores determine which detections are useful.

**Architecture diagram description:** A six-layer encoder feeds a six-layer decoder with 100 learned queries. Training matches the resulting slots to labelled objects.

```text
image -> ResNet-50 -> projected spatial features + positions -> 6-layer encoder
100 learned object queries --------------------------------> 6-layer decoder
                                                              |
                                                  shared class/box heads
training: Hungarian assignment -> matched set loss + no-object supervision
```

**Activation functions used and why:** Feed-forward layers use ReLU. Attention softmax forms lookup weights; class softmax scores categories. Sigmoid keeps box coordinates within a normalized range. Position information distinguishes locations and query slots.

**Loss function(s):** Matched objects receive classification loss plus L1 and generalized-IoU box losses. L1 measures coordinate differences; generalized IoU adds a box-overlap-based measure. Unmatched slots receive reduced-weight no-object loss. Extra losses at intermediate decoder layers also assist training.

**Optimization algorithm(s):** AdamW uses separate step sizes for new and pretrained parts. **Optional math:** The Transformer rate is $`10^{-4}`$, backbone rate $`10^{-5}`$, and weight decay $`10^{-4}`$. The main result uses **500 epochs, with a tenfold rate drop after epoch 400**. The 300-epoch experiment with a drop at 200 is a different test of design choices.

**Regularization techniques:** ResNet starts with ImageNet weights and fixed batch-normalization statistics. Transformer dropout, random resizing/cropping, and the released recipe's gradient controls support training. The long schedule is a genuine cost, not free extra capacity.

**Backpropagation considerations:** Matching chooses pairs discretely. Error signals then flow through the chosen class and box losses, not through a smooth Hungarian solver. Intermediate decoder losses help slots learn useful assignments earlier.

**Parameter count / scaling behavior:** ResNet-50 DETR has **41 million parameters**, Transformer width 256, and 100 queries. More queries cost more decoder and matching work. Finer image-feature maps cost more encoder attention.

**Training paradigm:** Labelled objects supervise detection after supervised backbone pretraining. Object queries are learned parameters, not unlabelled examples or user retrieval requests.

**Hardware/parallelism considerations:** Encoder and decoder matrix calculations run in parallel. Original DETR uses a long training schedule and global image-feature attention. A dilated backbone preserves finer maps, but raises memory and computation needs.

**Strengths and limitations:** Set prediction simplifies final box selection and performs well on large objects in the original comparison. Weaknesses include small objects and slow training progress. Later deformable versions make different trade-offs and are separate models.

**Computational complexity / scalability notes:** More image features raise all-pairs encoder work; more output slots raise decoder and matching work. **Optional math:** With N image tokens, Q queries, and width d, attention includes $`O(N^2d)`$ encoder work, $`O(QNd)`$ cross-attention, and $`O(Q^2d)`$ query self-attention. Projections and feed-forward layers add costs. Straightforward matching can take $`O(Q^3)`$ work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 1](https://arxiv.org/html/2005.12872v3) reports **42.0 box AP on COCO 2017 validation** for **ResNet-50 DETR**, trained on COCO train2017 for 500 epochs. Average precision (AP) combines detection quality across score thresholds. This COCO score also averages box-overlap (IoU) thresholds **0.50 through 0.95**, unlike VOC mAP at 0.5.

DETR matches the strengthened Faster R-CNN-FPN+ baseline's 42.0 overall AP, with weaker small-object AP but stronger large-object AP. Image features feed queries that jointly produce labelled boxes without NMS. This is a benchmark, not evidence of a named retail, autonomous-driving, or surveillance deployment.

**Notable vendor implementations/libraries:** Meta's original release, Hugging Face, and detection frameworks offer DETR versions. Deformable DETR, DINO detectors, and real-time variants require separate source and training records.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| ViT | Images with extensive labelled pretraining data | Weighted lookups connect all image patches | Many data and high-resolution calculations | ImageNet-21k-pretrained ViT-L/16 transfer test |
| Swin | Images needing features at several scales | Local lookups connect through shifting windows | Distant context takes stages; window handling adds work | Swin-T ImageNet-1K classification test |
| U-Net | Images with pixel labels, including microscopy | Combines wide context with saved fine detail | Large maps, imperfect boundaries, and unfamiliar data | ISBI neuronal-membrane segmentation test |
| Faster R-CNN | Images labelled with object boxes | Reuses features to propose and refine regions | Several matching and box-selection steps | VGG-based VOC 2007 detection test |
| YOLOv1 | Fixed-resolution images for object detection | One network predicts boxes directly | Coarse cells struggle with crowded small objects | Original VOC 2007 result, not a later YOLO |
| DETR | Images labelled with sets of objects | Matches one prediction to each object without NMS | Original model trains slowly and struggles with small objects | COCO 2017 validation box-AP test |

## 1.9 Graph networks

A **graph** records things and their connections. Nodes represent the things, such as proteins or posts. Edges represent relationships between them. A node can also have features, such as measurements or text-derived numbers. Graph layers combine information along edges rather than across image pixels.

Two questions must stay separate. First, which answers have labels: whole graphs, all training nodes, or only some nodes? Second, can the model see the evaluation graph during training? **Inductive** evaluation uses new nodes or graphs. **Transductive** training already sees evaluation nodes' features and edges, though not their hidden answers. Neither word alone tells us the supervision type. The examples here learn from labelled training targets; related semi-supervised and unsupervised tasks are identified separately.

### 1.9.1 Graph convolutional network (GCN)
**In plain English:** A GCN mixes each node's features with its connected neighbors' features. This example then averages the node results to classify an entire protein graph.

**Name:** Graph convolutional network, using the Kipf-Welling normalized neighbor-combining rule. The example adds a supervised whole-graph classifier.

**Category & sub-category:** Supervised learning; graph networks. Nodes share scaled messages along edges, then their outputs are pooled for a graph label.

**Originating paper/vendor/year:** Thomas Kipf and Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/html/1609.02907v4), 2016 preprint and ICLR 2017 paper. The title correctly describes the original citation-network task. The fully supervised example here instead comes from the [DGL graph-classification tutorial](https://www.dgl.ai/dgl_docs/en/1.1.x/tutorials/blitz/5_graph_classification.html).

**Core mechanism:** Add a self-loop so each node keeps its own features alongside neighbors' features. Scale contributions using connection counts, then combine and transform them with shared weights. The same rule can process a new graph. Here, averaging final node vectors gives one graph prediction without depending on node order. This pooling step is an added choice, not part of the original node classifier.

**Optional math:** $`\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}`$ and $`H'=\phi(\hat AHW)`$. A records edges; I adds self-loops; D-tilde records resulting connection counts. The inverse square roots scale messages. H holds node features, W learned weights, phi the activation, and H-prime new features. A-hat is the scaled connection matrix.

**Inputs/outputs and typical data types:** Node-feature tables and edge lists enter. Layers produce new node features; this example turns them into one class distribution per protein graph.

**Architecture diagram description:** Two graph layers change three input features into sixteen hidden features, then two class scores. The final mean is over nodes, not across different graphs.

```text
protein graph: node features + edges + self-loops
 -> normalized GraphConv(3 -> 16) -> ReLU
 -> normalized GraphConv(16 -> 2) -> mean over nodes
 -> two graph logits -> predicted graph class
```

**Activation functions used and why:** ReLU changes the first layer's output. The second produces linear class scores, averaged before cross-entropy normalizes them. Scaling the graph connections is a fixed sharing rule, not another activation.

**Loss function(s):** Cross-entropy compares predictions with **labels on training graphs**. This example does not train on unlabelled test nodes or hide most node labels as in citation-network experiments.

**Optimization algorithm(s):** The DGL example uses Adam at 0.01, batches of five graphs, and 20 epochs. It specifies no learning-rate decay.

**Regularization techniques:** The shown tutorial uses neither dropout nor weight decay. Its small hidden layer and scaled averaging limit capacity. A serious application would separately choose regularization and a sound split; the minimal demo is not an optimized recipe.

**Backpropagation considerations:** Error signals pass through node transformations, edge-based combining, and graph averaging. The model does not learn which edges exist. Repeated mixing can make nodes too alike, called oversmoothing. Averaging may also erase differences needed for the graph label.

**Parameter count / scaling behavior:** All nodes and graphs reuse the same weights. **Optional math:** For the biased 3-16-2 network, $`3(16)+16+16(2)+2=\mathbf{98}`$ trainable parameters, calculated from the tutorial.

**Training paradigm:** This is fully supervised graph classification, tested on separate new graphs. Original Cora experiments label only a small set of nodes while making the other nodes' features and edges visible. Those settings belong in [semi-supervised learning](03-semi-supervised.md).

**Hardware/parallelism considerations:** Software can batch separate graphs as one disconnected graph while keeping their final averages separate. Sparse edge access often moves through memory less efficiently than dense matrix multiplication.

**Strengths and limitations:** GCNs use relationships without imposing an arbitrary node order. Simple averaging can fail when edges connect dissimilar nodes. A small mean-pooled network cannot distinguish every structure or capture every biochemical interaction.

**Computational complexity / scalability notes:** Sparse storage follows existing edges rather than every possible node pair. **Optional math:** Transforming first costs $`O(Vd_{\rm in}d_{\rm out}+(m+V)d_{\rm out})`$ per layer. V counts nodes, m edges, and d input/output widths; the extra V counts self-loops. Sparse graph storage is $`O(V+m)`$. A dense V-by-V table loses that saving.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** DGL's [PROTEINS demonstration](https://www.dgl.ai/dgl_docs/en/1.1.x/tutorials/blitz/5_graph_classification.html) uses three features per node and two possible graph labels. The first 80% of dataset indices go to training; the rest go to testing. Random samplers shuffle **within those partitions**, not between them. The documentation prints **0.273542600896861 test accuracy**, about **27.35%**, for one illustrative run. This poor score is preserved, not replaced with an imagined improvement.

Protein-node neighborhoods become averaged class scores and a graph label. Relationships motivate trying a GCN instead of flattening everything into a list. But this result needs validation before scientific use. The tutorial is neither a robust biological benchmark averaged over random seeds nor a production discovery claim.

**Notable vendor implementations/libraries:** DGL `GraphConv` and PyTorch Geometric `GCNConv` supply versions. Set normalization, self-loops, reused adjacency calculations, and whole-graph readout explicitly.

### 1.9.2 Graph attention network (GAT)
**In plain English:** A GAT learns how much each connected neighbor should contribute to a node's new features. Here it uses labelled protein graphs to predict several possible functions for each protein.

**Name:** Graph attention network, the original multi-head **GAT**, used for supervised protein-function prediction on unseen graphs.

**Category & sub-category:** Supervised learning; graph networks. Learned weights control information gathered from connected neighbors.

**Originating paper/vendor/year:** Petar Velickovic and colleagues, [Graph Attention Networks](https://arxiv.org/html/1710.10903v3), 2017 preprint and ICLR 2018 paper. The study includes semi-supervised citation tasks and a separately supervised protein-protein interaction (PPI) task.

**Core mechanism:** Transform node features, score connected pairs, and turn each node's neighbor scores into weights. The next state is a weighted sum of neighboring features. Several heads learn different weighting rules. Hidden heads can be joined side by side; output heads can be averaged. The lookup follows existing graph edges, not every possible pair. "Attention" here means weighted calculation, not human reasoning.

**Inputs/outputs and typical data types:** A protein-interaction graph has 50 features per node. The PPI model outputs 121 independent function-label scores for each protein node.

**Architecture diagram description:** Two hidden layers concatenate four heads each. Six output heads are averaged before separate function probabilities are produced.

```text
PPI node features + adjacency
 -> GAT: 4 heads x 256 features -> concatenate -> ELU
 -> GAT: 4 heads x 256 features -> concatenate -> ELU, skip connection
 -> GAT: 6 heads x 121 scores -> average -> sigmoid -> function labels
```

**Activation functions used and why:** The scorer uses leaky ReLU, which retains a small negative response. Softmax makes each neighborhood's weights sum to one. Hidden layers use ELU, which smoothly handles negative values. Output sigmoid scores each function independently. Output softmax would wrongly force the 121 biological labels to compete as mutually exclusive choices.

**Loss function(s):** Binary cross-entropy checks each function label on labelled protein nodes in training graphs. This differs from the single-label class loss used in citation-network tests.

**Optimization algorithm(s):** PPI training uses Adam at 0.005 with two graphs per batch. Early stopping waits up to 100 epochs without validation improvement. The paper specifies no separate decaying rate for this experiment.

**Regularization techniques:** The authors explicitly use **no dropout and no L2 penalty** for PPI, judging the labelled training set large enough. A skip connection crosses the intermediate attention layer. Strong dropout from small citation datasets does not belong in this recipe.

**Backpropagation considerations:** Errors change both node features and neighbor weights. Some weights can become heavily concentrated. Nodes with many neighbors need more intermediate storage and work. A large learned weight does not prove a causal biological interaction.

**Parameter count / scaling behavior:** More heads and wider features add weights, but more edges do not directly add parameters. **Optional math:** A simple layer has about $`a(df+2f)`$ transform/scoring parameters, before biases and skip projections. Here a counts heads, d input width, and f head width. Edge count does affect saved attention values.

**Training paradigm:** **Supervised inductive node classification** uses 20 labelled PPI graphs for training, two for validation, and two unseen graphs for testing. Cora/Citeseer/Pubmed experiments instead expose evaluation graph structure during training and use only some node labels. Those are transductive, semi-supervised tasks, not this example.

**Hardware/parallelism considerations:** Batching graphs and using sparse edge operations helps. Uneven neighbor counts create irregular workloads. More heads increase data movement and feature storage.

**Strengths and limitations:** GAT can learn unequal neighbor contributions instead of fixing them through connection counts. Available edges still limit what it sees. The original scoring rule has limits, repeated mixing can blur nodes, and unfamiliar graphs may cause failure.

**Computational complexity / scalability notes:** Work follows actual nodes and edges, multiplied by head sizes. **Optional math:** A sparse layer costs about $`O(Vadf+maf)`$, plus softmax reductions. V counts nodes, m edges, a heads, d input width, and f head width. Attention weights need $`O(ma)`$ storage. All-pairs attention would instead assume $`O(V^2)`$ connections.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [PPI study](https://arxiv.org/html/1710.10903v3) reports **micro-F1 0.973 +/- 0.002** on **two held-out tissue graphs**, versus **0.934 +/- 0.006** for a constant-attention control with the same broad architecture. Micro-F1 combines all label decisions to balance missed labels and false positives; higher is better. The 24 graphs contain 44,906 training, 6,514 validation, and 5,524 test nodes.

Protein features and interaction edges become predicted function labels. Comparing learned weights with constant ones tests the value of neighbor weighting. Predictions are not experimentally verified functions. The study establishes neither clinical validation nor a deployed drug-discovery system.

**Notable vendor implementations/libraries:** The authors' release, DGL `GATConv`, and PyTorch Geometric `GATConv` offer GAT. GATv2 changes the scoring rule and cannot inherit the original PPI score without evaluation.

### 1.9.3 GraphSAGE
**In plain English:** GraphSAGE learns about a node by sampling its neighbors and combining their features. Sampling helps it handle large graphs and produce features for nodes it has not seen before.

**Name:** GraphSAGE, focusing on the **supervised mean-aggregator** version, not its unsupervised random-walk training.

**Category & sub-category:** Supervised learning; graph networks. Sampled neighborhoods support learning rules that can work on new nodes.

**Originating paper/vendor/year:** William Hamilton, Rex Ying, and Jure Leskovec, Stanford, [Inductive Representation Learning on Large Graphs](https://arxiv.org/html/1706.02216v4), NeurIPS 2017. The cited revision corrects and clarifies earlier preprints.

**Core mechanism:** Sample a fixed number of neighbors at each step outward from a node. Summarize their features, join that summary with the node's own features, apply learned weights, and normalize. The model learns this reusable rule instead of storing a separate learned vector for every node. Mean, pooling, LSTM, and GCN-style combining rules are different versions. The GCN-style one does not use ordinary GraphSAGE-mean's separate self/neighbor concatenation.

**Inputs/outputs and typical data types:** Node attributes and connections become learned feature vectors and class predictions. Reddit inputs use features derived from post text and edges derived from user interactions.

**Architecture diagram description:** Sampling collects a small neighborhood. Calculations then move from farther nodes toward the root before classification.

```text
root node -> sample neighbors -> sample their neighbors
           -> aggregate outer-hop features -> aggregate nearer-hop features
           -> concatenate self + neighbor mean -> learned transform/ReLU
           -> normalize embedding -> supervised classification head
```

**Activation functions used and why:** Reported variants use ReLU and normalize output embeddings, or learned feature vectors. Reddit's single-label classifier uses softmax. Pooling and LSTM versions add their own transformations.

**Loss function(s):** This version uses supervised class cross-entropy. The paper's **unsupervised** alternative treats nearby random-walk pairs as positives and sampled other pairs as negatives. Its results appear in separate table columns.

**Optimization algorithm(s):** Adam's learning rate is chosen using validation data from 0.01, 0.001, and 0.0001. **Optional math:** Two aggregation depths use sample counts $`S_1=25,S_2=10`$. S1 and S2 are the respective neighbor budgets. The paper does not establish one decay schedule for every variant.

**Regularization techniques:** Random neighbor sampling, normalized features, and validation-selected model size limit training choices. They do not justify importing a later PPI-specific dropout and normalization recipe into this original Reddit result.

**Backpropagation considerations:** Error signals follow the sampled message calculations and learned combining rules. They do not differentiate through the discrete sampling choice. Repeatedly sampled nodes may share cached calculations or be processed more than once, depending on the implementation.

**Parameter count / scaling behavior:** Shared weights need not grow as more nodes join the graph, though stored inputs and edges do. **Optional math:** A mean transform from concatenated width $`2d`$ to output width h has about $`2dh`$ weights, plus optional biases and the classifier. Here d is each input vector's width.

**Training paradigm:** The chosen task is supervised classification on new nodes. GloVe input features were learned in another process; GraphSAGE itself uses class labels here. See [representation learning](05-unsupervised-neural.md) for its self-supervised alternative.

**Hardware/parallelism considerations:** Building sampled batches can be limited by CPU memory or access to the stored graph. GPUs speed feature transformations, but sampling and transfers still count in total runtime.

**Strengths and limitations:** Sampling limits per-batch work and supports new nodes. It also adds randomness and can miss useful neighbors. Several sampling depths can still expand rapidly, even with a fixed neighbor budget at each depth.

**Computational complexity / scalability notes:** Each sampled neighbor can bring more neighbors, multiplying work across depths. **Optional math:** For B root nodes and L hop budgets $`s_1,\ldots,s_L`$, there can be $`O(B[1+s_1+s_1s_2+\cdots+\prod_\ell s_\ell])`$ node appearances before removing duplicates. Each needs features and transformations. "Constant-time" only makes sense with fixed depth, sample sizes, widths, and data-access assumptions.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [Stanford Reddit study](https://arxiv.org/html/1706.02216v4) builds a graph of 232,965 September-2014 posts from 50 communities. Posts connect when the same user comments on both. Features combine averaged title/comment GloVe vectors with post statistics. The first 20 days train the model. Remaining days supply held-out data, with 30% used for validation.

**Supervised GraphSAGE-mean reaches test micro-F1 0.950**, versus 0.585 for the listed raw-feature baseline. Micro-F1 combines classification decisions into a score balancing missed and false predictions; higher is better. A new post and its available neighborhood produce a vector and predicted community. This tests learning for unseen nodes, **not a claim that Reddit deployed GraphSAGE for ranking**.

**Notable vendor implementations/libraries:** Stanford's release, DGL `SAGEConv`, and PyTorch Geometric `SAGEConv` offer variants. Reproduction needs the sampling order, replacement rule, per-layer neighbor counts, normalization, and combining method.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| GCN | Labelled graphs whose neighbors carry useful information | Simple sharing that does not depend on node order | Mixing and averaging can erase important differences | DGL PROTEINS demo; its poor recorded score is disclosed |
| GAT | Graphs where neighbors matter by different amounts | Learns weights along existing edges | Uneven edge work and limits of the scoring rule | Fully supervised PPI function prediction on new graphs |
| GraphSAGE | Large changing graphs with node features | Samples neighbors to handle unseen nodes | Sampling can miss clues; deeper neighborhoods grow | Stanford's supervised Reddit community-classification test |

## 1.10 Metric and event-based networks

These two families do not share one design or hardware type. They appear together because they change how examples are compared or represented. Siamese networks learn relationships between pairs. Spiking neural networks (SNNs) pass time-dependent spike signals. Either choice can sit alongside other kinds of feature networks.

### 1.10.1 Supervised Siamese networks
**In plain English:** A Siamese network applies the same learned feature extractor to two inputs, then compares the results. This can help recognize a new class from just one labelled example.

**Name:** Supervised Siamese networks. The example is Koch, Zemel, and Salakhutdinov's convolutional same/different classifier for one-shot character recognition.

**Category & sub-category:** Supervised learning; metric and event-based networks. Two branches share a feature encoder for pair comparison.

**Originating paper/vendor/year:** Bromley and colleagues' [1993 signature-verification work](https://doi.org/10.1142/S0218001493000339) is an early Siamese reference. This architecture and result come from [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), ICML Deep Learning Workshop, 2015. It was not a main-conference ICML paper.

**Core mechanism:** Each input passes through an identical copy of the same encoder, using shared weights. The comparison head takes absolute differences between their feature values, learns how to weight those differences, and applies sigmoid. During testing, a query is compared with one labelled support example per candidate class. The highest similarity chooses the class without retraining a class-specific output layer.

**Inputs/outputs and typical data types:** Pairs of grayscale characters and same/different labels guide training. Outputs are similarity probabilities, not automatically trustworthy confidence estimates. One-shot decisions also require the support examples' class labels.

**Architecture diagram description:** The two branches share both convolution and dense weights. Their outputs join only at the difference calculation.

```text
image A -> shared Conv[64,128,128,256] -> shared Dense4096 --+
image B -> shared Conv[64,128,128,256] -> shared Dense4096 --+-> abs difference
                                                        -> weighted sum/sigmoid
query versus each labelled support image -> highest similarity -> class
```

**Activation functions used and why:** Convolutions use ReLU; the final feature layer and comparator use sigmoid. This produces a learned similarity score, not necessarily a mathematical distance. For example, it need not obey the triangle inequality, the rule that a direct distance cannot exceed a two-part route.

**Loss function(s):** Binary cross-entropy checks same/different labels, with separate L2 weight penalties by layer. Margin-based contrastive and triplet losses are other Siamese-family choices, **not this result's loss**.

**Optimization algorithm(s):** Momentum SGD uses batches of 128. Layerwise starting rates are chosen from a range, then all are multiplied by 0.99 each epoch. Momentum begins at 0.5 and rises to selected layerwise values. One-shot validation guides stopping, with a maximum of 200 epochs. **Optional math:** The rate-search range is $`10^{-4}`$ to $`10^{-1}`$, or 0.0001 to 0.1.

**Regularization techniques:** Training uses shared weights, layerwise L2 penalties, affine image distortions such as changes in geometric position, and validation stopping. How pairs are formed, and how classes and writers are separated, matters as much as weight penalties.

**Backpropagation considerations:** Both branches send error signals into the **same** encoder weights. Giving each branch independent weights changes the model. Pair sampling determines which similarities are taught and can overemphasize easy comparisons.

**Parameter count / scaling behavior:** Two branches do not double stored encoder weights. The 4,096-unit final feature layer is large. **Optional math:** From a 6-by-6-by-256 map, its matrix alone has $`9,216\times4,096`$ weights, about **37.7 million**, calculated from the architecture. This is not the entire network's parameter count.

**Training paradigm:** Known same/different pairs supervise learning. Testing then uses labelled support examples to classify unseen classes. This is not label-free contrastive pretraining; one-shot support labels still supply information at evaluation.

**Hardware/parallelism considerations:** The branches can run together. Repeated queries can reuse saved support embeddings. Training still stores intermediate outputs for both inputs even though weights are shared.

**Strengths and limitations:** The comparator can handle new categories without a fixed output unit per class. Results depend on varied, useful training pairs. Strong character similarity scores do not prove reliable biometric identification.

**Computational complexity / scalability notes:** One pair needs roughly two encoder passes and a feature-by-feature comparison. Cached support vectors reduce repeated work. **Optional math:** With k support classes and d features per vector, a new query needs one encoder pass and $`O(kd)`$ comparison work. Forming every pair from n examples would grow quadratically; sampled-pair training need not do that.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 2 of the 2015 study](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) reports **92.0% accuracy** across **400 trials of 20-way within-alphabet Omniglot one-shot classification**. Each trial compares a new character with twenty labelled support examples. The study uses 30 alphabets for training, ten for validation, and ten for testing, with separate writers. Other common Omniglot splits cannot replace these details.

The most similar support image supplies the class. The goal is to transfer learned pair-comparison features to unseen classes. This is character recognition, not measured signature-fraud reduction or deployed identity verification.

**Notable vendor implementations/libraries:** Shared Keras models and PyTorch modules can form the branches. Metric-learning libraries also support contrastive and triplet losses. A wrapper alone does not identify the encoder or objective.

### 1.10.2 Spiking neural networks (SNNs)
**In plain English:** An SNN carries a changing numerical state and sends a spike when that state crosses a threshold. This example learns digit labels using approximate backward signals through those sharp decisions.

**Name:** Spiking neural network, specifically the **supervised surrogate-gradient leaky-integrate-and-fire model** in snnTorch Tutorial 5.

**Category & sub-category:** Supervised learning; metric and event-based networks. Neurons communicate using spikes over simulated time steps.

**Originating paper/vendor/year:** SNNs are a broad historical family, not one invention. [Neftci, Mostafa, and Zenke's 2019 review](https://arxiv.org/abs/1901.09948) explains surrogate-gradient training. The concrete recipe is [Eshraghian's snnTorch tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html), based on the project's published guidance.

**Core mechanism:** A neuron adds incoming current to a state that gradually leaks away. This state is called its membrane potential. Crossing a threshold emits a spike, then a reset changes the later state. The sharp spike decision has no useful ordinary training slope. The backward pass therefore uses a smooth substitute, or surrogate, slope. Here, a static image supplies repeated current for 25 simulation steps. It needs neither Poisson random-spike encoding nor an event camera.

**Inputs/outputs and typical data types:** SNNs can read continuous currents or genuine event streams. This example takes 784 normalized MNIST pixel values and produces output spike trains. The digit with the largest output spike count wins.

**Architecture diagram description:** LIF means leaky integrate-and-fire: accumulate input, leak state, emit a spike, and reset. Two learned linear layers drive these states.

```text
static MNIST vector, repeated for 25 steps
 -> Linear(784,1000) -> LIF membrane/spikes
 -> Linear(1000,10) -> LIF membrane/spikes -> spike counts -> digit
each LIF: previous membrane -> leak + input current -> threshold spike -> reset
```

**Activation functions used and why:** A Heaviside threshold makes the forward spike an on/off event. The tutorial uses an arctangent-based smooth curve to supply backward slopes. This does not turn actual spikes into continuous signals. Membrane leak is 0.95 in the example.

**Loss function(s):** Training sums class cross-entropy applied to **output membrane potentials at every time step**. Evaluation instead uses spike counts. Calling this particular training loss "cross-entropy on spike counts" would be wrong.

**Optimization algorithm(s):** The displayed loop trains for one epoch with Adam and no explicit rate decay. **Optional math:** The rate is $`5\times10^{-4}`$, or 0.0005; $`\beta_1=0.9,\beta_2=0.999`$ set Adam's running averages.

**Regularization techniques:** The minimal tutorial specifies neither dropout nor weight decay. Leak, reset, and a fixed simulation length limit state behavior. They do not replace tests on unseen data.

**Backpropagation considerations:** BPTT traces state changes through time and substitutes slopes at thresholds. These are approximate gradients, not exact derivatives of the sharp spike rule. How software handles reset gradients and surrogate scale affects which earlier steps get credit. Changing implementations requires stating these choices.

**Parameter count / scaling behavior:** Membrane states need runtime memory but are not necessarily learned weights. **Optional math:** With two biased linear layers and fixed neuron constants, $`784(1000)+1000+1000(10)+10=\mathbf{795,010}`$ trainable parameters, calculated from the tutorial.

**Training paradigm:** Digit labels supervise this network. Unmodulated spike-timing-dependent plasticity (**STDP**) instead changes nearby connections using spike timing. It is a different local learning rule, covered in the [optimization glossary](10-glossary.md#62-optimization-and-neural-computation), not a separate entry in this edition. Neither all SNNs nor all STDP variants share one supervision type.

**Hardware/parallelism considerations:** Sparse spikes may help on suitable neuromorphic chips, hardware designed for this kind of event processing. A dense GPU simulation still performs repeated time-step calculations and stores states. **No energy saving is claimed here.** Measured energy depends on the chip, event rates, input encoding, numerical precision, memory transfers, simulation length, and whether training or prediction is measured.

**Strengths and limitations:** Explicit state over time can suit event-driven data. Approximate slopes, simulation length, input conversion, and limited hardware portability make fair comparisons with ordinary networks difficult.

**Computational complexity / scalability notes:** More simulation steps repeat the dense-layer work and add saved history. **Optional math:** This version uses roughly $`O(Tp)`$ arithmetic per example and $`O(BT H_{\rm neurons})`$ saved activations for straightforward BPTT. T counts steps, p weights, B batch size, and H-neurons relevant neuron states. True event-driven work instead follows actual connection events. These are different cost models.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html) reports **9,387 correct out of all 10,000 MNIST test images: 93.87% accuracy**. The final test loader keeps the last partial batch. Repeated image currents produce hidden spikes, which affect ten output states. Total output spikes choose the digit.

This shows that the selected spiking model can learn from labels. It does not show superiority over the CNN above, whose training recipe differs. The tutorial reports no neuromorphic-device energy test, sensor deployment, or commercial efficiency gain.

**Notable vendor implementations/libraries:** snnTorch, SpikingJelly, and hardware-oriented neuromorphic tools support related models. Simulating a model in one library does not prove it can run, or save energy, on a particular chip.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Supervised Siamese networks | Labelled pairs and small labelled support sets | Shared comparison features can help with new classes | Pair choice can bias scores; probabilities may be uncalibrated | Omniglot 20-way one-shot character test |
| Supervised surrogate-gradient SNN | Event streams or inputs represented over time | Learns with spikes and carried state | Backward slopes are approximate; efficiency depends on hardware | snnTorch static-MNIST simulation, with no energy claim |

## 1.11 Supervised mixtures

A mixture of experts combines several smaller networks with a gate, or router, that decides their contributions. Experts are subnetworks, not separate people or chatbots. Their training objective determines the supervision category. The 1991 mixture learns labelled examples; GShard learns paired translations. Later sparse language models trained on unpaired text belong in [the model catalog](07-moe-models.md). The [MoE deep dive](08-moe-deep-dive.md) explains shared routing and computing-system ideas.

### 1.11.1 Adaptive mixture of local experts
**In plain English:** This model lets several small networks specialize in different kinds of input. A learned gate combines their predictions and learns which ones tend to work best for each case.

**Name:** Adaptive mixture of local experts. The reference is the **Jacobs/Jordan/Nowlan/Hinton 1991** model.

**Category & sub-category:** Supervised learning; supervised mixtures. Input-dependent weighting encourages different networks to specialize.

**Originating paper/vendor/year:** Robert Jacobs, Michael Jordan, Steven Nowlan, and Geoffrey Hinton, [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jacobs.pdf), Neural Computation 3:79-87, 1991. Researchers at MIT and the University of Toronto developed it, not a later language-model vendor.

**Core mechanism:** All experts read the same input. A gate assigns mixing probabilities, and training encourages each expert to handle cases where it performs relatively well. The paper distinguishes scoring an averaged prediction from scoring individual expert errors. In its probability-mixture version, the known target helps decide each expert's responsibility for a case. More successful experts receive stronger learning signals, while the gate learns where they tend to succeed.

**Inputs/outputs and typical data types:** Numeric features and explicit target vectors become a prediction or output distribution. The original application uses two formants, measurements of sound resonances, to distinguish four vowel classes.

**Architecture diagram description:** Every expert contributes through the gate's weights. Labels train both the experts and the gate.

```text
                       +-> expert 1 -> output distribution --+
input x ---------------+-> expert 2 -> output distribution --+-> weighted mixture
                       +-> ... expert k --------------------+
        \-> gating network -> softmax probabilities ---------+
label y -> expert errors/responsibilities -> update experts and gate
```

**Activation functions used and why:** Softmax makes gate probabilities sum to one. Expert activations depend on the task's output model. The vowel test restricts experts to simple linear boundaries. The paper gives no universal hidden activation recipe and implies no modern ReLU stack.

**Loss function(s):** The paper considers both gate-weighted expert errors and the likelihood of a mixture of expert outputs. Neither is simply squared error after averaging all predictions. The experiment stops at average squared error 0.08.

**Optional math:** The losses include $`\sum_i g_i\|y-f_i(x)\|^2`$ and $`-\log\sum_i g_i\exp[-\|y-f_i(x)\|^2/2]`$. Here x is input, y target, i an expert, fi its prediction, and gi its gate probability. The squared norm adds squared target differences. The second formula uses a Gaussian-mixture-style likelihood, with fixed-scale constants left out.

**Optimization algorithm(s):** Vowel experiments use **full-batch gradient descent with a fixed step size**. Each update uses the entire training set. Limited trials of training progress choose step sizes for each system. There is **no momentum**. Missing numerical schedules are not filled in from modern MoE recipes.

**Regularization techniques:** Small experts and restricted decision boundaries limit capacity. Training encourages specialization, but can leave an expert nearly unused. The model does not have today's extra load-balancing penalty.

**Backpropagation considerations:** Gate and experts receive related but different error signals. Swapping expert identities can leave the same prediction, so specialization need not be unique. Bad starting weights can also lead to unhelpful divisions of work. **Optional math:** A likelihood responsibility is proportional to $`g_i p_i(y\mid x)`$, where gi is gate probability and pi is expert i's probability for target y given input x.

**Parameter count / scaling behavior:** More experts usually mean proportionally more stored expert weights. **Optional math:** Total count is about $`kp_{\rm expert}+p_{\rm gate}`$, with k equal-size experts, p-expert weights per expert, and p-gate gate weights. The study compares four/eight experts with roughly size-matched six/twelve-hidden-unit conventional networks. Modern billion-parameter counts do not apply.

**Training paradigm:** Labelled vowel cases supervise learning. The expert identity may be hidden, but known output targets still make this supervised learning.

**Hardware/parallelism considerations:** Experts can run together, but the classical training objective generally evaluates all of them. This is **not automatically sparse per-token execution** like later top-k Transformer mixtures, where only selected experts run.

**Strengths and limitations:** Specialization can reduce conflict between different subtasks. It need not produce understandable experts, equal workloads, or better results. Some experts may stay unused, while evaluating the full mixture costs more than evaluating one.

**Computational complexity / scalability notes:** If every expert runs, adding experts adds prediction work. **Optional math:** Per example, cost is $`O(kC_{\rm expert}+C_{\rm gate})`$. k counts experts; the C terms are costs per expert and for the gate. Training adds their backward work. Fewer epochs do not automatically mean an equal reduction in elapsed time.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original study](https://www.cs.toronto.edu/~hinton/absps/jacobs.pdf) uses Peterson-Barney vowel formants from **75 speakers**. The first 50 train the system; the remaining 25 test it. Across **25 simulations per configuration**, mixtures and conventional networks both report **90% test classification accuracy**. Four experts take an average **1,124 epochs** to the error stopping point, versus **2,209** for a six-hidden-unit backpropagation network.

The advantage is fewer epochs to the stated error level, **not better test accuracy**. Two formants enter the experts and gate; weighted class distributions choose a vowel. Reducing interference between different input regions motivates the design. The study establishes no speech-service deployment, runtime speedup, or commercial benefit.

**Notable vendor implementations/libraries:** Ordinary network layers and mixture-distribution tools can implement the method. A reconstruction must choose expert output probabilities and the gate explicitly. A modern MoE package is not automatically an exact 1991 reproduction.

### 1.11.2 GShard multilingual translation
**In plain English:** GShard keeps huge sets of subnetworks but uses at most two per token in each expert layer. It spreads the work across many devices to learn translations into English.

**Name:** **GShard's sparsely gated multilingual translation Transformer**, reported in 2020. GShard also names the system that automatically divides the model across devices.

**Category & sub-category:** Supervised learning; supervised mixtures. A distributed encoder-decoder routes translation tokens through selected expert networks.

**Originating paper/vendor/year:** Dmitry Lepikhin and colleagues, Google, [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/html/2006.16668v1), June 2020 technical report, later presented at ICLR 2021.

**Core mechanism:** Every other feed-forward layer in both Transformer stacks is replaced by a mixture-of-experts (MoE) layer. A softmax router chooses up to two expert subnetworks per token. Tokens are grouped, experts have capacity limits, and the second route is partly random. Overloaded experts cannot accept unlimited tokens. The model weights accepted expert outputs, combines them, and adds the result through the residual path.

Automatic sharding splits weights and calculations across devices. The SPMD approach runs the same program on different portions of the data and model. This avoids manually rewriting the network for each device arrangement.

**Inputs/outputs and typical data types:** Source-language subword sequences, paired with English translations, become probabilities for English output tokens. Source and target use distinct vocabularies: multilingual source and English target.

**Architecture diagram description:** Both encoder and decoder alternate dense feed-forward and routed expert layers. The decoder also retrieves source information through cross-attention.

```text
source tokens -> encoder: attention + alternating dense FFN / routed MoE
                                                        |
target prefix -> decoder: masked attention + cross-attention + dense FFN / MoE
                                                        -> English-token scores
MoE: softmax router -> capacity-limited top-2 dispatch -> expert FFNs -> combine
                         [experts sharded across TPU cores]
```

**Activation functions used and why:** Each expert's two-layer feed-forward network uses ReLU. Softmax supplies attention lookup weights, router probabilities, and output-class probabilities. Residual additions and normalization follow the Transformer design. This is not a later sparse language model using SwiGLU.

**Loss function(s):** Translation negative log-likelihood penalizes low probability on paired English targets. An extra balancing loss discourages sending too much work to only a few experts. This supports efficient routing; it does not replace translation supervision.

**Optimization algorithm(s):** Appendix A.2 uses Adafactor. It stores compressed summaries of squared gradients, called factored second moments, and no first-moment average. Update clipping threshold is one. Learning rate is one, followed by inverse-square-root decay after 10,000 steps. These adaptive settings are not interchangeable with an Adam learning rate. **Optional math:** The second-moment schedule is described as $`1-t^{-0.8}`$, where t counts training steps.

**Regularization techniques:** Input, residual, and attention dropout are 0.1, alongside routing balance. Capacity limits and random second-expert routing change which calculations occur. They are training and system choices, not guarantees that all experts receive equal work.

**Backpropagation considerations:** Selected experts receive translation error signals. Router probabilities and balancing loss also receive signals that guide routing. Discrete selection and capacity decisions have no ordinary continuous derivative. Sending tokens to experts and combining outputs must preserve the right token matches in both directions.

**Parameter count / scaling behavior:** The main large model has about **600 billion total weights**, mostly in experts. It uses **2,048 experts per MoE layer** and **36 combined encoder-plus-decoder layers**, not 36 in each stack. The appendix lists model width 1,024, expert/FFN width 8,192, and sixteen attention heads with 128-dimensional keys and values.

**The original 2020 report does not give a precise whole-model active-parameters-per-token inventory.** A route fraction cannot establish that total. **Optional math:** Multiplying 600B by $`2/2048`$ is not a valid shortcut. The fraction represents two routes out of 2,048 experts. It ignores shared layers, embeddings, attention, depth, and whether capacity allows both routes.

A later comparison, [Du et al.'s GLaM Table 2, v2 dated 2022-08-01](https://arxiv.org/html/2112.06905v2#S2.T2), lists **GShard-M4: 600B total / 1.5B activated parameters per input token**. This is the GLaM authors' rounded count under that table's rules for a representative encoder-decoder. It is not an independent calculation here or a universal GShard count. It neither supplies the original's missing precise inventory nor changes the supervised translation objective.

**Training paradigm:** **Supervised translation, not next-token unsupervised pretraining.** The mined parallel corpus has about 25 billion examples across directions; the many-to-English training uses about **13 billion examples**. Web-sourced sentence pairs can be noisy, but their paired translations still supply the learning targets.

**Hardware/parallelism considerations:** Experts must exchange many tokens and outputs across devices. The 600B experiment uses **2,048 TPU v3 cores** for about **four days**. TPUs are specialized learning accelerators. These research measurements do not establish equivalent GPU timing or commercial energy and cost savings.

**Strengths and limitations:** Running only selected experts grows stored model capacity without evaluating every expert per token. Limits remain: uneven routing, communication, memory for deployment, transfer to low-resource languages, and step-by-step output generation. More stored weights do not guarantee better sharing across all languages.

**Computational complexity / scalability notes:** Routing must score experts, selected experts must run, and devices must exchange their data. **Optional math:** With batch size B, length T, model width d, and k experts, routing can cost $`O(BTdk)`$. Selected two-layer experts add about $`O(BT r d f)`$, with expert hidden width f and at most $`r=2`$ accepted routes. Shared Transformer and attention work are additional. Expert storage grows roughly with MoE-layer count times $`kdf`$. Group capacities and device partitioning affect real scaling.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Table 3](https://arxiv.org/html/2006.16668v1) gives **MoE(2048E,36L) 44.3 average BLEU** on the held-out **100-language-to-English evaluation**, versus **36.9** for the listed dense **T(96L)** baseline. BLEU compares wording with reference translations, not percent correct. These are the authors' multilingual test sets and averaging rules, not WMT newstest2014 or a universally comparable aggregate. The mined corpus and details needed to rebuild the exact tests are not fully released.

A source sentence is encoded. Successive English tokens use selected experts while retrieving source information, and beam search chooses a translation. This shows supervised translation scaling, not that Google Translate commercially deployed this exact model. Later sparse language models appear in [07](07-moe-models.md) and [08](08-moe-deep-dive.md).

**Notable vendor implementations/libraries:** The TensorFlow/XLA SPMD sharding approach and related research code show the systems contribution. Modern expert-parallel tools use related ideas, but do not prove that every training artifact or an exact public 600B checkpoint was released.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Adaptive mixture of local experts, 1991 | Labelled data with different kinds of input | Specialists can reduce conflict between subtasks | Every expert runs; some may be unused or unstable | Vowels with separate test speakers: equal accuracy, fewer epochs |
| GShard translation, 2020 | Huge collections of paired multilingual sentences | Runs selected experts across automatically divided devices | Token transfers, capacity limits, and incomplete public artifacts | Supervised translation from 100 languages into English |

## Coverage and continuation manifest

This first-edition chapter covers **30 entries** within the stated limits. Every entry explains its training signal, keeps nine common and nine neural-specific fields, and includes a text diagram. Each worked example labels its evidence type.

| Covered range | Sub-category | Entries |
|---|---|---:|
| 1.5.1-1.5.2 | Basic decision networks: perceptron and MLP | 2 |
| 1.6.1-1.6.10 | Image-filter networks: generic CNN, LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet, ConvNeXt, DenseNet, MobileNet | 10 |
| 1.7.1-1.7.5 | Sequence models: RNN, LSTM, GRU, Bahdanau-attention Seq2Seq, original Transformer | 5 |
| 1.8.1-1.8.6 | Image features and predictions: ViT, Swin, U-Net, Faster R-CNN, YOLOv1, DETR | 6 |
| 1.9.1-1.9.3 | Connected-data networks: GCN, GAT, GraphSAGE | 3 |
| 1.10.1-1.10.2 | Pair comparisons and spikes: supervised Siamese networks and surrogate-gradient SNNs | 2 |
| 1.11.1-1.11.2 | Supervised mixtures: original adaptive local experts and GShard translation | 2 |

**Reading connections.** Start with the [reading guide and evidence policy](00-reading-guide.md) and [supervised classical models](01-supervised-classical.md). [Semi-supervised learning](03-semi-supervised.md) explains graphs with only some node labels, consistency learning, and pseudo-labels. [Unsupervised classical learning](04-unsupervised-classical.md) covers non-neural features and clustering. [Unsupervised neural learning](05-unsupervised-neural.md) covers autoencoders, contrastive and masked-input objectives, and the STDP cross-reference.

[Foundation models](06-foundation-models.md) separates BERT/GPT/T5 pretraining from the original supervised Transformer. [MoE model families](07-moe-models.md) and the [MoE deep dive](08-moe-deep-dive.md) extend the routing discussion. Use the [comparative guide](09-comparative-guide.md) and [glossary](10-glossary.md) to connect terms across chapters.

**Evidence boundaries.** Documentation demonstrations remain identified as demonstrations. That includes the perceptron's score on its training data and the poor PROTEINS result. Historical crops, ensembles, checkpoints, datasets, and losses remain separate. Missing schedules and unpublished artifacts are not replaced by another library's defaults. No proprietary product recipe is assigned to a neural family without evidence. SNN energy claims remain tied to hardware. GShard-M4's active count is credited to the later GLaM comparison, not calculated from the fraction of selected experts.

**Further non-required depth not included.** Additional vision entries could cover Inception-v2/v3, EfficientNetV2, ConvNeXt V2, MobileNet-v2/v3, later YOLO generations, Mask R-CNN, RetinaNet, Deformable DETR, three-dimensional U-Nets, and nnU-Net's configuration algorithm. These do not receive separate full entries here.

Other extensions include graph isomorphism networks and networks with different node or edge types, called relational/heterogeneous GNNs. Temporal GNNs handle changing connections over time. Graph Transformers and advanced neighbor samplers are also further topics. Further sequence topics include continuous-time recurrent models, neural differential equations, and state-space models. These are outside this edition's scope.

Detailed lower-precision, or quantization, recipes and compiler/kernel implementations also need separate treatment. So do deployment studies checking whether predicted chances match real frequencies, and controlled energy comparisons on neuromorphic hardware. Each needs its own evidence and experiments. These limits do not mean the listed architectures exhaust neural learning.
