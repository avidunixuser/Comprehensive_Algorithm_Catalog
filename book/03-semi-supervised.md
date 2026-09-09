# 2. Semi-Supervised Learning Algorithms

This chapter covers 22 methods and clearly defined method families in six categories. It does not cover every semi-supervised method. **Evidence policy date: 2026-09-08.** The papers describe past research, not necessarily today's newest models. Where needed, we separate conference dates from earlier arXiv uploads. Public research code shows that an implementation exists. It does not show that a company uses it in a product.

Imagine a photo collection with a few pictures labeled "cat" or "dog" and many unlabeled pictures. A label is a known answer. **Semi-supervised learning** uses both groups to learn the same task. The labeled pictures teach the class names. The unlabeled pictures may help reveal how pictures vary and which ones belong together. But an unlabeled picture is not a "not-cat" example. A model's guessed label is also not a human-checked answer.

Extra pictures help only when their patterns relate to the task. If the unlabeled collection contains birds, a cat-or-dog model may confidently give them wrong labels. More data can then make learning worse. The methods below differ in how they use the extra data and limit such mistakes.

Use the [reading guide](00-reading-guide.md) for the book's categories and evidence rules. See [supervised neural learning](02-supervised-neural.md) for network designs and [unsupervised neural learning](05-unsupervised-neural.md) for ways to learn useful input features. Features are input details, such as pixel values, or patterns a network learns from them. A **vector** is an ordered list of numbers that can hold those features.

A variational autoencoder, or VAE, can learn without labels inside a larger semi-supervised system. A Transformer can first learn from targets made from its own text, called **self-supervised pretraining**, then learn a task with some labeled examples. The network's design alone does not tell us how it was trained.

### Reading the objectives and costs

A **classifier** predicts a class, such as a digit or a document topic. A neural classifier learns **weights**, numbers that control its calculations. A **loss** is a score for what training should improve; training usually tries to make it smaller. Most neural methods here combine a loss on known labels with a loss on unlabeled examples. The latter may reward confidence, agreement after a small change, agreement with another model, or successful reconstruction of an input.

Several recurring terms will help:

- **Cross-entropy** penalizes giving too little predicted probability to the target answer. A target can name one class or give weights to several classes.
- **Mean-squared error (MSE)** averages squared differences between predictions and targets; lower is better.
- **KL divergence** measures how one probability distribution differs from another. Its direction matters: swapping the two distributions can change the value.
- **Entropy** measures how spread out a prediction's probabilities are. Similar chances for every class mean high entropy. Nearly all the probability on one class means low entropy. Low entropy means confidence, not correctness.
- **Softmax** turns class scores into probabilities that sum to one. Those numbers are not automatically trustworthy chances.
- **Backpropagation** works backward through a network to calculate how changing each weight would change the loss. These change signals are called **gradients**. An **optimizer**, such as SGD or Adam, uses them to update weights. The **learning rate** controls update size.
- A **batch** is a group of examples processed together. An **epoch** is one pass through a dataset. **Dropout** temporarily switches off some network signals during training. **Normalization** rescales or recenters values to help training.

**Optional math:** Write the labeled set as $`D_L=\{(x_i,y_i)\}_{i=1}^{n_l}`$ and the unlabeled set as $`D_U=\{u_i\}_{i=1}^{n_u}`$. Here $`x_i`$ and $`u_i`$ are inputs, $`y_i`$ is a known label, and $`n_l,n_u`$ count the examples. Thus $`n=n_l+n_u`$ is the total. A common loss is:

$$
\mathcal L=\mathcal L_s(D_L)+\lambda(t)\mathcal L_u(D_U).
$$

Here $`\mathcal L_s`$ measures labeled-task loss, $`\mathcal L_u`$ measures the chosen unlabeled loss, and $`\lambda(t)`$ sets its weight at training time $`t`$. The same form does not make different unlabeled losses interchangeable. A sum, a batch average, and an average over only accepted examples have different scales.

**Optional math:** For $`C`$ classes, $`p_\theta(y\mid x)`$ is the probability assigned to class $`y`$ for input $`x`$ by a model with weights $`\theta`$. For a probability list $`q`$, entropy is $`H(q)=-\sum_{c=1}^{C}q_c\log q_c`$. The term $`q_c`$ is the probability of class $`c`$; the sum combines all classes.

Cost notes first describe how work grows, then sometimes give Big-O notation. Unless a field says otherwise, $`d`$ is the number of input features or a named hidden-layer width, $`p`$ is the number of learned parameters, $`B`$ is batch size, and $`E`$ is the number of epochs. $`F`$ is the work for one forward prediction by the stated main network, or **backbone**. Backpropagation adds a network-dependent multiple of that work. For graphs, $`m`$ counts stored links and $`s`$ counts solver steps. The method's name alone cannot determine its cost.

### Assumptions, failure modes, and evaluation

- **Groups should relate to labels.** Methods often assume that examples in a crowded group tend to share a class. This is the **cluster assumption**. One class may fill several groups. The groups do not reveal their names: without a labeled example or other information, a new class usually cannot be identified.
- **The dividing line should pass through gaps.** **Low-density separation** means placing a class boundary where few examples lie. This can fail when real classes overlap or the desired label divides a crowded group.
- **Nearby examples should change in useful, smooth ways.** Inputs may have many numbers but vary mainly along a simpler underlying shape. This lower-dimensional shape is called a **manifold**. The assumption is that inputs lie near such a shape and small moves along its relevant directions change the target smoothly. Closeness in raw pixels, a chosen graph, or a pretrained feature space may not reflect the task.
- **Different views must keep useful information.** Co-training needs two useful ways to describe the same object. Consistency methods need changes that preserve its answer. An image change, also called an **augmentation**, that suits photos may change a digit. Translating text to another language and back, called **back-translation**, may alter "not" or reverse sentiment.
- **Mistakes can teach more mistakes.** In self-training, a wrong guess becomes a training target. The next model may become even more sure of it. This feedback loop is **confirmation bias**. Predicted chances should match how often those answers are actually right. That match is called **calibration**; high confidence alone does not provide it. Confidence cutoffs, a slow start, teacher averaging, and disagreement checks can reduce different risks. None proves a guessed label is right.
- **The extra data may come from the wrong classes.** Such data are **out of distribution (OOD)** for the intended task. A model may confidently force an unknown class into a known one. A graph may also link unrelated groups. Check filtering, handling of unknown classes, and performance within relevant subgroups. Compare with a labeled-only model on matching data. These protections are not automatic. [Oliver et al.'s realistic evaluation, full text v4](https://arxiv.org/html/1804.09170v4) directly shows harm from out-of-class unlabeled examples.

**Transductive versus inductive evaluation.** There are two different questions. Can a method label this particular unlabeled collection? Or can it predict new inputs it has never seen? **Transductive evaluation** answers the first: training may see the evaluation inputs, but not their labels. That is valid if stated clearly. **Inductive evaluation** tests a separate pool whose inputs and labels played no part in fitting or model selection. It better matches the second question.

Graph methods often label only the points already in their graph. To use them on new inputs, specify a prediction rule or how to rebuild the graph. A transductively trained SVM can still have a rule for new inputs. That fact does not change how its original experiment was evaluated.

**Label-budget protocol.** Count labels used for training, validation, teacher training, augmentation choices, and model selection separately. A "100 labels" result might use only 100 answers in its training loss, yet thousands more to choose settings. Validation data help choose settings. Test data should be saved for the final check. A **checkpoint** is a saved model during training; do not choose the best test checkpoint and present it as an estimate for future use.

Report how labels were chosen: equal numbers per class, the population's natural imbalance, groups such as patients or document sources, or another rule. Repeat both the label selection and the random start of training. Give averages or other stated summaries and their uncertainty. A **standard deviation** describes how much results vary across runs. For example, error is the share of wrong answers; its standard deviation in percentage points is a separate measure of variation.

Also record the unlabeled pool's size and source, duplicate removal, overlap of classes, pretraining, image changes, network, training steps, and checkpoint rule. Some historical results below use the best checkpoint. We retain that warning rather than treating them as equivalent to results with a test set kept untouched.

In each worked example, a prediction "decision" means how the answer serves the research task. It does not mean a hospital, search engine, or company deployed the model. Explanations of why a method fits a task are this chapter's analysis unless credited to a source.

## 2.1 Self-labeling and multiple-view approaches

These methods turn guesses on many unlabeled examples into extra training targets. They differ in who makes a guess, when it is accepted, and how they avoid repeating the same mistakes.

### 2.1.1 Self-training and Pseudo-Label

**In plain English:** Learn from a few known answers, guess answers for unlabeled examples, and train on those guesses too. This can use extra data cheaply, but it can also make early mistakes stick.

**Name:** Self-training / pseudo-labeling. The neural example here is Lee's Pseudo-Label.

**Category & sub-category:** Semi-supervised learning; one classifier creates extra labels for itself.

**Originating paper/vendor/year:** Self-training existed before deep learning. Dong-Hyun Lee's [2013 ICML workshop paper, *Pseudo-Label*](https://www.kaggle.com/blobs/download/forum-message-attachment-files/746/pseudo_label_final.pdf) made a simple neural version widely known. It did not invent the whole family. This link is a public copy of the original workshop paper.

**Core mechanism:** First train on labeled examples. Next predict answers for unlabeled inputs and use some predictions as targets. A **hard target** names one class rather than spreading probability across classes. Many later versions accept it only above a confidence cutoff, or **threshold**. Lee's original method instead keeps making fresh hard targets during training and slowly raises their loss weight. A cutoff is not part of its defining procedure. Some versions label a pool with a frozen model between training rounds; others refresh guesses during training.

**Optional math:** A common later rule chooses $`\hat y=\arg\max_c p_\theta(c\mid u)`$. Here $`u`$ is an unlabeled input, $`p_\theta`$ is the current classifier, $`c`$ ranges over classes, and $`\hat y`$ is the highest-probability class. A separate threshold decides whether to use it.

**Inputs/outputs and typical data types:** Start with labeled and unlabeled examples described by the same kinds of features. The result is guessed labels, called **pseudo-labels**, and a classifier for new examples. The base learner may handle images, word-count vectors, or tables.

**Strengths and limitations:** This is a simple addition to many learners and needs no graph of similar examples. It is easy to compare with labeled-only training. But if a dog picture is wrongly labeled "cat," learning that guess can reinforce the error. A confidence cutoff can also leave rare or difficult classes with too little training data. Repeating a prediction is not independent evidence that it is true.

**Computational complexity / scalability notes:** Each labeling round must run the model on all candidate inputs, then pay for retraining. More examples usually mean proportionally more work when passes per example stay fixed. Saving one class per example takes less space than saving every class probability.

**Optional math:** With $`n_u`$ unlabeled inputs and forward cost $`F`$, an offline labeling pass costs about $`O(n_uF)`$. Online training is roughly $`O(EnF)`$ for $`E`$ epochs over $`n`$ examples, under fixed passes per example. Labeled and unlabeled sampling rates may differ. Hard-label storage is $`O(n_u)`$; storing probabilities for $`C`$ classes is $`O(n_uC)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Lee tests handwritten-digit recognition on MNIST with **600 labeled examples**. Pixel vectors enter a network with one hidden layer. Its current digit guesses become targets for unlabeled training images. The final network predicts a digit for each test image.

Table 2 reports **8.57% test error for the dropout network and 5.03% after adding Pseudo-Label**. These rows do not use the extra denoising-autoencoder pretraining found in another row. Keeping them separate makes the contribution of pseudo-labeling clearer. Settings were chosen with a validation set, so 600 counts training labels, not all annotations used for development. The paper reports no production optical character recognition (OCR) deployment or business performance measure (KPI). [Original experiment](https://www.kaggle.com/blobs/download/forum-message-attachment-files/746/pseudo_label_final.pdf).

**Notable vendor implementations/libraries:** [scikit-learn's `SelfTrainingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html) adds self-training to other estimators and offers rules for selecting guesses. It does not reproduce every part of Lee's neural training schedule.

**Architecture diagram description:** Lee's MNIST network is `784 pixels -> dense 5,000 -> dense 10 outputs`. "Dense" means each unit receives inputs from the previous layer. Optional denoising-autoencoder initialization is a separate stage, not a required part of self-training.

**Activation functions used and why:** ReLU keeps positive hidden values and sets negative ones to zero. The original experiment uses **independent sigmoid outputs**, not softmax. Each sigmoid squeezes its own value between zero and one. The paper favors their nearly flat end regions even though each digit belongs to only one class. Many later versions use softmax instead.

**Loss function(s):** The original network adds binary cross-entropy losses for the outputs. Its target has one entry set to one and the others to zero, whether the label is known or guessed. The unlabeled loss gets an epoch-dependent weight. A modern softmax version usually uses categorical cross-entropy instead. Do not swap these losses when trying to reproduce the original result.

**Optimization algorithm(s):** Mini-batch stochastic gradient descent (SGD) with momentum, which carries part of earlier updates forward. The learning rate starts at 1.5 and is multiplied by 0.998 each epoch. Momentum rises from 0.5 to 0.99 over 500 epochs. The update also multiplies the new gradient by one minus momentum. Copying only the learning rate into another optimizer will not reproduce that rule. These settings belong to this sigmoid/dropout network.

**Regularization techniques:** Dropout and a slow start for the unlabeled loss reduce dependence on early guesses. Without pretraining, that loss starts increasing at epoch 100 and reaches weight 3 at epoch 600. Denoising pretraining uses a different schedule and is reported separately.

**Backpropagation considerations:** Hold the chosen class fixed during its update; do not take gradients through the class-selection step. Wrong targets near sigmoid's flat ends can send misleading or weak correction signals. Computing cross-entropy from the raw output scores, called logits, helps avoid numerical overflow.

**Parameter count / scaling behavior:** The two dense layers contain **3,975,010 parameters**, calculated from their sizes. This excludes any optional pretraining decoder. Self-training itself adds no learned parameters.

**Optional math:** The count is $`784(5000)+5000+5000(10)+10=3,975,010`$. Each product counts connections; each following term counts the output units' bias values.

**Training paradigm:** Start with labeled learning, then train jointly on true and self-made labels. Any label-free pretraining is an extra stage and must be counted.

**Hardware/parallelism considerations:** One GPU can handle this reference multilayer perceptron (MLP), a network of dense layers. Larger jobs can split prediction and training across devices. Record which model made each set of pseudo-labels and when they were refreshed.

### 2.1.2 Co-training

**In plain English:** Describe each object in two useful ways and train one classifier on each description. Each classifier can teach the other using confident guesses based on information the other view lacks.

**Name:** Co-training.

**Category & sub-category:** Semi-supervised learning; classifiers exchange guesses across different views of the same examples.

**Originating paper/vendor/year:** Avrim Blum and Tom Mitchell, [*Combining Labeled and Unlabeled Data with Co-Training*, COLT 1998](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf).

**Core mechanism:** Train two classifiers on a small labeled set, one per view. Let confident guesses from each add training information for the other. For example, one view may contain webpage words and the other words in links pointing to that page. The theory assumes that either view can predict the class. It also assumes the views are independent once the class is known, plus conditions that make learning possible. Simply cutting a feature list in half does not meet these conditions. In practice, views that make different errors may still help even without exact independence.

**Inputs/outputs and typical data types:** Each object needs two matching descriptions, a few known labels, and a larger unlabeled pool. The original views are webpage text and incoming-link text. Training returns two classifiers, possibly combined, plus guesses for previously unlabeled examples.

**Strengths and limitations:** A second view can contribute evidence missing from the first. This is less circular than a classifier teaching only itself. Still, both may share and reinforce a mistake. Missing or weak views, unequal class sizes, and confidence scores that are hard to compare across classes make selection difficult. Two available views are not automatically two suitable views.

**Computational complexity / scalability notes:** Every round trains two learners and labels candidate examples. The fits can run at the same time, but must wait for label exchange between rounds. The original word-count learners can work mostly on words that actually appear, rather than a full table of all possible words.

**Optional math:** For $`R`$ rounds, add both learners' training and candidate-prediction costs over all $`R`$ rounds. Sparse multinomial naive Bayes scales with processed nonzero word counts, not necessarily $`nd`$ entries for $`n`$ documents and $`d`$ possible features. There is no one cost bound for every possible pair of learners.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The original study uses **1,051 pages from four universities** to distinguish course homepages from other pages. Each run holds out 263 pages, starts with **3 positive and 9 negative labels**, and treats the remaining 776 pages as unlabeled.

Separate naive Bayes classifiers process page words and incoming-link words. They exchange predictions, and a combined classifier identifies pages for a course-oriented index. Over five random splits, Table 2 reports error falling from **11.1% with supervised training to 5.0% with co-training**. Link text can help because it says things the page itself may not say. This was not evidence of a deployed university search product, and no business KPI was reported. [Study and protocol](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf).

**Notable vendor implementations/libraries:** The paper gives the original procedure using naive Bayes. [scikit-learn's `MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) can provide the word-count classifiers, but not the whole exchange procedure. This entry covers the non-neural original. Using neural networks for the views changes training and hardware needs.

### 2.1.3 Tri-training

**In plain English:** Train three classifiers from slightly different samples of the known answers. When two agree on an unlabeled example, their answer may help train the third.

**Name:** Tri-training.

**Category & sub-category:** Semi-supervised learning; several classifiers use agreement to select extra labels, without requiring separate views.

**Originating paper/vendor/year:** Zhi-Hua Zhou and Ming Li, [*Tri-Training: Exploiting Unlabeled Data Using Three Classifiers*, IEEE TKDE, 2005](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde05.pdf).

**Core mechanism:** Start three classifiers with **bootstrap samples**: random samples of the labeled data that allow repeated examples. Two classifiers' agreement can supply a target for the third. The original method also estimates the agreeing pair's error on labeled examples and limits how many new examples it adds. Adding every agreement is not the full algorithm. The third classifier need not disagree with the pair. Repeating these steps seeks useful differences among classifiers without needing two natural views.

**Inputs/outputs and typical data types:** A labeled set and an unlabeled set with the same fields go in. Three updated classifiers come out. Their majority vote gives the final class. The reference here uses decision trees on table columns, not neural networks.

**Strengths and limitations:** Tri-training needs neither two views that each contain enough information nor trustworthy probability estimates. It uses agreement instead. But agreement is not independent proof: models trained from similar data may share blind spots. With very few labels, their estimated error is also unreliable. Limits on added examples and update checks help because agreement alone can be wrong.

**Computational complexity / scalability notes:** Each round makes candidate predictions with three classifiers and may retrain all three. For trees, work depends on depth, how splits are searched, and the growing training sets. Predicting with a fitted tree follows one path, so deeper paths take more work. Storing several changing pseudo-labeled sets and repeated fits can cost substantial time and memory.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Zhou and Li test the Wisconsin Diagnostic Breast Cancer dataset, listed as **569 records with 30 attributes**. About one quarter is held out. In their 80%-unlabeled setting, labels are hidden for 80% of the remaining training pool.

Three bootstrapped J4.8 trees process the features. Examples selected by agreement help train the third tree, and majority voting predicts the dataset's diagnostic class. Table III reports error changing from **0.094 to 0.075**, from the initial to the final ensemble, averaged over three random partitions. This task lacks the two suitable views that co-training needs. The study classifies past records; it does not show clinical deployment, patient benefit, or a verified medical business KPI. Exact label counts per split depend on rounding. [Original tables and evaluation](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde05.pdf).

**Notable vendor implementations/libraries:** The study uses J4.8 trees and separately tests other base learners. [Weka's J48](https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/J48.html) supplies a tree learner, not the whole tri-training procedure. Neural learners are possible extensions, not the version documented here.

### 2.1.4 Noisy Student

**In plain English:** A teacher model labels a large image collection, then a student learns those answers while seeing noisier images and training conditions. The improved student can become the next teacher.

**Name:** Noisy Student training.

**Category & sub-category:** Semi-supervised learning; large-scale self-training with separate teacher and student networks.

**Originating paper/vendor/year:** Qizhe Xie and colleagues, [*Self-training with Noisy Student improves ImageNet classification*, CVPR 2020; arXiv submission 2019](https://arxiv.org/html/1911.04252v4). The work is associated with Google Research.

**Core mechanism:** Train the teacher on labeled images. Have it label candidates without the noise used to challenge the student. Train an equal-sized or larger student on real labels and teacher guesses, while adding image and network noise. The student must give stable answers under harder conditions, rather than merely copy an easy input-output pair. Replace the teacher with the student and repeat. Teacher size, filtering, class balancing, and noise all matter to the method.

**Inputs/outputs and typical data types:** Use labeled images, a very large image pool, and teacher predictions. The intended result is a better image classifier and possibly a new round of pseudo-labels. A teacher can provide one class per image or a probability list, called a **soft target**. These choices need different storage and losses.

**Strengths and limitations:** It can improve an already strong model, not just one trained on very few labels. A larger student is not limited to copying every teacher mistake. Still, it needs substantial computing power and relevant unlabeled images. Class balancing can repeat wrong assignments. A private image collection makes exact reproduction by others difficult.

**Computational complexity / scalability notes:** Each round labels the candidate pool and trains the student for multiple epochs. A large student may cost more per prediction than its teacher. Labeling the pool is a major task even when done before training. Soft-target storage grows with the number of images times the number of class probabilities saved.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The ImageNet study uses the full labeled ImageNet training set plus **300 million candidate JFT images used without their labels**. Filtering and rebalancing yield **130 million sampled training items representing 81 million unique images**. These counts differ because sampled items need not all be different images.

A teacher labels candidates, and an EfficientNet student learns from the filtered mix. The final model chooses an ImageNet category. EfficientNet-L2 reports **88.4% top-1 accuracy**, meaning its highest-scoring class is correct that often, with 480 million parameters. Student noise helps explain the approach, but the result also depends on size, data, repeated rounds, and image resolution. It is a benchmark, not a demonstrated commercial image-search KPI. [Data preparation and Tables 2 and 8](https://arxiv.org/html/1911.04252v4).

**Notable vendor implementations/libraries:** Google's [Noisy Student research repository](https://github.com/google-research/noisystudent) and released EfficientNet models provide research artifacts. They do not establish use in a particular Google product.

**Architecture diagram description:** The example is **EfficientNet-L2**: `image -> convolutional stem -> repeated expanded mobile inverted-bottleneck blocks with depthwise convolution and squeeze-excitation -> pooling -> classification head`. The stem finds early image patterns. Each repeated block expands then narrows its features, filters channels separately, and adjusts their importance. Pooling summarizes them for classification. Teacher and student are separate networks; Noisy Student does not define a new block design.

**Activation functions used and why:** EfficientNet uses smooth Swish-type functions to control how feature values pass through the network. Sigmoid gates adjust channel importance in squeeze-excitation. Softmax produces the class probabilities. The final projection inside an inverted bottleneck is linear, so not every convolution has the same activation afterward.

**Loss function(s):** Cross-entropy trains on real labels and teacher labels. With soft targets, it rewards matching the teacher's whole probability list. Labeled and unlabeled examples are joined when the paper forms the average loss.

**Technical detail (optional):** Soft-target cross-entropy equals forward KL divergence plus the fixed teacher distribution's entropy. That extra term does not change student gradients.

**Optimization algorithm(s):** The [reference optimizer](https://github.com/google-research/noisystudent/blob/master/utils.py) uses RMSProp with momentum to scale and smooth gradient updates. The paper's large-model schedule starts at learning rate 0.128 for labeled batch size 2,048. It multiplies the rate by 0.97 every 2.4 epochs in a 350-epoch run. The implementation also includes a gradual start and later fine-tuning for resolution.

**Regularization techniques:** RandAugment changes training images; dropout and stochastic depth switch off some signals or blocks in the student. The paper reports final-layer dropout 0.5 and final-block survival probability 0.8. Filtering candidates and balancing classes control data selection, but do not guarantee correct labels.

**Backpropagation considerations:** Keep the teacher's saved targets fixed during each student update. Do not update an offline teacher through the student's loss. Large-batch normalization, resolution changes, and strong image changes affect training separately from the label-guessing rule.

**Parameter count / scaling behavior:** EfficientNet-L2 has **480M parameters** in this study. Its reported training/test resolutions are 475/800. Keeping a teacher adds storage during training or earlier labeling work. Ordinary prediction needs only the final student.

**Training paradigm:** Train a labeled teacher, generate and filter pseudo-labels, train a noisy student, repeat teacher replacement, then make the final resolution adjustment.

**Hardware/parallelism considerations:** This example uses distributed accelerators. Teacher prediction can be divided across machines. Student training must coordinate gradient updates and normalization statistics. Its cost is not representative of self-training on a small table.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Self-training / Pseudo-Label | Images, text, or tables the base learner can handle | Reuses model guesses with little extra machinery | Wrong guesses can reinforce themselves; some classes get too few targets | Lee's MNIST digit benchmark |
| Co-training | Objects with two useful, matching descriptions | One view can supply evidence missing from the other | Both views must be informative enough | Course-page classification from page and link text |
| Tri-training | Tables or vectors with one shared feature set | Uses agreement without requiring two natural views | All three models may share mistakes | Wisconsin Diagnostic Breast Cancer benchmark |
| Noisy Student | Very large labeled and unlabeled image collections | Can improve even a well-trained teacher | Expensive computing and hard-to-reproduce image collections | ImageNet with unlabeled JFT candidates |

## 2.2 Low-density boundaries and manifold regularization

These methods look at where labeled and unlabeled examples sit in feature space. One tries to place class boundaries in gaps. The other links similar examples and discourages their predictions from changing too sharply. This second idea is **manifold regularization**: using the data's simpler underlying shape to guide learning. Neither approach needs the neural image-change pipeline used by many later methods.

### 2.2.1 Transductive and semi-supervised SVM

**In plain English:** Use many unlabeled examples to help place a dividing line between classes. Try to keep the line away from crowded regions while still respecting the few known answers.

**Name:** Transductive SVM (TSVM) / semi-supervised SVM (S3VM). This entry covers support vector machines that choose a class boundary and unknown labels together.

**Category & sub-category:** Semi-supervised learning; class boundaries that avoid regions crowded with examples.

**Originating paper/vendor/year:** Vapnik's statistical learning framework developed the idea of transductive inference: predicting a particular unlabeled pool. Thorsten Joachims's [*Transductive Inference for Text Classification using Support Vector Machines*, ICML 1999](https://www.cs.cornell.edu/people/tj/publications/joachims_99c.pdf) gives the influential text version and practical training procedure used here.

**Core mechanism:** An SVM seeks a boundary with a wide safety gap, called a **margin**, between classes. This version also chooses labels for unlabeled examples. It discourages placing those examples within the margin. A constraint or prior expectation about class proportions matters; otherwise, assigning nearly everything to one class may look attractive. Practical training alternates label changes with SVM fitting, often increasing the influence of unlabeled data gradually. It approximates a hard search rather than finding a guaranteed global best solution.

**Optional math:** A representative two-class objective is:

$$
\tfrac12\lVert w\rVert^2+
C_l\sum_{i\in L}[1-y_if(x_i)]_+
+C_u\sum_{i\in U}[1-|f(x_i)|]_+.
$$

Here $`L`$ and $`U`$ index labeled and unlabeled inputs $`x_i`$. Known labels $`y_i`$ are $`-1`$ or $`+1`$. The score is $`f(x)=w^\top\phi(x)+b`$: $`\phi(x)`$ supplies features, $`w`$ supplies their weights, and $`b`$ shifts the boundary. The notation $`[v]_+`$ means the larger of $`v`$ and zero. $`C_l`$ and $`C_u`$ set the two penalties' strengths. The first term limits weight size; the second penalizes labeled margin errors; the third penalizes unlabeled points near the boundary.

**Inputs/outputs and typical data types:** Use labeled feature vectors and a particular unlabeled pool. Training returns labels for that pool and an SVM scoring rule. The original focus is text with many possible features but mostly zero values. A semi-supervised study may also test the resulting rule on separate, future examples.

**Strengths and limitations:** Unlabeled data can reveal gaps that the labeled set misses. But those gaps may not match the true classes, and the assumed class ratio may be wrong. Searching over unknown labels can get stuck in different locally good solutions depending on the starting point. An ordinary two-class supervised SVM does not have that same nonconvex label-search problem.

**Computational complexity / scalability notes:** Cost depends on the inner SVM solver, similarity rule, cached values, label switches, and schedule for the unlabeled penalty. A full table of pairwise similarities needs about four times the memory when examples double. Linear text versions can exploit the many zero features. Counting only one supervised SVM fit misses the repeated outer search.

**Optional math:** A dense kernel matrix for $`n`$ inputs needs $`O(n^2)`$ memory. A **kernel** supplies the pairwise comparisons used by the SVM; solver and search choices determine the remaining cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Joachims uses Reuters-21578 with the ModApte split: **9,603 training documents and 3,299 test documents**. The study considers the ten most frequent categories while retaining the documents. One experiment supplies only **17 labeled training documents** and exposes the 3,299 target document vectors without labels.

Word endings are reduced, and weighted word features enter the joint boundary-and-label search. The predicted categories organize the target news collection. The paper reports better precision/recall-breakeven performance than an inductive SVM in this low-label setting. Precision measures how often a chosen category is right; recall measures how much of that category is found. Breakeven summarizes where those measures meet. No numerical value is guessed from a plot here. This is explicitly **transductive**: access to the target collection is part of the method's fit. It is not an untouched-test result or a production news-service result. No business KPI is reported. [Sections 5.1-5.3](https://www.cs.cornell.edu/people/tj/publications/joachims_99c.pdf).

**Notable vendor implementations/libraries:** Joachims's [SVMlight](https://www.cs.cornell.edu/people/tj/svm_light/) supports transductive learning. An ordinary `SVC` fitted only to known labels is not a TSVM, even if someone later inspects its predictions on unlabeled examples.

### 2.2.2 Laplacian SVM and manifold regularization

**In plain English:** Connect similar examples, then train a classifier that both respects known labels and changes gently across those links. Unlike a list of graph labels, its learned rule can also classify a new input.

**Name:** Laplacian SVM (LapSVM), part of manifold regularization. Laplacian regularized least squares is a related version with a different loss, not another name for LapSVM.

**Category & sub-category:** Semi-supervised learning; a kernel classifier guided by smooth predictions across a similarity graph.

**Originating paper/vendor/year:** Mikhail Belkin, Partha Niyogi, and Vikas Sindhwani, [*Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples*, JMLR 2006](https://www.jmlr.org/papers/v7/belkin06a.html).

**Core mechanism:** Build a graph whose points are examples and whose links connect similar ones. Fit a kernel-based prediction rule with three goals: match known labels, keep the rule controlled in size, and avoid large prediction differences across strong links. LapSVM uses hinge loss, which penalizes wrong answers and points inside the safety margin. Laplacian regularized least squares, or Laplacian RLS, uses squared error instead. The learned rule combines kernel comparisons with the observed inputs, so it can score new inputs too.

**Optional math:** The framework minimizes:

$$
\frac1{n_l}\sum_{i\in L}\ell(y_i,f(x_i))
+\gamma_A\lVert f\rVert_{\mathcal H}^2
+\frac{\gamma_I}{n^2}\mathbf f^\top L_G\mathbf f.
$$

Here $`L`$ indexes the $`n_l`$ labeled examples, and $`n`$ counts all examples. $`x_i,y_i`$ are an input and label, $`f`$ is the prediction rule, and $`\ell`$ is its labeled loss. $`\lVert f\rVert_{\mathcal H}^2`$ measures the rule's size in the kernel's function space, called an RKHS. $`\mathbf f`$ lists its scores on the graph. The **graph Laplacian** $`L_G`$ encodes linked-score differences. $`\gamma_A,\gamma_I`$ control the two penalties. A representer theorem justifies the finite kernel-based rule described above; the proof is not needed to use the idea.

**Inputs/outputs and typical data types:** Inputs are feature vectors, labels for some of them, a kernel, and a similarity graph. Outputs include a fitted classifier and predictions for new examples. Research examples use image, speech, and document features.

**Strengths and limitations:** It combines a prediction rule for new data with structure from unlabeled data. For a fixed graph and suitable loss, it avoids TSVM's search over label assignments. Still, bad neighbors or poorly scaled distances can hurt. Smoothing across a real class boundary is also harmful. Use validation data to choose the graph and penalty strengths.

**Computational complexity / scalability notes:** Full similarity and kernel tables need roughly four times the storage if the number of examples doubles. Some exact dense solvers can need about eight times the work. A graph with few links makes graph calculations cheaper, but may still leave a full kernel table. Faster iterative solvers and approximations have different costs; report when and why they stop.

**Optional math:** Dense storage is $`O(n^2)`$ for $`n`$ examples, and dense solving can reach $`O(n^3)`$ time. Multiplying a sparse graph by one score vector costs $`O(m)`$, where $`m`$ is its stored link count. This alone does not remove dense kernel costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The USPS digit experiment uses **50 labeled and 1,957 unlabeled examples**, with ten random splits. Each class is learned against the remaining classes, a scheme called **one-versus-rest**. Image features build the kernel and neighborhood graph. Training balances correct labeled predictions with smooth graph scores. The largest class score selects the digit.

Table 3 reports mean error **12.7% for LapSVM versus 23.6% for SVM**. This comparison uses the USPS test-set pool during semi-supervised/transductive learning. It is not the paper's separate experiment on previously unseen images. The result fits the idea that neighboring handwriting features carry useful class information; it does not prove this for every recognition dataset. No production OCR KPI is reported. [Original experimental section](https://www.jmlr.org/papers/volume7/belkin06a/belkin06a.pdf).

**Notable vendor implementations/libraries:** The [Melacci Laplacian SVM library](https://www3.diism.unisi.it/~melacci/lapsvmp/) provides primal training, which optimizes the classifier directly, and related methods. Its documentation says it still stores the whole kernel matrix. It is research software, not a claimed hosted vendor service.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Transductive / semi-supervised SVM | Text or vectors with useful gaps between classes | Uses the target inputs to place a class boundary | Label search can get stuck; assumed class proportions may be wrong | Reuters-21578 topic assignment |
| Laplacian SVM / manifold regularization | Features with useful similarity measures and neighbors | Uses graph links while retaining a rule for new inputs | Bad neighbors and large kernel tables can cause problems | USPS digit classification |

## 2.3 Graph label inference

Here a **graph** holds both labeled and unlabeled examples as points, or vertices. Links show which examples are related. A few known answers help spread class scores across the graph. These two methods use a fixed graph; they are not trainable graph-convolution or graph-attention networks.

### 2.3.1 Label propagation through Gaussian fields and harmonic functions

**In plain English:** Keep the few known labels fixed and let each unlabeled point borrow information from its neighbors. Repeating this averaging spreads the known answers through connected groups.

**Name:** Label propagation, specifically the Gaussian-field / harmonic-function version with fixed labeled scores. Other algorithms also use the broader name "label propagation."

**Category & sub-category:** Semi-supervised learning; predicting labels for the points in an existing graph, a transductive task.

**Originating paper/vendor/year:** Xiaojin Zhu, Zoubin Ghahramani, and John Lafferty, [*Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions*, ICML 2003](https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf). This entry does not treat every label-propagation method as the same objective.

**Core mechanism:** Build links with nonnegative similarity weights that are the same in both directions. Fix each labeled point's class scores to its known answer; this is **hard clamping**. Choose scores for the other points that keep linked scores as close as possible. Each unlabeled point becomes a weighted average of its neighbors. Repeated averaging, with known labels reset after each step, can reach this **harmonic** solution when the graph has the needed connections.

**Optional math:** Let $`W`$ contain link weights, $`D`$ contain each point's total link weight on its diagonal, and $`L_G=D-W`$ be the graph Laplacian. $`F_i`$ is the class-score list at point $`i`$. The smoothness cost is $`\frac12\sum_{ij}W_{ij}\lVert F_i-F_j\rVert^2`$, which adds weighted squared differences over pairs. Fix the labeled scores $`F_L=Y_L`$. The unlabeled scores satisfy:

$$
F_U=-L_{UU}^{-1}L_{UL}Y_L.
$$

Here $`L`$ and $`U`$ mark labeled and unlabeled points. $`L_{UU}`$ and $`L_{UL}`$ are the corresponding blocks of $`L_G`$, and $`Y_L`$ contains known label scores. In practice, solve the linear equations rather than explicitly computing the inverse shown in the formula.

**Inputs/outputs and typical data types:** Start with a weighted graph, or features used to build one, and labels at some points. The output gives class scores at every unlabeled point. A disconnected group with no labeled point lacks a fixed answer to build from. It needs an added rule or prior; otherwise its class solution is not uniquely determined.

**Strengths and limitations:** With suitable labeled anchors, the smoothness rule has a unique solution. Hard clamping preserves trusted labels exactly, but also preserves wrong labels exactly. The graph should mostly connect same-class points, a property called **homophily**. Its connections and similarity scale matter greatly. Scores are not automatically calibrated probabilities whose stated chances match real frequencies.

**Computational complexity / scalability notes:** A full graph needs about four times the construction work and storage when its point count doubles. A dense direct solve can need about eight times the work when the unlabeled count doubles. Sparse graphs allow repeated link-based updates, but finding exact nearest neighbors may itself be expensive.

**Optional math:** Dense construction/storage are quadratic in total points $`n`$. A direct solve is cubic in unlabeled points $`n_u`$, apart from handling several class-score columns. Sparse iteration costs about $`O(smC)`$ for $`s`$ steps, $`m`$ links, and $`C`$ classes. How quickly it settles depends on how well-conditioned the graph equations are.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Zhu and colleagues study handwritten digits from the **CEDAR/Buffalo digit database**. Preprocessing makes each image a 256-dimensional vector. Labeled digits fix scores at some points; image similarities create links. Harmonic averaging fills in scores for other points, then a score-based rule assigns digit classes.

Section 7 plots results at different label budgets against nearest-neighbor and radial-basis-function (RBF) classifiers. It also tests **class mass normalization**, a step that adds prior information about class proportions. That step is not part of the unchanged hard-clamped solution. No exact error is guessed from the curves here, and no production OCR KPI is reported. The method fits the idea of borrowing answers through handwriting neighborhoods, rather than only from an isolated labeled example. [Experiment and formulation](https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf).

**Notable vendor implementations/libraries:** [scikit-learn's `LabelPropagation`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html) offers hard-clamped propagation with supported graph kernels. Graph-building choices and optional class-prior adjustments differ from the study. Its rule for new inputs is also separate from labeling points in the training graph.

### 2.3.2 Label spreading and local-and-global consistency

**In plain English:** Spread label information between neighbors, but keep gently pulling scores toward the known answers. Unlike fixed-label propagation, this method lets even a labeled point's scores change.

**Name:** Label spreading / learning with local and global consistency.

**Category & sub-category:** Semi-supervised learning; graph-based score sharing with adjusted link scales and soft label retention.

**Originating paper/vendor/year:** Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Scholkopf, [*Learning with Local and Global Consistency*, NIPS 2003, proceedings volume published 2004](https://papers.nips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf).

**Core mechanism:** Adjust link weights using how strongly each point is connected overall. Then repeatedly mix two sources: scores passed along those adjusted links and the initial known-label scores. This is **soft clamping**. Labeled points can move away from their initial scores instead of staying fixed. Adjusting for connection strength, or degree normalization, also changes which score patterns count as smooth.

**Optional math:** Let $`W`$ contain nonnegative similarities and $`D`$ contain total link weights on its diagonal. Define adjusted similarities $`S=D^{-1/2}WD^{-1/2}`$. With score table $`F`$ and initial label indicators $`Y`$, update:

$$
F^{(t+1)}=\alpha SF^{(t)}+(1-\alpha)Y,\qquad 0<\alpha<1.
$$

Here $`t`$ counts updates and $`\alpha`$ sets how much to rely on the graph. The settled scores are $`(1-\alpha)(I-\alpha S)^{-1}Y`$, where $`I`$ is the identity matrix. Multiplying every final score by one positive constant leaves the winning class unchanged, but can affect how scores are interpreted as probabilities.

**Inputs/outputs and typical data types:** Use a graph of labeled and unlabeled points with nonnegative links. Output is a class-score list at each point; the highest score usually selects the class. Specify feature scaling, how links are built, and what happens to points with no links.

**Strengths and limitations:** Soft retention can limit harm from one wrong label, and degree normalization controls the influence of highly connected points. Neither identifies which label is wrong. Too much sharing blurs real class differences, called **oversmoothing**. Too little misses useful unlabeled structure. The method is poorly suited to graphs that often link different classes, called **heterophilous** graphs.

**Computational complexity / scalability notes:** On a sparse graph, each step processes links and class scores. Storage must hold both. A full direct solve is much more costly: doubling points can mean eight times the work and four times the storage. Update count depends on the graph, the stopping tolerance, and how strongly scores rely on neighbors. Very strong neighbor weighting can slow settling.

**Optional math:** With $`s`$ steps, $`m`$ links, $`n`$ points, and $`C`$ classes, sparse work is $`O(smC)`$ and storage is $`O(m+nC)`$. Dense solving can take $`O(n^3)`$ time and $`O(n^2)`$ memory. Convergence depends on $`\alpha`$, tolerance, and the graph's spectral properties, which govern how score patterns fade during repeated sharing. Taking $`\alpha`$ near one can be slow.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The digit experiment uses **3,874 USPS images from digits 1-4**, with class sizes 1,269, 929, 824, and 852. Pixels set link strengths. A few labeled points start score sharing, and the largest resulting score selects each unlabeled digit.

The main curves average 100 trials and favor this consistency-based approach over the tested supervised baselines. A separate **one-pixel-jittered affinity variant**, which changes how similarities handle small image shifts, reaches approximately **1% error with 30 labeled points**. This is not the result of every ordinary RBF label-spreading model. The paper also acknowledges using optimal parameters for comparison methods, which limits realistic label-budget claims. Neighbor sharing suits changes in digit shapes better than relying only on a few examples. Still, this is a transductive research result, not a reported business KPI. [Section 4.2 and model-selection discussion](https://papers.nips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf).

**Notable vendor implementations/libraries:** [scikit-learn's `LabelSpreading`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html) implements normalized graph sharing with a clamping factor. It is not a graph neural network and needs no backpropagation through learned layers.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Harmonic label propagation | Similarity graphs whose links mostly join the same class | Spreads scores by a clear averaging rule while fixing known labels | Unlabeled disconnected groups lack anchors; wrong fixed labels stay wrong | CEDAR/Buffalo handwritten-digit study |
| Label spreading | Similarity graphs where known scores should not be fully fixed | Controls highly connected points' influence and allows label scores to move | Bad links or too much sharing can blur classes | USPS digits 1-4 benchmark |

## 2.4 Entropy and consistency

These methods ask for either more decisive answers or more stable answers. Entropy measures how spread out class probabilities are, so reducing it encourages confidence. **Consistency** means agreement after a suitable input change or between model predictions. Neither is the same as correctness. A classifier can be uncertain every time, confidently wrong once, or wrong in the same way every time. Known labels and careful tests are still essential.

### 2.4.1 Entropy minimization

**In plain English:** Encourage the classifier to give more decisive answers on unlabeled examples. This can help separate groups, but confidence alone cannot tell it which answer is right.

**Name:** Minimum-entropy regularization / entropy minimization.

**Category & sub-category:** Semi-supervised learning; encouraging confident class probabilities as an extra training goal.

**Originating paper/vendor/year:** Yves Grandvalet and Yoshua Bengio, [*Semi-supervised Learning by Entropy Minimization*, NIPS 2004](https://papers.nips.cc/paper_files/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf). Their experiments include logistic and kernel-logistic classifiers. The neural example below is the added entropy term tested in [Miyato et al.'s VAT study](https://arxiv.org/html/1704.03976v2), not the original paper's network.

**Core mechanism:** Train on known answers, then add a penalty for uncertain predictions on unlabeled inputs. If dense groups mostly share classes, this can move class boundaries away from those groups. But it cannot name a group correctly without other evidence. Unlike selecting one hard pseudo-label, this method adjusts a smooth function of the full probability list.

**Optional math:** The added term is $`\lambda\,\mathbb E_{u\in D_U}H(p_\theta(\cdot\mid u))`$. Here $`D_U`$ is the unlabeled set, $`u`$ an input, $`p_\theta`$ the classifier, $`H`$ entropy, $`\mathbb E`$ an average, and $`\lambda`$ the penalty weight. Equal probabilities across $`C`$ classes have entropy $`\log C`$; putting all probability on one class gives zero.

**Inputs/outputs and typical data types:** Use a classifier that produces probabilities, some labeled examples, and unlabeled examples. The output is the same kind of classifier. This rule neither reconstructs inputs nor adds a new network design. It can work with differentiable models for tables, images, or language.

**Strengths and limitations:** The extra calculation is cheap and easy to combine with other losses. Too much weight can make the model overconfident or make it choose one class for nearly everything, called **class collapse**. Low entropy does not mean calibrated chances, reliable uncertainty estimates, or detection of out-of-distribution inputs. The original paper examines when unlabeled data can actually help.

**Computational complexity / scalability notes:** Once probabilities exist, calculating their entropy is small work compared with running a large network. Still, processing extra unlabeled inputs requires predictions and gradients. The rule itself needs no graph, teacher, or pairwise-distance table.

**Optional math:** Entropy calculation costs $`O(BC)`$ for $`B`$ examples and $`C`$ classes. That does not include the network's forward and backward passes.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Miyato and colleagues test CIFAR-10 classification with **4,000 labeled training examples**. A Conv-Large network processes changed versions of images. VAT encourages stable predictions near each image; the added entropy term makes unlabeled predictions more decisive. The largest of ten class scores selects an object category.

With image augmentation, the study reports **11.36% error for VAT and 10.55% for VAT+EntMin**, with standard deviations 0.34 and 0.05 percentage points. This tests adding entropy to that recipe, **not entropy minimization alone**. It shows how confidence and stability can complement each other in this experiment, not general superiority over pseudo-labeling. No deployed recognition system or business KPI is reported. [Original ablation table](https://arxiv.org/html/1704.03976v2).

**Notable vendor implementations/libraries:** PyTorch or TensorFlow can compute entropy with stable log-softmax operations, which calculate log probabilities safely. The [VAT TensorFlow reference](https://github.com/takerum/vat_tf) gives a concrete neural example. An entropy function alone is not a complete semi-supervised training and evaluation procedure.

**Architecture diagram description:** The example uses **Conv-Large** from the VAT paper: `RGB image -> three 128-channel convolutions -> pool/dropout -> three 256-channel convolutions -> pool/dropout -> 512-channel convolution -> 256- and 128-channel 1x1 convolutions -> global pooling -> 10-way head`. Convolutions find local patterns; channels hold different learned features. Pooling summarizes them. The final head produces ten class scores.

**Activation functions used and why:** The CNN uses leaky ReLU with slope 0.1: negative values keep a small response rather than becoming zero. This preserves gradient signals there. Softmax provides the probability list whose entropy is penalized.

**Loss function(s):** The basic rule adds positive-weight predictive entropy to labeled cross-entropy and minimizes the total. The worked example also adds VAT's KL penalty for local stability. Reducing uncertainty for each image is different from encouraging varied class use across a whole batch.

**Optimization algorithm(s):** The neural study uses Adam starting at 0.001. Appendix D gives a 48,000-update validation schedule, with linear learning-rate decay over the final 16,000 updates. It separately extends final CIFAR-10 runs to 200,000 updates. This training budget is a study choice, not a requirement of entropy minimization.

**Regularization techniques:** The example also uses batch normalization, dropout, image augmentation, and VAT. Their effects must not all be credited to entropy. The entropy weight controls pressure toward confidence, rather than directly shrinking network weights.

**Backpropagation considerations:** Let the gradient account for how every probability changes the entropy. Treating a copy of the current probabilities as a fixed cross-entropy target is not the same calculation. It can give zero gradient on the output scores.

**Optional math:** In $`-\sum_c p_c\log p_c`$, $`p_c`$ is the current probability of class $`c`$. Both its appearances must receive derivatives. Stable log-softmax avoids directly evaluating $`\log 0`$.

**Parameter count / scaling behavior:** Entropy adds no learned parameters. The listed Conv-Large layers have approximately 3.1M convolution/head weights by arithmetic from their widths. Normalization adds a little extra state.

**Training paradigm:** Learn from labels and unlabeled regularization together. In the example, entropy and VAT train jointly; entropy is not a separate pretraining stage.

**Hardware/parallelism considerations:** Entropy arithmetic is small beside CNN work and can be split across batches on different devices. Safe probability calculations and the makeup of unlabeled batches matter more than special hardware for this loss.

### 2.4.2 Pi Model

**In plain English:** Show the same example to one network twice with different random changes. Train it to give matching answers, even for examples without labels.

**Name:** Pi Model, usually written with the Greek letter Pi as the $`\Pi`$-Model.

**Category & sub-category:** Semi-supervised learning; one model learns to agree with itself under random changes.

**Originating paper/vendor/year:** Samuli Laine and Timo Aila describe the named method in [*Temporal Ensembling for Semi-Supervised Learning*, ICLR 2017; arXiv submission 2016](https://arxiv.org/pdf/1610.02242).

**Core mechanism:** Make two independently changed versions of the same example. Also allow different network noise, such as dropout. Run both through the same weights and penalize differences between their probability lists. Add ordinary classification loss wherever a known label exists. Many unlabeled examples can then teach the model which changes should leave its answer stable.

**Optional math:** The two predictions are $`p_\theta(y\mid a_1(x),\xi_1)`$ and $`p_\theta(y\mid a_2(x),\xi_2)`$. Here $`x`$ is the input, $`y`$ a class, $`\theta`$ the shared weights, $`a_1,a_2`$ the input changes, and $`\xi_1,\xi_2`$ independent network noise. Training penalizes their squared difference.

**Inputs/outputs and typical data types:** Use labeled and unlabeled inputs plus a classifier with random changes during training. The output is one ordinary classifier. The source tests images. Other data need changes that can reasonably preserve their labels.

**Strengths and limitations:** It needs neither saved guesses for every example nor a separate teacher. But two predictions can agree on the wrong answer. Requiring agreement after an unsuitable change can remove useful information. Choose image changes carefully and start with a restrained consistency-loss weight.

**Computational complexity / scalability notes:** Each example needs about two changed-view evaluations, with gradients through both. Only one set of weights is learned, but training keeps intermediate values from both paths. With fixed image-change cost, work still grows roughly with examples and epochs.

**Optional math:** Training remains $`O(EnF)`$ for $`E`$ epochs, $`n`$ examples, and per-example forward cost $`F`$, but with a larger multiplier than labeled-only training.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Laine and Aila use CIFAR-10 with **4,000 labeled images** and the remaining training images unlabeled. Cropped or shifted views pass through one CNN under different noise. Matching their output probabilities teaches stable recognition. The final highest-scoring class selects one of ten categories.

The augmented Pi Model reports **12.36% test error, with standard deviation 0.31 percentage points**. Its useful addition is that unlabeled images also teach which changes to ignore. The result depends on the paper's validation and augmentation procedure. It is a controlled recognition benchmark, not a commercial vision deployment; no business KPI is reported. [Table 1 and implementation appendix](https://arxiv.org/pdf/1610.02242).

**Notable vendor implementations/libraries:** Laine's [temporal-ensembling research code](https://github.com/smlaine2/tempens) includes the Pi Model comparison. Later combined libraries may change the backbone and do not automatically reproduce its numbers.

**Architecture diagram description:** The reference CNN is `32x32 RGB -> 128/128/128 convolutions -> max-pool/dropout -> 256/256/256 convolutions -> max-pool/dropout -> 512 convolution -> 256 then 128 pointwise convolutions -> global average pool -> dense 10`. Each listed width counts feature channels. Pointwise convolutions combine channels at the same image location. The two evaluations share this one network.

**Activation functions used and why:** Leaky ReLU with slope 0.1 keeps a small response and gradient for negative values. Softmax gives probabilities that can be compared across views. Weight normalization rescales weights. Mean-only batch normalization recenters features without the full variance rescaling of ordinary batch normalization. These are this implementation's choices.

**Loss function(s):** Labeled cross-entropy is combined with a gradually increased mean-squared difference between the two probability lists. The comparison uses the appropriate labeled and unlabeled inputs. Averaging over classes and examples affects what the consistency weight means.

**Optimization algorithm(s):** Adam uses maximum learning rate 0.003. Training lasts 300 epochs. The learning rate and consistency weight ramp up during the first 80 epochs; the learning rate drops during the final 50. These are reference settings, not the definition of the Pi loss.

**Regularization techniques:** Gaussian input noise, dropout, weight normalization, mean-only batch normalization, and suitable geometric image changes. Do not automatically transfer horizontal flips from photo tasks to digit tasks.

**Backpropagation considerations:** In the original Pi Model, **both predictions receive gradients**. Freezing one path as a teacher changes the method. Independent noise matters: two identical deterministic paths already agree and give no useful consistency signal.

**Parameter count / scaling behavior:** The listed convolution/head widths give approximately 3.1M weights, plus normalization parameters. Running the model twice does not create two independently learned weight sets.

**Training paradigm:** Train on labels and view agreement together. A warm-up period delays strong trust in the model's unreliable early answers.

**Hardware/parallelism considerations:** A GPU can handle the reference CNN. The two views can share a batch, provided their noise stays independent and normalization matches the intended calculation. Count memory for both paths.

### 2.4.3 Temporal ensembling

**In plain English:** Save a running average of each example's earlier predictions. Use that steadier history to teach the current model instead of making two fresh predictions every time.

**Name:** Temporal ensembling.

**Category & sub-category:** Semi-supervised learning; averaging each example's predictions across training epochs.

**Originating paper/vendor/year:** Samuli Laine and Timo Aila, [*Temporal Ensembling for Semi-Supervised Learning*, ICLR 2017](https://arxiv.org/pdf/1610.02242), first posted in 2016.

**Core mechanism:** Give each training example a saved probability list. After each epoch, combine its old list with its newest prediction, giving recent answers more weight. This is an **exponential moving average (EMA)**. Correct for the fact that the saved list started at zero. The current noisy prediction learns to match the target built from earlier epochs. This averages outputs, not network weights, and avoids two fresh predictions per update.

**Optional math:** For example $`i`$ at epoch $`t`$, update $`Z_i^{(t)}=\beta Z_i^{(t-1)}+(1-\beta)p_i^{(t)}`$. Here $`p_i^{(t)}`$ is the current probability list, $`Z_i^{(t)}`$ the saved average, and $`\beta`$ the old-history weight. Starting from zero requires the corrected target $`\tilde Z_i^{(t)}=Z_i^{(t)}/(1-\beta^t)`$.

**Inputs/outputs and typical data types:** Use labeled and unlabeled examples with stable identifiers. The output is one classifier; the extra stored data hold prediction histories for training examples. Shuffling or moving examples across devices must not mix up those histories.

**Strengths and limitations:** Averaging can make targets less noisy and needs less repeated current-network work than the Pi Model. But targets stay unchanged within an epoch. New examples in a growing stream need cache management, and an old wrong guess can influence the target for several epochs.

**Computational complexity / scalability notes:** Training needs about one fresh noisy evaluation per example plus history updates. The history grows with both the number of examples and the number of classes, which can become costly. Prediction after training uses one model, not an ensemble of old networks.

**Optional math:** For $`n`$ examples and $`C`$ classes, target-cache storage and refresh work are $`O(nC)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The SVHN cropped-house-number experiment uses **500 labels within the official 73,257-image training set**, without its extra-image split. Earlier predictions of each digit image are averaged. Current noisy predictions learn to match that average. The final model remains a single digit classifier.

Table 2 reports **5.12% error with standard deviation 0.13 percentage points**, versus **6.65% with standard deviation 0.53** for the augmented Pi Model in that experiment. This shows how a smoother target can help without doubling current-model evaluation. It does not measure savings in a production address-reading service. No business KPI is reported. [Original SVHN table and appendix](https://arxiv.org/pdf/1610.02242).

**Notable vendor implementations/libraries:** The [authors' implementation](https://github.com/smlaine2/tempens) provides the method. A utility that averages network weights does **not** implement the per-example prediction cache required here.

**Architecture diagram description:** The backbone is the same 128/256/512-channel CNN described for the Pi Model: `noisy image -> CNN -> current probability vector -> consistency against stored historical target`. The cache is extra training storage, not another network layer.

**Activation functions used and why:** Leaky ReLU with slope 0.1 preserves negative-value gradients. Softmax makes class probabilities that can be averaged across epochs. Averaging raw scores, or logits, instead produces a different target.

**Loss function(s):** Add labeled cross-entropy to a gradually increased squared difference between current probabilities and the corrected historical target. During the first epoch, do not punish disagreement with an all-zero, uninitialized cache.

**Optimization algorithm(s):** Adam uses maximum learning rate **0.001 for temporal ensembling on SVHN**, not the paper's usual 0.003. Training lasts 300 epochs, with an 80-epoch ramp-up and a 50-epoch ramp-down. The prediction-average decay is 0.6 in the source experiments.

**Regularization techniques:** Dropout, input noise, weight normalization, mean-only batch normalization, and changes suited to the dataset. Historical averaging steadies targets; it does not replace these other controls.

**Backpropagation considerations:** Treat earlier saved targets as fixed values. Do not send gradients through the full training history. Wrong example identifiers or incorrect zero-start correction can silently make the model learn the wrong objective.

**Parameter count / scaling behavior:** The CNN still has approximately 3.1M convolution/head weights. The extra cache grows as the number of examples times classes. Mean Teacher, below, instead stores another set of weights whose size does not grow with the dataset.

**Training paradigm:** Train jointly with labels and unlabeled targets, refreshing targets each epoch. Afterward, the classifier can predict new examples without keeping their histories.

**Hardware/parallelism considerations:** Distributed training must either merge histories correctly or assign each history consistently to a device. Moving caches and tracking identities may cost more than the averaging calculation itself.

### 2.4.4 Mean Teacher

**In plain English:** Keep a teacher whose weights are a running average of the student's recent weights. This steadier teacher supplies targets for unlabeled examples, including ones it has not seen before.

**Name:** Mean Teacher.

**Category & sub-category:** Semi-supervised learning; learning to agree with a teacher made by averaging model weights.

**Originating paper/vendor/year:** Antti Tarvainen and Harri Valpola, [*Mean teachers are better role models*, NeurIPS 2017](https://arxiv.org/html/1703.01780v6). The cited expanded arXiv version dates from 2018.

**Core mechanism:** Train a student using known labels and agreement with teacher predictions under independently changed inputs or network noise. After each student update, mix the teacher's old weights with the student's new weights. This exponential moving average smooths the teacher's changes. It stores weights rather than each example's prediction history. The teacher can therefore make a target for a newly sampled image immediately and refreshes every step rather than every epoch.

**Optional math:** The update is $`\theta'_t=\alpha\theta'_{t-1}+(1-\alpha)\theta_t`$. Here $`\theta_t`$ is the current student weight set, $`\theta'_t`$ is the teacher weight set, $`t`$ counts updates, and $`\alpha`$ controls how much old teacher weight to retain.

**Inputs/outputs and typical data types:** Use labeled and unlabeled inputs, independent student/teacher changes, and two weight sets. The result is one chosen classifier, commonly the final averaged teacher. The paper tests both small CNNs and networks with skip connections, called residual networks. Their results are not interchangeable.

**Strengths and limitations:** It avoids a cache that grows with the dataset and can stabilize targets. But the teacher is built from the student's history, not independent truth. Too much averaging makes it slow to adapt. Poor image changes and unreliable early guesses can still cause failure.

**Computational complexity / scalability notes:** Student training is joined by one teacher forward pass. Averaging and storing teacher weights cost more as the network grows, not as the dataset grows. Only the student needs ordinary training gradients. Final prediction does not run every historical model.

**Optional math:** With $`p`$ parameters, averaging costs $`O(p)`$ per step and teacher storage is $`O(p)`$. These costs do not depend on the number of examples $`n`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On SVHN house-number recognition with **250 labels**, the paper's non-residual convolutional Mean Teacher reports **4.35% error**. Normalized digit images receive independent changes. The averaged teacher supplies probabilities, and the student combines those targets with the few known labels. Prediction selects one of ten digits.

Compared with temporal ensembling, it updates targets every step without keeping a record per image. This number belongs to the CNN experiment, not the separate, substantially different residual-network results. The standard SVHN training data and the authors' validation-label procedure apply. No production address-processing KPI is reported. [Paper abstract and experimental appendix](https://arxiv.org/html/1703.01780v6).

**Notable vendor implementations/libraries:** [Curious AI's Mean Teacher repository](https://github.com/CuriousAI/mean-teacher) contains TensorFlow and PyTorch research implementations. A framework's weight-averaging utility provides only one piece of the method.

**Architecture diagram description:** The example uses the paper's Table 6 CNN: convolutional groups with 128, 256, then 512/256/128 channels, followed by a pooled ten-class head. `Student view -> CNN(theta)` and `teacher view -> identical CNN(theta_EMA)` meet at a consistency loss. The paths have the same layer design but different weight states.

**Activation functions used and why:** Leaky ReLU with slope 0.1 keeps gradients on the negative side. Softmax places both models' outputs on a common probability scale. The small CNN uses weight normalization and mean-only batch normalization. The paper's residual models make separate design and regularization choices.

**Loss function(s):** The small-CNN experiment adds labeled student cross-entropy to mean-squared differences between teacher and student probabilities. The paper also studies KL-based agreement. Choosing that instead changes the loss's behavior and scale, so its coefficient is not directly interchangeable.

**Optimization algorithm(s):** Adam uses maximum learning rate 0.003 for the small CNN. On semi-supervised SVHN, the rate and consistency weight rise over 40,000 steps, with no final ramp-down. The no-extra-data run lasts 180,000 steps. Teacher decay and Adam's second-moment coefficient, which smooths squared gradients, change from 0.99 during ramp-up to 0.999 afterward.

**Regularization techniques:** Input noise, translations, dropout, normalization, and teacher averaging. The reference SVHN sampler uses one labeled and 99 unlabeled examples per minibatch. Other datasets use different ratios.

**Backpropagation considerations:** Hold teacher targets fixed while updating the student. Teacher averaging is an explicit weight update, not gradient descent on a teacher loss or differentiation through earlier training steps. Deliberately choose how to copy or average normalization state as well as weights.

**Parameter count / scaling behavior:** Each reference CNN has approximately 3.1M convolution/head weights. Training stores roughly two model states plus the student's optimizer state. Prediction needs only one selected model.

**Training paradigm:** Train the student with labels and agreement with its continually averaged teacher. The basic recipe needs no separately pretrained teacher.

**Hardware/parallelism considerations:** Teacher predictions add work, but need less saved intermediate state than a second fully trainable network. Across devices, keep teacher averages and normalization state consistent.

### 2.4.5 Virtual adversarial training

**In plain English:** Find a small input change that most unsettles the model's current answer. Then train the model to resist that change, without needing a true label for the example.

**Name:** Virtual adversarial training (VAT).

**Category & sub-category:** Semi-supervised learning; stable predictions under deliberately challenging nearby input changes.

**Originating paper/vendor/year:** Takeru Miyato and colleagues introduced a [distributional-smoothing formulation](https://arxiv.org/abs/1507.00677) in 2015. The expanded [*Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning*](https://arxiv.org/html/1704.03976v2) was posted in 2017 and revised in 2018 for the journal work.

**Core mechanism:** First predict a probability list for an input. Search for a small change that most alters that list. Then train the classifier to keep the changed prediction close to the original, while also learning from known labels. "Virtual" means the target is the model's own prediction, not the unknown true answer. Unlike random noise, the change deliberately follows a direction where the model is locally fragile.

**Optional math:** VAT approximates:

$$
r_{\rm vadv}\approx\arg\max_{\lVert r\rVert\le\epsilon}
D_{\rm KL}\!\left(\operatorname{sg}[p_\theta(\cdot\mid x)]
\parallel p_\theta(\cdot\mid x+r)\right).
$$

Here $`x`$ is the input, $`r`$ its change, and $`\epsilon`$ the allowed size under the chosen norm $`\lVert r\rVert`$. $`p_\theta`$ is the classifier and $`D_{\rm KL}`$ compares its two probability lists. $`\operatorname{sg}`$ means hold the original list fixed for gradients. $`r_{\rm vadv}`$ is the estimated most disruptive allowed change. A finite-difference power iteration estimates that direction using gradient changes, rather than building a full Hessian table of second derivatives.

**Inputs/outputs and typical data types:** Use inputs or learned input vectors for which gradients can be computed, some labeled examples, unlabeled examples, and a rule limiting change size. The output is a classifier trained for local stability. The changed inputs help training; they are not newly verified examples.

**Strengths and limitations:** It finds challenging directions without true labels and does not rely only on hand-chosen image changes. But the allowed radius makes sense only relative to input scaling and preprocessing. Stability nearby does not prove protection against all adversarial changes. A changed image can also leave the pattern followed by real images, the data manifold.

**Computational complexity / scalability notes:** Each search iteration adds gradient work, followed by a prediction on the changed input. A fixed small number of iterations multiplies ordinary training cost by a constant factor. It does not require storing a dense Hessian. More search iterations, longer inputs, or larger networks still increase the actual bill.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Miyato and colleagues classify CIFAR-10 images with **4,000 labeled examples**. Conv-Large first predicts probabilities. VAT estimates a sensitive input direction and trains the network to keep its answer stable there, while respecting known labels.

With image augmentation, the paper reports **11.36% test error with standard deviation 0.34 percentage points for VAT**. The separate 10.55% VAT+EntMin result includes the extra confidence penalty described above. VAT fits tasks where stability near unlabeled images helps beyond available geometric image changes. This is not evidence of a deployed adversarially secure system or a business KPI. [Original comparison and Appendix D](https://arxiv.org/html/1704.03976v2).

**Notable vendor implementations/libraries:** [The authors' VAT TensorFlow code](https://github.com/takerum/vat_tf) provides the reference. A general adversarial-attack library may use a different fixed target, change-size rule, or loss.

**Architecture diagram description:** The example uses **Conv-Large** from the entropy entry: 128/256-channel convolutional groups, a final 512/256/128-channel group, pooling, and a ten-class head. `Input -> prediction -> perturbation search -> perturbed input -> same CNN` describes extra training steps, not an added prediction-time network.

**Activation functions used and why:** Leaky ReLU with slope 0.1 preserves negative-side gradients. Softmax supplies the probability list compared by KL divergence. The source CNN also uses batch normalization. The search must not accidentally change the normalization behavior it is testing.

**Loss function(s):** Add labeled cross-entropy to the average KL difference between the fixed original prediction and the prediction in the estimated challenging direction. Entropy minimization is an optional **additional** term, not part of VAT's definition.

**Optimization algorithm(s):** Appendix D starts Adam at 0.001. Validation runs use 48,000 updates with linear decay over the final 16,000; final CIFAR-10 runs separately extend to 200,000 updates. Choosing the radius uses validation data, so its labels count toward the tuning budget.

**Regularization techniques:** VAT itself, plus the experiment's dropout, batch normalization, and optional image augmentation. Neither the choice of norm nor the radius is one universal setting for all images.

**Backpropagation considerations:** Hold the original probabilities fixed and normally treat the completed input change as fixed during the outer model update. This avoids sending gradients through the inner search. Very small finite-difference steps need enough numerical precision. A full Hessian is unnecessary.

**Parameter count / scaling behavior:** VAT adds no learned parameters to the approximately 3.1M-weight Conv-Large network. Extra memory holds changed-input activations and input gradients, not another independently trained classifier.

**Training paradigm:** Learn from true labels and local stability together. The stability term needs no labels and can use unlabeled inputs, and potentially labeled inputs too.

**Hardware/parallelism considerations:** GPUs and automatic differentiation help compute input gradients efficiently. In distributed training, measure the allowed change separately for each example. Accidentally normalizing across an entire multi-device batch changes the method.

### 2.4.6 Unsupervised Data Augmentation

**In plain English:** Make a much-changed version of an unlabeled image or document that should keep the same meaning. Train its prediction to match the answer for the original or gently changed version.

**Name:** Unsupervised Data Augmentation for Consistency Training (UDA).

**Category & sub-category:** Semi-supervised learning; agreement under strong changes suited to the data.

**Originating paper/vendor/year:** Qizhe Xie and colleagues, [*Unsupervised Data Augmentation for Consistency Training*, NeurIPS 2020; arXiv submission 2019](https://arxiv.org/html/1904.12848v6). Despite "Unsupervised" in the name, the classifier training here also uses known labels.

**Core mechanism:** Predict on an original or weakly changed unlabeled input. Train the classifier to keep that answer on a strongly changed version. A **weak** image change is relatively gentle; a **strong** one is more challenging. Both should preserve the task's answer. For text, translating to another language and back can create a changed version. UDA emphasizes making these changes useful, not just adding arbitrary noise.

The framework also discusses rejecting low-confidence targets, sharpening probabilities toward stronger preferences, and **training-signal annealing**. The last option temporarily limits the influence of already-easy labeled examples early in training. State which options and weights each experiment uses.

**Inputs/outputs and typical data types:** Use labeled and unlabeled images or documents plus a way to make changed versions. The output is a classifier and task probabilities. A back-translation model has its own training data and cost. Those resources are not free just because the final task has few labels.

**Strengths and limitations:** Rich changes can teach more useful stability than small random noise, and UDA can use pretrained features. But even a fluent rewrite can change a review's sentiment. A large pretrained model already reflects substantial earlier data and computing. Fine-tuning with twenty labels is not learning from scratch with twenty examples.

**Computational complexity / scalability notes:** Training needs original-view predictions, changed-view learning, and the work of making the changes. For text, comparing every token position with every other can grow quickly with sequence length. Other Transformer calculations also matter. Back-translation done beforehand still counts as resource use.

**Optional math:** A dense Transformer layer costs $`O(T^2d+Td^2)`$ for $`T`$ token positions and hidden width $`d`$, when its feed-forward width grows with $`d`$. Attention supplies the first part, not the full layer cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Table 4 tests IMDb review sentiment with **20 labeled reviews**. The **BERT_FINETUNE** model is BERT-Large additionally pretrained on unlabeled text from the same domain, then trained with UDA. Reviews and their back-translations become token sequences. The model predicts positive/negative probabilities and selects the review's sentiment.

For that initialization, the paper reports **4.20% error with UDA versus 6.50% without UDA**. This is not a randomly initialized Transformer result or a plain BERT-Large result. It also does not show that only twenty annotations were used across all development stages. No additional unlabeled-corpus count is inferred from the headline table. No deployed review-analysis service or business KPI is reported. [Table 4 and Appendix E.1](https://arxiv.org/html/1904.12848v6).

**Notable vendor implementations/libraries:** Google's [UDA research repository](https://github.com/google-research/uda) includes text and image code. A library that only changes inputs does not supply the full consistency, sampling, and model-selection procedure.

**Architecture diagram description:** The text example uses **BERT-Large**: `WordPiece and position embeddings -> 24 Transformer encoder layers, hidden width 1,024 and 16 attention heads -> pooled [CLS] representation -> two-class head`. WordPiece splits text into tokens; embeddings turn tokens and positions into vectors. Attention combines information across positions. The final classifier uses a summary from the special `[CLS]` token. The [BERT paper](https://arxiv.org/html/1810.04805v2) establishes this design. UDA's image experiments use separate CNN backbones.

**Activation functions used and why:** GELU smoothly controls values in BERT's feed-forward layers. Softmax turns attention scores and class scores into normalized weights or probabilities. The original tanh pooler bounds its transformed summary values. LayerNorm rescales features, and residual connections provide shortcut paths that help training. These are BERT choices, not UDA requirements.

**Loss function(s):** Add labeled cross-entropy to weighted KL agreement between a fixed original-view target and the changed-view prediction. The text setup reports unlabeled weight 1. Do not assume image-specific temperature or confidence settings also produced the IMDb result without checking.

**Optimization algorithm(s):** The [text reference optimizer](https://github.com/google-research/uda/blob/master/text/bert/optimization.py) uses an Adam-family update with weight decay, which shrinks weights. Its rate rises linearly at the start and then falls linearly. Appendix E.1 explores fine-tuning rates $`10^{-5},2\times10^{-5},5\times10^{-5}`$. These are candidate learning rates, not one claimed setting for every dataset.

**Regularization techniques:** Back-translation, consistency, and BERT dropout of 0.1 in the cited fine-tuning setup. Check that a changed review keeps its sentiment; plausible wording alone is not enough.

**Backpropagation considerations:** Hold the original-view target fixed and send gradients through the changed-view classifier. Ordinary UDA does not differentiate through the discrete translation process. Keep tokenization and sequence-shortening rules consistent across views.

**Parameter count / scaling behavior:** BERT-Large has approximately **340M parameters**, plus its small task head. UDA adds no required learned backbone parameters. A model that generates augmentations is a separate model and resource.

**Training paradigm:** First use self-supervised BERT pretraining. BERT_FINETUNE then adds in-domain pretraining, followed by semi-supervised task fine-tuning. These are different stages with different sources of training targets.

**Hardware/parallelism considerations:** The text experiments use a v3-32 Cloud TPU Pod and length-512 sequences. This states the source hardware, not the minimum for every UDA task. Generating changed inputs and storing long sequences can dominate cost.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Entropy minimization | Probability-based classifiers where gaps help separate classes | Adds confidence pressure cheaply | Can make wrong answers too confident or favor one class | Added entropy in the CIFAR-10 VAT comparison |
| Pi Model | Images or other inputs with safe random changes | Needs no saved history or separate teacher weights | Trains through two views that may share a wrong answer | CIFAR-10 with 4,000 labels |
| Temporal ensembling | A fixed dataset whose examples keep stable identities | Averages earlier answers with fewer fresh evaluations | Saves targets per example and refreshes only each epoch | SVHN with 500 labels |
| Mean Teacher | Large or changing unlabeled pools | Refreshes teacher weights every step without per-example histories | Teacher can inherit the student's mistakes | SVHN with 250 labels |
| Virtual adversarial training | Images or vectors with usable input gradients | Finds small challenging changes without labels | Sensitive to change size; requires extra gradient work | CIFAR-10 with 4,000 labels |
| UDA | Images and text with strong changes that preserve answers | Learns stability under more than mild noise | Changes may alter labels; earlier models have their own costs | IMDb sentiment with BERT_FINETUNE and 20 labels |

## 2.5 Combined modern recipes

These methods combine training ideas rather than each inventing a new network. Their small-image examples often use **Wide ResNet-28-2 (WRN-28-2)**, with approximately 1.5M parameters. It starts with a convolution, then uses three groups of four residual units. A residual unit has a shortcut that adds earlier features to newly calculated ones. Feature widths grow through roughly 32/64/128 channels, followed by global pooling and a class head.

The cited implementations normalize features before residual-layer operations and use leaky ReLU. These are choices in the reference code, not requirements of every Wide ResNet. See the [Google reference backbone](https://github.com/google-research/fixmatch/blob/master/libml/models.py) and [TorchSSL backbone](https://github.com/TorchSSL/TorchSSL/blob/main/models/nets/wrn.py).

Do not read results from different papers as one controlled ranking. Later TorchSSL tables reimplement earlier methods and report checkpoints differently. Each example below identifies its actual comparison. A changed score cannot always be credited to a changed threshold rule.

### 2.5.1 MixMatch

**In plain English:** Average several guesses for an unlabeled image, then train on blends of images and their target answers. This combines learning from guesses with learning to behave smoothly between examples.

**Name:** MixMatch.

**Category & sub-category:** Semi-supervised learning; combining soft label guesses, stronger confidence, and training on mixed examples.

**Originating paper/vendor/year:** David Berthelot and colleagues, [*MixMatch: A Holistic Approach to Semi-Supervised Learning*, NeurIPS 2019](https://arxiv.org/html/1905.02249v2).

**Core mechanism:** Make several changed views of an unlabeled image and average the model's predictions. **Sharpen** that average by giving its larger probabilities more emphasis. Keep it as a soft probability target, not necessarily a single chosen class. Then blend pairs of images and blend their targets by the same amount, a step called **MixUp**. Train on these mixed pairs as well as the effects of image changes.

**Optional math:** Sharpening uses $`q_c\propto\bar p_c^{1/\tau_s}`$. Here $`\bar p_c`$ is the average guessed probability for class $`c`$, $`\tau_s`$ is a temperature controlling sharpness, and $`q_c`$ is the new target, normalized so its entries sum to one. Mixing uses $`x'=\lambda x_a+(1-\lambda)x_b`$. Here $`x_a,x_b`$ are two inputs, $`x'`$ their blend, and $`\lambda`$ sets the share of each; targets are blended with the same share.

**Inputs/outputs and typical data types:** Use labeled and unlabeled images and changes that should preserve their labels. The output is a classifier for new inputs. A blended cat-and-truck picture has a blended training target by design; no person has independently observed a fractional class label for it.

**Strengths and limitations:** Several controls work together, and soft targets let the method use less-confident examples. But it is more involved than plain pseudo-labeling. Errors can enter during guessing, sharpening, or mixing. Sharpening a wrong target can make it harder to correct. Mixing raw inputs may be unsuitable for some discrete data or tasks with strict structure.

**Computational complexity / scalability notes:** Guessing labels needs several forward passes for each unlabeled image. Training then processes mixed examples. If the number of views stays fixed, work grows roughly in proportion to examples and epochs, with a network-dependent multiplier. No graph or dataset-sized prediction cache is required.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The CIFAR-10 experiment exposes **250 training labels** from the 50,000-image training pool. Changed images produce averaged, sharpened guesses. Mixed images and matching targets train WRN-28-2. The largest class score selects an object category on the held-out test set.

The paper reports **11.08% error with standard deviation 0.87 percentage points**. It reports median error over the last twenty checkpoints and uses a separate **5,000-example validation resource for hyperparameter selection**. Thus 250 is not the whole development-label budget. Compared with plain pseudo-labeling, the recipe keeps soft guesses and adds pressure for sensible behavior between examples. No commercial image-recognition KPI is reported. [Implementation details and CIFAR-10 results](https://arxiv.org/html/1905.02249v2).

**Notable vendor implementations/libraries:** Google's [MixMatch research code](https://github.com/google-research/mixmatch) includes its [training implementation](https://github.com/google-research/mixmatch/blob/master/mixmatch.py). Record runtime options rather than assuming their defaults match every published experiment.

**Architecture diagram description:** The example uses **WRN-28-2**: `augmented image views -> shared residual CNN -> averaged/sharpened targets; mixed images -> same CNN -> supervised and unlabeled losses`. Building targets adds calculations, not a new feature-extracting network.

**Activation functions used and why:** The reference residual network uses leaky ReLU with slope 0.1, preserving some gradient for negative values. Batch normalization helps control feature scales. Softmax gives probabilities suitable for averaging. Sharpening changes target probabilities, not hidden-layer activations.

**Loss function(s):** Mixed examples that started in the labeled group use cross-entropy. Those that started in the unlabeled group use squared probability error. Both use mixed targets. This two-loss choice differs from methods that use cross-entropy for both groups.

**Optimization algorithm(s):** The public code uses Adam with default learning rate 0.002. The paper describes a rate that does not decay, evaluation using averaged model weights, and a 16,000-step linear increase of the unlabeled-loss weight. Later reimplementations do not necessarily use Adam.

**Regularization techniques:** Image changes, low-temperature sharpening, MixUp, weight shrinkage, and an exponential moving average of weights. Record the actual shrinkage rule: the code multiplies its weight-decay option by the learning rate. The option alone is not the fraction removed from weights each step.

**Backpropagation considerations:** Hold guessed targets fixed when computing training gradients. Otherwise, the model could move its own target toward an easier answer in that same update. Keep images aligned with their targets during mixing and minibatch interleaving. Interleaving also affects the batch-normalization statistics.

**Parameter count / scaling behavior:** The cited WRN-28-2 has approximately **1.5M parameters**. A weight-average copy adds stored state for training/evaluation, not a new learned design. Making layers twice as wide makes many convolutional weight counts about four times as large.

**Training paradigm:** Learn from labels and guessed targets jointly. The standard reference setting makes two changed-view predictions per unlabeled example. The paper commonly uses sharpening temperature 0.5.

**Hardware/parallelism considerations:** GPUs can process batches directly, but extra guesses and mixed-image training raise throughput needs. Shuffling data across devices must move targets with their images.

### 2.5.2 ReMixMatch

**In plain English:** Use a gentle image change to make a target, then teach the model to keep that answer under several harder changes. Also adjust guesses when the model is using some classes too often.

**Name:** ReMixMatch.

**Category & sub-category:** Semi-supervised learning; combining class-frequency adjustment with gentle-to-strong image agreement.

**Originating paper/vendor/year:** David Berthelot and colleagues, [*ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring*, ICLR 2020; arXiv submission 2019](https://arxiv.org/html/1911.09785v2).

**Core mechanism:** Track how often the model predicts each class. Adjust a guess using the ratio of expected class frequency to that running predicted frequency, then rescale probabilities to sum to one and sharpen them. This is **distribution alignment**. It counters class bias across many predictions, but depends on having useful expected frequencies.

A weakly changed image provides the target, or **anchor**, for several strongly changed views. This avoids averaging potentially damaging strong views into the target. The recipe also keeps MixUp and adds a task that predicts image rotation. Its CTAugment procedure adapts image-change choices during training.

**Inputs/outputs and typical data types:** Use labeled and unlabeled images, estimated overall class frequencies, and weak/strong image changes. Overall frequencies are also called the **class marginal**. The output is a classifier plus extra training state. An auxiliary rotation head, when used, is not needed for ordinary class predictions.

**Strengths and limitations:** Better targets can make stronger image changes useful. But alignment helps only if its expected class frequencies fit the task. Equal class counts in a small labeled seed do not prove an equally balanced real population. Images from unknown classes can be forced into known ones. More components also mean more interacting choices to validate.

**Computational complexity / scalability notes:** Several strong views increase network work; the paper's standard configuration uses eight augmentations. Mixing examples and predicting rotations add more work. Tracking class frequencies is relatively small. This is not as cheap as a recipe with one strong view.

**Optional math:** Storing running frequencies takes $`O(C)`$ space for $`C`$ classes, apart from the networks and batch data.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The CIFAR-10 study uses **250 labeled training images**. A weak view supplies a guess. Running class statistics adjust it, and strong views learn against the adjusted target. The final classifier chooses an image category.

Table 1 reports **6.27% error with standard deviation 0.34 percentage points**, compared with **11.08% and 0.87** for the MixMatch row in that table. A gentle-view target may be more dependable than an average across difficult image changes. However, the paper warns that some externally reported methods use different implementations. Not every row isolates the effect of one component. No production computer-vision or business KPI is reported. [Original Table 1 and comparison notes](https://arxiv.org/html/1911.09785v2).

**Notable vendor implementations/libraries:** Google's [ReMixMatch repository](https://github.com/google-research/remixmatch) includes this research recipe. Its use of CTAugment does not establish that a vendor product uses the trained classifier.

**Architecture diagram description:** The reference uses **WRN-28-2**, with approximately 1.5M parameters. `Weak image -> shared CNN -> distribution-aligned target; several strong images and mixed images -> shared CNN -> class losses`; a small rotation head branches from the learned features. All image paths share the main CNN.

**Activation functions used and why:** Leaky ReLU preserves negative-side gradients in the residual network, and batch normalization controls feature values. Softmax produces class and rotation probabilities. Temperature sharpening acts on targets, not in place of a hidden-layer activation.

**Loss function(s):** Use cross-entropy for mixed labeled and unlabeled targets, extra agreement on unmixed strong views, and rotation-classification cross-entropy. Unlike MixMatch, the unlabeled loss is not squared probability error. Reproduction requires the stated loss weights and class-alignment normalization.

**Optimization algorithm(s):** The paper uses Adam with fixed learning rate 0.002. Evaluation uses averaged weights with decay 0.999. The reported coefficients belong to this implementation. As with MixMatch, a weight-decay option is meaningful only alongside its update rule.

**Regularization techniques:** Class-frequency alignment, weak-to-strong targets, CTAugment, MixUp, sharpening, weight decay, and rotation prediction. The reference sharpening temperature is 0.5. Its Beta interpolation parameter is 0.75; this parameter controls the distribution used to draw mixing amounts.

**Backpropagation considerations:** Guessed targets and running class frequencies are stored information used to build targets. Gradients update the classifier and rotation head, not the discrete image-change choices. A wrong alignment ratio can distort targets for every example of a class.

**Parameter count / scaling behavior:** The main network still has approximately **1.5M parameters**, plus a small auxiliary head and averaged-weight state. Most extra work comes from additional views, not a much larger classifier.

**Training paradigm:** Train semi-supervised classification together with a self-supervised rotation task. The rotation target comes from the applied change. Its presence does not make the whole recipe unsupervised.

**Hardware/parallelism considerations:** Multiple views need more stored activations and faster image processing. With several devices, running class frequencies should reflect the intended global data mix, not an accidental local mix.

### 2.5.3 FixMatch

**In plain English:** Accept a guess from a gently changed image only when confidence is high enough. Then use that class as the answer for a strongly changed version of the same image.

**Name:** FixMatch.

**Category & sub-category:** Semi-supervised learning; confidence-filtered guesses that link weak and strong image changes.

**Originating paper/vendor/year:** Kihyuk Sohn and colleagues, [*FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*, NeurIPS 2020](https://arxiv.org/pdf/2001.07685v2). The cited full paper is **arXiv v2, dated 2020-11-25**, not a claim about a current model release.

**Core mechanism:** Predict on a weakly changed unlabeled image. If its largest probability meets the threshold, choose that class as a hard target for a strongly changed version. A low-confidence example contributes no unlabeled loss at that step. Unlike MixMatch, the basic recipe needs neither averaging of soft targets nor MixUp. Unlike Mean Teacher, it needs no separately averaged teacher to make targets.

**Inputs/outputs and typical data types:** Use labeled images, unlabeled images, weak and strong image changes, and a confidence threshold. The result is a classifier for new images. The unlabeled pool must fit the target classes, and the changes must keep their meaning for the task.

**Strengths and limitations:** The rule is compact and easy to test by removing or changing a component. It may accept few unlabeled examples early and more as confidence grows. But a fixed cutoff can ignore difficult classes or accept confidently wrong out-of-distribution images. Results may also depend strongly on which few labeled images start training.

**Computational complexity / scalability notes:** Each step processes the labeled batch and two views per unlabeled image. Larger unlabeled batches mean more image work even though checking a threshold is cheap.

**Optional math:** With labeled batch size $`B`$ and unlabeled-to-labeled ratio $`\mu`$, the step uses $`B`$ labeled views and about $`2\mu B`$ unlabeled views.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** On CIFAR-10 with **250 labels**, original-paper Table 2 reports **5.07% error with standard deviation 0.65 percentage points for FixMatch (RA)** over five folds. This equals **94.93% accuracy**. RA means the RandAugment variant. The separately reported CTAugment variant has a different standard deviation.

A weak image change produces a candidate class. Only a confident enough guess teaches its strongly changed counterpart. A trained WRN-28-2 then chooses the object category. This suits images whose changes preserve labels and whose small labeled seed can start useful predictions. The number is not a production success rate or a controlled comparison with later TorchSSL versions. No business KPI is reported. [Full paper v2, Table 2 and Section 4.1](https://arxiv.org/pdf/2001.07685v2).

**Notable vendor implementations/libraries:** Google's [FixMatch repository](https://github.com/google-research/fixmatch) includes the [training recipe](https://github.com/google-research/fixmatch/blob/master/fixmatch.py). Ports to other frameworks can change normalization, image changes, and checkpoint averaging.

**Architecture diagram description:** The example uses **WRN-28-2**. `Weak unlabeled view -> CNN -> detached argmax and confidence mask; strong view -> same CNN -> masked cross-entropy`. Argmax selects the highest-scoring class; "detached" means it is fixed during the update. A separate labeled branch supplies ordinary supervised learning.

**Activation functions used and why:** The cited residual implementation uses leaky ReLU with slope 0.1, batch normalization, and softmax. These keep negative-side gradients, control feature values, and supply class probabilities for the cutoff. Choosing one class by argmax builds a target; it is not a differentiable hidden activation.

**Loss function(s):** Add labeled cross-entropy to cross-entropy on accepted strong-view pseudo-labels. Average the unlabeled term over the **whole unlabeled batch**, not just accepted examples. Otherwise, especially early in training, the effective unlabeled weight changes.

**Optional math:** The unlabeled term is:

$$
\frac{\lambda_u}{\mu B}\sum_b
\mathbf1[\max q_b\ge\tau]\,
\operatorname{CE}(\arg\max q_b,p_\theta(\cdot\mid a_s(u_b))).
$$

Here $`B`$ is labeled batch size, $`\mu B`$ is unlabeled batch size, and $`b`$ indexes that batch. $`u_b`$ is an unlabeled image, $`q_b`$ its weak-view probabilities, and $`a_s`$ the strong image change. $`p_\theta`$ is the classifier. $`\tau`$ is the cutoff, $`\mathbf1`$ is one when the cutoff is met and zero otherwise, and $`\lambda_u`$ weights the loss. $`\operatorname{CE}`$ is cross-entropy; $`\arg\max`$ chooses the target class.

**Optimization algorithm(s):** The reference uses SGD with Nesterov momentum 0.9, a momentum rule with a look-ahead adjustment. The initial learning rate is 0.03 and follows a cosine curve. The standard small-image run uses $`2^{20}`$ updates.

**Optional math:** The rate at update $`s`$ is $`\eta_s=\eta_0\cos(7\pi s/(16S))`$. Here $`\eta_0`$ is the initial rate, $`S`$ the configured total update budget, and $`\pi`$ the circle constant. This states the exact curve, not a generic "cosine schedule."

**Regularization techniques:** Weak changes, strong RandAugment or separately configured CTAugment policies, weight decay, and confidence filtering. Typical reference settings are threshold 0.95, unlabeled-to-labeled batch ratio 7, and unlabeled weight 1. These are choices, not universally best constants.

**Backpropagation considerations:** Hold pseudo-labels and accept/reject decisions fixed for gradients. Keep weak and strong views paired. Batch normalization links examples through shared statistics, so changing batch interleaving can change results even if the written loss stays the same.

**Parameter count / scaling behavior:** WRN-28-2 has approximately **1.5M learned parameters**. A threshold adds none. Optional model-weight averaging for evaluation adds state, but does not turn it into a Mean Teacher target generator.

**Training paradigm:** Train jointly on true labels and pseudo-labels made during training. No separate generator, graph solver, or pretraining stage is required.

**Hardware/parallelism considerations:** Training batches can be split across GPUs or TPUs. Strong-image processing, the unlabeled ratio, and synchronized normalization can limit speed. Changing device count can change training unless those details are controlled.

### 2.5.4 FlexMatch

**In plain English:** Give each class its own cutoff for accepting guessed labels. Lower the cutoff for classes the model appears to be learning more slowly, so they are not left out.

**Name:** FlexMatch, which adds Curriculum Pseudo Labeling (CPL) to FixMatch.

**Category & sub-category:** Semi-supervised learning; adapting confidence cutoffs separately for each class.

**Originating paper/vendor/year:** Bowen Zhang and colleagues, [*FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling*, NeurIPS 2021](https://arxiv.org/html/2110.08263v3).

**Core mechanism:** Count how many unlabeled examples are confidently assigned to each class above a base cutoff. Compare these counts to estimate relative learning progress. Lower the admission cutoff for classes with lower estimated progress. CPL also includes a threshold warm-up and can reshape the progress-to-threshold relationship with a nonlinear rule. Crucially, the count is **not measured class accuracy**. Using it as progress depends on assumptions about class balance and confidence.

**Optional math:** The paper calls normalized progress for class $`c`$ at step $`t`$ $`\beta_t(c)`$. It uses this estimate to set a class-specific admission threshold $`\tau_t(c)`$. The admission threshold and the base threshold used to count progress are distinct.

**Inputs/outputs and typical data types:** Use FixMatch-style labeled and unlabeled images, class counts, and a saved record of sufficiently confident assignments for individual examples. The output remains a normal classifier. Thresholds and assignment records are only needed during training.

**Strengths and limitations:** Class-specific cutoffs can admit useful examples that one fixed cutoff misses. Estimating progress does not require repeatedly predicting on a validation set. But few confident guesses could mean a rare class, ambiguous images, changed data, or poor learning. Counts cannot tell these causes apart. The original experiments do not improve every dataset uniformly.

**Computational complexity / scalability notes:** The method reuses weak-view predictions, so it needs no extra backbone forward or backward pass. It does need per-example assignment storage and class counts. Updating those records efficiently avoids rescoring the entire unlabeled pool at every step.

**Optional math:** With $`n_u`$ unlabeled examples, saved assignments take about $`O(n_u)`$ extra space, plus class-count storage.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The CIFAR-10 experiment uses **40 labels, four per class**. WRN-28-2 predicts weak-view classes. CPL lowers cutoffs for classes with fewer confident assignments, and accepted targets train strong views. The final classifier chooses an object category.

Table 1 reports **4.97% error with standard deviation 0.06 percentage points**, versus **7.47% and 0.28** for the same study's FixMatch implementation. These are **best-checkpoint results over three runs**. The paper separately gives statistics for a window of last checkpoints. Do not treat these as the original FixMatch score or as future-use estimates selected without test feedback. The method fits uneven learning across classes when labels are scarce. No business KPI is reported. [Methods, evaluation rule, and Table 1](https://arxiv.org/html/2110.08263v3).

**Notable vendor implementations/libraries:** The authors' [TorchSSL framework](https://github.com/TorchSSL/TorchSSL) supplies the comparison implementations. They are research baselines, not evidence of a vendor choosing this method for production.

**Architecture diagram description:** The CIFAR-10 example uses **WRN-28-2**, with FixMatch's weak, strong, and labeled branches. Its extra control path is `Weak-view probabilities -> cached confidence counts -> class threshold -> strong-view loss mask`. The mask decides which losses count, without adding neural layers.

**Activation functions used and why:** TorchSSL's reference WRN uses leaky ReLU with slope 0.1, batch normalization, and softmax. They preserve negative-side gradients, control feature values, and give probabilities for counting progress. Changing a cutoff affects example selection, not activations within the residual units.

**Loss function(s):** Add supervised cross-entropy to masked cross-entropy on hard pseudo-labels. Replace FixMatch's single cutoff with the class-specific admission threshold based on progress. Do not confuse that lower admission cutoff with the base cutoff used to estimate progress.

**Optimization algorithm(s):** The source uses SGD with momentum 0.9 and initial learning rate 0.03. It runs $`2^{20}`$ updates with the stated cosine schedule. Evaluation uses averaged model weights with decay 0.999. These settings train the backbone; they are not learned curriculum parameters.

**Optional math:** The learning rate is $`\eta_0\cos(7\pi s/(16S))`$. Here $`\eta_0`$ is the initial rate, $`s`$ the current step, $`S=2^{20}`$ the update budget, and $`\pi`$ the circle constant.

**Regularization techniques:** RandAugment, weight decay, weak-to-strong agreement, threshold warm-up, and per-class selection. The usual base cutoff is 0.95. Cutoffs need not rise steadily; changing assignments can make them fall too.

**Backpropagation considerations:** Do not differentiate the counts or admission choices. Gradients update the classifier through selected losses. Counting all current pseudo-labels instead of the saved sufficiently confident assignments changes the curriculum.

**Parameter count / scaling behavior:** WRN-28-2 has approximately **1.5M learned parameters**; CPL adds no learned layers. Its per-example storage differs from FreeMatch's smaller, class-level running statistics.

**Training paradigm:** Learn from labels and online guesses with a confidence-based curriculum. It is not a separate curriculum trained from held-out class accuracies.

**Hardware/parallelism considerations:** Accelerator work is similar to FixMatch, with extra bookkeeping. Keep example identities and class statistics consistent across devices. Letting each worker develop an independent curriculum gives a different procedure.

### 2.5.5 FreeMatch

**In plain English:** Let confidence cutoffs change as the model learns, both overall and by class. Also discourage the model from using only a narrow set of classes across its predictions.

**Name:** FreeMatch.

**Category & sub-category:** Semi-supervised learning; adapting overall and per-class cutoffs while encouraging varied class use.

**Originating paper/vendor/year:** Yidong Wang and colleagues, [*FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning*, ICLR 2023; arXiv submission 2022](https://arxiv.org/html/2205.07246v3).

**Core mechanism:** Keep an exponential moving average of the highest predicted probabilities on unlabeled examples. It weights recent predictions more and estimates overall confidence. Separately average each class's predicted probability. Use the overall value as a starting cutoff, then lower it for classes receiving smaller average probabilities. These statistics start with equal-class values and change during training.

An added **self-adaptive fairness** term encourages a spread of class use across the predictions. It uses both probability averages and histograms, or counts, of chosen classes. This is a batch-level diversity rule. It does not assert that every real population has equal class frequencies.

**Optional math:** For class $`c`$ at step $`t`$, the cutoff is:

$$
\tau_t(c)=\tau_t^{\rm global}
\frac{\bar p_t(c)}{\max_j\bar p_t(j)}.
$$

Here $`\tau_t^{\rm global}`$ is the running overall-confidence cutoff, $`\bar p_t(c)`$ the running probability average for class $`c`$, and $`j`$ ranges over classes. The denominator is the largest such class average.

**Inputs/outputs and typical data types:** Use labeled and unlabeled images, weak and strong views, and running confidence, probability, and chosen-class statistics. Training produces a classifier and changing cutoffs used to accept pseudo-labels.

**Strengths and limitations:** There is no need to fix one admission confidence for the entire run. This can help with extremely few labels. But "FreeMatch" does not mean free of settings: averaging rates, loss weights, image changes, and optimizer settings remain. Biased confidence, missing classes, and out-of-distribution images can spoil both the cutoffs and the fairness statistics. **Class fairness here does not guarantee demographic fairness.**

**Computational complexity / scalability notes:** Network work is similar to other weak/strong-view pseudo-labeling. Extra statistics grow with class count, not with a saved history for every image. Computing and sharing these averages is generally small beside CNN training.

**Optional math:** Running statistics take $`O(C)`$ space for $`C`$ classes, plus ordinary batch probabilities.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The CIFAR-10 experiment uses **ten labels, one per class**. Weak-view predictions update overall and class-specific cutoffs. Accepted targets train strong views, while the fairness term discourages using too few classes across predictions. The final model chooses an object category.

Table 1 reports **8.07% error with standard deviation 4.24 percentage points**, compared with **13.85% and 12.04** for its FlexMatch comparison. That large run-to-run variation matters. Results use three random seeds and the paper's **best-error-over-checkpoints reporting rule**. They are not from a deployment-style protocol that locks model choice before checking test performance. The approach fits the problem of an unreliable fixed confidence cutoff early in training. It does **not** show that one label per class will reliably work in a new domain. No business KPI is reported. [Setup and Table 1](https://arxiv.org/html/2205.07246v3).

**Notable vendor implementations/libraries:** The paper links the [TorchSSL ecosystem](https://github.com/TorchSSL/TorchSSL). [Microsoft's Semi-supervised-learning/SemiLearn repository](https://github.com/microsoft/Semi-supervised-learning) includes modern recipe implementations. Repository ownership does not identify a deployed product that uses them.

**Architecture diagram description:** The CIFAR-10 example uses **WRN-28-2**: `weak view -> CNN -> running confidence and class statistics -> adaptive mask; strong view -> shared CNN -> class and marginal-diversity losses`. "Marginal" refers to class use averaged across examples, not an extra image feature.

**Activation functions used and why:** Leaky ReLU keeps negative-side feature gradients, and batch normalization controls the reference WRN's feature values. Softmax produces probabilities used both for classification and for running averages of class use.

**Loss function(s):** Combine labeled loss, adaptively filtered pseudo-label cross-entropy, and a particular class-diversity loss. The last term corrects average probabilities using counts of chosen classes. It is not ordinary positive cross-entropy toward fixed equal class frequencies. It is also not entropy minimization for each example.

**Optional math:** The total is $`\mathcal L_s+w_u\mathcal L_u+w_f\mathcal L_f`$, where $`\mathcal L_s`$ is labeled cross-entropy, $`\mathcal L_u`$ the masked pseudo-label loss, and $`w_u,w_f`$ their stated weighting choices for unlabeled and fairness terms. Equation 11 defines $`\mathcal L_f=-\operatorname{CE}(a,b)`$. Here $`\operatorname{CE}`$ is cross-entropy; $`a`$ comes from running class probabilities and $`b`$ from accepted-batch class probabilities. Each is divided by its corresponding hard-prediction histogram, then normalized. The negative sign belongs to the source's histogram-corrected diversity proxy; replacing it with ordinary positive cross-entropy changes the objective.

**Optimization algorithm(s):** The source uses SGD with momentum 0.9, initial learning rate 0.03, and $`2^{20}`$ updates on the stated cosine curve. Model evaluation uses averaged weights with decay 0.999. Averaging adaptation statistics and averaging model weights are separate operations.

**Optional math:** The learning rate is $`\eta_0\cos(7\pi s/(16S))`$. Here $`\eta_0`$ is the initial rate, $`s`$ is the step, $`S=2^{20}`$ the total update budget, and $`\pi`$ the circle constant.

**Regularization techniques:** Strong image changes, weight decay, adaptive pseudo-label selection, and the self-adaptive fairness term. Its weight and the amount of statistic smoothing still need validation despite the method's name.

**Backpropagation considerations:** Treat running estimates and hard accept/reject choices as extra training state. Differentiate the current probability losses. Guard histogram divisions when a class count is zero. Otherwise, exactly the low-label cases of interest can produce undefined calculations.

**Parameter count / scaling behavior:** WRN-28-2 has approximately **1.5M learned parameters**. Extra adaptation storage is class-sized beyond normal training buffers. It needs no new expert network or learned head for predicting thresholds.

**Training paradigm:** Learn semi-supervised classification jointly while updating confidence and class statistics. There is no separately validation-driven curriculum.

**Hardware/parallelism considerations:** Image work is similar to FixMatch, plus small class-statistic calculations. If devices see different class mixtures, combine the intended statistics. Otherwise, they silently develop different thresholds.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| MixMatch | Images suited to gentle changes and blending | Uses soft guesses and teaches behavior between examples | Several target-building choices can go wrong together | CIFAR-10 with 250 labels |
| ReMixMatch | Images with reliable weak targets and useful strong changes | Adjusts class use and trains across richer changes | Wrong expected class frequencies; many extra views | CIFAR-10 with 250 labels |
| FixMatch | Images whose weak and strong changes preserve labels | Accepts confident guesses with a simple rule | One fixed cutoff can exclude useful classes | CIFAR-10 with 250 labels |
| FlexMatch | Images from known classes that are learned at different rates | Gives each class its own admission cutoff | Few confident guesses may mean rarity, not difficulty | CIFAR-10 with 40 labels |
| FreeMatch | Known-class image tasks with very few labels | Adapts confidence cutoffs overall and by class | Running statistics can be biased; low-label results vary greatly | CIFAR-10 with ten labels |

## 2.6 Generative and reconstruction approaches

These methods learn from unlabeled inputs by trying to explain or rebuild them. A **generative model** learns a way to produce inputs. A **reconstruction model** tries to recover an input or its internal features. Either task may help a classifier learn useful patterns. But realistic-looking images, accurate reconstructions, high model likelihood, and correct class predictions measure different things. **Likelihood** describes how well a probability model accounts for the observed data. None of these scores automatically stands in for the others.

### 2.6.1 Semi-supervised variational autoencoders: M1 and M2

**In plain English:** Learn a compact description of many unlabeled images, then use a few known answers to learn classes. M1 learns features before classification; M2 learns classes and image generation together; M1+M2 combines those stages.

**Name:** Semi-supervised deep generative models M1, M2, and the stacked M1+M2 system. These are three related but different setups.

**Category & sub-category:** Semi-supervised learning; learning hidden input features and probability models that can generate inputs.

**Originating paper/vendor/year:** Diederik Kingma, Shakir Mohamed, Danilo Rezende, and Max Welling, [*Semi-supervised Learning with Deep Generative Models*, NIPS 2014](https://arxiv.org/html/1406.5298v2).

**Core mechanism:** **M1** trains a variational autoencoder (VAE) on all inputs. Its encoder turns an input into a probability-based short description, called a **latent representation**. A decoder tries to generate the input from it. A separate classifier then learns from these features; the paper tests a TSVM in that role.

**M2** adds a class variable to the generative model. One network guesses the class; another estimates hidden information given the input and a class; a decoder generates inputs from class and hidden information. Known labels specify the class during labeled learning. For unlabeled inputs, training considers every possible class and averages their contributions using guessed class probabilities. **M1+M2** first learns M1 features, then trains M2 on those features. Their benchmark scores must stay separate.

**Optional math:** M2 models $`p_\theta(x,y,z)=p(y)p(z)p_\theta(x\mid y,z)`$. Here $`x`$ is an input, $`y`$ a class, $`z`$ hidden information, and $`\theta`$ generative-model weights. $`p(y),p(z)`$ are prior distributions, and $`p_\theta(x\mid y,z)`$ describes generating an input. Inference networks with weights $`\phi`$ estimate $`q_\phi(y\mid x)`$, the class probabilities, and $`q_\phi(z\mid x,y)`$, the hidden-information distribution.

**Inputs/outputs and typical data types:** Use labeled and unlabeled vectors or images. Depending on the setup, outputs include latent features, inferred class probabilities, and a class-conditioned generator. M1's VAE stage itself uses no class labels. Its later classifier makes the complete workflow supervised or semi-supervised.

**Strengths and limitations:** The model explicitly connects classes to how inputs are generated. It may help separate a digit's identity from its writing style. But describing common input details well does not guarantee learning the details that distinguish classes. Approximate inference and a poorly chosen input probability model can hurt. So can **latent collapse**, where the generator ignores the hidden code. Considering every class also becomes expensive when there are many.

**Computational complexity / scalability notes:** A VAE trains both encoder and decoder. For a fixed number of random latent samples, their forward and backward passes set the batch cost. M2 additionally evaluates class-conditioned paths for each possible class on unlabeled examples. Doubling class count can therefore roughly double that part, unlike an ordinary single-output-head classifier.

**Optional math:** The exact unlabeled-class sum is approximately $`O(BCF)`$ plus classifier work. Here $`B`$ is batch size, $`C`$ class count, and $`F`$ now means one class-conditioned inference/generation evaluation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The MNIST study uses **100 labeled training examples** in the permutation-invariant task. This task treats each image as a flat pixel list, without building neighboring-pixel structure into the model. M1 first produces hidden features. M2 uses them to infer digit class and style, and its inferred class probabilities select the digit.

Table 1 reports **3.33% error with standard deviation 0.14 percentage points for M1+M2**. In contrast, **M2 alone reports 11.97% with standard deviation 1.71**. Do not assign the combined result to M2 alone. The approach fits the idea that many unlabeled images can teach useful variation before scarce labels name the classes. It is research, not evidence of a deployed document-processing system, and no business KPI is reported. [Original model definitions and MNIST table](https://arxiv.org/html/1406.5298v2).

**Notable vendor implementations/libraries:** The authors provide [NIPS 2014 semi-supervised code](https://github.com/dpkingma/nips14-ssl). Modern tensor and probability-model libraries can express these calculations. A generic VAE example does not automatically include M2's average over unknown classes or its extra classification loss.

**Architecture diagram description:** M1 uses `input -> two 600-unit hidden layers -> mean/log-variance of 50-dimensional z`, with decoder `z -> 600 -> 600 -> input likelihood parameters`. Its output describes the center and spread of the hidden-code distribution. M2 also uses 50-dimensional `z`, with one 500-unit hidden layer in each component MLP: `x -> q(y|x)`; `(x,y) -> q(z|x,y)`; `(y,z) -> p(x|y,z)`. These are dense networks. The three paths predict class, infer hidden information, and generate an input, respectively.

**Activation functions used and why:** Softplus gives smooth hidden-layer responses, and softmax normalizes class probabilities. The MNIST Bernoulli model treats each pixel as a binary outcome; sigmoid keeps its probability between zero and one. Each Gaussian hidden coordinate has a bell-shaped distribution with separate mean and variance outputs. Variance must stay positive. This is different from choosing a class activation.

**Loss function(s):** M1 balances reconstruction likelihood with a KL penalty that keeps hidden-code distributions near their prior, a chosen starting distribution. It then trains a separate classifier. For M2, the objective combines labeled and unlabeled generative bounds with extra cross-entropy on known class labels. A **variational bound**, or ELBO, is a computable lower estimate of the data's log probability. Training maximizes that bound, or equivalently minimizes its negative.

**Optional math:** Let $`\operatorname{ELBO}_l(x,y)`$ be the labeled bound for input $`x`$ and class $`y`$. With inferred class probabilities $`q_\phi(y\mid x)`$, M2's unlabeled bound is:

$$
\operatorname{ELBO}_u(x)=
\sum_y q_\phi(y\mid x)\operatorname{ELBO}_l(x,y)
+H(q_\phi(y\mid x)).
$$

Here $`\phi`$ denotes inference-network weights and $`H`$ is entropy. The sum averages each possible class's bound; the entropy term accounts for uncertainty over the unknown class. Minimize negative labeled/unlabeled bounds plus weighted labeled cross-entropy for $`q_\phi(y\mid x)`$. The **positive entropy in this ELBO** is not the same as a positive entropy penalty in a minimized loss. Negating the bound changes that sign. M1's negative bound is expected negative reconstruction log-likelihood plus latent KL.

**Optimization algorithm(s):** Training uses gradient estimates with random latent samples. **The source is inconsistent about its optimizer.** Section 3.2 says experimental results used AdaGrad. The detailed implementation discussion instead describes a momentum/bias-corrected RMSProp variant with **constant learning rate 0.0003**. We retain both statements rather than invent one unambiguous Adam recipe.

**Regularization techniques:** Hidden-code KL penalties, prior distributions, random latent sampling, and M2's reported parameter prior/weight penalty. For MNIST, normalized pixel intensities also provide probabilities for sampling binary inputs. Do not assume batch normalization or dropout merely because later VAE code uses them.

**Backpropagation considerations:** Make a random hidden sample by scaling and shifting fixed noise. This lets gradients pass through its learned mean and scale with less sampling noise. With M2's small class set, explicitly consider all classes and differentiate their weighted sum instead of randomly choosing an unknown label.

**Optional math:** The reparameterization is $`z=\mu+\sigma\odot\epsilon`$. Here $`z`$ is a latent sample, $`\mu`$ the learned mean, $`\sigma`$ the learned standard-deviation vector, $`\epsilon`$ standard Gaussian noise, and $`\odot`$ element-by-element multiplication. Gradients follow this explicit path through the sample. For the class sum, listing every class avoids an unnecessary score-function estimator, a noisier way to estimate gradients by sampling labels.

**Parameter count / scaling behavior:** M1, raw-input M2, and stacked M1+M2 do not share one parameter count. A Gaussian head needs separate mean and variance outputs. Stacking changes the input width of M2, so simply adding counts for two raw-input models is misleading.

**Optional math:** A dense layer with $`a`$ input units and $`b`$ output units has $`(a+1)b`$ parameters: $`ab`$ connection weights and $`b`$ biases. A diagonal Gaussian head provides both mean and variance values for each latent dimension.

**Training paradigm:** Choose the stated setup: M1 feature pretraining then a classifier; joint class-and-generation learning in M2; or staged M1+M2. If the later classifier is a TSVM, check again whether evaluation sees the target inputs during training.

**Hardware/parallelism considerations:** GPUs can train these MLPs efficiently. Class-conditioned branches can be processed together, but memory and computing grow with class count. Approximations for very large class sets are beyond this entry.

### 2.6.2 Semi-supervised GAN with a K+1 classifier

**In plain English:** Teach a classifier both the known classes and the difference between real and generated examples. A generator supplies extra training examples, while a few true labels teach the real class names.

**Name:** Semi-supervised GAN with K+1 categories. Here K equals the number C of real classes; the extra category is generated, or fake, data.

**Category & sub-category:** Semi-supervised learning; a generator and a real/fake classifier also learn features for labeled classification.

**Originating paper/vendor/year:** The representative method is Tim Salimans and colleagues' [*Improved Techniques for Training GANs*, NeurIPS 2016](https://arxiv.org/html/1606.03498v1). A related independent proposal is Augustus Odena's [*Semi-Supervised Learning with Generative Adversarial Networks*, 2016](https://arxiv.org/abs/1606.01583).

**Core mechanism:** A GAN has a **generator**, which makes examples, and a **discriminator**, which judges them. Here the discriminator learns known real classes from labeled examples. It also learns that unlabeled real inputs belong to some real class and generated inputs belong to the fake class.

The successful reference setup trains the generator with **feature matching**: make its outputs produce the same average intermediate discriminator features as real data. This shapes the classifier's learning without requiring the most realistic-looking images. The paper finds that a technique improving image appearance need not produce the best semi-supervised classifier.

**Inputs/outputs and typical data types:** Use labeled real images, unlabeled real images, and random generator inputs. Outputs include real-class predictions and generated samples. The fake category is learned against this generator's outputs. It is not a generally validated detector of unknown classes or out-of-distribution inputs.

**Strengths and limitations:** Generated examples can help learn useful features without hand-designing every harmless input change. The system also produces samples. But alternating two learning processes is harder to stabilize than one classification loss. The generator may make too little variety, called **mode collapse**. An overly strong discriminator or artificial image artifacts can also spoil the learning signal.

**Computational complexity / scalability notes:** Each training step runs the discriminator on real and generated inputs, runs the generator, and performs separate updates. Cost depends on both networks and how often each is updated. Ordinary class prediction can discard the generator, so inference costs less than the whole training setup.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The permutation-invariant MNIST task uses flat digit vectors and **100 labeled examples**. The discriminator receives real digit vectors, and the generator makes synthetic vectors. Real/fake learning shapes features; labeled cross-entropy teaches digit names. The highest real-class score gives the recognition decision.

Original Table 1 reports **93 incorrectly classified test images on average, standard deviation 6.5, out of the 10,000-image test set**, averaged over ten seeds. This is the single-model row, not the separate ten-model ensemble row. Feature matching fits the goal of learning useful class features, even when generated samples are not the most realistic. No production OCR performance or business KPI is reported. [MNIST experiment and feature-matching discussion](https://arxiv.org/html/1606.03498v1).

**Notable vendor implementations/libraries:** OpenAI's historical [improved-gan repository](https://github.com/openai/improved-gan) contains the research code. The neural details here describe its public [MNIST feature-matching script](https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_mnist_feature_matching.py), not an undocumented commercial checkpoint.

**Architecture diagram description:** The inspected script uses discriminator `784 -> 1000 -> 500 -> 250 -> 250 -> 250 -> 10 real logits`. A logit is a raw class score. The fake logit is implicitly fixed to zero, yielding an equivalent K+1-class distribution without a separately learned fake output. The generator is `100-dimensional noise -> 500 -> 500 -> 784`. These are script settings, not a claim that every paper experiment used exactly this generator.

**Activation functions used and why:** ReLU gives the discriminator nonlinear features by zeroing negative responses. Softplus gives the generator smooth hidden responses, and sigmoid bounds output image intensities. Normalized real-class logits select the digit. A log-sum-exp calculation safely normalizes scores including the implicit fake category.

**Loss function(s):** Use labeled cross-entropy over real classes, real-versus-fake losses on real and generated examples, and generator feature matching. A generic GAN minimax generator loss is not a substitute for the feature-matching setup behind this example.

**Optional math:** The feature-matching loss is $`\lVert\mathbb E_x h(x)-\mathbb E_z h(G(z))\rVert_2^2`$. Here $`x`$ is a real input, $`z`$ random noise, $`G`$ the generator, and $`h`$ the discriminator's intermediate features. Each $`\mathbb E`$ averages over the relevant inputs. The squared norm measures the difference between real and generated average features.

**Optimization algorithm(s):** The public MNIST script alternates Adam-family updates. It uses learning rate **0.003**, with no explicit decay schedule in that script, and first-moment coefficient 0.5 for smoothing gradients. These are inspected code settings, not a universal GAN recipe.

**Regularization techniques:** Weight normalization and Gaussian noise in the discriminator, normalization in the generator, and feature matching. Do not assume every technique in *Improved Techniques* was used together for its best semi-supervised result. For example, minibatch discrimination, which lets a discriminator use information across examples, is not automatically part of this configuration.

**Backpropagation considerations:** A discriminator update must not update generator weights through generated inputs. During a generator update, gradients pass through the fixed discriminator's features into generator outputs. Hold the real-feature target fixed as appropriate for this alternating objective.

**Parameter count / scaling behavior:** Training stores both networks, while classification uses only the discriminator. Its displayed layers have approximately 1.54M connection weights and biases by arithmetic, plus normalization state. These weights and biases are the affine parameters. The two networks can grow independently.

**Optional math:** If discriminator and generator counts are $`p_D`$ and $`p_G`$, training holds $`p_D+p_G`$ learned parameters, while class inference uses $`p_D`$.

**Training paradigm:** Train labeled classification, real/fake learning from unlabeled data, and the generator together. This is not an unlabeled GAN that is later tested using a separate external classifier.

**Hardware/parallelism considerations:** A GPU supports the reference MLP setup. Larger CNN image versions need more memory and throughput. Across devices, control how feature means are averaged and how often discriminator and generator updates occur.

### 2.6.3 Ladder Networks

**In plain English:** Teach a network to classify the few labeled examples while also cleaning noise from its internal features. Unlabeled examples can then train several layers, not just the final class output.

**Name:** Semi-supervised Ladder Network.

**Category & sub-category:** Semi-supervised learning; learning classes and removing noise from features at each layer together.

**Originating paper/vendor/year:** Antti Rasmus and colleagues, [*Semi-Supervised Learning with Ladder Networks*, NIPS 2015](https://arxiv.org/html/1507.02672v2). It extends the earlier Ladder architecture associated with Harri Valpola.

**Core mechanism:** Pass each input through clean and noise-corrupted encoders that share weights. A decoder works downward from higher layers. At each level, it receives both a top-down signal and a sideways link from that level's noisy features. It tries to reconstruct the matching clean features. Known labels train the noisy encoder's class output; all examples train the denoising losses. Sideways information helps restore local details without requiring the topmost features to retain everything about the input.

**Inputs/outputs and typical data types:** Use labeled and unlabeled vectors or images. Training produces class probabilities and reconstructed features at multiple layers. Ordinary classification keeps only the clean encoder. Unlike M2, this is not the same kind of latent-variable likelihood model.

**Strengths and limitations:** Unlabeled inputs send learning signals throughout the network when true labels are scarce. There is no need to pretrain each layer separately. But the decoder, normalization, noise levels, and layer-specific loss weights require care. Reconstruction can preserve irrelevant detail unless labeled learning guides the features toward the task.

**Computational complexity / scalability notes:** Training runs both clean and noisy encoders plus the denoising decoder. With fixed design and noise, work grows roughly with the number of examples per epoch, but exceeds a simple classifier's work. Saving intermediate values for several paths can dominate memory. Class prediction discards the decoder.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In permutation-invariant MNIST recognition, the dense Ladder model uses **100 labels in the supervised training objective**. It reports **1.06% test error with standard deviation 0.37 percentage points**. Flat pixel vectors enter clean and noisy encoders. The decoder learns to clean intermediate features, and the clean encoder's highest class score chooses the digit.

**The authors used 10,000 labeled validation images for model and hyperparameter development.** Thus this was not a complete development process using only 100 known answers. Final runs used all 60,000 training images, with training labels restricted as stated. Occasional failures affected the average, so the authors increased the 100-label evaluation to forty runs. The method fits the goal of teaching lower layers from unlabeled handwriting, not just producing a better pixel reconstruction score. No deployed OCR system or business KPI is reported. [Section 4.1 and Table 1](https://arxiv.org/html/1507.02672v2).

**Notable vendor implementations/libraries:** The authors provide a [Ladder research implementation](https://github.com/CuriousAI/ladder). A stack of denoising autoencoders without sideways and top-down denoising at each level is not the same design.

**Architecture diagram description:** The encoder is `784 -> 1000 -> 500 -> 250 -> 250 -> 250 -> 10`. Clean and noisy copies share weights. At every level, `top-down decoder signal + lateral corrupted activation -> denoising function -> reconstructed clean activation`. "Lateral" means the sideways connection at the matching layer. Training compares each reconstructed feature set with its clean counterpart.

**Activation functions used and why:** ReLU makes encoder features nonlinear, and softmax gives class probabilities for cross-entropy. Denoising functions use sigmoid and affine parts, which scale and shift values. Top-down signals control estimated means and scales for reconstruction. These are not extra class outputs at every layer.

**Loss function(s):** Combine noisy-encoder labeled cross-entropy with weighted mean-squared denoising errors at several layers. The source normalizes both clean targets and reconstructed quantities in a specific way. Different layers have different weights. Replacing this with arbitrary MSE on raw activations changes the method.

**Optimization algorithm(s):** MNIST runs use Adam with learning rate **0.002 for 100 epochs**, followed by **50 epochs of linear decay to zero**. Minibatch size is 100. All parts train together; separate autoencoder fits are not a required sequence.

**Regularization techniques:** Gaussian corruption, batch normalization, and layer-by-layer denoising. For the reported 100-label setup, the paper gives noise standard deviation 0.3. Cost weights are 1000 at the input level, 10 at the first hidden level, and 0.1 at higher levels.

**Backpropagation considerations:** Shared encoder weights receive both class and denoising gradients. Keep clean and noisy features aligned and normalize variances consistently. Otherwise, the model may reduce the loss merely by changing scales, or gradients may become unstable. Sideways links and layer losses give shorter routes for learning signals than the final classifier alone.

**Parameter count / scaling behavior:** The listed encoder has approximately **1.54M affine parameters**, meaning connection weights and biases, calculated from its widths. Normalization adds parameters too. Training also adds decoder matrices and denoising parameters for individual units. Ordinary prediction needs only the encoder.

**Training paradigm:** Start from random weights and jointly learn semi-supervised classification and self-supervised denoising. Clean features provide reconstruction targets, but the whole method still includes true class labels.

**Hardware/parallelism considerations:** GPUs benefit from batching the paths, but training stores more intermediate state than a classifier alone. Across devices, use consistent batches for normalization and layer losses. Prediction is much simpler than the training graph.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
|---|---|---|---|---|
| Semi-supervised VAE M1 / M2 | Vectors or images with useful hidden structure for generation | Connects class learning with a model of how inputs arise | Generating well may not aid classification; approximations and class sums cost work | MNIST M1+M2 with 100 training labels |
| K+1 semi-supervised GAN | Images where generated examples can help teach features | Links classification to feature learning from generated data | Alternating updates can be unstable; realism is not classification quality | MNIST feature-matching GAN with 100 labels |
| Ladder Network | Images or vectors helped by denoising at several layers | Unlabeled examples teach multiple layers | Decoder and normalization need careful implementation | MNIST with 100 training labels and a separate large validation budget |

## Evidence and reproduction boundaries

All 22 worked examples are **research benchmarks**, not verified production deployments. They address real kinds of recognition, document classification, and scientific classification problems. But the cited evidence supplies no unreported business outcome. "No business KPI reported" does not mean a method has never been used in production. It means this evidence does not establish that result.

The exact metrics retain the checks against linked primary papers or copies of the original papers. Graph results described through curves stay qualitative where no exact number was established. Counts marked as arithmetic come from the stated layer sizes. They do not claim that a vendor released a checkpoint with exactly that count. These checks are not independent reruns of the experiments.

Take care when reproducing older code. Repository branches can change, older framework dependencies may be unavailable, and public scripts may not identify every setting behind a paper's score. Record a code commit and the full configuration before claiming numerical reproduction. Keep the known limitations visible: conflicting M1/M2 optimizer descriptions, GAN script-versus-paper architecture details, labels used for validation, private JFT images, and best-checkpoint reporting. Do not fill these gaps with guesses.

## Coverage and continuation manifest

- **2.1.1-2.1.4:** Self-training/Pseudo-Label, co-training, tri-training, and Noisy Student.
- **2.2.1-2.2.2:** Transductive/semi-supervised SVM and Laplacian SVM/manifold regularization.
- **2.3.1-2.3.2:** Harmonic label propagation with fixed known labels and normalized label spreading with soft label retention.
- **2.4.1-2.4.6:** Entropy minimization, Pi Model, temporal ensembling, Mean Teacher, VAT, and UDA.
- **2.5.1-2.5.5:** MixMatch, ReMixMatch, FixMatch, FlexMatch, and FreeMatch.
- **2.6.1-2.6.3:** Semi-supervised VAE M1/M2, K+1 semi-supervised GAN, and Ladder Networks.
- **Total:** 22 scoped entries and six category comparison tables. The 16 neural deep dives explain specific reference implementations, not a required architecture for every use of a method name.
- **Related volumes:** [Supervised classical methods](01-supervised-classical.md), [supervised neural architectures](02-supervised-neural.md), [unsupervised classical methods](04-unsupervised-classical.md), [unsupervised neural and self-supervised learning](05-unsupervised-neural.md), [foundation models and their training stages](06-foundation-models.md), [cross-cutting comparisons](09-comparative-guide.md), and [glossary](10-glossary.md).
- **Further non-required depth not included:** predicting numerical values or structured outputs with semi-supervised methods; formal proofs about generalization or graph convergence; theory for missing views or labels whose absence depends on the data; full tests for unknown classes or highly unequal class sizes beyond the failure modes discussed here; learning across separate data holders, called federated SSL; managing caches or teachers in continuous streams; policies for choosing new examples to label; and complete searches for image-change policies. Networks that learn graph representations differ from the two fixed-graph methods here. Read about them in the related neural volumes.
