# Preface: How to Read an Algorithm Catalog

Machine learning means teaching a computer to find useful patterns in data.
Instead of writing a rule for every situation, we give it examples. A learning
method uses those examples to build a model.

You do not need a college course to start this book. Basic algebra helps, but
the main ideas come first. Equations are marked as optional. On a first read,
focus on what a method takes in, what it does, and what it gives back.

Four words will help you keep the pieces straight:

| Word | Plain meaning | Example |
|---|---|---|
| Algorithm | A set of steps for solving a problem or learning from data | Steps that group similar records |
| Architecture | The way a model's parts connect | The layers and shortcuts in a neural network |
| Trained model | The result after a learning method has adjusted its settings | A network whose learned numbers help it recognize digits |
| Product | A complete application that may use models, search, rules, and people | A chatbot that also searches documents |

These are related, but they are not the same thing. A Transformer is a network
design, not a complete chatbot. Downloading a model's learned numbers does not
mean its training data are also available. A good lab test does not prove that
the model works well in a live business.

The book keeps the history, sources, and technical details. It now explains
them in shorter steps, so you can build understanding before tackling the math.

## P.1 The book's organization

The first three parts group methods by the kind of feedback they learn from:

1. **Supervised Learning Algorithms:** learn from examples with known answers,
   such as pictures labeled "cat" or past houses with known sale prices.
2. **Semi-Supervised Learning Algorithms:** learn from a few examples with
   answers and many more examples without them.
3. **Unsupervised Learning Algorithms:** find patterns without the task's
   answer labels. This part also covers self-supervised learning, which creates
   practice questions from the data itself.

Part 4 gives **Mixture of Experts (MoE)** a closer look. An MoE model chooses
which small networks to use for an input. Its individual models also appear
in the learning section that fits their training. The original MoE and GShard
translation models used supplied answers, so they stay in supervised learning.

Use the [complete contents](../CONTENTS.md) to jump to a method. Comparison
tables help you compare methods within a group. Near the end, a
[cross-family guide](09-comparative-guide.md) compares broader choices, and
the [glossary](10-glossary.md) explains key terms.

A chapter's "coverage and continuation manifest" simply says what it covers
and what a later edition could add.

## P.2 What determines a learning category?

**Look at the training feedback, not just the network's shape.** The same kind
of layer can help classify a labeled picture, learn from unlabeled pictures,
or rebuild a damaged picture. The goal and data decide the learning category.

| Learning setting | What the model receives | What it tries to do | Keep in mind |
|---|---|---|---|
| Supervised | Examples and supplied answers | Make predictions close to those answers | Supplied answers can still be wrong or incomplete |
| Semi-supervised | Some answers plus extra examples without answers | Use both sets to improve the same task | The method must actually use the unlabeled examples |
| Unsupervised | Examples without the task's answer labels | Find groups, compact summaries, or patterns | A discovered group may not have a useful real-world meaning |
| Self-supervised | Practice questions made from the data | Guess a hidden word, next word, or missing image part | It still has a target, but people did not label each practice question |
| Weak or naturally paired supervision | Noisy clues, rules, captions, or paired data | Learn from useful but imperfect guidance | A caption supplies information; it is not "no supervision" |
| Reinforcement learning | Actions and rewards from an environment | Learn actions that lead to better rewards | This is a separate learning approach, not a type of clustering |

Large general-purpose models often learn in several stages. **Pretraining**
is the broad first stage. **Fine-tuning** adjusts the model later for a task.
People may also rate answers, or one model may teach another through its
outputs. This last approach is called **distillation**.

The book usually places these models by their base pretraining stage. Each
entry then explains the later stages. A model does not become wholly
unsupervised just because its first stage used self-made questions.

Some cases cross the boundaries. CLIP learns from pictures paired with text.
The original paper calls this *natural language supervision*. The captions
give clues about the pictures, even without a fixed list of class labels.
We discuss CLIP beside other methods that learn reusable features, but do not
pretend its text supplies no guidance.

Text-to-image models have a similar mix. They may learn to remove noise from
an image while a caption tells them what the image should show. The noise
task and the caption provide different kinds of training information.

For **graphs**, the inputs are items called nodes and links called edges.
A graph model may learn from labels on only some nodes. If it can also see
the unlabeled nodes it must later classify, the test is **transductive**.
That means the test inputs, but not their answers, were visible during
learning. A model tested on wholly new graphs faces a different setting.

## P.3 Scope and the meaning of comprehensive

This is a wide first-edition guide, not a list of every method ever published.
It contains 170 entries across 32 sub-categories. Some entries cover a family
of related models and explain the differences. Mentioning several versions
does not turn one family entry into several counted entries.

The book covers prediction, trees, kernels, neural networks, and methods using
partly labeled data. It also covers grouping, smaller data summaries, unusual
record detection, text and image generation, large model families, and MoE.
The names may be new; each entry explains its purpose.

Some areas need their own future volumes. We do not fully catalog learning
through rewards, cause-and-effect methods, recommendation systems, every
weight-update rule, or every scientific network. Related terms still appear
when they help explain a method in this book.

**Evidence cutoff: September 8, 2026.** The simpler wording does not update
the research cutoff. A source may be much older. Model versions keep their
dates, and an older model is not presented as the latest release.

For closed, company-run models, we describe only what public sources support.
If the company has not shared a detail, that gap stays visible.

## P.4 Anatomy of an entry

Every method starts with **In plain English**. Read that first. Then try the
core mechanism and worked example. Return to the other fields when you need
them. The same field names appear throughout the book so details are easy
to find.

Here is what the nine common field names mean:

| Field | Plain meaning |
|---|---|
| Name | What the method is called, including short names |
| Category & sub-category | Which learning group it belongs to |
| Originating paper/vendor/year | Who introduced it, when, and which version we mean |
| Core mechanism | How it works, step by step |
| Inputs/outputs and typical data types | What goes in and what comes out |
| Strengths and limitations | Where it helps and where it can fail |
| Computational complexity / scalability notes | How the work and memory needs grow as the problem gets bigger |
| Real-world problem solved - REQUIRED WORKED EXAMPLE | A named use or research test, with clear limits on what it proves |
| Notable vendor implementations/libraries | Software that provides the method; not proof of who uses it in a business |

Neural-network entries add nine more fields:

| Field | Plain meaning |
|---|---|
| Architecture diagram description | A map of the network's layers and connections |
| Activation functions used and why | Small rules that change signals between layers |
| Loss function(s) | The score the model tries to reduce while learning |
| Optimization algorithm(s) | The rule for adjusting the model's learned numbers |
| Regularization techniques | Controls that help the model avoid learning only training-set quirks |
| Backpropagation considerations | How error information travels backward to guide updates |
| Parameter count / scaling behavior | How many numbers the model learns, and how size changes its needs |
| Training paradigm | The stages used to teach the model |
| Hardware/parallelism considerations | The computers it needs and how they can share the work |

An answer of **not publicly disclosed** means the source does not tell us.
It is not permission to fill the field with another model's design.
Even a widely used chatbot may have an unknown size or training recipe.

## P.5 Evidence policy for worked examples

Each worked example tells you which kind of evidence it uses:

| Label | Meaning | What it does not prove |
|---|---|---|
| **Sourced application** | A named study or organization really used the method | That it caused a business gain, unless the source shows that |
| **Research benchmark** | A named test on specified data | That a hospital or company put it into everyday use |
| **Illustrative (not a claimed deployment)** | A clear teaching example or calculation | That the example happened in the real world |

Some examples include both a real study and a small made-up calculation.
The text tells you which is which. A toy calculation can explain an update;
it cannot prove that a company saved money.

**KPI** means "key performance indicator": a useful business measure, such as
cost or downtime. **Production KPI not reported** means the source gives
no such result. It does not mean the effect was zero or the project failed.

A worked example follows this path:

**problem -> data -> method -> output -> decision -> reported result**.

For example, a code test gives a model a function description. The model
writes code, and the test checks it. A pass rate describes that code test.
It does not tell us how much time a programmer saved at work.

Sometimes we explain why a method is a sensible choice. That is our technical
reasoning unless the source says the organization chose it for that reason.
"This method can find curved patterns" does not mean a hospital publicly
gave that reason for buying it.

Links lead to original papers, model reports, code, or official documentation.
A company's own comparison remains a **vendor-reported comparison**.
The model version, data split, and scoring rules are part of the result,
not small details that can be dropped.

## P.6 Reading numbers without being misled

**Accuracy and error are different.** If error falls from 10% to 8%, it
falls by 2 percentage points. That is a 20% relative reduction in error.
It is not a 20% rise in accuracy.

**Speed, work, memory, and price are also different.** A GPU is a processor
often used for many calculations at once. GPU-hours count how long such
processors were used. FLOPs count arithmetic work. **Latency** is the wait
for an answer. **Throughput** is how much work finishes in a given time.
None of these alone gives the price of running a complete service.

**Read the test rules.** A model allowed eight tries and a majority vote
has a different task from one allowed one answer. Top-5 accuracy accepts
an answer anywhere in five guesses; top-1 accepts only the first.
Changing how answers are extracted can change a score without changing
the model.

**Good comparisons need more than a number.** A model may have seen test
items during training. Labels may be wrong. The tested traffic may differ
from real use. Results from different studies should not be combined into
a single ranking without checking those differences.

**Scaling laws describe measured trends, not promises.** Researchers have
studied how model size, training data, and compute relate to performance.
The [language-model scaling study](https://arxiv.org/abs/2001.08361) and
[compute-optimal training study](https://arxiv.org/abs/2203.15556) are
important examples. Their results apply to their tested conditions.
They do not prove that every new dataset or network will follow the same rule.

## P.7 Mathematical and systems notation

**Optional math:** you can skip the formulas on a first read. These symbols
let authors write a short expression instead of a long sentence. Entries
explain their local meanings.

| Symbol | Usual meaning | Keep in mind |
|---|---|---|
| $`n`$ | Number of training examples | Sometimes it counts tokens instead; the text says when |
| $`d`$ | Number of input details or width of a hidden layer | Check the definition beside the formula |
| $`p`$ | Number of learned settings, called parameters | MoE total and active counts are different |
| $`k`$ | Number of neighbors, groups, or chosen experts | The local meaning takes priority |
| $`T`$ | Sequence length or number of time steps | Not usually the number of passes through training data |
| $`L`$ | Number of layers | A loss may use the different symbol $`\mathcal{L}`$ |
| $`B`$ | Batch size: examples processed together | It may count whole sequences rather than tokens |
| $`E`$ | Training passes or experts | The MoE chapters explain which |
| $`X,y`$ | Input data and target answers | An answer can be a number, class, or sequence |
| $`\theta`$ | The model's learned numbers | This can include the router's learned numbers |
| $`\lambda`$ | The strength of an extra penalty or loss | The same symbol need not mean the same setting in two methods |

**Big-O** describes how work grows, not how many seconds a computer takes.
For example, $`O(n)`$ means work grows roughly in proportion to the number
of examples. With $`O(n^2)`$, doubling that number can require about four
times the work. These are growth patterns under stated assumptions.

Different ways to fit the same kind of model can have different costs.
A method that stores every pair of examples may need far more memory than
one that uses a smaller approximation. The simpler method may also give
a different answer.

In ordinary dense attention, each sequence position can compare with every
other position. That part grows roughly with sequence length squared.
A Transformer also does other work, including moving features through
learned layers. A memory-saving attention method can store less without
removing all those pairwise calculations.

## P.8 From a catalog to a deployed system

Start with the problem, not the newest model name. Decide what a useful
answer looks like and what a wrong answer would cost. Choose a simple
**baseline**: a starting method that a more complex one should beat.

Keep training and test data separate. That rule includes learning how to
scale inputs or fill missing values. If a model learns those steps from
test data, the test is no longer fully fresh. This is called **data leakage**.

Randomly splitting rows is not always enough. Records from the same patient,
machine, or time period may share clues. A fair test should match how new
cases will arrive.

For grouping methods, a pretty plot is not proof that the groups matter.
For text and image generators, check errors, privacy, data rights, and what
happens if someone trusts a wrong answer.

Vertex AI, Azure Machine Learning, and Amazon SageMaker are hosting and
development platforms, not learning algorithms. Their available models and
rules can change. A software link tells you where to start, not that a model
is ready for your use without further work.

## Coverage and continuation manifest

This guide explains the book's groups, field names, sources, notation, and
limits. Continue with [classical supervised methods](01-supervised-classical.md)
or choose a topic from the [complete contents](../CONTENTS.md).

Later editions could expand the areas listed in P.3 and add more well-documented
real-world results. The word "comprehensive" does not claim those additions
are already present.
