# Comprehensive Algorithm Catalog

## A reference book of machine learning algorithms and deep learning architectures

How does a computer learn from examples? How does it find patterns or create
text and images? This book explains the methods behind these tasks.

The language is written for a high-school reader with basic algebra. Each
entry starts with **In plain English**, then explains how the method works.
You can skip the **optional math** on a first read. Technical names stay in
the book so you can recognize them in papers and software.

**Catalog:** 170 algorithm, architecture, and model-family entries across
32 sub-categories, including 108 entries with neural-network deep dives.

**First-edition evidence cutoff:** September 8, 2026. The plain-language
revision changes the explanations, not the research cutoff. Older models keep
their version names; this book does not present them as the latest releases.

**Scope:** a wide guide, not a list of every method ever invented. The reading
guide explains what is covered, how the methods are grouped, and what the
sources can tell us.

**About this edition:** eleven Markdown chapters and reference files, written
with AI assistance. Links lead to the original papers and reports. The book
explains their results; it does not claim to have repeated their experiments.

Start with the [reading guide](book/00-reading-guide.md), browse the
[complete table of contents](CONTENTS.md), or choose a part below.

## Table of contents

| Part | Chapters |
|---|---|
| Front matter | [Preface, taxonomy, notation, and evidence policy](book/00-reading-guide.md) |
| 1. Supervised Learning Algorithms | [Classical methods](book/01-supervised-classical.md); [neural architectures](book/02-supervised-neural.md) |
| 2. Semi-Supervised Learning Algorithms | [Methods combining labeled and unlabeled data](book/03-semi-supervised.md) |
| 3. Unsupervised Learning Algorithms | [Classical methods](book/04-unsupervised-classical.md); [generative and representation models](book/05-unsupervised-neural.md); [foundation-model families](book/06-foundation-models.md); [sparse MoE model catalog](book/07-moe-models.md) |
| 4. Mixture of Experts | [Foundations, routing, systems, and evidence](book/08-moe-deep-dive.md) |
| Reference material | [Cross-cutting comparisons and selection guide](book/09-comparative-guide.md); [glossary](book/10-glossary.md) |

## Reading the evidence

Each example tells you what kind of evidence it uses. A **sourced application**
describes a real use. A **research benchmark** is a test on a named dataset.
An **illustrative example** is a teaching example, not a claimed real-world
result. A high test score does not prove that a company saved money.

If a company has not shared a detail, the book says so. It does not guess
missing numbers. Source links sit near the claims they support.

All entries follow the same layout. You can find the method's purpose, inputs,
steps, strengths, limits, and example in the same places. Neural-network entries
also explain their layers, how they learn, and the computers they need.
The [field guide](book/00-reading-guide.md#p4-anatomy-of-an-entry) translates
the technical field names into everyday language.

Each sub-category ends with a comparison table. Each chapter ends with a
"coverage and continuation manifest": a short note on what it covers and what
could be added later.

## Coverage and continuation

The book covers supervised, semi-supervised, and unsupervised learning.
A separate part explains Mixture of Experts, or MoE. These models choose
which small parts of a network to use for each input.

Later editions could add more on learning through rewards, cause-and-effect
questions, recommendations, and new model releases. Those are future additions,
not sections that this edition claims to cover fully.

Unknown model sizes and unreported business results remain clearly marked.
We do not turn a research test into a claimed business success.
