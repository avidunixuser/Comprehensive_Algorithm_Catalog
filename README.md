# Comprehensive Algorithm Catalog

## A reference book of machine learning algorithms and deep learning architectures

This book organizes classical machine learning, neural architectures, and
foundation models by their primary training signal. It connects mathematical
mechanisms to documented applications, explains implementation tradeoffs, and
separates public evidence from illustrative examples and undisclosed details.

**Catalog:** 170 algorithm, architecture, and model-family entries across
32 sub-categories, including 108 entries with neural-network deep dives.

**First-edition evidence cutoff:** September 8, 2026. Historical releases are
identified by version; inclusion does not imply that a release is current.

**Scope:** a broad, bounded reference, not a claim to enumerate every algorithm
or vendor release ever published. The reading guide explains the taxonomy,
evidence standards, and the limits of the catalog.

**About this edition:** an AI-assisted technical synthesis in eleven Markdown
chapters and reference files. The linked primary sources remain authoritative;
reported research results are not independent experimental reproductions.

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

Each entry identifies a **sourced application**, a **research benchmark**, or an
**illustrative example**. A benchmark score is not a production business outcome.
Missing public information is reported as missing, not filled with plausible
numbers. Links alongside factual claims lead to papers, technical reports,
model cards, or official documentation.

Every algorithm entry follows a common field schema. Neural entries additionally
describe architecture, activations, losses, optimization, regularization,
backpropagation, scaling, training stages, and hardware. Every sub-category ends
with a comparison table; each chapter ends with a coverage and continuation
manifest.

## Coverage and continuation

The first edition is organized as a complete reading sequence across the three
requested learning categories and the standalone MoE part. Its chapter
manifests distinguish included entries from further extensions, such as a full
reinforcement-learning catalog, causal inference, additional scientific and
recommender architectures, and newer checkpoint-specific disclosures.

An absent public parameter count or production KPI is recorded as unknown or
unreported. A research example is not relabeled as a commercial deployment to
fill that gap.
