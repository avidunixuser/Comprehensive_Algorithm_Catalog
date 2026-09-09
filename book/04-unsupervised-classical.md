# 3. Unsupervised Learning Algorithms: Classical Methods

This chapter explains ways to find patterns without training on a supplied answer for each example. Clustering puts similar items into groups. Representation learning makes a smaller, useful summary of each item. Other methods find things that occur together, map crowded areas, or flag unusual observations. Features are the input details, such as a pixel's color or a penguin's body mass. A vector is a list of these numeric features.

These methods learn from the data itself. They may compare distances, try to rebuild the input, or measure how well a model describes it. Labels used only to color a plot or check a result do not train the model. However, using known labels to select records, start groups, or choose settings adds outside information. We must say when that happens. Learning from examples of one reference group is not the same as learning from labeled examples of two classes.

**Evidence policy: 2026-09-08.** A research example is not proof that a company uses a method in a live product. This chapter keeps those claims separate. When a source shows only a plot or a descriptive finding, we do not invent an accuracy score. We also note when it reports no production KPI: a measure of results in a live service or business. Reasons to prefer a method are this book's analysis unless we credit the source. Versioned links identify the example checked, not the newest release. Settings and results from those examples should not be assumed to match later software or changing documentation pages.

The cost notes describe how work and storage grow as data grows. They separate preparing distances, fitting a model, storing it, and using it on new examples. They do not promise a particular running time or number of fitting steps.

**Technical detail (optional):** In the cost formulas, $`n`$ counts observations and $`d`$ counts input features. Depending on the entry, $`k`$ counts groups or neighbors. $`I`$ counts update steps, $`E`$ counts complete passes through the data, and $`B`$ is the batch size. $`r`$ is the number of features in a smaller representation. Big-O notation describes growth while leaving out constant factors.

Keep four cautions in mind:

- **You choose what "similar" means.** Changing units or rescaling features changes which differences matter most. Rescaling documents can make word proportions matter more than document length. Locations on Earth need a suitable geographic distance, not always straight-line distance between coordinates. When testing future predictions, learn preprocessing settings from training data only. Using the whole group to describe that same group is allowed, but answers a different question.
- **A group is not automatically a real-world category.** One method may favor compact groups; another may follow links or crowded neighborhoods. Try different data samples and reasonable preprocessing choices. Check the original measurements and evidence from the subject area. A silhouette score compares closeness within and between groups using your chosen distance. It cannot prove that a group is a species or a useful customer segment.
- **Different-looking answers can mean the same thing.** Swapping group names changes nothing. Some summaries can reverse signs, rotate equally strong directions, or rescale their parts without changing what they describe or rebuild. Compare the relationships or reconstructions that stay the same, not just component numbers across runs.
- **A map of old data may not describe new data well.** Some methods learn a rule that directly maps new observations. Others place all training observations together. A library's new-point feature may hold the old map fixed or estimate a position from nearby points. That is not always the same as fitting again with the new data included.

The [reading guide](00-reading-guide.md) explains the different sources of training information. The [neural unsupervised learning](05-unsupervised-neural.md) chapter covers neural networks that learn more flexible input summaries.

## 3.1 Centroid, prototype, mixture, and hierarchical clustering

These methods group items using a shared representative, a model of their spread, or a tree of smaller groups. A centroid is an average and need not be a real item. A medoid is a chosen item from the data. A Gaussian component describes a cloud of values, not necessarily a meaningful class. A dendrogram is a tree showing past merges. It records decisions that cannot be undone, not the one true way to classify things.

### 3.1.1 k-means

**In plain English:** k-means groups items whose measured features are close together. It represents each group by an average, which is useful for tasks such as reducing a photo's color palette.

**Name:** k-means; this entry uses Lloyd's method of alternating two update steps.

**Category & sub-category:** Unsupervised clustering around average representatives. It also does vector quantization: replacing many input vectors with a smaller set of representatives.

**Originating paper/vendor/year:** MacQueen's 1967 work helped establish the k-means name. Lloyd published [*Least squares quantization in PCM* in 1982](https://doi.org/10.1109/TIT.1982.1056489). These were research contributions, not inventions by a software vendor.

**Core mechanism:** Choose the number of groups and some starting centers. Assign each item to its nearest center, then replace each center with its group's average. Repeat these steps to reduce the total squared distance from items to their centers. Each exact step avoids increasing that total, but the starting centers can lead to different answers. Trying several starts or using k-means++ to spread starting centers helps. The method does not choose the group count for you: adding centers always allows an equal or smaller training error.

**Optional math:** The target is $`J=\sum_i\min_{1\le j\le k}\|x_i-\mu_j\|_2^2`$. Here, $`x_i`$ is item $`i`$'s feature vector, $`\mu_j`$ is center $`j`$, and $`k`$ is the group count. The squared-distance term measures how far an item lies from a center. $`J`$, also called inertia, adds the squared nearest-center distances over all items. A good result need not be the smallest possible $`J`$.

**Inputs/outputs and typical data types:** Inputs are lists of numbers, including image measurements or text converted into numeric features. Text can use sparse storage, which saves space by leaving out zeros. Outputs are the $`k`$ centers, each item's group, and distance summaries. A new item goes to its nearest saved center. Rescaling features changes their influence. Giving every vector length one emphasizes direction, but ordinary k-means still differs from some specialized spherical-k-means methods.

**Strengths and limitations:** It is easy to use when squared straight-line distance expresses meaningful differences. It favors compact groups that spread roughly equally in all directions, but does not require equal group sizes. Long thin groups, unequal spreads, extreme values, and irrelevant features can mislead it. Swapping group names changes nothing, and different answers can have similar errors. An average color can be useful even if no original pixel has exactly that color.

**Computational complexity / scalability notes:** Each update compares every item with every center across its features. More items, groups, features, or repeated starts mean more work. Saving bounds on distances can speed up comparisons but uses extra memory. The cost formulas do not guarantee a fixed stopping time or the best possible grouping.

**Optional math:** With $`n`$ items, $`d`$ features, and $`k`$ centers, one dense update costs $`O(nkd)`$. $`I`$ updates cost $`O(Inkd)`$, before initialization and restarts. Basic working storage beyond the input is $`O(n+kd)`$; distance-bound methods can add $`O(nk)`$. Assigning one new item costs $`O(kd)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn 1.5 Summer Palace example](https://scikit-learn.org/1.5/auto_examples/cluster/plot_color_quantization.html) reduces colors in `china.jpg`, a photograph with 96,615 distinct colors. It scales red, green, and blue values to the range from 0 to 1 (**Optional math:** $`[0,1]`$). It fits 64 centers using 1,000 sampled pixels. Every pixel is then replaced by its nearest center's color, rebuilding the image with a 64-color palette.

The displayed result keeps the photo's overall appearance and is compared with a random palette. The reason to use k-means here is to reduce color differences, not to identify objects in the photo. The source reports no measured score for perceived image quality, deployed compression service, or production KPI. A smaller palette does not by itself tell us the file-compression ratio.

**Notable vendor implementations/libraries:** Options include `sklearn.cluster.KMeans`, SciPy's vector-quantization tools, and Spark ML `KMeans`. A library offering the method does not prove that a product uses it.

### 3.1.2 Mini-batch k-means

**In plain English:** Mini-batch k-means updates group averages using small batches instead of all the data at once. It can make grouping a large collection more practical, though its groups may be less consistent.

**Name:** Mini-batch k-means, the small-batch version of average-center clustering.

**Category & sub-category:** Unsupervised clustering using sampled updates. It is intended for large datasets or data arriving over time.

**Originating paper/vendor/year:** D. Sculley described this version in [*Web-scale k-means clustering*, WWW 2010](https://doi.org/10.1145/1772690.1772862). Methods for updating numeric representatives as data arrives existed earlier.

**Core mechanism:** Take a small sample of items and assign each to its nearest center. Move each center toward its newly assigned items, using a count of its past assignments. Repeat with more batches. The goal is still to reduce k-means' total squared distance, but each batch only approximates a full-data update. Libraries may also choose starting samples, use stopping rules, and replace centers that attract too few items.

**Optional math:** In a running-average update, the update rate is $`1/c_j`$ after assignment number $`c_j`$ to center $`j`$. As that count grows, each new item moves the center less. This is not an exact Lloyd step over all observations.

**Inputs/outputs and typical data types:** Inputs are numeric batches or sparse feature tables that store mainly nonzero values. Outputs are centers and, if requested, group assignments for all items. New items use the nearest saved center. The method can accept a stream, but shrinking updates may respond poorly when the stream changes. Forgetting old data or fitting again is an extra training choice, not an automatic feature of the basic update.

**Strengths and limitations:** It needs only a small working batch and avoids repeatedly scanning all items. A small or biased batch can miss rare groups. Starting centers, data order, and unused centers all matter. Sparse inputs can still produce centers with many nonzero values. For text, truncated singular value decomposition (SVD) makes smaller feature vectors; rescaling those vectors afterward can make comparisons more useful. Both steps have their own costs and change what similarity means.

**Computational complexity / scalability notes:** A batch update grows with batch size, center count, and feature count. Getting final assignments or exact error for the whole collection still requires a full pass. Loading and preparing the data can cost more than the updates. On a small dataset, or when demanding equal final quality, mini-batches need not be faster.

**Optional math:** Let $`B`$ be batch size, $`k`$ center count, $`d`$ feature count, and $`I`$ update count. One dense update costs $`O(Bkd)`$; all updates cost $`O(IBkd)`$. Streamed working storage can be $`O(Bd+kd+k)`$, not counting saved input. Final labels or exact inertia for $`n`$ items add $`O(nkd)`$ work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn document-clustering study](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html) uses 3,387 posts from four 20 Newsgroups categories: atheism, religion, graphics, and space. It removes headers, signatures, and quoted replies. TF-IDF turns words into weights that emphasize words distinctive to a document. Latent semantic analysis (LSA), using truncated SVD, reduces these to 100 features. The example rescales the resulting vectors and fits four centers with batch size 1,000. The groups help inspect topics; they are not authoritative topic labels.

On this small collection, the example reports more variation between runs than full k-means. Our reason to consider mini-batches is handling a larger collection, not a demonstrated advantage here. Labels do not guide center updates. However, the known four categories set the requested group count and help evaluate it. Timing comparisons must include LSA fitting. No production search-quality or cost KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.cluster.MiniBatchKMeans` provides incremental updates through `partial_fit`. A distributed k-means tool does not necessarily use Sculley's update merely because it divides data into pieces.

### 3.1.3 k-medoids and PAM

**In plain English:** k-medoids groups similar items around real examples from the data, not invented averages. PAM searches for better representatives by swapping them with other items.

**Name:** k-medoids, fitted here with Partitioning Around Medoids (PAM).

**Category & sub-category:** Unsupervised clustering around observed representatives. A chosen measure of difference determines which items belong together.

**Originating paper/vendor/year:** Kaufman and Rousseeuw described medoid methods in 1987 and explained PAM in *Finding Groups in Data*, 1990. The [R `cluster::pam` manual](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html) credits them and separates their method from later FastPAM/FasterPAM improvements.

**Core mechanism:** Choose a set of actual items as representatives, called medoids. Assign every item to its closest representative. Original PAM builds a starting set, then tests swaps between selected and unselected items. It stops when no single swap reduces the total difference. This need not find the best possible set. A cheaper method that updates one medoid within each group is a different optimizer. FasterPAM speeds up and changes swap handling, but keeps the medoid goal rather than switching to k-means.

**Optional math:** Choose a set $`M`$ of $`k`$ medoids, so $`|M|=k`$, to reduce $`\sum_i\min_{m\in M}D(x_i,m)`$. Here, $`x_i`$ is input item $`i`$, $`m`$ is a medoid, and $`D`$ measures their difference. The expression adds each item's distance to its closest medoid.

**Inputs/outputs and typical data types:** Inputs can be numeric features or a table of pairwise differences. A suitable difference measure also allows words, structured objects, or records mixing categories and numbers. Outputs identify the selected real items, group assignments, and total costs. To place a new item, you must compute its differences from the saved medoids. A table containing only old-to-old differences cannot supply that information.

**Strengths and limitations:** A real representative is often easy to inspect. Using differences without squaring them can reduce the influence of extreme points compared with k-means. It does not protect against every bad record or poor distance choice. Units, missing values, and the relative importance of different feature types can decide the result. You still choose the group count, and changing group names changes nothing.

**Computational complexity / scalability notes:** Comparing and saving every pair can be costly. Doubling the number of items can roughly quadruple pairwise work and memory. PAM also tests many possible swaps, sometimes over several rounds. FasterPAM reduces each round's work. CLARA instead works with samples, trading some accuracy for scale.

**Optional math:** Let $`n`$ be item count, $`k`$ medoid count, and $`C_D`$ the cost of one distance. All distances cost $`O(n^2C_D)`$ time and $`O(n^2)`$ memory. With saved distances, original BUILD costs $`O(kn^2)`$. A SWAP round that saves best and second-best distances costs $`O(kn(n-k))`$; multiply by the rounds actually used. FasterPAM can reduce a round to quadratic work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [R `cluster` agriculture example](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/agriculture.html) uses Eurostat's 1993 records for twelve European Union countries. Its two features are gross national product (GNP) per person and the share of workers in agriculture. The command `pam(agriculture, 2)` makes two groups and plots them. The documentation describes a more-agricultural group containing Greece, Portugal, Spain, and Ireland.

The groups support comparison of economies. Unlike k-means, each representative is a real country rather than an invented average economy. The default example does not standardize the measurements, so their units affect the result. These groups are not policy recommendations. The source reports no policy intervention, evidence of cause and effect, or production KPI.

**Notable vendor implementations/libraries:** R `cluster::pam`, ELKI, and `sklearn_extra.cluster.KMedoids` offer medoid methods. For scikit-learn-extra, specify `method="pam"`: its documented default, `method="alternate"`, uses a different update.

### 3.1.4 Gaussian mixtures with expectation-maximization

**In plain English:** A Gaussian mixture describes data as overlapping clouds of values. EM improves those clouds by giving each item a share of membership in each one, rather than forcing an immediate yes-or-no assignment.

**Name:** Gaussian mixture model (GMM), fitted with expectation-maximization (EM).

**Category & sub-category:** Unsupervised grouping with model-based probabilities. It also estimates density: where the fitted model expects observations to be more concentrated.

**Originating paper/vendor/year:** Mixture modeling reaches back to the nineteenth century, including Pearson's 1894 analysis. Dempster, Laird, and Rubin set out the general EM framework in [*Maximum Likelihood from Incomplete Data via the EM Algorithm*, 1977](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x). They did not invent all mixture models.

**Core mechanism:** Start with several Gaussian, or bell-shaped, clouds. In the expectation step, calculate each item's fractional membership in each cloud. These soft assignments are called responsibilities. In the maximization step, use those fractions to update each cloud's share, center, and spread. Repeat. Exact EM does not reduce likelihood, the model's measure of fit to the observed data. Under suitable conditions it can approach a stationary point, where the local slope is zero. That point need not give the best possible fit.

**Optional math:** The density is $`p(x)=\sum_{j=1}^k\pi_j\mathcal N(x\mid\mu_j,\Sigma_j)`$. Here, $`x`$ is an input vector, $`k`$ counts clouds, and $`\pi_j`$ is cloud $`j`$'s share. $`\mathcal N`$ is a Gaussian density with center $`\mu_j`$ and covariance $`\Sigma_j`$, which describes spread and how features vary together. The responsibility for item $`x_i`$ is proportional to $`\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)`$, then scaled so its shares across clouds sum to one. Full covariance allows freely shaped tilted clouds. Tied covariance shares one shape across clouds; diagonal covariance prevents tilting; spherical covariance uses equal spread in every direction.

**Inputs/outputs and typical data types:** Inputs are continuous numeric measurements. Outputs include each cloud's settings, density or log-density scores, soft memberships, and optional single-group assignments. Log density is density expressed on a logarithmic scale. New points can be scored directly. A responsibility is a probability about the fitted model's components, not automatically about real-world classes. It is not guaranteed to be calibrated: predicted chances need not match observed class frequencies.

**Strengths and limitations:** Overlapping oval-shaped groups fit this model better than k-means' hard assignments. But one meaningful population may need several Gaussian components. Swapping component names changes nothing; duplicate components make interpretation still less clear. A component can shrink around one point and make unconstrained likelihood grow without limit. Minimum allowed spreads or prior assumptions guard against this failure.

Feature scaling, allowed cloud shapes, starting points, and component count all matter. The Bayesian information criterion (BIC) balances fit against model size; smaller is conventionally better. For mixtures, collapsed or overlapping components can break assumptions behind simple textbook interpretations of BIC.

**Computational complexity / scalability notes:** Allowing every pair of features to vary together is much more expensive than tracking each feature's spread separately. The work includes evaluating clouds, updating their spreads, and preparing matrix calculations. Saving every soft membership also takes space. Processing batches and keeping only accumulated totals can reduce that storage. The number of updates and safeguards on spread affect running time.

**Optional math:** With $`n`$ items, $`d`$ features, and $`k`$ components, a full-covariance EM step commonly costs $`O(nkd^2+kd^3)`$. Diagonal covariance reduces the main data-dependent cost to $`O(nkd)`$. Beyond the input, memberships can use $`O(nk)`$ storage and full covariances $`O(kd^2)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** [Hunt and Reffert's 2023 Gaia DR3 study](https://ar5iv.labs.arxiv.org/html/2303.13424) uses Gaussian mixtures to separate overlapping candidate star clusters. Examples include Collinder 394/NGC 6716 and UBC 76/UBC 77. Measurements of stars' positions and motions feed the mixture fits. Their assignments help revise the catalogue after HDBSCAN first detects candidate groups. The paper identifies close pairs that hard density clustering did not separate well.

Soft, overlapping clouds can help where a density method returns one connected group. This is a later step in a larger workflow, not evidence that EM alone made the catalogue. Detailed solver settings are not established here. The reported result is a revised account of cluster membership, not a universal accuracy score for separating two clusters. No business KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.mixture.GaussianMixture`, R `mclust`, and Spark ML `GaussianMixture` provide mixture models. Their allowed cloud shapes and safeguards against tiny spreads differ.

### 3.1.5 Agglomerative hierarchical clustering

**In plain English:** This method starts with each item alone, then repeatedly joins the closest groups. The resulting tree lets you inspect broad groups and smaller groups within them.

**Name:** Agglomerative hierarchical clustering, including single, complete, average, and Ward linkage rules.

**Category & sub-category:** Unsupervised grouping from the bottom up. Small groups merge into a hierarchy of larger ones.

**Originating paper/vendor/year:** Several researchers contributed to this family; it has no single originating implementation. Ward explained his spread-based rule in [*Hierarchical Grouping to Optimize an Objective Function*, 1963](https://doi.org/10.1080/01621459.1963.10500845).

**Core mechanism:** Begin with one group per item. A linkage rule defines the distance between groups, and the closest pair merges. Single linkage uses the closest pair of members across the two groups. Complete linkage uses the farthest pair, while average linkage uses the average cross-group distance. Ward instead picks the merge that adds the least squared error around group averages. Continue merging to build a tree.

**Optional math:** Ward's added error is $`\Delta(A,B)=|A||B|\|\mu_A-\mu_B\|^2/(|A|+|B|)`$. $`A`$ and $`B`$ are the groups, $`|A|`$ and $`|B|`$ their sizes, and $`\mu_A`$ and $`\mu_B`$ their averages. $`\Delta`$ is the increase in within-group squared error. This interpretation depends on Euclidean distance; it does not carry over to any arbitrary measure of difference.

**Inputs/outputs and typical data types:** Inputs are features or a suitable table of pairwise differences. You may also provide a graph, a set of links specifying which groups may merge. Outputs are the merge tree and the distance or cost recorded at each merge. Cutting the tree at a chosen level gives groups. Adding a new point can change the whole tree. Simply attaching it to a saved group is a separate prediction rule.

**Strengths and limitations:** One tree supports several levels of detail without a fresh fit for each group count. But linkage defines the result. Single linkage can join groups through a thin bridge of points. Complete linkage discourages groups with far-apart members, and Ward favors small within-group spread. Early mistakes cannot be reversed. Ties and allowed links affect merges. A tree's merge height is not a trustworthy probability that two groups are different real-world categories.

**Computational complexity / scalability notes:** Saving all pairwise distances can make both work and memory grow about fourfold when the item count doubles. Efficient methods chain nearest-neighbor searches to speed up merging. A simple repeated search can be much slower. Allowing only selected links can save work, but new links formed during merges and merge order still matter.

**Optional math:** For $`n`$ items with $`d`$ features, dense Euclidean distances cost $`O(n^2d)`$ work and $`O(n^2)`$ memory. Efficient nearest-neighbor-chain methods for common linkage rules add $`O(n^2)`$ merging work. Naive repeated searches can take cubic work. Sparse allowed links do not guarantee work proportional to $`n`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn structured Ward example](https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html) divides the `skimage.data.coins` image into regions. It smooths and shrinks the image, then uses pixel brightness and links between neighboring pixels. The displayed run groups 4,697 pixels into 27 regions and draws their boundaries over the image.

Restricting merges to linked regions helps avoid grouping distant pixels solely because they share a brightness. Ordinary brightness-based k-means lacks that restriction. The source uses more regions than coins because the background also needs regions. This demonstrates image segmentation, not verified coin counting. It reports no object-level accuracy or production KPI.

**Notable vendor implementations/libraries:** SciPy `cluster.hierarchy.linkage`, `sklearn.cluster.AgglomerativeClustering`, R `hclust`, and `fastcluster` provide these methods. Check that the selected distance, linkage, and allowed links work together.

### 3.1.6 BIRCH

**In plain English:** BIRCH first compresses a large collection into small summaries of nearby items. It can then group those summaries instead of repeatedly comparing all the original items.

**Name:** BIRCH, short for Balanced Iterative Reducing and Clustering using Hierarchies.

**Category & sub-category:** Unsupervised clustering that summarizes small groups before grouping the whole collection.

**Originating paper/vendor/year:** Tian Zhang, Raghu Ramakrishnan, and Miron Livny introduced [*BIRCH: An Efficient Data Clustering Method for Very Large Databases*, SIGMOD 1996](https://doi.org/10.1145/233269.233324). Their [1997 extended paper](https://research.ibm.com/publications/birch-a-new-data-clustering-algorithm-and-its-applications) also describes applications.

**Core mechanism:** Store small-group summaries in a branching tree kept balanced in height. Each summary records the group size and totals that describe its center and spread. For each incoming item, follow nearby summaries down the tree. Add the item if the group stays within a chosen radius or diameter limit. Split a tree node if it becomes too full. Optional later steps shrink the tree further or cluster its summaries into larger groups.

**Optional math:** A summary stores count $`N`$, vector sum $`LS=\sum x_i`$, and squared-length sum $`SS=\sum\|x_i\|^2`$. Here, $`x_i`$ is a member's feature vector; the sums cover members of that small group. These totals add when groups merge. They allow center and spread calculations without saving every pairwise distance.

**Inputs/outputs and typical data types:** Inputs are numeric records, usually compared with straight-line distance, arriving in batches or streams. Outputs are small-group summaries and, if a final clustering step is requested, larger-group labels. New records can be matched to saved representatives. The summaries do not retain every record, every original shape, or arbitrary kinds of distance.

**Strengths and limitations:** BIRCH helps when a large collection will not fit comfortably in working memory. Allowing broader small groups saves space but loses detail. That threshold does not directly set the number of final groups. Input order and early compression can change the result. Many features or long, irregular groups can make the summaries less useful. When clustering summaries, check whether group sizes are used as weights. Treating a tiny and a huge group equally changes what the final step optimizes.

**Computational complexity / scalability notes:** Inserting a record is fast when the tree stays shallow and each branch offers few choices. Claims of work proportional to the number of records assume that these search costs stay bounded. Splits, rebuilding, and later refinement add work. The final clustering method may itself become expensive when many summaries remain.

**Optional math:** Let $`b`$ bound choices per branch, $`h`$ be tree height, $`d`$ feature count, and $`n`$ record count. A normal insertion examines about $`O(bh)`$ summaries and costs $`O(bhd)`$. Total insertion work is $`O(nbhd)`$ before splits, rebuilds, and refinement. Saving $`m`$ summaries takes about $`O(md)`$ space. Final clustering adds its own cost on $`m`$, possibly quadratic.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** The authors' [1997 BIRCH application study](https://research.ibm.com/publications/birch-a-new-data-clustering-algorithm-and-its-applications) reports an interactive pixel-classification tool and a way to create starting codebooks for image compression. A codebook is a set of representative vectors used in place of many original vectors. For this task, image measurements become small-group summaries. Their representatives form the starting codebook for later image coding.

The study describes implemented applications, not a named commercial image product. Compared with merging groups over all original data, summarizing first limits memory needs before costly global work. The public abstract establishes the applications but supplies no reproducible compression-quality measure or production KPI. We do not invent either.

**Notable vendor implementations/libraries:** `sklearn.cluster.Birch` supports incremental fitting. Check its optional final clustering and use of group-size weights. Do not assume that it reproduces every stage of the original BIRCH workflow.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| k-means | Numeric features with useful straight-line distances | Reduces squared distance to group averages | Favors compact groups; can settle on a poor answer | Reducing colors in a Summer Palace photo |
| Mini-batch k-means | Large or arriving batches of numeric features | Updates with a small working batch | Results vary; rare groups can be missed | Grouping posts from four 20 Newsgroups categories |
| k-medoids / PAM | Items with a meaningful measure of difference | Uses real items as representatives | Pairwise comparisons and swaps can be costly | Comparing Eurostat agricultural-workforce records |
| Gaussian mixtures with EM | Numeric measurements forming overlapping clouds | Allows shared membership between clouds | Clouds can collapse or lack a clear meaning | Separating overlapping candidate Gaia star clusters |
| Agglomerative clustering | Moderate datasets, possibly with allowed links | Shows broad groups and smaller groups within them | Cannot undo merges; the merge rule matters | Dividing a coin image into connected Ward regions |
| BIRCH | Large streams of numeric records | Summarizes groups to save working memory | Loses detail and depends on input order | Creating a starting codebook for image compression |

## 3.2 Density and connectivity clustering

Density methods look for crowded neighborhoods separated by sparsely occupied areas. Graph methods follow links between items; stronger links mean greater similarity. Both approaches depend on choices: how to measure distance, define neighbors, build links, and turn the result into groups. How densely the data was sampled matters too. An isolated item might be a measurement error, a rare but valid event, or a member of a previously unseen population.

### 3.2.1 DBSCAN

**In plain English:** DBSCAN links crowded neighborhoods into groups and leaves isolated points outside them. It can follow curved shapes without needing you to choose the number of groups.

**Name:** DBSCAN, short for Density-Based Spatial Clustering of Applications with Noise.

**Category & sub-category:** Unsupervised clustering based on crowding within a chosen radius. It can mark points as noise rather than force them into groups.

**Originating paper/vendor/year:** Martin Ester, Hans-Peter Kriegel, Jorg Sander, and Xiaowei Xu introduced [*A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*, KDD 1996](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf).

**Core mechanism:** Choose a neighborhood radius and a minimum number of nearby observations. A point meeting that minimum is a core point. Under the usual scikit-learn convention, `min_samples` includes the point itself. Connected chains of core points form groups. A point near a core can join as a border point even if its own neighborhood is not crowded enough. All other points are marked as noise. Border points cannot extend a chain. If a border point touches two groups, its assignment may depend on processing order; the core groups stay the same.

**Optional math:** The radius is often written $`\varepsilon`$ (epsilon). It is a distance cutoff, not a probability or a requested group count.

**Inputs/outputs and typical data types:** Inputs are numeric features, coordinates, or a suitable table of distances. Outputs give group labels, core-point information, and noise flags. Latitude and longitude need suitable geographic distances and units; degrees should not automatically be treated as flat-map coordinates. Adding data can turn old border points into core points or connect old groups, so a new full fit can change past assignments.

**Strengths and limitations:** DBSCAN can find irregular connected groups and leave items unassigned. However, one radius often fails when useful groups have very different crowding levels. Rescaling features changes what that radius means. With many features, distances may become too alike to separate near from far. A plot of distances to each point's chosen-number neighbor can suggest settings, but cannot prove one correct cutoff. A tightly packed group of unusual events may look like an ordinary cluster rather than noise.

**Computational complexity / scalability notes:** A spatial index can avoid many comparisons when there are few features and neighborhoods are manageable. A simple all-pairs search is much more expensive. The original approach can find neighborhoods as needed, rather than store them all. Libraries that save all neighbors may use much more memory when the radius is large.

**Optional math:** Let $`n`$ be point count, $`d`$ feature count, and $`Q`$ the total number of neighbor appearances across searches. Useful low-dimensional indexes can approach $`O(n\log n+Q)`$ work. Brute-force distances cost $`O(n^2d)`$. Saving all neighborhoods adds $`O(Q)`$ space, which can reach $`O(n^2)`$. Thus no unconditional linear or log-linear cost claim applies.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original paper](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) tests spatial-data processing on SEQUOIA 2000's point dataset. It contains 62,584 California landmark names and locations from the US Geological Survey's Geographic Names Information System. Runtime tests use subsets containing 2% to 20% of those records. This is not a modern full-scale deployment. Coordinates become connected crowded groups plus noise for geographic inspection.

DBSCAN runs faster than CLARANS in the paper's tested setup. Separate tests on artificial data show recovery of irregular shapes. The reason to use it rather than medoid grouping is following connections without fixing a representative count. The paper establishes neither accuracy against verified landmark groups nor a production KPI for a geographic information system.

**Notable vendor implementations/libraries:** `sklearn.cluster.DBSCAN`, ELKI, and PostGIS `ST_ClusterDBSCAN` provide the method. Its availability in a database does not show that every geographic workload uses it.

### 3.2.2 OPTICS

**In plain English:** OPTICS orders points so you can inspect crowded groups at several neighborhood sizes. It helps when one fixed crowding threshold would hide useful structure.

**Name:** OPTICS, short for Ordering Points To Identify the Clustering Structure.

**Category & sub-category:** Unsupervised density analysis. It first creates an ordering, then uses a separate rule to turn that ordering into groups.

**Originating paper/vendor/year:** Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, and Jorg Sander described [*OPTICS: Ordering Points To Identify the Clustering Structure*, SIGMOD 1999](https://web.ece.ucsb.edu/Faculty/Manjunath/courses/ece594S03/Papers/optics.pdf).

**Core mechanism:** Visit points in an order that favors easy-to-reach neighbors. Record the distance needed to reach each point while accounting for how crowded the preceding point's neighborhood is. Crowded connected regions often appear as valleys when these reachability distances are plotted in order. The method does not immediately choose one final grouping. A distance cutoff, called an epsilon cut, can create groups afterward. The `xi` rule instead looks for steep changes in the plot. These choices can produce different groups from the same ordering.

**Optional math:** Through a core point $`p`$, point $`o`$ has candidate reachability $`\max\{\operatorname{coreDist}(p),D(p,o)\}`$. $`D(p,o)`$ is their distance. $`\operatorname{coreDist}(p)`$ is the radius needed for $`p`$ to meet the minimum-neighborhood requirement. Taking the larger value prevents a very short step from ignoring that requirement.

**Inputs/outputs and typical data types:** Inputs are numeric features or suitable distances. Outputs are the visit order, reachability values, and the preceding points used to reach them. A further extraction rule supplies group labels. Adding data can change the ordering. Assigning a later point to a saved group is an extension, not an update of that original ordering.

**Strengths and limitations:** OPTICS helps inspect nested groups or groups with different crowding levels without rerunning DBSCAN for every radius. It still needs settings: minimum neighborhood size, maximum search radius, distance measure, and extraction rule. Searching without a radius limit can be expensive. A valley may simply reflect where more data was collected. Many irrelevant features can hide useful neighborhood relationships.

**Computational complexity / scalability notes:** A good search index and a heap, a structure for quickly choosing the next point, can reduce work. This depends on a fixed feature count and bounded neighborhoods. Large neighborhoods add both search results and priority updates to process. The [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html) does not use a heap for choosing expansions and documents quadratic time.

**Optional math:** With $`n`$ points, favorable indexed versions can approach $`O(n\log n)`$. Brute-force distance work with $`d`$ features can be $`O(n^2d)`$. The ordering arrays grow linearly with $`n`$, but input data, indexes, and any supplied dense distance matrix require additional memory.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original OPTICS study](https://web.ece.ucsb.edu/Faculty/Manjunath/courses/ece594S03/Papers/optics.pdf) analyzes 30,000 industrial-part outlines. Sixteen Fourier-derived attributes summarize each outline using wave-based shape measurements. The OPTICS ordering and the authors' Circle Segments display reveal small and medium groups, plus a larger group separated by noise. Linked views of the attributes help engineers inspect the differences.

The output suggests possible shape groups, rather than proving defect categories. Unlike one fixed DBSCAN cut, it leaves several crowding levels available for inspection. This is the paper's industrial-contour study, not a disclosed factory deployment. The manufacturer, later quality-control decisions, defect-detection accuracy, and production KPI are not established.

**Notable vendor implementations/libraries:** Options include `sklearn.cluster.OPTICS`, ELKI, and R `dbscan`. Compare how they extract groups and how much work they require, not just their shared algorithm name.

### 3.2.3 HDBSCAN

**In plain English:** HDBSCAN searches for crowded groups that remain together across several density levels. It can find groups with different crowding levels and leave other points unassigned.

**Name:** HDBSCAN; commonly this means the HDBSCAN* hierarchy followed by stable-group selection.

**Category & sub-category:** Unsupervised clustering that builds a tree of density-based groups and marks some observations as noise.

**Originating paper/vendor/year:** Ricardo Campello, Davoud Moulavi, and Jorg Sander introduced [*Density-Based Clustering Based on Hierarchical Density Estimates*, PAKDD 2013](https://doi.org/10.1007/978-3-642-37456-2_14). Later research and software add ways to build the hierarchy, select groups, and score unusual points.

**Core mechanism:** First adjust distances so points in sparse neighborhoods are harder to link. Connect all points with a minimum spanning tree: links with the smallest possible total adjusted distance and no loops. Use these links to build nested groups, as in single-linkage clustering. Simplify branches using a minimum group size. Stable-group selection chooses branches that last across density levels. Selecting the final, smallest branches instead gives a different, finer grouping. The [implementation authors explain these separate stages](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).

**Optional math:** The adjusted, mutual-reachability distance is

$$
D_{\mathrm{mr}}(a,b)=\max\{\operatorname{coreDist}(a),\operatorname{coreDist}(b),D(a,b)\}.
$$

Here, $`a`$ and $`b`$ are points, and $`D(a,b)`$ is their ordinary distance. Each $`\operatorname{coreDist}`$ is the radius needed to meet the chosen neighborhood-count requirement. $`D_{\mathrm{mr}}`$ takes the largest of these three values. This prevents a short link involving a sparse neighborhood from looking too strong.

**Inputs/outputs and typical data types:** Inputs are numeric measurements or suitable distances. Outputs include a simplified tree, labels, noise flags, and, in some libraries, membership strengths. Minimum group size and the neighbor count used to judge crowding are separate controls. Check whether the library counts the point itself among its neighbors.

**Strengths and limitations:** HDBSCAN handles varying crowding levels more flexibly than DBSCAN with one radius. It still cannot reliably separate populations that overlap in the features you provide. Persistence across density levels is not proof of scientific importance. Membership strength is not a verified class probability or a calibrated significance measure.

New data can change the grouping. The Python package's [`approximate_predict`](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html) instead holds existing groups fixed. It can assign new points to that old structure, but cannot discover new groups or reproduce every merge that a fresh fit would make.

**Computational complexity / scalability notes:** Computing all distances can make work and memory grow roughly fourfold when the point count doubles. Tree-based searches can avoid the full distance table when there are few features and the distance measure is supported. That speedup depends on shape, dimension, and whether the calculations are exact.

**Optional math:** With $`n`$ points and $`d`$ features, a dense exact route uses $`O(n^2d)`$ distance work and $`O(n^2)`$ storage. Dense Prim-style minimum-spanning-tree construction adds quadratic work. Sorting the completed tree's $`n-1`$ links costs $`O(n\log n)`$. Favorable low-dimensional neighbor and tree algorithms can approach log-linear scaling, but there is no universal guarantee.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** [Hunt and Reffert's 2023 all-sky Gaia DR3 study](https://ar5iv.labs.arxiv.org/html/2303.13424) searches 729 million sources down to approximately magnitude $`G=20`$, a limit on brightness in Gaia's G band. It divides the sky into overlapping fields instead of comparing all sources in one huge distance table. Measurements of positions and motions feed HDBSCAN. The candidate groups then face statistical crowding checks. A separate Bayesian convolutional neural network (CNN) checks patterns in diagrams of star color versus brightness. Further steps combine and check the catalogue entries.

The stricter catalogue selection contains 4,105 highly reliable clusters, including 739 new objects. **These are counts from the whole pipeline, not an accuracy score for HDBSCAN alone.** The paper also discusses false detections and groups that may not be held together by gravity. Varying crowding levels and an unknown group count motivate HDBSCAN rather than fixed-radius DBSCAN. No commercial KPI is reported.

**Notable vendor implementations/libraries:** Python `hdbscan`, `sklearn.cluster.HDBSCAN`, and other scientific tools implement related versions. New-point prediction, parameter definitions, and speedups differ, so they are not interchangeable.

### 3.2.4 Spectral clustering

**In plain English:** Spectral clustering groups items by their links to one another, not just their distance from an average. It can separate connected shapes that ordinary center-based grouping misses.

**Name:** Spectral clustering; this entry uses a normalized graph-Laplacian representation followed by grouping.

**Category & sub-category:** Unsupervised clustering of a similarity graph. A graph is a set of items joined by links.

**Originating paper/vendor/year:** Using graph calculations to divide groups has earlier roots. Shi and Malik's [*Normalized Cuts and Image Segmentation*, 2000](https://doi.org/10.1109/34.868688), and Ng, Jordan, and Weiss's 2001 method are influential later versions. They are not identical algorithms.

**Core mechanism:** Build links between similar items and give stronger links larger, nonnegative weights. A matrix calculation finds numerical patterns that change little across strong links. These patterns give each item a new set of coordinates. Depending on the version, rescale each item's coordinates and use k-means or another assignment rule to form groups. This approximates a goal of cutting the graph into suitable parts; it does not generally solve that discrete grouping problem exactly.

**Optional math:** Let $`W`$ contain link weights and let diagonal matrix $`D`$ contain each item's total link weight. The normalized graph Laplacian is $`L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}`$, where $`I`$ is the identity matrix. The factors involving $`D`$ adjust for unequal link totals. Eigenvectors are patterns returned by this matrix calculation; those with small eigenvalues provide the new coordinates used for clustering.

**Inputs/outputs and typical data types:** Inputs are similarity tables or features used to build weighted neighbor links. A kernel is one possible rule for converting feature comparisons into similarities. Outputs are group labels and sometimes the intermediate coordinate patterns. Do not supply raw distances as link strengths: far-apart items should usually have weaker links, not stronger ones. New points require a fresh graph fit or an extension such as the Nystrom approximation.

**Strengths and limitations:** It can follow curved, connected groups and use links meaningful to a particular task. But the similarity scale, neighbor count, disconnected pieces, and unequal link totals can determine the answer. A large gap between eigenvalues can suggest a group count, not prove it. Equally valued eigenvectors can rotate without changing their shared information. The later clustering step can also vary with its starting point.

**Computational complexity / scalability notes:** Connecting every pair uses much more memory than linking only selected neighbors. A full dense matrix calculation can make work grow about eightfold when item count doubles. Sparse methods save work, but must still build the graph and run an iterative matrix solver. The later clustering stage adds another cost.

**Optional math:** With $`n`$ items and $`d`$ features, basic dense similarities cost $`O(n^2d)`$ to build and $`O(n^2)`$ to store. A full eigendecomposition costs $`O(n^3)`$. For $`m`$ links, $`r`$ coordinate patterns, and $`I_e`$ solver steps, sparse matrix and orthogonalization work is roughly $`O(I_e(mr+nr^2))`$. Orthogonalization keeps the patterns from duplicating directions. K-means with $`k`$ groups and $`I_c`$ steps then costs $`O(I_c nkr)`$. Graph construction and the steps needed for the solver to settle cannot be omitted.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn Greek-coin example](https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_segmentation.html) turns `skimage.data.coins` into a graph of neighboring pixels. After smoothing, larger brightness differences produce exponentially weaker links. Spectral clustering makes region labels and draws their boundaries over the image. The example compares three ways to assign groups after the spectral step: k-means, a discretization rule, and a QR-based matrix-factorization rule.

Using neighbor links helps follow even weak image boundaries, unlike k-means on pixel brightness alone. The result divides the image into partly uniform regions using manually chosen settings. It does not automatically determine the number of coins. No score against verified segmentation labels or production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.cluster.SpectralClustering`, SciPy's sparse matrix solvers, and R `kernlab::specc` provide useful tools. A sparse solver still needs the graph to be constructed.

### 3.2.5 Mean shift

**In plain English:** Mean shift repeatedly moves a starting point toward the average of nearby observations. Points that move toward the same crowded spot can become a group; a related update can track an object in video.

**Name:** Mean shift, used here for finding density peaks and for related tracking tasks.

**Category & sub-category:** Unsupervised search for crowded spots, called density modes. It does not assume a fixed number of Gaussian clouds.

**Originating paper/vendor/year:** Fukunaga and Hostetler's [1975 density-gradient paper](https://doi.org/10.1109/TIT.1975.1055330) helped establish the method. Cheng's [*Mean Shift, Mode Seeking, and Clustering*, 1995](https://doi.org/10.1109/34.400568), and Comaniciu and Meer's 2002 work developed influential versions and uses.

**Core mechanism:** Choose starting points, called seeds. Around each seed, calculate a weighted average of nearby observations and move the seed there. Repeat to move toward a density peak. Seeds that reach the same peak can be merged into one group. A flat neighborhood weights all points within its radius equally. Other weighting rules must match the density-smoothing rule if the update is to follow rising density.

**Optional math:** The update is $`x_{\mathrm{new}}=\sum_i w_i(x)x_i/\sum_i w_i(x)`$. Here, $`x`$ is the current seed, $`x_i`$ are observations, and $`w_i(x)`$ are their local weights. Dividing the weighted sum by the total weight gives the new average. With suitable kernel-derived weights, this moves toward increasing density. A kernel is the rule for how influence changes with distance.

**Inputs/outputs and typical data types:** Inputs can be numeric features, combined position and color measurements, or pixel coordinates weighted by matching scores. Clustering returns peak representatives and labels. Position and color scales must be chosen so one does not overwhelm the other. In tracking, a histogram-backprojection image gives each pixel a score for matching a target's color distribution. Mean shift moves a tracking window over that image; it does not return a full-data clustering.

**Strengths and limitations:** No group count is required, but the bandwidth, or neighborhood width, strongly affects the number of peaks. Too small a width breaks groups apart; too large a width joins them. A peak is not necessarily a meaningful category. Many features, many seeds, and broad neighborhoods increase cost. A fixed-size tracking window has trouble when an object changes apparent size. CamShift addresses related issues as a separate extension.

**Computational complexity / scalability notes:** The simple approach compares every seed with every observation on every update. Starting a seed at every observation can therefore be costly. Search indexes, fewer grid-based seeds, and kernels that ignore distant points can help. Choosing bandwidth and merging peaks add work. Moving a local image window can be far cheaper than clustering the whole dataset.

**Optional math:** With $`q`$ seeds, $`I`$ updates per seed, $`n`$ observations, and $`d`$ features, naive work is $`O(Iqnd)`$. A seed for every observation gives $`O(In^2d)`$. Savings from indexes and restricted neighborhoods depend on the data.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [OpenCV 4.13.0 mean-shift tutorial](https://docs.opencv.org/4.13.0/d7/d00/tutorial_meanshift.html) tracks an object in `slow_traffic_small.mp4`. A selected starting region supplies a hue histogram, which counts its color tones. Each later frame receives pixel scores for matching that histogram. Mean shift moves a fixed-size window toward a concentration of matching pixels, producing a tracked bounding box.

Following a local peak avoids reclustering all colors with k-means on each frame. However, the selected starting window supplies information about the target. The whole tracking workflow is therefore not wholly unsupervised just because each update uses no class labels. The source reports no tracking-accuracy benchmark, deployed traffic service, or production KPI.

**Notable vendor implementations/libraries:** `sklearn.cluster.MeanShift` clusters data. OpenCV `meanShift` and `pyrMeanShiftFiltering` use related ideas for different image tasks; their outputs and surrounding workflows differ.

### 3.2.6 Affinity propagation

**In plain English:** Affinity propagation lets items compete to become representatives of similar items. It repeatedly updates numerical messages until representatives and their groups emerge.

**Name:** Affinity propagation, a message-passing method for selecting representatives.

**Category & sub-category:** Unsupervised clustering around real examples, using comparisons between pairs of items.

**Originating paper/vendor/year:** Brendan J. Frey and Delbert Dueck introduced [*Clustering by Passing Messages Between Data Points*, Science 2007](https://doi.org/10.1126/science.1136800).

**Core mechanism:** Supply pairwise similarities and a preference for each item becoming a representative. The algorithm updates two kinds of numbers. A responsibility compares one candidate representative with its competitors for a particular item. An availability reflects support from other items for choosing that representative. These are numerical messages, not communication between thinking agents. Repeated updates choose representatives and groups together. Damping blends old and new messages to reduce back-and-forth swings. Raising preferences often creates more representatives, but does not set an exact group count.

**Optional math:** $`s(i,j)`$ is the supplied similarity between items $`i`$ and $`j`$. The diagonal values $`s(j,j)`$ encode preferences for choosing item $`j`$ as a representative. They are settings, not measured probabilities.

**Inputs/outputs and typical data types:** Inputs are pairwise similarities from images, text, or other meaningful comparisons. Outputs identify representative items and their groups. Negative squared straight-line distance is a common similarity choice, but not required. A table of training similarities alone may lack the features or comparison rule needed to place new items near saved representatives.

**Strengths and limitations:** It chooses real representatives without starting from a fixed-size medoid set. It can use comparisons for objects that have no meaningful average. Yet preferences still guide the model, so selection is not assumption-free or fully automatic. Messages can keep oscillating or fail to settle. Damping does not guarantee the best answer. Duplicate items and ties can make representative choices unclear. Fitting with new data can change old groups, even if a library also offers prediction with frozen representatives.

**Computational complexity / scalability notes:** Dense versions keep messages for every pair, so doubling item count can roughly quadruple storage and work per update. Reusing each row's largest values avoids unnecessary repeated searches. Similarity construction adds a separate cost. Restricted-link versions may save work, but you must actually use such an implementation; calling dense input a graph does not make scikit-learn's calculations sparse.

**Optional math:** With $`n`$ items and $`I`$ updates, dense message updates cost $`O(n^2)`$ per step and $`O(In^2)`$ overall, with $`O(n^2)`$ memory. For $`d`$-feature vectors, preparing similarities may add $`O(n^2d)`$ work. Sparse variants depend on which candidate links are kept.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Frey and Dueck's University of Toronto [2007 study and author abstract](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=17218491&retmode=xml) describe face-image grouping, gene detection in microarray measurements, representative sentences, and cities accessible by airline travel. In the face-image experiment, pairwise similarities produce representative faces and groups of associated images for visual inspection.

Unlike starting with a random subset of medoids, the method lets possible representatives compete together. The paper reports favorable grouping-error and runtime results. They depend on the experiment and comparison method, so no universal speedup is claimed here. This is a research study, not evidence of a commercial face-identification deployment. No production KPI is established.

**Notable vendor implementations/libraries:** `sklearn.cluster.AffinityPropagation`, R `apcluster`, and the authors' reference code provide implementations. Check whether the updates actually settled before interpreting the representatives.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| DBSCAN | Locations or a few features with useful distances | Follows connected shapes and allows noise | One radius can miss differently crowded groups | Grouping SEQUOIA 2000 landmarks |
| OPTICS | Distance-based data with several crowding levels | Shows a density ordering for inspection | Group extraction choices and searches can be costly | Exploring industrial-part outlines |
| HDBSCAN | Distance-based observations with varying crowding | Selects groups that persist across density levels | Persistent groups are not automatically meaningful | Finding candidates in the Gaia DR3 catalogue pipeline |
| Spectral clustering | Graphs with a limited set of similarity links | Uses links to separate connected groups | Building the graph and solving matrices costs work | Dividing a Greek-coin image into regions |
| Mean shift | Numeric features or images with several density peaks | Finds peaks without a fixed group count | Neighborhood width controls the peaks found | Moving a tracking window in an OpenCV traffic video |
| Affinity propagation | Moderate-sized tables of similarities | Chooses real representatives together | Pairwise storage is large; updates may not settle | Selecting faces in Frey-Dueck research experiments |

## 3.3 Representation and dimensionality reduction

A representation gives each observation a new, often smaller set of features. What makes it useful depends on what you need to keep. You might want to rebuild the input, separate mixed signals, keep nearby items together, or display gradual changes. No two-dimensional picture can faithfully preserve every relationship in a complicated dataset with many features.

Test the property you actually need. If you want compression, check reconstruction error. If you want useful neighbors, check which neighbors survive. If the summary will help predict outcomes, test that later task on data held out from fitting.

In particular, **t-SNE and UMAP plots are not proof of clusters**. Their drawing rules can enlarge gaps, squeeze some neighborhoods, spread others out, and rearrange groups. A pretty plot is a starting point for questions, not proof of separate populations. Check the original measurements, several settings and random starts, and how observations were sampled. Seek independent evidence before giving an island a scientific meaning.

### 3.3.1 Principal component analysis

**In plain English:** PCA turns many measurements into a smaller useful summary that keeps as much variation as possible. It can simplify data or reduce noise, but the biggest variations are not always the most important ones.

**Name:** Principal component analysis, usually called PCA.

**Category & sub-category:** Unsupervised representation learning using weighted combinations of the original features. Those combinations follow perpendicular directions and can approximately rebuild the input.

**Originating paper/vendor/year:** Karl Pearson described [*On Lines and Planes of Closest Fit to Systems of Points in Space*, 1901](https://doi.org/10.1080/14786440109462720). Harold Hotelling developed the statistical principal-component formulation in 1933.

**Core mechanism:** Subtract each feature's average so the data is centered. Find a weighted combination of features that varies as much as possible across observations. Find further perpendicular directions that capture the most remaining variation. Keep only the chosen number of directions. Each observation becomes a few numbers describing its position along them. These numbers provide a smaller summary and allow an approximate reconstruction.

**Optional math:** Singular value decomposition (SVD) writes the centered table as $`X_c=U\Sigma V^\top`$. $`X`$ is the original data table, $`X_c`$ its centered version, $`V`$ contains feature directions, $`\Sigma`$ contains their strengths, and $`U`$ describes observation-side patterns. The superscript $`\top`$ transposes rows and columns. If $`V_r`$ contains the first $`r`$ directions, the scores are $`X_cV_r`$. These directions retain the most variance and give the least squared reconstruction error among rank-at-most-$`r`$ linear approximations. Rank here limits the number of independent directions used. This guarantee concerns the given centered data and squared Euclidean error, not every later task. Uncentered truncated SVD, often used for text LSA, solves a related but different problem.

**Inputs/outputs and typical data types:** Inputs are continuous numeric features. Outputs include loadings, the feature weights defining each direction; scores, each item's new coordinates; and explained variances, how much spread each direction keeps. The model also stores the mean and can reconstruct approximate inputs. New items use the same learned linear rule.

Standardizing features first changes the question from variation in original units to variation relative to each feature's spread. This is the difference between covariance-style and correlation-style PCA. Whitening takes another step: it gives the retained directions equal spread. That can make small noisy directions more influential.

**Strengths and limitations:** PCA is a useful starting point for compression, plots, noise reduction, and making later numerical calculations easier. Yet a low-variation feature may still predict a rare outcome. Outliers, differences between collection batches, or large measurement units can dominate the summary. Keeping high variation does not prove meaning or cause and effect. A direction can reverse sign without changing the result. Equally strong directions can rotate together, so their shared information can be stable even when individual displayed axes change.

**Computational complexity / scalability notes:** A full matrix decomposition becomes costly as both observations and features grow. Randomized methods can focus on a smaller set of directions, trading extra passes and accuracy settings against work. After fitting, mapping a new observation is much cheaper: apply the saved weights and mean.

**Optional math:** Let $`n`$ count observations, $`d`$ input features, and $`r`$ retained directions. Dense economy SVD costs $`O(nd\min(n,d))`$. An alternative forms covariances in $`O(nd^2)`$, then solves a $`d\times d`$ eigenproblem. Randomized methods use passes costing about $`O(ndr)`$ each, plus work keeping directions perpendicular. Oversampling and power iterations, extra calculations that improve the approximation, affect accuracy. The saved transform uses $`O(dr+d)`$ values and $`O(dr)`$ work per new observation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** [Novembre and colleagues' *Genes mirror geography within Europe*, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2735096/) studies POPRES genetic measurements. After quality checks and ancestry-based selection, its main analysis uses 197,146 genetic locations, or loci, in 1,387 people. PCA turns that variation into two coordinates. The authors rotate the display to highlight its resemblance to European geography.

The result shows gradual geographic variation, not separate biological races. It helps researchers understand population structure that could otherwise confuse studies linking genetic variants to traits. Compared with a nonlinear plot, PCA offers inspectable linear features that can be reused as adjustment variables. The paper's separate ancestry-location predictions require additional modeling; they are not results of PCA alone. No clinical or commercial KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.decomposition.PCA`, NumPy/SciPy SVD, R `prcomp`, and Spark ML `PCA` are common choices. State the solver and centering choices so others can reproduce the analysis.

### 3.3.2 Kernel PCA

**In plain English:** Kernel PCA makes a smaller summary using a chosen rule for how similar items are. It can follow curved patterns that ordinary PCA's straight-line combinations miss.

**Name:** Kernel principal component analysis, or kernel PCA.

**Category & sub-category:** Unsupervised representation learning using a kernel, a valid rule for computing feature-based similarity.

**Originating paper/vendor/year:** Bernhard Scholkopf, Alexander Smola, and Klaus-Robert Muller published [*Nonlinear Component Analysis as a Kernel Eigenvalue Problem*, Neural Computation 1998](https://doi.org/10.1162/089976698300017467).

**Core mechanism:** Compare every training item with the others using a kernel. Center the resulting similarity table, then find its main numerical patterns. Use those patterns to give each item fewer coordinates. This acts like PCA on an expanded feature description without explicitly building that description. The result is usually nonlinear in the original features: a straight direction in the expanded description can correspond to a curved pattern in the input. For a new item, compare it with saved training items and center those comparisons using the training information.

**Optional math:** The Gram matrix, or kernel similarity table, has entries $`K_{ij}=k(x_i,x_j)`$. Here, $`x_i`$ and $`x_j`$ are items and $`k`$ is a positive-semidefinite kernel, the mathematical requirement for this PCA interpretation. Center it as $`HKH`$, where $`H=I-\mathbf1\mathbf1^\top/n`$. $`n`$ counts training items, $`I`$ is the identity matrix, and $`\mathbf1`$ is a vector of ones. This subtracts the needed means from the similarity table. Its eigenvectors give the new directions.

**Inputs/outputs and typical data types:** Inputs are feature vectors or a valid kernel table. Outputs are new coordinates and eigenvalues describing the strength of the extracted patterns. Saved reference items plus a callable kernel allow new-point projection. An old-to-old kernel table alone cannot provide new-to-old similarities. Rebuilding the input is a separate inverse, or pre-image, problem. It may have several answers or no exact answer.

**Strengths and limitations:** It captures curved variation without a neural network. However, the kernel and its scale decide which similarities matter. With a radial basis function (RBF) kernel, a poor bandwidth can make almost all items look alike or almost all look isolated. Centering raw features is not a substitute for centering the kernel table. Directions can still reverse sign, and equally valued directions can rotate together. Reconstruction depends on a separately learned backward mapping, not just on the forward coordinates.

**Computational complexity / scalability notes:** A full pairwise kernel table grows quickly: doubling the number of items roughly quadruples storage. A full matrix solve can grow about eightfold in work. Using selected reference landmarks can reduce costs but gives an approximation. Reconstructing inputs with kernel ridge regression, a fitted mapping with a penalty against excessive complexity, adds its own training and prediction work.

**Optional math:** With $`n`$ items, $`d`$ input features, and $`r`$ output coordinates, basic dense kernel construction costs $`O(n^2d)`$ and storage $`O(n^2)`$. A full eigenproblem costs $`O(n^3)`$. Partial solvers depend on requested rank and the steps needed to settle. Mapping one new item against all references costs $`O(nd+nr)`$. Nystrom landmark approximations reduce this reference burden.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn 1.5 USPS denoising study](https://scikit-learn.org/1.5/auto_examples/applications/plot_digits_denoising.html) uses 16-by-16 handwritten-digit images. It separates 1,000 training and 100 test images and adds Gaussian random noise. Linear PCA and RBF kernel PCA fit the noisy training images, then reconstruct noisy test images. Kernel ridge regression learns kernel PCA's backward mapping.

Linear PCA has lower MSE: mean squared error, the average squared reconstruction error, where lower is better. Kernel PCA produces smoother-looking backgrounds. The models use different component counts, so this does not prove superiority at equal summary size. Kernel PCA is worth considering when curved image patterns resist linear summaries, but a nicer-looking background is not proof of lower error. No postal-service deployment, improved recognition accuracy, or production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.decomposition.KernelPCA` and R `kernlab::kpca` provide the method. An `inverse_transform` option offers an approximate reconstruction, not a guaranteed mathematical inverse.

### 3.3.3 Independent component analysis

**In plain English:** ICA tries to separate signals that were mixed together, such as several sources recorded by the same sensors. It uses patterns of independence rather than needing labeled examples of each source.

**Name:** Independent component analysis (ICA); FastICA is the representative fitting method here.

**Category & sub-category:** Unsupervised linear source separation. "Blind" separation means the original source signals and mixing weights are not supplied.

**Originating paper/vendor/year:** Source-separation research predates the ICA name. Pierre Comon set out a statistical view in [*Independent Component Analysis, a New Concept?*, 1994](https://doi.org/10.1016/0165-1684(94)90029-9). Hyvarinen and Oja introduced their [FastICA fixed-point algorithm in 1997](https://doi.org/10.1162/neco.1997.9.7.1483).

**Core mechanism:** Assume each recorded channel is approximately a weighted sum of hidden source signals. Center and whiten the data: remove means, remove straight-line correlations, and equalize spread. Then repeatedly adjust an unmixing rule to recover signals that behave as independently as possible. FastICA commonly does this by seeking signals that differ from a Gaussian bell-shaped distribution while keeping recovered directions separate. Independence is stronger than PCA's lack of straight-line correlation; two uncorrelated signals can still depend on one another.

**Optional math:** The mixing assumption is $`x=As`$. $`x`$ contains recorded channel values, $`s`$ contains the hidden source values, and matrix $`A`$ holds mixing weights. ICA estimates a transformation in the opposite direction. FastICA uses fixed-point updates, repeatedly applying an update rule until it changes little, rather than treating PCA alone as source separation.

**Inputs/outputs and typical data types:** Inputs include recordings from several sensors over time, spectra, or other roughly linear mixtures of numeric signals. Outputs include recovered component signals, mixing and unmixing weights, and reconstructed recordings. The same unmixing rule can process later samples if the mixing process remains sufficiently stable.

**Strengths and limitations:** ICA can separate nuisance signals without labeled examples of every artifact, meaning unwanted recording contamination. Under standard linear ICA assumptions, at most one source may have a Gaussian distribution for all sources to be uniquely separable. Even then, source order, sign, and nonzero scale remain arbitrary. Independence alone does not tell you that a component is brain activity or eye movement. Outside evidence is needed. Related sources, nonlinear or changing mixtures, a poor component count, and sensor problems can undermine the model.

**Computational complexity / scalability notes:** Preparing the whitened data needs a matrix decomposition or covariance calculation. Repeated FastICA updates then depend on how many dimensions remain. Different starting points and scoring rules can require different numbers of updates. Keeping all samples and the learned mixing weights adds storage.

**Optional math:** With $`n`$ samples reduced to $`r`$ whitened dimensions, a typical parallel FastICA update costs $`O(nr^2+r^3)`$ for projected scores and decorrelation. Multiply by the actual update count. Sample storage is $`O(nr)`$. Mixing and unmixing weights add about $`O(dr)`$, where $`d`$ is the original channel or feature count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [MNE-Python ICA correction tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) removes eye-blink and heartbeat contamination from a sample MEG/EEG recording. MEG measures magnetic signals associated with brain activity; EEG measures electrical signals. A high-pass filter removes slow changes from a copy used to fit fifteen ICA components. Component waveforms, scalp patterns, and EOG/ECG reference recordings of eye and heart activity help choose unwanted components. Removing those components and reconstructing the original recording visibly reduces the artifacts.

Separating unwanted sources can preserve more observations than discarding whole time intervals. However, manual or sensor-assisted component selection adds information beyond unsupervised fitting. A component may contain both brain activity and contamination, so removing it can lose useful signal. This is an analysis tutorial, not a patient-diagnosis accuracy test or production KPI.

**Notable vendor implementations/libraries:** `sklearn.decomposition.FastICA`, MNE-Python, EEGLAB's ICA workflows, and R `fastICA` are available. Infomax and Picard are other ways to fit ICA, not different names for FastICA's update.

### 3.3.4 Non-negative matrix factorization

**In plain English:** NMF describes each item by adding together reusable patterns, using no negative amounts. It can summarize images or text in parts that are easier to inspect.

**Name:** Non-negative matrix factorization, or NMF.

**Category & sub-category:** Unsupervised representation learning that builds a smaller description from additive patterns.

**Originating paper/vendor/year:** Earlier work includes positive matrix factorization. Daniel D. Lee and H. Sebastian Seung popularized the parts-based view in [*Learning the Parts of Objects by Non-negative Matrix Factorization*, Nature 1999](https://doi.org/10.1038/44565).

**Core mechanism:** Start with a nonnegative data table. Learn a smaller collection of reusable patterns and a nonnegative amount of each pattern for every item. Alternate between improving the patterns and improving their amounts, so their sum rebuilds the data as closely as possible. Choose an error measure suited to the data. Optional penalties can encourage many amounts to be zero. Different update methods improve different pieces of this problem, and none of the general methods guarantees the best possible overall answer.

**Optional math:** Approximate $`X\in\mathbb R_+^{n\times d}`$ by $`WH`$, where $`W\in\mathbb R_+^{n\times r}`$ and $`H\in\mathbb R_+^{r\times d}`$. $`n`$ counts items, $`d`$ input features, and $`r`$ patterns. $`\mathbb R_+`$ means nonnegative entries; $`W`$ holds amounts and $`H`$ the patterns. Squared Frobenius error adds squared differences between corresponding table entries. Generalized KL divergence is another way to measure mismatch. Multiplicative updates rescale values, coordinate descent changes selected values, and alternating nonnegative least squares fits one nonnegative table while holding the other fixed.

**Inputs/outputs and typical data types:** Inputs include nonnegative brightness, concentrations, counts, or TF-IDF word weights. TF-IDF emphasizes words that help distinguish documents. Outputs are additive patterns and the amounts used for each observation. Centering the data may create negative entries, which the basic model does not allow. Rescaling rows or columns changes which errors matter and may change an interpretation from amounts to proportions.

**Strengths and limitations:** Without negative amounts, one pattern cannot cancel another through subtraction. This often helps interpretation, but does not guarantee local image parts, many zero entries, or the uniquely correct topics. Different patterns can rebuild the same input. A pattern might reflect collection-batch differences or preprocessing instead of a meaningful source. For a new item, usually keep the patterns fixed and solve for allowed nonnegative amounts; there is no simple perpendicular projection as in PCA.

**Optional math:** If $`S`$ is a diagonal matrix with positive scale factors, $`WH=(WS)(S^{-1}H)`$. Making a pattern smaller while increasing its amount can leave the reconstruction unchanged. Reordering patterns, or finding other valid factorizations, creates further ambiguity. Thus the separate tables are not necessarily uniquely determined.

**Computational complexity / scalability notes:** Work grows with data size and the number of patterns. Sparse input can save multiplications, though the learned patterns and amounts may still contain many nonzero entries. The starting values, stopping tolerance, and chosen error measure affect how many updates are needed. Some multiplicative updates lock a value at zero once it reaches zero, which can limit later improvement.

**Optional math:** For $`n`$ items, $`d`$ features, and $`r`$ patterns, a typical dense alternating step costs about $`O(ndr+(n+d)r^2)`$, depending on the solver and loss. Suitable sparse squared-error methods replace the leading $`ndr`$ work by $`\operatorname{nnz}(X)r`$, where $`\operatorname{nnz}(X)`$ counts nonzero input entries. Factor storage is $`O((n+d)r)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn 1.5 Olivetti face-decomposition example](https://github.com/scikit-learn/scikit-learn/blob/1.5.X/examples/decomposition/plot_faces_decomposition.py) compares several ways to summarize face images. NMF uses the original nonnegative pixels, not the centered values used by PCA and ICA. It learns six nonnegative basis images and a list of amounts for each input face. The example displays the basis images for inspection.

The reason to use NMF rather than signed PCA components is to explain brightness by addition without positive and negative parts canceling. This demonstrates a representation, not a deployed face-recognition system or a tested identity classifier. It claims no recognition accuracy, privacy protection, or production KPI.

**Notable vendor implementations/libraries:** `sklearn.decomposition.NMF`, `MiniBatchNMF`, R NMF packages, and scientific optimization tools offer implementations. Before comparing results, check the beta-divergence, the chosen mismatch measure, along with rescaling and penalties that constrain the fit.

### 3.3.5 t-SNE

**In plain English:** t-SNE draws a small map that tries to keep similar items close to one another. It is useful for exploring many measurements, but gaps and island sizes on the map can be misleading.

**Name:** t-distributed stochastic neighbor embedding, usually called t-SNE.

**Category & sub-category:** Unsupervised visualization focused on nearby items, usually in a two-dimensional plot.

**Originating paper/vendor/year:** Laurens van der Maaten and Geoffrey Hinton introduced [*Visualizing Data using t-SNE*, JMLR 2008](https://jmlr.org/papers/v9/vandermaaten08a.html). It builds on earlier stochastic neighbor embedding.

**Core mechanism:** Compare each item with other items in the original feature space. Give nearby items stronger neighbor weights, adjusting the neighborhood scale separately around each item. The perplexity setting controls roughly how broad those neighborhoods are. Place the items on a smaller map, then repeatedly move them so important original neighbors remain close. The method strongly discourages losing close neighbors; it does not try to preserve every long distance.

The starting map, update size, and number of updates affect the result. So does early exaggeration, a starting phase that temporarily strengthens neighbor attraction.

**Optional math:** t-SNE reduces $`\mathrm{KL}(P\|Q)`$. $`P`$ is the table of neighbor probabilities built from the original features. $`Q`$ is the corresponding table on the map, using a heavy-tailed Student distribution that allows distant points more room. KL divergence measures mismatch between those tables, with a strong cost when an important neighbor in $`P`$ is weak in $`Q`$. These probabilities describe neighbor relationships, not verified class membership.

**Inputs/outputs and typical data types:** Inputs are numeric features or suitable distances, often after PCA or another noise-reduction step. Output is a coordinate list for each training observation, usually two numbers. Ordinary t-SNE learns where to put these points, not a general rule for encoding all future items or rebuilding their inputs. New-point interpolation and parametric t-SNE add methods and assumptions beyond this original setup.

**Strengths and limitations:** It helps inspect local similarities in data with many features. But an island's plotted area does not measure its original spread or crowding. Two islands far apart on the page need not be more different than two islands nearby. Settings and sampling can create dramatic gaps. [Wattenberg, Viegas, and Johnson's controlled experiments](https://distill.pub/2016/misread-tsne/) even show apparent clumps in random data. Compare several perplexities and random starts, and check neighbors using the original features. Do not choose the prettiest plot and treat it as proof.

**Computational complexity / scalability notes:** Exact pairwise comparisons and point movements can become costly as the dataset grows. Doubling the item count can roughly quadruple this work and storage. Barnes-Hut t-SNE approximates the forces used to move plotted points, reducing that part of the cost. It still needs neighbors, accuracy settings, and repeated updates.

**Optional math:** With $`n`$ items and $`d`$ input features, dense exact neighbor weights use $`O(n^2)`$ storage and often $`O(n^2d)`$ distance work. For a map with $`r`$ coordinates per item, exact force calculations cost $`O(n^2r)`$ per update. [Barnes-Hut t-SNE](https://www.jmlr.org/papers/v15/vandermaaten14a.html) uses about $`O(n\log n)`$ per force update in the low-dimensional setting. That is not a log-linear guarantee for preparing and fitting the entire pipeline.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn handwritten-digit comparison](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html) maps 64-pixel feature vectors from the first six digit classes. t-SNE starts from PCA coordinates. Digit identities color and label the finished map; they do not enter its fitting objective. The two-dimensional arrangement helps inspect related writing styles and overlap between classes.

Unlike linear PCA, t-SNE can reveal local curved relationships. The source compares visual representations, not recognition accuracy. Selecting six classes does not establish six naturally separate populations. The example also rescales its display, so plotted distances must not be read as original-feature distances. No production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.manifold.TSNE`, openTSNE, and FIt-SNE implement t-SNE methods. Some add a way to place new points, but the original method still fits coordinates rather than a general encoder.

### 3.3.6 UMAP

**In plain English:** UMAP builds links between nearby items and draws a smaller map that tries to keep those links useful. It can reveal patterns to investigate, but a neat island is not proof of a real group.

**Name:** Uniform Manifold Approximation and Projection, or UMAP.

**Category & sub-category:** Unsupervised representation learning and visualization using a graph of neighbor links.

**Originating paper/vendor/year:** Leland McInnes, John Healy, and James Melville first posted [*UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*](https://arxiv.org/abs/1802.03426) in February 2018. The linked arXiv record also has later revisions. That initial posting date is not a conference-publication date.

**Core mechanism:** Find nearby items using a chosen distance measure. Build directed links with strengths adjusted to the crowding around each item, then combine the two directions into shared link strengths. Place points in a smaller space and repeatedly adjust them. Strong links pull points together; sampled comparison pairs provide repulsion. Practical software samples these updates instead of checking every unlinked pair.

The setting `n_neighbors` controls how broad the neighborhood view is. `min_dist` controls how tightly points may pack on the output map. The fitting objective is called cross-entropy: it measures disagreement between link strengths in the original graph and the smaller map.

**Inputs/outputs and typical data types:** Inputs are numeric features with a supported distance measure, or another appropriate distance representation. Outputs give each item a chosen number of coordinates, often written $`r`$. Standard unsupervised UMAP uses no target labels. Supervised UMAP does use them and is a different training setup. The common implementation can place new vectors by finding their training neighbors and adjusting their positions while keeping the old map fixed.

**Strengths and limitations:** UMAP is practical for exploration and for building features used by later models. The output can have more than two dimensions. But global distances, original crowding, and the full pattern of connections and holes need not survive. The fitting process can create tight islands. Methods for better preserving density are additions, not guarantees of ordinary UMAP.

The [authors' new-point tutorial](https://umap-learn.readthedocs.io/en/latest/transform.html) assumes training and test observations come from similar distributions. A new kind of item may still be assigned a position near old items. Being drawn there does not prove it belongs to their population. Check original-feature relationships and several settings, not just the finished picture.

**Computational complexity / scalability notes:** Exact neighbor search may compare every pair. Approximate search can save work, depending on its index and search effort. A limited-neighbor graph uses far fewer links than a full pairwise table. Fitting cost also depends on output size, repeated passes, and the number of repulsion samples.

**Optional math:** Let $`n`$ count items, $`d`$ input features, $`k`$ neighbors, and $`r`$ output coordinates. Exact search can cost $`O(n^2d)`$, while the graph stores $`O(nk)`$ links. With $`E`$ passes and $`s`$ negative, or repulsion, samples per sampled link, a broad fitting estimate is $`O(E nkr(1+s))`$, assuming that many link updates per pass. Coordinates use $`O(nr)`$ space; reference data and indexes add more. A universal $`O(n\log n)`$ claim would omit these dependencies.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [UMAP authors' Palmer Penguins tutorial](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) keeps 333 complete records. It standardizes bill length, bill depth, flipper length, and body mass, putting them on comparable scales. UMAP turns these four measurements into two coordinates. Species labels supply colors only afterward. The tutorial compares the result with plots of the original feature pairs and shows recognizable organization related to species.

UMAP offers a curved neighborhood summary rather than a raw projection. Yet four features also allow direct inspection, so that comparison remains valuable. The picture does not prove discovery of a new biological population. No ecological-management outcome or production KPI is reported.

**Notable vendor implementations/libraries:** `umap-learn` and R `uwot` provide UMAP. Parametric UMAP adds a neural encoder, so it is not the non-neural version described here.

### 3.3.7 Isomap

**In plain English:** Isomap tries to unfold a curved pattern while preserving travel distances along it. It uses chains of nearby observations instead of assuming a straight shortcut always measures similarity well.

**Name:** Isomap, short for isometric feature mapping.

**Category & sub-category:** Unsupervised representation learning based on paths through a neighbor graph. It assumes the data traces a lower-dimensional shape, often called a manifold.

**Originating paper/vendor/year:** Joshua B. Tenenbaum, Vin de Silva, and John C. Langford introduced [*A Global Geometric Framework for Nonlinear Dimensionality Reduction*, Science 2000](https://doi.org/10.1126/science.290.5500.2319).

**Core mechanism:** Link nearby observations and weight each link by distance. Find the shortest path along those links between each pair of observations. These path lengths estimate distances along the underlying curved shape. Then use classical multidimensional scaling (MDS) to place points in fewer dimensions while representing those distances. Unlike PCA, it tries to preserve long-range distance along the shape, not distance straight through the surrounding feature space.

**Technical detail (optional):** Geodesic distance means shortest travel along a shape. Isomap estimates it using graph paths. MDS squares the path distances and double-centers the distance table, subtracting row and column means and accounting for the overall mean. Eigenvectors from that matrix calculation supply the output coordinates.

**Inputs/outputs and typical data types:** Inputs are numeric vectors or differences that make useful local neighborhoods. Outputs include coordinates, neighbor links, and path-distance information. Possible inputs include images with gradually changing pose or closely sampled physical movement. These uses rely on local feature distances actually tracking smooth changes.

**Strengths and limitations:** It can unfold curved patterns when sampling and geometry support the assumption. Too many neighbors create shortcuts across folds; too few leave separate pieces with no path between them. Noise, holes, uneven sampling, or several unrelated shapes can make paths misleading. Negative eigenvalues in the distance calculation show that no exact Euclidean map can match all the inferred distances. The map's orientation is arbitrary. A good-looking unfolding does not prove that it preserves the true hidden distances.

**Computational complexity / scalability notes:** Searching all pairs, finding paths from every point, and saving every path distance can be expensive. Landmark methods reduce this work by approximation. A library's `transform` typically attaches new points to the old graph and extends the map. It does not rebuild the joint geometry with all points included.

**Optional math:** With $`n`$ items, $`d`$ features, and $`k`$ neighbors, exact neighbor search can cost $`O(n^2d)`$. For $`O(nk)`$ links, binary-heap Dijkstra searches from all points cost about $`O(n^2k\log n)`$. The alternative Floyd-Warshall path algorithm costs $`O(n^3)`$. All path distances use $`O(n^2)`$ storage, and a full dense MDS eigenproblem also takes cubic work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn digits study](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html) turns 8-by-8 images from six digit classes into 64-feature vectors. Isomap uses the study's thirty-neighbor setting and draws a two-dimensional map for inspecting writing-style changes. The output is a plot compared with PCA-like and local-neighborhood methods, not a classifier or deployed service.

The reason to try Isomap is to keep variation along curved paths that a straight-line projection might flatten. However, distinct digit classes also make the assumption of one connected shape questionable. The example supplies no score for preservation of original-space distances and no production KPI.

**Notable vendor implementations/libraries:** `sklearn.manifold.Isomap`, scientific MDS and eigenproblem tools, and landmark-Isomap implementations are available. Check what each does with disconnected graphs and new observations.

### 3.3.8 Locally linear embedding

**In plain English:** LLE describes each item using a weighted recipe of its neighbors. It then draws a smaller map where those same local recipes still work.

**Name:** Locally linear embedding (LLE); this entry focuses on standard LLE.

**Category & sub-category:** Unsupervised representation learning that preserves local reconstruction relationships. It assumes small neighborhoods can be described with linear combinations even when the overall shape curves.

**Originating paper/vendor/year:** Sam T. Roweis and Lawrence K. Saul introduced [*Nonlinear Dimensionality Reduction by Locally Linear Embedding*, Science 2000](https://doi.org/10.1126/science.290.5500.2323).

**Core mechanism:** Find a chosen number of neighbors for every item. Learn weights that combine those neighbors to approximately rebuild the item. Keep the weights fixed and find fewer coordinates where the same combinations still rebuild each point well. Centering and scale constraints stop the whole map from collapsing into a trivial answer during this step. The weights sum to one, but can include negative values. They are reconstruction instructions, not membership probabilities.

**Optional math:** For item vector $`x_i`$, choose neighbor weights $`w_{ij}`$ to reduce $`\|x_i-\sum_jw_{ij}x_j\|^2`$. Here, $`x_j`$ is a neighbor and only neighbors receive weights. Next choose smaller vectors $`y_i`$ to reduce $`\sum_i\|y_i-\sum_jw_{ij}y_j\|^2`$, keeping the required center and scale. $`W`$ is the weight matrix. An eigenproblem for $`(I-W)^\top(I-W)`$, with identity matrix $`I`$ and transpose $`\top`$, finds the coordinates after discarding the constant direction that gives every point the same value.

**Inputs/outputs and typical data types:** Inputs are closely sampled numeric observations whose local neighborhoods are roughly linear. Outputs include map coordinates and the neighbor reconstruction weights. A new-point extension can find weights using saved training neighbors and apply them to those neighbors' map positions. This estimates a local position; it is not a general backward mapping or a complete refit.

**Strengths and limitations:** LLE can capture curves without calculating shortest paths between every pair. Neighborhood size, feature scales, noise, and safeguards in the local matrix calculations matter. If many neighbors vary in too few independent directions, the weight calculation can lack a unique stable answer. Disconnected pieces create extra unconstrained directions, and standard LLE can collapse or distort parts of a map. Modified LLE, Hessian LLE, and local tangent-space alignment address different problems; they are not identical methods. Rotating or reflecting a valid map preserves the relationships it represents.

**Computational complexity / scalability notes:** Besides finding neighbors, LLE solves a small matrix problem around every point. Larger neighborhoods can greatly increase that cost. Keeping only neighbor weights saves memory. A later eigenproblem still needs repeated calculations, whose difficulty depends on numerical stability. Falling back to dense matrix methods can be expensive.

**Optional math:** With $`n`$ points, $`d`$ features, and $`k`$ neighbors, each local Gram matrix costs $`O(dk^2)`$ to build and each dense local solve costs $`O(k^3)`$. Together these give $`O(n(dk^2+k^3))`$ beyond neighbor search. Sparse $`W`$ uses $`O(nk)`$ space. Applying $`(I-W)^\top(I-W)`$ can use two sparse products without storing a larger dense matrix. Eigenproblem work depends on the iterations and conditioning, or sensitivity to small numerical changes; a dense fallback can take cubic work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn handwritten-digit comparison](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html) includes standard, modified, Hessian, and tangent-alignment versions on the same six-class, 64-feature dataset. Local pixel relationships become two-dimensional coordinates, shown with digit thumbnails to help inspect neighbors.

Unlike Isomap, LLE preserves local reconstruction recipes instead of trusting long paths that might cross different digit shapes. The source compares plots and running times. It does not establish a universal ranking or improved recognition accuracy. Labels annotate the map, rather than guide standard LLE's fit. No production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.manifold.LocallyLinearEmbedding` and its function-based API offer these variants. Report `method`, neighbor count, safeguards called regularization, and the eigenproblem solver, not just "LLE."

### 3.3.9 Self-organizing maps

**In plain English:** A self-organizing map learns representative measurement patterns arranged on a grid. Similar observations tend to use nearby grid locations, making a large collection easier to inspect.

**Name:** Self-organizing map (SOM); this entry uses the classical online Kohonen map.

**Category & sub-category:** Unsupervised neural learning in which units compete to represent an input. Neighboring grid units are encouraged to learn similar patterns.

**Originating paper/vendor/year:** Teuvo Kohonen introduced [*Self-Organized Formation of Topologically Correct Feature Maps*, 1982](https://doi.org/10.1007/BF00337288). It is a competitive network, not a modern stack of layers trained by backpropagation.

**Core mechanism:** Put a learned representative vector, or prototype, at every location on a fixed grid. For each input, find the closest prototype: the best-matching unit. Move that winner and nearby grid prototypes toward the input. Repeat across observations. Early updates often reach a broad neighborhood; later updates reach a smaller one to refine details. This encourages nearby grid positions to represent similar inputs, but does not guarantee it.

**Optional math:** There are $`U`$ prototypes $`w_j\in\mathbb R^d`$, each with $`d`$ numeric features. For input $`x`$, the winner is $`c=\arg\min_j\|x-w_j\|`$, meaning the index with smallest distance. Update prototype $`j`$ as $`w_j\leftarrow w_j+\eta_t h_{cj}(t)(x-w_j)`$. At step $`t`$, $`\eta_t`$ sets update size and $`h_{cj}(t)`$ sets how much winner $`c`$ influences grid neighbor $`j`$. The difference $`x-w_j`$ points toward the input.

**Inputs/outputs and typical data types:** Inputs include numeric feature lists, spectra, or preprocessed measurements from several channels. Outputs include prototypes, winning-unit assignments, counts per unit, and grid displays. A new item can match a saved prototype. Check its quantization error, the mismatch from that representative, rather than assuming every assigned item is familiar.

**Strengths and limitations:** SOM combines a representative summary with an organized display. Its representatives need not be actual observed items. Grid size and the way grid positions connect limit what it can show. The map can fold, leave units empty, or change with starting values. Shared neighbor updates can blur real boundaries. A location is not a uniquely meaningful hidden coordinate, a probability, or a cell-type label. Feature scales should reflect scientifically meaningful differences.

**Computational complexity / scalability notes:** Each input may need comparison with every grid prototype. A larger grid or more features increases that work. Updating fewer nearby prototypes saves update time but does not speed up an exhaustive winner search. Batch SOM versions collect observations before changing prototypes and allow different kinds of parallel work.

**Optional math:** With $`U`$ units and $`d`$ features, exhaustive winner search costs $`O(Ud)`$ per item; updating all units has the same order. For $`n`$ items and $`E`$ online passes, work is $`O(EnUd)`$. Prototypes use $`O(Ud)`$ storage beyond the input.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Ghent University's [FlowSOM cytometry vignette](https://github.com/SofieVG/FlowSOM/blob/master/vignettes/FlowSOM.Rnw), linked to [Van Gassen and colleagues' 2015 paper](https://doi.org/10.1002/cyto.a.22625), analyzes the `68983.fcs` sample. Cytometry measures properties of individual cells. The example first corrects signal spillover between channels, called compensation, and rescales values with a logicle transformation. It then uses seven marker channels and a 7-by-7 map.

Cells are assigned to map units. A separate metaclustering step groups those units, and displays help compare them with manually gated B-, T-, and NK-cell populations. Gating means selecting cells using expert-chosen measurement boundaries. The code requests ten metaclusters, **not ten SOM neurons**. Compared with an unorganized k-means display, the grid helps inspect marker relationships across representatives. This is a cell-population analysis workflow, not a clinical diagnosis accuracy result or production KPI.

**Notable vendor implementations/libraries:** R `kohonen`, MiniSom, and Bioconductor FlowSOM are available. The extra fields below explain the online Kohonen method, not every FlowSOM wrapper's specific training choices.

**Architecture diagram description:** Compare the input with all prototypes, select one winner, and use the grid to decide which prototypes move:

`d input measurements -> distances to U learned prototypes -> hard best-matching unit -> neighborhood-weighted updates on a fixed 2-D grid`

FlowSOM then adds separate grouping of map units and a minimum-spanning-tree display, which links them without loops using minimum total link distance. Those later steps are outside the SOM diagram.

**Activation functions used and why:** A negative squared distance can be a matching score: the closest prototype receives the largest score. Hard winner selection means picking one unit, not assigning class probabilities. A Gaussian neighborhood makes nearby grid units change more than distant ones.

**Optional math:** A common rule is $`h_{cj}=\exp[-\|g_c-g_j\|^2/(2\sigma_t^2)]`$. $`g_c`$ and $`g_j`$ are the winner's and another unit's grid positions; $`\sigma_t`$ is the neighborhood width at step $`t`$. The exponential makes influence fade with grid distance. These units are not ReLU hidden layers, which clip negative values to zero, or a softmax output that turns class scores into model probabilities.

**Loss function(s):** With assignments and the neighborhood held fixed, the update aims to reduce a neighbor-weighted sum of squared representation errors. In general online training, winners and neighborhoods change. Therefore the entire process is not guaranteed to follow the downhill slope of one fixed, smooth error function.

**Optimization algorithm(s):** Inputs compete for winners, and prototypes receive direct incremental updates. Adam or backpropagation is not required. Common schedules reduce both the update size, called the learning rate, and the neighborhood radius over time.

**Optional math:** Generic choices are $`\eta_t=\eta_0e^{-t/\tau_\eta}`$ and $`\sigma_t=\sigma_0e^{-t/\tau_\sigma}`$. At update $`t`$, $`\eta_t`$ is the learning rate and $`\sigma_t`$ the neighborhood width. $`\eta_0`$ and $`\sigma_0`$ are starting values; $`\tau_\eta`$ and $`\tau_\sigma`$ control how slowly they shrink. $`e`$ is the base of the natural exponential. These illustrate possible schedules, not undocumented FlowSOM settings.

**Regularization techniques:** Broad early neighbor updates encourage an organized map. Limiting grid size and reducing updates constrain how much detail it can fit. Use representation error and checks of preserved neighbor relationships to decide when to stop. Preprocessing and starting values also matter. Dropout, batch normalization, and weight decay are not built-in requirements; these other neural techniques respectively drop units during training, rescale activations, or penalize large weights.

**Backpropagation considerations:** Classical SOM does not send error derivatives backward through a network. It updates prototypes directly. Choosing one winner changes abruptly when two distances swap order, so that selection is not differentiable. Gradients fading away or growing uncontrollably are not its main training issues. Unstable winner updates and a poorly organized grid are more relevant concerns.

**Parameter count / scaling behavior:** Each map unit stores one learned number per input feature. Preprocessing and models fitted after the map have their own parameters.

**Optional math:** The basic map has $`p=Ud`$ learned coordinates, with $`U`$ units and $`d`$ features. The vignette's seven-feature, 49-unit map therefore has 343 prototype coordinates. This is arithmetic for the SOM alone, not a claimed total parameter count for all of FlowSOM.

**Training paradigm:** The SOM learns by competition without class labels. Experts may label the resulting map, or another method may cluster it afterward. Manual gating used to interpret or evaluate cells is not itself the SOM's training target.

**Hardware/parallelism considerations:** Moderate maps work on CPUs. Many prototype distances can be calculated together, and large maps may benefit from GPUs. Sequential online updates depend on input order. Batch SOM can assign observations in parallel, add their totals, and then update prototypes. Combining workers' totals requires communication roughly proportional to the codebook, the saved collection of prototypes.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| PCA | Centered numeric measurements | Best linear summary for squared reconstruction error | Large variation may not be useful variation | Summarizing POPRES genetic patterns across Europe |
| Kernel PCA | Data with a suitable similarity rule | Summarizes curved patterns | Large pairwise table; rebuilding inputs is approximate | Removing noise from USPS digit images |
| ICA | Signals formed by roughly linear mixtures | Separates statistically independent sources | Sources may not separate under real recording conditions | Reducing eye and heart contamination in MNE recordings |
| NMF | Nonnegative brightness values, counts, or amounts | Builds summaries by adding patterns | Several pattern sets may fit; fitting can get stuck | Displaying additive patterns in Olivetti faces |
| t-SNE | Many features with useful nearby-item relationships | Helps visually inspect local neighbors | Island sizes, gaps, and long distances can mislead | Plotting images from six digit classes |
| UMAP | Numeric features with meaningful neighbors | Makes flexible maps and can place new points | Neat islands are not proof of real groups | Plotting four Palmer Penguins measurements |
| Isomap | Closely sampled observations along a curved shape | Tries to preserve travel distances along the shape | Bad links create shortcuts; distance storage is large | Comparing maps of handwritten digits |
| LLE | Data that is roughly linear within small neighborhoods | Keeps neighbor reconstruction recipes | Unstable local calculations and neighbor choices matter | Comparing standard and modified digit maps |
| Self-organizing maps | Several meaningfully scaled measurements per item | Organizes learned representatives on a grid | Grid can distort; assignments are not probabilities | Exploring cell populations in a FlowSOM sample |

## 3.4 Pattern and topic discovery

Frequent-itemset methods find sets of things that often occur together, such as products in shopping baskets. Topic models instead describe documents as mixtures of themes, each represented by word frequencies. Neither tells us what caused a pattern. Two products may sell together because of season, availability, household needs, or an earlier promotion. Their association does not show what would happen if we recommended one.

An itemset is simply a set of items. Three measures help describe an association rule, such as "baskets with tea also contain biscuits":

- **Support count** is how many transactions contain the whole itemset. **Support** is that count divided by the number of transactions. For a rule, count baskets containing both its starting and ending items.
- **Confidence** asks: among baskets containing the starting items, what fraction also contain the ending items? It is an observed frequency, not a guarantee about future customers.
- **Lift** compares that co-occurrence with what we would expect if the starting and ending sets occurred independently. It adjusts for how common the ending items already are. It is not a measured sales increase.

In a **toy calculation**, 100 baskets contain tea in 20, biscuits in 30, and both in 12. Support is 0.12 because 12 of 100 contain both. Confidence for tea-to-biscuits is 0.60 because 12 of the 20 tea baskets contain biscuits. Lift is 2: the co-occurrence is twice the independence baseline. This is explanatory arithmetic, not a retailer's measured uplift or a randomized marketing experiment.

**Optional math:** Let $`n`$ be the transaction count. For itemset $`A`$, support is its count divided by $`n`$. In a rule $`A\Rightarrow B`$, $`A`$ is the antecedent, or starting set, and $`B`$ the consequent, or ending set. Both are nonempty and share no items. $`A\cup B`$ means all items from both sets.

$$
\begin{aligned}
\operatorname{support}(A\Rightarrow B)&=\operatorname{support}(A\cup B),\\
\operatorname{confidence}(A\Rightarrow B)&=
\frac{\operatorname{support}(A\cup B)}{\operatorname{support}(A)},\\
\operatorname{lift}(A\Rightarrow B)&=
\frac{\operatorname{support}(A\cup B)}
{\operatorname{support}(A)\operatorname{support}(B)}.
\end{aligned}
$$

The denominators must be nonzero. These formulas describe observed co-occurrence, not cause and effect. The [arulesViz authors document the definitions](https://github.com/mhahsler/arulesViz/blob/master/vignettes/arulesViz.Rnw).

Check rules on later or held-out transactions. Look at actual counts, not just impressive ratios. Searching many rules makes some strong-looking results likely by chance. Choose cutoffs based on repeatability and how many rules people can review. In ordinary binary itemset mining, several copies of an item in one basket count only once. Quantities, purchase order, and profit need different methods.

### 3.4.1 Apriori

**In plain English:** Apriori finds combinations of items that appear together often enough to meet a chosen cutoff. It saves work by refusing to extend combinations already known to be too rare.

**Name:** Apriori, a frequent-itemset and association-rule method.

**Category & sub-category:** Unsupervised pattern discovery. It lists common item combinations and can turn them into co-occurrence rules.

**Originating paper/vendor/year:** Rakesh Agrawal and Ramakrishnan Srikant introduced Apriori in [*Fast Algorithms for Mining Association Rules in Large Databases*, VLDB 1994](https://www.vldb.org/conf/1994/P487.PDF). The 1993 Agrawal-Imielinski-Swami paper introduced the problem and earlier algorithms, not this specific 1994 method.

**Core mechanism:** Count individual items first. Combine frequent items into candidate pairs, then continue with larger combinations. If a smaller combination is too rare, any larger combination containing it must also be too rare. Apriori removes those candidates before another scan counts the survivors. Once frequent sets are found, form rules and filter them by confidence or other measures. Confidence filtering happens after support-based pruning; it does not replace it.

**Inputs/outputs and typical data types:** Inputs are transactions, each stored as a set of item categories, often using space-saving sparse storage. Outputs are frequent sets and counts, plus optional rules with support, confidence, and lift. Continuous measurements must first be put into meaningful categories. Rescaling a measurement to a standard spread does not by itself turn it into a useful transaction item.

**Strengths and limitations:** Its logic is easy to inspect and constrain. It works well when frequent combinations are short and few candidates survive. A very low support cutoff can create huge candidate lists and require costly repeated scans. High confidence may just mean the ending item is common. High lift from very few baskets can vanish in another sample. Duplicate receipts, wrongly grouped transactions, or fields derived from an outcome can create impressive but useless rules.

**Computational complexity / scalability notes:** Work depends heavily on the number and length of candidate sets, not just transaction count. Each additional distinct item creates many possible combinations. Storage must hold candidates and outputs as well as inputs. Any method that lists all qualifying sets must spend time producing that list; generating all possible rules can be larger still.

**Optional math:** Let $`n`$ be transaction count, $`m`$ distinct item count, and $`C_\ell`$ the candidate sets of length $`\ell`$. A naive scan-and-test approach costs $`O(n\sum_\ell \ell|C_\ell|)`$, plus candidate construction. Here, $`|C_\ell|`$ counts candidates at that length. Prefix trees called tries and other counting improvements change the practical work. There can be $`2^m-1`$ nonempty frequent sets, an exponential output size.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Hahsler and Chelluboina's [arulesViz study](https://github.com/mhahsler/arulesViz/blob/master/vignettes/arulesViz.Rnw) applies Apriori to Groceries. Its [documentation](https://github.com/mhahsler/arules/blob/master/man/Groceries.Rd) describes thirty days of real checkout records: 9,835 baskets grouped into 169 product categories. The study uses support 0.001 and confidence 0.5, then plots and ranks the rules. With whole baskets as counts, that support setting requires at least ten baskets.

The output provides common combinations and possible retail questions for an analyst to investigate. Unlike a topic model, Apriori provides exact co-occurrence counts that can be checked. The source notes that high-lift rules often have low support. It reports no store intervention, sales uplift, or production KPI.

**Notable vendor implementations/libraries:** R `arules::apriori`, Weka `Apriori`, mlxtend, and SPMF offer implementations. Limits on rule length, ending-set size, or mining time can stop the search before all requested results are found.

### 3.4.2 FP-growth

**In plain English:** FP-growth compresses repeated basket patterns into a shared tree, then searches that tree for frequent combinations. It avoids many of Apriori's repeated candidate checks.

**Name:** Frequent-pattern growth, or FP-growth.

**Category & sub-category:** Unsupervised discovery of frequent itemsets using a compressed tree of transactions.

**Originating paper/vendor/year:** Jiawei Han, Jian Pei, and Yiwen Yin introduced [*Mining Frequent Patterns without Candidate Generation*, SIGMOD 2000](https://doi.org/10.1145/335191.335372). Later parallel versions and an expanded journal treatment are separate contributions.

**Core mechanism:** Count individual items and remove those below minimum support. Give frequent items a consistent order, then insert transactions into an FP-tree. Transactions with the same ordered beginning share a path, which saves space. Extra links join tree nodes holding the same item. To find larger patterns, collect paths associated with an item, build smaller conditional trees from them, and repeat. The imposed item order compresses the data; it does not describe the order in which customers acted.

**Inputs/outputs and typical data types:** Inputs are transactions with unique item identifiers and a minimum-support setting. Outputs are frequent sets and their counts. Creating association rules is a later step. A recommendation system might suggest a rule's ending items when its starting items match a basket. Whether that helps users or sales requires separate evaluation, not just frequent counts.

**Strengths and limitations:** It avoids Apriori's level-by-level candidate lists and benefits when many transactions share beginnings in the chosen order. Little sharing, very low support, or too many results can still exhaust time or memory. Even a tiny tree with one path can represent exponentially many frequent subsets. Compressing the tree or distributing work cannot remove the cost of listing all those outputs.

**Computational complexity / scalability notes:** The first pass counts transaction entries. Sorting each transaction and building the tree add work. Mining then depends on all the smaller conditional trees and the number of patterns returned. Memory holds the main tree, active smaller trees, and outputs. Parallel FP-growth, called PFP, also pays to divide and move data between workers.

**Optional math:** If $`t`$ denotes one transaction and $`|t|`$ its item count, comparison sorting by the shared global rank costs $`O(\sum_t |t|\log |t|)`$. Building the initial tree then takes work proportional to retained entries. Total mining cost cannot be described by transaction count $`n`$ alone; it depends on the conditional trees and emitted patterns.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Borgelt's 2005 FP-growth implementation study](https://borgelt.net/papers/fpgrowth.pdf) tests BMS-WebView-1 and other named datasets. The [public dataset collection](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php) identifies BMS-WebView-1 as e-commerce browsing data. The itemset task treats session items as sets; it does not preserve browsing order. Item identifiers become a compressed tree and sets of items often viewed together.

The reason to try FP-growth rather than Apriori is less repeated candidate counting. Yet the runtime results are not a universal win: Relim is slightly faster at the higher tested support settings on BMS-WebView-1. The study reports no recommendation click-through rate or production KPI. Faster mining does not establish better recommendations.

**Notable vendor implementations/libraries:** Spark ML [`FPGrowth`](https://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html), Borgelt's code, SPMF, and mlxtend are options. Spark documents a parallel PFP version. An itemset count and a rule's confidence describe different things, so do not confuse their outputs.

### 3.4.3 Eclat

**In plain English:** Eclat finds common item combinations by keeping a list of transactions for each item. It finds shared combinations by checking which transaction IDs appear in both lists.

**Name:** Eclat, a list-intersection method for frequent itemsets.

**Category & sub-category:** Unsupervised pattern discovery using a vertical format. Instead of listing items per basket, it lists baskets per item or itemset.

**Originating paper/vendor/year:** Mohammed J. Zaki, Srinivasan Parthasarathy, Mitsunori Ogihara, and Wei Li described *New Algorithms for Fast Discovery of Association Rules*, KDD 1997. The [official `arules` Eclat documentation](https://github.com/mhahsler/arules/blob/master/man/eclat.Rd) cites that work and identifies Borgelt's separate implementation.

**Core mechanism:** Store the transaction IDs containing each itemset. To extend a set, intersect lists, keeping only IDs found in both. The surviving count tells how often the larger combination appears. Explore a branch of larger sets before returning to other branches, and stop extending a set below minimum support. Bitsets and difference sets store or compare the same information more efficiently in some cases; they do not change what support means.

**Optional math:** $`T(A\cup B)=T(A)\cap T(B)`$. $`T(A)`$ is the set of transaction IDs containing all items in $`A`$, and similarly for $`B`$. The union $`A\cup B`$ contains both itemsets, so its transaction list is the intersection $`\cap`$ of their lists.

**Inputs/outputs and typical data types:** Inputs are transactions recording whether each item is present, often converted to sorted posting lists of IDs. Outputs are frequent itemsets, supports, and optionally their supporting transaction IDs. Rules can be created afterward.

As a separate **toy illustration**, ID lists $`\{1,2,4\}`$ and $`\{2,3,4\}`$ share $`\{2,4\}`$. The combined itemset therefore appears in two transactions. These are explanatory IDs, not published counts from the Adult dataset.

**Strengths and limitations:** Reusing intersections avoids many repeated scans of whole transactions, especially when the lists fit in memory. Long dense lists, many frequent extensions, or saving all supporting IDs can still be costly. How you define transactions and turn measurements into categories decides which associations can appear. Treating census income as one item among many describes associations; it does not automatically train a supervised income classifier.

**Computational complexity / scalability notes:** An intersection of sorted lists needs work proportional to their combined lengths. Bitsets pack yes/no membership into machine words, allowing many positions to be compared together. Total cost adds up across all attempted extensions. Initial lists, active search branches, and saved outputs use memory. There can still be exponentially many frequent sets.

**Optional math:** For sorted ID lists $`T_A`$ and $`T_B`$, intersection costs $`O(|T_A|+|T_B|)`$, where bars denote list length. A dense bitset for $`n`$ transactions costs $`O(n/w)`$ machine-word operations to intersect, with $`w`$ bits per word. Counting the remaining set bits can add work if done separately. Initial posting storage is proportional to transaction entries; recursion and outputs add more.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [`arules::eclat` example](https://github.com/mhahsler/arules/blob/master/man/eclat.Rd) uses [prepared UCI Adult transactions](https://github.com/mhahsler/arules/blob/master/man/Adult.Rd). After the documented conversion into categories and measurement ranges, there are 48,842 records and 115 items. It uses minimum support 0.1, maximum itemset length five, and then rule confidence 0.9. Records become ID lists, common combinations of attributes, and rules for inspection.

Shared list intersections are the reason to try Eclat rather than repeatedly scanning data with Apriori. The output describes associations, not improved benefit allocation or income prediction. Sensitive demographic attributes need careful interpretation. The example reports no policy outcome or production KPI.

**Notable vendor implementations/libraries:** R `arules::eclat`, Borgelt's Eclat/PyFIM implementations, and SPMF provide the method. Requesting supporting transaction IDs can substantially increase memory use.

### 3.4.4 Latent Dirichlet allocation

**In plain English:** Topic-model LDA describes each document as a mixture of themes, such as science and politics. It learns themes from words that tend to occur in similar documents, rather than needing topic labels.

**Name:** Latent Dirichlet allocation (LDA). It is **not** supervised linear discriminant analysis, a different method with the same abbreviation.

**Category & sub-category:** Unsupervised topic modeling with mixed membership. Bayesian prior assumptions guide how document-topic and topic-word proportions are fitted.

**Originating paper/vendor/year:** David M. Blei, Andrew Y. Ng, and Michael I. Jordan introduced [*Latent Dirichlet Allocation*, JMLR 2003](https://www.jmlr.org/papers/v3/blei03a.html).

**Core mechanism:** Treat a topic as a pattern of word probabilities and a document as a mixture of topics. The model imagines making each word occurrence by first choosing a topic from that document's mixture, then choosing a word from that topic. Fitting works backward from observed words to estimate the hidden choices and probabilities. Because an exact calculation is difficult, methods use approximations. Variational inference fits a simpler probability description; collapsed Gibbs sampling repeatedly samples hidden topic assignments. A document can belong partly to several topics, rather than one cluster.

**Optional math:** Topic $`j`$ has word probabilities $`\beta_j`$. A document has topic proportions $`\theta`$, drawn from a Dirichlet distribution, a distribution over proportions that sum to one. Each token, meaning a word occurrence, chooses a topic from $`\theta`$, then a word from that topic. Common Bayesian versions also use a Dirichlet prior to guide the topic-word probabilities before seeing all the evidence.

**Inputs/outputs and typical data types:** Inputs are documents split into tokens or a nonnegative table of word counts per document. Outputs include word probabilities for each topic, topic proportions for each document, and measures of how well the model explains the words. The standard word-generating interpretation uses raw counts. Arbitrary TF-IDF weights are not those counts and change the interpretation. For a new document, the learned topics can stay fixed while its mixture is estimated.

**Strengths and limitations:** LDA helps browse large collections whose documents mix themes. Its bag-of-words view ignores word order and grammar. Swapping topic numbers changes nothing. Shared vocabulary, different starting solutions, and prior settings can make topics hard to distinguish. A human gives a topic its readable name; that name is not a verified label learned by the model. Held-out perplexity measures how surprising test words are to the fitted model. It depends on how documents and words were held out and does not guarantee clear or useful topics.

**Computational complexity / scalability notes:** A simple sampling pass considers topic choices for every word occurrence, so more tokens and topics add work. Faster sampling methods use sparse information or lookup structures. Variational methods repeatedly update each document, so local update counts matter too. Online fitting uses smaller working batches, but still stores the full topic-by-vocabulary table.

**Optional math:** Let $`M`$ count tokens, $`n`$ documents, $`V`$ vocabulary terms, and $`k`$ topics. A straightforward collapsed Gibbs pass costs $`O(Mk)`$, with storage including $`O(M+nk+kV)`$. Sparse or alias-based samplers change these costs. A local variational step commonly needs work proportional to topic count times nonzero document-term entries. Batching does not shrink the global $`k\times V`$ topic representation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn NMF/LDA topic-extraction study](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html) uses 2,000 20 Newsgroups posts. It removes headers, signatures, and quotations and limits the vocabulary to 1,000 terms. The LDA branch uses word counts and fits ten topics with online variational learning. Bar charts show the highest-weighted words in each topic so readers can inspect themes.

Unlike k-means, the model allows several topics in one document. Unlike NMF, it explicitly describes how topic mixtures could generate word counts. The source demonstrates extracted topics but reports no human relevance score, production search benefit, or business KPI.

**Notable vendor implementations/libraries:** `sklearn.decomposition.LatentDirichletAllocation`, Gensim, MALLET, and Spark ML `LDA` provide topic models. [Supervised classical learning](01-supervised-classical.md) covers linear discriminant analysis. The two LDAs have different goals and use different training information.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Apriori | Baskets with short common combinations | Clearly rules out extensions of rare sets | Too many candidates and repeated scans | Inspecting Groceries rules with arulesViz |
| FP-growth | Transactions sharing many ordered beginnings | Reuses a compressed tree to find patterns | Smaller trees and result lists can still become huge | Mining items viewed together in BMS-WebView-1 |
| Eclat | Transactions whose item-to-basket lists fit in memory | Counts combinations by intersecting ID lists | Long transaction-ID lists can use much space | Describing attribute combinations in UCI Adult |
| Latent Dirichlet allocation | Word counts per document | Allows several themes within one document | Ignores word order; topic meanings can be unclear | Displaying topics from 20 Newsgroups posts |

## 3.5 Density and anomaly modeling

Density estimation describes where a model places more or less concentration of observations. Some outputs are properly normalized densities; others support only relative comparisons. Anomaly detection asks a different question: how unusual is this item compared with a reference group? With many features, density, rarity, typical behavior, and real-world harm need not agree. An expensive but legitimate transaction can be rare without being fraud.

**Outlier detection** usually fits an unlabeled collection that may already contain unusual items, then ranks items in that same collection. **Novelty detection** learns from a reference collection assumed to be mostly normal and scores later observations. Selecting normal reference data supplies some outside information. For that reason, this is sometimes called semi-supervised anomaly detection. That use of "semi-supervised" differs from classification using a few labeled examples and many unlabeled ones. See the [semi-supervised volume](03-semi-supervised.md) and the [official explanation of novelty versus outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html).

**Thresholds are decisions, not discoveries.** First check whether larger or smaller scores mean more unusual. Choose the flagging cutoff using held-out normal records, expert-checked anomalies, review capacity, and the costs of false alarms and missed problems. A cutoff at the empirical 99th percentile is exceeded by roughly 1% of its calibration reference. It does not guarantee that future normal data will produce that false-alarm rate.

A `contamination` setting usually chooses a score cutoff. It neither verifies the true fraction of anomalies nor turns scores into probabilities. Say when labels help choose the cutoff or model settings, even if fitting itself is unsupervised. Watch for changes in incoming data. Do not evaluate with the same known anomalies used to tune the cutoff.

### 3.5.1 Kernel density estimation

**In plain English:** KDE places a small smooth bump around each observation and adds the bumps. The result shows crowded and sparse areas without forcing the data into one bell-shaped cloud.

**Name:** Kernel density estimation, or KDE.

**Category & sub-category:** Unsupervised density estimation using local smoothing rather than a fixed number of fitted components. Density can also help score unusual items.

**Originating paper/vendor/year:** Foundational papers are Murray Rosenblatt's [*Remarks on Some Nonparametric Estimates of a Density Function*, 1956](https://doi.org/10.1214/aoms/1177728190), and Emanuel Parzen's [*On Estimation of a Probability Density Function and Mode*, 1962](https://doi.org/10.1214/aoms/1177704472).

**Core mechanism:** Choose a bump shape, called a kernel, and a width, called the bandwidth. Put a bump at every observed value, then average their contributions at locations of interest. Nearby observations create higher combined density. In a properly normalized model, adding density over a region gives that region's probability. A density value at one exact point is not the probability of that point.

**Optional math:** For a normalized Euclidean kernel $`K`$ and scalar bandwidth $`h`$,

$$
\hat p(x)=\frac{1}{nh^d}\sum_{i=1}^{n}K\!\left(\frac{x-x_i}{h}\right).
$$

Here, $`x`$ is a query location, $`x_i`$ are the $`n`$ observations, and $`d`$ is the feature count. $`\hat p(x)`$ is the estimated density. The divisor corrects for the number and width of the bumps. A bandwidth matrix allows different smoothing widths and directions instead of one width everywhere. Its determinant supplies the corresponding volume correction.

**Inputs/outputs and typical data types:** Inputs are continuous measurements or coordinates, especially with few features. Outputs are density estimates, logarithms of density, and, for some kernels, generated samples. A separate cutoff can turn density-based unusualness into flags. Rescaling features changes both neighborhoods and density units. Transforming a density back requires a Jacobian correction, which accounts for how the transformation stretches or shrinks space.

**Strengths and limitations:** KDE is flexible and does not require choosing a count of Gaussian components. But bandwidth largely decides the result. Narrow bumps can overfit individual observations; wide bumps can erase separate peaks. Select bandwidth on held-out data or with each scored point left out of its own fit. Otherwise that point's own narrow bump can look falsely impressive. Boundaries distort bumps, rare tails provide little evidence, and many features leave most combinations sparsely sampled. Low estimated density can reflect how data was collected, not a real problem.

**Computational complexity / scalability notes:** Direct scoring compares each query with every saved observation. Tree indexes can help with a few features, sometimes using a controlled approximation. With many features, their advantage may disappear. Trying several bandwidths adds work. Grid methods using a fast Fourier transform (FFT) need suitable grids and kernels, and their grid sizes can grow badly with dimension.

**Optional math:** With $`n`$ reference observations and $`d`$ features, storage and one direct query each cost $`O(nd)`$. For $`G`$ query locations, direct work is $`O(Gnd)`$. Naive leave-one-out bandwidth evaluation costs $`O(n^2d)`$ per candidate bandwidth. Search acceleration depends on the data and permitted approximation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn species-KDE example](https://scikit-learn.org/stable/auto_examples/neighbors/plot_species_kde.html) uses recorded locations of *Bradypus variegatus* and *Microryzomys minutus* from the Phillips and colleagues dataset. Latitude and longitude are expressed in radians. Each species gets a separate KDE using Haversine distance, a distance rule for points on a sphere. The output maps concentrations of observations in South America.

Unlike one Gaussian cloud, KDE can show several geographic concentrations. The code explicitly does not learn environmental suitability. These are smoothed occurrence maps, not validated habitat predictions. Also, the [KDE API guarantees density normalization only for Euclidean distances](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html). The maps must not be called calibrated probabilities per square kilometre. No conservation outcome or production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.neighbors.KernelDensity`, SciPy `gaussian_kde`, R `density`, and statsmodels offer KDE tools. Their bandwidth definitions and support for several features differ.

### 3.5.2 Isolation Forest

**In plain English:** Isolation Forest flags items that are easy to separate from others with random splits. A high unusualness score means "worth checking," not "proved fraud" or another confirmed problem.

**Name:** Isolation Forest, an ensemble of random isolation trees.

**Category & sub-category:** Unsupervised anomaly scoring based on how quickly random partitions separate observations.

**Originating paper/vendor/year:** Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou introduced [*Isolation Forest*, ICDM 2008](https://doi.org/10.1109/ICDM.2008.17), then published expanded work on isolation-based anomaly detection.

**Core mechanism:** Build many trees, each using a sample of the data. At each branch, choose a feature and randomly split its observed range. Keep splitting the resulting groups. An item separated after few splits is easier to isolate and receives a higher score under the original convention. Average the behavior across trees to reduce reliance on one random partition. Libraries may reverse the score sign, so check which direction means more unusual.

**Optional math:** The original normalized score is $`2^{-\mathbb E[h(x)]/c(\psi)}`$. $`x`$ is the scored item, $`h(x)`$ its path length, and $`\mathbb E`$ averages over trees. Path length includes an adjustment when a leaf still contains several observations. $`\psi`$ is the sample size per tree, and $`c(\psi)`$ adjusts the path scale for that size. Shorter adjusted paths yield larger scores.

**Inputs/outputs and typical data types:** Inputs are numeric tables or categories encoded in a suitable numeric form. Outputs are unusualness scores and, after a cutoff is chosen, flags. Saved trees can score new observations without rebuilding a neighbor graph. They do not use task labels or choose splits to make labeled classes purer, as supervised trees do.

**Strengths and limitations:** Small bounded samples make the method practical at scale. It can find unusual combinations without estimating a full density. But a dense group of anomalies may escape, while harmless rare items may be flagged. Splits follow feature axes, so rotating the feature coordinates can change results.

Changing units by a positive scale factor and shifting the zero point need not change ideal random-split partitions. Nonlinear transformations and arbitrary number codes for categories can change them. Standardizing every feature is therefore less central than for LOF, but feature design still matters. Scores are not calibrated probabilities of anomalies.

**Computational complexity / scalability notes:** Keeping each tree's sample and depth limited keeps fitting and scoring relatively inexpensive. Loading and preparing the entire input is still a separate cost. Trying more features per branch or allowing extremely deep, unbalanced trees changes the estimate.

**Optional math:** For $`F`$ trees and sample size $`\psi`$, fitting is about $`O(F\psi\log\psi)`$ when depths are capped or balanced and feature-selection costs are fixed. Model storage is $`O(F\psi)`$. Depth-capped scoring costs $`O(F\log\psi)`$ per observation, apart from input preparation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn outlier-estimator study](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html) uses KDDCup99's SA subset of network connections. The displayed sample has 10,065 records and 338 injected-attack anomalies. Categories in connection records are encoded before Isolation Forest fits without labels. Its scores rank suspicious connections.

The study reports slightly better ROC-AUC than its configured LOF on this dataset. ROC-AUC measures how well scores rank anomalies above comparison records across possible cutoffs; higher is better under the same test setup. It is not a percent-correct score. Cheap random partitions motivate the method at this sample size. However, known labels guide sampling, and evaluation scores the same contaminated collection used for fitting, not a future stream. The attacks were generated in a controlled network. No live intrusion-detection or financial-loss KPI is established.

**Notable vendor implementations/libraries:** `sklearn.ensemble.IsolationForest` and tools such as `isotree` provide implementations. Extended or hyperplane versions use differently oriented splits and are not the original axis-aligned forest.

### 3.5.3 Local outlier factor

**In plain English:** LOF checks whether an item sits in a much sparser neighborhood than its neighbors do. It can flag locally unusual items even when one global rarity cutoff would miss them.

**Name:** Local outlier factor, or LOF.

**Category & sub-category:** Unsupervised anomaly scoring that compares an item's local crowding with its neighbors' crowding.

**Originating paper/vendor/year:** Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jorg Sander introduced [*LOF: Identifying Density-Based Local Outliers*, SIGMOD 2000](https://doi.org/10.1145/335191.335388).

**Core mechanism:** Find neighbors around each observation. Measure distances with a lower limit based on each neighbor's own neighborhood size, so an extremely close pair does not dominate. Short average adjusted distances mean high local crowding. Compare the neighbors' average density with the item's own density. A ratio near one means similar local crowding; a much larger ratio suggests that the item is unusually isolated nearby.

**Optional math:** From item $`x`$ to neighbor $`o`$, reachability distance is $`\max\{k\text{-distance}(o),D(x,o)\}`$. $`D(x,o)`$ is their distance and $`k\text{-distance}(o)`$ reaches $`o`$'s $`k`$-th neighbor. Local reachability density is one divided by the mean of those adjusted distances. LOF divides average neighbor density by the item's density. Distance ties can make the mathematical neighborhood contain more than exactly $`k`$ observations.

**Inputs/outputs and typical data types:** Inputs are feature vectors or suitable distances. Outputs are relative-density scores and flags after choosing a cutoff for the fitted collection. Rescaling numbers and encoding categories can strongly change neighbors. Giving categories an arbitrary integer order can therefore mislead LOF. For training samples, scikit-learn returns the negative of the conventional outlier factor.

**Strengths and limitations:** LOF can notice a sparse point beside a dense group even if that point is not globally rare. The neighborhood size decides which population it is compared with. Tiny neighborhoods are noisy; broad ones blur local structure. A dense group of anomalies can support itself and escape detection. A score near one or a chosen contamination cutoff does not prove that an item is normal, safe, or legitimate.

**Computational complexity / scalability notes:** Finding neighbors usually dominates. A full pairwise search can roughly quadruple its work when the item count doubles. Good indexes help with few features, but often lose their advantage with many. Only the chosen neighbors need be stored, not necessarily every pair. Approximate searches can change scores, so check them especially near the flagging cutoff.

**Optional math:** With $`n`$ items, $`d`$ features, and $`k`$ neighbors, brute-force search costs $`O(n^2d)`$. Reachability and ratio calculations then take roughly $`O(nk)`$. Neighbor indices and distances use $`O(nk)`$ storage.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn Ames Housing experiment](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html) deliberately constructs an anomaly task. It divides sale price by **lot area, not living-floor area**. It keeps ratios below 40 or above 70 USD per square foot, leaving 2,714 records and thirty high-ratio records used as stand-ins for anomalies. Encoded housing features become LOF rankings.

The study reports better ROC-AUC than its Isolation Forest configuration here. ROC-AUC measures ranking across cutoffs, not the percentage of verified fraud correctly found. Local comparisons are useful to consider because properties differ widely. Labels help construct the test and set the neighborhood scale, so model selection is not fully label-blind. These flags do not establish fraud or appraisal errors. No production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.neighbors.LocalOutlierFactor`, ELKI, and R `dbscan::lof` provide LOF. In scikit-learn, `novelty=True` scores new points against a frozen reference. [Do not use that prediction path on training data and treat it as equivalent to `fit_predict`](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-with-lof): they use different scoring setups.

### 3.5.4 One-class SVM

**In plain English:** A one-class SVM learns a boundary around a reference set, such as mostly normal observations. It can flag new items outside that boundary without needing examples of every possible anomaly.

**Name:** One-class support vector machine, or one-class SVM.

**Category & sub-category:** Unsupervised or one-class novelty detection. It estimates a region occupied by reference data, rather than learning from labeled examples of two opposing classes.

**Originating paper/vendor/year:** Bernhard Scholkopf, John C. Platt, John Shawe-Taylor, Alexander J. Smola, and Robert C. Williamson published [*Estimating the Support of a High-Dimensional Distribution*, Neural Computation 2001](https://doi.org/10.1162/089976601750264965), following earlier novelty-detection work.

**Core mechanism:** Supply reference observations, not paired lists of normal and abnormal examples. Use a kernel similarity rule to fit a boundary that contains much of the reference distribution while allowing some training exceptions. A common radial basis function (RBF) kernel allows a curved boundary without explicitly making all the expanded features. New items receive scores relative to that boundary. No labeled negative class is required.

**Optional math:** The fitting problem is

$$
\min_{w,\rho,\xi\ge0}\frac12\|w\|^2+\frac{1}{\nu n}\sum_i\xi_i-\rho,
\qquad
w^\top\phi(x_i)\ge\rho-\xi_i.
$$

Here, $`x_i`$ is reference item $`i`$, $`n`$ is reference size, and $`\phi`$ is the feature mapping supplied implicitly by the kernel. $`w`$ holds learned weights, $`\rho`$ sets the boundary level, and each nonnegative $`\xi_i`$ allows a training exception. $`\nu`$ controls the trade-off. The first term limits weight size, the second penalizes exceptions, and the last adjusts the boundary. The decision score is $`f(x)=w^\top\phi(x)-\rho`$, with $`\top`$ denoting transpose. Its sign locates item $`x`$ relative to the learned boundary; it is not a probability.

**Inputs/outputs and typical data types:** Inputs are numeric reference records, a kernel and its scale, and the setting nu, written $`\nu\in(0,1]`$: greater than zero and at most one. Outputs include support vectors, the training points used to define the decision rule, plus scores and novelty flags. Learn feature-normalization settings consistently from reference data. The RBF bandwidth can substantially change the fitted boundary.

**Strengths and limitations:** It helps when you have representative normal examples but cannot gather representative anomalies. A contaminated or unrepresentative reference set can distort the boundary. Choosing a kernel and checking a cutoff against separate data remain essential. Scores are neither normalized densities nor calibrated probabilities.

**Optional math:** Under the standard solution conditions, $`\nu`$ is an upper bound on the fraction of training margin errors and a lower bound on the fraction of support vectors. Margin errors are training examples that fail the model's required boundary margin. These bounds do not measure real anomaly prevalence or guarantee a future false-alarm rate.

**Computational complexity / scalability notes:** A full table of pairwise kernel comparisons can make memory and preparation costs grow roughly fourfold when record count doubles. Fitting then solves a constrained quadratic optimization problem. Its steps depend on the solver, required precision, and numerical stability; there is no universal cubic total-time guarantee. A limited kernel cache saves memory by recalculating some comparisons. Linear and finite-feature approximations change both cost and accuracy.

**Optional math:** For $`n`$ records and $`d`$ features, a dense RBF Gram matrix costs $`O(n^2d)`$ to build and $`O(n^2)`$ to store. With $`s`$ support vectors, dense RBF scoring costs $`O(sd)`$ per query. This does not include every possible solver iteration during fitting.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn species-distribution example](https://scikit-learn.org/stable/auto_examples/applications/plot_species_distribution_modeling.html) uses observed locations and fourteen environmental features for *Bradypus variegatus* and *Microryzomys minutus*. For each species, the training features come only from places where it was observed. Those features are standardized, then separate RBF models use `gamma=0.5` and `nu=0.1`. Environmental grid cells receive relative suitability scores. This is scikit-learn's model on the Phillips dataset, not a claim that Phillips's original method was one-class SVM.

The displayed ROC-AUCs are 0.868443 and 0.993919. ROC-AUC measures how well the scores rank held-out occurrences above comparison locations, not percent-correct habitat predictions. The comparison uses 10,000 randomly sampled grid-background points with seed thirteen. **Background locations are not verified absences.** The code also gives ocean cells low map scores. These results depend on that presence/background test protocol and include no controlled alternative-model comparison or production KPI. The reason to use a one-class method rather than a binary classifier is to avoid inventing negative training examples.

**Notable vendor implementations/libraries:** LIBSVM, `sklearn.svm.OneClassSVM`, and R `e1071` offer kernel one-class SVMs. `sklearn.linear_model.SGDOneClassSVM` is a distinct linear method using stochastic updates.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Kernel density estimation | Continuous measurements with few features | Smoothly shows local concentrations | Bump width matters; many features leave sparse evidence | Mapping South American species observations |
| Isolation Forest | Large tables with suitable numeric encodings | Quickly scores how easily items separate | Dense anomaly groups may hide; features affect splits | Ranking KDDCup99 SA network connections |
| Local outlier factor | Data with meaningful neighbors and varied crowding | Compares unusualness with nearby items | Neighbors matter; old-cohort and new-point scoring differ | Ranking constructed Ames Housing price-ratio anomalies |
| One-class SVM | A mostly normal numeric reference collection | Learns a curved boundary without negative examples | Bad reference data, kernel settings, and fitting cost matter | Scoring species suitability from presence-only training data |

## Coverage and continuation manifest

- **3.1.1-3.1.6:** Six methods that use averages, representatives, clouds, or merge trees: k-means, mini-batch k-means, k-medoids/PAM, Gaussian mixtures with EM, agglomerative clustering, and BIRCH.
- **3.2.1-3.2.6:** Six methods based on crowding or links: DBSCAN, OPTICS, HDBSCAN, spectral clustering, mean shift, and affinity propagation.
- **3.3.1-3.3.9:** Nine ways to represent data: PCA, kernel PCA, ICA, NMF, t-SNE, UMAP, Isomap, LLE, and SOM. SOM retains all nine extra neural-network fields. The other entries explain their representative non-neural versions.
- **3.4.1-3.4.4:** Four methods for recurring patterns or document themes: Apriori, FP-growth, Eclat, and latent Dirichlet allocation.
- **3.5.1-3.5.4:** Four methods for density or unusualness: KDE, Isolation Forest, LOF, and one-class SVM.

**Total: 29 algorithm entries and five category comparison tables.** Each worked example names a public study, documented application, or dataset. Extra toy calculations are labeled and are not commercial results. A catalogue count, a demonstration setting, and a score from one test protocol do not measure the same thing as impact in a deployed system.

Continue with [neural unsupervised and self-supervised learning](05-unsupervised-neural.md) and [foundation-model pretraining](06-foundation-models.md). A Gaussian mixture's probability clouds are not automatically neural experts chosen to handle particular inputs. The [MoE model catalogue](07-moe-models.md) and [dedicated MoE deep dive](08-moe-deep-dive.md) explain expert routing, capacity, and training. Related chapters cover [semi-supervised learning](03-semi-supervised.md), [supervised classical learning](01-supervised-classical.md), the [comparative guide](09-comparative-guide.md), and the [glossary](10-glossary.md).

This first-edition chapter has a limited scope. It does not claim to cover every unsupervised method. Further topics not developed here include:

- Robust and sparse PCA, which change how outliers or feature choices affect a summary, and full treatments of factor analysis.
- Other ways to build maps, including diffusion maps and standalone MDS, plus clustering families CURE/ROCK/CHAMELEON.
- Dirichlet-process and Bayesian nonparametric mixtures, which allow more flexible component collections; subspace clustering, which can focus on selected feature directions; and co-clustering, which groups rows and columns together.
- Maintaining density clusters as streams change, rather than just fitting a fixed dataset.
- Closed and maximal pattern mining, which reduce redundant pattern outputs; high-utility mining, which accounts for value; and sequential mining, which includes order.
- Dynamic and correlated topic models, which handle changing or related themes.
- Extreme-value tail calibration, which uses models of rare extremes to help set cutoffs, and formal guarantees for approximate neighbor searches.

Neural extensions of classical methods need their own explanations of network architecture and training information. They cannot simply inherit the claims made for the versions in this chapter.
