# 3. Unsupervised Learning Algorithms: Classical Methods

This volume covers classical clustering, representation learning, pattern discovery, density estimation, and anomaly detection. Its representative training signals come from the observations themselves: distances, reconstruction, likelihood, neighborhood relationships, or co-occurrence. Labels used to color a plot or evaluate an experiment do not necessarily train its model. Conversely, selecting a cohort, initializing from known classes, or tuning against labels introduces information that must be disclosed. One-class training on a reference population is not ordinary supervised binary classification.

**Evidence policy: 2026-09-08.** Examples below distinguish scientific applications and research demonstrations from commercial deployments. Named public datasets count as concrete research examples, not evidence that a vendor deployed the algorithm. Published qualitative findings are reported as such; absence of a production KPI is explicit. Technical reasons to prefer a method are this book's analysis unless attributed to the source. Versioned documentation links identify the example inspected, not the newest software release.

Here, $`n`$ is the number of observations, $`d`$ the input feature count, $`k`$ the number of clusters or neighbors as locally specified, $`I`$ the number of solver iterations, $`E`$ the number of complete passes, and $`B`$ a batch size. Reduced dimension is $`r`$. Complexity statements distinguish distance construction, optimization, storage, and prediction; they are not promises about convergence or wall-clock performance.

Several cautions apply throughout:

- **Geometry is a modeling decision.** Changing currency units, standardizing features, normalizing documents, or choosing a geographical rather than Euclidean metric changes what "similar" means. Fit preprocessing on training data for predictive evaluation. Fitting on the entire cohort is legitimate for a descriptive analysis, but is a different protocol.
- **A partition is not a discovered ground truth.** Compactness, connectivity, density, and likelihood define different cluster concepts. Assess stability under resampling and reasonable preprocessing changes, inspect original features, and use external domain evidence. A silhouette score favors its chosen geometry; it cannot establish that a cluster is a biological species or a useful customer segment.
- **Non-identifiability is often intrinsic.** Relabeling clusters does not change a solution. Component signs, rotations within repeated-eigenvalue subspaces, and factor rescalings can change a representation without changing its fitted reconstruction or distribution. Compare invariant quantities rather than component numbers across runs.
- **A fitted coordinate map need not generalize.** Some methods learn an explicit transformation; others jointly arrange the training observations. An implementation's out-of-sample extension may freeze an old graph or interpolate locally rather than solve the original problem on the expanded dataset.

See the [reading guide](00-reading-guide.md) for the book's supervision taxonomy and [neural unsupervised learning](05-unsupervised-neural.md) for learned nonlinear encoders.

## 3.1 Centroid, prototype, mixture, and hierarchical clustering

These methods summarize observations with representatives, probability distributions, or nested groups. A centroid need not be an observed object; a medoid must be. A Gaussian component describes a distribution rather than necessarily a semantic class. A dendrogram records a sequence of irreversible merges, not a uniquely correct taxonomy.

### 3.1.1 k-means

**Name:** k-means, represented by Lloyd-style alternating minimization.

**Category & sub-category:** Unsupervised learning; centroid-based partitioning and vector quantization.

**Originating paper/vendor/year:** MacQueen's 1967 work is a foundational source for the k-means name; Lloyd's published formulation is [*Least squares quantization in PCM*, 1982](https://doi.org/10.1109/TIT.1982.1056489). These are historical algorithmic contributions, not vendor inventions.

**Core mechanism:** Minimize $`J=\sum_i\min_{1\le j\le k}\|x_i-\mu_j\|_2^2`$. Alternate nearest-centroid assignments with arithmetic-mean updates. Each exact step does not increase the objective, but the result depends on initialization and need not be globally optimal. Multiple starts and k-means++ seeding reduce avoidable initialization failures. Cluster-number selection remains external: lower training inertia alone always favors adding representatives.

**Inputs/outputs and typical data types:** Numeric vectors, including suitably encoded images and sparse text; outputs are $`k`$ centroids, assignments, and squared-distance summaries. A new observation has a natural nearest-centroid assignment. Standardization weights features differently; unit-length normalization changes the problem toward directional similarity but does not make ordinary k-means identical to every spherical-k-means formulation.

**Strengths and limitations:** Simple, interpretable, and effective when squared Euclidean distortion is meaningful. Its Voronoi cells favor compact, roughly isotropic groups; equal cluster size is not a hard constraint. Elongated structures, unequal dispersions, outliers, and irrelevant high-dimensional features can produce misleading partitions. Cluster labels are permutation-invariant, and different local minima can have similar inertia. A mean-colored pixel is useful even when no photographed object has that exact color.

**Computational complexity / scalability notes:** A dense Lloyd assignment/update iteration costs $`O(nkd)`$; $`I`$ iterations cost $`O(Inkd)`$, excluding initialization and restarts. A straightforward implementation stores the data plus $`O(n+kd)`$ working state. Distance-bound accelerations can use additional $`O(nk)`$ memory. Prediction costs $`O(kd)`$ per observation. Neither a fixed iteration count nor globally optimal convergence follows from this accounting.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn 1.5 Summer Palace color-quantization example](https://scikit-learn.org/1.5/auto_examples/cluster/plot_color_quantization.html) addresses image representation. Its named `china.jpg` photograph has 96,615 distinct colors. RGB values scaled to $`[0,1]`$ feed a 64-centroid model fitted on 1,000 sampled pixels; nearest-centroid indices then reconstruct the entire image using a 64-color palette. The documented result preserves the overall appearance and includes a random-palette comparison. The technical rationale is direct optimization of color distortion rather than arbitrary palette selection, not discovery of image objects. No perceptual-quality score, deployed compression service, or production KPI is reported; palette size is not itself a measured file-compression ratio.

**Notable vendor implementations/libraries:** `sklearn.cluster.KMeans`, SciPy's vector-quantization routines, and Spark ML `KMeans`. Availability in a library does not establish product adoption.

### 3.1.2 Mini-batch k-means

**Name:** Mini-batch k-means.

**Category & sub-category:** Unsupervised learning; stochastic centroid clustering for large or streamed datasets.

**Originating paper/vendor/year:** D. Sculley, [*Web-scale k-means clustering*, WWW 2010](https://doi.org/10.1145/1772690.1772862). Online vector quantization predates this particular mini-batch formulation.

**Core mechanism:** Approximate the same squared-distortion objective as k-means using small sampled batches. Assign batch points to their closest centroids and update each centroid using accumulated assignment counts. A streaming-average update has learning rate $`1/c_j`$ after the $`c_j`$-th assignment to centroid $`j`$. Implementations add initialization samples, stopping heuristics, and reassignment of poorly used centers. This is not an exact full-data Lloyd iteration.

**Inputs/outputs and typical data types:** Dense numeric batches or sparse feature matrices; outputs are centroids and optionally full-data labels. New observations use nearest-centroid prediction. Online input is possible, but an ever-shrinking learning rate is not automatically suitable for concept drift: forgetting or periodic refitting changes the training policy.

**Strengths and limitations:** Reduces working-set size and can reach a useful quantizer without repeatedly scanning all observations. Small or unrepresentative batches increase variance and can miss rare groups. Initialization, data order, and dead-center handling matter. Sparse input does not imply sparse centroids. Re-normalizing latent text vectors after truncated SVD can improve the geometry, but the SVD and normalization are separate modeling steps and costs.

**Computational complexity / scalability notes:** One dense batch update costs $`O(Bkd)`$; $`I`$ updates cost $`O(IBkd)`$. Streaming storage can be $`O(Bd+kd+k)`$, excluding retained input. Computing final labels or exact inertia over all observations adds $`O(nkd)`$. Full-data ingestion and preprocessing may dominate; a mini-batch solver is not necessarily faster on a small dataset or at equal final objective quality.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn document-clustering study](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html) analyzes 3,387 posts from four named 20 Newsgroups categories: atheism, religion, graphics, and space. It strips headers, signatures, and quoted replies, builds TF-IDF vectors, reduces them to 100 LSA dimensions, normalizes them, and fits four mini-batch centroids with batch size 1,000. Document vectors become group assignments for topic inspection, not authoritative topic labels. The example reports greater run-to-run variability than full k-means on this small corpus. Our rationale for mini-batches is scaling to a larger document collection, not a claimed advantage in this particular experiment. Labels do not enter centroid updates, but their known category count sets the requested four clusters and they support evaluation. LSA fitting time must not disappear from comparisons. No production search-quality or cost KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.cluster.MiniBatchKMeans`, including `partial_fit`. Distributed k-means implementations should not be assumed to implement Sculley's mini-batch update merely because they process partitions.

### 3.1.3 k-medoids and PAM

**Name:** k-medoids; Partitioning Around Medoids (PAM) is the representative optimizer.

**Category & sub-category:** Unsupervised learning; exemplar-based partitioning using dissimilarities.

**Originating paper/vendor/year:** Kaufman and Rousseeuw's medoid work includes their 1987 formulation and the detailed PAM treatment in *Finding Groups in Data*, 1990. The [R `cluster::pam` manual](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html) identifies the original authors and distinguishes later FastPAM/FasterPAM improvements.

**Core mechanism:** Select observed representatives $`M`$, $`|M|=k`$, minimizing $`\sum_i\min_{m\in M}D(x_i,m)`$. Original PAM builds an initial medoid set, then searches swaps between medoids and non-medoids until no single improving swap remains. Alternating within-cluster medoid updates are a different, cheaper optimizer. FasterPAM changes swap evaluation and execution; it does not turn the objective into k-means.

**Inputs/outputs and typical data types:** Numeric vectors or a precomputed dissimilarity matrix. With an appropriate dissimilarity, mixed categorical/continuous records, strings, and structured objects are possible. Outputs include actual exemplar indices, cluster assignments, and costs. Assigning new data requires distances to the saved medoids; an arbitrary training-only distance matrix provides no such distances automatically.

**Strengths and limitations:** Observed exemplars are interpretable, and unsquared dissimilarities can reduce extreme-point influence relative to squared-error k-means. Robustness is not immunity to contamination or a badly chosen metric. Distances must be constructed thoughtfully: mixed-data weighting, missingness, and units can determine the result. The objective remains nonconvex, $`k`$ must be selected, and label permutations are immaterial.

**Computational complexity / scalability notes:** If one distance costs $`C_D`$, materializing all distances costs $`O(n^2C_D)`$ time and $`O(n^2)`$ memory. Original BUILD costs $`O(kn^2)`$ using cached distances; a cached best/second-best SWAP sweep costs $`O(kn(n-k))`$. Multiply sweeps by their actual count. FasterPAM can reduce a sweep to quadratic order, while CLARA uses subsampling and changes the accuracy/scalability trade-off.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [R `cluster` agriculture example](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/agriculture.html) uses Eurostat's 1993 observations for twelve European Union countries: per-capita GNP and agricultural-workforce share. It explicitly fits `pam(agriculture, 2)` and plots the partition. The documentation describes a more-agricultural group containing Greece, Portugal, Spain, and Ireland. Country records become two exemplar-based groups for socioeconomic exploration. Our rationale versus k-means is that the representative is a real country rather than an invented average economy. The example's default unstandardized metric makes units consequential; it is not a recommendation to infer policy from this partition. No policy intervention, causal effect, or production KPI is reported.

**Notable vendor implementations/libraries:** R `cluster::pam`; `sklearn_extra.cluster.KMedoids` with explicit `method="pam"`; ELKI. In scikit-learn-extra, the documented default `method="alternate"` is not PAM.

### 3.1.4 Gaussian mixtures with expectation-maximization

**Name:** Gaussian mixture model (GMM), fitted by expectation-maximization (EM).

**Category & sub-category:** Unsupervised learning; probabilistic clustering and parametric density estimation.

**Originating paper/vendor/year:** Gaussian-mixture modeling has nineteenth-century roots, including Pearson's 1894 mixture analysis. Dempster, Laird, and Rubin formalized the general EM framework in [*Maximum Likelihood from Incomplete Data via the EM Algorithm*, 1977](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x); they did not invent every mixture model.

**Core mechanism:** Model $`p(x)=\sum_{j=1}^k\pi_j\mathcal N(x\mid\mu_j,\Sigma_j)`$. The E-step computes responsibilities proportional to $`\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)`$; the M-step updates weights, means, and covariances using these fractional memberships. Exact EM gives nondecreasing likelihood; under suitable regularity conditions it can converge to a stationary solution rather than the global maximum. Full, tied, diagonal, and spherical covariance restrictions represent different assumptions.

**Inputs/outputs and typical data types:** Continuous vectors; outputs include component parameters, log densities, responsibilities, and optional hard assignments. New points can be scored directly. Responsibilities are posterior component probabilities under the fitted model, not automatically calibrated probabilities of meaningful classes.

**Strengths and limitations:** Represents overlapping ellipsoidal groups and uncertainty more naturally than hard k-means. Component permutations leave the density unchanged. Identical or redundant components add further ambiguity, and one semantic population may require several Gaussians. Unconstrained likelihood can diverge when a component collapses onto a point; covariance floors or priors are substantive safeguards. Feature scaling, covariance structure, restarts, and component count matter. BIC uses a fit/complexity trade-off, conventionally with smaller values preferred, but mixture singularities complicate textbook asymptotic interpretations.

**Computational complexity / scalability notes:** A full-covariance EM iteration commonly costs $`O(nkd^2+kd^3)`$, including density evaluation, covariance updates, and factorization. Diagonal covariance reduces the leading data-dependent work to $`O(nkd)`$. Storage can include $`O(nk)`$ responsibilities and $`O(kd^2)`$ covariance parameters in addition to data. Batch sufficient-statistic accumulation can reduce responsibility storage. Convergence count and covariance regularization materially affect runtime.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** In astronomy, [Hunt and Reffert's 2023 Gaia DR3 cluster-catalogue study](https://ar5iv.labs.arxiv.org/html/2303.13424) uses Gaussian mixtures to split residual overlapping cluster candidates, including Collinder 394/NGC 6716 and UBC 76/UBC 77. Astrometric member distributions feed mixture fits; component assignments help revise the catalogue after initial HDBSCAN detection. The paper explicitly identifies close pairs that hard density clustering could not separate satisfactorily. Our technical rationale is soft overlapping components versus a single hard connected group. This is a downstream mixture application, not a claim that EM alone produced the catalogue; detailed solver settings are not established here. The result is corrected scientific membership structure, with no business KPI or universal binary-cluster accuracy reported.

**Notable vendor implementations/libraries:** `sklearn.mixture.GaussianMixture`, R `mclust`, and Spark ML `GaussianMixture`. Covariance families and regularization differ between implementations.

### 3.1.5 Agglomerative hierarchical clustering

**Name:** Agglomerative hierarchical clustering; single, complete, average, and Ward linkage.

**Category & sub-category:** Unsupervised learning; bottom-up hierarchical grouping.

**Originating paper/vendor/year:** A family with multiple historical contributors rather than one originating implementation. Ward's variance-based criterion is documented in [*Hierarchical Grouping to Optimize an Objective Function*, 1963](https://doi.org/10.1080/01621459.1963.10500845).

**Core mechanism:** Start with singleton clusters and repeatedly merge the closest pair under a linkage rule. Single linkage uses the minimum cross-cluster distance, complete linkage the maximum, and average linkage the mean. Ward chooses the smallest increase in within-cluster squared error, $`\Delta(A,B)=|A||B|\|\mu_A-\mu_B\|^2/(|A|+|B|)`$. Ward's Euclidean variance interpretation must not be applied indiscriminately to arbitrary dissimilarities.

**Inputs/outputs and typical data types:** Feature vectors or appropriate pairwise dissimilarities, optionally with a graph constraining allowed merges. Outputs are a merge tree, merge heights, and partitions obtained by a cut. Most standard implementations are transductive: inserting a new point can change the tree. Attaching it to the nearest existing cluster is an added prediction rule, not the original algorithm.

**Strengths and limitations:** Supports inspection at several resolutions without refitting for every $`k`$. Linkage choice changes the meaning of a group: single linkage can chain through bridges; complete linkage resists large diameters; Ward favors low-variance regions. Early mistakes cannot be undone. Ties and connectivity constraints affect merges. Dendrogram height is a linkage-dependent quantity, not a calibrated probability of taxonomic separation.

**Computational complexity / scalability notes:** Dense distance construction costs $`O(n^2d)`$ for ordinary Euclidean vectors and uses $`O(n^2)`$ memory. Efficient nearest-neighbor-chain implementations of common linkages take $`O(n^2)`$ linkage work after distances, whereas naive repeated searches can be cubic. Sparse connectivity can reduce work substantially but does not give a universal linear bound; graph fill-in and merge order matter.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn structured Ward coin-segmentation example](https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html) processes the named `skimage.data.coins` image. After smoothing and downsampling, pixel intensities and a grid-neighbor connectivity graph feed Ward clustering. Its displayed run partitions 4,697 pixels into 27 regions, producing a contour overlay for inspecting coins and background. Spatially constrained merging is technically preferable to unconstrained intensity k-means when disconnected equal-intensity pixels should not become one region. The source notes that more regions than coins are needed because background structure also receives segments. This is a segmentation demonstration, not verified coin counting; no object-level accuracy or production KPI is reported.

**Notable vendor implementations/libraries:** SciPy `cluster.hierarchy.linkage`, `sklearn.cluster.AgglomerativeClustering`, R `hclust`, and `fastcluster`. Check linkage, distance, and connectivity compatibility.

### 3.1.6 BIRCH

**Name:** BIRCH: Balanced Iterative Reducing and Clustering using Hierarchies.

**Category & sub-category:** Unsupervised learning; clustering-feature compression followed by global clustering.

**Originating paper/vendor/year:** Tian Zhang, Raghu Ramakrishnan, and Miron Livny, [*BIRCH: An Efficient Data Clustering Method for Very Large Databases*, SIGMOD 1996](https://doi.org/10.1145/233269.233324). Their [1997 extended paper](https://research.ibm.com/publications/birch-a-new-data-clustering-algorithm-and-its-applications) documents applications as well as the algorithm.

**Core mechanism:** Maintain a height-balanced clustering-feature tree. A subcluster summary stores $`N`$, the linear sum $`LS=\sum x_i`$, and squared-norm sum $`SS=\sum\|x_i\|^2`$. These additive statistics recover a centroid and within-subcluster dispersion without retaining every pairwise distance. Insertions descend toward nearby summaries; a radius/diameter threshold controls absorption, and overflowing nodes split. Optional condensation and global clustering operate on the compressed representation.

**Inputs/outputs and typical data types:** Numeric, usually Euclidean, observations arriving in batches or streams. Outputs include subcluster summaries and, when a final clustering stage is requested, macrocluster labels. New observations can be mapped to representatives. A clustering-feature summary does not preserve arbitrary dissimilarities, original shapes, or all individual records.

**Strengths and limitations:** Useful when the main problem is fitting a large dataset into limited working memory. The compression threshold trades resolution against memory and is not a direct request for a particular number of final groups. Input order and early compression can change results. High-dimensional distances and broad nonspherical groups weaken compact-summary assumptions. When globally clustering summaries, whether their population weights are used matters: treating a tiny and huge subcluster equally changes the objective.

**Computational complexity / scalability notes:** With branching bound $`b`$ and tree height $`h`$, a normal insertion examines approximately $`O(bh)`$ summaries and costs $`O(bhd)`$. Thus $`O(nbhd)`$ describes the insertion work before accounting for splits, rebuilds, and optional refinement. "Linear in $`n`$" assumes tree-search costs remain bounded. If $`m`$ summaries survive, their storage is approximately $`O(md)`$, and the chosen final algorithm adds its own cost on $`m`$, possibly quadratic.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** Zhang, Ramakrishnan, and Livny's [1997 BIRCH application study](https://research.ibm.com/publications/birch-a-new-data-clustering-algorithm-and-its-applications) reports an interactive pixel-classification tool and initial-codebook generation for image compression. In the latter problem, image-derived numeric vectors are compressed into clustering features, then representative vectors become an initial quantization codebook for subsequent image coding. The documented outcome is an implemented application of BIRCH, not a named commercial image product. Our rationale versus full-data agglomeration is bounded-memory summarization before expensive global work. The public abstract establishes these applications but does not supply a reproducible compression-quality metric or production KPI; neither is invented here.

**Notable vendor implementations/libraries:** `sklearn.cluster.Birch`, including incremental fitting. Its optional final clustering and weighting behavior should be checked against the intended objective rather than assumed identical to every original BIRCH phase.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| k-means | Euclidean numeric vectors | Direct squared-distortion optimization | Compact-cell bias and local optima | Summer Palace photograph palette |
| Mini-batch k-means | Large batches or streamed feature vectors | Small working set | Stochastic variability and rare-group loss | Four-category 20 Newsgroups study |
| k-medoids / PAM | Objects with meaningful dissimilarities | Actual observed exemplars | Expensive pairwise distances and swaps | Eurostat agricultural-workforce data |
| Gaussian mixtures with EM | Continuous overlapping distributions | Soft memberships and density scores | Covariance degeneracy and component ambiguity | Gaia overlapping cluster pairs |
| Agglomerative clustering | Moderate datasets or constrained graphs | Nested partitions at several resolutions | Irreversible, linkage-sensitive merges | Structured Ward coin segmentation |
| BIRCH | Large numeric streams | Additive, memory-conscious summaries | Order-sensitive lossy compression | Published image-codebook application |

## 3.2 Density and connectivity clustering

Density methods ask whether observations occupy locally crowded regions separated by sparse ones. Graph methods instead use the connectivity encoded in an affinity matrix. Neither approach eliminates assumptions: the metric, sampling density, neighborhood size, edge construction, and extraction rule define what can be found. An isolated observation may be measurement noise, a rare valid event, or a genuinely new population.

### 3.2.1 DBSCAN

**Name:** DBSCAN: Density-Based Spatial Clustering of Applications with Noise.

**Category & sub-category:** Unsupervised learning; fixed-scale density-connected clustering with explicit noise.

**Originating paper/vendor/year:** Martin Ester, Hans-Peter Kriegel, Jorg Sander, and Xiaowei Xu, [*A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*, KDD 1996](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf).

**Core mechanism:** For radius $`\varepsilon`$, a core point has at least `min_samples` observations in its neighborhood, including itself under the usual scikit-learn convention. Connected chains of core points form clusters. Non-core points neighboring a core join as border points; remaining observations are noise. Border points do not propagate connectivity. A border point reachable from two core components can receive an order-dependent assignment without changing the core components.

**Inputs/outputs and typical data types:** Vectors, coordinates, or suitable precomputed distances; outputs are labels, core-point information, and noise indicators. Latitude/longitude require an appropriate geographical distance and units, not automatic Euclidean treatment of degrees. Canonical DBSCAN is transductive: new observations can turn old border points into cores or connect existing components.

**Strengths and limitations:** Finds nonspherical connected groups without specifying $`k`$ and can leave observations unassigned. One global radius struggles when meaningful groups have markedly different densities. Standardization changes the physical meaning of the radius; high-dimensional distance concentration can erase useful contrast. A $`k`$-distance plot helps propose parameters but does not objectively determine a unique threshold. Dense anomalous groups may be found as ordinary clusters rather than rejected.

**Computational complexity / scalability notes:** With a useful low-dimensional spatial index, neighborhood search can approach $`O(n\log n+Q)`$, where $`Q`$ is the number of reported neighbor incidences. Brute-force distances cost $`O(n^2d)`$. Original on-demand processing need not store every neighborhood, but implementations that bulk-materialize them can use $`O(Q)`$ extra memory, reaching $`O(n^2)`$ for large radii. These qualifications matter more than an unconditional "linear" or "log-linear" label.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original paper](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf) evaluates spatial-data processing using SEQUOIA 2000's point dataset: 62,584 California landmark names and locations derived from the US Geological Survey's Geographic Names Information System. Runtime tests use subsets containing 2% to 20% of that dataset, not a modern full-scale deployment. Coordinates become density-connected groups and noise; these outputs support geographic inspection. The study reports lower runtimes than CLARANS under its tested setup, while its synthetic experiments separately demonstrate arbitrary-shape recovery. Our rationale versus medoid partitioning is connectivity without a fixed number of representatives. Neither a landmark-level ground-truth accuracy nor a production GIS KPI is established.

**Notable vendor implementations/libraries:** `sklearn.cluster.DBSCAN`, ELKI, and PostGIS `ST_ClusterDBSCAN`. Database availability does not imply that every GIS workload uses DBSCAN.

### 3.2.2 OPTICS

**Name:** OPTICS: Ordering Points To Identify the Clustering Structure.

**Category & sub-category:** Unsupervised learning; density ordering and multiscale cluster extraction.

**Originating paper/vendor/year:** Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, and Jorg Sander, [*OPTICS: Ordering Points To Identify the Clustering Structure*, SIGMOD 1999](https://web.ece.ucsb.edu/Faculty/Manjunath/courses/ece594S03/Papers/optics.pdf).

**Core mechanism:** Produce an ordering with core and reachability distances rather than immediately committing to one flat partition. For a core predecessor $`p`$, candidate $`o`$'s reachability through $`p`$ is $`\max\{\operatorname{coreDist}(p),D(p,o)\}`$. Expanding the most reachable pending observation creates valleys in a reachability plot. An epsilon cut or a steepness-based `xi` extraction rule subsequently produces clusters; these are different interpretations of the ordering.

**Inputs/outputs and typical data types:** Numeric or metric data, including precomputed distances. Primary outputs are the ordering, reachability values, and predecessors; labels require extraction choices. The original ordering is transductive. A nearest-cluster assignment rule for later observations is an extension, not an update of the original density ordering.

**Strengths and limitations:** Helps inspect nested or differently dense groups without rerunning DBSCAN at every radius. It is not parameter-free: minimum neighborhood size, maximum search radius, metric, and extraction settings remain consequential. An unlimited radius can be expensive. Valleys can reflect sampling density rather than meaningful categories, and high-dimensional irrelevant features can overwhelm the neighborhood structure.

**Computational complexity / scalability notes:** An indexed implementation with a heap can approach $`O(n\log n)`$ under fixed dimension and bounded neighborhood sizes; large neighborhoods add their enumeration and priority-update costs. Brute-force search can require $`O(n^2d)`$ distance work. The [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html) does not use a heap for expansion selection and documents quadratic time. Retained ordering arrays are linear, but neighbor indexes, input, and any supplied dense distance matrix add storage.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [original OPTICS study](https://web.ece.ucsb.edu/Faculty/Manjunath/courses/ece594S03/Papers/optics.pdf) analyzes 30,000 industrial-part contour records represented by sixteen Fourier-derived attributes. An OPTICS ordering plus the authors' Circle Segments visualization reveals small and medium groups and a larger group separated by noise; linked attribute displays help explain their differences. Shape descriptors become an ordering, then visual cluster hypotheses for engineering inspection. Our rationale versus one global DBSCAN cut is retaining several density levels for examination. This is the paper's named industrial-contour study, not a disclosed factory deployment: the manufacturer, downstream quality-control decisions, defect-detection accuracy, and production KPI are not established.

**Notable vendor implementations/libraries:** `sklearn.cluster.OPTICS`, ELKI, and R `dbscan`. Compare extraction conventions and implementation complexity, not only the algorithm name.

### 3.2.3 HDBSCAN

**Name:** HDBSCAN, commonly referring to the HDBSCAN* hierarchy and stable-cluster extraction.

**Category & sub-category:** Unsupervised learning; hierarchical density clustering with noise.

**Originating paper/vendor/year:** Ricardo Campello, Davoud Moulavi, and Jorg Sander, [*Density-Based Clustering Based on Hierarchical Density Estimates*, PAKDD 2013](https://doi.org/10.1007/978-3-642-37456-2_14). Later work and implementations extend the hierarchy, extraction, and outlier scoring.

**Core mechanism:** Define mutual-reachability distance as


$$
D_{\mathrm{mr}}(a,b)=\max\{\operatorname{coreDist}(a),\operatorname{coreDist}(b),D(a,b)\}.
$$


Construct its minimum spanning tree, derive a single-linkage hierarchy, and condense branches using a minimum cluster size. Stable-cluster extraction selects branches that persist over density levels; leaf extraction yields a different, finer result. The [implementation authors' explanation](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) separates these stages.

**Inputs/outputs and typical data types:** Numeric or metric observations; outputs include a condensed tree, labels, noise, and implementation-dependent membership strengths. Minimum cluster size and neighborhood-density smoothing are distinct controls. Check whether a library's neighbor count includes the observation itself.

**Strengths and limitations:** Handles differing density scales more flexibly than one global DBSCAN radius, but cannot reliably separate populations that overlap in the chosen feature space. Stability and membership strength are not calibrated scientific significance or posterior class probabilities. The method remains transductive. The Python package's [`approximate_predict`](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html) assigns new points while freezing existing structure; it cannot discover new clusters or correctly reproduce every merge a refit would make.

**Computational complexity / scalability notes:** A dense exact route uses $`O(n^2d)`$ distance work and $`O(n^2)`$ storage; dense Prim-style MST construction adds quadratic work. Once an MST exists, sorting its $`n-1`$ edges is $`O(n\log n)`$. Low-dimensional tree-based neighbor/MST algorithms can avoid the full matrix and approach log-linear scaling. Dimension, metric support, exactness, and cluster geometry prevent a universal log-linear guarantee.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** [Hunt and Reffert's 2023 all-sky Gaia DR3 study](https://ar5iv.labs.arxiv.org/html/2303.13424) searches 729 million sources down to approximately magnitude $`G=20`$, partitioned into overlapping sky fields rather than one enormous dense-distance problem. Astrometric positions and motions feed HDBSCAN; candidate groups then undergo statistical density checks, color-magnitude validation with a separate Bayesian CNN, and catalogue reconciliation. Its more stringent catalogue cut contains 4,105 highly reliable clusters, including 739 new objects. Those are whole-pipeline catalogue counts, not HDBSCAN accuracy. The study also discusses false positives and potentially unbound groups. Our rationale versus fixed-radius DBSCAN is varying density and unknown group count; no commercial KPI is reported.

**Notable vendor implementations/libraries:** Python `hdbscan`, `sklearn.cluster.HDBSCAN`, and other scientific implementations. Their prediction facilities, parameter conventions, and optimizations are not interchangeable.

### 3.2.4 Spectral clustering

**Name:** Spectral clustering, represented by normalized graph-Laplacian embeddings followed by partitioning.

**Category & sub-category:** Unsupervised learning; affinity-graph partitioning.

**Originating paper/vendor/year:** Spectral graph partitioning has earlier roots. Shi and Malik's [*Normalized Cuts and Image Segmentation*, 2000](https://doi.org/10.1109/34.868688), and Ng, Jordan, and Weiss's 2001 formulation are influential modern variants, not identical algorithms.

**Core mechanism:** Build a nonnegative affinity graph $`W`$, with degree matrix $`D`$. A normalized Laplacian is $`L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}`$. Selected low-eigenvalue eigenvectors yield a representation in which connected groups can separate. Depending on the formulation, rows are normalized and then clustered with k-means or discretized by another rule. The continuous spectral relaxation approximates a discrete graph-cut objective; it is not generally an exact minimum-cut partition.

**Inputs/outputs and typical data types:** Similarity matrices or feature vectors used to construct kernels or neighbor graphs. Outputs are graph-dependent labels and optionally eigenvectors. Distances cannot simply be passed as affinities: large distances mean weak, not strong, connections. New-point assignment needs an extension such as Nystrom approximation or refitting the graph.

**Strengths and limitations:** Captures connected, nonconvex groups that centroid clustering in input space misses. It can exploit application-specific graph structure. However, kernel bandwidth, neighbor count, disconnected components, and degree imbalance may determine the result. An apparent eigengap is suggestive rather than proof of the correct $`k`$. Repeated eigenvalues permit rotations of the eigenbasis; later clustering can add initialization variability.

**Computational complexity / scalability notes:** A dense affinity matrix costs $`O(n^2d)`$ to construct for basic vector kernels and $`O(n^2)`$ to store; a full dense eigendecomposition is $`O(n^3)`$. With $`m`$ edges and $`r`$ requested eigenvectors, iterative sparse eigensolvers involve matrix-vector and orthogonalization work, roughly $`O(I_e(mr+nr^2))`$ for the specified solver iterations. Subsequent k-means adds $`O(I_c nkr)`$. Neighbor-graph construction and solver convergence must be included.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn Greek-coin segmentation example](https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_segmentation.html) converts the `skimage.data.coins` image into a neighboring-pixel graph. Smoothed intensity differences become exponentially decreasing edge affinities; spectral partitioning generates region labels and contour overlays. The study compares k-means, discretization, and QR-based assignment after the spectral stage. Our rationale versus raw-pixel k-means is respecting weak image boundaries and spatial connectivity. The documented output is a partly homogeneous image partition with manually chosen region settings, not automatic determination of the number of coins. No segmentation ground-truth score or production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.cluster.SpectralClustering`, SciPy sparse eigensolvers, and R `kernlab::specc`. Sparse eigensolver support does not remove the cost of constructing the graph.

### 3.2.5 Mean shift

**Name:** Mean shift; mode-seeking clustering and related tracking procedures.

**Category & sub-category:** Unsupervised learning; nonparametric density-mode discovery.

**Originating paper/vendor/year:** Fukunaga and Hostetler's [1975 density-gradient estimation paper](https://doi.org/10.1109/TIT.1975.1055330) is foundational; Cheng's [*Mean Shift, Mode Seeking, and Clustering*, 1995](https://doi.org/10.1109/34.400568), and Comaniciu and Meer's 2002 feature-space treatment developed influential formulations and applications.

**Core mechanism:** Move each seed toward a locally weighted average of observations. For suitable kernel-derived weights, $`x_{\mathrm{new}}=\sum_i w_i(x)x_i/\sum_i w_i(x)`$ follows a density-ascent direction. Seeds reaching the same mode are merged into a group. A flat neighborhood gives the simple mean of points within a bandwidth; general mean-shift weights must be consistent with the chosen density-kernel profile.

**Inputs/outputs and typical data types:** Continuous feature vectors, joint spatial/color features, or probability-weighted pixel coordinates. Clustering outputs mode representatives and labels. Spatial and color bandwidths need compatible scaling. Tracking applies the mode-seeking step to a histogram-backprojection image and outputs a moved window, which is not the same output as clustering an entire feature dataset.

**Strengths and limitations:** Avoids specifying $`k`$ and can adapt representatives to density peaks. Bandwidth nevertheless controls how many modes survive: too small creates fragments, too large merges groups. Density peaks are not necessarily semantic categories. High dimension, seed count, and broad neighborhoods limit scalability. Ordinary fixed-window mean-shift tracking also struggles with scale changes; CamShift is a distinct extension.

**Computational complexity / scalability notes:** With $`q`$ seeds, $`I`$ updates each, and naive evaluation against $`n`$ dense points, work is $`O(Iqnd)`$. Seeding every observation gives $`O(In^2d)`$. Spatial indexes, bin seeding, and compact-support kernels can reduce this, depending on neighborhoods. Bandwidth estimation and merging candidate modes add costs. A local image-window tracker can be much cheaper than full-data clustering.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [OpenCV 4.13.0 mean-shift tutorial](https://docs.opencv.org/4.13.0/d7/d00/tutorial_meanshift.html) tracks an object in the named `slow_traffic_small.mp4` video. An initial region supplies a hue histogram; each later frame becomes a histogram-backprojection image, and mean shift moves a fixed-size window toward concentrated matching pixels. The displayed output is a tracked bounding box. Our rationale versus framewise k-means is local mode following without repeatedly partitioning all colors. The initialized target window supplies task information, so the whole tracking pipeline is not wholly unsupervised merely because its update is label-free. No tracking-accuracy benchmark, deployed traffic service, or production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.cluster.MeanShift` for clustering; OpenCV `meanShift` and `pyrMeanShiftFiltering` for related vision operations. These APIs solve different surrounding tasks.

### 3.2.6 Affinity propagation

**Name:** Affinity propagation.

**Category & sub-category:** Unsupervised learning; exemplar selection by pairwise message passing.

**Originating paper/vendor/year:** Brendan J. Frey and Delbert Dueck, [*Clustering by Passing Messages Between Data Points*, Science 2007](https://doi.org/10.1126/science.1136800).

**Core mechanism:** Supply similarities $`s(i,j)`$ and diagonal preferences for choosing each observation as an exemplar. Responsibility messages express how suitable $`j`$ is for $`i`$ compared with competing exemplars; availability messages express whether other observations support choosing $`j`$. Iterative updates exchange these forms of evidence, with damping to reduce oscillation. Exemplars and assignments emerge jointly. Increasing preference often increases exemplar count, but it is not a calibrated cluster-number parameter.

**Inputs/outputs and typical data types:** Pairwise similarities, possibly derived from images, text, or domain-specific comparisons; outputs are exemplar indices and assignments. Negative squared Euclidean distance is one common similarity, not a requirement. A generic similarity matrix need not supply the feature representation required for later nearest-exemplar prediction.

**Strengths and limitations:** Produces observed representatives without initializing a fixed-size medoid set. It can use similarities that lack a simple centroid interpretation. Preferences remain substantive assumptions, and uniformly "automatic" model selection should not be claimed. Message passing can oscillate or fail to converge; damping is not a guarantee of an optimal solution. Duplicate observations and ties can produce ambiguity. Its global optimization setup is transductive, even when an implementation adds frozen-exemplar prediction.

**Computational complexity / scalability notes:** Dense responsibility and availability updates cost $`O(n^2)`$ per iteration when row maxima are reused, giving $`O(In^2)`$ optimization time and $`O(n^2)`$ memory. Computing vector similarities may additionally cost $`O(n^2d)`$. Sparse variants can exploit restricted candidate relationships, but those restrictions and their implementation must be stated; dense scikit-learn behavior is not made sparse by describing the input abstractly as a graph.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Frey and Dueck's University of Toronto [2007 study and author abstract](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=17218491&retmode=xml) explicitly apply affinity propagation to face images, gene detection in microarray data, representative sentences, and airline-accessible cities. In the face-image experiment, pairwise similarities become exemplar faces and groups of associated images, allowing representative-based visual inspection. Our rationale versus a randomly initialized medoid subset is the algorithm's simultaneous competition among possible exemplars. The paper reports favorable clustering-error and runtime findings; no universal speedup is quoted here because it depends on the experiment and baseline. This is a named research study, not evidence of deployment in a commercial face-identification product; no production KPI is established.

**Notable vendor implementations/libraries:** `sklearn.cluster.AffinityPropagation` and R `apcluster`, alongside the authors' reference implementations. Check convergence status before interpreting returned exemplars.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| DBSCAN | Low-dimensional spatial or metric data | Arbitrary connected shapes plus noise | One global density scale | SEQUOIA 2000 landmark benchmark |
| OPTICS | Metric data with multiscale density structure | Inspectable reachability ordering | Extraction choices and costly searches | Industrial-part contour study |
| HDBSCAN | Metric observations with varying density | Stable hierarchical cluster extraction | Stability is not scientific significance | Gaia DR3 cluster catalogue |
| Spectral clustering | Sparse affinity graphs | Connectivity-sensitive partitioning | Graph construction and eigensolver cost | Greek-coin image segmentation |
| Mean shift | Continuous multimodal features or image densities | Mode discovery without fixed $`k`$ | Bandwidth dependence | OpenCV traffic-video tracking tutorial |
| Affinity propagation | Moderate-sized similarity matrices | Joint observed-exemplar selection | Quadratic storage and convergence issues | Frey-Dueck face-image experiments |

## 3.3 Representation and dimensionality reduction

A useful representation depends on the property it should preserve: variance, independence, nonnegativity, local neighbors, geodesic distances, or topographic organization. There is no universally faithful two-dimensional view of a complicated high-dimensional distribution. Evaluate reconstruction when reconstruction matters, neighborhood preservation when neighborhoods matter, and held-out downstream behavior when the representation will support prediction.

In particular, **t-SNE and UMAP plots are not proof of clusters**. Their local rescaling and optimization can exaggerate gaps, change apparent density, and rearrange separated groups. Check original-space relationships, alternative parameters and seeds, sampling artifacts, and independent domain measurements before assigning scientific meaning to islands.

### 3.3.1 Principal component analysis

**Name:** Principal component analysis (PCA).

**Category & sub-category:** Unsupervised learning; linear orthogonal representation and low-rank reconstruction.

**Originating paper/vendor/year:** Karl Pearson, [*On Lines and Planes of Closest Fit to Systems of Points in Space*, 1901](https://doi.org/10.1080/14786440109462720); Harold Hotelling's 1933 work developed the principal-component statistical formulation.

**Core mechanism:** Center $`X`$, then compute an SVD $`X_c=U\Sigma V^\top`$. The first $`r`$ columns of $`V`$ maximize retained variance and minimize squared reconstruction error among approximations of rank at most $`r`$. Scores are $`X_cV_r`$. This optimality is for the specified centered matrix and Euclidean loss, not for every downstream task. Uncentered truncated SVD, commonly used for text LSA, solves a related but different problem.

**Inputs/outputs and typical data types:** Continuous feature matrices; outputs are loadings, component scores, explained variances, a mean, and approximate reconstructions. New observations have an explicit linear transform. Standardizing columns produces a correlation-style analysis rather than the original covariance analysis. Whitening additionally removes retained variance scales and can amplify weak, noisy directions.

**Strengths and limitations:** Strong baseline for compression, visualization, conditioning, and noise reduction. It discards low-variance directions even if they predict a rare outcome and can be dominated by outliers, batch effects, or feature units. Eigenvector signs are arbitrary. With repeated eigenvalues, individual axes can rotate inside the tied subspace; the subspace can be identifiable while its displayed axes are not. Retaining high variance does not establish causal or semantic importance.

**Computational complexity / scalability notes:** Dense economy SVD costs $`O(nd\min(n,d))`$. Explicit covariance formation costs $`O(nd^2)`$, followed by a $`d\times d`$ eigensolve, when that route is appropriate. Randomized methods targeting rank $`r`$ use matrix passes costing approximately $`O(ndr)`$ each plus orthogonalization, with accuracy depending on oversampling and power iterations. The deployed transform stores $`O(dr+d)`$ values and costs $`O(dr)`$ per observation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** [Novembre and colleagues' *Genes mirror geography within Europe*, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2735096/) analyzes POPRES genotypes. Its principal analysis uses 197,146 loci in 1,387 individuals after quality and ancestry-selection criteria. Genotype variation becomes two principal-component coordinates; the authors rotate the display to emphasize its geographical resemblance. The result is a geographic continuum useful for understanding population structure and possible confounding in association studies, not a discovery of discrete biological races. Our rationale versus nonlinear visualization is an inspectable linear summary with reusable covariates. The paper's separate ancestry-location predictions involve additional modeling and are not attributed to PCA alone. No clinical or commercial KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.decomposition.PCA`, NumPy/SciPy SVD, R `prcomp`, and Spark ML `PCA`. Solver and centering conventions should accompany reproducibility claims.

### 3.3.2 Kernel PCA

**Name:** Kernel principal component analysis (kernel PCA).

**Category & sub-category:** Unsupervised learning; nonlinear representation through a kernel eigenproblem.

**Originating paper/vendor/year:** Bernhard Scholkopf, Alexander Smola, and Klaus-Robert Muller, [*Nonlinear Component Analysis as a Kernel Eigenvalue Problem*, Neural Computation 1998](https://doi.org/10.1162/089976698300017467).

**Core mechanism:** Replace explicit feature inner products by a positive-semidefinite kernel $`K_{ij}=k(x_i,x_j)`$. Center the Gram matrix using $`HKH`$, where $`H=I-\mathbf1\mathbf1^\top/n`$, and diagonalize it. Principal axes are linear in the implicit feature space but usually nonlinear in the original coordinates. A new point is projected using its correctly training-centered kernel evaluations against the reference observations.

**Inputs/outputs and typical data types:** Vectors or a valid kernel matrix; outputs are nonlinear coordinates and eigenvalues. A model with a callable kernel and saved reference data supports out-of-sample projection. A precomputed training kernel alone does not provide new-to-training similarities. The inverse mapping is a separate pre-image problem and may have no exact or unique solution.

**Strengths and limitations:** Captures nonlinear variation without designing a neural encoder. The kernel and its scale define the representation; an RBF bandwidth can make observations nearly indistinguishable or nearly isolated. Centering features before a nonlinear kernel is not a substitute for centering the Gram matrix. Signs and degenerate eigenspaces remain non-identifiable. Reconstruction quality depends on any separately learned inverse, not only the forward eigenvectors.

**Computational complexity / scalability notes:** For a basic dense vector kernel, Gram construction costs $`O(n^2d)`$ and storage $`O(n^2)`$. A full eigensolve costs $`O(n^3)`$; partial solvers depend on requested rank and convergence. Projection of one new point costs $`O(nd+nr)`$ with all $`n`$ references. Nystrom landmarks reduce this at the expense of approximation. A kernel-ridge inverse adds its own training and evaluation work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn 1.5 USPS denoising study](https://scikit-learn.org/1.5/auto_examples/applications/plot_digits_denoising.html) takes 16-by-16 handwritten digits, separates 1,000 training and 100 test images, and adds Gaussian noise. Its code fits linear PCA and RBF kernel PCA on noisy training images, then reconstructs noisy test images; the kernel inverse is learned using kernel ridge regression. The documented result is nuanced: linear PCA has lower MSE, while kernel PCA produces smoother-looking backgrounds. The configurations use different component counts, so this is not an equal-rank superiority claim. Our rationale is nonlinear denoising when linear subspaces miss image structure. No postal-service deployment, recognition improvement, or production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.decomposition.KernelPCA` and R `kernlab::kpca`. An available `inverse_transform` is an approximation facility, not a mathematical inverse guarantee.

### 3.3.3 Independent component analysis

**Name:** Independent component analysis (ICA), with FastICA as a representative solver.

**Category & sub-category:** Unsupervised learning; linear blind source separation.

**Originating paper/vendor/year:** Blind source-separation work predates the modern name. Pierre Comon's [*Independent Component Analysis, a New Concept?*, 1994](https://doi.org/10.1016/0165-1684(94)90029-9) formalizes the statistical perspective; Hyvarinen and Oja's [FastICA fixed-point algorithm appeared in 1997](https://doi.org/10.1162/neco.1997.9.7.1483).

**Core mechanism:** Assume observed channels mix latent sources approximately linearly, $`x=As`$. Estimate an unmixing transformation whose recovered components are statistically independent, often by maximizing non-Gaussianity after centering and whitening. FastICA uses fixed-point contrast updates and decorrelation; PCA only decorrelates second-order statistics, which is weaker than independence.

**Inputs/outputs and typical data types:** Multichannel time series, spectra, or other approximately linearly mixed continuous signals. Outputs are component time courses, mixing/unmixing matrices, and reconstructed observations. A learned linear unmixing can transform later samples if the mixing process stays sufficiently stable.

**Strengths and limitations:** Useful for separating sources without labeled examples of every artifact. Under standard linear ICA assumptions, at most one source may be Gaussian for full source identifiability. Components are recoverable only up to permutation and nonzero scaling, including sign. Independence does not identify a component as neural activity or eye movement; interpretation requires external evidence. Correlated sources, nonlinear or changing mixtures, poor rank selection, and sensor artifacts can invalidate the model.

**Computational complexity / scalability notes:** Whitening requires a decomposition or covariance calculation. After reducing to $`r`$ whitened dimensions, a typical parallel FastICA iteration costs $`O(nr^2+r^3)`$ for projected contrasts and decorrelation. Multiply by the actual iteration count; convergence is contrast- and initialization-dependent. Retaining samples costs $`O(nr)`$, and the mixing/unmixing representation adds approximately $`O(dr)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [MNE-Python ICA artifact-correction tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) uses a sample MEG/EEG recording to remove eye-blink and heartbeat contamination. A high-pass-filtered copy feeds a fifteen-component ICA fit. Component waveforms, scalp patterns, and EOG/ECG references support selection; excluding identified artifact components and reconstructing the original recording visibly repairs those artifacts. Our rationale versus rejecting whole time intervals is separating nuisance sources while preserving more observations. Manual or sensor-assisted component selection is extra information beyond unsupervised fitting, and removing a mixed neural/artifact component can discard useful signal. The tutorial reports an analysis result, not patient-diagnostic accuracy or a production KPI.

**Notable vendor implementations/libraries:** `sklearn.decomposition.FastICA`, MNE-Python, EEGLAB's ICA workflows, and R `fastICA`. Infomax and Picard are other ICA solvers, not synonyms for the FastICA update.

### 3.3.4 Non-negative matrix factorization

**Name:** Non-negative matrix factorization (NMF).

**Category & sub-category:** Unsupervised learning; additive low-rank representation.

**Originating paper/vendor/year:** Nonnegative factorization has earlier roots, including positive matrix factorization. Daniel D. Lee and H. Sebastian Seung popularized parts-based NMF in [*Learning the Parts of Objects by Non-negative Matrix Factorization*, Nature 1999](https://doi.org/10.1038/44565).

**Core mechanism:** Approximate $`X\in\mathbb R_+^{n\times d}`$ by $`WH`$, with $`W\in\mathbb R_+^{n\times r}`$ and $`H\in\mathbb R_+^{r\times d}`$. Minimize a chosen loss, commonly squared Frobenius error or generalized KL divergence, optionally with sparsity penalties. Multiplicative updates, coordinate descent, and alternating nonnegative least squares solve different subproblems; no generic solver guarantees the global nonconvex optimum.

**Inputs/outputs and typical data types:** Nonnegative intensities, concentrations, counts, or TF-IDF weights; outputs are additive components and nonnegative sample coefficients. Centering into negative values violates the basic model. Column or row normalization changes the loss weighting and can alter the interpretation of amounts versus proportions.

**Strengths and limitations:** Nonnegativity avoids cancellation and often yields interpretable additive structure. It does not guarantee localized parts, sparse factors, or uniquely correct topics. For any positive diagonal $`S`$, $`WH=(WS)(S^{-1}H)`$; permutations and additional factorizations also cause non-identifiability. An apparent factor can reflect batch effects or preprocessing. For new data, coefficients usually require a constrained optimization with $`H`$ fixed, not a simple orthogonal projection.

**Computational complexity / scalability notes:** A typical dense alternating iteration involves approximately $`O(ndr+(n+d)r^2)`$ arithmetic, depending on solver and objective. Suitable sparse Frobenius implementations replace the leading $`ndr`$ multiplication by $`\operatorname{nnz}(X)r`$, but factors may be dense. Factor storage is $`O((n+d)r)`$. Rank, initialization, zero-locking in some multiplicative updates, and stopping tolerances affect the number of iterations.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn 1.5 Olivetti face-decomposition example](https://github.com/scikit-learn/scikit-learn/blob/1.5.X/examples/decomposition/plot_faces_decomposition.py) compares several unsupervised representations of face images. Its NMF branch explicitly uses the original nonnegative pixels rather than the centered array used by PCA and ICA. Images become six nonnegative basis images and coefficient vectors, and the basis images are displayed for inspection. Our rationale versus signed PCA components is an additive account of image intensity without positive/negative cancellation. The source demonstrates a representation, not a facial-recognition deployment or an evaluated identity classifier. No recognition accuracy, privacy-preserving property, or production KPI is claimed.

**Notable vendor implementations/libraries:** `sklearn.decomposition.NMF` and `MiniBatchNMF`, R NMF packages, and scientific optimization libraries. Check the chosen beta-divergence, normalization, and regularization before comparing factors.

### 3.3.5 t-SNE

**Name:** t-distributed stochastic neighbor embedding (t-SNE).

**Category & sub-category:** Unsupervised learning; neighborhood-oriented low-dimensional visualization.

**Originating paper/vendor/year:** Laurens van der Maaten and Geoffrey Hinton, [*Visualizing Data using t-SNE*, JMLR 2008](https://jmlr.org/papers/v9/vandermaaten08a.html), building on earlier stochastic neighbor embedding.

**Core mechanism:** Construct high-dimensional pairwise neighborhood probabilities, with local bandwidths adjusted to a target perplexity. Optimize low-dimensional coordinates to minimize $`\mathrm{KL}(P\|Q)`$, where $`Q`$ uses a heavy-tailed Student distribution. Missing a high-probability neighbor is strongly penalized; preserving every large distance is not the objective. Initialization, early exaggeration, learning rate, and iteration schedule influence the result.

**Inputs/outputs and typical data types:** Numeric feature vectors or suitable distances, often after PCA or another denoising stage. Output is one coordinate vector per training observation, usually in two dimensions. Ordinary nonparametric t-SNE does not learn a general encoder or inverse; new-point interpolation and parametric t-SNE are additional methods with their own assumptions.

**Strengths and limitations:** Excellent for inspecting local relationships in difficult feature spaces. Apparent island areas do not estimate original variance or density; inter-island distances and positions are generally not reliable global geometry. A visually dramatic gap can be induced by settings or sampling. [Wattenberg, Viegas, and Johnson's controlled experiments](https://distill.pub/2016/misread-tsne/) demonstrate these distortions, including apparent clumps in random data. Compare several perplexities, seeds, and original-space neighborhoods rather than selecting the prettiest plot.

**Computational complexity / scalability notes:** Dense exact affinities require $`O(n^2)`$ storage and often $`O(n^2d)`$ distance work. Exact low-dimensional force evaluation costs $`O(n^2r)`$ per iteration. [Barnes-Hut t-SNE](https://www.jmlr.org/papers/v15/vandermaaten14a.html) approximates forces in approximately $`O(n\log n)`$ per iteration for the low-dimensional setting; neighbor construction, accuracy controls, and total iterations remain additional costs. This is not a log-linear guarantee for the entire pipeline.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn handwritten-digit manifold comparison](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html) embeds 64-pixel vectors from the first six classes of its digits dataset. Its t-SNE uses PCA initialization; digit identities color and annotate the resulting map rather than train the t-SNE objective. Vectors become a two-dimensional arrangement for inspecting visually related writing styles and class overlap. Our rationale versus linear PCA is revealing local nonlinear relationships. The published result is an embedding comparison, not measured recognition accuracy or proof of six naturally separated populations. It reports no production KPI; apparent distances after the example's display rescaling should not be interpreted as original-space distances.

**Notable vendor implementations/libraries:** `sklearn.manifold.TSNE`, openTSNE, and FIt-SNE. Transform facilities in some libraries do not change the nonparametric nature of the original formulation.

### 3.3.6 UMAP

**Name:** Uniform Manifold Approximation and Projection (UMAP).

**Category & sub-category:** Unsupervised learning; graph-based nonlinear representation and visualization.

**Originating paper/vendor/year:** Leland McInnes, John Healy, and James Melville, [*UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*](https://arxiv.org/abs/1802.03426), initially posted in February 2018; the cited record also contains later revisions. An arXiv submission year is not a conference publication year.

**Core mechanism:** Construct a nearest-neighbor graph with locally adjusted distance scales, combine directed fuzzy memberships, and optimize a low-dimensional graph representation using an attractive/repulsive cross-entropy formulation. Practical optimization samples edges and negative examples rather than evaluating every non-edge exactly. `n_neighbors` changes the scale of the relationships represented; `min_dist` changes how tightly nearby points can pack in the output.

**Inputs/outputs and typical data types:** Numeric vectors with a supported metric or an appropriate distance representation; outputs are coordinates in chosen dimension $`r`$. Standard unsupervised fitting uses no target labels. Supervised UMAP adds target information and belongs to a different training setup. The common implementation can transform new vectors by locating training neighbors and optimizing positions relative to a fixed reference embedding.

**Strengths and limitations:** Practical for exploration and reusable feature pipelines, including output dimensions larger than two. It does not guarantee preservation of global distances, density, or topology. Tight islands can be optimization effects, and density correction requires additional methods rather than ordinary UMAP. The [authors' transform tutorial](https://umap-learn.readthedocs.io/en/latest/transform.html) assumes similar training/test distributions; projecting out-of-distribution observations into an old map is not evidence that they belong there.

**Computational complexity / scalability notes:** Exact neighbor search can cost $`O(n^2d)`$; approximate search depends on its index and search effort. A $`k`$-neighbor graph stores $`O(nk)`$ edges. With $`s`$ negative samples per sampled edge, a broad optimization accounting is $`O(E nkr(1+s))`$, assuming that many edge updates per pass. Reference data, search indexes, and $`O(nr)`$ coordinates add memory. Avoid a universal $`O(n\log n)`$ claim.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [UMAP authors' Palmer Penguins tutorial](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) retains 333 complete records and standardizes bill length, bill depth, flipper length, and body mass. Four-dimensional measurements become two coordinates; species labels color the result only afterward. The tutorial compares the visual structure with the original pairwise plots and shows recognizable species-related organization. Our rationale versus raw projection is nonlinear neighborhood summarization, while acknowledging that four features already permit direct inspection. This result does not prove that a new biological population was discovered, and no ecological-management or production KPI is reported.

**Notable vendor implementations/libraries:** `umap-learn` and R `uwot`. Parametric UMAP adds a neural encoder and is not the non-neural formulation described in this entry.

### 3.3.7 Isomap

**Name:** Isomap: isometric feature mapping.

**Category & sub-category:** Unsupervised learning; geodesic-distance manifold embedding.

**Originating paper/vendor/year:** Joshua B. Tenenbaum, Vin de Silva, and John C. Langford, [*A Global Geometric Framework for Nonlinear Dimensionality Reduction*, Science 2000](https://doi.org/10.1126/science.290.5500.2319).

**Core mechanism:** Connect nearby observations in a weighted graph. Approximate manifold geodesic distances using shortest paths, then apply classical multidimensional scaling to the squared-distance matrix. Eigenvectors of the double-centered distance matrix supply coordinates. Unlike PCA, the method tries to retain long-range distances measured along the inferred manifold rather than straight through ambient space.

**Inputs/outputs and typical data types:** Dense vectors or neighborhood-compatible dissimilarities; outputs are coordinates, a neighborhood graph, and graph-geodesic information. Image collections with changing pose and densely sampled physical trajectories are plausible inputs when local distances genuinely track continuous variation.

**Strengths and limitations:** Can unfold a curved manifold and provide interpretable global coordinates under strong sampling and geometry assumptions. Too many neighbors create shortcuts across folds; too few disconnect the graph. Noise, holes, variable sampling density, and several unrelated manifolds can invalidate shortest-path distances. A non-Euclidean geodesic matrix can produce negative eigenvalues, indicating that an exact Euclidean realization is unavailable. Coordinate orientation is arbitrary; a good-looking unfolding is not proof of an isometric latent space.

**Computational complexity / scalability notes:** Exact neighbor search can require $`O(n^2d)`$. For $`O(nk)`$ edges, running binary-heap Dijkstra from every vertex costs approximately $`O(n^2k\log n)`$; Floyd-Warshall instead costs $`O(n^3)`$. Storing all geodesic distances requires $`O(n^2)`$, and a full dense MDS eigensolve is cubic. Landmark methods reduce costs by approximation. Library `transform` methods attach new points to the existing graph and extend the embedding; they do not recompute the original joint geometry.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In the [scikit-learn digits manifold study](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html), 8-by-8 digit images from six classes become 64-dimensional vectors. The Isomap branch constructs local relationships using the study's thirty-neighbor setting and produces a two-dimensional map for inspecting writing-style variation. Its documented output is a plotted embedding alongside PCA-like and local manifold alternatives, not a reported classifier or deployment. Our rationale is preserving pathwise variation that straight-line projection can flatten; distinct digit classes also illustrate why a single connected-manifold assumption may be questionable. No original-space distance-preservation score or production KPI is supplied by this example.

**Notable vendor implementations/libraries:** `sklearn.manifold.Isomap`, scientific MDS/eigensolver toolkits, and landmark-Isomap implementations. Inspect how disconnected graphs and new-point extensions are handled.

### 3.3.8 Locally linear embedding

**Name:** Locally linear embedding (LLE), with standard LLE as the representative formulation.

**Category & sub-category:** Unsupervised learning; local affine-reconstruction manifold embedding.

**Originating paper/vendor/year:** Sam T. Roweis and Lawrence K. Saul, [*Nonlinear Dimensionality Reduction by Locally Linear Embedding*, Science 2000](https://doi.org/10.1126/science.290.5500.2323).

**Core mechanism:** Find $`k`$ neighbors for each observation. Choose reconstruction weights minimizing $`\|x_i-\sum_jw_{ij}x_j\|^2`$, with weights restricted to neighbors and summing to one. Then find low-dimensional coordinates minimizing $`\sum_i\|y_i-\sum_jw_{ij}y_j\|^2`$, subject to centering and scale constraints. An eigenproblem for $`(I-W)^\top(I-W)`$ supplies the solution after excluding the trivial constant direction. Weights can be negative; they are not mixture probabilities.

**Inputs/outputs and typical data types:** Densely sampled numeric observations with approximately linear local neighborhoods. Outputs are coordinates and local reconstruction structure. A new-point extension can compute reconstruction weights against training neighbors and apply those weights to their embedded coordinates. This is local interpolation, not a learned global inverse or a full refit.

**Strengths and limitations:** Captures curved variation without computing all-pairs geodesics. Neighborhood size, feature scaling, noise, and local covariance regularization are crucial. Neighborhood covariance becomes singular when too many neighbors lie in too few independent directions. Disconnected graphs introduce additional null directions, and standard LLE may collapse or distort structures. Modified LLE, Hessian LLE, and local tangent-space alignment address different issues and should not be treated as identical algorithms. Rotations and reflections of a solution preserve its relevant geometry.

**Computational complexity / scalability notes:** Beyond neighbor search, building each local Gram matrix costs $`O(dk^2)`$ and a dense local solve $`O(k^3)`$, for $`O(n(dk^2+k^3))`$ total. Store $`W`$ sparsely in $`O(nk)`$. Multiplying by $`(I-W)^\top(I-W)`$ can use two sparse products without explicitly materializing a denser matrix. Eigensolver iterations and conditioning determine further cost; a dense fallback can be cubic.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn handwritten-digit comparison](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html) includes standard, modified, Hessian, and tangent-alignment variants on the same six-class, 64-feature dataset. Local pixel-space relationships become two-dimensional coordinates, displayed with digit thumbnails for interpreting neighborhoods. Our rationale versus Isomap is preserving local reconstruction relationships rather than trusting long shortest paths across possibly different digit manifolds. The source supplies comparative visual outputs and runtime measurements, not a validated universal ranking or a recognition-accuracy improvement. Labels annotate the map; they do not supervise standard LLE. No production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.manifold.LocallyLinearEmbedding` and its functional API. Specify `method`, neighbor count, regularization, and eigensolver rather than reporting only "LLE."

### 3.3.9 Self-organizing maps

**Name:** Self-organizing map (SOM), represented by the classical online Kohonen map.

**Category & sub-category:** Unsupervised learning; competitive neural prototype learning and topographic representation.

**Originating paper/vendor/year:** Teuvo Kohonen, [*Self-Organized Formation of Topologically Correct Feature Maps*, 1982](https://doi.org/10.1007/BF00337288). This competitive network is not a modern multilayer backpropagation architecture.

**Core mechanism:** Place $`U`$ prototype vectors $`w_j\in\mathbb R^d`$ on a fixed grid. For observation $`x`$, find the best-matching unit $`c=\arg\min_j\|x-w_j\|`$. Update the winner and its grid neighbors toward $`x`$: $`w_j\leftarrow w_j+\eta_t h_{cj}(t)(x-w_j)`$. A shrinking neighborhood changes broad topographic organization into local refinement. Nearby grid positions are encouraged, not guaranteed, to represent similar observations.

**Inputs/outputs and typical data types:** Numeric vectors, spectra, or transformed multichannel measurements. Outputs include prototypes, best-matching-unit assignments, occupancy counts, and grid visualizations. New observations can map to a saved prototype. Their quantization error should be inspected instead of forcing every new observation to be treated as familiar.

**Strengths and limitations:** Combines an exemplar-like summary with an organized display. Grid size and topology constrain the representation, and folds, empty units, and initialization effects occur. Neighborhood regularization can smear boundaries. A grid location is not an identifiable latent coordinate, probability, or cell-type label. Input scaling must match the scientific meaning of feature differences.

**Computational complexity / scalability notes:** Exhaustive winner search costs $`O(Ud)`$ per observation; updating all prototypes has the same order. $`E`$ online passes therefore cost $`O(EnUd)`$, with $`O(Ud)`$ prototype storage beyond input. A restricted neighborhood lowers update work but not exhaustive winner-search work. Batch variants aggregate observations before prototype updates and have different parallelization behavior.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Ghent University's [FlowSOM cytometry vignette](https://github.com/SofieVG/FlowSOM/blob/master/vignettes/FlowSOM.Rnw), associated with [Van Gassen and colleagues' 2015 paper](https://doi.org/10.1002/cyto.a.22625), analyzes the named `68983.fcs` sample. Compensation and logicle transformation precede an example using seven marker channels and a 7-by-7 map. Cells become map-node assignments, then separate metaclustering and visualization steps support comparison with manually gated B-, T-, and NK-cell populations. The code requests ten metaclusters; those are not ten SOM neurons. Our rationale versus an unorganized k-means display is inspecting marker relationships across prototypes. The documented outcome is a cell-population analysis workflow, not clinical diagnostic accuracy or a production KPI.

**Notable vendor implementations/libraries:** R `kohonen`, MiniSom, and Bioconductor FlowSOM. The extra fields below describe the online Kohonen formulation, not every wrapper-specific FlowSOM training detail.

**Architecture diagram description:** `d input measurements -> distances to U learned prototypes -> hard best-matching unit -> neighborhood-weighted updates on a fixed 2-D grid`. FlowSOM adds metaclustering and a minimum-spanning-tree display after its SOM stage.

**Activation functions used and why:** Negative squared distance can serve as a matching score; hard winner selection supplies competition. A common neighborhood is Gaussian in grid distance, $`h_{cj}=\exp[-\|g_c-g_j\|^2/(2\sigma_t^2)]`$, to spread updates locally. These are not ReLU hidden layers or class-softmax probabilities.

**Loss function(s):** Neighborhood-weighted squared quantization distortion describes the update with assignments and neighborhood held fixed. General online SOM training is not guaranteed gradient descent on one fixed smooth global objective because winners and neighborhoods change.

**Optimization algorithm(s):** Competitive stochastic updates, not mandatory Adam or backpropagation. Common schedules decrease both learning rate and neighborhood radius; an explicit generic choice is $`\eta_t=\eta_0e^{-t/\tau_\eta}`$, $`\sigma_t=\sigma_0e^{-t/\tau_\sigma}`$. These are qualified schedule choices, not undocumented settings attributed to FlowSOM.

**Regularization techniques:** Early broad neighborhood sharing, limited map capacity, decreasing update sizes, and stopping based on quantization/topographic diagnostics. Preprocessing and initialization matter; dropout, batch normalization, and weight decay are not inherent requirements.

**Backpropagation considerations:** Classical SOM needs no backward computational graph. Hard winner selection is nondifferentiable; prototype adaptation is explicit. Vanishing/exploding gradients are not the central issue, whereas unstable competitive updates or poor organization can be.

**Parameter count / scaling behavior:** $`p=Ud`$ learned prototype coordinates for the basic map, excluding preprocessing and downstream models. The vignette's seven-feature, 49-unit configuration therefore has 343 prototype coordinates by arithmetic, not a claimed total parameter count for all of FlowSOM.

**Training paradigm:** Unlabeled competitive learning, followed optionally by externally informed annotation or separate clustering. Manual gating used to interpret or evaluate cells is not itself the SOM's training target.

**Hardware/parallelism considerations:** Moderate maps are practical on CPUs. Prototype-distance calculations vectorize; large maps can benefit from GPUs. Sequential online updates are order-dependent, while batch SOM allows parallel assignment and reduction of prototype sufficient sums, with communication roughly proportional to the codebook size.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| PCA | Centered continuous vectors | Optimal linear squared-error subspace | Variance need not equal relevance | POPRES genetic-geography study |
| Kernel PCA | Data with an appropriate nonlinear kernel | Nonlinear eigenfeatures | Quadratic Gram storage and approximate inverse | USPS digit denoising |
| ICA | Approximately linearly mixed signals | Independent-source separation | Strong identifiability and mixing assumptions | MNE MEG/EEG artifact correction |
| NMF | Nonnegative intensities or counts | Additive representations | Nonunique factors and nonconvex fitting | Olivetti face decompositions |
| t-SNE | High-dimensional neighborhood structure | Local visual exploration | Distorted sizes, gaps, and global distances | Six-class digit visualization |
| UMAP | Metric vectors with useful local neighborhoods | Practical nonlinear maps and new-point transforms | Embedding islands are not verified clusters | Palmer Penguins measurements |
| Isomap | Densely sampled curved manifolds | Graph-geodesic global geometry | Shortcuts, disconnections, and quadratic storage | Handwritten-digit manifold comparison |
| LLE | Locally affine manifold neighborhoods | Preserves reconstruction relationships | Rank deficiency and neighborhood sensitivity | Standard/modified digit embeddings |
| Self-organizing maps | Scaled multivariate measurements | Organized neural prototype map | Grid distortion and nonprobabilistic assignments | FlowSOM cytometry sample |

## 3.4 Pattern and topic discovery

Frequent-itemset methods discover recurring sets, while topic models represent each document as a mixture of latent word distributions. Neither directly establishes causation. A frequently co-purchased pair may reflect availability, season, household structure, or a previous promotion rather than the effect of recommending one item.

For $`n`$ transactions, define the **support count** of itemset $`A`$ as the number containing every item in $`A`$, and its **support** as that count divided by $`n`$. For disjoint nonempty antecedent $`A`$ and consequent $`B`$:


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


These require nonzero denominators. Lift compares co-occurrence with an independence baseline, not with a randomized marketing intervention. In a **toy calculation**, 100 baskets contain tea in 20, biscuits in 30, and both in 12: support is 0.12, confidence for tea-to-biscuits is 0.60, and lift is 2. This arithmetic is illustrative, not a retailer's measured uplift.

The [arulesViz authors' treatment](https://github.com/mhahsler/arulesViz/blob/master/vignettes/arulesViz.Rnw) documents these definitions. Validate discovered rules on later or held-out transactions, check absolute counts and multiple-testing effects, and choose thresholds based on stability and review capacity. Repeated occurrences of one item in a basket count once in ordinary binary itemset mining; quantities, sequence, and profit require different formulations.

### 3.4.1 Apriori

**Name:** Apriori.

**Category & sub-category:** Unsupervised learning; frequent-itemset enumeration and association-rule generation.

**Originating paper/vendor/year:** Rakesh Agrawal and Ramakrishnan Srikant, [*Fast Algorithms for Mining Association Rules in Large Databases*, VLDB 1994](https://www.vldb.org/conf/1994/P487.PDF). The 1993 Agrawal-Imielinski-Swami paper introduced the association-rule mining problem and earlier algorithms; it should not be conflated with the specific 1994 Apriori algorithm.

**Core mechanism:** Use downward closure of support: every subset of a frequent itemset is frequent. Count singletons, join frequent sets to propose larger candidates, prune candidates containing an infrequent subset, and rescan transactions to count survivors. After frequent sets are available, generate rules and filter by confidence or other criteria. A confidence threshold does not replace the support pruning principle.

**Inputs/outputs and typical data types:** A transaction database of categorical item sets, typically represented sparsely. Outputs are frequent itemsets with counts and optionally rules with support, confidence, and lift. Continuous measurements must be meaningfully discretized before becoming items; scaling them to unit variance alone does not create appropriate transactions.

**Strengths and limitations:** Transparent, easy to constrain, and effective when frequent sets are short and candidate counts remain manageable. Low minimum support can cause candidate explosion and repeated expensive scans. High-confidence rules can merely predict a common consequent; high lift with tiny counts can be unstable. Duplicate receipts, incorrect transaction boundaries, and accidental inclusion of outcome-derived fields can create impressive but useless rules.

**Computational complexity / scalability notes:** Let $`m`$ be the number of distinct items and $`C_\ell`$ the candidates of length $`\ell`$. A naive scan-and-test scheme costs $`O(n\sum_\ell \ell|C_\ell|)`$, plus candidate construction; tries and counting optimizations change constants and traversal work. There can be $`2^m-1`$ nonempty frequent sets. Peak storage includes candidate representations and output, not just the original transactions. Enumeration is inherently output-sensitive, and unrestricted rule generation can be even larger.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Hahsler and Chelluboina's [arulesViz worked study](https://github.com/mhahsler/arulesViz/blob/master/vignettes/arulesViz.Rnw) applies Apriori to the named Groceries dataset. Its [data documentation](https://github.com/mhahsler/arules/blob/master/man/Groceries.Rd) identifies thirty days of actual point-of-sale records: 9,835 baskets aggregated to 169 product groups. The study mines at support 0.001 and confidence 0.5, then plots and ranks rules; the support setting requires at least ten baskets by integer arithmetic. Baskets become frequent conjunctions and candidate retail hypotheses for analyst inspection. Our rationale versus a topic model is exact, auditable co-occurrence counts. The source observes that high-lift rules often have low support; no store intervention, sales uplift, or production KPI is reported.

**Notable vendor implementations/libraries:** R `arules::apriori`, Weka `Apriori`, mlxtend, and SPMF. Library limits on rule length, consequent size, or mining time can truncate the requested search.

### 3.4.2 FP-growth

**Name:** Frequent-pattern growth (FP-growth).

**Category & sub-category:** Unsupervised learning; compressed pattern-growth itemset mining.

**Originating paper/vendor/year:** Jiawei Han, Jian Pei, and Yiwen Yin, [*Mining Frequent Patterns without Candidate Generation*, SIGMOD 2000](https://doi.org/10.1145/335191.335372). Later parallel variants and the expanded journal treatment are separate contributions.

**Core mechanism:** Count frequent individual items, order them consistently, and insert filtered transactions into a prefix-sharing FP-tree. Header links connect nodes carrying the same item. Recursively construct conditional pattern bases and conditional trees to enumerate larger frequent sets. The ordering is a compression strategy, not the temporal ordering of customer behavior.

**Inputs/outputs and typical data types:** Transactions containing unique item identifiers, with minimum support. Outputs are frequent itemsets and counts; association-rule induction is a further stage. A system can recommend consequents when antecedents match a new basket, but that application needs relevance and business evaluation beyond counting.

**Strengths and limitations:** Avoids Apriori's explicit levelwise candidate generation and exploits repeated prefixes. It often works well when transactions share common combinations. Weak prefix sharing, very low support, or a huge output set can still exhaust memory or time. A tiny one-path tree may encode exponentially many frequent subsets. Neither tree compression nor distributed execution removes that output-size lower bound.

**Computational complexity / scalability notes:** Initial counting scans the transaction entries. If each transaction $`t`$ is sorted by global rank using comparison sorting, ordering costs $`O(\sum_t |t|\log |t|)`$; building the initial tree is then linear in retained entries. Mining cost depends on the cumulative sizes of conditional trees and emitted patterns, not merely $`n`$. Memory includes the initial tree, active conditional structures, and outputs. Distributed PFP adds partitioning and communication costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Borgelt's 2005 FP-growth implementation study](https://borgelt.net/papers/fpgrowth.pdf) benchmarks BMS-WebView-1 alongside other named datasets. BMS-WebView-1 contains e-commerce clickstream data, as documented in the [public dataset collection](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php); the itemset formulation uses transaction sets rather than claiming to preserve browsing order. Session item identifiers become a compressed tree and frequent viewed-together sets for inspection. Our rationale versus Apriori is reducing repeated candidate counting. The study's runtime curves are not a universal win: it explicitly notes that Relim is slightly faster at higher tested supports on BMS-WebView-1. No recommendation click-through rate or production KPI is reported, and no such metric is inferred from mining speed.

**Notable vendor implementations/libraries:** Spark ML [`FPGrowth`](https://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html), Borgelt's implementation, SPMF, and mlxtend. Spark documents a parallel PFP variant; frequent-itemset counts and rule confidence are different output quantities.

### 3.4.3 Eclat

**Name:** Eclat.

**Category & sub-category:** Unsupervised learning; vertical-format frequent-itemset mining.

**Originating paper/vendor/year:** Mohammed J. Zaki, Srinivasan Parthasarathy, Mitsunori Ogihara, and Wei Li, *New Algorithms for Fast Discovery of Association Rules*, KDD 1997. The [official `arules` Eclat documentation](https://github.com/mhahsler/arules/blob/master/man/eclat.Rd) identifies that reference and its separate Borgelt implementation.

**Core mechanism:** Represent an itemset by the transaction IDs in which it occurs. An extension's transaction list is obtained by intersection: $`T(A\cup B)=T(A)\cap T(B)`$. Traverse related itemsets depth-first and prune below-support branches. Bitsets and difference-set variants change the efficiency of these operations without changing ordinary support's meaning.

**Inputs/outputs and typical data types:** Binary transaction data, often converted into sorted vertical posting lists. Outputs are frequent sets, supports, and optionally supporting transaction IDs. Rules can be induced afterward. For a separate toy illustration, lists $`\{1,2,4\}`$ and $`\{2,3,4\}`$ intersect at $`\{2,4\}`$: their union itemset appears in two transactions. These are explanatory IDs, not published Adult-dataset counts.

**Strengths and limitations:** Reuses efficient set intersections instead of repeatedly scanning every original transaction, especially when useful vertical lists fit in memory. Long dense lists, many frequent prefixes, or retaining every supporting-ID list can be expensive. As with Apriori, discretization and transaction definitions determine the discovered associations. Mining a census income attribute as one item among many is descriptive association analysis, not automatically training a supervised income classifier.

**Computational complexity / scalability notes:** Intersecting two sorted lists costs $`O(|T_A|+|T_B|)`$; dense bitset intersection costs $`O(n/w)`$ machine-word operations for word width $`w`$, excluding support population counts if implemented separately. Total time sums these costs over visited extensions. Initial postings require space proportional to transaction entries; recursion and retained outputs add memory. Worst-case frequent-set output remains exponential in the number of items.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [`arules::eclat` example](https://github.com/mhahsler/arules/blob/master/man/eclat.Rd) mines the [prepared UCI Adult transactions](https://github.com/mhahsler/arules/blob/master/man/Adult.Rd), containing 48,842 records and 115 items after categorical conversion and documented discretization. It uses minimum support 0.1 and maximum itemset length five, then induces rules at confidence 0.9. Records become posting lists, frequent attribute conjunctions, and inspectable rules. Our rationale versus repeated Apriori scans is shared vertical intersections. The documented output is descriptive itemsets and rules, not evidence of improved benefit allocation or income prediction. Sensitive demographic attributes require care in interpretation; the example reports no policy or production KPI.

**Notable vendor implementations/libraries:** R `arules::eclat`, Borgelt's Eclat/PyFIM implementations, and SPMF. Returning supporting transaction IDs can materially increase memory use.

### 3.4.4 Latent Dirichlet allocation

**Name:** Latent Dirichlet allocation (LDA). This is **not** supervised linear discriminant analysis, which is also abbreviated LDA.

**Category & sub-category:** Unsupervised learning; Bayesian mixed-membership topic modeling.

**Originating paper/vendor/year:** David M. Blei, Andrew Y. Ng, and Michael I. Jordan, [*Latent Dirichlet Allocation*, JMLR 2003](https://www.jmlr.org/papers/v3/blei03a.html).

**Core mechanism:** Each topic has a word distribution $`\beta_j`$; each document has topic proportions $`\theta`$ drawn from a Dirichlet distribution. Each token selects a topic from $`\theta`$, then a word from that topic. Inference estimates latent memberships and topic parameters through approximations such as variational inference or collapsed Gibbs sampling. Common Bayesian implementations also place a Dirichlet prior on topic-word distributions. This is a model of mixed membership, not one-cluster-per-document partitioning.

**Inputs/outputs and typical data types:** Tokenized documents or nonnegative document-term counts. Outputs are topic-word distributions, document-topic mixtures, and likelihood-related diagnostics. Raw counts correspond to the standard generative model; arbitrary TF-IDF weighting changes that interpretation. New documents can infer proportions with global topics frozen.

**Strengths and limitations:** Useful for browsing and summarizing large collections when documents mix themes. Bag-of-words exchangeability discards syntax and order. Topic permutation leaves the model unchanged, while overlapping vocabularies, local optima, and prior settings create further ambiguity. A topic's human-readable name is an interpretation, not a learned ground-truth label. Held-out perplexity depends on the document/token evaluation protocol and does not guarantee coherent or actionable topics.

**Computational complexity / scalability notes:** Let $`M`$ be total tokens, $`V`$ vocabulary size, and $`k`$ topic count. A straightforward collapsed Gibbs sweep costs $`O(Mk)`$, with storage including $`O(M+nk+kV)`$; sparse or alias-based samplers change this. Variational methods have nested document-level iterations, commonly with work proportional to topic count times nonzero term entries per local iteration. Online variational inference reduces batch working sets, not the size of the global $`k\times V`$ topic representation.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn NMF/LDA topic-extraction study](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html) processes 2,000 20 Newsgroups posts, strips headers, signatures, and quotations, and caps the vocabulary at 1,000 terms. Its LDA branch uses term counts and fits ten topics with online variational learning. Counts become topic-word weights displayed as top-word bars, enabling thematic inspection of posts. Our rationale versus k-means is allowing one document to contain several topics; versus NMF, LDA provides an explicit count-generating mixed-membership model. The source demonstrates extracted topics but reports no human relevance score, production search benefit, or business KPI.

**Notable vendor implementations/libraries:** `sklearn.decomposition.LatentDirichletAllocation`, Gensim, MALLET, and Spark ML `LDA`. See [supervised classical learning](01-supervised-classical.md) for linear discriminant analysis; the two LDAs have different objectives and supervision.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Apriori | Sparse baskets with short frequent sets | Transparent support-based pruning | Candidate explosion and repeated scans | Groceries / arulesViz study |
| FP-growth | Transactions with shared prefixes | Compressed pattern growth | Conditional-tree and output explosion | BMS-WebView-1 benchmark |
| Eclat | Transactions with manageable vertical postings | Efficient support intersections | Large transaction-ID lists | UCI Adult association-mining example |
| Latent Dirichlet allocation | Document-term counts | Mixed-topic document representation | Exchangeability and ambiguous topics | 20 Newsgroups topic extraction |

## 3.5 Density and anomaly modeling

Density estimation assigns relative or normalized mass under a model. Anomaly detection assigns unusualness relative to a reference population. These are related but not identical: high-dimensional density, typicality, rarity, and operational harm need not agree. An expensive legitimate transaction can be rare without being fraud.

**Outlier detection** usually fits a contaminated, unlabeled cohort and ranks unusual members of that same cohort. **Novelty detection** fits a reference cohort assumed mostly normal and scores later observations. The latter is sometimes called semi-supervised anomaly detection because normality is supplied through cohort selection; that usage differs from classification with a small labeled subset and a large unlabeled subset. See the [semi-supervised volume](03-semi-supervised.md) and the [official novelty/outlier distinction](https://scikit-learn.org/stable/modules/outlier_detection.html).

**Thresholds are decisions, not discoveries.** Choose the score direction explicitly, then calibrate against held-out normal data, adjudicated anomalies, review capacity, and costs of false alarms and misses. An empirical 99th-percentile threshold gives roughly 1% exceedance on its calibration reference, not a guaranteed future false-positive rate. A `contamination` setting usually fixes a score cutoff; it does not verify the true anomaly fraction or convert scores into probabilities. Label-informed threshold or hyperparameter selection must be disclosed even when the underlying fit is unsupervised. Monitor drift and avoid evaluating on data whose anomalies were used to tune the threshold.

### 3.5.1 Kernel density estimation

**Name:** Kernel density estimation (KDE).

**Category & sub-category:** Unsupervised learning; nonparametric density estimation and density-based scoring.

**Originating paper/vendor/year:** Murray Rosenblatt, [*Remarks on Some Nonparametric Estimates of a Density Function*, 1956](https://doi.org/10.1214/aoms/1177728190), and Emanuel Parzen, [*On Estimation of a Probability Density Function and Mode*, 1962](https://doi.org/10.1214/aoms/1177704472).

**Core mechanism:** For normalized Euclidean kernel $`K`$ and scalar bandwidth $`h`$, estimate


$$
\hat p(x)=\frac{1}{nh^d}\sum_{i=1}^{n}K\!\left(\frac{x-x_i}{h}\right).
$$


Each observation contributes a local bump. A bandwidth matrix generalizes isotropic smoothing, with the corresponding determinant normalization. Density integrates over a region to yield probability; a density value at one point is not its probability.

**Inputs/outputs and typical data types:** Continuous observations, especially low-dimensional measurements or coordinates. Outputs are estimated densities, log densities, samples for some kernels, and optionally thresholded unusualness scores. Standardization changes both geometry and density units; transforming back requires the appropriate Jacobian.

**Strengths and limitations:** Flexible without choosing a parametric component count. Bandwidth dominates the bias/variance trade-off: small bandwidths overfit individual samples, while large ones erase multimodality. Use held-out or leave-one-out selection rather than rewarding a sample's own increasingly narrow kernel. Boundary bias, sparse tails, and the curse of dimensionality matter. A low density may reflect the data-collection process rather than a real anomaly.

**Computational complexity / scalability notes:** Storing the reference observations costs $`O(nd)`$. A direct evaluation costs $`O(nd)`$ per query, or $`O(Gnd)`$ for $`G`$ query points. Trees accelerate favorable low-dimensional queries, often using controlled approximation; their advantage can vanish in high dimension. Naive leave-one-out bandwidth evaluation costs $`O(n^2d)`$ per candidate. Grid/FFT techniques require grid and kernel assumptions and can scale poorly with dimension.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn species-KDE example](https://scikit-learn.org/stable/auto_examples/neighbors/plot_species_kde.html) smooths observed locations of *Bradypus variegatus* and *Microryzomys minutus* from the Phillips and colleagues dataset. Latitude/longitude in radians feed separate Haversine-distance KDEs, yielding South American occurrence-concentration maps for exploration. Our rationale versus fitting a single Gaussian is accommodating several geographical concentrations. The code explicitly does not learn environmental suitability; observations become smoothed occurrence maps, not validated habitat predictions. Moreover, the [KDE API only guarantees density normalization for Euclidean metrics](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html), so these maps must not be advertised as calibrated probability per square kilometre. No conservation outcome or production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.neighbors.KernelDensity`, SciPy `gaussian_kde`, R `density`, and statsmodels KDE facilities. Their bandwidth parameterizations and multivariate capabilities differ.

### 3.5.2 Isolation Forest

**Name:** Isolation Forest.

**Category & sub-category:** Unsupervised learning; randomized partition-based anomaly scoring.

**Originating paper/vendor/year:** Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou, [*Isolation Forest*, ICDM 2008](https://doi.org/10.1109/ICDM.2008.17), followed by their expanded isolation-based anomaly-detection work.

**Core mechanism:** Train many trees on subsamples. At each node, select a feature and randomly split within its observed range. Observations isolated after fewer splits receive higher anomaly scores. The original normalized score is $`2^{-\mathbb E[h(x)]/c(\psi)}`$, where $`h(x)`$ includes leaf-size adjustment and $`c(\psi)`$ normalizes path lengths for subsample size $`\psi`$. Library score signs can differ from this convention.

**Inputs/outputs and typical data types:** Tabular numeric or appropriately encoded categorical data. Outputs are continuous unusualness scores and, after threshold selection, flags. A saved forest scores new observations without rebuilding training neighborhoods. The trees do not use task labels or ordinary supervised impurity reduction.

**Strengths and limitations:** Scales well with bounded subsamples and can find multivariate isolation without estimating a full density. It may miss dense anomalous groups or find harmless rare values. Axis-aligned splits are not rotation-invariant. Positive affine changes of a feature's units need not change ideal random-split partitions; nonlinear transformations and arbitrary ordinal encodings can. Thus automatic standardization is less central than for LOF, but feature design remains consequential. Scores are not calibrated anomaly probabilities.

**Computational complexity / scalability notes:** For $`F`$ trees and subsample size $`\psi`$, construction is approximately $`O(F\psi\log\psi)`$ with capped/balanced depths and fixed feature-selection costs, apart from reading/preparing the full input. Model storage is $`O(F\psi)`$, and depth-capped scoring costs $`O(F\log\psi)`$ per observation. More extensive per-node feature searches or uncapped, very unbalanced trees change these assumptions.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn outlier-estimator study](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html) uses KDDCup99's SA network-connection subset. Its displayed sample has 10,065 records and 338 injected-attack anomalies; categorical connection fields are encoded before fitting Isolation Forest without labels. Records become scores for ranking suspicious connections. The source reports slightly better ROC-AUC for Isolation Forest than its configured LOF on this dataset. Our rationale is inexpensive randomized partitions at this sample size. The benchmark samples using known labels and evaluates scores on the fitted contaminated cohort, not a future production stream; the attacks were generated in a controlled network. No live intrusion-detection or financial-loss KPI is established.

**Notable vendor implementations/libraries:** `sklearn.ensemble.IsolationForest` and scientific isolation-forest implementations such as `isotree`. Extended/hyperplane variants differ from the original axis-aligned forest.

### 3.5.3 Local outlier factor

**Name:** Local outlier factor (LOF).

**Category & sub-category:** Unsupervised learning; neighborhood-relative density-ratio anomaly scoring.

**Originating paper/vendor/year:** Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jorg Sander, [*LOF: Identifying Density-Based Local Outliers*, SIGMOD 2000](https://doi.org/10.1145/335191.335388).

**Core mechanism:** Define reachability distance from $`x`$ to neighbor $`o`$ as $`\max\{k\text{-distance}(o),D(x,o)\}`$. Local reachability density is the inverse mean reachability distance to the neighborhood. LOF compares average neighbor density with the observation's own density. A ratio near one indicates similar local density; a much larger value suggests local sparsity. Ties can make the mathematical neighborhood contain more than exactly $`k`$ observations.

**Inputs/outputs and typical data types:** Vectors or a valid distance representation. Outputs are relative-density scores and thresholded flags for a fitted cohort. Numeric scaling and categorical encoding are particularly important: an arbitrary integer ordering changes neighbors. Scikit-learn exposes the negative of the conventional outlier factor for training samples.

**Strengths and limitations:** Can detect a sparse point near a dense group even when its absolute density is not globally unusual. Neighborhood size defines the comparison population. Very small neighborhoods are noisy; very large ones obscure local structure. A sufficiently dense anomalous group can support itself and escape detection. Neither LOF near one nor a chosen contamination percentile establishes semantic normality.

**Computational complexity / scalability notes:** Brute-force neighbor discovery costs $`O(n^2d)`$, followed by roughly $`O(nk)`$ reachability and ratio computations. Useful low-dimensional indexes can reduce neighbor-search work; high dimensions often defeat pruning. Storing neighbor indices and distances requires $`O(nk)`$, without requiring the full pairwise matrix. Approximate neighbors change scores and should be validated, especially near the decision threshold.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn Ames Housing outlier experiment](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html) defines a deliberately constructed anomaly task using sale price divided by **lot area**, not living-floor area. It retains ratios below 40 or above 70 USD per square foot, giving 2,714 records and thirty high-ratio proxies for anomalies. Encoded housing attributes become LOF rankings; the published study finds better ROC-AUC than its Isolation Forest configuration here. Our rationale is local comparison among heterogeneous properties. Labels help construct the benchmark and set neighborhood scale, so this is not fully label-blind model selection. The result does not establish fraud or appraisal error, and no production KPI is reported.

**Notable vendor implementations/libraries:** `sklearn.neighbors.LocalOutlierFactor`, ELKI, and R `dbscan::lof`. Scikit-learn's `novelty=True` enables new-point scoring against a frozen reference; [do not apply that prediction path to the training data and equate it with `fit_predict`](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-with-lof).

### 3.5.4 One-class SVM

**Name:** One-class support vector machine (one-class SVM).

**Category & sub-category:** Unsupervised/one-class learning; kernel support estimation and novelty detection, not ordinary supervised binary classification.

**Originating paper/vendor/year:** Bernhard Scholkopf, John C. Platt, John Shawe-Taylor, Alexander J. Smola, and Robert C. Williamson, [*Estimating the Support of a High-Dimensional Distribution*, Neural Computation 2001](https://doi.org/10.1162/089976601750264965), following their earlier novelty-detection work.

**Core mechanism:** In feature space, solve


$$
\min_{w,\rho,\xi\ge0}\frac12\|w\|^2+\frac{1}{\nu n}\sum_i\xi_i-\rho,
\qquad
w^\top\phi(x_i)\ge\rho-\xi_i.
$$


The decision function $`f(x)=w^\top\phi(x)-\rho`$ estimates a region containing much of the reference distribution. A kernel, commonly RBF, supplies nonlinear boundaries without explicit high-dimensional features. No labeled negative class is required.

**Inputs/outputs and typical data types:** A reference set of numeric observations, a kernel and scale, and $`\nu\in(0,1]`$. Outputs are support vectors, decision scores, and novelty flags. Feature normalization must be fitted consistently on the reference data; RBF bandwidth changes the boundary substantially.

**Strengths and limitations:** Useful when representative normal observations exist but representative anomalies do not. Contaminated or unrepresentative training cohorts can distort the boundary. Under the standard solution conditions, $`\nu`$ bounds the fraction of training margin errors from above and the support-vector fraction from below; it is not a measured anomaly prevalence or a guaranteed future false-alarm rate. Scores are not normalized densities or probabilities. Threshold calibration and kernel selection remain essential.

**Computational complexity / scalability notes:** Forming a full dense RBF Gram matrix costs $`O(n^2d)`$ and $`O(n^2)`$ storage. Kernel optimization is a constrained quadratic program with solver-, tolerance-, and conditioning-dependent iterations; there is no universal cubic total-runtime guarantee. Bounded kernel caches trade memory for recomputation. With $`s`$ support vectors, dense RBF scoring costs $`O(sd)`$ per query. Linear or finite-feature approximations have different scaling and accuracy.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn species-distribution example](https://scikit-learn.org/stable/auto_examples/applications/plot_species_distribution_modeling.html) uses observed locations and fourteen environmental covariates for *Bradypus variegatus* and *Microryzomys minutus*. Each species' presence-only training covariates are standardized and fitted separately with RBF `gamma=0.5`, `nu=0.1`; environmental grid cells become relative suitability scores. This is scikit-learn's model on the Phillips dataset, not a claim that Phillips's original method was one-class SVM. The displayed ROC-AUCs are 0.868443 and 0.993919 using held-out occurrences versus 10,000 randomly sampled grid-background points with seed thirteen. Background is not verified absence, and the code gives ocean cells low map scores; these are protocol-dependent presence/background results, with no controlled alternative-model comparison or production KPI. Our rationale versus a binary classifier is not inventing negative training examples.

**Notable vendor implementations/libraries:** LIBSVM, `sklearn.svm.OneClassSVM`, R `e1071`, and `sklearn.linear_model.SGDOneClassSVM` for the distinct linear/stochastic formulation.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Kernel density estimation | Low-dimensional continuous data | Flexible local density estimate | Bandwidth and dimensionality sensitivity | South American species occurrence maps |
| Isolation Forest | Large encoded tabular datasets | Cheap subsampled isolation scores | Dense anomaly groups and feature dependence | KDDCup99 SA connection benchmark |
| Local outlier factor | Metric data with heterogeneous local density | Relative rather than global unusualness | Neighbor selection and transductive semantics | Ames Housing constructed anomaly task |
| One-class SVM | Mostly normal reference observations | Nonlinear novelty boundary without negatives | Kernel tuning, contamination, and solver cost | Presence-only species-distribution study |

## Coverage and continuation manifest

- **3.1.1-3.1.6:** Six centroid, prototype, mixture, and hierarchical entries: k-means, mini-batch k-means, k-medoids/PAM, Gaussian mixtures with EM, agglomerative clustering, and BIRCH.
- **3.2.1-3.2.6:** Six density/connectivity entries: DBSCAN, OPTICS, HDBSCAN, spectral clustering, mean shift, and affinity propagation.
- **3.3.1-3.3.9:** Nine representation entries: PCA, kernel PCA, ICA, NMF, t-SNE, UMAP, Isomap, LLE, and SOM. SOM includes the nine additional neural-network fields; the other entries describe their non-neural representative formulations.
- **3.4.1-3.4.4:** Four pattern/topic entries: Apriori, FP-growth, Eclat, and latent Dirichlet allocation.
- **3.5.1-3.5.4:** Four density/anomaly entries: KDE, Isolation Forest, LOF, and one-class SVM.

**Total: 29 algorithm entries and five category comparison tables.** Every worked example identifies a public study, documented application, or named dataset; supplementary toy arithmetic is labeled and is not a commercial metric. Catalogue counts, demonstration settings, and protocol-specific results are not interchangeable with deployment impact.

Continue with [neural unsupervised and self-supervised learning](05-unsupervised-neural.md) and [foundation-model pretraining](06-foundation-models.md). Gaussian mixture density components are not automatically conditional neural experts: consult the [MoE model catalogue](07-moe-models.md) and the [dedicated MoE deep dive](08-moe-deep-dive.md) for routing, expert capacity, and training details. Related context appears in [semi-supervised learning](03-semi-supervised.md), [supervised classical learning](01-supervised-classical.md), the [comparative guide](09-comparative-guide.md), and the [glossary](10-glossary.md).

This is a bounded first-edition volume, not an assertion that these methods exhaust unsupervised learning. Further non-required depth not developed here includes robust and sparse PCA, full factor-analysis treatments, diffusion maps and standalone MDS, CURE/ROCK/CHAMELEON, Dirichlet-process and Bayesian nonparametric mixtures, subspace and co-clustering, streaming density-cluster maintenance, closed/maximal/high-utility and sequential pattern mining, dynamic/correlated topic models, extreme-value tail calibration, and formal guarantees for approximate neighbor search. Neural extensions of otherwise classical methods require their own architecture and supervision descriptions rather than inheriting claims from the entries above.
