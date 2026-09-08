# 1. Supervised Learning Algorithms: Classical Models

Classical supervised learning is not a single statistical philosophy. A linear model compresses observations into coefficients; a neighbor method retains examples; a tree partitions feature space; a probabilistic model specifies distributions; and a structured predictor chooses mutually dependent outputs. What unites the representative formulations in this chapter is a known training target: a measurement, class, sequence annotation, relevance judgment, or observed event time with a censoring indicator.

This bounded first edition covers 28 entries in sections 1.1-1.4. It uses non-neural instantiations throughout. A sigmoid in logistic regression, a differentiable objective, or a kernel does not by itself make an algorithm a neural network. Conversely, the same model family can appear in other learning settings: a hidden Markov model can be fitted without state labels, and a Bayesian network can be elicited from experts rather than learned from labeled cases.

**Evidence policy, 2026-09-08.** Origins, implementations, and applications are separate claims. Author manuscripts and historical implementation examples retain their own dates and protocols; an old result is not a claim about a library's latest release. "Research benchmark" includes an explicitly identified teaching experiment on public data. Neither a benchmark nor a research prototype establishes production deployment. An application report can establish use without establishing financial benefit. Numerical results below are source-reported, not new experiments performed for this book.

Let $`n`$ denote training examples and $`d`$ input features after the stated preprocessing. Additional symbols are defined locally. Time bounds count arithmetic operations unless stated otherwise; memory bounds count stored scalar values, not bytes. Hyperparameter search multiplies the cost of a single fit. Train/test separation must also apply to scaling, feature selection, imputation, categorical target statistics, and any learned representation. Reported discrimination is not necessarily calibration, causation, fairness, or operational value.

## 1.1 Linear, generalized, and discriminant models

These models make their structural assumptions unusually visible. Regularization changes which coefficient estimates are preferred; links change how a linear predictor describes a response; smooth functions relax linearity; and discriminant models derive a classifier from assumptions about class-conditional distributions.

### 1.1.1 Ordinary least squares

**Name:** Ordinary least squares regression, usually abbreviated OLS.

**Category & sub-category:** Supervised learning; linear regression for continuous targets.

**Originating paper/vendor/year:** Legendre published the least-squares method in 1805; Gauss's 1809 treatment belongs to its early development. The [public translation of Legendre's appendix](https://www.york.ac.uk/depts/maths/histstat/legendre.pdf) documents its astronomical and geodetic roots. Modern statistics packages are implementations, not inventors.

**Core mechanism:** Estimate an intercept $`b`$ and coefficients $`\beta`$ by minimizing $`\sum_i(y_i-b-x_i^\top\beta)^2`$. The fitted values are an orthogonal projection onto the design matrix's column space. A QR or singular-value decomposition is normally preferable to explicitly computing $`(X^\top X)^{-1}`$. "Linear" means linear in the coefficients: polynomial terms, indicators, or prespecified transformations may still be features. Gaussian errors are not required to calculate OLS; they support particular likelihood and inferential interpretations.

**Inputs/outputs and typical data types:** A numeric design matrix, possibly obtained from categorical encoding, and a real-valued response. Outputs are coefficients and conditional-mean predictions. Confidence intervals for the mean and prediction intervals for a new observation answer different questions and require additional assumptions about errors.

**Strengths and limitations:** OLS is a transparent baseline with inexpensive prediction and interpretable conditional associations. Correlated columns can make coefficients unstable; exact dependencies make the coefficient solution non-unique. Outliers have large influence under squared loss, and an unmodeled nonlinear relationship can dominate the error. Standard-error calculations must accommodate heteroskedasticity or dependent observations where appropriate. Coefficients do not establish an intervention's causal effect.

**Computational complexity / scalability notes:** For dense $`n\ge d`$, a conventional QR or SVD fit costs $`O(nd^2)`$ time; storing the input costs $`O(nd)`$. Prediction is $`O(d)`$ per example. Sparse iterative solvers instead depend on nonzero entries, conditioning, and convergence tolerance; these are not universally one-pass algorithms.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn diabetes regression example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge.html) uses standardized body-mass index to predict the dataset's quantitative disease-progression measurement. Its single-feature model, with the final 20 observations reserved for testing, reports test MSE **2548.07** and $`R^2`$ **0.47**. BMI enters a fitted straight line; the output is a progression estimate; the demonstration's decision is whether this baseline adequately predicts held-out measurements, not which treatment to prescribe. Technically, OLS makes the one-feature relationship inspectable, whereas a tree could model thresholds at greater variance. That rationale is not a claim that clinicians selected this model. The example does not demonstrate clinical deployment, improved patient outcomes, or a production KPI.

**Notable vendor implementations/libraries:** R `stats::lm`, statsmodels `OLS`, scikit-learn `LinearRegression`, and MATLAB linear-model routines. Their defaults for intercepts, missing values, weighting, and inference differ.

### 1.1.2 Ridge regression

**Name:** Ridge regression; linear regression with a squared $`L_2`$ penalty.

**Category & sub-category:** Supervised learning; regularized linear regression.

**Originating paper/vendor/year:** Hoerl and Kennard's [1970 paper, "Ridge Regression: Biased Estimation for Nonorthogonal Problems"](https://doi.org/10.1080/00401706.1970.10488634), established the ridge-regression formulation in statistics. Related quadratic regularization has a separate inverse-problems history, including Tikhonov regularization.

**Core mechanism:** Add $`\lambda\|\beta\|_2^2`$ to squared prediction error, usually leaving the intercept unpenalized. For centered data and an unnormalized residual sum of squares, the solution satisfies $`(X^\top X+\lambda I)\beta=X^\top y`$. Positive $`\lambda`$ stabilizes poorly determined directions rather than deleting individual variables. Standardization matters because the penalty measures coefficient size in the chosen units. Library definitions may divide the loss by $`n`$, changing the numerical interpretation of $`\lambda`$.

**Inputs/outputs and typical data types:** Dense or sparse numerical features and continuous labels. The model returns a generally dense coefficient vector and scalar or multi-output predictions. One-hot categorical designs are possible, but the chosen coding and treatment of the intercept affect the penalty.

**Strengths and limitations:** Ridge trades some bias for lower estimation variance and is useful when many correlated features each contain signal. Unlike lasso, it normally retains all predictors. This can improve stability but does not provide a compact variable-selection explanation. It does not automatically correct nonlinear misspecification, bad measurements, or distribution shift. A Bayesian interpretation uses a Gaussian coefficient prior; a point estimate alone is not a full posterior uncertainty calculation.

**Computational complexity / scalability notes:** Forming and solving dense primal normal equations takes $`O(nd^2+d^3)`$ time and $`O(d^2)`$ working memory beyond the data. QR/SVD and iterative alternatives have different numerical trade-offs. When $`d\gg n`$, a dual solve can be preferable. Dense prediction remains $`O(d)`$ per case.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Zou and Hastie's [August 2004 elastic-net manuscript, section 4 and Table 1](https://hastie.su.domains/Papers/elasticnet.pdf), compares ridge on Stamey and colleagues' prostate-cancer data. Eight clinical measurements enter a model of log prostate-specific antigen; 67 observations form the training set and 30 the test set, with ten-fold tuning confined to training. Ridge retains all eight variables and reports test MSE **0.566**, compared with **0.586** for OLS, in squared log-response units. The workflow is measurements -> shrunken linear predictor -> estimated log PSA -> held-out predictive comparison. Correlation among clinical measurements makes ridge a technically sensible alternative to unstable OLS; the manuscript does not establish a hospital's model-selection policy. This small historical experiment is not diagnostic validation, and production clinical KPIs are unreported.

**Notable vendor implementations/libraries:** scikit-learn `Ridge`/`RidgeCV`, R `glmnet` with `alpha=0`, and Spark ML linear regression with a pure $`L_2`$ penalty.

### 1.1.3 Lasso

**Name:** Least absolute shrinkage and selection operator, or lasso.

**Category & sub-category:** Supervised learning; sparse regularized linear regression.

**Originating paper/vendor/year:** Robert Tibshirani's [1996 "Regression Shrinkage and Selection Via the Lasso"](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). Sparse estimation also has related signal-processing and optimization lineages; the name does not encompass every $`L_1`$-regularized method.

**Core mechanism:** Minimize $`\|y-b-X\beta\|_2^2/(2n)+\lambda\|\beta\|_1`$. The corner at zero in the absolute-value penalty permits exactly zero coefficients. Coordinate descent updates one coefficient at a time, using soft thresholding of its partial residual correlation. A regularization path describes how the active set changes with $`\lambda`$; it is not a ranking of causal importance. Least-angle regression is another route to parts of that path.

**Inputs/outputs and typical data types:** Standardized numerical or encoded tabular features with a continuous response, especially designs containing many potentially irrelevant variables. Outputs include predictions and the selected nonzero coefficient set. Sparse input matrices can avoid materializing absent features.

**Strengths and limitations:** Lasso combines prediction and variable selection and can make a high-dimensional model easier to inspect. Among strongly correlated predictors, however, small data perturbations may change which variable survives. Selection consistency requires conditions that ordinary cross-validation does not guarantee. Inference performed after selecting variables on the same observations needs special care. Sparsity can also discard weak but collectively useful signals that ridge retains.

**Computational complexity / scalability notes:** With residual updates, one dense coordinate-descent sweep costs approximately $`O(nd)`$; total training is $`O(I nd)`$ for $`I`$ sweeps at one penalty value, ignoring screening overhead. Sparse implementations depend on $`\operatorname{nnz}(X)`$. Convergence varies with correlation, tolerances, and warm starts. If $`s`$ coefficients remain nonzero, prediction can cost $`O(s)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In the same [prostate-cancer experiment in Zou and Hastie's August 2004 manuscript](https://hastie.su.domains/Papers/elasticnet.pdf), lasso predicts log PSA from eight candidate clinical measurements using the 67/30 training/test split and training-only ten-fold tuning. Table 1 reports test MSE **0.499** and five selected predictors: log cancer volume, log prostate weight, log benign prostatic hyperplasia, seminal-vesicle invasion, and percentage Gleason score 4 or 5. Thus patient measurements -> selected linear combination -> estimated log PSA -> comparison with held-out observed log PSA is a concrete supervised workflow. The technical advantage over ridge is a smaller retained measurement set, not proven cheaper clinical practice. These selected associations do not show that changing a predictor changes disease, and no deployment, diagnostic safety claim, or production KPI follows from the reported MSE.

**Notable vendor implementations/libraries:** scikit-learn `Lasso`, `LassoCV`, and `LassoLars`; R `glmnet` with `alpha=1`; and Spark ML linear regression with a pure $`L_1`$ penalty.

### 1.1.4 Elastic net

**Name:** Elastic-net regularization.

**Category & sub-category:** Supervised learning; combined sparse and quadratic regularization for linear or generalized linear models.

**Originating paper/vendor/year:** Hui Zou and Trevor Hastie, [2005, "Regularization and Variable Selection Via the Elastic Net"](https://doi.org/10.1111/j.1467-9868.2005.00503.x). The publicly available [author manuscript revised August 2004](https://hastie.su.domains/Papers/elasticnet.pdf) supplies the historical examples used here.

**Core mechanism:** A common modern regression objective is


$$
\frac{1}{2n}\|y-b-X\beta\|_2^2+
\lambda\left[\alpha\|\beta\|_1+\frac{1-\alpha}{2}\|\beta\|_2^2\right].
$$


The $`L_1`$ component creates zeros; the $`L_2`$ component stabilizes correlated predictors and encourages a grouping effect. Both overall penalty and mixing proportion require selection. The original paper distinguishes a "naive elastic net" from a rescaled estimator that corrects double shrinkage. Consequently, its parameterization should not be equated mechanically with a modern library's `alpha` argument.

**Inputs/outputs and typical data types:** Numeric designs with correlated columns, including gene-expression profiles and sparse text features. The representative squared-loss model returns real scores; with a logistic likelihood, the same penalty gives a different estimator returning class probabilities.

**Strengths and limitations:** Elastic net offers a compromise between ridge's stable dense fit and lasso's sparse fit. Under a positive quadratic penalty, it can select groups even when the feature count exceeds the sample count. The selected group need not be biologically or causally meaningful. Tuning two penalty dimensions, preprocessing leakage, and unstable small-sample validation remain concerns.

**Computational complexity / scalability notes:** A residual-maintaining coordinate-descent sweep is $`O(nd)`$ for dense data, with total cost depending on sweeps and the two-dimensional tuning grid. Storage can follow the sparse design. Prediction costs $`O(s)`$ with $`s`$ active coefficients; constructing the features is an additional cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [manuscript's leukemia experiment](https://hastie.su.domains/Papers/elasticnet.pdf) uses the Golub gene-expression study to distinguish acute lymphoblastic from acute myeloid leukemia. Importantly, this is **squared-loss regression on 0/1 labels followed by a 0.5 threshold**, not elastic-net logistic regression. Gene screening is repeated inside each training fold; ten-fold tuning uses 38 training samples, and 34 separate samples test the selected model. The reported model selects **45 genes**, with **3/38** cross-validation errors and **0/34** test errors (Table 4). Expression measurements -> screened and regularized score -> thresholded subtype -> research classification is the complete path. Grouped selection is technically attractive relative to choosing one arbitrary correlated gene, but zero errors on this small test set is not zero clinical risk. Clinical deployment and patient-outcome KPIs are unreported.

**Notable vendor implementations/libraries:** R `glmnet`, scikit-learn `ElasticNet`/`ElasticNetCV`, and elastic-net penalties in suitable generalized-linear-model solvers.

### 1.1.5 Logistic regression: binary and multinomial

**Name:** Logistic regression, distinguishing binary logistic, multinomial softmax, and one-versus-rest constructions.

**Category & sub-category:** Supervised learning; generalized linear classification.

**Originating paper/vendor/year:** D. R. Cox's [1958 "The Regression Analysis of Binary Sequences"](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x) is a foundational modern binary treatment, not the invention of every logistic model. Multinomial choice models have additional statistical and econometric lineages. Software vendors implement these formulations.

**Core mechanism:** Binary logistic regression models $`\Pr(y=1\mid x)=\sigma(b+x^\top\beta)`$, fitting coefficients by binomial negative log-likelihood, often with regularization. Multinomial regression jointly fits class scores $`z_c`$ and probabilities $`\exp(z_c)/\sum_j\exp(z_j)`$ under multiclass cross-entropy. An identifiability constraint or regularization addresses the invariance to adding a shared score. One-versus-rest instead trains separate binary problems; its independent probabilities are not automatically the jointly fitted multinomial probabilities. Ordinal logistic regression imposes another structure and is outside this entry's detailed scope.

**Inputs/outputs and typical data types:** Numerical or encoded categorical features and binary or categorical labels. Outputs are class scores, estimated probabilities, and decisions produced by a threshold or decision rule. The threshold can reflect costs rather than defaulting to 0.5.

**Strengths and limitations:** Efficient, inspectable, and often strong for sparse features. It assumes linear log odds unless feature interactions or basis expansions are supplied. Complete separation can make unregularized estimates diverge. Calibration, class imbalance, and distribution shift require explicit checks. A coefficient's odds ratio is a conditional model association, not a risk ratio or causal effect.

**Computational complexity / scalability notes:** For $`C`$ jointly modeled classes, a dense first-order gradient pass costs $`O(ndC)`$, with $`O(dC)`$ model storage. Binary Newton/IRLS methods may require $`O(nd^2+d^3)`$ per iteration. No fixed iteration count applies across datasets.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 classification lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html) predicts S&P 500 daily direction in `Smarket`, using `Lag1` and `Lag2`, training before 2005 and testing on 252 days in 2005. The binary model correctly classifies **141/252**, or **55.95%**, exactly the accuracy of always predicting "Up" on that test period. Its supplied input `(1.2, 1.1)` produces probability **0.4791462**, hence "Down" at the demonstration's threshold. Lagged returns -> probability -> direction illustrates interpretable classification versus a nonlinear tree; it does not establish profitable trading. Same-day `Today` must not enter the predictor. No production profit, transaction-cost result, or trading deployment is reported.

**Notable vendor implementations/libraries:** R `glm(family=binomial)` for binary models; R `nnet::multinom` for classical multinomial regression; scikit-learn `LogisticRegression`; statsmodels `Logit` and `MNLogit`.

### 1.1.6 Generalized additive models

**Name:** Generalized additive models, or GAMs.

**Category & sub-category:** Supervised learning; interpretable nonlinear extensions of generalized linear models.

**Originating paper/vendor/year:** Trevor Hastie and Robert Tibshirani, [1986, "Generalized Additive Models"](https://doi.org/10.1214/ss/1177013604), followed by their 1990 monograph. Later implementations, notably Simon Wood's `mgcv`, add substantial smoothing-parameter estimation and computational methods.

**Core mechanism:** Replace a purely linear predictor by $`g(\mathbb E[y\mid x])=b+\sum_j f_j(x_j)`$, optionally with explicitly specified interaction smooths. Each $`f_j`$ may use a spline basis; roughness penalties control its wiggliness. Identifiability constraints, such as centering smooths, separate them from the intercept. Backfitting, penalized iteratively reweighted least squares, and smoothing-parameter optimization are alternative or combined fitting components, not one universal GAM algorithm.

**Inputs/outputs and typical data types:** Continuous, categorical, seasonal, or spatial covariates with a response family appropriate to the target: Gaussian measurements, binomial outcomes, or Poisson counts, for example. Outputs include response predictions, smooth partial effects, and assumption-dependent uncertainty intervals.

**Strengths and limitations:** GAMs capture nonlinear component effects while preserving an inspectable decomposition. They often need fewer observations than an unrestricted interaction model. Additivity can nevertheless miss important joint effects; adding many tensor-product smooths sacrifices simplicity. Correlated smooth terms cause concurvity, the nonlinear analogue of collinearity. Extrapolation depends on the basis and constraints, not on a general guarantee that a smooth curve continues sensibly.

**Computational complexity / scalability notes:** Let $`q`$ be the total number of basis coefficients, including parametric terms. A dense penalized weighted-least-squares iteration can cost $`O(nq^2+q^3)`$, with $`O(nq+q^2)`$ storage. Local bases, sparsity, discretization, or streaming construction can change this substantially. Basis evaluation and prediction are typically $`O(q)`$ per example.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 nonlinear-modeling lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch7-nonlin-lab.html) analyzes the `Wage` data using year, age, and education. The published comparison prefers a model with a linear year term, a smooth age effect, and education indicators: adding year as a nonlinear smooth to that model has an ANOVA $`p`$-value of **0.3485661**. This is an in-sample model-comparison result, not held-out accuracy. Covariates -> additive wage curve -> estimated wage -> inspection of the age and calendar-year associations is the demonstrated workflow. Relative to straight-line OLS, the smooth age term permits a changing slope without an opaque collection of tree interactions. The lab does not establish causal returns to education, a wage-setting deployment, or a production economic KPI.

**Notable vendor implementations/libraries:** R `mgcv` and `gam`; Python pyGAM and statsmodels GAM support. These packages do not share identical bases, default penalties, or inferential conventions.

### 1.1.7 Linear discriminant analysis

**Name:** Linear discriminant analysis, or LDA; not latent Dirichlet allocation.

**Category & sub-category:** Supervised learning; Gaussian generative classification and supervised dimensionality reduction.

**Originating paper/vendor/year:** R. A. Fisher's [1936 "The Use of Multiple Measurements in Taxonomic Problems"](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) is the foundational linear-discrimination reference. Modern Gaussian generative classification is a related formulation, not a claim that every implementation follows the original procedure literally.

**Core mechanism:** Estimate a mean $`\mu_c`$ for each class, a shared within-class covariance $`\Sigma`$, and class priors $`\pi_c`$. Comparing Gaussian posteriors cancels the quadratic term common to all classes, leaving linear scores $`x^\top\Sigma^{-1}\mu_c-\frac12\mu_c^\top\Sigma^{-1}\mu_c+\log\pi_c`$. The associated projection separates class means relative to within-class scatter. Unlike PCA, that projection uses labels.

**Inputs/outputs and typical data types:** Numeric measurements and class labels. Outputs can include posterior class probabilities, class decisions, and at most $`\min(d,C-1)`$ discriminant coordinates for $`C`$ classes. The feature distribution within each class is modeled, rather than only the class probability given features.

**Strengths and limitations:** A pooled covariance reduces estimation variance and often works well with modest labeled datasets. Multiclass prediction is natural. Gaussian class shapes and equal covariances can be poor approximations, and empirical covariance becomes unstable in high dimensions. Shrinkage or alternative covariance estimation can help; it does not fix mislabeled cases or a completely inappropriate density model. A useful projection need not imply calibrated posteriors.

**Computational complexity / scalability notes:** A dense covariance-based fit costs $`O(nd^2+d^3+Cd^2)`$, with $`O(d^2+Cd)`$ model storage. Once linear score vectors are precomputed, prediction is $`O(Cd)`$ per example. SVD implementations can avoid explicitly constructing the covariance and have different advantages when $`d`$ is large.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 `Smarket` experiment](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html) fits LDA using two lagged market returns before 2005 and predicts the 252 daily directions in 2005. Its confusion matrix is identical to binary logistic regression's: **141 correct**, or **55.95%**. For the first test observation, the reported posterior for "Down" is **0.4901792**, so the classifier chooses "Up." The path is lagged numeric returns -> shared-covariance posterior -> direction -> out-of-period comparison. Pooling covariance is a technical reason to prefer LDA over the more variable QDA when labels are scarce, not an investment firm's documented selection rationale. This benchmark does not outperform the always-Up accuracy baseline, establish a trading strategy, or report production profit.

**Notable vendor implementations/libraries:** R `MASS::lda` and scikit-learn `LinearDiscriminantAnalysis`; their covariance, prior, and numerical-solver options require attention.

### 1.1.8 Quadratic discriminant analysis

**Name:** Quadratic discriminant analysis, or QDA.

**Category & sub-category:** Supervised learning; class-specific Gaussian generative classification.

**Originating paper/vendor/year:** QDA belongs to classical normal-theory discrimination rather than to a software vendor. C. A. B. Smith's ["Some Examples of Discrimination"](https://doi.org/10.1111/j.1469-1809.1946.tb02368.x), dated 1946 in the journal's DOI metadata, is an early lineage reference; no unique invention claim for all QDA variants is made here. The formulation below is the standard [Gaussian Bayes rule](https://scikit-learn.org/stable/modules/lda_qda.html).

**Core mechanism:** Estimate a separate covariance $`\Sigma_c`$, mean $`\mu_c`$, and prior for every class. The discriminant contains $`-\frac12\log|\Sigma_c|-\frac12(x-\mu_c)^\top\Sigma_c^{-1}(x-\mu_c)+\log\pi_c`$. Because covariance now varies by class, the quadratic terms no longer cancel. Classification boundaries can curve even though the model is not a neural network. Restricting each covariance to a diagonal matrix produces the usual class-specific Gaussian naive-Bayes formulation.

**Inputs/outputs and typical data types:** Numeric measurements and class labels, with enough observations in each class to estimate its covariance. Outputs are posterior probabilities and categorical predictions. The model represents ellipsoidal distributions of measurements, not arbitrary multimodal class densities.

**Strengths and limitations:** QDA captures different correlations and spreads across classes, providing more flexibility than LDA without a general nonlinear optimizer. That flexibility requires roughly $`C d(d+1)/2`$ covariance entries before symmetry is exploited computationally. Small or imbalanced classes can yield singular or noisy estimates. Regularized discriminant methods interpolate toward simpler covariance assumptions. Raw QDA is generally unsuitable when class sizes are small relative to dimension.

**Computational complexity / scalability notes:** With dense covariances, fitting is approximately $`O(nd^2+Cd^3)`$; model storage is $`O(Cd^2)`$. Prediction costs $`O(Cd^2)`$ per case when evaluating full quadratic forms. These bounds exclude feature engineering and assume conventional dense factorizations.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** In the [ISLR2 `Smarket` lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html), the same two lagged returns and pre-2005 training period produce **151 correct predictions out of 252** in 2005: **59.92%**, compared with LDA's **55.95%** on that exact test period. The confusion matrix contains 30 correctly predicted Down days and 121 correctly predicted Up days. Lagged returns -> two class-specific Gaussian models -> posterior direction -> held-out comparison is the worked path. Allowing different covariance shapes is a technical explanation for QDA's extra capacity, not proof that markets truly follow those densities. A single short test period does not establish persistent advantage, statistical arbitrage, or positive net returns. Production KPIs are unreported.

**Notable vendor implementations/libraries:** R `MASS::qda` and scikit-learn `QuadraticDiscriminantAnalysis`. Regularized variants and singular-covariance handling differ between implementations.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Ordinary least squares | Numeric features, continuous labels | Transparent conditional-mean baseline | Collinearity and squared-loss sensitivity | Diabetes progression from BMI; teaching benchmark |
| Ridge regression | Correlated numeric measurements | Stable dense coefficient estimates | Does not select variables | Historical prostate log-PSA prediction |
| Lasso | Many potentially irrelevant features | Joint prediction and sparse selection | Unstable choices within correlated groups | Five-variable prostate log-PSA model |
| Elastic net | High-dimensional correlated features | Grouped shrinkage with sparsity | Two-dimensional tuning and selection uncertainty | Golub leukemia-expression benchmark |
| Logistic regression | Encoded tables or sparse vectors | Direct binary/multinomial probabilities | Linear log odds unless features are expanded | `Smarket` direction benchmark; no trading KPI |
| Generalized additive models | Smooth continuous effects plus factors | Inspectable nonlinear effects | Additivity can miss interactions | `Wage` age/year/education analysis |
| Linear discriminant analysis | Numeric classes with shared covariance | Efficient multiclass generative model | Pooled Gaussian assumptions | `Smarket` shared-covariance classifier |
| Quadratic discriminant analysis | Numeric classes with different spreads | Curved boundaries with explicit densities | Per-class covariance data requirements | `Smarket` class-specific covariance comparison |

## 1.2 Trees and ensembles

A tree learns conditions; an ensemble learns how to combine many such conditions. Bagging primarily reduces variance through resampling and aggregation. Boosting sequentially improves a loss, often reducing bias as well. XGBoost, LightGBM, and CatBoost are distinct algorithm-and-system contributions within gradient boosting, not independent inventions of decision trees.

For this section, $`M`$ denotes trees, $`h`$ maximum tree depth, $`q`$ candidate features at a split, and $`v`$ stored nodes per tree. A balanced tree need not result from unconstrained greedy training. Complexity statements distinguish candidate evaluation, sorting, and prediction; categorical search and missing-value handling can add work.

### 1.2.1 CART

**Name:** Classification and regression trees, or CART.

**Category & sub-category:** Supervised learning; binary recursive partitioning.

**Originating paper/vendor/year:** Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone, *Classification and Regression Trees*, 1984. The [scikit-learn tree documentation](https://scikit-learn.org/stable/modules/tree.html) identifies its optimized CART implementation and distinguishes CART from ID3, C4.5, and C5.0.

**Core mechanism:** Repeatedly choose a feature and split that improve a node's objective, such as Gini impurity for classification or squared error for regression. Each terminal node supplies a class distribution or target estimate. Cost-complexity pruning balances fit against leaf count, producing nested subtrees from which validation selects an appropriate size. This greedy procedure does not generally find the globally optimal tree. Categorical partitioning and surrogate splits are features of particular CART formulations, not universal library capabilities.

**Inputs/outputs and typical data types:** Tabular features and categorical or continuous targets. Numeric thresholds work directly; nominal variables may require encoding or native categorical support. The output is a sequence of conditions ending at a leaf prediction. Missing-value behavior must be checked for the chosen implementation.

**Strengths and limitations:** Trees expose local decision rules, model interactions without prespecifying products, and need little numerical scaling. A small tree can be inspected end to end. However, minor data changes can produce a different hierarchy, axis-aligned boundaries can approximate oblique relationships inefficiently, and unpruned trees overfit. Regression leaves do not extrapolate a continuous trend beyond their fitted values. A rule's readability does not prove that its variables are causal or fair.

**Computational complexity / scalability notes:** With reusable sorted numeric features and a balanced tree, split scanning can be $`O(dn\log n)`$; sorting independently at every node may instead yield $`O(dn\log^2 n)`$. Deeply unbalanced growth is worse. Prediction is $`O(h)`$ for one fully observed example; tree storage is $`O(v)`$, excluding training data.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn pruning demonstration](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html) classifies the Wisconsin Diagnostic Breast Cancer measurements derived from breast-mass cell nuclei, using the library's default 75%/25% training/test split with `random_state=0`. Measurements -> threshold path -> benign/malignant label -> comparison with recorded labels is the benchmark workflow, not a clinical recommendation. The documented unpruned tree reaches **100% training accuracy** but only **88% testing accuracy**; pruning improves the displayed held-out curve. Because the example uses that curve to choose pruning strength, it serves as validation, not an untouched final clinical test. A tree permits inspection of threshold interactions that logistic regression would need to encode explicitly; that is a technical rationale rather than a hospital's documented choice. Production diagnostic outcomes are unreported.

**Notable vendor implementations/libraries:** R `rpart`, scikit-learn `DecisionTreeClassifier`/`DecisionTreeRegressor`, and CART-derived tree learners in Spark ML. Implementation options are not identical to the complete 1984 system.

### 1.2.2 C4.5

**Name:** C4.5 decision-tree and rule induction.

**Category & sub-category:** Supervised learning; entropy-based classification trees.

**Originating paper/vendor/year:** J. Ross Quinlan, *C4.5: Programs for Machine Learning*, 1993, published by Morgan Kaufmann. Quinlan's [author software page](https://www.rulequest.com/Personal/) supplies historical C4.5 Release 8 and distinguishes it from the later C5.0 product.

**Core mechanism:** Extend the ID3 lineage with continuous-attribute thresholds, gain-ratio-based split selection, missing-value handling, and pruning. Gain ratio normalizes information gain by the information in the split, reducing the attraction of attributes with many outcomes; selection also needs safeguards against ratios based on negligible gain. Nominal variables can produce multiway branches, unlike strictly binary CART. The system can convert a learned tree into a pruned ruleset; a ruleset's error and size need not match the original tree's.

**Inputs/outputs and typical data types:** Mixed numeric and nominal tabular features with class labels. Predictions can be represented as a tree path or a matching rule. Historical C4.5 distributes cases with unknown values across branches using weights rather than simply treating every missing value as a numeric zero.

**Strengths and limitations:** C4.5 provides inspectable symbolic models and handles mixed data naturally in its native representation. Greedy induction, noisy labels, large nominal domains, and unstable splits remain limitations. Gain ratio reduces a particular selection bias but does not make all variables equally interpretable. Neither C4.5's name nor its capabilities should be transferred to a generic library tree that actually implements CART.

**Computational complexity / scalability notes:** A useful implementation-dependent accounting is $`\sum_u O(d n_u\log n_u)`$ for numeric sorting at nodes $`u`$ containing $`n_u`$ cases. Presorting can reduce this work; arbitrary depth and weighted missing-case propagation complicate simple bounds. Prediction follows approximately $`h`$ tests when one branch is determined, but missing values may require multiple branches. Rule extraction has its own potentially substantial cost.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [RuleQuest's February 2017 comparison](https://www.rulequest.com/see5-comparison.html) uses the forest-cover dataset: environmental attributes -> tree conditions -> one of seven cover classes -> comparison with held-out cover labels. The vendor benchmark splits 581,012 cases into equal training and test halves. **C4.5 Release 8 trees** have **6.1% test error** and **10,169 leaves**; the separate C4.5 ruleset result is different and must not be substituted. Mixed-attribute rules are technically useful for explaining classifications relative to an opaque ensemble, but this is not evidence of a forestry agency's deployment decision. The comparison comes from the developer of the successor product and is not an independent evaluation. Production forestry KPIs are unreported.

**Notable vendor implementations/libraries:** Quinlan's historical C4.5 source and Weka's `J48` implementation. RuleQuest See5/C5.0 is a successor with additional algorithms, not a rename of C4.5.

### 1.2.3 Bagging

**Name:** Bootstrap aggregating, or bagging.

**Category & sub-category:** Supervised learning; resampling-based ensemble construction.

**Originating paper/vendor/year:** Leo Breiman's [1996 "Bagging Predictors"](https://doi.org/10.1007/BF00058655), preceded by a Berkeley technical report in 1994. Bagging is a general ensemble procedure; it is not intrinsically a tree architecture.

**Core mechanism:** Draw bootstrap training sets, fit a base learner independently on each, and combine their predictions. Regression commonly averages outputs; classification can use voting or averaged probabilities. Resampling perturbs unstable predictors so aggregation reduces their variance. Under a simplified equal-variance, equal-correlation model, averaging $`M`$ predictors gives variance $`\sigma^2[\rho+(1-\rho)/M]`$: shared errors remain even as the ensemble grows. Out-of-bag predictions use only models whose bootstrap samples omit the evaluated case.

**Inputs/outputs and typical data types:** Any labeled data supported by the base estimator. This entry's representative implementation uses regression trees on tabular measurements. The output is an aggregate prediction, plus possible out-of-bag diagnostics. A generic bagging wrapper does not create new neural layers or a unique network design.

**Strengths and limitations:** Independent fits parallelize naturally and often stabilize trees. Bagging can offer little benefit to an already stable learner and does not reliably remove bias. Naively bootstrapping individual rows can be inappropriate for grouped, spatial, or time-dependent observations. Out-of-bag evaluation is compromised if supervised preprocessing has already used the omitted observations, or if extensive model selection is judged on the same out-of-bag score.

**Computational complexity / scalability notes:** If the base fit costs $`F(n,d)`$, training $`M`$ models costs approximately $`O(MF(n,d)+Mn)`$, including resampling. Tree prediction costs $`O(Mh)`$ per example and storage $`O(Mv)`$ plus data. Parallel execution reduces elapsed time only within processor, memory-bandwidth, and data-replication limits.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 tree-ensemble lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch8-baggboost-lab.html) fits 500 bagged trees to its historical `Boston` housing data, considering all 12 supplied predictors at each split. The published 253/253 training/test experiment reports test MSE **23.41916**, in squared thousands-of-dollars units, for neighborhood median home value. Attributes -> bootstrap tree predictions -> their average -> comparison with observed median value demonstrates variance reduction relative to a single tree. This is a legacy teaching benchmark, not a recommendation for valuation or lending: its demographic proxies and historical sampling require particular caution. The result also depends on the documented R/package setup and must not be compared directly with another paper's different Boston split. No production property-pricing KPI is reported.

**Notable vendor implementations/libraries:** scikit-learn `BaggingClassifier`/`BaggingRegressor`; R `randomForest` with all predictors considered per split for tree bagging; ensemble facilities in MATLAB.

### 1.2.4 Random forest

**Name:** Random forest.

**Category & sub-category:** Supervised learning; randomized tree ensembles.

**Originating paper/vendor/year:** Leo Breiman's [2001 "Random Forests"](https://doi.org/10.1023/A:1010933404324) formalized this influential construction, building on bagging and earlier randomized-tree/subspace work, including Tin Kam Ho's contributions. The name should not erase those antecedents.

**Core mechanism:** Combine bootstrap sampling with random feature subsets at individual splits. Each tree searches a subset of predictors, rather than repeatedly allowing a few strong variables to dominate every tree. Aggregate votes, probabilities, or regression values. This decreases tree correlation at a possible cost to individual-tree strength. The classification ensemble in Breiman's account grows trees without pruning; modern libraries also expose depth and leaf-size controls.

**Inputs/outputs and typical data types:** Mixed tabular measurements after implementation-appropriate encoding, with class or continuous labels. Outputs include predictions, per-tree results, and often out-of-bag estimates. Impurity importance and permutation importance measure different quantities and should not be labeled interchangeably.

**Strengths and limitations:** Random forests handle nonlinearities, interactions, and heterogeneous feature scales with limited preprocessing. They generally extrapolate poorly for regression and can consume considerable memory. Impurity importance favors some high-cardinality features; correlated variables can divide or hide permutation importance. More trees stabilize Monte Carlo variability, but do not guarantee immunity to overfitting through depth, tuning, or leakage. Forest probabilities may need calibration.

**Computational complexity / scalability notes:** With $`q`$ candidate features, reusable orderings, and depth $`h`$, a rough split-scanning cost is $`O(Mqnh)`$, plus sorting, sampling, and bookkeeping. Re-sorting at nodes changes the bound. Storage is $`O(Mv)`$ beyond the data, and prediction $`O(Mh)`$ per example. Individual trees are parallelizable.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Breiman and Cutler's [author-hosted microarray case study](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#micro) classifies an early lymphoma dataset containing **81 cases, 4,682 expression variables, and three classes**. With **1,000 trees** and 150 candidate variables at a split, the documented run reports **1.23% out-of-bag error**, corresponding to one misclassified case. Expression profile -> votes from trees that did not train on that case -> predicted class -> out-of-bag comparison is the evaluation path. Random feature selection is technically useful for a many-genes/few-samples design relative to a single unstable tree; it does not establish a clinical selection rationale. This is neither an independent external test nor evidence of improved lymphoma care. Production diagnostic KPIs are unreported.

**Notable vendor implementations/libraries:** R `randomForest` and `ranger`, scikit-learn random-forest estimators, and Spark ML random forests. Defaults for candidate features, probability aggregation, and missing values differ.

### 1.2.5 Extremely randomized trees

**Name:** Extremely randomized trees, or Extra-Trees.

**Category & sub-category:** Supervised learning; randomized split-point ensembles.

**Originating paper/vendor/year:** Pierre Geurts, Damien Ernst, and Louis Wehenkel, [2006, "Extremely Randomized Trees"](https://orbi.uliege.be/handle/2268/9357). The originating institution's repository supplies the [full author paper](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf).

**Core mechanism:** At a node, select candidate features, draw random cut points, and choose the best of those candidate splits using the supervised criterion. The original default uses the full learning sample rather than bootstrap samples. This randomizes thresholds as well as feature choice, increasing diversity and reducing threshold-search work. It does **not** make the standard algorithm label-independent: labels still choose among candidates and determine leaf predictions. The paper also studies more extreme randomized variants.

**Inputs/outputs and typical data types:** Tabular numeric or encoded features with classification or regression labels. Predictions average or vote across trees. A library may optionally enable bootstrapping, but this is a configuration change, not a defining requirement of Extra-Trees.

**Strengths and limitations:** Extra-Trees can be fast and competitive where many approximately useful partitions exist. Strong randomization may reduce variance but increase bias, particularly when a narrow, precise threshold matters. Like other axis-aligned forests, it is not intrinsically suitable for extrapolation or arbitrary categorical identifiers. Its apparent robustness does not replace appropriate splitting, calibration checks, or sensitivity analysis.

**Computational complexity / scalability notes:** If each of $`q`$ features supplies one candidate cut and each node scans its cases, training costs approximately $`O(Mqnh)`$. It can avoid the global sorting required by exact split searches. Prediction is $`O(Mh)`$, with $`O(Mv)`$ tree storage. Actual speed depends on node sizes, memory access, stopping rules, and the chosen number of candidate thresholds.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Geurts and colleagues' [Letter-recognition experiment, Tables 7-8](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf), uses 16 numeric image-summary features to classify 26 letters. It repeatedly divides the data into 10,000 learning and 10,000 test cases; this dataset receives ten random-split runs. The default Extra-Trees column reports mean test error **3.80%**, while the paper's best-setting random-forest comparator, labeled `RF*`, reports **4.87%**. These are the authors' comparison settings, not a claim about default contemporary libraries. Features -> randomized thresholds -> letter votes -> recognition-error evaluation is the worked path. Avoiding exhaustive cut-point search is the technical advantage relative to ordinary forests. No postal-processing deployment, labor savings, or production transcription KPI is established.

**Notable vendor implementations/libraries:** scikit-learn `ExtraTreesClassifier` and `ExtraTreesRegressor`; other forest toolkits expose extra-randomized splitting options whose details should be checked against the original formulation.

### 1.2.6 AdaBoost

**Name:** Adaptive boosting, or AdaBoost.

**Category & sub-category:** Supervised learning; sequential reweighting ensembles.

**Originating paper/vendor/year:** Yoav Freund and Robert Schapire, "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting," [1997 journal publication](https://doi.org/10.1006/jcss.1997.1504), following the 1995 conference contribution. Later multiclass variants are not identical to the original binary procedure.

**Core mechanism:** Fit a weak classifier to weighted observations, increase the influence of misclassified cases, and combine classifiers with learned weights. In the binary $`\{-1,+1\}`$ formulation, an error $`0<\epsilon_m<1/2`$ gives weight $`\frac12\log[(1-\epsilon_m)/\epsilon_m]`$; a zero-error learner requires a separate stopping or limiting treatment. A final weighted sum supplies the decision. This corresponds to stagewise reduction of exponential loss. Multiclass SAMME uses a different coefficient formula and condition on the weak learner; averaging binary formulas across classes is not equivalent.

**Inputs/outputs and typical data types:** Labeled feature vectors, frequently tabular data or hand-engineered image features, and a base learner supporting observation weights. Outputs are ensemble scores and class labels. A margin is not automatically a calibrated probability.

**Strengths and limitations:** AdaBoost can transform simple weak learners into a rich classifier while focusing computation on difficult cases. Its reweighting also makes persistent label errors and outliers influential. Training rounds are sequential; increasing rounds or base-tree depth can hurt performance. Feature selection emerges naturally when stumps choose one feature, but a boosted collection may be harder to explain than a single tree.

**Computational complexity / scalability notes:** For $`M`$ rounds and base-fitting cost $`F(n,d)`$, training is $`O(M[F(n,d)+n])`$. Reusable sorted features can make decision-stump fitting roughly $`O(nd)`$ per round after preprocessing. Prediction evaluates the $`M`$ base learners; for trees of depth $`h`$ this is $`O(Mh)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Viola and Jones's [2001 boosted-cascade face detector, documented by MERL](https://www.merl.com/publications/TR2004-043), learns from face and non-face image windows. Integral-image rectangle features -> AdaBoost-selected weak tests -> a cascade decision -> candidate face boxes is the demonstrated pipeline. The authors report detection rates comparable to leading earlier systems and real-time operation. This is face **detection**, not personal identification; the report does not establish a particular commercial deployment or its KPI. The documented method uses boosting to select informative inexpensive features, while the cascade rejects easy background windows before costly later stages. A single fixed template lacks that learned combination, although this technical comparison is not evidence of a purchaser's intent. Throughput depends on the complete cascade and implementation, not AdaBoost alone.

**Notable vendor implementations/libraries:** scikit-learn `AdaBoostClassifier`, Weka AdaBoost variants, and boosted-cascade facilities in OpenCV. OpenCV's different boost types should not all be called original AdaBoost.

### 1.2.7 Gradient-boosted decision trees

**Name:** Gradient-boosted decision trees, or GBDT; gradient boosting machines with tree base learners.

**Category & sub-category:** Supervised learning; stagewise additive optimization.

**Originating paper/vendor/year:** Jerome Friedman's [2001 "Greedy Function Approximation: A Gradient Boosting Machine"](https://doi.org/10.1214/aos/1013203451) is the central gradient-boosting reference. Boosting has earlier antecedents; particular vendor implementations are not the origin of the general method.

**Core mechanism:** Start from a constant predictor and iteratively fit a tree to the negative derivative of the loss with respect to current predictions. Add the tree with a shrinkage factor, optionally optimizing leaf values for the chosen loss. For squared loss, these targets are residuals; for logistic loss, they are not simply the original class labels. Subsampling, shallow trees, leaf constraints, and early stopping regularize the additive model. Ranking variants change the optimization signal and should be identified explicitly.

**Inputs/outputs and typical data types:** Tabular features with numeric labels, classes, or relevance judgments. Outputs can be real predictions, log-odds scores, probabilities after a link, or ranking scores. Individual tree leaves can also become categorical features for another model.

**Strengths and limitations:** GBDT learns interactions and nonlinear effects while accommodating task-specific losses. Compared with bagging, it can reduce systematic underfit rather than only average independent fits. Its sequential nature, sensitivity to depth/learning-rate/round choices, and vulnerability to leakage demand careful validation. Explanations of a combined score are not causal explanations.

**Computational complexity / scalability notes:** For $`M`$ trees, gradient calculation is typically $`O(Mn)`$ for a simple scalar loss, plus $`M`$ tree fits. Exact sorting-based and histogram-based learners differ substantially. Prediction costs $`O(Mh)`$; ranking losses may additionally depend on query sizes or document pairs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** He and colleagues' [2014 Facebook advertising report](https://quinonero.net/Publications/predicting-clicks-facebook.pdf) uses logged impressions and click/no-click outcomes from one week in Q4 2013, with a shared offline training/testing partition across comparisons. Features -> boosted-tree leaf identifiers -> a sparse logistic classifier -> estimated click probability forms the reported hybrid, rather than a standalone tree ensemble. The probability supports ad evaluation; the numerical comparisons are offline predictive evaluations. Table 1 rescales normalized cross-entropy to a tree-only reference of **100%**: logistic regression alone is **99.43%**, and trees plus logistic regression **96.58%**, with lower values better. The table, not a loosely paraphrased percentage in surrounding prose, anchors this comparison. Learned interaction features explain the technical fit relative to plain logistic regression. The authors explicitly evaluate prediction rather than profit or revenue; a production revenue KPI is unreported.

**Notable vendor implementations/libraries:** R `gbm`, scikit-learn gradient-boosting estimators, Spark ML GBT, and the separately described XGBoost, LightGBM, and CatBoost systems.

### 1.2.8 XGBoost

**Name:** XGBoost, here specifically its boosted-tree formulation.

**Category & sub-category:** Supervised learning; regularized gradient boosting with scalable split search and systems engineering.

**Originating paper/vendor/year:** Tianqi Chen and Carlos Guestrin, [2016, "XGBoost: A Scalable Tree Boosting System," KDD](https://arxiv.org/html/1603.02754v3). The inspected arXiv version is v3, dated 2016-06-10. XGBoost is an open-source project; later cloud offerings are implementation or hosting providers.

**Core mechanism:** Expand the objective using first- and second-order derivatives, then score candidate partitions using their aggregated gradients and Hessians. With a quadratic leaf penalty, a representative leaf weight is $`-\sum_i g_i/(\sum_i h_i+\lambda)`$. Penalizing leaf count and coefficient size controls complexity alongside shrinkage and subsampling. The paper adds sparsity-aware default directions, a weighted quantile sketch for approximate split proposals, and cache-aware/out-of-core organization. These contributions go beyond merely putting trees on several processors.

**Inputs/outputs and typical data types:** Dense or sparse feature tables with regression, classification, or ranking targets. Tree scores become predictions through the selected objective's link. Other boosters exist in the package, but a linear booster is not the tree model described in this entry.

**Strengths and limitations:** Flexible objectives, missing/sparse-value handling, and scalable implementations make XGBoost broadly useful. Sparse absence and a numeric zero are not automatically interchangeable under every interface. Powerful interaction fitting can still leak targets, overfit validation, or produce poor extrapolation. Probability calibration and drift monitoring are separate responsibilities. Native categorical support in later releases should not be retroactively assigned to the 2016 paper.

**Computational complexity / scalability notes:** Let $`z=\operatorname{nnz}(X)`$. A simplified sparsity-aware scan over $`M`$ trees of depth $`h`$ costs $`O(Mhz)`$, in addition to sorting or sketching and tree bookkeeping. A histogram implementation also depends on bin count and processed nodes. Out-of-core I/O and distributed communication can dominate wall time. Prediction is $`O(Mh)`$ for a fully traversed tree ensemble.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper's Higgs-1M classification comparison](https://arxiv.org/html/1603.02754v3) uses the HIGGS dataset of Monte Carlo particle events, not a live detector deployment. Its features include kinematic measurements and derived physical quantities; signal/background labels supervise the trees. Event features -> boosted score -> signal-like classification -> held-out discrimination evaluation is the scientific workflow. Section 6.3 reports classification quality comparable to scikit-learn's exact tree boosting with substantially shorter training time in the authors' CPU setup. This is a useful fit for nonlinear feature interactions relative to a linear decision boundary, but it is not evidence that XGBoost discovered the Higgs boson. The comparison uses a one-million-example subset for the exact-greedy experiment; its conclusions are not a universal library-speed ranking. Production scientific KPIs are unreported.

**Notable vendor implementations/libraries:** The [XGBoost project](https://xgboost.readthedocs.io/en/stable/), language bindings, distributed integrations, and managed-service wrappers. Availability on a cloud platform does not prove use in that provider's products.

### 1.2.9 LightGBM

**Name:** LightGBM.

**Category & sub-category:** Supervised learning; histogram-based gradient-boosted trees with data- and feature-reduction techniques.

**Originating paper/vendor/year:** Guolin Ke and collaborators, [2017, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree), a Microsoft-led research contribution. The [original paper](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) defines the techniques used in its experiments.

**Core mechanism:** Quantize feature values into bins and aggregate gradient statistics in histograms. Leaf-wise growth expands a promising leaf rather than necessarily expanding a whole depth level. Gradient-based one-side sampling, GOSS, retains large-gradient cases and reweights a sample of small-gradient cases; it does not simply discard every easy example. Exclusive feature bundling, EFB, combines features that are rarely nonzero together. These are distinct optimizations. Contemporary configuration choices need not enable the exact GOSS-plus-EFB setup from the paper.

**Inputs/outputs and typical data types:** Large dense or sparse tabular datasets with numerical, encoded, or appropriately supplied categorical features. Objectives support regression, classification, and learning to rank. Outputs are additive tree scores or linked probabilities.

**Strengths and limitations:** Histogram construction and compact feature representations can make large datasets tractable. Leaf-wise growth can fit strong interactions efficiently, but very small leaves can overfit. GOSS trades computational work against gradient-estimation variance; bundling relies on feature co-occurrence properties. Category codes must represent categories rather than fabricated numerical magnitudes. Model quality still depends on task loss, temporal validity, and tuning.

**Computational complexity / scalability notes:** With $`b`$ bins and $`q`$ effective features, a dense node histogram costs roughly $`O(n_u q)`$ to construct and $`O(qb)`$ to scan at node $`u`$. Histogram subtraction can reuse sibling information. Overall cost sums over grown nodes and boosting rounds; it is not universally $`O(n\log n)`$. Storage depends on binned data, histograms, and leaves; distributed runs add histogram communication.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper's Microsoft LETOR workload](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) uses query-document features and relevance labels: features -> ranking trees -> relevance scores -> ordered search results evaluated by NDCG. Table 2 reports **0.31 seconds per training iteration** for LightGBM versus **0.49 seconds** for its histogram baseline without GOSS/EFB, on the authors' two-E5-2670-v3 server with 256 GB memory and **16 training threads**. This is per-round timing on that benchmark, not total retraining latency or a general speedup guarantee; the paper reports similar held-out ranking quality. Histograms and reduced gradient work are the documented technical rationale relative to more expensive split scans. The experiment does not establish use in live Bing ranking, improved search revenue, or a production KPI.

**Notable vendor implementations/libraries:** The [Microsoft LightGBM project](https://lightgbm.readthedocs.io/), its Python/R interfaces, command-line learner, and distributed integrations. CPU/GPU capabilities depend on the build and selected learner.

### 1.2.10 CatBoost

**Name:** CatBoost.

**Category & sub-category:** Supervised learning; gradient boosting with ordered statistics and ordered boosting.

**Originating paper/vendor/year:** Developed at Yandex; Liudmila Prokhorenkova and colleagues' [2018 "CatBoost: Unbiased Boosting with Categorical Features"](https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) is the principal algorithm paper. The project's release history and subsequent implementations are distinct from that publication.

**Core mechanism:** Encode categorical values with statistics designed to avoid using an observation's own target. Under a permutation, an observation's training statistic uses suitable earlier observations and a prior. Ordered boosting additionally constructs residual-related quantities from models that have not already fitted the current observation's label, addressing prediction shift. These are separate ideas: leak-resistant category encoding alone is not ordered boosting. Symmetric, or oblivious, trees use the same split at every node of a depth level, enabling compact regular structure. Plain boosting and other tree-growth options exist in later configurations.

**Inputs/outputs and typical data types:** Tables combining numeric and categorical features, including high-cardinality categories, with supervised labels. Outputs are scores, probabilities, regressions, or rankings according to the objective. Category combinations can model interactions, but their construction must avoid target leakage.

**Strengths and limitations:** CatBoost reduces manual categorical preprocessing and addresses a subtle source of training/inference mismatch. Symmetric trees can be efficient and regularized, but may require additional trees to represent irregular boundaries. Rare categories, changing vocabularies, concept drift, and leakage from future observations remain risks. Ordered statistics are not permission to mix future and past records in a forecasting evaluation.

**Computational complexity / scalability notes:** A symmetric tree of depth $`h`$ has $`2^h`$ leaves and $`h`$ split tests along a prediction path. Scalar-output ensemble leaf storage is therefore $`O(M2^h)`$, excluding category-statistic tables; multiple outputs add their own factor. Training additionally depends on candidate features, permutations, category combinations, and histogram bins; there is no single dimension-independent runtime. Increasing depth can sharply increase memory even when per-example traversal remains $`O(Mh)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [2018 paper's Amazon employee-access task](https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) predicts recorded access outcomes from categorical employee/resource attributes. Its four-fifths training/tuning and one-fifth test protocol reports **0.139 log loss** and **0.044 zero-one loss** for CatBoost's Ordered mode in Table 2. The authors also apply ordered target-statistic preprocessing to the compared learners, an important detail when interpreting differences. Attributes -> leak-resistant category statistics -> tree score -> predicted access label is an offline classifier, not an authorization policy. The technical fit is handling interacting categories without an enormous naive one-hot design. The paper does not establish Amazon production use, reduced access-review costs, or improved security; production KPIs are unreported.

**Notable vendor implementations/libraries:** The [CatBoost project](https://catboost.ai/docs/), Python/R interfaces, command-line tools, and model-serving/export facilities. Yandex's authorship is not evidence about an unrelated organization's deployment.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| CART | Mixed tabular measurements | Inspectable threshold interactions | Instability and overfitting | Wisconsin breast-cancer pruning demonstration |
| C4.5 | Numeric and nominal classification tables | Mixed-data trees and rules | Greedy structure; costly ruleset construction | RuleQuest forest-cover benchmark |
| Bagging | Data suited to an unstable base learner | Variance reduction through independent fits | Shared bias and correlated errors remain | Historical Boston median-value benchmark |
| Random forest | Nonlinear tabular or many-feature data | Decorrelated trees and out-of-bag diagnostics | Memory, extrapolation, and importance biases | Lymphoma microarray research case |
| Extremely randomized trees | Numeric or encoded tables | Cheap randomized split thresholds | Extra randomization can increase bias | Letter-recognition benchmark |
| AdaBoost | Weighted tabular/image-feature examples | Combines weak classifiers adaptively | Sensitivity to persistent label errors | Viola-Jones research face detector |
| Gradient-boosted decision trees | Interaction-rich labeled tables | Task-specific sequential loss reduction | Sequential fitting and tuning sensitivity | Facebook tree-feature plus logistic CTR model |
| XGBoost | Dense/sparse regression, classification, ranking | Regularized boosting and scalable systems | Strong models still require leakage control | Monte Carlo HIGGS event classification |
| LightGBM | Large binned dense/sparse tables | Histogram efficiency, GOSS, and EFB | Small-leaf overfit and configuration dependence | Microsoft LETOR offline ranking workload |
| CatBoost | Tables with categorical interactions | Ordered statistics and boosting | Category tables and exponential depth storage | Amazon access-data research benchmark |

## 1.3 Neighbors and kernels

These methods make similarity a central modeling choice. A neighbor classifier directly reuses nearby labeled observations. A kernel method expresses a prediction through pairwise comparisons in an implicit feature space. Similarity must match the problem: Euclidean distance on arbitrary category codes is usually meaningless, and a valid positive-semidefinite kernel is more than an unrestricted similarity score.

### 1.3.1 k-nearest neighbors

**Name:** k-nearest neighbors, or kNN, for classification and regression.

**Category & sub-category:** Supervised learning; instance-based local prediction.

**Originating paper/vendor/year:** Nearest-neighbor rules predate modern machine learning. Cover and Hart's [1967 "Nearest Neighbor Pattern Classification"](https://doi.org/10.1109/TIT.1967.1053964) is a foundational statistical analysis, not a claim that they invented every neighborhood estimator.

**Core mechanism:** Store labeled observations and, for a query, find its $`k`$ closest training examples under a chosen distance. Classification votes or averages class indicators; regression averages numerical responses, possibly weighting by distance. Increasing $`k`$ smooths the prediction and usually trades lower variance for greater bias. The metric, scaling, feature selection, and treatment of ties are part of the method rather than incidental preprocessing.

**Inputs/outputs and typical data types:** Numeric feature vectors with labels; other objects are possible when an appropriate distance is available. Outputs include predicted classes or values and the retrieved neighbors. A neighbor-vote fraction is an empirical local class estimate, not a universally calibrated posterior. Sparse text commonly needs a similarity and normalization suited to text rather than raw Euclidean word counts.

**Strengths and limitations:** kNN is simple, local, and capable of irregular decision boundaries without a fitted global equation. It can incorporate new reference observations cheaply. Prediction may be expensive, memory grows with the reference set, irrelevant coordinates distort distances, and high-dimensional distance concentration weakens locality. Duplicate or nearly duplicate records across a split can make evaluation misleading. A highly similar historical case is not necessarily a safe analogy after distribution shift.

**Computational complexity / scalability notes:** Brute-force fitting mainly stores $`O(nd)`$ data. One exact query requires $`O(nd)`$ distance work, plus neighbor selection: expected $`O(n)`$ with an appropriate selection algorithm, or $`O(n\log k)`$ with a heap. Spatial indexes can help in low effective dimension but do not guarantee logarithmic queries in high dimensions. Approximate retrieval changes the prediction procedure.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 `Smarket` lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html) predicts 2005 market direction from the first two lagged returns, using only pre-2005 cases as neighbors. With $`k=1`$, it correctly predicts **126/252** days; with $`k=3`$, **135/252**, or **53.57%**. The latter remains below the always-Up baseline on that test period. Lagged returns -> nearest historical return patterns -> their class vote -> a test-day direction is the worked path. Local voting is a technical alternative to a global linear boundary, but here added flexibility does not establish better prediction. The negative result is informative: neither the method's simplicity nor an apparent local analogy demonstrates trading profitability. Production financial KPIs are unreported.

**Notable vendor implementations/libraries:** scikit-learn neighbor estimators, R `class::knn`, and MATLAB nearest-neighbor learners. FAISS and other vector indexes provide retrieval infrastructure, not by themselves a complete supervised kNN estimator.

### 1.3.2 Support-vector classification

**Name:** Support-vector classification, or SVC; commonly called a support-vector machine classifier.

**Category & sub-category:** Supervised learning; maximum-margin linear and kernel classification.

**Originating paper/vendor/year:** Corinna Cortes and Vladimir Vapnik, [1995, "Support-Vector Networks"](https://doi.org/10.1007/BF00994018), is the canonical soft-margin reference, with earlier maximum-margin and kernel antecedents.

**Core mechanism:** Balance a large separating margin against violations. A binary formulation minimizes $`\frac12\|w\|^2+C\sum_i\max(0,1-y_i f(x_i))`$, where $`f(x)=w^\top\phi(x)+b`$. A positive-semidefinite kernel evaluates inner products of implicit features $`\phi(x)`$. The dual prediction uses training examples with nonzero coefficients, the support vectors. Multiclass systems may combine one-versus-one or one-versus-rest classifiers; these decompositions are not interchangeable with one jointly optimized multiclass objective.

**Inputs/outputs and typical data types:** Labeled numerical vectors, or structured objects for which a suitable kernel is defined. Outputs are decision scores and class labels. Probabilities require an additional calibration procedure; the original hinge-loss score is not a probability.

**Strengths and limitations:** Margin regularization can work well with high-dimensional inputs and relatively modest sample counts. Kernels introduce nonlinear decision boundaries without explicitly enumerating their features. Scaling, kernel bandwidth, and $`C`$ are crucial. Large numbers of examples or support vectors make training and serving costly. Kernel choice encodes strong assumptions, and a mathematically valid kernel can still describe the wrong notion of similarity.

**Computational complexity / scalability notes:** If one kernel evaluation costs $`c_K`$, materializing a Gram matrix costs $`O(n^2c_K)`$ time and $`O(n^2)`$ memory. Optimization depends on the solver, active set, cache, and tolerance; cubic dense linear algebra may occur, but is not a universal total-training bound. A binary prediction costs $`O(s c_K)`$ for $`s`$ support vectors. A specialized linear solver can instead use roughly $`O(nd)`$ work per pass.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn handwritten-digit example](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) flattens 8-by-8 grayscale digit images into 64 features and fits an RBF SVC with `gamma=0.001`. Its non-shuffled half-training/half-test split reports rounded accuracy **0.97 on 899 test images**. Pixels -> kernel decision scores -> a digit label -> comparison with the recorded transcription is the workflow; the displayed first test prediction is 8. A nonlinear similarity boundary is technically appropriate for varied handwriting compared with a single linear score, but this demonstration does not establish superiority to a tuned alternative. It is not a deployed postal or banking OCR system, and no production processing KPI is reported.

**Notable vendor implementations/libraries:** LIBSVM, scikit-learn `SVC`, R `e1071` interfaces, and LIBLINEAR/scikit-learn `LinearSVC` for related linear formulations with different solver and multiclass conventions.

### 1.3.3 Support-vector regression

**Name:** Support-vector regression, or SVR.

**Category & sub-category:** Supervised learning; regularized kernel regression with an insensitive loss region.

**Originating paper/vendor/year:** Drucker, Burges, Kaufman, Smola, and Vapnik, ["Support Vector Regression Machines"](https://papers.nips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf), presented at NIPS 1996 and published in its volume 9 proceedings. This extends the support-vector framework from classification to numerical prediction.

**Core mechanism:** Minimize a function-complexity penalty plus an $`\epsilon`$-insensitive loss, $`\max(0,|y-f(x)|-\epsilon)`$. Errors inside the tube cost nothing; deviations outside it incur a penalty controlled by $`C`$. A kernel version uses positive and negative dual coefficients, with many observations contributing zero weight. The tube's width has response units and is not a statistically calibrated prediction interval. Squared insensitive loss and $`\nu`$-SVR are related but distinct variants.

**Inputs/outputs and typical data types:** Numerical or kernel-represented observations with continuous targets. Outputs are real predictions and, in suitable interfaces, support-vector diagnostics. Scaling both inputs and the response may change effective regularization, so transformations must be recorded and inverted correctly.

**Strengths and limitations:** SVR permits nonlinear regression without explicitly constructing a large feature expansion and can be less driven by very large residuals than squared loss. It does not automatically handle uncertainty, heteroskedasticity, or censored observations. Hyperparameter selection can be expensive, and dense kernel computation limits sample scaling. An uninformative bandwidth can make the fit either nearly constant or excessively local.

**Computational complexity / scalability notes:** As for SVC, a fully stored Gram matrix needs $`O(n^2)`$ memory and $`O(n^2c_K)`$ kernel work, with solver-dependent optimization beyond that. Prediction costs $`O(s c_K)`$ for $`s`$ support vectors. Sparsity of the solution is data- and tolerance-dependent; SVR is not guaranteed to retain only a small fraction of cases.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Drucker and colleagues' original paper](https://papers.nips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf) compares SVR and bagging on historical Boston housing observations. It uses **401 training, 80 validation, and 25 test cases**, repeating random partitions **100 times**. The paper reports mean squared prediction error **7.2 for SVR versus 12.4 for bagging**, in squared dataset-response units, with SVR better on 71 of the 100 trials. Numerical neighborhood attributes -> kernel regression score -> estimated median home value -> test-error comparison is the demonstrated chain. The implicit polynomial representation was useful relative to explicitly fitting a high-dimensional polynomial expansion with few training examples. This is not the ISLR2 split or feature setup, so its errors must not be ranked against that chapter's bagging numbers. The dataset is a legacy benchmark with demographic and sampling concerns, not a recommended valuation policy. Production economic KPIs are unreported.

**Notable vendor implementations/libraries:** LIBSVM, scikit-learn `SVR` and `NuSVR`, R `e1071`, and specialized linear-SVR solvers.

### 1.3.4 Kernel ridge regression

**Name:** Kernel ridge regression, or KRR.

**Category & sub-category:** Supervised learning; squared-loss regression in a reproducing-kernel Hilbert space.

**Originating paper/vendor/year:** Quadratic regularization in function spaces has a longer history; Saunders, Gammerman, and Vovk's [1998 "Ridge Regression Learning Algorithm in Dual Variables"](https://eprints.soton.ac.uk/258942/) is an influential machine-learning treatment. A library's kernel interface is not the origin of the theory.

**Core mechanism:** Minimize squared errors plus $`\lambda\|f\|_{\mathcal H}^2`$, where the kernel defines the function-space norm. The representer theorem gives $`f(x)=\sum_i\alpha_i K(x_i,x)`$. Under the unnormalized squared-loss convention, $`(K+\lambda I)\alpha=y`$. Centering or a separate intercept must be handled explicitly. The penalty on the represented function is not generally $`\lambda\|\alpha\|^2`$; confusing those objectives changes the estimator.

**Inputs/outputs and typical data types:** Labeled vectors, molecules, strings, or other objects with a suitable positive-semidefinite kernel. Output is a continuous prediction. Unlike SVR, the solution is generally dense in training examples. For fixed matched kernel and noise settings, its prediction can coincide with a Gaussian-process posterior mean, but KRR alone does not supply that model's posterior variance.

**Strengths and limitations:** KRR offers a direct regularized solve and clean control of smoothness through the kernel. It can model nonlinear relationships with modest datasets. Its cost grows sharply with sample count, descriptor quality controls what similarity means, and extrapolation is kernel-dependent. Low-rank or random-feature approximations reduce work but alter the exact estimator.

**Computational complexity / scalability notes:** Exact dense training costs $`O(n^2c_K+n^3)`$ time and $`O(n^2)`$ Gram-matrix memory. Prediction is $`O(nc_K)`$ per query. An approximation with $`r\ll n`$ basis functions can reduce the solve to approximately $`O(nr^2+r^3)`$, in addition to constructing those features; $`r`$ is approximation rank, not the number of original input features.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Rupp and colleagues' [molecular-energy preprint v1, dated 2011-09-12](https://arxiv.org/html/1109.2618v1), precedes the [2012-01-31 journal publication](https://doi.org/10.1103/PhysRevLett.108.058301). It uses 7,165 small organic molecules from GDB, with reference atomization energies computed using PBE0 density-functional theory. Nuclear charges and atomic positions -> Coulomb-matrix-based descriptors -> Gaussian-kernel regression -> predicted atomization energy is the demonstrated surrogate. The inspected preprint reports **9.9 kcal/mol mean absolute error**, using nested five-fold selection/evaluation rather than a production screening campaign. It compares predictions with quantum-chemical reference calculations; those references are approximations to physics, not exact experimental ground truth. The method fits a repeated-computation problem because a learned energy surrogate can be evaluated without resolving the electronic-structure problem for every query. That technical motivation does not establish validated reaction energetics, drug-discovery success, or a production cost-saving KPI.

**Notable vendor implementations/libraries:** scikit-learn `KernelRidge`, kernel-toolbox implementations, and scientific packages that solve regularized kernel systems. A general linear-algebra solver needs additional kernel and preprocessing logic.

### 1.3.5 Gaussian processes: regression and classification

**Name:** Gaussian processes, with Gaussian-process regression (GPR) and Gaussian-process classification (GPC) explicitly distinguished.

**Category & sub-category:** Supervised learning; Bayesian nonparametric prediction over functions.

**Originating paper/vendor/year:** Gaussian stochastic processes and kriging have multiple historical origins rather than one vendor or uniquely defining paper. Rasmussen and Williams's [2006 *Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/chapters/) is the canonical modern reference used here, not a claim that GPs were invented in 2006.

**Core mechanism:** A mean function and covariance kernel define joint Gaussian distributions for finite collections of latent function values. **Regression with Gaussian observation noise** permits analytic conditioning: the posterior mean is a kernel-weighted prediction and the covariance quantifies model-based uncertainty. **Classification** passes latent functions through a Bernoulli or categorical likelihood; the posterior is generally non-Gaussian and needs Laplace approximation, expectation propagation, variational inference, or sampling. Applying the regression equations directly to labels does not give the usual GPC posterior.

**Inputs/outputs and typical data types:** Numeric or structured covariates and measured responses or class labels. GPR returns predictive means and covariances; GPC returns approximate class probabilities. Noise-free latent-function uncertainty and noisy future-observation uncertainty differ. Hyperparameters may be fitted by marginal likelihood or assigned their own priors.

**Strengths and limitations:** Kernels express prior assumptions about smoothness, periodicity, or spatial dependence, and uncertainty is an explicit part of the model. It remains conditional on those assumptions: misspecified kernels can produce misleading confidence. Exact methods do not scale like ordinary linear regression. Stationary kernels and homogeneous noise assumptions are not always adequate; nonstationary and sparse approximations require additional modeling choices.

**Computational complexity / scalability notes:** Exact dense GPR at fixed hyperparameters requires **$`O(n^3)`$ time and $`O(n^2)`$ memory**. Hyperparameter optimization repeats expensive calculations. After factorization, one predictive mean costs $`O(nc_K)`$; an exact variance additionally requires roughly $`O(n^2)`$ work. Dense GPC repeats comparable matrix work inside approximate-inference iterations. Inducing-variable approximations have different bounds and are not exact GPs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [Mauna Loa CO2 example](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html), based on the GP book, models monthly atmospheric concentrations from **1958-2001**. Date -> a kernel combining smooth trend, locally periodic variation, irregularity, and noise -> concentration in ppm plus posterior uncertainty -> inspection of fitted and extrapolated trajectories is the workflow. The published output reproduces seasonal variation and provides post-2001 model extrapolations; it does **not** report an independent future-period accuracy score. A periodic covariance is a technical advantage over a bare linear trend when uncertainty and recurring seasons matter. The graph's uncertainty is not a validated climate forecast or evidence of an observatory's operational system. Production forecasting KPIs are unreported.

**Notable vendor implementations/libraries:** GPML, scikit-learn Gaussian-process estimators, GPyTorch, and GPflow. This entry describes classical kernels; using a learned neural feature extractor would require a separately specified neural instantiation.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| k-nearest neighbors | Low/moderate-dimensional meaningful distances | Local prediction without a global equation | Serving cost and high-dimensional distances | `Smarket` neighbor-voting benchmark |
| Support-vector classification | Moderate sample counts, rich numeric features | Margin control with nonlinear kernels | Kernel scaling and calibration requirements | 8-by-8 handwritten-digit classification |
| Support-vector regression | Nonlinear continuous targets | Insensitive loss and implicit feature expansion | Expensive tuning; no native uncertainty interval | Original Boston housing comparison |
| Kernel ridge regression | Moderate-sized datasets with useful kernels | Direct regularized nonlinear solve | Dense $`n`$-dependent training and prediction | GDB molecular atomization energies |
| Gaussian processes | Small/moderate scientific or spatial datasets | Explicit covariance and predictive uncertainty | Cubic exact fitting; classification is approximate | Mauna Loa CO2 fit and extrapolation |

## 1.4 Probabilistic, structured, and survival models

Labels need not be independent scalar categories. A structured output can be a sequence of grammatical tags, and a survival label can combine a duration with incomplete event observation. These models make different factorization assumptions; "probabilistic" does not mean that all use the same fitting algorithm or that their probabilities are automatically calibrated.

### 1.4.1 Naive Bayes: Gaussian, multinomial, and Bernoulli

**Name:** Naive Bayes classifiers, including Gaussian, multinomial, and Bernoulli variants.

**Category & sub-category:** Supervised learning; simple generative probabilistic classification.

**Originating paper/vendor/year:** There is no single inventor of all naive-Bayes variants. M. E. Maron's [1961 "Automatic Indexing: An Experimental Inquiry"](https://doi.org/10.1145/321075.321084) is a landmark in probabilistic text classification. Bayes's theorem predates these computer-learning applications; a modern library is not the source of that theorem.

**Core mechanism:** Combine class priors with a simplified class-conditional likelihood and select the largest posterior. **Gaussian NB** models each numerical feature with a separate class-specific normal distribution. **Multinomial NB** models conditionally repeated token events, yielding a count likelihood proportional to $`\prod_j\theta_{jc}^{x_j}`$; its fixed-total counts are not themselves independent random variables. **Bernoulli NB** uses binary presence indicators and explicitly includes both presence and absence probabilities. Repetition therefore matters for multinomial NB but not a binarized Bernoulli model. Smoothing avoids zero-probability failures; log scores avoid numerical underflow.

**Inputs/outputs and typical data types:** Gaussian NB expects continuous measurements; multinomial NB expects nonnegative counts or a deliberately chosen count-like representation; Bernoulli NB expects binary indicators. Outputs are class labels and approximate posterior probabilities. TF-IDF may work empirically with a multinomial implementation without literally following a multinomial data-generating process.

**Strengths and limitations:** Training is fast, sufficient statistics can be updated incrementally, and the models provide valuable small-data or sparse-text baselines. Conditional-independence assumptions can be badly violated, especially by correlated measurements. Classification can remain useful while probabilities become overconfident. Different event models are substantive modeling choices, not interchangeable parameter settings.

**Computational complexity / scalability notes:** Dense supervised sufficient-statistic accumulation is approximately $`O(nd)`$, with $`O(Cd)`$ parameter storage for $`C`$ classes. Dense prediction is $`O(Cd)`$ per case. Sparse count implementations replace much of the input scan with nonzero-entry work; vocabulary initialization and smoothing still require storage proportional to class-by-vocabulary size.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn Gaussian NB example](https://scikit-learn.org/stable/modules/naive_bayes.html) uses the Iris flower measurements and species labels. It holds out half the data with `random_state=0` and reports **four mislabeled flowers among 75 test examples**. Sepal/petal measurements -> class-specific one-dimensional Gaussian likelihoods -> posterior species choice -> comparison with recorded species is the worked path. Fewer covariance parameters make Gaussian NB a technically sensible small-sample baseline relative to full QDA, although correlated flower dimensions challenge its assumptions. This particular result validates neither the multinomial nor Bernoulli variants; those require their own count/binary tasks. The demonstration is not a botanical production deployment, and no operational identification KPI is reported.

**Notable vendor implementations/libraries:** scikit-learn `GaussianNB`, `MultinomialNB`, and `BernoulliNB`; R `e1071::naiveBayes`; and Spark ML naive-Bayes variants with implementation-specific support.

### 1.4.2 Bayesian networks

**Name:** Bayesian networks; here including supervised Bayesian-network classifiers.

**Category & sub-category:** Supervised learning; directed graphical probability models used to predict a designated target. The broader family also supports expert-elicited models and unsupervised density/structure learning.

**Originating paper/vendor/year:** Judea Pearl's 1980s work and 1988 *Probabilistic Reasoning in Intelligent Systems* established central Bayesian-network foundations. Friedman, Geiger, and Goldszmidt's [1997 "Bayesian Network Classifiers"](https://doi.org/10.1023/A:1007465528199) is a specific supervised reference. A graphical representation is not an invention of a current software provider.

**Core mechanism:** A directed acyclic graph factorizes a joint distribution as $`\prod_j P(X_j\mid\mathrm{Pa}(X_j))`$. For supervised classification, a designated class variable is observed during fitting and inferred for new cases. Fully observed, fixed-structure discrete models can estimate conditional-probability tables from counts and priors. Tree-augmented naive Bayes, TAN, adds a learned tree among predictors conditional on the class, relaxing naive Bayes's independence restriction. General structure search, parameter estimation, and inference are separate problems. An edge does not establish causation without additional assumptions and evidence.

**Inputs/outputs and typical data types:** Discrete or appropriately modeled continuous variables, optionally with missing evidence, and labeled targets for the supervised formulation. Outputs are marginal or conditional distributions, most-probable states, or expected utilities when a separate decision model is supplied.

**Strengths and limitations:** Graphs expose conditional dependence and can combine measurements with expert knowledge. They also support predictions with only some variables observed. Poor structure, sparse conditional tables, and misspecified continuous distributions can undermine predictions. General graph search is computationally difficult; inference may be intractable even when parameter fitting is easy. The [bnlearn classifier documentation](https://www.bnlearn.com/examples/classifiers/) explicitly distinguishes predictive classifier structure from causal structure.

**Computational complexity / scalability notes:** For $`q`$ fully observed discrete variables and fixed graph, counting takes approximately $`O(nq)`$, plus table initialization. With at most $`r`$ states and $`b`$ parents per node, parameter storage can be $`O(qr^{b+1})`$. Exact inference is exponential in graph treewidth $`w`$, with factors on the order of $`r^{w+1}`$; a linear data scan does not imply cheap inference.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** Microsoft's [Lumiere project, reported by Horvitz and colleagues in 1998](https://erichorvitz.com/lumiere.htm), reasons about software users' goals and assistance needs from their background, actions, and queries. Observed interaction events -> Bayesian user-model evidence -> posterior help-topic/need estimates -> candidate assistance is the documented path. The authors report that prototypes supplied the basis for components of the **Office Assistant in Office '97**, not that the entire research system shipped unchanged. Structured dependence and uncertain partial evidence are the technical fit relative to a fixed keyword rule. Crucially, the [paper](https://erichorvitz.com/ftp/lum.pdf) includes expert assessment and usability studies: this is a broader-family application, **not evidence of a wholly supervised parameter-learning recipe**. The supervised classifier formulation above is therefore distinguished from this historical hybrid construction. Production productivity or support-cost KPIs are unreported.

**Notable vendor implementations/libraries:** pgmpy, R `bnlearn`, Bayes Server, and GeNIe/SMILE. Their supported variable families, learning procedures, causal facilities, and decision-analysis extensions differ.

### 1.4.3 Supervised hidden Markov models

**Name:** Supervised hidden Markov models, or supervised HMMs.

**Category & sub-category:** Supervised learning; generative sequence labeling with observed training-state annotations.

**Originating paper/vendor/year:** HMMs developed from 1960s work on probabilistic functions of Markov chains, including Baum and Petrie. Rabiner's [1989 tutorial](https://doi.org/10.1109/5.18626) is a canonical synthesis, not their invention. Brants's [2000 TnT tagger](https://aclanthology.org/A00-1031/) is the concrete supervised example below.

**Core mechanism:** A first-order HMM models an initial state distribution, transitions $`P(z_t\mid z_{t-1})`$, and emissions $`P(x_t\mid z_t)`$. With annotated training states, estimate transitions and emissions from supervised counts or class-conditional estimators, usually with smoothing. Viterbi finds the most probable complete state path; forward-backward computes marginal probabilities. Those are different inference objectives. When training states are **unobserved**, Baum-Welch/EM estimates expected counts instead: see [unsupervised probabilistic models](04-unsupervised-classical.md), rather than relabeling that fitting procedure as supervised.

**Inputs/outputs and typical data types:** Ordered observations such as words, acoustic feature vectors, or sensor readings, paired with state sequences for fully supervised training. Outputs include sequence likelihoods, decoded labels, and state marginals. The word "hidden" refers to the latent-state generative representation and inference-time uncertainty; training annotations can still reveal those states.

**Strengths and limitations:** HMMs give an explicit temporal model, efficient dynamic programming, and a natural treatment of ambiguous observations. Conditional independence and short-memory transitions can miss long-range or overlapping contextual evidence. Sparse transition/emission counts require smoothing. A single state sequence can also be an imperfect annotation of genuinely ambiguous language or physical states.

**Computational complexity / scalability notes:** For $`C`$ states and sequence length $`T`$, dense first-order forward-backward or Viterbi takes $`O(TC^2)`$, plus emission computation; storing a decoded path typically requires $`O(TC)`$ backpointers. Supervised discrete counting is linear in the number of labeled tokens. A second-order transition model can require $`O(TC^3)`$ unpruned inference; beam search trades exactness for speed.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Brants's [TnT paper, arXiv v1 dated 2000-03-13](https://arxiv.org/html/cs/0003055v1), trains a **second-order**, trigram-transition HMM for part-of-speech tagging, with interpolation and unknown-word suffix handling. Annotated Penn Treebank words -> transition/emission estimates -> a decoded tag sequence -> token-level annotation comparison is the workflow. The reported Penn Treebank overall accuracy is **96.7%**, averaged over ten experiments using disjoint **90% training/10% test** partitions with contiguous test segments. This is not a modern standard-split neural benchmark. Sequence transitions resolve ambiguity that independent word classification cannot, while suffix handling addresses unseen words; these are documented design components. The paper establishes a research tagger, not a particular deployed writing assistant or its productivity KPI.

**Notable vendor implementations/libraries:** NLTK's supervised HMM and TnT implementations; `hmmlearn` for HMM inference and fitting. In particular, [`hmmlearn.fit`](https://hmmlearn.readthedocs.io/en/stable/tutorial.html) commonly performs unsupervised EM; using that library does not by itself establish supervised training.

### 1.4.4 Conditional random fields

**Name:** Conditional random fields, or CRFs; the representative formulation is a classical linear-chain CRF with fixed feature functions.

**Category & sub-category:** Supervised learning; globally normalized discriminative sequence labeling.

**Originating paper/vendor/year:** John Lafferty, Andrew McCallum, and Fernando Pereira, "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data," ICML 2001. The [original paper](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf) establishes the model family; CRFsuite is a later implementation.

**Core mechanism:** Model a complete label sequence conditionally on observations:


$$
P(y\mid x)=\frac{1}{Z(x)}
\exp\left(\sum_t\sum_r w_r f_r(y_{t-1},y_t,x,t)\right).
$$


Train by conditional negative log-likelihood, often with $`L_1`$ or $`L_2`$ regularization. Forward-backward computes the partition function and expected feature counts; Viterbi decodes the highest-scoring sequence. Unlike an HMM, the CRF need not model the observation distribution. Unlike a locally normalized label-transition model, its sequence-level normalizer avoids that model's characteristic label-bias mechanism. Features may inspect broad input context while label dependence remains local.

**Inputs/outputs and typical data types:** Annotated token sequences, categorical/numeric feature dictionaries, and label sequences. Outputs include tags and marginal probabilities. BIO entity constraints can be explicitly encoded, but an unconstrained CRF does not automatically forbid every invalid BIO transition.

**Strengths and limitations:** CRFs combine correlated observational features with dependencies between adjacent labels. The fixed-feature linear-chain objective is convex; replacing those features with a learned neural encoder changes the training problem and requires a separate architecture description. Feature engineering, limited label-dependence range, and sequence inference cost are practical limitations. General loopy-graph CRFs can require approximate inference and should not inherit the chain's complexity bound.

**Computational complexity / scalability notes:** For $`C`$ labels and length $`T`$, chain inference costs $`O(TC^2)`$ plus feature scoring. A training pass sums this cost over sequences and also computes feature gradients; total optimization depends on iterations and solver. Dynamic-programming storage is typically $`O(TC)`$, and parameter storage depends on active feature-label and transition weights.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [sklearn-crfsuite CoNLL-2002 tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html) uses Spanish named-entity annotations: `esp.train` for fitting and `esp.testb` for testing. Word identity, suffixes, shape, supplied part-of-speech tags, and neighboring-word features -> joint BIO labels -> entity candidates -> annotation comparison is the workflow. Its initial L-BFGS configuration, with both regularization coefficients set to 0.1 and at most 100 iterations, reports **0.7698023 weighted token-label F1 excluding `O`**. This is **not entity-span F1**, so it should not be ranked directly against CoNLL leaderboard scores. Joint label transitions are a technical advantage over independent token decisions; the tutorial does not establish an enterprise extraction deployment. Production document-processing KPIs are unreported.

**Notable vendor implementations/libraries:** CRFsuite, `python-crfsuite`, `sklearn-crfsuite`, CRF++, and MALLET. Neural encoder-plus-CRF systems belong with the [neural supervised chapter](02-supervised-neural.md), not this fixed-feature instantiation.

### 1.4.5 Cox proportional-hazards regression

**Name:** Cox proportional-hazards regression, or the Cox PH model.

**Category & sub-category:** Supervised learning; semiparametric time-to-event regression with censoring.

**Originating paper/vendor/year:** D. R. Cox, [1972, "Regression Models and Life-Tables"](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x). Subsequent work developed computational methods, residual diagnostics, and extensions; contemporary survival libraries are implementation providers.

**Core mechanism:** Model the hazard as $`h(t\mid x)=h_0(t)\exp(x^\top\beta)`$. Partial likelihood compares each observed event with the subjects still at risk at that time, allowing coefficient estimation without specifying a parametric baseline hazard. Censored subjects contribute while they remain at risk; a censoring time is not a known event time. Tied event times require a declared approximation or exact treatment. To obtain absolute survival probabilities, estimate the cumulative baseline hazard as well.

**Inputs/outputs and typical data types:** Covariates, observed follow-up durations, and event indicators; more advanced formulations also use entry times or time-varying covariates. Outputs include log relative hazards, hazard ratios, and estimated survival curves. A hazard ratio is neither a probability nor a ratio of survival times.

**Strengths and limitations:** Cox regression uses incompletely observed follow-up without pretending censored outcomes are fully known, while leaving baseline hazard shape unspecified. It assumes proportional covariate effects in the stated formulation and appropriately non-informative censoring conditional on the model. Non-proportional hazards, omitted covariates, competing events, and sparse event counts can invalidate simple interpretations. Residual diagnostics and external calibration remain necessary.

**Computational complexity / scalability notes:** Sorting event times costs $`O(n\log n)`$. For fixed covariates and ordinary risk sets, cumulative sums can give an $`O(nd)`$ likelihood/gradient evaluation; a dense Newton Hessian can require $`O(nd^2)`$ work plus an $`O(d^3)`$ solve per iteration. Risk-score prediction is $`O(d)`$ per subject. Survival prediction also needs the estimated baseline; ties and extensions alter implementation costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 survival lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch11-surv-lab.html) analyzes `BrainCancer` follow-up using diagnosis, sex, tumor location, Karnofsky index, gross tumor volume, and treatment type. Its multivariable fit uses **87 complete cases and 35 events** and reports coefficient **2.15457**, hazard ratio **8.62414**, for high-grade glioma relative to meningioma. Clinical covariates plus event/censoring records -> partial-likelihood coefficients -> relative hazards and adjusted survival curves -> prognostic association analysis is the workflow. The number is an adjusted instantaneous hazard ratio, **not** 8.62 times the probability of dying by a chosen date. Cox fits this censored-data problem more naturally than OLS on observed follow-up times, but this analysis does not establish treatment causation, a validated bedside calculator, or improved patient outcomes. Production clinical KPIs are unreported.

**Notable vendor implementations/libraries:** R `survival::coxph`, Python lifelines `CoxPHFitter`, scikit-survival, and statsmodels `PHReg`. Baseline estimation, ties, penalization, and time-dependent-data interfaces differ.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Naive Bayes | Continuous measurements, token counts, or binary indicators by variant | Fast sufficient-statistic learning | Independence assumptions and overconfident probabilities | Iris Gaussian-NB teaching benchmark |
| Bayesian networks | Structured variables with partial evidence | Explicit conditional-dependence model | Expensive inference and structure uncertainty | Microsoft Lumiere/Office Assistant components; hybrid expert construction |
| Supervised HMMs | State-annotated sequences | Efficient generative sequence inference | Short-memory and emission assumptions | TnT Penn Treebank POS tagging |
| Conditional random fields | Annotated sequences with rich fixed features | Joint discriminative sequence labels | Feature engineering; general graphs are harder | CoNLL-2002 Spanish entity-labeling tutorial |
| Cox proportional hazards | Covariates and censored follow-up | Semiparametric event-time modeling | Proportional-hazard and censoring assumptions | `BrainCancer` prognostic association analysis |

## Public bibliography and evidence notes

The inline links identify the source supporting each origin or application claim. These grouped references provide longer reading paths rather than substituting a bibliography for evidence:

- **Early regression and regularization:** [Legendre's least-squares appendix](https://www.york.ac.uk/depts/maths/histstat/legendre.pdf); [Hoerl and Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634); [Tibshirani (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x); [Zou and Hastie (2005 publication)](https://doi.org/10.1111/j.1467-9868.2005.00503.x), with the [August 2004 manuscript](https://hastie.su.domains/Papers/elasticnet.pdf) used for the prostate and leukemia results. Manuscript provenance matters when comparing numbers across editions.
- **Generalized and discriminant models:** [Cox's binary-sequence paper (1958)](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x); [Hastie and Tibshirani's GAM paper (1986)](https://doi.org/10.1214/ss/1177013604); [Fisher (1936)](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x); [Smith's early discrimination paper](https://doi.org/10.1111/j.1469-1809.1946.tb02368.x). The historical venues and terminology do not imply endorsement of their broader social assumptions.
- **Reproducible teaching analyses:** The ISLR2 authors' [classification](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html), [nonlinear modeling](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch7-nonlin-lab.html), [tree ensembles](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch8-baggboost-lab.html), and [survival](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch11-surv-lab.html) labs. These are documented analyses, not deployment studies or uniform benchmark protocols.
- **Resampling and randomized trees:** [Breiman's bagging paper (1996)](https://doi.org/10.1007/BF00058655), [random-forest paper (2001)](https://doi.org/10.1023/A:1010933404324), and [Breiman-Cutler case studies](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm); [Geurts, Ernst, and Wehenkel (2006)](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf). Historical authors' broad promotional statements are not adopted as universal guarantees.
- **Symbolic trees and boosting applications:** [Quinlan's software archive](https://www.rulequest.com/Personal/); [RuleQuest's 2017 comparison](https://www.rulequest.com/see5-comparison.html); [Freund and Schapire (1997)](https://doi.org/10.1006/jcss.1997.1504); [MERL's Viola-Jones record](https://www.merl.com/publications/TR2004-043); [Friedman (2001)](https://doi.org/10.1214/aos/1013203451); [He and colleagues' Facebook report (2014)](https://quinonero.net/Publications/predicting-clicks-facebook.pdf).
- **Scalable boosting systems:** [XGBoost (2016, arXiv v3)](https://arxiv.org/html/1603.02754v3), [LightGBM (2017)](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf), and [CatBoost (2018)](https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf). Their hardware, feature transformations, split strategies, and tuning protocols differ; results do not form one cross-paper ranking.
- **Neighbors and kernel methods:** [Cover and Hart (1967)](https://doi.org/10.1109/TIT.1967.1053964); [Cortes and Vapnik (1995)](https://doi.org/10.1007/BF00994018); [Drucker and colleagues' SVR paper](https://papers.nips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf); [Saunders, Gammerman, and Vovk (1998)](https://eprints.soton.ac.uk/258942/); [Rupp and colleagues (2011 preprint v1; 2012 journal publication)](https://arxiv.org/html/1109.2618v1); [Rasmussen and Williams (2006)](https://gaussianprocess.org/gpml/chapters/).
- **Graphical, sequence, and survival models:** [Maron (1961)](https://doi.org/10.1145/321075.321084); [Friedman, Geiger, and Goldszmidt (1997)](https://doi.org/10.1023/A:1007465528199); [Lumiere (1998)](https://erichorvitz.com/lumiere.htm); [Rabiner (1989)](https://doi.org/10.1109/5.18626); [Brants (2000)](https://aclanthology.org/A00-1031/); [Lafferty, McCallum, and Pereira (2001)](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf); [Cox (1972)](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x).

**Comparability limits.** Classification error, $`R^2`$, log loss, token-label F1, ranking NDCG, and a survival hazard ratio are not substitutes. An out-of-bag estimate differs from an external test; a selected validation maximum differs from a fresh test result. The CRF example uses token-label F1, not entity-span F1. The Facebook numbers are relative normalized cross-entropies, not click-rate or revenue improvements. The GP example's future curve is model extrapolation, not measured future accuracy. The Lumiere application establishes a broader graphical-model use and a documented product connection, not a fully supervised training recipe.

## Coverage and continuation manifest

- **Completed required scope:** 28 individual entries: **1.1.1-1.1.8** linear, generalized, and discriminant models; **1.2.1-1.2.10** trees and ensembles; **1.3.1-1.3.5** neighbors and kernels; **1.4.1-1.4.5** probabilistic, structured, and survival models. Each sub-category has a comparison table covering every entry.
- **Variants made explicit:** binary versus multinomial versus one-versus-rest logistic regression; Gaussian/multinomial/Bernoulli naive Bayes; GP regression versus classification; supervised HMM counting versus unsupervised EM; Bayesian-network classifiers versus expert-elicited or unsupervised uses. No neural instantiation is specified in this chapter.
- **Continue the supervised part:** [Neural supervised models, sections 1.5-1.11](02-supervised-neural.md). A neural encoder can supply inputs to several classical heads, but that composite requires its own architecture, training, and evidence description.
- **Other learning signals:** [Semi-supervised learning](03-semi-supervised.md), [unsupervised classical models](04-unsupervised-classical.md), and [unsupervised/self-supervised neural models](05-unsupervised-neural.md). In particular, unannotated-state HMM EM fitting belongs with unsupervised learning.
- **Larger model families and mixtures:** [Foundation models](06-foundation-models.md), [MoE model catalog](07-moe-models.md), and the [Mixture of Experts deep dive](08-moe-deep-dive.md). An ordinary forest's aggregation is not automatically a learned sparse expert router.
- **Navigation and shared interpretation:** [Reading guide and evidence policy](00-reading-guide.md), [comparative guide](09-comparative-guide.md), and [glossary](10-glossary.md).
- **Non-required extensions not developed here:** robust and quantile regression; probit, ordinal, Poisson, and negative-binomial GLMs as separate entries; MARS and distributional GAMs; oblique/model trees, BART, quantile forests, and online ensembles; full learning-to-rank algorithms; large-scale approximate kernel derivations; Bayesian-network causal discovery and influence diagrams; factorial and semi-Markov sequence models; general-graph CRF inference; competing-risks, accelerated-failure-time, recurrent-event, left-truncation, and time-varying survival-model treatments.
- **Further evidence depth not claimed:** independent reruns of the historical experiments, prospective clinical validation, causal business-impact studies, and an exhaustive inventory of commercial deployments. No benchmark result in this chapter is converted into an invented cost, downtime, productivity, or patient-outcome improvement.
