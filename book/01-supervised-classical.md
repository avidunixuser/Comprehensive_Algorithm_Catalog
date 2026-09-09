# 1. Supervised Learning Algorithms: Classical Models

Supervised learning finds patterns in examples that already have known answers. **Features** are the input details, such as age or temperature. **Labels**, also called targets, are the answers used for learning. Predicting a number is **regression**; choosing a category is **classification**. A **model** is the learned rule that makes those predictions.

The methods in this chapter learn different kinds of rules. Linear models multiply features by learned numbers called **weights**, or coefficients, and add the results. Neighbor methods reuse similar examples. Trees ask a series of questions. Other methods describe chances, label whole sequences, or predict when an event may happen. Event records can include incomplete follow-up, as the Cox entry explains.

This first edition covers 28 entries in sections 1.1-1.4, all without neural networks. A curved probability formula, a rule for measuring similarity, or calculus-based training does not make a model neural. Some families also work without labeled examples. Hidden Markov models can learn without state labels; experts can build Bayesian networks from their knowledge.

**Evidence policy, 2026-09-08.** A method's origin, software implementation, and reported use are separate facts. Historical results keep their original dates, software versions, and test procedures. They do not describe the latest library release. "Research benchmark" includes identified teaching experiments on public data. A benchmark or prototype does not prove that a system was used in everyday operations. A use report does not, by itself, prove financial benefit. All numerical results below come from the cited sources; this book did not rerun the experiments.

Training examples teach the model. Validation examples help choose settings; test examples check the finished choice. **Regularization** means controls that prevent over-reliance on training examples, such as limiting large weights or tree size. It reduces a risk; it does not guarantee good predictions. Settings chosen outside the learned weights are called **hyperparameters**. Trying many settings adds to training cost.

Keep test information out of every learning step. This includes changing feature scales, choosing features, filling missing values, and learning new input summaries. It also includes summaries of labels for categories. **Accuracy** is the fraction of checked predictions that are correct. A model's **probability** is only its estimated chance for one outcome. **Calibration** means those estimated chances match observed frequencies across many cases. Good class separation does not establish calibration, fairness, usefulness, or cause and effect.

The optional cost formulas use $`n`$ for training examples and $`d`$ for features after preparation. Big-O describes how work or storage grows, not elapsed seconds. Unless stated otherwise, time counts arithmetic operations and memory counts stored numbers, not bytes. Each entry explains other symbols and the assumptions behind its costs.

## 1.1 Linear, generalized, and discriminant models

These models make their main choices fairly easy to inspect. Some add weighted inputs to predict a number. Others turn that total into a chance or let each input follow a smooth curve. Discriminant models first describe each category's typical measurements, then decide which description best fits a new case.

### 1.1.1 Ordinary least squares

**In plain English:** OLS fits a line, or a multi-input version of a line, to training examples. Use it as a simple starting point for predicting numbers.

**Name:** Ordinary least squares regression, usually abbreviated OLS.

**Category & sub-category:** Supervised learning; linear regression, which predicts a measured number from weighted inputs.

**Originating paper/vendor/year:** Legendre published least squares in 1805; Gauss developed an early treatment in 1809. The [public translation of Legendre's appendix](https://www.york.ac.uk/depts/maths/histstat/legendre.pdf) describes its roots in astronomy and measuring the Earth. Modern statistics packages implement the method; they did not invent it.

**Core mechanism:** Multiply each input by its weight, add the results, and add a starting value called the intercept. Compare each prediction with its known answer. Square each mistake so positive and negative mistakes cannot cancel. Choose weights that make the total squared error as small as possible. "Linear" describes how weights enter the rule. The inputs can still include squared measurements, category indicators, or other changes chosen beforehand.

**Optional math:** Minimize $`\sum_i(y_i-b-x_i^\top\beta)^2`$. Here $`i`$ indexes examples, $`y_i`$ is the known answer, $`x_i`$ lists its features, $`\beta`$ lists weights, and $`b`$ is the intercept. The expression $`x_i^\top\beta`$ means multiply matching features and weights, then add. Geometrically, the fitted predictions are the closest reachable point to the observed answers, called an orthogonal projection. QR and singular-value decomposition (SVD) are ways to solve the equations more reliably than directly forming $`(X^\top X)^{-1}`$. Here $`X`$ is the table of features, $`\top`$ swaps rows and columns, and the inverse undoes a matrix operation when possible. Bell-shaped, or Gaussian, errors are not needed to calculate OLS. They are assumptions used for some further statistical conclusions.

**Inputs/outputs and typical data types:** A table of numbers and a numerical answer for each training row. Categories must first be represented numerically. The outputs are weights and an estimated average answer for cases with those inputs. A confidence interval describes uncertainty about that average. A prediction interval describes uncertainty about one new observation. Both need extra assumptions about prediction errors.

**Strengths and limitations:** OLS is easy to inspect and cheap to use after training. It describes associations while accounting for the other included inputs. If inputs closely track each other, their separate weights may change sharply between samples. Exact duplication or dependence can leave several equally good weight choices. Unusual observations matter strongly because errors are squared. A straight-line rule can also miss a curved pattern. Uncertainty calculations must account for changing error spread or related observations. A weight does not prove what would happen if someone deliberately changed an input.

**Computational complexity / scalability notes:** For an ordinary filled-in table, more input columns can make training much more expensive. Doubling the columns can make this fitting step take about four times the work. Prediction only multiplies and adds along one row.

**Technical detail (optional):** With $`n`$ examples, $`d`$ features, and $`n\ge d`$, conventional dense QR or SVD costs $`O(nd^2)`$ time. Input storage is $`O(nd)`$; each prediction costs $`O(d)`$. Sparse solvers skip many zero entries and repeatedly refine the answer. Their cost also depends on numerical difficulty and the requested precision; one pass is not always enough.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn diabetes regression example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge.html) predicts a numerical disease-progression measure from body-mass index (BMI). BMI is standardized: its values are put on a common scale. The model uses this single feature and reserves the final 20 observations for testing. It reports test MSE **2548.07** and $`R^2`$ **0.47**. Mean squared error (MSE) averages the squared prediction mistakes; lower is better. $`R^2`$ compares squared error with predicting the test-set average. It is not a percent-correct score.

The model puts BMI into a fitted straight line and returns a progression estimate. The test asks whether that simple rule predicts unseen measurements well enough, not which treatment to prescribe. A tree could learn thresholds, but its fitted rule could vary more between samples. This is a technical comparison, not a claim that clinicians chose OLS. The example shows no clinical deployment, improved patient outcomes, or measured operating benefit.

**Notable vendor implementations/libraries:** R `stats::lm`, statsmodels `OLS`, scikit-learn `LinearRegression`, and MATLAB linear-model routines implement OLS. Check their defaults for intercepts, missing values, example weights, and uncertainty calculations; these differ.

### 1.1.2 Ridge regression

**In plain English:** Ridge fits a weighted sum while discouraging very large weights. This can make predictions steadier when several inputs carry similar information.

**Name:** Ridge regression; linear regression with a penalty on squared weight sizes, called an $`L_2`$ penalty.

**Category & sub-category:** Supervised learning; linear regression with regularization, meaning controls against over-reliance on training examples.

**Originating paper/vendor/year:** Hoerl and Kennard's [1970 paper, "Ridge Regression: Biased Estimation for Nonorthogonal Problems"](https://doi.org/10.1080/00401706.1970.10488634), established ridge regression in statistics. Related penalties also developed in inverse problems, which recover unknown quantities from measurements. Tikhonov regularization is part of that separate history.

**Core mechanism:** Fit a weighted sum, as in OLS, but also charge a cost for large weights. The model then balances fitting the answers against keeping weights small. It usually does not penalize the intercept, the starting value added to every prediction. Unlike lasso, ridge usually reduces weights without removing inputs. Put inputs on suitable scales: measuring the same quantity in different units changes the penalty.

**Optional math:** Add $`\lambda\|\beta\|_2^2`$ to the sum of squared errors. Here $`\beta`$ lists the weights, the squared norm means add their squares, and $`\lambda`$ controls the penalty. For centered inputs and answers, the solution obeys $`(X^\top X+\lambda I)\beta=X^\top y`$. Here $`X`$ is the feature table, $`y`$ lists answers, $`\top`$ swaps rows and columns, and $`I`$ is the identity matrix. A positive $`\lambda`$ steadies weight choices that the data alone cannot pin down. This equation uses a sum of errors, not an average. Libraries that divide that sum by $`n`$, the number of examples, give the same numerical penalty setting a different meaning.

**Inputs/outputs and typical data types:** Numerical input tables, including sparse tables that store mainly nonzero entries, with numerical labels. Ridge returns weights and one or more number predictions. It usually keeps a nonzero weight for every input. Categories can use one-hot coding, with a separate yes/no column for each category. That coding and the intercept choice affect the penalty.

**Strengths and limitations:** Ridge often gives steadier results when many related inputs are useful. It accepts some systematic error to reduce changes caused by the particular training sample. Keeping all inputs can improve stability, but does not give a short list of selected measurements. It cannot automatically fix a curved relationship, bad measurements, or changed future data. A Bayesian view treats weights as initially following a bell-shaped distribution. The fitted weights alone do not describe the full uncertainty after seeing data.

**Computational complexity / scalability notes:** Standard dense fitting becomes expensive as the number of features grows. Predicting a new answer still only requires one weighted sum. If there are far more features than examples, equations based on examples may be smaller.

**Technical detail (optional):** For $`n`$ examples and $`d`$ features, forming and solving dense feature-based normal equations costs $`O(nd^2+d^3)`$ time and $`O(d^2)`$ extra memory. QR/SVD factorizations and repeated-update solvers make different numerical trade-offs. When $`d\gg n`$, meaning many more features than examples, a sample-based "dual" solve can help. Dense prediction costs $`O(d)`$ per case.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Zou and Hastie's [August 2004 elastic-net manuscript, section 4 and Table 1](https://hastie.su.domains/Papers/elasticnet.pdf), tests ridge on Stamey and colleagues' prostate-cancer data. Eight clinical measurements predict log prostate-specific antigen (PSA). A logarithm compresses the scale of the PSA values. There are 67 training observations and 30 test observations. Ten-fold tuning divides only the training data into ten parts, repeatedly fitting on nine and checking the remaining part.

Ridge keeps all eight variables. It reports test MSE **0.566**, versus **0.586** for OLS, in squared log-response units. MSE averages squared prediction errors; lower is better. The model multiplies measurements by reduced weights, adds them, and predicts log PSA for comparison with test answers. Related clinical measurements can make ridge steadier than OLS. That does not show why a hospital would choose it. This small historical experiment does not validate a diagnostic tool or report benefits in clinical operations.

**Notable vendor implementations/libraries:** scikit-learn `Ridge`/`RidgeCV`, R `glmnet` with `alpha=0`, and Spark ML linear regression support ridge. In Spark, use the pure $`L_2`$ option, which penalizes squared weight sizes.

### 1.1.3 Lasso

**In plain English:** Lasso fits a weighted sum and can set some input weights to zero. Use it when you want predictions and a smaller list of inputs to inspect.

**Name:** Least absolute shrinkage and selection operator, or lasso.

**Category & sub-category:** Supervised learning; linear regression with regularization and sparse weights, meaning many weights can be zero.

**Originating paper/vendor/year:** Robert Tibshirani introduced lasso in [1996, "Regression Shrinkage and Selection Via the Lasso"](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). Related methods also developed in signal processing and optimization. Not every method that penalizes absolute weight sizes, called $`L_1`$ regularization, is lasso.

**Core mechanism:** Start with the prediction errors from a weighted sum. Add a cost for the absolute size of each weight. This cost can make exactly zero the best choice for a weight, removing that input from the prediction. One common solver, coordinate descent, adjusts one weight at a time. It checks the errors left after accounting for other inputs, then reduces the proposed weight toward zero. Small proposals become zero; this step is called soft thresholding.

**Optional math:** Minimize $`\|y-b-X\beta\|_2^2/(2n)+\lambda\|\beta\|_1`$. Here $`y`$ lists answers, $`X`$ is the feature table, $`\beta`$ lists weights, $`b`$ is the intercept, and $`n`$ counts examples. The first norm sums squared errors; the second sums absolute weights. The setting $`\lambda`$ controls the penalty. Its absolute-value shape has a corner at zero, which allows exact zero weights. A "regularization path" tracks selected inputs as that setting changes. It does not rank causes. Least-angle regression is another way to find parts of this path.

**Inputs/outputs and typical data types:** Scaled numerical inputs, or numerically encoded categories, paired with numerical answers. Lasso is often used when many candidate inputs may be unhelpful. It returns predictions and the inputs whose weights remain nonzero. Sparse tables can avoid storing entries for absent features.

**Strengths and limitations:** A shorter input list is easier to inspect. However, when several inputs closely track one another, small data changes can change which one survives. Cross-validation does not guarantee that lasso selects the truly relevant inputs. Statistical claims made after choosing inputs on the same data need special methods. Lasso may also remove weak inputs that are useful together; ridge may retain them.

**Computational complexity / scalability notes:** A full round of weight updates scans roughly the whole input table. Training repeats those rounds until the answer is precise enough. Predictions can skip inputs whose weights are zero.

**Technical detail (optional):** With $`n`$ examples and $`d`$ features, maintaining current errors gives about $`O(nd)`$ work per dense sweep. At one penalty setting, $`I`$ sweeps cost $`O(I nd)`$, excluding work to screen out features. Sparse versions depend on $`\operatorname{nnz}(X)`$, the number of nonzero entries in feature table $`X`$. Related inputs, requested precision, and reusing a nearby solution affect the number of sweeps. With $`s`$ nonzero weights, prediction can cost $`O(s)`$.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [prostate-cancer experiment in Zou and Hastie's August 2004 manuscript](https://hastie.su.domains/Papers/elasticnet.pdf) also tests lasso. Eight candidate clinical measurements predict log PSA, the logarithm of prostate-specific antigen. It uses the same 67/30 training/test split and training-only ten-fold tuning. Table 1 reports test MSE **0.499**. MSE is the average squared prediction mistake; lower is better.

Lasso keeps five predictors: log cancer volume, log prostate weight, log benign prostatic hyperplasia, seminal-vesicle invasion, and percentage Gleason score 4 or 5. These record tumor size, prostate measurements, spread, and cell grading. The model combines the retained measurements into a log-PSA estimate, then compares it with held-out answers. Its smaller input list is an advantage over ridge, but does not prove cheaper clinical practice. The selected associations do not show that changing an input changes disease. This error score establishes neither deployment nor diagnostic safety nor a benefit in clinical operations.

**Notable vendor implementations/libraries:** scikit-learn provides `Lasso`, `LassoCV`, and `LassoLars`. R `glmnet` uses `alpha=1`; Spark ML linear regression uses a pure $`L_1`$ penalty, the penalty on absolute weight sizes.

### 1.1.4 Elastic net

**In plain English:** Elastic net combines ridge's small-weight control with lasso's ability to remove inputs. It can keep groups of related inputs rather than choosing just one.

**Name:** Elastic-net regularization.

**Category & sub-category:** Supervised learning; combined controls on absolute and squared weight sizes. These work with linear models and models that transform a weighted score into another kind of prediction.

**Originating paper/vendor/year:** Hui Zou and Trevor Hastie published [2005, "Regularization and Variable Selection Via the Elastic Net"](https://doi.org/10.1111/j.1467-9868.2005.00503.x). The historical examples here come from their [public author manuscript revised August 2004](https://hastie.su.domains/Papers/elasticnet.pdf).

**Core mechanism:** Fit a weighted sum while applying two penalties. One adds absolute weight sizes and can remove inputs. The other adds squared weight sizes and steadies groups of related inputs. Choose both the total penalty strength and the mixture of the two. These are regularization controls against relying too closely on the particular training examples.

**Optional math:** A common modern regression objective is


$$
\frac{1}{2n}\|y-b-X\beta\|_2^2+
\lambda\left[\alpha\|\beta\|_1+\frac{1-\alpha}{2}\|\beta\|_2^2\right].
$$


Here $`n`$ counts examples, $`X`$ is the feature table, $`y`$ lists answers, $`b`$ is the intercept, and $`\beta`$ lists weights. The first term averages squared errors with a factor of $`1/2`$. The setting $`\lambda`$ controls penalty strength; $`\alpha`$ controls the mixture. The $`L_1`$ norm sums absolute weights; the squared $`L_2`$ norm sums their squares. The original paper separates "naive elastic net" from a rescaled version that corrects weights being reduced twice. Do not assume its parameter names mean the same thing as a modern library's `alpha` setting.

**Inputs/outputs and typical data types:** Numerical tables with related columns, such as gene-activity measurements or sparse text features. The squared-error version returns numerical scores. Applying the same penalties to logistic regression creates a different model that estimates class probabilities. Those chances still need checking against observed outcomes.

**Strengths and limitations:** Elastic net balances ridge's steady weights with lasso's shorter input list. With a positive squared-weight penalty, it can keep groups even when features outnumber examples. A selected group is not automatically a meaningful biological group or a set of causes. Choosing two settings takes extra work. Leaking test information during preparation and drawing strong conclusions from small validation sets remain risks.

**Computational complexity / scalability notes:** Each round of one-weight-at-a-time updates scans roughly the input table. Total work grows with the number of rounds and the combinations of penalty settings tried.

**Technical detail (optional):** With $`n`$ examples and $`d`$ features, a dense coordinate-descent sweep that maintains current errors costs $`O(nd)`$. Sparse tables can also save storage. Prediction costs $`O(s)`$ for $`s`$ nonzero weights, plus the separate cost of creating the input features.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [manuscript's leukemia experiment](https://hastie.su.domains/Papers/elasticnet.pdf) uses the Golub gene-expression study to distinguish acute lymphoblastic from acute myeloid leukemia. These are two leukemia types. It uses **squared-loss regression on 0/1 labels followed by a 0.5 threshold**, not elastic-net logistic regression. The fitted score chooses a type according to which side of the threshold it falls on; it is not automatically a reliable probability.

Ten-fold tuning uses 38 training samples. Gene screening happens again inside each training fold, keeping that fold's check data out of selection. A separate 34 samples test the chosen model. Table 4 reports **45 genes** selected, **3/38** cross-validation errors, and **0/34** test errors. Gene-activity measurements become a reduced, weighted score, then a predicted leukemia type. Keeping related genes can be useful compared with arbitrarily picking one. Zero mistakes in this small test set does not mean zero clinical risk. The paper reports no clinical deployment or improvement in patient outcomes.

**Notable vendor implementations/libraries:** R `glmnet` and scikit-learn `ElasticNet`/`ElasticNetCV` provide implementations. Some generalized-linear-model solvers also offer elastic-net penalties; check which prediction rule and loss they use.

### 1.1.5 Logistic regression: binary and multinomial

**In plain English:** Logistic regression adds weighted inputs and turns the total into estimated chances for categories. Use it when you need a simple classification rule you can inspect.

**Name:** Logistic regression. This entry distinguishes binary logistic, multinomial softmax, and one-versus-rest versions.

**Category & sub-category:** Supervised learning; classification using transformed linear scores. Despite its name, logistic regression usually predicts categories, not unrestricted numbers.

**Originating paper/vendor/year:** D. R. Cox's [1958 "The Regression Analysis of Binary Sequences"](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x) is a key modern account of the binary version. It is not the origin of every logistic model. Models for multiple choices have other roots in statistics and economics. Software vendors implement these methods.

**Core mechanism:** In the binary version, multiply inputs by learned weights and add an intercept. Pass this total through an S-shaped curve that returns a chance between zero and one. Training penalizes assigning low chances to the known answers, especially confident mistakes. Regularization can limit weights to reduce over-reliance on training examples.

Multinomial softmax learns scores for all classes together and converts them into chances that sum to one. One-versus-rest instead trains a separate yes/no classifier for each class. These separate estimates are not automatically the same as a jointly fitted multinomial model. Ordinal logistic regression treats ordered categories differently and is outside this entry's detailed scope.

**Optional math:** Binary logistic regression uses $`\Pr(y=1\mid x)=\sigma(b+x^\top\beta)`$. Here $`\Pr`$ means probability, $`y=1`$ denotes one class, $`x`$ lists inputs, $`\beta`$ lists weights, and $`b`$ is the intercept. The product $`x^\top\beta`$ is a weighted sum; $`\sigma`$ is the S-shaped sigmoid function. For multinomial scores $`z_c`$, class $`c`$ gets $`\exp(z_c)/\sum_j\exp(z_j)`$. The exponential function makes each score positive; $`j`$ runs over all classes in the total. The training losses are binary negative log-likelihood and multiclass cross-entropy, both costs for poorly assigned chances. Adding the same value to every class score leaves these probabilities unchanged. A reference-score constraint or regularization resolves that ambiguity.

**Inputs/outputs and typical data types:** Numerical inputs or encoded categories, paired with binary or multi-category labels. Outputs are scores, estimated probabilities, and choices made using a threshold or rule. The threshold need not be 0.5: different mistakes may have different costs. A predicted chance is not the model's measured accuracy.

**Strengths and limitations:** Logistic regression is efficient, inspectable, and often useful for sparse inputs such as text features. Its basic weighted-sum rule may need extra features for curves or joint effects. It assumes linear log odds: the logarithm of an outcome's chance divided by its alternative's chance follows the weighted sum. If the training classes can be perfectly separated, weights may keep growing without regularization. Check rare classes, changing future data, and calibration, meaning whether predicted chances match observed frequencies. A weight-based odds ratio describes an association with other included inputs held fixed. It is not a ratio of probabilities or proof of cause and effect.

**Computational complexity / scalability notes:** Each basic update pass works through the inputs for each class. More powerful update methods can take much more work per pass, and datasets need different numbers of passes.

**Technical detail (optional):** With $`n`$ examples, $`d`$ features, and $`C`$ jointly modeled classes, a dense gradient pass costs $`O(ndC)`$ and model storage is $`O(dC)`$. A gradient tells which direction reduces the loss. Binary Newton or iteratively reweighted least-squares (IRLS) updates can cost $`O(nd^2+d^3)`$ per iteration. These methods use extra information to choose an update; there is no universal iteration count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 classification lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html) predicts S&P 500 daily direction in `Smarket`. Its inputs, `Lag1` and `Lag2`, are the first two previous daily returns. It trains before 2005 and tests on 252 days in 2005. The model gets **141/252**, or **55.95%**, correct. Always predicting "Up" achieves exactly the same test accuracy.

For the supplied input `(1.2, 1.1)`, the model estimates probability **0.4791462** for "Up" and therefore chooses "Down" at the demonstration's threshold. That is a chance estimate for one input, not an accuracy score. Past returns enter a weighted score, which becomes a chance and then a direction. This is an inspectable alternative to a nonlinear tree, not evidence of profitable trading. Same-day `Today` must not be an input: it reveals information about the answer. The lab reports no trading deployment, profit, or results after transaction costs.

**Notable vendor implementations/libraries:** R `glm(family=binomial)` fits binary models; R `nnet::multinom` fits classical multinomial regression. Other options are scikit-learn `LogisticRegression` and statsmodels `Logit` and `MNLogit`.

### 1.1.6 Generalized additive models

**In plain English:** A GAM learns a separate smooth curve for each input and adds their contributions. Use it when straight lines are too rigid but you still want to inspect individual patterns.

**Name:** Generalized additive models, or GAMs.

**Category & sub-category:** Supervised learning; models that add understandable curved effects instead of requiring every input to follow a straight line.

**Originating paper/vendor/year:** Trevor Hastie and Robert Tibshirani published [1986, "Generalized Additive Models"](https://doi.org/10.1214/ss/1177013604), followed by a 1990 book. Later software, especially Simon Wood's `mgcv`, added important ways to choose smoothness and speed up fitting.

**Core mechanism:** Learn a smooth curve for each input, such as how predicted wage changes with age. Add the curve contributions and a starting value. For some targets, a final conversion turns that total into the required scale, such as a probability. Penalties discourage curves from bending around every training example. Joint effects of multiple inputs can be added, but must be specified.

**Optional math:** $`g(\mathbb E[y\mid x])=b+\sum_j f_j(x_j)`$. Here $`y`$ is the answer, $`x`$ lists the inputs, and $`\mathbb E[y\mid x]`$ means the average answer expected for those inputs. The function $`g`$, called a link, converts that average to the score scale. The starting value is $`b`$; $`f_j`$ is the curve for input $`x_j`$; the sum runs over inputs $`j`$. A spline basis builds curves from smooth joined pieces. Centering each curve separates its contribution from the intercept. Fitting can repeatedly update curves ("backfitting"), solve penalized weighted equations, and choose smoothness settings. These are different, sometimes combined steps, not one universal GAM algorithm.

**Inputs/outputs and typical data types:** Measurements, categories, dates or seasons, and locations, with an appropriate kind of answer. The chosen response model might describe bell-shaped measurement errors (Gaussian), yes/no outcomes (binomial), or event counts (Poisson). Outputs include predictions, each input's fitted curve, and uncertainty intervals that depend on the model's assumptions.

**Strengths and limitations:** Separate curves reveal patterns while often needing fewer examples than a model with unrestricted joint effects. However, adding contributions can miss cases where one input changes another input's effect. Combining many multi-input curves, including tensor-product smooths, makes the model harder to explain. If two curves can mimic each other, their separate contributions become hard to pin down. This is called concurvity, a curved-pattern version of closely related linear inputs. Beyond the observed input range, the chosen curve pieces and constraints determine predictions. Smoothness does not guarantee sensible extension.

**Computational complexity / scalability notes:** More curve pieces let a GAM describe finer shapes, but also create more weights to fit. Dense fitting can become much more expensive as that number grows.

**Technical detail (optional):** Let $`n`$ count examples and $`q`$ count all curve and other model weights. A dense penalized weighted-least-squares update can cost $`O(nq^2+q^3)`$ time and $`O(nq+q^2)`$ storage. Local curve pieces, sparse storage, grouping nearby values, or building inputs in batches can change these costs. Evaluating curve pieces and making a prediction typically costs $`O(q)`$ per example.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 nonlinear-modeling lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch7-nonlin-lab.html) analyzes `Wage` using year, age, and education. Its comparison favors a straight-line year effect, a smooth age curve, and category indicators for education. Making year curved too gives an analysis-of-variance (ANOVA) $`p`$-value of **0.3485661**. This test asks whether the added curve improves the fit beyond what chance could explain under the simpler model's assumptions. The $`p`$-value is not the chance that the model is correct. It is an in-sample comparison, not accuracy on unseen data.

The inputs become separate contributions, which add to a wage estimate. The curves let readers inspect age and calendar-year associations. Unlike straight-line OLS, the age effect can change slope without a complicated set of tree rules. The lab does not show that education causes a particular wage increase. It reports no wage-setting deployment or measured economic benefit in practice.

**Notable vendor implementations/libraries:** R `mgcv` and `gam`, Python pyGAM, and statsmodels support GAMs. Their curve pieces, default penalties, and uncertainty calculations differ.

### 1.1.7 Linear discriminant analysis

**In plain English:** LDA describes each class as a cloud of measurements with its own center but a shared shape. It classifies a new case by checking which cloud best explains its measurements.

**Name:** Linear discriminant analysis, or LDA; not latent Dirichlet allocation.

**Category & sub-category:** Supervised learning; classification by modeling the measurements within each class. It can also turn many inputs into fewer coordinates that help separate labeled groups.

**Originating paper/vendor/year:** R. A. Fisher's [1936 "The Use of Multiple Measurements in Taxonomic Problems"](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) is the key early reference. Modern classification with Gaussian class models is related, but software does not necessarily follow Fisher's original procedure exactly.

**Core mechanism:** For each class, estimate the average of every input. Estimate a shared pattern of spread and of which inputs move together. That pattern is the **covariance**; it describes the cloud's shape. LDA assumes bell-shaped, or Gaussian, clouds with this same shape. It also accounts for how common each class is, called its **prior** probability. Compare how well the clouds explain a new case and choose a class. The shared shape makes the boundaries between classes straight.

**Optional math:** The score for class $`c`$ is $`x^\top\Sigma^{-1}\mu_c-\frac12\mu_c^\top\Sigma^{-1}\mu_c+\log\pi_c`$. Here $`x`$ lists the new measurements, $`\mu_c`$ lists class averages, $`\Sigma`$ is the shared covariance, and $`\pi_c`$ is the class prior. The transpose $`\top`$ turns a column into a row; $`\Sigma^{-1}`$ adjusts for shared spread and linked inputs; $`\log`$ is a logarithm. Comparing the Gaussian class scores cancels their common squared-input term, leaving linear scores. A related projection chooses new coordinates that separate class centers relative to spread within classes. Unlike principal component analysis (PCA), which summarizes inputs alone, this projection uses labels.

**Inputs/outputs and typical data types:** Numerical measurements and known classes. Outputs include predicted classes and probabilities updated after seeing the measurements, called posterior probabilities. LDA models the measurements within each class, not just a direct input-to-class rule. It can also return a smaller set of separating coordinates.

**Technical detail (optional):** With $`d`$ input features and $`C`$ classes, there are at most $`\min(d,C-1)`$ such coordinates: the smaller of the input count and one less than the class count.

**Strengths and limitations:** Sharing the shape across classes reduces how much must be learned and can help with modest datasets. The same method handles multiple classes. However, real classes may not have bell-shaped clouds or matching shapes. With many inputs, estimated shapes can become unreliable. Pulling estimates toward simpler shapes, called shrinkage, may help. It cannot fix wrong labels or badly chosen assumptions. Clear separation in a plot does not prove the estimated probabilities match real frequencies.

**Computational complexity / scalability notes:** Learning relationships between all pairs of inputs can be costly when there are many columns. Once training is finished, a new prediction needs only a weighted score for each class.

**Technical detail (optional):** With $`n`$ examples, $`d`$ features, and $`C`$ classes, dense covariance-based fitting costs $`O(nd^2+d^3+Cd^2)`$ time. Model storage is $`O(d^2+Cd)`$. Precomputed linear scores give $`O(Cd)`$ prediction work per example. Singular-value decomposition (SVD), a way of factoring the data table, can avoid building the covariance explicitly and may help when $`d`$ is large.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 `Smarket` experiment](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html) fits LDA with two previous daily market returns. Training uses data before 2005; testing uses 252 daily directions in 2005. Its confusion matrix, a table of predicted versus actual classes, matches binary logistic regression's exactly: **141 correct**, or **55.95%**. The first test observation gets estimated probability **0.4901792** for "Down", so LDA chooses "Up". That chance is not a measured success rate.

LDA compares past-return measurements with its shared-shape class models, chooses a direction, and checks it in the later period. Sharing shape information may help when there are too few labels to fit QDA's separate shapes well. This is a technical reason, not an investment firm's documented choice. The benchmark does not beat always predicting "Up", establish a trading strategy, or report profit in practice.

**Notable vendor implementations/libraries:** R `MASS::lda` and scikit-learn `LinearDiscriminantAnalysis` provide LDA. Check how each estimates covariance, sets prior class chances, and solves the equations.

### 1.1.8 Quadratic discriminant analysis

**In plain English:** QDA gives each class its own measurement cloud, including its own shape and spread. This allows curved class boundaries, but needs more examples than sharing one shape.

**Name:** Quadratic discriminant analysis, or QDA.

**Category & sub-category:** Supervised learning; classification by fitting a separate bell-shaped measurement model for each class.

**Originating paper/vendor/year:** QDA comes from classical statistics, not a software vendor. C. A. B. Smith's ["Some Examples of Discrimination"](https://doi.org/10.1111/j.1469-1809.1946.tb02368.x) is an early reference. The journal's DOI record dates it to 1946. This is not a claim that Smith uniquely invented every QDA variant. The version below uses the standard [Gaussian Bayes rule](https://scikit-learn.org/stable/modules/lda_qda.html).

**Core mechanism:** Estimate each class's average measurements, how widely they vary, and which measurements move together. The spread and linked movement form that class's covariance. Also estimate the class's starting frequency, or prior. For a new case, compare how well each Gaussian, bell-shaped cloud explains its measurements. Different cloud shapes produce curved boundaries without a neural network. Ignoring all within-class links between different inputs gives the usual class-specific Gaussian naive-Bayes model.

**Optional math:** A class score contains $`-\frac12\log|\Sigma_c|-\frac12(x-\mu_c)^\top\Sigma_c^{-1}(x-\mu_c)+\log\pi_c`$. Here $`c`$ identifies the class, $`x`$ lists measurements, $`\mu_c`$ lists class averages, $`\Sigma_c`$ describes its covariance, and $`\pi_c`$ is its prior. The determinant $`|\Sigma_c|`$ measures the cloud's overall spread; the middle term measures distance adjusted for its shape. The inverse $`\Sigma_c^{-1}`$ supplies that adjustment, $`\top`$ transposes a vector, and $`\log`$ means logarithm. Unlike LDA, the squared terms differ across classes and do not cancel. A diagonal covariance keeps each input's spread but removes links between inputs.

**Inputs/outputs and typical data types:** Numerical measurements and class labels, with enough examples in each class to estimate its shape. Outputs are class choices and updated class chances, called posterior probabilities. Each class is modeled as one oval-like cloud in the measurement space. QDA cannot describe every possible shape or several separate clusters within one class.

**Strengths and limitations:** QDA can learn different spreads and linked-input patterns for different classes. It is more flexible than LDA without needing a general search over nonlinear prediction rules. However, a small class may not supply enough data for a reliable, usable shape estimate. This is especially risky when classes have very different sizes. Regularization can move estimates toward simpler shapes to reduce over-reliance on a small training sample. Unmodified QDA is usually a poor choice when each class has few examples relative to the number of inputs.

**Technical detail (optional):** With $`C`$ classes and $`d`$ features, the symmetric covariance estimates require roughly $`C d(d+1)/2`$ entries. Software can exploit symmetry, but there are still many relationships to estimate.

**Computational complexity / scalability notes:** QDA must store a shape for every class and check pairs of input values during prediction. Doubling the input count can make this prediction work about four times larger.

**Technical detail (optional):** With $`n`$ examples, $`d`$ features, and $`C`$ classes, dense fitting takes approximately $`O(nd^2+Cd^3)`$. Model storage is $`O(Cd^2)`$. Full quadratic-score prediction costs $`O(Cd^2)`$ per case. These costs assume conventional dense factorizations and exclude creating the features.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 `Smarket` lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html) uses the same two previous daily returns and pre-2005 training period as LDA. QDA gets **151 correct predictions out of 252** in 2005: **59.92%**, versus LDA's **55.95%** on that exact period. The prediction table contains 30 correctly predicted Down days and 121 correctly predicted Up days.

The two return measurements are checked against separate Gaussian models for Up and Down. The resulting chances choose a direction, which is compared with the held-out answer. Different covariance shapes let QDA fit more patterns. They do not prove that real markets follow those shapes. One short test period establishes neither a lasting advantage nor a profitable statistical trading strategy. No positive net return or measured benefit in trading operations is reported.

**Notable vendor implementations/libraries:** R `MASS::qda` and scikit-learn `QuadraticDiscriminantAnalysis` implement QDA. They differ in regularization options and in handling covariance estimates that cannot be inverted.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Ordinary least squares | Numerical inputs and number answers | Easy-to-inspect starting model for average answers | Related inputs and large errors can distort weights | Diabetes progression from BMI; teaching test |
| Ridge regression | Related numerical measurements | Steadier weights while keeping inputs | Does not shorten the input list | Historical prediction of log PSA in prostate data |
| Lasso | Many inputs, some possibly unhelpful | Predicts and removes some inputs | May swap choices among similar inputs | Five-input prostate log-PSA model |
| Elastic net | Many related inputs | Can keep groups while removing other inputs | Two settings to tune; selected inputs may change | Golub leukemia gene-activity research test |
| Logistic regression | Numerical or encoded tables, including many zeros | Direct chances for two or more classes | Basic weighted score misses some curved patterns | `Smarket` direction test; no measured trading benefit |
| Generalized additive models | Measurements with smooth trends and categories | Separate curves are easy to inspect | Adding separate effects can miss joint effects | `Wage` analysis of age, year, and education |
| Linear discriminant analysis | Numerical classes with similar cloud shapes | Shares shape estimates across classes | Assumes matching bell-shaped clouds | `Smarket` classifier with shared shape |
| Quadratic discriminant analysis | Numerical classes with different spreads | Separate cloud shapes allow curved boundaries | Needs enough data for each class's shape | `Smarket` comparison with class-specific shapes |

## 1.2 Trees and ensembles

A tree asks a sequence of questions about the inputs. Each answer leads to another question or a final prediction. An **ensemble** combines several models. Bagging trains models on resampled data and averages or votes to make results steadier. Boosting builds models in sequence to improve earlier predictions. It can correct systematic mistakes as well as reduce variation.

XGBoost, LightGBM, and CatBoost add different algorithms and engineering choices to gradient boosting. They are not separate inventions of decision trees. A tree's **depth** is the number of questions along a path; its final prediction points are **leaves**. Choosing the best next split, called greedy training, does not guarantee a balanced or globally best tree.

Optional cost formulas below use $`M`$ for trees, $`h`$ for maximum depth, $`q`$ for candidate inputs at a split, and $`v`$ for stored nodes per tree. A node is a question or a leaf. Sorting inputs, checking possible splits, and making predictions have separate costs. Categories and missing values can add work.

### 1.2.1 CART

**In plain English:** CART learns a series of yes/no questions that lead to a class or number. A small tree is useful when you want to follow a prediction step by step.

**Name:** Classification and regression trees, or CART.

**Category & sub-category:** Supervised learning; trees that repeatedly divide examples into two groups.

**Originating paper/vendor/year:** Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone described CART in *Classification and Regression Trees*, 1984. The [scikit-learn tree documentation](https://scikit-learn.org/stable/modules/tree.html) describes its optimized CART implementation. It also explains that CART, ID3, C4.5, and C5.0 are different methods.

**Core mechanism:** Try questions such as whether a measurement exceeds a threshold. Choose the question that best improves the current groups, then repeat within each group. For classification, Gini impurity measures how mixed the labels are; a useful split reduces that mixing. For regression, a useful split reduces squared prediction errors. Each final leaf stores a class distribution or a number estimate.

Pruning removes branches to balance fit against the number of leaves. Cost-complexity pruning builds a sequence of smaller trees; validation data help choose a size. This control reduces over-reliance on training examples. Choosing the best immediate split does not generally find the best possible whole tree. Some CART versions support category-group splits or substitute questions when a value is missing. Not every library provides those features.

**Inputs/outputs and typical data types:** Tables with class labels or numerical answers. Numerical thresholds work directly. Unordered categories may need encoding unless the library handles them itself. Prediction follows conditions to a leaf. Check how the chosen implementation handles missing values.

**Strengths and limitations:** Small trees are easy to inspect and need little input scaling. They learn joint effects because an earlier answer changes which questions come next. However, small changes in training data can produce a different tree. Splitting one input at a time may need many steps to approximate a slanted boundary involving several inputs. An unpruned tree can memorize training details. Regression leaves repeat learned values rather than extending a smooth trend beyond the data. A readable rule is not automatically fair or causal.

**Computational complexity / scalability notes:** Training checks many candidate questions. Reusing sorted inputs saves work. A balanced tree needs relatively few questions per prediction; a long, uneven tree can be much slower.

**Technical detail (optional):** For $`n`$ examples and $`d`$ features, reusable numeric sort orders and balanced growth can give $`O(dn\log n)`$ split-scanning work. Sorting again at every node may give $`O(dn\log^2 n)`$. Very unbalanced trees cost more. With maximum depth $`h`$ and $`v`$ stored nodes, prediction costs $`O(h)`$ for one fully observed case and storage is $`O(v)`$, excluding training data. The logarithm here reflects the number of balanced splitting levels.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn pruning demonstration](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html) uses Wisconsin Diagnostic Breast Cancer data. Its inputs are measurements of cell nuclei from breast masses. It uses the library's default 75%/25% training/test split with `random_state=0`. Each case follows measurement thresholds to a benign or malignant label, then is checked against its recorded label. This is not a clinical recommendation.

The unpruned tree reaches **100% training accuracy** but only **88% testing accuracy**. Accuracy means the fraction correctly classified. Pruning improves the displayed held-out results. However, the example uses that curve to choose pruning strength, so those data serve as validation, not an untouched final clinical test. The tree makes combined thresholds visible; logistic regression would need those combinations encoded as inputs. This does not show a hospital's reason for choosing CART. Results from diagnostic use in practice are unreported.

**Notable vendor implementations/libraries:** R `rpart`, scikit-learn `DecisionTreeClassifier`/`DecisionTreeRegressor`, and Spark ML provide CART or CART-derived learners. Their available options do not reproduce every feature of the complete 1984 system.

### 1.2.2 C4.5

**In plain English:** C4.5 learns questions for classifying mixed numerical and category data. It can also turn the tree into a set of if-then rules.

**Name:** C4.5 decision-tree and rule induction, meaning learning trees and rules from examples.

**Category & sub-category:** Supervised learning; classification trees that choose splits by how much they reduce uncertainty about labels.

**Originating paper/vendor/year:** J. Ross Quinlan described it in *C4.5: Programs for Machine Learning*, 1993, published by Morgan Kaufmann. His [author software page](https://www.rulequest.com/Personal/) supplies historical C4.5 Release 8. It distinguishes this software from the later C5.0 product.

**Core mechanism:** C4.5 extends ID3 with numerical thresholds, missing-value handling, and pruning. It scores splits using information gain: how much a split reduces uncertainty about labels. It adjusts that score for how the split divides cases across branches, producing a gain ratio. This reduces the tendency to favor inputs with many possible values. Further checks prevent a large ratio based on almost no useful gain.

An unordered category can create several branches, unlike CART's strictly two-way splits. After growing a tree, C4.5 can simplify it or convert it into a pruned ruleset. The ruleset may have a different size and error rate from the tree.

**Inputs/outputs and typical data types:** Tables containing numerical measurements and unordered categories, with class labels. A prediction follows a tree path or matches a rule. Historical C4.5 shares cases with unknown values across branches using weights, which specify how much each branch receives. It does not simply replace every missing value with zero.

**Strengths and limitations:** C4.5 handles mixed inputs and produces rules people can inspect. However, it chooses locally useful splits rather than the best whole tree. Wrong labels, categories with many possible values, and small data changes can cause problems. Gain ratio reduces one split-selection bias; it does not make every input equally easy to interpret. A library that implements CART should not be described as C4.5.

**Computational complexity / scalability notes:** Repeatedly sorting measurements can be a large part of training. Sorting beforehand may help. Missing values can send a case along several branches, so prediction need not follow just one short path. Extracting rules adds separate, sometimes substantial work.

**Technical detail (optional):** One implementation-dependent sorting cost is $`\sum_u O(d n_u\log n_u)`$. Here $`u`$ indexes nodes, $`n_u`$ counts cases at that node, and $`d`$ counts features. Add the sorting costs across all nodes. Tree depth and sharing weighted missing cases complicate the total. A single determined path takes approximately $`h`$ tests, where $`h`$ is its depth; missing values may require more branches.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [RuleQuest's February 2017 comparison](https://www.rulequest.com/see5-comparison.html) predicts one of seven forest-cover classes from environmental inputs. The tree follows conditions and checks its chosen class against a held-out label. The vendor splits 581,012 cases into equal training and test halves. **C4.5 Release 8 trees** have **6.1% test error** and **10,169 leaves**. Test error is the fraction of wrong classifications; lower is better. The separate C4.5 ruleset has a different result and must not be substituted.

Rules using mixed input types can be easier to explain than a large ensemble. That is not evidence of a forestry agency's deployment choice. The comparison comes from the developer of C4.5's successor, not an independent evaluator. It reports no measured benefit in forestry operations.

**Notable vendor implementations/libraries:** Options include Quinlan's historical C4.5 source and Weka's `J48` implementation. RuleQuest See5/C5.0 adds algorithms as a successor product; it is not simply a new name for C4.5.

### 1.2.3 Bagging

**In plain English:** Bagging trains several models on different resampled versions of the same data, then combines their answers. It can make an unstable model, such as a tree, give steadier predictions.

**Name:** Bootstrap aggregating, or bagging.

**Category & sub-category:** Supervised learning; ensembles built by resampling training examples and combining fitted models.

**Originating paper/vendor/year:** Leo Breiman published [1996, "Bagging Predictors"](https://doi.org/10.1007/BF00058655), following a Berkeley technical report in 1994. Bagging can combine many kinds of learner, not just trees.

**Core mechanism:** Draw training rows at random with replacement, so a row can appear more than once. This creates a bootstrap sample. Repeat to make several samples and train a separate base model on each. Average numerical predictions; for classes, use votes or average estimated probabilities. The different samples change unstable learners in different ways. Combining their answers can reduce that variation, but shared mistakes remain.

**Optional math:** If $`M`$ predictors have equal variance $`\sigma^2`$ and equal pairwise correlation $`\rho`$, their average has variance $`\sigma^2[\rho+(1-\rho)/M]`$. Variance describes how predictions vary across training samples; correlation describes how closely they vary together. Adding models reduces the separate-error part, not the shared part. An **out-of-bag** prediction uses only models whose bootstrap samples left out the case being checked.

**Inputs/outputs and typical data types:** Any labeled data that the chosen base learner can handle. Here the representative version uses regression trees and tabular measurements. It returns a combined prediction and may provide out-of-bag checks. A generic bagging wrapper does not add neural layers or define a new neural network.

**Strengths and limitations:** Independent fits can run at the same time and often stabilize trees. Bagging may add little to an already steady model, and cannot reliably remove shared systematic mistakes. Resampling rows independently may be wrong for related people, nearby locations, or time sequences. Out-of-bag checks are misleading if earlier label-based preparation already used the omitted cases. Repeatedly selecting models using the same out-of-bag score also weakens it as a final check.

**Computational complexity / scalability notes:** More models mean proportionally more fits, predictions, and storage. Running fits together can shorten elapsed time, but processors and memory still limit speed.

**Technical detail (optional):** Let $`F(n,d)`$ be one model's fitting cost for $`n`$ examples and $`d`$ features. Training $`M`$ models costs approximately $`O(MF(n,d)+Mn)`$, including resampling. For trees of depth $`h`$ with $`v`$ nodes each, prediction costs $`O(Mh)`$ per example and storage is $`O(Mv)`$ plus data. Memory-transfer speed and making data copies can limit parallel work.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 tree-ensemble lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch8-baggboost-lab.html) fits 500 bagged trees to historical `Boston` housing data. Each split considers all 12 supplied inputs. The 253/253 training/test experiment reports test MSE **23.41916** for neighborhood median home value. MSE averages squared prediction errors; lower is better. Its units here are squared thousands of dollars, not dollars.

Each tree predicts from neighborhood inputs, then the model averages the predictions and checks observed median values. This shows how averaging can steady a single-tree approach. The data are a legacy teaching benchmark, not recommended evidence for lending or valuation. Demographic proxy variables can stand in for personal characteristics, and the historical sampling needs particular caution. The result depends on the documented R/package setup and cannot be directly compared with another paper's different Boston split. No benefit in property-pricing operations is reported.

**Notable vendor implementations/libraries:** scikit-learn provides `BaggingClassifier`/`BaggingRegressor`; MATLAB also has ensemble tools. R `randomForest` can perform tree bagging when all predictors are considered at each split.

### 1.2.4 Random forest

**In plain English:** A random forest combines trees that see different samples and different choices of inputs. This helps prevent the whole group from repeating the same tree's mistakes.

**Name:** Random forest.

**Category & sub-category:** Supervised learning; ensembles that randomize both training samples and candidate tree inputs.

**Originating paper/vendor/year:** Leo Breiman's [2001 "Random Forests"](https://doi.org/10.1023/A:1010933404324) formalized this influential method. It built on bagging and earlier work on randomized trees and random input subsets, including Tin Kam Ho's contributions.

**Core mechanism:** Make a bootstrap sample by drawing training rows with replacement. Grow a tree on it, but at each split allow only a random subset of inputs as candidates. Repeat for many trees, then combine votes, probabilities, or numerical predictions. This stops a few strong inputs from dominating every tree. Trees may make fewer shared mistakes, although each individual tree may become weaker. Breiman's classification version grows trees without pruning. Modern libraries also offer controls on depth and leaf size.

**Inputs/outputs and typical data types:** Tables with mixed measurements, prepared as the library requires, and class or numerical labels. Outputs include combined predictions, individual tree results, and often out-of-bag checks using trees that did not train on the checked case. Input-importance scores need careful naming. Impurity importance counts improvements at splits; permutation importance checks how predictions worsen when an input's values are shuffled. These measure different things.

**Strengths and limitations:** Forests learn curved patterns and joint effects with little scaling of measurements. However, regression predictions generally do not extend trends well beyond observed data, and many trees can use substantial memory. Split-based importance can favor inputs with many distinct values. Closely related inputs can share or hide the importance seen by shuffling. More trees reduce variation from random sampling, not every form of overfitting. Depth, repeated tuning, and leaked answers can still cause trouble. Check calibration: predicted chances may not match observed frequencies.

**Computational complexity / scalability notes:** More trees, candidate inputs, or levels mean more split checks. Trees can be fitted in parallel. Memory and prediction work grow with the size of the forest.

**Technical detail (optional):** With $`M`$ trees, $`q`$ candidate features per split, $`n`$ examples, depth $`h`$, and reusable sort orders, split scanning is roughly $`O(Mqnh)`$. Sorting, sampling, and bookkeeping add work; sorting again at each node changes the bound. With $`v`$ nodes per tree, storage is $`O(Mv)`$ beyond data. Prediction costs $`O(Mh)`$ per case.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Breiman and Cutler's [author-hosted microarray case study](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#micro) classifies early lymphoma data. A microarray measures activity across many genes. This dataset has **81 cases, 4,682 expression variables, and three classes**. The run uses **1,000 trees** and 150 candidate variables at each split. It reports **1.23% out-of-bag error**, corresponding to one incorrectly classified case.

For each gene-activity profile, only trees that left that case out contribute evaluation votes. Their chosen class is checked against the recorded label. Random input choices can help when genes greatly outnumber samples and one tree would be unstable. This is a technical argument, not a clinical model-selection decision. Out-of-bag error is not an independent external test or evidence of improved lymphoma care. No measured benefit in diagnostic operations is reported.

**Notable vendor implementations/libraries:** R `randomForest` and `ranger`, scikit-learn, and Spark ML provide random forests. Defaults differ for candidate inputs, combining probabilities, and handling missing values.

### 1.2.5 Extremely randomized trees

**In plain English:** Extra-Trees tries randomly chosen split points instead of searching every possible threshold. Combining many such trees can save training work while still learning useful patterns.

**Name:** Extremely randomized trees, or Extra-Trees.

**Category & sub-category:** Supervised learning; ensembles that randomize both candidate features and split thresholds.

**Originating paper/vendor/year:** Pierre Geurts, Damien Ernst, and Louis Wehenkel published [2006, "Extremely Randomized Trees"](https://orbi.uliege.be/handle/2268/9357). Their institution provides the [full author paper](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf).

**Core mechanism:** At each node, choose candidate inputs and random threshold values. Use the known labels to choose the best of those proposed splits. Repeat to grow trees, then combine their predictions. The original default trains on the full learning sample rather than bootstrap samples. Random thresholds make trees differ and reduce search work. The standard algorithm is **not** label-independent: labels choose among candidate splits and determine leaf predictions. The paper also studies more extreme forms of randomization.

**Inputs/outputs and typical data types:** Numerical tables or encoded categories, paired with class or number answers. Predictions are tree averages or votes. Some libraries allow bootstrap sampling, but it is an optional setting, not a requirement of Extra-Trees.

**Strengths and limitations:** Extra-Trees can work quickly when many roughly useful splits are available. Randomization may steady results across samples while making each fit less precise. This can hurt when an exact, narrow threshold matters. Like other one-input-at-a-time forests, it does not automatically extend trends beyond the data or give meaning to arbitrary category ID numbers. Still use suitable train/test splits, check estimated chances against observed frequencies, and test sensitivity to data changes.

**Computational complexity / scalability notes:** Trying random thresholds can avoid the global sorting used in exact split searches. Training still scans cases to score the proposed splits. More trees, deeper paths, and more proposed thresholds add work.

**Technical detail (optional):** Suppose each of $`q`$ features supplies one candidate cut and each node scans its cases. With $`M`$ trees, $`n`$ examples, and depth $`h`$, training costs approximately $`O(Mqnh)`$. Prediction costs $`O(Mh)`$; storing $`v`$ nodes per tree costs $`O(Mv)`$. Node sizes, memory access, stopping rules, and the number of candidate thresholds affect actual speed.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Geurts and colleagues' [Letter-recognition experiment, Tables 7-8](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf), uses 16 numerical summaries of images to classify 26 letters. It divides the data into 10,000 learning and 10,000 test cases, repeating this dataset's experiment for ten random splits. The default Extra-Trees column reports mean test error **3.80%**. The paper's best-setting random-forest comparison, labeled `RF*`, reports **4.87%**. These are average wrong-classification rates; lower is better.

Image summaries pass through randomized tree thresholds, produce letter votes, and are checked against test labels. Avoiding exhaustive threshold searches is a technical advantage over ordinary forests. The numbers describe the authors' settings, not today's default libraries. They establish no postal deployment, labor savings, or measured transcription benefit in production.

**Notable vendor implementations/libraries:** scikit-learn provides `ExtraTreesClassifier` and `ExtraTreesRegressor`. Other forest toolkits offer extra-randomized splits, but check whether their details match the original method.

### 1.2.6 AdaBoost

**In plain English:** AdaBoost builds simple classifiers one after another, giving more attention to earlier mistakes. It combines their votes into a stronger classifier.

**Name:** Adaptive boosting, or AdaBoost.

**Category & sub-category:** Supervised learning; ensembles that repeatedly change how much training examples influence the next classifier.

**Originating paper/vendor/year:** Yoav Freund and Robert Schapire published "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting" in a [1997 journal article](https://doi.org/10.1006/jcss.1997.1504), following a 1995 conference paper. Later versions for multiple classes differ from the original binary method.

**Core mechanism:** Give each training example an importance weight, then fit a simple, or weak, classifier. Increase the relative importance of cases it gets wrong. Fit the next classifier with these new example weights. Also give each classifier a voting weight based on its performance. The final answer combines these weighted votes. Example weights control which cases matter in training; voting weights control which classifiers matter in the final decision.

**Optional math:** In the binary version, labels are $`\{-1,+1\}`$. If classifier $`m`$ has weighted error $`0<\epsilon_m<1/2`$, its voting weight is $`\frac12\log[(1-\epsilon_m)/\epsilon_m]`$. Here $`\log`$ means logarithm. A lower error gives a larger positive vote. A zero-error classifier needs a separate stopping rule or limiting treatment. These steps reduce exponential loss, a cost that rises sharply for confidently wrong predictions. Multiclass SAMME uses a different weight formula and requirement for a useful weak learner. Averaging binary formulas across classes is not equivalent.

**Inputs/outputs and typical data types:** Labeled numerical input lists, often table rows or manually designed image measurements. The base learner must support weighted examples. Outputs are combined scores and classes. A margin, which measures how strongly the combined score supports a class, is not automatically a trustworthy probability.

**Strengths and limitations:** AdaBoost can combine weak classifiers into a flexible model and focus learning on difficult cases. However, wrong labels and unusual cases may keep receiving more influence. Training rounds depend on earlier rounds, so they run in sequence. More rounds or deeper base trees can hurt performance. Decision stumps, trees with one split, select inputs as they are fitted. The whole boosted collection may still be harder to explain than one tree.

**Computational complexity / scalability notes:** Every round fits another learner and updates example weights. More rounds increase both training work and the number of learners evaluated for each prediction.

**Technical detail (optional):** Let $`n`$ count examples, $`d`$ count features, $`F(n,d)`$ be one base fit's cost, and $`M`$ count rounds. Training costs $`O(M[F(n,d)+n])`$. After preprocessing, reusable sorted inputs can give roughly $`O(nd)`$ stump fitting per round. Evaluating $`M`$ trees of depth $`h`$ costs $`O(Mh)`$ per prediction.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Viola and Jones's [2001 boosted-cascade face detector, documented by MERL](https://www.merl.com/publications/TR2004-043), learns from small image regions labeled face or non-face. It uses rectangle-based brightness features. An integral image, a table of cumulative brightness totals, makes these features quick to calculate. AdaBoost chooses useful weak tests. A cascade then applies stages of tests, rejecting easy background regions before using more costly later stages. The outputs are candidate boxes around faces.

The authors report detection rates comparable to leading earlier systems and real-time operation. This is face **detection**, not identifying a person. Boosting selects inexpensive useful features that a single fixed template would lack. That technical comparison says nothing about a buyer's intent. The report establishes no particular commercial deployment or measured operating benefit. Speed depends on the complete cascade and implementation, not AdaBoost alone.

**Notable vendor implementations/libraries:** scikit-learn has `AdaBoostClassifier`; Weka offers AdaBoost variants; OpenCV has boosted-cascade tools. OpenCV supports different boosting types, not all of which are original AdaBoost.

### 1.2.7 Gradient-boosted decision trees

**In plain English:** Gradient boosting adds small trees that correct the model's current mistakes. Use it to learn curved patterns and combinations of inputs in labeled tables.

**Name:** Gradient-boosted decision trees, or GBDT. These are gradient boosting machines whose added models are trees.

**Category & sub-category:** Supervised learning; ensembles built by repeatedly adding a model that reduces the current prediction loss.

**Originating paper/vendor/year:** Jerome Friedman's [2001 "Greedy Function Approximation: A Gradient Boosting Machine"](https://doi.org/10.1214/aos/1013203451) is the central reference. Boosting has earlier roots; vendors did not originate the general method by implementing it.

**Core mechanism:** Start with the same prediction for every case. Measure prediction mistakes using a loss function, a rule assigning a cost to mistakes. For each example, work out which small change to its prediction would lower that cost. Fit a tree to these requested changes, then add a reduced version of its output to the running prediction. The reduction factor is the learning rate. Leaf values may also be adjusted for the chosen loss.

**Technical detail (optional):** The requested changes are negative derivatives of the loss with respect to predictions. A derivative describes how a small prediction change affects loss. For squared loss, the targets are residuals: known answers minus current predictions. For logistic loss, they are not simply class labels. Training on subsets, using shallow trees, limiting leaves, and stopping early are regularization controls against over-reliance on training examples. Ranking versions change the update signal and should be named explicitly.

**Inputs/outputs and typical data types:** Tables paired with number answers, classes, or judgments about which items are more relevant. Outputs may be numerical predictions, class scores, converted probabilities, or ranking scores. A classification score may represent log odds: a logarithm of one outcome's chance relative to its alternative's chance. Tree leaf identifiers can also become categorical inputs to another model.

**Strengths and limitations:** GBDT learns joint effects and curves while using a loss suited to the task. Unlike bagging's independent fits, it can directly correct systematic underfitting. However, trees are fitted in sequence, and depth, learning rate, and round count matter greatly. Test-answer leakage and excessive tuning remain risks. Explaining why a score changed is not proof of cause and effect.

**Computational complexity / scalability notes:** Each added tree requires new correction targets and another tree fit. Sorting exact candidate thresholds differs from grouping values into bins and counting within them.

**Technical detail (optional):** With $`M`$ trees and $`n`$ examples, gradient calculation for a simple single-number loss is typically $`O(Mn)`$, plus $`M`$ tree fits. At depth $`h`$, prediction costs $`O(Mh)`$. Ranking losses can add costs based on documents per query or document pairs. Exact and histogram-based tree fits have different costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** He and colleagues' [2014 Facebook advertising report](https://quinonero.net/Publications/predicting-clicks-facebook.pdf) uses logged ad displays, called impressions, and click/no-click outcomes from one week in Q4 2013. All comparisons share the same offline training/testing split. Inputs first pass through boosted trees. Their leaf identifiers become sparse inputs to logistic regression, which estimates click probability. This is a tree-plus-logistic hybrid, not a standalone tree ensemble.

The chance estimate supports ad evaluation; the numerical comparisons check predictions offline. Cross-entropy penalizes assigning low chances to actual outcomes. Table 1 rescales normalized cross-entropy to a tree-only reference of **100%**: logistic regression alone is **99.43%**, and trees plus logistic regression is **96.58%**. Lower is better. These percentages are relative error scores, not accuracy, click-rate gains, or revenue gains. The table anchors the comparison, not a loosely paraphrased percentage elsewhere in the prose. Trees learn joint-input features that plain logistic regression would lack. The authors evaluate prediction, not profit or revenue; no production revenue benefit is reported.

**Notable vendor implementations/libraries:** R `gbm`, scikit-learn gradient-boosting estimators, and Spark ML GBT provide tree boosting. XGBoost, LightGBM, and CatBoost have their own entries below because their methods and systems differ.

### 1.2.8 XGBoost

**In plain English:** XGBoost builds corrective trees while controlling their size and using efficient split searches. It is useful for strong predictions on large numerical or sparse tables.

**Name:** XGBoost; this entry describes its boosted-tree version.

**Category & sub-category:** Supervised learning; gradient-boosted trees with regularization and methods for efficient training on large data.

**Originating paper/vendor/year:** Tianqi Chen and Carlos Guestrin presented [2016, "XGBoost: A Scalable Tree Boosting System," at KDD](https://arxiv.org/html/1603.02754v3). The inspected arXiv version is v3, dated 2016-06-10. XGBoost is open source. Later cloud services implement or host it; they did not originate the paper's method.

**Core mechanism:** At each boosting round, calculate how prediction errors would change under small corrections. Also calculate how quickly that rate of change itself changes. These are first- and second-order derivatives. Add this information within proposed groups to score tree splits. Penalties on leaf count and leaf output sizes, smaller updates, and training subsets control over-reliance on the training examples.

The paper also learns default directions for missing or sparse entries. A weighted quantile sketch summarizes ranked values and weights to propose approximate splits. Cache-aware organization reuses data close to the processor; out-of-core processing works with data stored beyond memory. These are more than simply running trees on several processors.

**Optional math:** With a squared leaf-output penalty, a representative leaf weight is $`-\sum_i g_i/(\sum_i h_i+\lambda)`$. Here $`i`$ indexes examples in the leaf, $`g_i`$ is the loss's first derivative, $`h_i`$ its second derivative, and $`\lambda`$ the penalty strength. The sums combine the examples' suggested changes and curvature. The result sets that leaf's corrective output.

**Inputs/outputs and typical data types:** Dense or sparse tables with numerical answers, class labels, or ranking targets. The chosen loss determines whether tree totals are used directly or converted, for example into probabilities. The package has other boosters. Its linear booster is not the tree model described here.

**Strengths and limitations:** XGBoost supports varied losses, missing and sparse inputs, and large-data implementations. An absent sparse entry does not always mean the same thing as a numerical zero; check the interface. Flexible trees can still learn leaked answers, overfit validation data, or extend patterns poorly beyond training data. Check whether probabilities match observed frequencies and whether future data change. Category support added in later releases is not a feature to credit to the 2016 paper.

**Computational complexity / scalability notes:** Skipping absent entries can save work on sparse tables. More trees and more levels still require more scans. Reading from disk or exchanging information between machines can take longer than the calculations.

**Technical detail (optional):** Let $`z=\operatorname{nnz}(X)`$ count nonzero entries in feature table $`X`$. A simplified sparse scan for $`M`$ trees of depth $`h`$ costs $`O(Mhz)`$, plus sorting or sketch construction and bookkeeping. Histogram versions also depend on bin count and processed nodes. Fully traversing the ensemble costs $`O(Mh)`$ per prediction.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper's Higgs-1M classification comparison](https://arxiv.org/html/1603.02754v3) uses HIGGS particle events generated by Monte Carlo simulation, which uses random sampling. These are not observations from a live deployed detector system. Inputs describe particle motion and derived physical quantities; labels mark signal or background. Trees combine the inputs into a score, choose a signal-like class, and check class separation on held-out data.

Section 6.3 reports classification quality comparable to scikit-learn's exact tree boosting, with substantially shorter training time on the authors' CPU setup. Tree interactions suit nonlinear relationships better than a simple linear boundary might. This is not evidence that XGBoost discovered the Higgs boson. The exact-greedy comparison uses a one-million-example subset, not a universal test of library speeds. Benefits in scientific operations are unreported.

**Notable vendor implementations/libraries:** The [XGBoost project](https://xgboost.readthedocs.io/en/stable/) provides language interfaces and tools for distributed use. Managed services also wrap it. Availability on a cloud platform does not prove that the provider uses it inside its own products.

### 1.2.9 LightGBM

**In plain English:** LightGBM speeds up tree boosting by grouping input values and reducing some repeated work. It is designed for large tables, including tables with many mostly empty columns.

**Name:** LightGBM.

**Category & sub-category:** Supervised learning; gradient-boosted trees using bins of values and techniques that reduce data and feature processing.

**Originating paper/vendor/year:** Guolin Ke and collaborators published [2017, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree), in a Microsoft-led project. The [original paper](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) defines the techniques used in the reported experiments.

**Core mechanism:** Put nearby input values into bins. For each bin, combine information about how the predictions should change. These summaries form histograms used to choose tree splits. Leaf-wise growth expands a promising leaf rather than necessarily expanding every node at the same level.

Gradient-based one-side sampling (GOSS) keeps cases with large gradients, meaning large rates of change in loss. It samples some small-gradient cases and increases their weights to compensate. It does not discard every easy example. Exclusive feature bundling (EFB) combines inputs that are rarely nonzero together, reducing the columns to process. These are separate techniques. Current settings do not necessarily enable the paper's exact GOSS-plus-EFB combination.

**Inputs/outputs and typical data types:** Large dense or sparse tables with numerical inputs, encoded inputs, or correctly supplied categories. Tasks include number prediction, classification, and ordering items by relevance. Outputs are added tree scores or converted probabilities.

**Strengths and limitations:** Bins and compact inputs can make large datasets practical to process. Leaf-wise growth learns joint effects efficiently, but leaves with very few cases can overfit. GOSS saves calculations at the cost of more variation in estimated updates. Bundling works best when the inputs are rarely active together. Category IDs must be treated as names, not invented numerical sizes. Good results still require the right loss, careful tuning, and time-valid data splits.

**Computational complexity / scalability notes:** Training builds and scans histograms at each grown node. Reusing a parent's and sibling's histogram can avoid rebuilding another from scratch. More bins, inputs, nodes, and boosting rounds add work.

**Technical detail (optional):** At node $`u`$ with $`n_u`$ cases, $`q`$ effective features, and $`b`$ bins, a dense histogram costs roughly $`O(n_u q)`$ to build and $`O(qb)`$ to scan. Sum these costs across nodes and rounds; total training is not universally $`O(n\log n)`$ for $`n`$ examples. Storage includes binned data, histograms, and leaves. Distributed training also exchanges histograms between machines.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [paper's Microsoft LETOR workload](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) pairs query-document inputs with relevance labels. Trees turn the inputs into relevance scores and order the search results. Evaluation uses NDCG, normalized discounted cumulative gain. This score rewards putting relevant results near the top; it is not a percent-correct measure.

Table 2 reports **0.31 seconds per training iteration** for LightGBM versus **0.49 seconds** for the histogram baseline without GOSS/EFB. The authors use a server with two E5-2670-v3 processors, 256 GB memory, and **16 training threads**. These are per-round times for this benchmark, not total retraining times or a general speedup guarantee. The paper reports similar held-out ranking quality. Histograms and reduced gradient work explain the saving relative to more expensive split scans. This does not establish live Bing use, higher search revenue, or a measured production benefit.

**Notable vendor implementations/libraries:** The [Microsoft LightGBM project](https://lightgbm.readthedocs.io/) provides Python/R interfaces, a command-line learner, and distributed integrations. Support for CPUs or graphics processors (GPUs) depends on the build and learner selected.

### 1.2.10 CatBoost

**In plain English:** CatBoost learns from tables with categories while trying not to let each training row reveal its own answer. It also uses a controlled order of learning to reduce this kind of prediction error.

**Name:** CatBoost.

**Category & sub-category:** Supervised learning; gradient boosting with ordered category summaries and a separate ordered fitting method.

**Originating paper/vendor/year:** CatBoost was developed at Yandex. Liudmila Prokhorenkova and colleagues' [2018 "CatBoost: Unbiased Boosting with Categorical Features"](https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) is the principal algorithm paper. Later releases and implementations should be distinguished from that publication.

**Core mechanism:** A category can be summarized using answers from other training rows in that category. Using a row's own answer in its summary would leak the answer into its input. CatBoost instead places rows in a permutation, or shuffled order. A row's training summary uses suitable earlier rows and a starting estimate called a prior.

Ordered boosting is a separate step. It calculates corrections using models that have not already fitted the current row's label. This reduces prediction shift: training-time predictions behaving differently because the model already saw the answer. Avoiding leaks in category summaries alone is not ordered boosting. Symmetric, also called oblivious, trees ask the same split question at every node on a level. This gives a compact, regular structure. Later settings also offer plain boosting and other ways of growing trees.

**Inputs/outputs and typical data types:** Labeled tables combining numbers and categories, including columns with many different category values. Depending on the task, outputs are scores, probabilities, numerical predictions, or rankings. Combining categories can capture joint effects, but those combinations must also avoid revealing labels.

**Strengths and limitations:** CatBoost reduces manual category preparation and one source of mismatch between training and later predictions. Symmetric trees are efficient and restrict model shape, but may need more trees for irregular boundaries. Rare categories, new category names, and changes in what inputs mean remain risks. Ordered summaries do not make it safe to mix future records into past training data when testing forecasts.

**Computational complexity / scalability notes:** A symmetric tree gains another full level of leaves each time depth increases. Memory can therefore grow much faster than the number of questions asked for one prediction. Category-summary tables need additional storage.

**Technical detail (optional):** A depth-$`h`$ symmetric tree has $`2^h`$ leaves and $`h`$ tests per prediction path. Leaf storage for $`M`$ single-output trees is $`O(M2^h)`$, excluding category tables. Multiple outputs add a further factor. Prediction traversal remains $`O(Mh)`$. Training also depends on candidate inputs, permutations, category combinations, and histogram bins; no single runtime ignores these dimensions.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [2018 paper's Amazon employee-access task](https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) predicts recorded access outcomes from employee and resource categories. It uses four-fifths of the data for training/tuning and one-fifth for testing. Table 2 reports **0.139 log loss** and **0.044 zero-one loss** for CatBoost's Ordered mode. Log loss checks assigned probabilities and heavily penalizes confident mistakes. Zero-one loss is the fraction of wrong class choices. Lower is better for both; neither number proves that future probabilities will be reliable.

The compared learners also receive ordered target-statistic preprocessing. This shared use of leak-resistant label summaries matters when reading the differences. Category inputs become summaries, then a tree score and an access prediction. The model is an offline classifier, not a policy deciding who should receive authorization. Handling category combinations avoids a potentially enormous one-hot table with a separate yes/no column per category. The paper establishes no Amazon production use, lower access-review costs, or improved security.

**Notable vendor implementations/libraries:** The [CatBoost project](https://catboost.ai/docs/) offers Python/R interfaces, command-line tools, and tools to serve or export trained models. Yandex's authorship does not establish deployment at another organization.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| CART | Tables of mixed measurements | Follow combined threshold questions step by step | Small data changes or excessive growth can hurt | Wisconsin breast-cancer pruning teaching test |
| C4.5 | Numerical and category inputs with class labels | Learns mixed-input trees and if-then rules | Local choices may miss a better tree; rules take work | RuleQuest forest-cover research comparison |
| Bagging | Data for a learner that changes sharply between samples | Combining separate fits steadies predictions | Models can still share systematic mistakes | Historical Boston median-home-value test |
| Random forest | Tables with curves, joint effects, or many inputs | Different trees share fewer mistakes; omitted-case checks | Memory use, weak trend extension, misleading importance | Lymphoma gene-activity research case |
| Extremely randomized trees | Numerical or encoded tables | Random thresholds reduce split-search work | Random cuts may miss a precise useful threshold | Research test of letter recognition |
| AdaBoost | Weighted rows or designed image measurements | Later classifiers focus on earlier mistakes | Wrong labels can gain too much influence | Viola-Jones research face detection, not identification |
| Gradient-boosted decision trees | Labeled tables with important joint effects | Adds trees to reduce the chosen error cost | Rounds are sequential; settings matter | Facebook trees plus logistic click-probability model |
| XGBoost | Dense or sparse tables for numbers, classes, or rankings | Controlled tree growth and efficient large-data training | Still vulnerable to leaked answers and overfitting | Simulated HIGGS event classification |
| LightGBM | Large tables grouped into value bins | Histograms, sampled updates, and bundled inputs save work | Tiny leaves can overfit; settings change behavior | Microsoft LETOR offline search-ranking test |
| CatBoost | Tables with combinations of categories | Reduces answer leakage in summaries and fitting | Category tables and deeper trees can use much memory | Research test on Amazon employee-access data |

## 1.3 Neighbors and kernels

These methods rely on a useful way to compare examples. A neighbor classifier reuses nearby labeled cases. A **kernel** is a comparison rule that can act as if the inputs had been transformed into a richer set of features, without constructing all those features. Different kernels define different kinds of similarity.

The comparison must fit the task. Ordinary straight-line, or Euclidean, distance between arbitrary category ID numbers usually has no meaning. Kernel methods also require more than any convenient similarity score. A valid positive-semidefinite kernel obeys mathematical rules that make its comparisons consistent with inner products: multiply matching transformed features and add them. A valid rule can still be the wrong comparison for the problem.

### 1.3.1 k-nearest neighbors

**In plain English:** kNN finds the most similar labeled examples and combines their answers. Use it when nearby cases are a sensible guide to a new case.

**Name:** k-nearest neighbors, or kNN, for classification and regression.

**Category & sub-category:** Supervised learning; predictions based directly on nearby stored examples.

**Originating paper/vendor/year:** Nearest-neighbor rules predate modern machine learning. Cover and Hart's [1967 "Nearest Neighbor Pattern Classification"](https://doi.org/10.1109/TIT.1967.1053964) provides a key statistical analysis. It is not a claim that they invented every method based on nearby examples.

**Core mechanism:** Store the training inputs and their labels. For a new input, calculate distances to stored inputs and find the $`k`$ closest, where $`k`$ is the chosen neighbor count. For classification, let the neighbors vote. For regression, average their numerical answers. Nearby neighbors may receive more weight. Using more neighbors usually steadies predictions but can blur useful local differences. Distance choice, feature scales, input selection, and tie rules are essential parts of the method.

**Inputs/outputs and typical data types:** Numerical input lists with labels, or other objects with a suitable distance rule. Outputs include the chosen neighbors and a class or number prediction. The fraction of neighbors voting for a class is a local estimate, not a guaranteed reliable probability. Sparse text usually needs a comparison and scaling suited to text, rather than ordinary straight-line distance between raw word counts.

**Strengths and limitations:** kNN is simple and can follow irregular class boundaries without one global equation. Adding another stored example is cheap. However, searching for each prediction can be costly, and storing more examples needs more memory. Unhelpful inputs distort distance. With many dimensions, nearest and farthest cases can become less meaningfully different. Duplicate or nearly duplicate cases on both sides of a train/test split can make results look misleadingly good. A similar past case may be a poor guide when future conditions change.

**Computational complexity / scalability notes:** Simple kNN does little fitting beyond storing examples. Instead, it does most work when a prediction is requested: compare the new case with stored cases, then choose the nearest ones.

**Technical detail (optional):** With $`n`$ examples and $`d`$ features, storage is $`O(nd)`$. One brute-force distance scan costs $`O(nd)`$. Selecting $`k`$ neighbors adds expected $`O(n)`$ work with a suitable selection algorithm, or $`O(n\log k)`$ with a heap, an organized list of current candidates. Spatial indexes group nearby cases and can help when few dimensions matter. They do not guarantee logarithmic query work with many dimensions. Approximate retrieval may choose different neighbors and therefore changes the prediction procedure.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 `Smarket` lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html) predicts 2005 market direction from the first two previous daily returns. Only pre-2005 cases can be neighbors. With $`k=1`$, one neighbor, it gets **126/252** days correct. With $`k=3`$, three neighbors, it gets **135/252**, or **53.57%**, correct. This remains below always predicting "Up" for that test period.

The method finds similar historical return pairs, collects their direction votes, and chooses a test-day direction. This local rule is an alternative to a single linear boundary, but its flexibility does not produce a demonstrated advantage here. A simple method or a plausible analogy with the past does not establish profitable trading. No financial benefit in production is reported.

**Notable vendor implementations/libraries:** scikit-learn, R `class::knn`, and MATLAB provide nearest-neighbor learners. FAISS and other vector indexes find similar items; they are not complete supervised kNN predictors by themselves.

### 1.3.2 Support-vector classification

**In plain English:** SVC finds a class boundary with a wide buffer around it while allowing some mistakes. A kernel lets that boundary bend to follow useful patterns in the inputs.

**Name:** Support-vector classification, or SVC; commonly called a support-vector machine classifier.

**Category & sub-category:** Supervised learning; classification that balances a wide separation margin with errors, using linear scores or kernels.

**Originating paper/vendor/year:** Corinna Cortes and Vladimir Vapnik's [1995 "Support-Vector Networks"](https://doi.org/10.1007/BF00994018) is the key soft-margin reference. "Soft" means some margin violations are allowed. Earlier work developed wide-margin classification and kernels.

**Core mechanism:** Seek a boundary that separates classes with a wide **margin**, the buffer around the boundary. Allow some cases inside that buffer or on the wrong side, but charge a cost for them. Balancing that cost with the margin is a form of regularization: it discourages a rule that follows training examples too closely. A kernel computes comparisons as if inputs had been transformed, allowing curved boundaries without listing every transformed feature. Cases with nonzero contributions to the fitted boundary are called **support vectors**.

**Optional math:** One binary objective is $`\frac12\|w\|^2+C\sum_i\max(0,1-y_i f(x_i))`$, where $`f(x)=w^\top\phi(x)+b`$. Here $`i`$ indexes examples, $`x_i`$ lists inputs, $`y_i`$ is a signed class label, $`w`$ lists weights, and $`b`$ is the intercept. The function $`\phi`$ transforms inputs; the transpose product is a weighted sum. The squared norm sums squared weights. The maximum term, called hinge loss, charges for insufficiently supported or wrong predictions. The setting $`C`$ controls that charge. A positive-semidefinite kernel gives valid inner products between the transformed inputs without constructing them. The sample-based, or dual, prediction uses the support vectors.

For multiple classes, software may fit each class against each other class or each class against all others. These one-versus-one and one-versus-rest combinations are not the same as optimizing one joint multiclass objective.

**Inputs/outputs and typical data types:** Labeled numerical input lists, or structured objects with a suitable kernel comparison. Outputs are decision scores and class labels. A raw hinge-loss score is not a probability. Producing probabilities needs an added calibration procedure, which must itself be checked against observed outcomes.

**Strengths and limitations:** SVC can work well when inputs are numerous but examples are relatively modest in number. Kernels add flexibility without explicitly storing huge transformed feature lists. Input scaling, the error-cost setting, and kernel bandwidth matter greatly. Bandwidth controls how far similarity extends around a case. Many examples make training costly; many support vectors make later predictions costly. A mathematically valid kernel can still compare cases in an unhelpful way.

**Computational complexity / scalability notes:** Comparing every training example with every other example grows quickly: twice as many cases means about four times as many comparisons. Linear-only solvers can avoid this full comparison table.

**Technical detail (optional):** For $`n`$ examples and kernel cost $`c_K`$ per pair, building the Gram matrix, the table of pairwise kernel values, costs $`O(n^2c_K)`$ time and $`O(n^2)`$ memory. Optimization also depends on the solver, currently used constraints, cache, and requested precision. Cubic dense calculations may occur, but are not a universal total-training bound. With $`s`$ support vectors, binary prediction costs $`O(s c_K)`$. A specialized linear solver can instead use roughly $`O(nd)`$ per pass for $`d`$ features.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn handwritten-digit example](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) turns each 8-by-8 grayscale image into a list of 64 pixel features. It fits an RBF, or radial-basis-function, SVC with `gamma=0.001`. This kernel makes nearby input lists more similar than distant ones. The non-shuffled half-training/half-test split reports rounded accuracy **0.97 on 899 test images**. Accuracy is the fraction of correct labels; the reported value is rounded.

Pixels become kernel decision scores, which choose a digit for comparison with the recorded answer. The displayed first test prediction is 8. A curved similarity boundary can suit varied handwriting better than one linear score might. This demonstration does not prove superiority to a tuned alternative. It is not a deployed postal or banking optical character recognition (OCR) system and reports no production processing benefit.

**Notable vendor implementations/libraries:** LIBSVM, scikit-learn `SVC`, and R `e1071` interfaces provide SVC. LIBLINEAR and scikit-learn `LinearSVC` provide related linear versions. Their solvers and ways of combining classes differ.

### 1.3.3 Support-vector regression

**In plain English:** SVR predicts a number while ignoring errors inside a chosen tolerance band. A kernel lets the prediction follow a curve without building a large explicit feature list.

**Name:** Support-vector regression, or SVR.

**Category & sub-category:** Supervised learning; regularized numerical prediction with a band of errors that receive no penalty.

**Originating paper/vendor/year:** Drucker, Burges, Kaufman, Smola, and Vapnik presented ["Support Vector Regression Machines"](https://papers.nips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf) at NIPS 1996, published in the volume 9 proceedings. It extends support-vector methods from class choices to number predictions.

**Core mechanism:** Choose an acceptable error tolerance around the predicted curve. Errors inside this tube cost nothing; larger errors incur a cost. Balance that cost against a penalty on the curve's complexity. This regularization discourages following every training detail. In a kernel version, training cases can make positive, negative, or zero contributions. Cases with nonzero contributions support the prediction.

**Optional math:** The insensitive loss is $`\max(0,|y-f(x)|-\epsilon)`$. Here $`x`$ lists inputs, $`y`$ is the true number, $`f(x)`$ is the prediction, and $`\epsilon`$ is the tolerance. The absolute-value term measures mistake size; subtract the tolerance and keep only positive excess. The setting $`C`$ controls the error penalty relative to complexity. Kernel fitting uses positive and negative sample-based, or dual, coefficients. Squared insensitive loss and $`\nu`$-SVR, a version with a different control parameter, are related but distinct methods.

The tube width uses the answer's units. It is a chosen tolerance, **not** an interval shown to contain future answers at a stated rate.

**Inputs/outputs and typical data types:** Numerical inputs, or objects with a suitable kernel, paired with numerical answers. Outputs are number predictions and, in some interfaces, information about support vectors. Changing the scale of either inputs or answers changes how penalties act. Record such changes and convert predictions back correctly.

**Strengths and limitations:** Kernels let SVR learn curves without creating a huge transformed input table. Very large mistakes may dominate less than with squared-error fitting. SVR does not automatically provide uncertainty, handle error spread that changes across cases, or account for follow-up ending before an event is known. Choosing settings can be expensive, and dense kernel tables limit sample size. A poor bandwidth, the range over which cases count as similar, can make predictions almost constant or too narrowly local.

**Computational complexity / scalability notes:** A complete pairwise comparison table grows with the square of the number of examples. Prediction uses retained support vectors, but many cases may remain.

**Technical detail (optional):** For $`n`$ examples and kernel comparison cost $`c_K`$, a full Gram matrix needs $`O(n^2)`$ memory and $`O(n^2c_K)`$ kernel work. Solver-dependent optimization adds work. Prediction costs $`O(s c_K)`$ for $`s`$ support vectors. Data and tolerance settings determine how many cases remain; a small retained fraction is not guaranteed.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** [Drucker and colleagues' original paper](https://papers.nips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf) compares SVR and bagging on historical Boston housing data. It uses **401 training, 80 validation, and 25 test cases**, repeating random partitions **100 times**. Mean squared prediction error is **7.2 for SVR versus 12.4 for bagging**, in squared dataset-response units. This averages squared mistakes; lower is better. SVR does better on 71 of the 100 trials.

Neighborhood inputs enter a kernel regression rule, which predicts median home value for comparison with test answers. The kernel's implicit polynomial representation handles powers and combinations of inputs without explicitly fitting a huge feature list to few examples. This is not the ISLR2 split or feature setup, so these errors must not be ranked against the earlier bagging numbers. Demographic variables and historical sampling make this a legacy benchmark, not a recommended valuation policy. Economic benefits in production are unreported.

**Notable vendor implementations/libraries:** LIBSVM, scikit-learn `SVR` and `NuSVR`, and R `e1071` provide SVR variants. Specialized linear-SVR solvers are available when only a linear prediction rule is needed.

### 1.3.4 Kernel ridge regression

**In plain English:** Kernel ridge predicts a number by combining comparisons with training examples. It uses a penalty to keep the resulting prediction rule from following the training data too closely.

**Name:** Kernel ridge regression, or KRR.

**Category & sub-category:** Supervised learning; squared-error regression with a kernel-defined penalty. Its formal setting is a reproducing-kernel Hilbert space: a family of functions with a kernel-defined way to measure their size.

**Originating paper/vendor/year:** Saunders, Gammerman, and Vovk's [1998 "Ridge Regression Learning Algorithm in Dual Variables"](https://eprints.soton.ac.uk/258942/) is an influential machine-learning account. Squared penalties on functions have a longer mathematical history. Library kernel interfaces did not originate that theory.

**Core mechanism:** Build a table comparing every training case with every other case using the chosen kernel. Fit a contribution weight for each case. For a new case, compare it with training cases, multiply comparisons by those weights, and add. Training balances squared prediction errors with a kernel-defined penalty on the whole function. This regularization controls how closely the rule follows the examples.

**Optional math:** Minimize squared errors plus $`\lambda\|f\|_{\mathcal H}^2`$. Here $`f`$ is the prediction function, $`\mathcal H`$ its kernel-defined function family, the norm measures its size in that family, and $`\lambda`$ controls the penalty. The representer theorem shows that $`f(x)=\sum_i\alpha_i K(x_i,x)`$. Here $`x_i`$ is training case $`i`$, $`x`$ the new case, $`K`$ the kernel, and $`\alpha_i`$ its learned contribution weight. Using a sum rather than an average of squared errors gives $`(K+\lambda I)\alpha=y`$. In that equation, $`K`$ is the pairwise kernel table, $`I`$ the identity matrix, $`\alpha`$ lists weights, and $`y`$ lists answers. Handle centering or an intercept separately. The function penalty is not generally $`\lambda\|\alpha\|^2`$, a simple sum of squared contribution weights; replacing it changes the model.

**Inputs/outputs and typical data types:** Labeled numerical inputs, molecules, strings, or other objects with a suitable positive-semidefinite kernel, a comparison rule meeting the required inner-product conditions. The output is a number. Unlike SVR, KRR usually retains contributions from all training examples. With fixed, matched kernel and noise settings, its predictions can equal a Gaussian process's updated mean predictions. KRR alone does not provide that model's uncertainty spread.

**Strengths and limitations:** KRR fits nonlinear patterns on modest datasets by solving a regularized set of equations. The kernel controls which kinds of smoothness it favors. Costs rise sharply as examples increase. Descriptors, the numerical summaries of objects, determine which similarities the model can see. Predictions outside observed data depend on the kernel. Smaller or random-feature approximations save work but change the exact fitted rule.

**Computational complexity / scalability notes:** Exact fitting stores pairwise comparisons and solves a system whose dense work grows cubically with examples. Doubling examples can make the solve take about eight times as much work. Later predictions usually compare against all training cases.

**Technical detail (optional):** With $`n`$ examples and kernel comparison cost $`c_K`$, exact dense training costs $`O(n^2c_K+n^3)`$ time and $`O(n^2)`$ Gram-matrix memory. Prediction costs $`O(nc_K)`$. Using $`r\ll n`$ basis functions, meaning far fewer building blocks than examples, can reduce the solve to approximately $`O(nr^2+r^3)`$, plus feature construction. The number $`r`$ is the approximation rank, not the original input count.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Rupp and colleagues' [molecular-energy preprint v1, dated 2011-09-12](https://arxiv.org/html/1109.2618v1), came before the [2012-01-31 journal publication](https://doi.org/10.1103/PhysRevLett.108.058301). It uses 7,165 small organic molecules from GDB. Reference atomization energies, energies associated with separating molecules into atoms, come from PBE0 density-functional theory, a quantum-chemistry calculation method. Nuclear charges and atomic positions form Coulomb-matrix-based descriptors, numerical summaries of the atoms and their arrangement. Gaussian-kernel regression maps these summaries to predicted atomization energy.

The inspected preprint reports **9.9 kcal/mol mean absolute error**. This averages absolute prediction mistakes, in kilocalories per mole; lower is better. It uses nested five-fold selection/evaluation: inner splits choose settings, while outer splits check them. This is not a production molecule-screening campaign. The reference calculations approximate physics; they are not exact experimental truth. A learned substitute for energy calculation can avoid solving the electronic-structure problem for every query. That motivation establishes neither validated reaction-energy predictions nor drug-discovery success nor measured production cost savings.

**Notable vendor implementations/libraries:** scikit-learn `KernelRidge`, kernel toolboxes, and scientific packages implement regularized kernel solves. A general matrix-equation solver still needs code to build the kernel and prepare inputs.

### 1.3.5 Gaussian processes: regression and classification

**In plain English:** A Gaussian process starts with assumptions about plausible prediction curves and updates them after seeing examples. It returns predictions and model-based uncertainty, but that uncertainty is only as reliable as its assumptions.

**Name:** Gaussian processes, distinguishing Gaussian-process regression (GPR) from Gaussian-process classification (GPC).

**Category & sub-category:** Supervised learning; Bayesian prediction over functions. "Bayesian" means updating starting assumptions with evidence. "Nonparametric" here means not choosing one fixed-length list of curve coefficients independent of the data size.

**Originating paper/vendor/year:** Gaussian processes and kriging, a related spatial-prediction method, have several historical origins. Rasmussen and Williams's [2006 *Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/chapters/) is the main modern reference used here. It does not imply that GPs were invented in 2006 or by one vendor.

**Core mechanism:** Choose a starting average curve and a covariance kernel. The kernel describes how values at different inputs are expected to vary together. Similar inputs may be expected to have similar outputs; another kernel may favor repeating seasons. A Gaussian process uses these rules to assign chances to possible function values. Training updates those chances using the observed answers. "Gaussian" describes the joint bell-shaped distributions assigned to any finite set of underlying function values.

**Regression with Gaussian observation noise** assumes measurements differ from the underlying function through bell-shaped noise. Matrix equations then give an updated average prediction and uncertainty exactly under the model. **Classification** instead converts underlying scores into chances for binary or multiple classes. That update generally no longer has an exact Gaussian form and needs approximation or sampling. Putting category labels directly into the regression equations does not give the usual GPC model.

**Technical detail (optional):** GPC may use a Laplace approximation, which builds a local bell-shaped approximation; expectation propagation, which matches simpler approximations to parts of the model; variational inference, which fits a simpler probability description; or sampling, which draws possible values. These are alternative inference methods, not the analytic GPR update.

**Inputs/outputs and typical data types:** Numerical or structured inputs paired with measurements or class labels. GPR returns predicted averages and covariances, which describe uncertainty and how predictions vary together. GPC returns approximate class probabilities. Uncertainty about an underlying noise-free curve differs from uncertainty about a future noisy measurement. Kernel and noise settings may be fitted using marginal likelihood, which assesses the data while accounting for possible functions, or given their own prior assumptions.

**Strengths and limitations:** GPs let you state assumptions about smoothness, repeating patterns, or linked nearby locations. They include uncertainty in the model, rather than adding it afterward. Wrong assumptions can still produce misleading confidence. A stationary kernel applies the same relationship rule everywhere; that may not suit changing conditions. Assuming the same noise level for every observation may also fail. Location-dependent models and smaller approximations need extra choices. Exact GPs become expensive much sooner than ordinary linear regression.

**Computational complexity / scalability notes:** Exact dense fitting grows cubically with the number of examples: doubling them can make the main solve take about eight times the work. Storing their relationships grows quadratically. Trying different kernel settings repeats costly calculations.

**Technical detail (optional):** With $`n`$ examples and fixed settings, exact dense GPR requires **$`O(n^3)`$ time and $`O(n^2)`$ memory**. After factoring the matrix, one mean prediction costs $`O(nc_K)`$, where $`c_K`$ is one kernel comparison's cost. An exact variance adds roughly $`O(n^2)`$ work. Dense GPC repeats comparable matrix work while refining approximate inference. Inducing-variable methods use a smaller set of representative function values; they have different costs and are not exact GPs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [Mauna Loa CO2 example](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html), based on the GP book, models monthly atmospheric concentrations from **1958-2001**. Its input is the date. The kernel combines a smooth trend, locally repeating seasonal variation, irregular variation, and noise. The output is a CO2 concentration in ppm, parts per million, with updated model-based uncertainty.

The graph reproduces seasonal variation and extends the model beyond 2001. It does **not** report an independent future-period accuracy score. A repeating-pattern kernel can express seasons and uncertainty that a bare straight-line trend misses. But the curve and uncertainty band are not a validated climate forecast. They do not show that an observatory uses this model in daily work. Production forecasting results are unreported.

**Notable vendor implementations/libraries:** GPML, scikit-learn Gaussian-process estimators, GPyTorch, and GPflow provide GP tools. This entry uses classical kernels. Learning the input features with a neural network would require a separate description of that combined model.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| k-nearest neighbors | Few or moderate inputs with useful distances | Reuses local examples without one global equation | Slow queries; distances can lose meaning with many inputs | `Smarket` research test of neighbor votes |
| Support-vector classification | Modest example counts with rich numerical inputs | Wide-margin control and curved kernel boundaries | Large pairwise tables; scores are not probabilities | Research classification of 8-by-8 digit images |
| Support-vector regression | Numerical answers with curved input relationships | Tolerates small errors without a huge explicit feature list | Costly tuning; tolerance tube is not an uncertainty interval | Original historical Boston housing comparison |
| Kernel ridge regression | Modest datasets with useful comparison rules | Direct fit with a kernel-defined complexity penalty | Training and predictions depend strongly on example count | Research prediction of GDB molecular atomization energies |
| Gaussian processes | Small or modest scientific or location-based datasets | Predictions include assumption-based uncertainty | Exact fitting grows cubically; classification needs approximations | Mauna Loa CO2 fit and unvalidated future extension |

## 1.4 Probabilistic, structured, and survival models

Not every answer is a single independent category. A sequence model can label all words in a sentence together. A survival model can use follow-up time even when the event has not yet been observed. The models below break their prediction problems into different linked parts. Calling them "probabilistic" does not mean they learn in the same way or produce automatically trustworthy chances.

### 1.4.1 Naive Bayes: Gaussian, multinomial, and Bernoulli

**In plain English:** Naive Bayes checks how well each input fits each class, then combines those clues. Its simplifying assumptions make it fast, but can make its probability estimates overconfident.

**Name:** Naive Bayes classifiers, including Gaussian, multinomial, and Bernoulli variants.

**Category & sub-category:** Supervised learning; classification by using a simple probability model of inputs within each class.

**Originating paper/vendor/year:** No single person invented every naive-Bayes variant. M. E. Maron's [1961 "Automatic Indexing: An Experimental Inquiry"](https://doi.org/10.1145/321075.321084) is a landmark in probability-based text classification. Bayes's theorem, the rule for updating chances with evidence, predates these applications and modern libraries.

**Core mechanism:** Start with how common each class is, its prior chance. For each class, check how consistent the inputs are with that class's training examples. Combine these clues under simplifying assumptions and update the class chances using Bayes's rule. Choose the class with the largest updated, or posterior, chance.

**Gaussian NB** fits a separate bell-shaped distribution for each numerical input within each class. It ignores links between inputs once the class is known. **Multinomial NB** models repeated tokens, such as word occurrences, so repetition matters. **Bernoulli NB** instead checks yes/no presence and includes evidence from absence too. Repeating a word does not change a binarized presence indicator. Smoothing prevents unseen events from forcing a zero chance. Calculating with log scores avoids multiplying many tiny numbers until the computer rounds them to zero.

**Optional math:** Multinomial NB's count likelihood is proportional to $`\prod_j\theta_{jc}^{x_j}`$. Here $`c`$ identifies the class, $`j`$ a token type, $`\theta_{jc}`$ that token's probability within the class, and $`x_j`$ its count. The product multiplies token probabilities repeatedly according to their counts. With a fixed total token count, the counts themselves are not independent random variables: more occurrences of one token leave fewer for others.

**Inputs/outputs and typical data types:** Gaussian NB expects measurements; multinomial NB expects nonnegative counts or deliberately chosen count-like values. Bernoulli NB expects binary indicators. Outputs are labels and approximate class probabilities. TF-IDF, term frequency-inverse document frequency, weights words that are frequent in one document but less common across documents. It may work with multinomial software even though those values are not literally generated as token counts.

**Strengths and limitations:** Training is fast and can update summaries as examples arrive. These summaries, called sufficient statistics, retain what this model needs without retaining every row. Naive Bayes is a useful starting model for small datasets or sparse text. Related measurements can badly violate its within-class independence assumptions. It may still choose useful classes while assigning overly confident chances. Gaussian, multinomial, and Bernoulli describe different kinds of events; they are not interchangeable settings.

**Computational complexity / scalability notes:** Training mainly scans inputs to update class summaries. Prediction checks each input's contribution for each class. Sparse count storage can skip many zero entries.

**Technical detail (optional):** With $`n`$ examples, $`d`$ features, and $`C`$ classes, dense summary accumulation costs approximately $`O(nd)`$. Parameter storage is $`O(Cd)`$; one dense prediction costs $`O(Cd)`$. Sparse count versions do much of their scanning over nonzero entries. Vocabulary setup and smoothing still need storage proportional to the number of classes times vocabulary size.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [scikit-learn Gaussian NB example](https://scikit-learn.org/stable/modules/naive_bayes.html) uses Iris flower measurements and species labels. It holds out half the data with `random_state=0` and reports **four mislabeled flowers among 75 test examples**. Here "mislabeled" means the model predicted the wrong species. Sepal and petal measurements enter separate bell-shaped models for each species. Their combined evidence produces a species choice, checked against the recorded species.

Gaussian NB estimates fewer relationships between inputs than full QDA, making it a sensible small-sample starting point. Related flower dimensions still challenge its assumptions. This result does not validate multinomial or Bernoulli NB; those need their own count or binary tasks. The demonstration is not a deployed botanical system and reports no benefit in operational identification.

**Notable vendor implementations/libraries:** scikit-learn provides `GaussianNB`, `MultinomialNB`, and `BernoulliNB`. R has `e1071::naiveBayes`; Spark ML also offers naive-Bayes variants. Check which input models each implementation supports.

### 1.4.2 Bayesian networks

**In plain English:** A Bayesian network links variables in a diagram and describes how their chances depend on one another. It can predict a target even when only some related facts are known.

**Name:** Bayesian networks; this entry includes supervised Bayesian-network classifiers.

**Category & sub-category:** Supervised learning; arrow-linked probability models used to predict a labeled target. The wider family also includes expert-built models and learning patterns or graph structure without target labels.

**Originating paper/vendor/year:** Judea Pearl's 1980s work and 1988 *Probabilistic Reasoning in Intelligent Systems* established key foundations. Friedman, Geiger, and Goldszmidt's [1997 "Bayesian Network Classifiers"](https://doi.org/10.1023/A:1007465528199) specifically addresses supervised use. Current software providers did not invent the graphical approach.

**Core mechanism:** Draw a node for each variable, such as an observation or target class. Arrows specify which variables directly help set another variable's probabilities. Arrows cannot lead around a loop back to their starting point. This makes a directed acyclic graph. In supervised fitting, the target class is known; for a new case, the network calculates its chance from the available evidence.

For a fixed graph with fully observed discrete values, count outcomes for each combination of parent values. Combine counts with prior information to estimate conditional-probability tables. A parent is simply a node whose arrow enters the current node. Tree-augmented naive Bayes (TAN) learns a tree of extra links between inputs, while still accounting for the class. It relaxes naive Bayes's assumption that inputs are independent within a class. Choosing the graph, estimating its probabilities, and calculating an answer are separate jobs. An arrow does **not** prove cause and effect.

**Optional math:** The joint probability, meaning the chance of a complete set of variable values, is $`\prod_j P(X_j\mid\mathrm{Pa}(X_j))`$. Here $`j`$ indexes variables, $`X_j`$ is one variable, and $`\mathrm{Pa}(X_j)`$ lists its parents. Each factor gives that variable's chance when its parents' values are known. Multiply the factors to describe the complete case.

**Inputs/outputs and typical data types:** Discrete values or appropriately modeled measurements, possibly with some evidence missing. Supervised training also needs target labels. Outputs include chances for individual variables, chances after observing evidence, and most-probable states. Expected utility, an average benefit or cost for an action, needs a separately supplied decision model.

**Strengths and limitations:** Graphs make assumptions about linked variables visible and can combine measurements with expert knowledge. They can use partial evidence. A poor graph, too few examples for probability-table entries, or an unsuitable model of measurements can spoil predictions. Searching for a graph is hard. Calculating an exact answer can also be impractical even when its probability tables are easy to fit. The [bnlearn classifier documentation](https://www.bnlearn.com/examples/classifiers/) explicitly separates a graph useful for prediction from one justified as causal.

**Computational complexity / scalability notes:** Counting observed combinations can be fast for a fixed graph. But adding parents multiplies the combinations a table must store. Answering queries can require handling large groups of linked variables together.

**Technical detail (optional):** With $`n`$ examples and $`q`$ fully observed discrete variables, fixed-graph counting takes about $`O(nq)`$, plus table setup. With at most $`r`$ states and $`b`$ parents per node, storage can be $`O(qr^{b+1})`$. Exact inference grows exponentially with treewidth $`w`$, a measure of how many variables must be handled together. Its factors can have about $`r^{w+1}`$ entries. A quick data scan therefore does not guarantee a quick prediction.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Sourced application.** Microsoft's [Lumiere project, reported by Horvitz and colleagues in 1998](https://erichorvitz.com/lumiere.htm), estimates software users' goals and help needs. It uses their background, actions, and queries as evidence. The Bayesian user model updates chances for help topics or needs, then suggests possible assistance. The authors report that prototypes supplied the basis for components of the **Office Assistant in Office '97**. They do not say the entire research system shipped unchanged.

Linked variables and incomplete, uncertain evidence suit this approach better than a fixed keyword rule might. Importantly, the [paper](https://erichorvitz.com/ftp/lum.pdf) includes expert assessment and usability studies. This is a wider-family application, **not evidence of a wholly supervised parameter-learning recipe**. The supervised classifier described above is distinct from this historical hybrid of methods. Production productivity gains and support-cost savings are unreported.

**Notable vendor implementations/libraries:** pgmpy, R `bnlearn`, Bayes Server, and GeNIe/SMILE provide Bayesian-network tools. They support different variable types, learning methods, causal-analysis tools, and decision-model extensions.

### 1.4.3 Supervised hidden Markov models

**In plain English:** A supervised HMM learns which labeled states follow one another and which observations each state produces. It uses those patterns to label a new sequence, such as the words in a sentence.

**Name:** Supervised hidden Markov models, or supervised HMMs.

**Category & sub-category:** Supervised learning; sequence labeling from training examples whose states have been annotated. The model describes both state sequences and the observations they produce.

**Originating paper/vendor/year:** HMMs grew from 1960s work on probability models of linked states, including work by Baum and Petrie. Rabiner's [1989 tutorial](https://doi.org/10.1109/5.18626) is a key synthesis, not their invention. Brants's [2000 TnT tagger](https://aclanthology.org/A00-1031/) supplies the supervised example below.

**Core mechanism:** A state is the label behind an observation, such as a word's grammatical role. Learn how sequences start, how one state follows another, and which observations occur in each state. A first-order HMM uses only the previous state to predict the next state. Supervised training knows the state labels, so it can count transitions and observations or fit a measurement model for each state. Smoothing prevents rare or unseen combinations from being treated as impossible.

For a new sequence, Viterbi reuses the best partial paths to find the most probable **complete** label path. Forward-backward instead combines paths to find each position's state probabilities. The most likely whole path and the most likely state at each position answer different questions. If training states are **unobserved**, Baum-Welch/expectation-maximization (EM) estimates expected counts instead of counting known labels. That is [unsupervised probabilistic learning](04-unsupervised-classical.md), not this supervised recipe.

**Optional math:** The first-order transition is $`P(z_t\mid z_{t-1})`$ and the emission is $`P(x_t\mid z_t)`$. Here $`t`$ is sequence position, $`z_t`$ its state, $`z_{t-1}`$ the previous state, and $`x_t`$ the observation. A transition gives the next state's chance given the previous state. An emission gives the observation's chance given its state. A separate initial distribution gives starting-state chances.

**Inputs/outputs and typical data types:** Ordered words, sound measurements, or sensor readings, paired with state sequences for fully supervised training. Outputs include scores for observed sequences, decoded labels, and individual-position state chances. "Hidden" means the state is not directly part of the observed input at prediction time. Training annotations can still reveal it.

**Strengths and limitations:** HMMs explicitly model order and handle ambiguous observations. Dynamic programming reuses partial calculations rather than enumerating every possible sequence. However, short-memory transitions and restricted links between observations can miss distant or overlapping clues. Sparse counts need smoothing. Even the supplied "correct" state sequence may oversimplify genuinely ambiguous language or physical states.

**Computational complexity / scalability notes:** At each position, first-order inference checks pairs of possible states. Longer sequences add work roughly in proportion to length; more possible states grow the work much faster. Counting fully labeled training tokens is linear in the token count.

**Technical detail (optional):** With $`C`$ states and length $`T`$, dense first-order Viterbi or forward-backward costs $`O(TC^2)`$, plus emission calculations. Decoding typically stores $`O(TC)`$ backpointers, records of earlier choices used to recover the path. A second-order model uses two previous states and may require $`O(TC^3)`$ without pruning. Beam search keeps only promising partial paths, saving work but giving up exactness.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** Brants's [TnT paper, arXiv v1 dated 2000-03-13](https://arxiv.org/html/cs/0003055v1), trains a **second-order** HMM for part-of-speech tagging, such as assigning grammatical roles to words. Its trigram transitions use the previous two tags when predicting the next. Interpolation combines estimates from different amounts of context; suffix handling uses word endings to help with unseen words.

Annotated Penn Treebank words supply transition and emission estimates. The model then decodes tag sequences and checks each word's predicted tag. Reported overall accuracy is **96.7%**, averaged over ten experiments with disjoint **90% training/10% test** partitions and contiguous test segments. These segments preserve stretches of the original sequence. This is not a modern standard-split neural benchmark. Transitions help resolve ambiguities that independent word classification misses. The paper documents a research tagger, not a particular deployed writing assistant or measured productivity gain.

**Notable vendor implementations/libraries:** NLTK provides supervised HMM and TnT implementations. `hmmlearn` provides HMM inference and fitting, but [`hmmlearn.fit`](https://hmmlearn.readthedocs.io/en/stable/tutorial.html) commonly uses unsupervised EM. Using that library alone does not establish supervised training.

### 1.4.4 Conditional random fields

**In plain English:** A CRF chooses labels for a whole sequence by weighing input clues and neighboring label choices together. It is useful when choosing each word's label separately would create an inconsistent sequence.

**Name:** Conditional random fields, or CRFs. This entry uses a classical linear-chain CRF with feature rules chosen before training.

**Category & sub-category:** Supervised learning; sequence labeling that learns directly from inputs and compares complete label sequences, not just separate local choices.

**Originating paper/vendor/year:** John Lafferty, Andrew McCallum, and Fernando Pereira introduced the family in "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data," ICML 2001. The [original paper](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf) describes the model; CRFsuite is later software.

**Core mechanism:** Define input clues such as a word's ending, capitalization, and nearby words. Also define clues about adjacent labels, such as whether one tag often follows another. Learn weights for these rules using annotated sequences. For each possible complete labeling, add the weighted clues to get a score. Compare scores across whole sequences to assign chances and choose labels.

Training lowers the cost of assigning low probability to correct sequences. Penalties on absolute or squared weights, called $`L_1`$ or $`L_2`$ regularization, help prevent over-reliance on training examples. Forward-backward reuses partial calculations to combine scores across paths and compute expected rule counts. Viterbi finds the highest-scoring complete labeling. Unlike an HMM, a CRF need not model how the input words themselves were generated. Its rules may inspect a wide input context while label-to-label links remain local.

**Optional math:** The conditional probability of a complete label sequence is:


$$
P(y\mid x)=\frac{1}{Z(x)}
\exp\left(\sum_t\sum_r w_r f_r(y_{t-1},y_t,x,t)\right).
$$


Here $`x`$ is the observed input sequence and $`y`$ the complete label sequence. Position $`t`$ has labels $`y_{t-1}`$ and $`y_t`$ on either side of a transition. Rule $`f_r`$ checks those labels, the inputs, and the position; $`w_r`$ is its learned weight. Sum over positions $`t`$ and rules $`r`$. The exponential turns a score positive. The partition function $`Z(x)`$ sums these positive scores over possible label sequences, making their probabilities add to one. Training uses conditional negative log-likelihood, the cost of assigning low chances to known sequences. Normalizing whole sequences avoids the characteristic label bias of locally normalized transition models, which can favor states simply because they have fewer outgoing choices.

**Inputs/outputs and typical data types:** Annotated sequences of tokens, such as words, and named numerical or category features. Outputs include tags and each position's label probabilities. Entity labeling often uses BIO: beginning of an entity, inside one, or outside one. Rules can explicitly forbid invalid BIO transitions. An unconstrained CRF does not automatically enforce every such rule.

**Strengths and limitations:** CRFs can combine overlapping input clues while keeping adjacent labels linked. With fixed features in a linear chain, the usual objective is convex. A minimum among nearby weight choices is also a minimum across all choices. Replacing fixed features with a learned neural encoder changes that training problem and needs a separate model description. Designing good clues, limited label-link range, and sequence-calculation costs remain limitations. More general CRF graphs with loops may need approximations; the chain's speed bounds do not apply to them.

**Computational complexity / scalability notes:** Chain inference checks possible adjacent label pairs at each position. Longer sequences add work roughly in proportion to length; more label types increase work quadratically. Training repeats this across sequences while updating rule weights.

**Technical detail (optional):** With $`C`$ labels and sequence length $`T`$, chain inference costs $`O(TC^2)`$ plus scoring input features. A training pass adds these costs over sequences and calculates feature gradients, directions for changing weights. Total training depends on solver and iteration count. Dynamic-programming storage is typically $`O(TC)`$. Model storage depends on the retained feature-label and transition weights.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [sklearn-crfsuite CoNLL-2002 tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html) labels Spanish named entities, such as names spanning one or more words. It uses `esp.train` for fitting and `esp.testb` for testing. Inputs include word identity, suffixes, word shape, supplied part-of-speech tags, and neighboring words. These clues produce joint BIO labels, then candidate entities to compare with annotations.

Its initial L-BFGS setup, a repeated weight-update solver, uses both regularization coefficients at 0.1 and at most 100 iterations. It reports **0.7698023 weighted token-label F1 excluding `O`**. For each tag type, precision is the fraction of predicted tags that are right. Recall is the fraction of actual tags found. F1 balances these, and the weighted average accounts for how often each included tag occurs. `O`, the outside-entity tag, is excluded. This is **not entity-span F1**, which checks complete entity boundaries, so it cannot be ranked directly against CoNLL leaderboard scores. Joint transitions can improve on separate word decisions, but the tutorial shows no enterprise extraction deployment or production document-processing benefit.

**Notable vendor implementations/libraries:** CRFsuite, `python-crfsuite`, `sklearn-crfsuite`, CRF++, and MALLET provide CRF tools. Systems that learn inputs with a neural encoder before the CRF belong in the [neural supervised chapter](02-supervised-neural.md), not this fixed-feature entry.

### 1.4.5 Cox proportional-hazards regression

**In plain English:** Cox regression compares how inputs relate to an event's timing, even when follow-up ends before some events are seen. It does not treat an unfinished record as if the event happened at the last observation.

**Name:** Cox proportional-hazards regression, or the Cox PH model.

**Category & sub-category:** Supervised learning; time-to-event regression with incomplete follow-up, called censoring. It is semiparametric: it learns a fixed set of input weights but does not prescribe a fixed shape for the baseline event-rate curve.

**Originating paper/vendor/year:** D. R. Cox published [1972, "Regression Models and Life-Tables"](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x). Later work added calculation methods, checks of model errors, and extensions. Today's survival-analysis libraries implement these methods.

**Core mechanism:** Record how long each subject was followed and whether the event was actually observed. For example, a person may still be alive when a study ends. Their record shows survival up to that time, not a death at that time or survival forever afterward. This is right censoring.

The **hazard** is the event rate at a moment among subjects who have not yet had the event. Whenever an event occurs, Cox fitting compares that subject's inputs with those of subjects still at risk. These comparisons, called partial likelihood, learn input weights without choosing a fixed baseline hazard shape. Censored subjects contribute while they remain at risk. Events recorded at the same time need a stated tie-handling approximation or exact treatment.

**Optional math:** $`h(t\mid x)=h_0(t)\exp(x^\top\beta)`$. Here $`t`$ is time, $`x`$ lists inputs, $`\beta`$ lists learned weights, and $`h_0(t)`$ is the baseline hazard. The product $`x^\top\beta`$ is a weighted sum; the exponential makes it a positive multiplier of the baseline rate. "Proportional hazards" means the input-based multiplier stays constant over time in this version. To predict absolute survival chances, also estimate the cumulative baseline hazard, the accumulated baseline rate over time.

**Inputs/outputs and typical data types:** Input measurements, also called covariates; observed follow-up durations; and indicators saying whether an event was seen. Advanced versions can use entry times or inputs that change over time. Outputs include log relative hazards, hazard ratios comparing event rates, and estimated survival curves. A hazard ratio is neither a probability nor a ratio of survival times.

**Strengths and limitations:** Cox uses incomplete records without inventing their eventual outcomes. It leaves the baseline rate's shape unspecified. However, this version assumes input effects multiply hazards proportionally over time. Censoring must also be appropriately non-informative. After accounting for modeled inputs, why follow-up ends must not give extra clues about remaining event risk. Changing hazard ratios, omitted inputs, or too few events can undermine simple conclusions. So can competing events that prevent the target event from occurring. Check patterns in model errors and test calibration on external data. Predicted survival chances must match observed frequencies, not be trusted merely because the model produces them.

**Computational complexity / scalability notes:** Sort the observed times, then compare each event with the appropriate still-at-risk group. With fixed inputs, cumulative sums avoid rebuilding every group from scratch. More detailed weight updates can require substantially more work.

**Technical detail (optional):** With $`n`$ subjects and $`d`$ inputs, sorting costs $`O(n\log n)`$. For fixed covariates and ordinary risk sets, likelihood and gradient evaluation can cost $`O(nd)`$. A dense Newton Hessian, a matrix describing how update directions change, can need $`O(nd^2)`$ work plus an $`O(d^3)`$ solve per iteration. A risk score costs $`O(d)`$ per subject. Survival prediction also needs the estimated baseline. Ties and advanced extensions change implementation costs.

**Real-world problem solved - REQUIRED WORKED EXAMPLE:**

**Evidence status: Research benchmark.** The [ISLR2 survival lab](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch11-surv-lab.html) analyzes `BrainCancer` follow-up. Inputs are diagnosis, sex, tumor location, Karnofsky index, gross tumor volume, and treatment type. The Karnofsky index describes a person's ability to carry out daily activities. The multivariable fit uses **87 complete cases and 35 events**. "Complete cases" means required input values are present, not that every eventual event was observed. It reports coefficient **2.15457** and hazard ratio **8.62414** for high-grade glioma relative to meningioma, two diagnosis groups.

The model combines clinical inputs with event/censoring records, learns weights from the at-risk comparisons, and produces relative hazards and adjusted survival curves. The ratio compares instantaneous event rates while accounting for the other included inputs. It is **not** 8.62 times the probability of dying by a chosen date. Cox handles unfinished follow-up more naturally than applying OLS to observed durations as if they were complete event times. This analysis establishes no treatment cause and effect, validated bedside calculator, improved patient outcomes, or production clinical benefit.

**Notable vendor implementations/libraries:** R `survival::coxph`, Python lifelines `CoxPHFitter`, scikit-survival, and statsmodels `PHReg` provide Cox models. Check differences in baseline estimation, tied-time handling, regularization penalties, and support for inputs that change over time.

| Algorithm | Best-fit data type | Key strength | Key limitation | Real-world example |
| --- | --- | --- | --- | --- |
| Naive Bayes | Measurements, token counts, or yes/no inputs, depending on version | Learns class summaries quickly | Assumes limited links; chances may be overconfident | Iris Gaussian-NB flower-classification teaching test |
| Bayesian networks | Linked variables with some facts missing | Makes probability dependencies visible | Graph choice and exact calculations can be difficult | Microsoft Lumiere/Office Assistant components; included expert knowledge |
| Supervised HMMs | Sequences with known training-state labels | Efficiently combines state order and observation clues | Short memory and restricted observation links | TnT Penn Treebank part-of-speech tagging |
| Conditional random fields | Labeled sequences with designed input clues | Chooses linked labels for a whole sequence | Requires feature design; non-chain graphs can cost more | CoNLL-2002 Spanish entity-tagging tutorial |
| Cox proportional hazards | Inputs and partly observed event times | Uses incomplete follow-up without inventing events | Needs suitable hazard-ratio and censoring assumptions | `BrainCancer` analysis of prognosis-related associations |

## Public bibliography and evidence notes

Links beside each claim show where its origin or application evidence comes from. The groups below offer further reading; a bibliography alone does not prove a claim.

- **Early regression and regularization:** [Legendre's least-squares appendix](https://www.york.ac.uk/depts/maths/histstat/legendre.pdf); [Hoerl and Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634); [Tibshirani (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x); [Zou and Hastie (2005 publication)](https://doi.org/10.1111/j.1467-9868.2005.00503.x). The prostate and leukemia results use the [August 2004 manuscript](https://hastie.su.domains/Papers/elasticnet.pdf). Check the source version before comparing numbers across editions.
- **Generalized and discriminant models:** [Cox's binary-sequence paper (1958)](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x); [Hastie and Tibshirani's GAM paper (1986)](https://doi.org/10.1214/ss/1177013604); [Fisher (1936)](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x); [Smith's early discrimination paper](https://doi.org/10.1111/j.1469-1809.1946.tb02368.x). Citing these historical publications does not endorse their wider social assumptions or terminology.
- **Reproducible teaching analyses:** The ISLR2 authors' labs cover [classification](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch4-classification-lab.html), [nonlinear modeling](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch7-nonlin-lab.html), [tree ensembles](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch8-baggboost-lab.html), and [survival](https://hastie.su.domains/ISLR2/Labs/Rmarkdown_Notebooks/Ch11-surv-lab.html). These are documented analyses, not deployment studies. Their test procedures are not all the same.
- **Resampling and randomized trees:** [Breiman's bagging paper (1996)](https://doi.org/10.1007/BF00058655), [random-forest paper (2001)](https://doi.org/10.1023/A:1010933404324), and [Breiman-Cutler case studies](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm); [Geurts, Ernst, and Wehenkel (2006)](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf). Broad promotional claims in historical sources are not universal guarantees adopted by this book.
- **Symbolic trees and boosting applications:** [Quinlan's software archive](https://www.rulequest.com/Personal/); [RuleQuest's 2017 comparison](https://www.rulequest.com/see5-comparison.html); [Freund and Schapire (1997)](https://doi.org/10.1006/jcss.1997.1504); [MERL's Viola-Jones record](https://www.merl.com/publications/TR2004-043); [Friedman (2001)](https://doi.org/10.1214/aos/1013203451); [He and colleagues' Facebook report (2014)](https://quinonero.net/Publications/predicting-clicks-facebook.pdf).
- **Scalable boosting systems:** [XGBoost (2016, arXiv v3)](https://arxiv.org/html/1603.02754v3), [LightGBM (2017)](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf), and [CatBoost (2018)](https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf). The papers use different hardware, feature preparation, splits, and tuning. Their results cannot be combined into one fair ranking.
- **Neighbors and kernel methods:** [Cover and Hart (1967)](https://doi.org/10.1109/TIT.1967.1053964); [Cortes and Vapnik (1995)](https://doi.org/10.1007/BF00994018); [Drucker and colleagues' SVR paper](https://papers.nips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf); [Saunders, Gammerman, and Vovk (1998)](https://eprints.soton.ac.uk/258942/); [Rupp and colleagues (2011 preprint v1; 2012 journal publication)](https://arxiv.org/html/1109.2618v1); [Rasmussen and Williams (2006)](https://gaussianprocess.org/gpml/chapters/).
- **Graphical, sequence, and survival models:** [Maron (1961)](https://doi.org/10.1145/321075.321084); [Friedman, Geiger, and Goldszmidt (1997)](https://doi.org/10.1023/A:1007465528199); [Lumiere (1998)](https://erichorvitz.com/lumiere.htm); [Rabiner (1989)](https://doi.org/10.1109/5.18626); [Brants (2000)](https://aclanthology.org/A00-1031/); [Lafferty, McCallum, and Pereira (2001)](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf); [Cox (1972)](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x).

**Comparability limits.** Different scores answer different questions. Classification error counts wrong choices; $`R^2`$ compares numerical prediction error with a mean-based reference. Log loss checks assigned chances. Token-label F1 balances precision and recall for individual labels; ranking NDCG rewards placing relevant items near the top. A survival hazard ratio compares instantaneous event rates, not accuracy or a probability.

An out-of-bag check is not an external test. The best result chosen during validation is not a fresh test result. The CRF score checks token labels, not complete entity spans. Facebook's relative normalized cross-entropies are not click-rate or revenue gains. The GP's future curve extends a model; it does not measure future accuracy. Lumiere shows a graphical-model application and a documented product connection, not a fully supervised training recipe.

## Coverage and continuation manifest

- **Completed required scope:** All 28 entries are included: **1.1.1-1.1.8** linear, generalized, and discriminant models; **1.2.1-1.2.10** trees and ensembles; **1.3.1-1.3.5** neighbors and kernels; **1.4.1-1.4.5** probabilistic, structured, and survival models. Each sub-category's comparison table includes every entry in that group.
- **Variants made explicit:** Logistic regression separates binary, multinomial, and one-versus-rest methods. Naive Bayes separates Gaussian, multinomial, and Bernoulli inputs. GP regression differs from classification. HMM counting with known states differs from unsupervised EM. Bayesian-network classifiers differ from expert-built or unsupervised models. This chapter specifies no neural version.
- **Continue the supervised part:** See [Neural supervised models, sections 1.5-1.11](02-supervised-neural.md). A neural encoder learns an input representation that a classical final predictor can use. Such a combined model needs its own structure, training, and evidence description.
- **Other learning signals:** See [Semi-supervised learning](03-semi-supervised.md), which combines labeled and unlabeled examples; [unsupervised classical models](04-unsupervised-classical.md), which learn without supplied target labels; and [unsupervised/self-supervised neural models](05-unsupervised-neural.md). Self-supervised methods make training tasks from the data itself. HMM EM with unannotated states belongs to unsupervised learning.
- **Larger model families and mixtures:** Continue with [Foundation models](06-foundation-models.md), the [MoE model catalog](07-moe-models.md), and the [Mixture of Experts deep dive](08-moe-deep-dive.md). Averaging an ordinary forest is not automatically a learned sparse expert router, a mechanism that selects only some model parts for each input.
- **Navigation and shared interpretation:** Use the [Reading guide and evidence policy](00-reading-guide.md) to interpret claims, the [comparative guide](09-comparative-guide.md) to compare methods, and the [glossary](10-glossary.md) to check terms.
- **Non-required extensions not developed here:** The following topics remain outside this chapter:
  - **Regression:** Robust methods that resist unusual cases; quantile methods that predict parts of a distribution; and separate probit, ordinal, Poisson, and negative-binomial generalized linear models (GLMs). These GLMs use different rules for binary, ordered, or count answers. Also excluded are multivariate adaptive regression splines (MARS), which add joined curve pieces, and distributional GAMs, which model more than an average.
  - **Trees:** Oblique trees with slanted splits; model trees with fitted models in leaves; Bayesian additive regression trees (BART); quantile forests; and online ensembles updated as data arrive.
  - **Ranking, kernels, and graphs:** Full learning-to-rank algorithms; large-scale approximate kernel derivations; Bayesian-network causal discovery, which searches for proposed causal structure; and influence diagrams for decisions.
  - **Sequences:** Factorial models with multiple state chains; semi-Markov models with explicit state durations; and inference for general CRF graphs rather than simple chains.
  - **Survival:** Competing risks, which distinguish different event types; accelerated-failure-time models, which model changes in duration; recurrent events in the same subject; left truncation, where observation begins only after survival to an entry time; and time-varying models.
- **Further evidence depth not claimed:** The chapter does not independently rerun the historical experiments, validate models in future clinical studies, prove causal business impact, or list every commercial deployment. No benchmark is turned into an invented reduction in cost or downtime, productivity gain, or improvement in patient outcomes.
