# What Does the Next Unit of Computation Buy?

## A staged scientific reader on adaptive inference, recurrent representations, and learned propagators

**Status:** Research and exposition plan, 19 September 2026. The existing results cited below are evidence; all new experiments and architectures are proposals. This document does not record their execution.

**Central question:** Can we identify which operation will improve an uncertain scientific inference, reuse a compact representation across those operations, and spend computation only where the improvement justifies its cost?

**Intended contribution:** A reproducible account of discovering the structure of useful computation. Quantum error correction supplies a demanding test case: decisions have algebraic constraints, probability matters, approximate solvers can fail together, and eventually the classical computation must meet a physical deadline.

**Reader format:** Twelve chapters in four parts, approximately 24,000–32,000 words for the eventual reader, excluding technical appendices. This is a scope estimate for the finished exposition, not a commitment to complete every proposed experiment. Each part can stand alone. Later chapters are conditional on earlier evidence and remain valuable as accounts of a rejected hypothesis.

The narrative follows a concrete problem rather than a succession of model names. We begin with two approximations that agree on the wrong answer. We ask what information their agreement omitted, intervene on individual refinement operations, and test whether useful operations can be learned and reused. Only then do we ask whether the resulting process admits a compact dynamical model or a scaling relationship.

## 1. The scientific starting point

The frozen experiment used qecsim `PlanarCode(5, 5)` with independent depolarizing code-capacity noise. It is a single-shot inference problem with perfect syndrome information, not a repeated noisy-measurement circuit or a Willow experiment.

| Observation already available | What it establishes | What it does not establish |
|---|---|---|
| The two-view tolerance policy changed the exact-reference logical class on 2/2,048 shots at p=0.10 and 6/2,048 at p=0.15. | The frozen exact-outcome preservation gate failed in both strata. | The failure of every possible adaptive policy. |
| Both approximate directions agreed on all eight reference-class mismatches. | Common-mode approximation error defeats an argmax-only agreement gate. | That posterior agreement or a spectral diagnostic will necessarily solve the problem. |
| The policy/fixed-chi8 estimated-work ratio was 0.112173, with a paired-bootstrap interval of [0.106324, 0.118246]. | This implemented policy used substantially fewer estimated arithmetic operations on the recorded shots. | Accuracy-preserving savings, wall-clock speedup, or FPGA performance. |
| Exact tensor inference had 109 versus calibrated CMWPM's 148 failures at p=0.10, and 323 versus 416 at p=0.15. | The exact reference outperformed the selected matching configuration on these samples; paired tests are in the report. | Superiority over all matching methods or a state-of-the-art claim. |
| Fixed column chi8 selected the reference class on all 4,096 shots. | A useful observed comparator exists. | That chi8 is exact or has zero population mismatch probability. |
| Earlier contraction studies recorded nonmonotonic approximation error as chi increased. | More capacity need not improve every individual approximation. | A universal law of tensor convergence or shot-level accuracy from the historical mispaired data. |

The authoritative starting evidence is the [physical-shot report](planar-shot-accuracy-results.md), its [summary](../evidence/planar-shot-accuracy/summary.json), and the [public evidence manifest](../evidence/planar-shot-accuracy/manifest.json). The [historical tensor study](tensor-network-adaptive-results.md) is usable for conditional solver behaviour with its [syndrome-pairing erratum](symplectic-syndrome-erratum.md) kept explicit.

Of the eight reference-class changes, four helped the realized physical-shot outcome and four harmed it. All eight violated the frozen preservation requirement. Calling them reference mismatches avoids conflating approximation error with eight additional physical-shot decoding failures.

These outcomes motivate the new investigation. They do not supply an untouched test set for it. Existing revealed samples may support development and examples, but the original frozen report remains unchanged.

## 2. The conceptual contract with the reader

### Three levels of explanation

Every chapter provides a plain-language opening, an engineering account of what was measured, and a mathematical box for readers who want the formal argument. A beginner should understand the decision and its consequence without deriving a tensor contraction. An engineer should be able to locate inputs, outputs, costs, and failure cases. A scientist should be able to identify the estimand, competing explanation, assumptions, and falsifying observation.

### Terms that must remain distinct

| Term | Meaning in this reader |
|---|---|
| Physical time t | Successive physical measurements; absent from the current code-capacity screen. |
| Computational time or depth tau | Repeated processing of the same available observations. |
| Solver parameters | Direction, tolerance, bond caps, numerical precision, and allocation rules. |
| Learned parameters theta | Stored weights defining an encoder, update operation, head, or controller. |
| Working representation h | Per-instance information retained while inference runs. |
| Logical posterior q | Probabilities of logical classes conditional on the observed syndrome and assumed channel. |
| Refinement family | An operationally defined intervention, such as increasing selected bond dimensions or adding a contraction direction. |
| Additional information from computation | Better extraction or representation of information already supplied by observations and the model; not an additional physical measurement. |
| World model | Initially a predictor of the consequences of inference actions. A physical-noise world model is a later, separate object. |
| Multiscale | Explicit representations or computations at different resolutions. |
| Fractal | A stronger claim requiring evidence of repeated structure or statistical self-similarity across scales. |

Neural channels, physical coordinates, Fourier modes, and tensor bonds cannot be substituted for one another merely because all have an index. Likewise, tensor bond dimension is a rank restriction in this inference representation; it is not automatically a measurement of the physical device's quantum entanglement.

### The five questions carried through the reader

1. Do independent computational views disagree, and where in their probability distributions?
2. Is the leading logical class separated from plausible alternatives?
3. Does the inferred distribution change under another scale of computation?
4. What representation was discarded, and how relevant is it to the logical answer?
5. Does the current state resemble situations for which the diagnostics were calibrated?

All five are observable diagnostics or hypotheses about diagnostics. None individually certifies correctness. Lower entropy, a stable hidden state, a small update, or agreement between models can accompany a stable wrong answer.

## 3. The chapter sequence

### Part I — What must be correct?

#### Chapter 1. Two approximations agreed. Why were they wrong?

**Reader question:** How can agreement be persuasive and still fail as a safety test?

Open with the eight accepted reference-class mismatches, then explain syndromes, valid corrections, stabilizer equivalence, and logical classes. Show why matching raw correction strings is the wrong endpoint. Develop the distinction between exact-model optimality and the outcome of one physical shot: an exact posterior can choose the most probable class and still be wrong on that particular draw.

**Experiment E0: evidence reconstruction.** Reproduce the tables from the public artifact without rerunning or retuning the original experiment. Verify paired counts, class vectors, outcome discordances, and all charged work. Include one worked shot with its four-class distributions.

**Competing explanations:** Numerical ties, invalid recoveries, unequal information, or a bookkeeping error could mimic approximation failure. Refer to the existing reference checks and replay evidence before interpreting the examples.

**Figure:** The row, column, and reference probability vectors for one common-mode failure, alongside a successful case with similar cheap diagnostics.

**Exit:** Readers can distinguish syndrome validity, reference agreement, posterior fidelity, and realized logical success. The chapter is writeable from existing evidence.

#### Chapter 2. A probability distribution is more than its favourite answer

**Reader question:** What should an approximate decoder preserve?

Introduce total variation, selected-class mismatch, and reference-conditional excess decision risk. Explain that an almost-tied class swap can have small expected decision cost, while an apparently confident posterior can be badly distorted. Keep actual physical-shot BLER as a separate endpoint.

**Experiment E1: diagnostic audit.** On development-open data, evaluate the five signals against exact-posterior error and class mismatch. Inspect the eight known failures as case studies, but compute diagnostics over the full sample so the account does not become a catalogue of hand-picked successes.

Only compute a feature from information available before the corresponding decision. A spectrum obtained during a more expensive contraction is charged to that contraction. Actual logical outcomes and exact-reference margins are labels, never deployable features.

**Controls:** Class agreement alone; constant risk estimate; cheap-view margin alone; simple scalar combinations before a learned classifier. Show how much each additional signal contributes after accounting for the others.

**Figure:** Error versus cross-view disagreement, coloured by cheap-view margin, with all reference mismatches marked. Include benign disagreements and dangerous agreements.

**Exit:** A small candidate feature set and explicit counterexamples to each proposed diagnostic. This stage selects hypotheses; it establishes no new safety claim.

#### Chapter 3. Safety is a measured budget

**Reader question:** How do we progress from caution to performance without changing the definition of success after seeing results?

Explain abstention and fallback. Separate three quantities: risk among accepted cheap answers, whole-system mismatch after fallback, and coverage—the fraction of shots accepted cheaply. An always-abstaining rule may have excellent final accuracy and no adaptive value.

**Experiment E2: frozen risk ladder.** Develop candidate rules, choose thresholds on a separate calibration role, and evaluate frozen policies on new confirmation shots. Start with zero observed mismatches plus an explicit upper confidence bound. Relax finite target budgets only as predeclared additional operating points.

Do not call delta=0 an empirically certified population-risk target. With zero observed failures, a one-sided exact binomial upper bound is `1 - alpha^(1/n)`. At alpha=0.025, demonstrating an upper bound at most 0.001 requires at least 3,688 independent relevant trials; at most 0.0001 requires 36,887. If the estimand is accepted-set risk, these are accepted trials, not total sampled shots. Multiple policies and strata require further multiplicity handling or an appropriate validated risk-control procedure.

**Controls:** Exact fallback, finite-chi fallback measured separately, fixed chi policies, the original tolerance policy, and a simple calibrated scalar gate. A finite-chi fallback contributes its own errors to the system endpoint.

**Figure:** Risk–coverage–work curves with uncertainty, including the full-fallback endpoint.

**Exit:** A feasible risk ladder with adequate sample sizes. Failure to achieve useful coverage is a result and redirects attention to better refinement operations.

### Part II — Which computation helps, and why?

#### Chapter 4. Find the useful knobs before learning to turn them

**Reader question:** Which characteristics predict the value of a particular refinement?

Separate direction, tolerance, global bond capacity, local bond allocation, and logical-class allocation. Begin with the three interventions already close to the implementation: direction, tolerance, and global chi. Local allocation and class-specific work require additional solver support and their own audit.

**Experiment E3: an intervention atlas.** For identical physical shots, measure complete action outcomes and costs across a declared small grid. Proposed discovery anchors are row/column direction, chi in {2,4,8,16}, and tolerance in {0.03,0.01,0.003}; treat fixed-chi and tolerance policies as distinct arms before studying their combinations. Use a bounded pilot to remove redundant or prohibitively expensive actions before freezing the scientific grid.

Estimate paired changes in exact-posterior error per additional cost. Condition those changes on spectral tails, margin, directional discrepancy, geometry, and scale drift. Define groups on development data and test their predictive value on independent shots.

**Controls:** Equal-budget random refinement, uniform capacity increases, direction-only changes, and an oracle that sees the exact target. The oracle measures available opportunity and cannot be deployed.

**Figure:** State characteristic by refinement family, showing mean gain, uncertainty, harmful-refinement frequency, and cost. Include nonmonotonic cases rather than sorting them away.

**Exit:** Either a reproducible pattern linking state characteristics to useful actions, or evidence that the available actions mostly differ by generic capacity. Only the first supports a specialised controller.

#### Chapter 5. Where did the missing information go?

**Reader question:** Does a large discarded component matter to the logical answer?

Explain singular spectra with a low-rank matrix example, then the extra complication of a surrounding tensor network. Local discarded norm is a representation diagnostic; it is not generally a bound on normalized logical-posterior error.

**Experiment E4: local relevance interventions.** On an audited local-refinement engine, enlarge selected bonds and compare with equally costly enlargement of random or uniformly chosen bonds. Use reference-derived local importance as an oracle control, and inference-visible spectral/environment features for deployable selection. Preserve the contribution of all logical classes when comparing their masses.

Use canonicalization or gauge-invariant summaries when comparing internal tensor states. Arbitrary tensor entries are not directly comparable across equivalent gauges. Validate the engine against the existing full contraction and small-code enumeration.

**Controls:** Largest-tail allocation, random allocation, uniform allocation, and a simple learned relevance score. Test combinations of interventions: the gain from two refinements may be complementary or redundant, not additive.

Attribute behaviour to defined operations or modules before attributing it to individual learned weights. Hidden representations can be reparameterized without changing the function. For a claimed specialised module, disable it, replace it with an equally costly generic update, and test whether the predicted class of problems selectively degrades. Repeat across training seeds; an attractive activation plot alone does not establish a reusable computational role.

**Figure:** Discarded spectral weight versus actual posterior improvement after intervention, with examples of large irrelevant tails and small consequential components.

**Exit:** A measurable advantage for targeted capacity or a reason to prefer the simpler global schedule. This chapter supplies the causal basis for the phrase “useful parameter family.”

#### Chapter 6. Learn the operation; reuse the machinery

**Reader question:** Can a shared update replace a sequence of separately parameterized transformations?

Introduce the proposed update

\[
h_{k+1}=h_k+F_{\theta_{a_k}}(h_k,s,\text{scale}),
\qquad q_k=\operatorname{softmax}(G(h_k)).
\]

The syndrome and permitted model information are s; a_k selects a refinement module. Explicitly distinguish a model that predicts a posterior directly from one that chooses actions for the numerical solver. Test the controller first; a direct surrogate has an additional approximation burden.

**Experiment E5: recurrence versus stored capacity.** Compare an untied-depth model, a shared recurrent block, a small bank of shared modules, and the best symbolic schedule. Proposed discovery loop counts are {1,2,4,8}; evaluate {12,16} as a predeclared extrapolation probe, not as assumed improvement. Use at least five training seeds for a confirmatory neural comparison, with training and tuning compute recorded.

Train on intermediate states using exact-posterior supervision where available, variable recurrence counts, and matched information. Do not require every per-step error decrease as a hard condition: useful computation may have nonmonotonic intermediate outputs. Evaluate the trajectory and its eventual decision.

**Controls:** Fixed repetitions, adaptive repetitions, shared versus untied weights, step/scale conditioning, and a simple linear or MLP controller. Compare at matched inference compute and separately at matched stored-weight bytes; a single comparison cannot generally match every resource simultaneously.

**Figure:** Posterior error versus inference work, with separate panels for stored weights and peak working memory.

**Exit:** Evidence that reuse buys accuracy or resource efficiency against the strongest simple comparator. If shared recurrence loses, keep the solver-based scheduler and document the representation limit.

### Part III — What must the representation remember?

#### Chapter 7. Does another loop think, remember, or repeat?

**Reader question:** What useful state accumulates across refinement steps?

Separate stored weights, live inference tensors, saved training activations, and any retained history. Weight sharing reduces one quantity; it does not guarantee reductions in the others. Explain why an embedding that discarded a distinction cannot recover it merely by receiving extra empty channels.

**Experiment E6: memory interventions.** Compare intact recurrence with hidden-state reset, within-stratum state shuffling, and independent repeated passes at matched work. Include models trained under the corresponding reset condition: an inference-only reset can simply cause distribution shift. Vary state width and quantization, and compare access to the original syndrome versus a compressed input alone.

Test nested active channel groups with a defined initialization or lifting map when new groups activate. Distinguish weights resident on device from parameters active in one step. Count both. Profile whether masks actually skip allocations and arithmetic.

**Controls:** No-history Markov features, a fixed window of diagnostics, and an equally wide nonrecurrent model. Quantize only after establishing a floating-point mechanism, then measure any change in convergence and calibration.

**Figure:** The recurrence-count by state-width surface, with iso-error contours and measured memory. Add a panel showing where extra iterations cannot compensate for too little state.

**Exit:** A measured claim about parameter storage, working memory, or useful retained state, stated separately. Representation sufficiency is empirical and specific to the tested tasks.

#### Chapter 8. Space, frequency, time, and correlation are different budgets

**Reader question:** Which notion of scale matches the missing information?

Present spatial resolution, retained Fourier modes, physical history length, computational depth, and tensor correlation capacity as different interventions. Spatial and frequency resolution are linked by sampling and aliasing; they are not independent labels to multiply without constraints. Physical time and computational depth remain separate throughout.

**Experiment E7: controlled structure families.** First build small tasks with known spatial bandwidth, localized defects, anisotropy, and controlled correlation structure. These are deliberate mechanism tests. Repeated-time tasks arrive later and require a generator with physical history and a causal information boundary.

Use staged factorial designs rather than a full Cartesian product. First test one factor at a time around a declared baseline, then the interactions suggested by the intervention atlas, then hold out entire combinations. Match stationary marginals when testing whether temporal memory adds value.

Compare spatial CNN/graph operations with FNO where the geometry supports it. For physical-time memory, compare finite windows and EWMA banks with HiPPO at matched state bytes. For correlation capacity, compare pairwise representations with selected higher-order interactions or larger tensor bonds. These are separate subexperiments until their individual contribution is established.

**Controls:** Spatial permutation consistent with a specified symmetry audit, missing-history controls, localized versus smooth fields, and held-out scales. Handle planar boundaries explicitly; the current planar geometry is not a cyclic LP lift. Complex Fourier diagonalization of a circulant embedding does not solve binary parity inference.

**Figure:** A map of which refinement wins in which controlled regime. A complementary pair must beat its single-component controls with a measured interaction or allocation advantage.

A later renormalization branch must define a coarse-graining map and the resulting effective model, including how constraints, noise, and logical observables are represented. Compare “refine then coarse-grain” with “coarse-grain then apply the corresponding effective update” on observables of interest. Test reuse of the same update rule across several held-out scales. Approximate agreement would motivate a scale-consistent propagator; hierarchical pooling alone would not establish an RG description. Spatial, temporal, and spectral scaling exponents must be estimated separately before any self-similarity claim.

**Exit:** An experimentally supported division of labour. “Fractal” remains an optional later hypothesis about repeated scale structure, not a synonym for these mechanisms.

#### Chapter 9. A world model of the next computation

**Reader question:** Can a model predict which action will be worth taking?

The initial world model predicts the outcome of a solver action, not the underlying quantum device. Its targets include the next diagnostic state, posterior change, validity, estimated or measured cost, and residual error learned from offline reference labels.

**Experiment E8: action-conditioned prediction and planning.** Collect branches from the same solver state using the real action interface. Train and assess one-step prediction first. Compare a greedy gain-per-cost rule with depth-two planning on trajectories where action order can actually change the next available state.

Independent full reruns form a selection table. They do not by themselves demonstrate resumable dynamics or the value of sequential planning. A resumed engine must account for the state it retained and cannot restore discarded tensor information without an explicit reconstruction or recomputation path.

**Controls:** A symbolic scheduler, lookup table, linear action-value predictor, contextual bandit, and an oracle action policy. Condition action predictions only on current information. Evaluate unsupported actions and shifted states as abstention cases rather than silently extrapolating rewards.

RL is a conditional extension if delayed gains and state-dependent trajectories defeat myopic selection. A GFlowNet is a separate diversity experiment only if multiple useful refinement plans need to be sampled; compare with beam search, random proposals, and simple stochastic policies before claiming a benefit. Any rewards must use training/reference information only within their permitted data role.

**Figure:** Predicted versus actual action gains, followed by deployed policy regret relative to the action oracle, including planning overhead.

**Exit:** A predictor that improves actual decisions after its own costs are charged. Low average transition-prediction error alone is insufficient.

### Part IV — From reusable computation to a propagator

#### Chapter 10. Does inference have learnable dynamics?

**Reader question:** Is there a simpler mathematical description of repeated refinement?

Compare three explicit hypotheses: a discrete nonlinear shared update; continuous computational dynamics `dh/dtau = F_theta(h,s,a)` with the required scale/step context supplied; and an approximately linear action-conditioned evolution in learned observables, `z_next = K_a z`, where `z = psi(h,s,scale,step)`. Supply identical syndrome, scale, action, and step information to all candidates. A reduced `psi(h)` is a separate sufficiency ablation, admissible only with explicit tests of what h retains; it is not the default comparison. Include fixed context in the augmented state and define how changing context is updated so apparent non-Markov behaviour is not caused by omitted inputs.

**Experiment E9: propagator comparison.** Fit all candidates to the same action trajectories and prediction targets. Include linear dynamics in the original feature space as a baseline. Match model-selection effort and count every neural-ODE function evaluation, rejected integration step, encoder/decoder operation, and planning call.

A Koopman interpretation requires an approximately closed observable representation with useful multi-step prediction. A Fourier multiplier followed by nonlinearities is not sufficient evidence. A finite learned representation need not inherit the exact closure of an infinite-dimensional operator.

**Controls:** One-step versus multi-step supervision, perturbed starting states, action sequences held out from training, and a stable-but-wrong fixed-point example. Test calibration and exact-posterior error alongside latent convergence. Numerical solver tolerance does not certify scientific inference accuracy.

**Figure:** Rollout error and decision quality versus horizon and actual work; add a stability plot under state perturbations.

**Exit:** Select the simplest propagator that improves downstream inference. If discrete recurrence is best, the scientific argument ends there without requiring an ODE or Koopman label.

#### Chapter 11. When a four-coordinate description earns its place

**Reader question:** Can physical evolution and computational refinement coexist in one structured representation?

A prospective planar streaming state is `h(x,y,t,tau)`: two spatial coordinates, physical measurement time, and computational refinement depth. Latent channel width is an additional representation budget, not automatically a fourth physical coordinate. Tensor bond indices have another role again.

**Experiment E10: causal streaming extension.** Build verified repeated-measurement circuits and use one shared circuit and detector error model per comparison. Establish a strong matching baseline, round and boundary conventions, final-observable scoring, and the availability of side information. Introduce an accumulating error/frame interpretation explicitly rather than reusing independent code-capacity shots as a physical trajectory.

Test factorized spatial mixing, causal physical-time memory, and recurrent computational updates before a dense four-axis operator. Temporal mixing must not see future measurements. General planar boundaries and a finite computational horizon do not justify circular convolution on every axis.

Introduce real device data as a separate transfer study only after its release contents, detector mapping, observables, sequence boundaries, and permitted information are audited. A Willow dataset does not provide exact latent noise states or the exact posterior used in the small synthetic oracle. Synthetic-to-device disagreement therefore needs different validation endpoints.

**Controls:** Fixed-window decoding, matched simple temporal filters, factorized versus joint operators, and stationary-noise controls. Measure forecast quality and decoder utility separately.

**Figure:** The causal timeline distinguishing when a syndrome arrives, which history is available, which computational actions run, and when the decision is returned.

**Exit:** Demonstrated value from modelling both physical and computational evolution. The current single-shot evidence cannot support this chapter's result in advance.

#### Chapter 12. What scales—and what fits on a device?

**Reader question:** Does the mechanism persist as the problem and resource budget grow?

**Experiment E11: scaling and implementation.** Measure achievable posterior error and logical risk over stored parameters P, peak working memory M, compute C, and a task descriptor including code size, noise, and geometry. Fit a response surface before proposing a power law. Compare power-law, saturating, piecewise, and simple nonparametric descriptions using held-out sizes and uncertainty estimates.

Record approximation floors and changing bottlenecks. More recurrence cannot overcome every loss of representation; larger bond capacity can change the dominant cost. A threshold variable such as p/p_threshold cannot be inserted without an independently established threshold and convention.

Use at least four sizes where trustworthy reference computation is feasible for an exploratory size-scaling analysis. If that range is unavailable, report resource curves at fixed size instead of claiming a size law. At larger sizes, label approximate reference calculations as such and distinguish their evidence from exact-oracle validation.

Only after the accuracy/resource frontier is established, implement fixed-point and FPGA emulation for a selected policy. Measure single-shot and per-round latency distributions, p95/p99 tails, peak memory, synthesis resources, and backlog under the target syndrome arrival process. Include routing, data movement, fallback, and host/device transfers. Weight reuse helps hardware only if the execution schedule realises it.

**Controls:** Best fixed policies and simple adaptive policies on the same target; matched risk; fixed versus adaptive channel activation; several precisions; workloads with clustered difficult shots. Batch throughput on a GPU is not a substitute for the stream deadline.

**Figure:** Empirical error–compute–memory surfaces, then a separate timeline and backlog plot for the chosen implementation.

**Exit:** A bounded scaling relationship or a measured counterexample, and separately a deployment claim supported by the relevant hardware measurements. Scaling, self-similarity, threshold behaviour, and speedup each require their own evidence.

## 4. Formal objects that unify the chapters

For a syndrome s, let q_star be the trusted reference distribution over logical classes, q_a an action's approximate distribution, and c_a its chosen class.

### Fidelity, decisions, and outcomes

Use total variation as an interpretable primary posterior discrepancy:

\[
E_{\mathrm{TV}}(q_a,q_*)=\tfrac12\sum_c |q_a(c)-q_*(c)|.
\]

Jensen–Shannon divergence can supplement it. Pairwise log-ratio error is useful for small class masses but requires a declared treatment of zero or invalid masses. Do not silently clip invalid solver results into apparently valid distributions.

The reference-conditional excess decision risk is

\[
R_*(a\mid s)=\max_c q_*(c\mid s)-q_*(c_a\mid s).
\]

It is nonnegative and quantifies the expected decision cost under the assumed reference model. It is not the realized error of a sampled physical shot, and it need not describe a mismatched device noise distribution. Report reference class mismatch and paired physical-shot BLER alongside it.

### The value of an action

For a real state h and refinement action a, define an offline measured utility

\[
U(a\mid h)=\frac{E(q_h,q_*)-E(q_{h\rightarrow a},q_*)}
{C(h\rightarrow a)}.
\]

The error measure E and charged cost C must be declared. Allow negative utility. The target is reduced error against a trusted target, not reduced entropy or mutual agreement alone. In deployment, utility is predicted from permitted state; q_star is unavailable.

For independent reruns, charge the whole new run and all earlier probes. For actual resumption, charge measured incremental computation and the retained state. A full-information oracle omits the difficulty of selecting actions; it is a bound on opportunity, not a claimed deployable saving.

### The resource surface

The eventual object of study is

\[
E_{\min}(P,M,C;\mathcal D),
\]

or its inverse, the least measured cost at a specified error/risk level on task distribution D. P is stored weight bytes or parameter count at a specified precision; M is measured peak working memory; C is an explicitly named work or timing measure. Do not collapse these into one undefined “model size.”

Measure the response surface first. A scaling law is a compact predictive model of that surface, with a stated range and out-of-sample tests. It is not guaranteed by drawing logarithmic axes.

## 5. Data, statistics, and review protocol

### Data roles

Use distinct immutable identities for historical development, new training, model selection, risk calibration, and final confirmation. Keep all actions and all trajectory steps from one physical shot in the same role. For streaming data, split by independent complete sequence or device run as appropriate; adjacent windows are not independent samples.

Historical confirmation data can become declared development data for a new protocol after publication, but can never confirm the retuned rule. Every final campaign binds source, configuration, circuit/code, solver versions, sampler identities, and model weights before opening confirmation results.

### Sampling and selection

Use a small pilot to estimate runtime, event rates, and feasible reference sizes. Freeze the confirmatory sample size from a prespecified effect or bound requirement. Fixed-shot campaigns are the default. Any sequential stopping rule needs valid sequential inference; repeatedly checking a fixed-sample confidence interval and stopping when it looks favourable is not the same procedure.

Risk selection over many thresholds must account for selection. Options include independent selection and calibration roles with appropriate simultaneous tests, or a fully specified risk-control method whose assumptions are checked. Report accepted-set risk, final-system risk, coverage, and fallback risk separately.

For physical shots, use paired comparisons and appropriate binomial intervals. For training variability, report each seed and distinguish variability across trained models from uncertainty across evaluated shots. Resample entire independent sequences for temporal experiments. Action-table rows are repeated measurements on the same shot, not additional independent shots.

A low number of discordances can motivate a failure analysis but cannot establish a precise population gap. Report both estimate and interval. A non-significant difference is not proof of equivalence; any noninferiority or equivalence margin must be frozen in advance.

### Information symmetry

The baseline and proposed system receive the same syndromes, channel assumptions, heralding/readout information, and history unless the study explicitly measures the value of extra information. Exact posteriors, sampled physical errors, and realized logical outcomes belong to offline labels or scoring. They never enter an inference feature through a diagnostic or cache.

At each proposed early exit, list the features available by that moment and their incurred costs. A controller cannot make an inexpensive decision using spectra from an uncharged expensive computation.

### Review checkpoints

Each execution stage requests an independent scientific review before its new confirmation domain is opened, followed by an artifact/replay review after evaluation. The reviewer checks the question and estimand before looking at the result.

Required decisions are: information parity; reference validity; correct logical outcome scoring; sufficient baseline strength; valid accounting of selection and sample dependence; realistic compute accounting; and the cheapest plausible disconfirming experiment. Findings are resolved before progression. Preserve negative results and errata with their provenance.

## 6. Execution order and bounded first deliverable

This is the program map, not permission to launch every costly branch. Later stages are conditional on observed evidence and receive their own concrete implementation plans and resource estimates.

| Stage | Deliverable | Required evidence to advance | If it fails |
|---|---|---|---|
| A: E0–E1 | Reproduced evidence and diagnostic atlas | Auditable feature availability and testable diagnostic hypotheses | Publish why agreement-based diagnostics fail; reconsider observations. |
| B: E2–E3 | Risk ladder and refinement intervention table | Useful accepted coverage or reproducible action-dependent gains | Retain the best fixed solver; do not add a controller. |
| C: E4 | Audited local/resumable refinement interface | Local allocation helps after all costs are charged | Use global actions and simplify the architecture. |
| D: E5–E6 | Recurrence and memory ablations | Shared updates improve a stated resource frontier | Keep symbolic or untied alternatives and explain the failed compression. |
| E: E7 | Controlled multiscale mechanism map | Reproducible scale-specific gains or interactions | Avoid a combined multiscale model. |
| F: E8 | Action-conditioned surrogate and deployed scheduler | Prediction improves decisions beyond a simple rule | Stop at greedy or symbolic scheduling. |
| G: E9 | Discrete/ODE/Koopman comparison | A richer propagator buys measured downstream value | Retain the discrete update. |
| H: E10 | Verified causal streaming experiment | Physical-time information improves the relevant task | Scope the contribution to single-shot computation. |
| I: E11 | Resource scaling and selected hardware implementation | Held-out predictive scaling and measured hardware feasibility | Report bounded empirical frontiers and the bottleneck. |

Execution order differs slightly from reading order. The temporal subexperiment discussed in E7 waits for the causal data and scoring prerequisites in E10; the spatial and correlation subexperiments can proceed earlier. A new correlated-noise task also needs a compatible trusted reference or exact small-instance enumeration: an iid-channel oracle cannot silently serve as ground truth for a different channel. The ODE/Koopman comparison can use computational trajectories before a physical-time experiment exists.

### First executable tranche

1. Reconstruct existing physical-shot aggregates from the published artifact and produce a compact evidence table. Preserve the original report and frozen inputs.
2. Extract inference-visible margins, cross-view discrepancies, and spectral summaries for all revealed development shots. Produce the five-signal availability ledger and cost ledger.
3. Audit common-mode failures alongside matched benign cases. Test whether apparent signals survive adjustment for cheap margin and physical error rate. Label all resulting thresholds exploratory.
4. Run a bounded discovery pilot on 64 fresh development shots per existing rate to measure the cost and utility of the proposed direction/chi/tolerance grid. The pilot is for engineering and design, not confirmation.
5. From those results, write a frozen first risk-ladder contract: exact estimands, action set, baseline, fallback, effect threshold, multiplicity, sample-size calculation, and independent seed domains. Review it before generating confirmation samples.

This tranche can use the existing CPU numerical infrastructure and does not require a neural model, GPU training, or hardware deployment. Estimate runtime from the pilot; do not extrapolate a full campaign budget from operation counts alone.

### Artifact layout

Proposed future outputs live under `evidence/adaptive-computation/<campaign>/`, with `manifest.json`, a compact `summary.json`, and figure-source tables. Large trajectories use a versioned external artifact with checksums, following the existing planar publication pattern.

Each stage's `docs/adaptive-computation/<stage>-results.md` records the question, frozen protocol, result, strongest counterexample, claim boundary, and next decision. Runnable commands are documented only when their implementation exists. A future action-state schema must include shot identity, action identity, pre-action feature provenance, state/recomputation policy, class probabilities, validity, cost, and a separately controlled reference/outcome join.

Existing integration points include [tensor instrumentation](../src/qldpc_fno/decision/tensor_network.py), [physical-shot generation](../src/qldpc_fno/decision/planar_shot_data.py), and [physical-shot scoring](../src/qldpc_fno/decision/planar_shot_accuracy.py). New studies should add separate modules rather than rewrite the frozen scientific record.

## 7. How each chapter becomes an experienced scientific exposition

Use a consistent rhythm without forcing every chapter into identical prose:

1. Start with a concrete decision or surprising observation.
2. Explain the smallest mathematical object needed to understand it.
3. Present the proposed mechanism and its strongest competing explanation.
4. Show the intervention that distinguishes them.
5. Present the paired evidence and uncertainty, including a failure case.
6. State exactly what changed in the next experiment because of the result.

The main text follows a small recurring collection of examples: an easy shot, a rescued disagreement, a common-mode failure, a case where more capacity harms the posterior, and a case where state compression loses useful distinctions. They are illustrations, never replacements for full-sample statistics.

Use one central figure per chapter and optional technical panels. Every scientific plot should have a saved source table, a reproduction command, units, and an independent-sample count. Show theoretical diagrams as diagrams; never draw imagined experimental curves as measured results.

Potential chapter titles are deliberately questions about mechanisms. FNO, HiPPO, RL, tensor networks, and Koopman models enter when they offer a testable answer. A reader should be able to learn from the program even if the strongest final system is a small symbolic controller around a conventional solver.

## 8. Primary literature and its role

These are conceptual anchors, checked against primary-source records on 19 September 2026. They are not a completed novelty search. Before making a contribution or state-of-the-art claim, perform a focused current review for the specific successful mechanism and compare published baselines under compatible tasks.

| Source | Where it belongs | What it supports | What remains our experiment |
|---|---|---|---|
| [Bravyi, Suchara, Vargo: Efficient Algorithms for Maximum Likelihood Decoding in the Surface Code](https://arxiv.org/abs/1405.4883) | Chapters 1, 4, 5 | Tensor-based logical-class inference as a reference algorithmic foundation. | The usefulness of the proposed adaptive allocation and learned controller. |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | Chapter 3 | A framework for calibration and statistical risk control. | The exact QEC loss, valid sampling unit, policy family, and selected procedure. |
| [Universal Transformers](https://arxiv.org/abs/1807.03819) | Chapters 6–7 | Shared recurrent transformations and adaptive computation. | QEC posterior fidelity, recurrence efficiency, and memory tradeoffs. |
| [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171) | Chapters 6–7 | Large-scale precedent for recurrent-depth latent computation. | Whether shared updates transfer to these inference trajectories. |
| [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) | Chapters 8, 10, 11 | Operator learning with Fourier-parameterized kernels and PDE resolution transfer. | Which QEC geometry and spectral families benefit; code-family transfer is separate. |
| [HiPPO](https://arxiv.org/abs/2008.07669) | Chapters 7–8, 11 | Online polynomial representations of history under specified measures. | Which causal history improves decoding at matched memory and work. |
| [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) | Chapter 10 | Continuous-depth parameterization and numerical integration of learned dynamics. | Accuracy and total cost compared with discrete refinement. |
| [Koopman neural operator](https://arxiv.org/abs/2301.10022) | Chapter 10 | Combining learned operator models with a Koopman-inspired description of dynamics. | Observable closure, controlled action transitions, and useful rollout prediction here. |

Optional later reading branches cover adaptive mesh refinement, tensor-network error estimation, multigrid, model-based control, and risk-sensitive sequential decision-making. Select exact papers when the corresponding stage is specified; their terminology should not substitute for an experiment.

## 9. Corrections carried forward from the initial discussion

- Zero observed errors is a finite-sample outcome with an upper bound, not a demonstrated zero population risk.
- A group-level calibrated risk statement is not a distribution-free conditional guarantee for each individual shot.
- Better agreement between approximate posteriors can mean shared bias; supervision and evaluation retain a trusted target.
- A singular-value truncation tolerance is not automatically a normalized-posterior error bound.
- Repeated shared layers can reduce weight storage while retaining substantial activation, state, and execution costs.
- Independent solver reruns do not establish incremental reuse or justify charging only a hypothetical delta cost.
- The current code-capacity evidence has no physical-time axis. A four-coordinate streaming model requires new data and causal validation.
- Neither operator resolution transfer nor FFT-compatible structure proves transfer between different quantum code instances.
- A fitted power law, an RG interpretation, and fractal self-similarity are distinct claims. Each requires its own held-out evidence and mathematical definitions.

## 10. What would make this reader substantial?

The contribution is a sequence of increasingly discriminating questions. First establish what failed. Then determine which computation changes the answer for the right reason. Test whether those operations can be reused with less stored capacity, whether their retained state is sufficient, and whether their consequences can be predicted cheaply enough to guide action. Finally ask how the resulting tradeoff scales and whether it survives a physical execution budget.

A successful ending could be a compact recurrent adaptive decoder. It could also be evidence that a symbolic schedule captures the available benefit, that memory is the limiting resource, or that certain attractive propagator descriptions do not improve decisions. Each is scientifically useful if the experiment distinguishes it from its alternatives and the reader can reproduce the distinction.
