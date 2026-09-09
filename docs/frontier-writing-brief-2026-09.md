# What should a learned quantum decoder learn?

A research and writing brief, checked against primary literature on 9 September 2026.

## The argument that can connect a series

Quantum error correction is a useful place to study a general problem in scientific machine learning: a model can make accurate predictions while contributing little to the decision that follows. Conversely, a weak predictor can sometimes provide a useful alternative solution to a strong solver. Understanding the interface between the learned component and the scientific algorithm is therefore a research problem in its own right.

The experiments in this repository give that argument substance. We tried learned corrections and learned priors, constructed controlled spatial and temporal noise, compared memory mechanisms, and eventually separated noise identifiability from decoding performance. These investigations support a series about how to identify useful learning problems inside a scientific system. They do not yet establish a superior FNO–HiPPO decoder, code-size transfer, or real-time hardware performance.

The closest recent literature concerns adaptive noise estimation, interpretable streaming memory, decoder ensembles, and calibration during computation. The most useful articles would explain one mechanism, reproduce a small consequence, and use the result to interrogate a recent paper. The recommendations below are editorial judgments; proposed experiments are not reported results.

## What the existing evidence can carry

| Investigation | Evidence available | Defensible use in an article |
|---|---|---|
| Fixed-shot learned-decoder comparison | At `p=0.0375`, 2,048 shared test shots gave 387 BP-LSD failures, 835 soft-prior failures, and 819 residual-repair failures. The report includes paired discordances and exact tests. | This particular frozen learned intervention harmed accuracy. One code, noise point, and training run do not settle the architectural question. |
| Crossed spatial/temporal screen | CNN/FNO × FIR/HiPPO, five noise regimes, but only two reused validation sequences and 32 scored rounds per arm/regime; no independent test set. FNO's forecast NLL was worse than the same-memory CNN in all ten comparisons. | A worked example of an unsuccessful preliminary screen and the next diagnostic it motivated. It is explicitly `reduced_non_scientific`, not evidence that CNNs generally beat FNOs or that memory is useless. |
| Temporal identifiability study | The confirmation manifest records a grid-Bayes expected cross-entropy gain of approximately `0.001550` nats/qubit/round over the known-marginal forecast, with a one-sided 95% lower bound of `0.001264`, across 64 independent drifting-noise test sequences. Stationary and mismatched-history controls are included. | Evidence for predictability in the specified synthetic latent-noise model. Preserve the distinction between the independently checked forecasting component and the unresolved full decoder replay. |
| Decoder replay investigation | The prior investigation found different higher-order LSD corrections for identical inputs, including different predicted logical signatures. The full confirmation replay did not pass its decoder comparison. | A reproducibility investigation, pending a packaged minimal reproducer and an intervention that establishes the cause. Do not use the persisted BLER table to claim confirmed equivalence or absence of decoder benefit. |
| Candidate pooling | Earlier exploratory inspection suggested some baseline failures had successful alternatives among learned candidates. The test set had already been inspected. | Motivation for a fresh, preregistered candidate-generation and selection study. It is not a confirmed deployable ensemble improvement. |

Sources within the project: [fixed-shot report](../artifacts/accuracy-disconfirm-p0375/summary/results.md), [crossed-screen report](causal-fno-hippo-results.md), [confirmation manifest](../artifacts/temporal-identifiability-confirmation-v2/manifest.json), [identifiability configuration](../configs/temporal_identifiability.json), and [independent replay implementation](../src/qldpc_fno/identifiability/screen.py). The artifact directories are local, ignored experiment outputs; those links will not supply the data in a fresh public clone. An article using their numbers needs a compact, immutable evidence release. The standalone replay-failure log and minimal reproducer also need to accompany any public account of that investigation.

The current scientific runs concern `lp(3,7)_16`, with `n=2610`, `k=744`, and lift order 45, under independent-Z code-capacity noise with perfect syndromes. The temporal generator resets the physical error each round while correlating its latent probability. It is not a repeated circuit-level memory experiment with an accumulating Pauli frame, nor an experiment on Willow. These boundaries must appear beside the results, not only in a limitations appendix.

## 1. Can a quantum computer's error checks act as a noise sensor?

**Best first technical walkthrough.** The reader learns something substantial before encountering a neural network: how a parity measurement can reveal information about an unobserved error process.

### The scientific argument

For independent binary errors on the qubits in a check support \(A\),

\[
\Pr(s_A=1)=\frac{1-\prod_{i\in A}(1-2p_i)}{2}.
\]

For a uniform error probability \(q\) and a weight-\(w\) check, this becomes

\[
r(q)=\frac{1-(1-2q)^w}{2}.
\]

This is an observation model. It explains both the opportunity and a limitation: the detector rate changes with the physical error rate, but its sensitivity can diminish as the parity approaches a fair coin. Checks sharing qubits also produce correlated observations. Counting every check as an independent sample would exaggerate the available information.

Our identifiability experiment selected 135 disjoint weight-10 checks. That makes an exact conditional observation likelihood tractable for the deliberately simple global latent channel. It is a useful teaching example of designing the experiment around what can actually be inferred. See the [observation model](../src/qldpc_fno/identifiability/observation.py) and [study design](superpowers/specs/2026-09-05-syndrome-identifiability-design.md).

### Closest literature

Wagner and colleagues established conditions under which syndrome measurements identify Pauli-channel information, including constrained correlations. Their result depends on the channel class and code properties; it does not say that arbitrary hardware noise can be uniquely reconstructed from arbitrary syndrome data. This supplies the foundational question: what is identifiable? [Wagner et al., *Quantum*, 2022](https://quantum-journal.org/papers/q-2022-09-19-809/).

Bhardwaj and colleagues move to drifting noise. They derive the frequency-filtering behavior of syndrome windows and use adaptive estimation in phenomenological and circuit-level decoding studies. Their paper appeared in *PRX Quantum* on 6 August 2026, following a November 2025 preprint. This is the closest current reference to our observation-only temporal work. [Bhardwaj et al., 2026](https://journals.aps.org/prxquantum/abstract/10.1103/z1hc-nqw5).

### Small experiment that makes the article ours

1. Verify the parity formula by exact enumeration for a small check, then by Bernoulli sampling for weight 10.
2. Generate stationary, slowly drifting, and abruptly changing scalar error rates. Keep each complete trajectory in one train, validation, or test split.
3. Compare a constant estimate, causal moving windows, EWMA, and the existing Bayesian filter. Give every estimator the same past observations; no centered window may inspect future rounds.
4. Sweep drift frequency and window size. Plot prediction error, tracking lag, and uncertainty. Keep forecasting one step ahead separate from estimating a rate using measurements from that same step.
5. Attach a stationary negative control and a whole-history derangement control. Evaluate uncertainty across independent trajectories, not across correlated rounds as if they were independent.

**Data and execution:** Start with the existing [generator](../src/qldpc_fno/identifiability/generator.py), [filters](../src/qldpc_fno/identifiability/filters.py), and [endpoints](../src/qldpc_fno/identifiability/endpoints.py). The small analytic example needs NumPy and a CPU; Stim becomes useful when extending to noisy repeated measurement circuits. No GPU or FPGA is needed to establish this mechanism.

**Diagnostic figure:** recovered drift amplitude and phase lag versus frequency for several memory lengths. This is more informative than one average loss number.

**Claim boundary:** Our scalar latent experiment is a controlled example, not a replication of all the paper's channel-estimation results. On hardware, latent error probabilities are not available as ground-truth labels.

## 2. When does a better noise forecast improve a decoder?

**Strongest article for a broad ML/AI4Science audience.** This takes the forecasting result seriously while asking what the downstream algorithm can use.

### The scientific argument

For independent errors with probabilities \(q_i\), a most-likely *individual error* consistent with syndrome \(s\) minimizes

\[
\widehat e=\arg\min_{He=s}\sum_i e_i\log\frac{1-q_i}{q_i}.
\]

If every \(q_i=q<1/2\), the objective is a positive scalar times Hamming weight. Changing that scalar does not change the set of minimum-weight solutions. A forecast can therefore improve a proper probability score while leaving this exact decision unchanged.

That is a narrow analytical observation, not an explanation already established for our BP-LSD results. Quantum maximum-likelihood decoding sums probabilities over stabilizer-equivalent errors; coset rankings can respond differently. Sum-product BP, approximate search, numerical clipping, and tie handling also need separate analysis. Uniform-prior invariance of an exact individual-error objective is not a theorem about every implementation.

The useful next question is whether estimating *relative spatial risks* or choosing between logical alternatives matters more than estimating the global rate accurately.

### Closest literature

Decision-focused learning studies predictions through the quality of the resulting optimization decisions. Mandi and colleagues formulate it through ranking feasible solutions, a particularly close analogy to choosing decoder candidates. [Mandi et al., ICML 2022](https://proceedings.mlr.press/v162/mandi22a.html).

More recent decision-focused fine-tuning work addresses a relevant tension: improving decisions can distort the semantics of predictions. A decoder-oriented learned weight vector should therefore not automatically be described as a calibrated physical noise estimate. [Yang et al., AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34891).

The adaptive-noise paper in article 1 provides the QEC counterpart: changing inferred noise can improve logical performance in its studied settings. Our task is to identify when that translation happens, with the decoder and noise model held explicit.

### Small experiment that makes the article ours

Use a tiny CSS code small enough to enumerate all relevant error strings and stabilizer classes. Compare exact individual-error decoding and exact coset decoding while sweeping assumed noise parameters. Then construct two noise conditions with the same mean physical error rate: one spatially uniform and one spatially heterogeneous.

Give the decoders either the nominal rate, the correct global mean, or the correct per-qubit field. Measure separately: probability-score improvement, changes of correction, changes of logical class, and logical failure probability. Exact enumeration removes Monte Carlo uncertainty from the toy example and supplies a reference before repeating it with BP-LSD.

For any learned ranking follow-up, train and select on fresh splits. Keep a calibration objective if the output will also be used as a probability; otherwise name it a decoding score. Candidate ranking losses do not solve the separate problem of generating a useful candidate set.

**Data and execution:** Enumerated toy data first, followed by the existing code-capacity generator. A CPU suffices for the toy. Decoder comparisons on the LP code require resolution of the replay issue and fresh frozen artifacts.

**Diagnostic figure:** forecast gain on one axis and probability of changing the selected logical class on the other, with logical failures shown separately. A second panel can show exact decision boundaries as the assumed noise field changes.

**Claim boundary:** The existing work supports asking this question. It does not support the headline “better forecasting does not help decoding.”

## 3. Three different meanings of spectral structure

**Best bridge back to FNO, HiPPO, and scientific modeling.** A possible title is “Which Fourier transform does quantum error correction need?”

### The scientific argument

Several distinct mathematical structures were initially bundled together. Separating them creates a much more informative article:

| Structure | Domain | What it describes |
|---|---|---|
| Ring Fourier modes | Circulant index in \(\mathbb Z_\ell\) | Spatial variation of a real-valued field or learned feature over code-block coordinates |
| Temporal frequencies or polynomial memory | Measurement history | Drift timescales, transients, and the history retained by a streaming estimator |
| Boolean Fourier characters | Error configurations in \(\mathbb Z_2^n\) | Parities such as \(\chi_A(e)=(-1)^{\sum_{i\in A}e_i}\), whose expectations carry noise information |

The parity formula in article 1 follows from the expectation of a Boolean character. That does not make it the same transform as an FFT over the lift index. Likewise, complex diagonalization of a real circulant embedding does not solve a binary parity-constrained, logically degenerate inference problem.

There is a legitimate FNO–HiPPO hypothesis: a nuisance channel might have a compact spatial representation and a compact predictive history. In that setting, spatial features and temporal state address different variables. Their combination is useful only if the task contains both kinds of structure, those structures are observable, and the retained information improves the downstream decision. Orthogonality of bases alone establishes none of these requirements. A circulant block coordinate also need not coincide with physical proximity on hardware.

### Closest literature

The original FNO work learns operators between function spaces and demonstrates resolution transfer in PDE settings. It motivates sharing a structured spatial computation, not automatic decoding transfer between different codes. [Li et al., ICLR 2021](https://arxiv.org/abs/2010.08895v3).

HiPPO defines online polynomial representations of history with respect to a chosen measure. The measure determines what history the representation emphasizes. [Gu et al., NeurIPS 2020](https://arxiv.org/abs/2008.07669).

The 2026 *HiPPO Zoo* work makes adaptive, multiscale, associative, and forecasting-oriented memory mechanisms explicit. Its synthetic sequence experiments inspire a more precise question for us: which information does a fixed memory budget preserve under changing drift timescales? The May revision lists ICML 2026 acceptance; it is not a QEC decoder result. [Goffinet et al., 2026](https://arxiv.org/abs/2602.21340v2).

### Small experiment that makes the article ours

First hold the spatial representation fixed. Compare EWMA, a bank of EWMAs, a causal finite window, and HiPPO at matched state bytes and measured update work. Use drift with one timescale, multiple timescales, and abrupt changes. Test on held-out timescales. Report forecast loss, burst-recovery delay, and any decoder benefit separately.

Only then cross CNN and FNO spatial encoders with the best simple memory and HiPPO. Use both smooth fields and deliberately mismatched localized fields. Match parameter budgets or present several size-matched points; the existing screen had roughly five times as many FNO parameters as CNN parameters. Separate a spatial effect, a memory effect, and their interaction rather than calling any hybrid improvement “complementarity.”

**Data and execution:** Reuse the five regimes documented in the [crossed screen](causal-fno-hippo-results.md), with independent test trajectories and adequate burn-in. The linear-memory comparison is a small CPU experiment. Neural comparisons require a training sweep over several seeds. Match discretization, reset conventions, precision, and history availability.

**Diagnostic figure:** performance over a grid of spatial bandwidth and temporal correlation time, including a basis-mismatch control. For the first, simpler article, show memory reconstruction and predictive error as separate panels.

**Claim boundary:** A HiPPO state is not generally a one-MAC-per-state hardware update. Its actual discretized structure determines cost. A change of LP lift and seed is not merely a refinement of the same PDE grid. Cross-code transfer remains untested here.

## 4. Can a worse decoder still provide a valuable candidate?

**Closest continuation of the learned-generator and multipass-consensus discussion.** A useful subtitle is “Search coverage, selection, and the cost of another attempt.”

### The scientific argument

An added generator is useful when it supplies successful logical alternatives that the existing portfolio misses, and a deployable selector can recognize enough of them without introducing excessive failures or cost. Its standalone failure rate is insufficient to answer that question.

For syndrome-valid candidates, distinguish physical-string diversity from logical-class diversity. A collection can contain many strings that all represent the same logical action. Count useful marginal rescues, harmful selections, and genuinely new logical alternatives. An oracle that selects any successful candidate measures available headroom; it cannot be deployed because it uses the actual logical outcome.

### Closest literature

Relay-BP modifies belief propagation with disordered memory strengths and passes information between successive decoding attempts. It can encounter multiple valid corrections. It is a necessary reference for any argument that repeated attempts should outperform a single BP run. Its algorithmic memory is also distinct from HiPPO memory across physical syndrome rounds. [Müller et al., 2025](https://arxiv.org/abs/2506.01779v2).

Multiple-bases BP list decoding is another recent qLDPC approach to obtaining and selecting alternatives. This means neither multiple candidates nor repeated BP attempts is itself a novel contribution. [Rabeti and Mahdavifar, May 2026](https://arxiv.org/abs/2605.14170).

The June 2026 Coset Ensemble Decoder explicitly aggregates logical alternatives in a union-find-based, surface-code-oriented algorithm/hardware design. It is close prior art for coset-level voting, but not a demonstrated LP-code baseline. [Coset Ensemble Decoder, 2026](https://arxiv.org/abs/2606.11076).

### Small experiment that makes the article ours

Freeze a conventional candidate pool, then compare adding a learned proposal with adding the same amount of conventional search. Tune a meaningful Relay-BP baseline. On the exact same new shots, persist every correction, syndrome-validity flag, logical signature, and work counter.

Report the oracle coverage curve first as a diagnostic, then an independently trained and frozen selector's actual curve. Include minimum-cost selection, a fixed cascade, and a simple syndrome-weight stopping rule. Compare matched total work and a common execution resource budget; seven candidates are not an equal-cost budget if one invokes expensive LSD and another uses a few BP iterations.

Consensus needs special care. Correlated restarts are not independent votes, and the frequency of an error string in a heuristic search is not automatically its posterior probability. Family-balanced voting is a useful control for a pool containing many near-duplicate conventional attempts and only one learned arm.

**Data and execution:** Fresh code-capacity shots plus the [portfolio design](superpowers/specs/2026-09-08-anytime-coset-portfolio-design.md), after auditing its baseline settings against current implementations. Begin with a small deterministic pool on CPU. No attention-based controller or reinforcement learning is required to determine whether new candidates add headroom.

**Diagnostic figure:** oracle coverage and deployable selection performance versus measured work, with the incremental learned arm compared to an equal-work conventional addition.

**Claim boundary:** The old inspected test set supplies a hypothesis, not new confirmatory evidence. The frontier opportunity is a reliable, budget-aware interface between generation and selection; this brief does not establish novelty or SOTA for such an interface.

## 5. Can error correction also learn to stabilize the hardware?

**Strongest newly identified frontier connection.** This is the natural place for the earlier world-model intuition, but it requires actions and feedback, not only forecasting.

### Closest literature and why it changes the story

Sivak and colleagues use error-detection events as a learning signal for continuously adjusting control parameters. Their July 2026 *Nature* paper demonstrates this on Willow, including improved stability against injected drift. It changes the object being optimized: the controller acts on the source of errors rather than only changing how a fixed stream is decoded. The custom implementation is proprietary; the paper supplies a mathematical description and public supporting data. [Sivak et al., 2026](https://www.nature.com/articles/s41586-026-10759-2).

An August 2026 preprint by Gong and Hu studies when detector rates provide a locally well-behaved calibration objective, under stated control-error and randomization assumptions, with online optimization guarantees. This is a theoretical and simulated result, not a universal guarantee that minimizing syndrome activity improves every quantum device. [Gong and Hu, 2026](https://arxiv.org/abs/2608.05686v1).

### A newly relevant real dataset

The Willow RL-control data are separate from the earlier below-threshold release. The verified version-2 listing is [Zenodo 18896801](https://zenodo.org/records/18896801), published 6 March 2026. It lists `google_reinforcement_learning_qec.zip`, approximately 7.8 GB, with MD5 `ca54323082fcd0e3671d5b90ce45d85c`; the root README describes its contents. The paper cites the all-version DOI [10.5281/zenodo.17566521](https://doi.org/10.5281/zenodo.17566521). The listing identifies surface-code data collected in 2026 and colour-code data collected in 2025.

The listing and its relation to the paper were checked for this brief. The [archive preview](https://zenodo.org/records/18896801/preview/google_reinforcement_learning_qec.zip?include_deleted=0) also exposes a useful schema: under `color_code_distance_5/traditional_calibration/X/r006`, it lists ideal and noisy Stim circuits, detector events, actual observable flips, metadata, and predicted observable flips from Tesseract with either frequency-calibrated or SI1000 priors. The preview is incomplete; the root README, binary payloads, and full archive have not been read. Do not assume this release contains a complete state/action/reward log, control propensities, or everything required for offline policy evaluation.

This suggests an especially small real-data companion to article 2: after reading the README and metadata, compare the two supplied prediction files against the same actual-observable file within one shard. Count paired logical failures and discordances, then repeat across separately reported round counts and bases. This needs no model training and directly examines the utility of different decoding priors. It is a reanalysis of released predictions, not an independently rerun decoder or a causal estimate of the RL controller's benefit. Do not infer shots or observable counts from byte length alone, or convert per-shot failures to per-round error without an explicit model.

The earlier [below-threshold dataset](https://zenodo.org/records/13273331) remains a different experimental resource. Neither release supplies labels for physical error strings on our LP code. Experimental logical outcomes, when available, are evaluation targets, not decoder input features.

### Small experiment that makes the article ours

Start with a transparent controlled simulator. Let a hidden drift variable \(\theta_i(t)\) change slowly and let action \(a_i(t)\) affect a bounded error probability through a toy relation such as

\[
q_i(t,a)=\operatorname{clip}\{q_{\rm floor}+c[\theta_i(t)-a_i(t)]^2\}.
\]

This is a declared model assumption, not a fitted Willow noise law. Run the same syndrome extraction and decoder under fixed controls, a two-point stochastic calibration method, and a small history-conditioned predictive controller. Equalize observation budgets and action constraints. Use latent state only for an explicitly privileged oracle and evaluation.

A controller cannot infer the sign of a miscalibration from a quadratic error rate alone. Deliberate perturbations reveal information that passive observation lacks. That makes this a compact example of partial observability and the trade-off between learning about a system and controlling it.

**Data and execution:** The first real-data task is an archive-schema audit: look for chronological runs, applied controls, detector records or counts, basis labels, and logical outcomes. Reproduce one descriptive curve only if the fields support it. Use a simulator for counterfactual action experiments; a static dataset cannot establish the effects of actions that were never taken. Stim can simulate a chosen action-dependent Pauli-noise approximation, but cannot on its own validate a coherent pulse-level control model.

**Diagnostic figure:** true drift, chosen controls, detector activity, and logical failures over time, comparing equal-budget controllers. Show where the detector proxy and logical objective disagree.

**Claim boundary:** Our existing observation-only work does not demonstrate control. This direction is a research proposal inspired by the papers, with unusually good continuity from the identifiability study.

## Two shorter technical articles worth keeping

### Reproducibility has three layers

“Same syndrome, different correction: what exactly must a decoder reproduce?” can explain bitwise equality, equality modulo stabilizers, and equality of aggregate performance. These are different contracts. A changed correction need not be a logical failure, but a change in predicted logical signature is not merely harmless stabilizer degeneracy either.

The [LSD paper](https://arxiv.org/abs/2406.18655) supplies the algorithmic background: BP guides localized algebraic post-processing. Our investigation supplies a concrete reason to test reproducibility inside that post-processing, rather than attributing every difference to GPU training seeds. Publish a single input, exact dependency/build information, repeated output hashes, logical signatures, and the smallest intervening code change. Describe pointer-container ordering as a suspected mechanism until changing it removes the observed bifurcation. This should be a precise debugging walkthrough, not an accusation about all LSD implementations.

### What does “real-time neural decoding” measure?

AlphaQubit 2's March 2026 revision reports sub-microsecond-per-cycle decoding on commercial accelerators for specified surface and colour codes. That is important context: broad claims that neural decoding is inherently too slow, or universally tied to one fixed-size implementation, are not a sound opening today. Its results also do not establish arbitrary zero-shot code-family transfer. [Senior et al., AlphaQubit 2](https://arxiv.org/abs/2512.07737v2).

The April 2026 Cascade preprint reports a geometric CNN decoder with strong accuracy and throughput claims on qLDPC benchmarks. It belongs in the reading list before writing as though FNO is the obvious architecture for structured qLDPC problems. Treat its rare-event estimates, noise assumptions, batching, hardware, and confidence use as things to inspect before comparing numbers. [Gu et al., 2026](https://arxiv.org/abs/2604.08358).

An accessible experiment is a queue replay using single-request latency samples. For one serial decoder processing one request every \(T\), waiting work evolves as \(Q_{t+1}=\max(0,Q_t+L_t-T)\). Contrast average service time, latency tails, and burst-induced backlog. State this simple queue model's assumptions; streaming windows, overlapping pipelines, and multiple engines need a richer model. Software queue simulation is not FPGA timing evidence.

## Recommended writing order and first article shape

Start with article 1, then article 2. They have the strongest continuity with the existing scientific work and give the reader tools for evaluating every architectural claim that follows. Article 3 can then explain the FNO–HiPPO hypothesis without requiring it to win. Article 4 covers learned proposals and test-time search. Article 5 opens the control and world-model direction once the dataset schema has been inspected.

For the first article, use this sequence:

1. Begin with the physical question: can measurements designed to detect errors also tell us how the noise is changing?
2. Derive the one-check observation model and verify it with a small experiment.
3. Introduce drift and demonstrate the memory-length trade-off.
4. Explain the independent-sequence and mismatched-history controls using the existing study.
5. End at the unresolved interface: a useful forecast still has to improve an action. Link the current adaptive-decoding and Willow-control papers as two distinct answers to pursue.

An opening in the intended technical voice could read:

> A quantum error-correction cycle produces a stream of parity measurements. The decoder uses them to decide which logical correction to apply, but the same measurements also depend on the noise that produced the errors. That gives us a second inference problem: can we estimate a changing noise process while the code is running?
>
> Our first attempts put learned spatial features and recurrent memory in front of a conventional decoder. The preliminary comparison was too small to establish whether either helped. It left a simpler question unanswered: did the observations contain enough information for a causal predictor to work at all? We built a controlled experiment to answer that question before returning to the architecture.
>
> The useful starting point is a single parity check. Its probability of firing can be written down exactly. From there, we can see what a syndrome tells us, how a memory window filters drifting noise, and why a better estimate is only one part of a better error-correction system.

## A repeatable experiment-to-article loop

For each article, maintain a compact claim record: the question, the minimal model and its assumptions, the strongest feasible baseline, the held-out comparison, the diagnostic that could disconfirm the explanation, and the exact supported conclusion. Separate derivations, our empirical observations, literature results, and proposed extensions in the text.

A small experiment earns its place by making an ambiguity visible. Examples here include correlated checks masquerading as independent data, an accurate forecast that leaves a decision unchanged, many correction strings representing one logical class, and passive data that cannot identify an action's effect. Explaining those distinctions with evidence is what will give the writing an experienced scientific voice.

Avoid claims that the current work does not establish: a threshold, superiority across a code family, quantum-information-bottleneck guarantees from classical regularization, certified distance preservation from a learned fallback, FPGA timing from Python throughput, or zero-shot code-size transfer from the existence of circulant blocks.

## Primary-source reading inventory

The linked sources above are the basis of this brief. This is a nearest-neighbor reading map, not an exhaustive review or a claim of novelty clearance.

| Read for | Source and version/date | Role |
|---|---|---|
| Noise identifiability | [Wagner et al.](https://quantum-journal.org/papers/q-2022-09-19-809/), *Quantum*, September 2022 | Conditions under which syndrome statistics identify noise information |
| Drifting-noise estimation | [Bhardwaj et al.](https://journals.aps.org/prxquantum/abstract/10.1103/z1hc-nqw5), *PRX Quantum*, August 2026 | Closest temporal-estimation reference |
| Decision-oriented predictions | [Mandi et al.](https://proceedings.mlr.press/v162/mandi22a.html), ICML 2022; [Yang et al.](https://ojs.aaai.org/index.php/AAAI/article/view/34891), AAAI 2025 | General ML bridge, not QEC results |
| Spatial operator learning | [Li et al.](https://arxiv.org/abs/2010.08895v3), ICLR 2021 | Original motivation and limits of the PDE analogy |
| Streaming memory | [Gu et al.](https://arxiv.org/abs/2008.07669), NeurIPS 2020; [Goffinet et al.](https://arxiv.org/abs/2602.21340v2), May 2026 revision | Polynomial-history representations and interpretable extensions |
| Strong classical decoding | [LSD](https://arxiv.org/abs/2406.18655), initial preprint June 2024; [Relay-BP](https://arxiv.org/abs/2506.01779v2), August 2025 revision | Algebraic repair and sequential candidate generation |
| Candidate ensembles | [MBBP list decoding](https://arxiv.org/abs/2605.14170), May 2026; [Coset Ensemble Decoder](https://arxiv.org/abs/2606.11076), June 2026 | Prior work limiting broad ensemble-novelty claims |
| Adaptive hardware control | [Sivak et al.](https://www.nature.com/articles/s41586-026-10759-2), *Nature*, July 2026 | Hardware demonstration and new Willow data connection |
| Calibration theory | [Gong and Hu](https://arxiv.org/abs/2608.05686v1), August 2026 preprint | Conditional guarantees, not experimentally established universality |
| Current learned decoding | [AlphaQubit 2](https://arxiv.org/abs/2512.07737v2), March 2026 revision; [Cascade](https://arxiv.org/abs/2604.08358), April 2026 preprint | Current accuracy, scale, confidence, and systems context |
| Experimental records | [Willow RL-control v2](https://zenodo.org/records/18896801); [below-threshold release](https://zenodo.org/records/13273331) | Distinct real-data resources; audit their schemas before promising an experiment |
