# Research notes: what failed, what it teaches, and where to go next

Companion to the [README](../README.md). All numbers come from `results/summary*.json`; the diagnostics are
reproducible with `python scripts/diagnose_failures.py` (figure: `results/figures/fig9_diagnosis.png`).

## 1. What did not work — and why

### 1a. Held-out photons did not flag the pasted objects (honest reconstructions)

**Diagnosis: physics-limited, not a failure of the check.** In simulation the true scene is known, so the *expected*
held-out z of every window can be computed — what any test on these photons could show at best ("oracle
detectability"):

| reconstruction | photons / pixel | object windows flaggable by the held-out photons | ... by the 3× larger fit set | actually flagged |
|---|---|---|---|---|
| honest | 2 | **0%** | 4% | 0% |
| honest | 8 | **0%** | 0% | 0% |
| honest | 32 | **0%** | 0% | 0% |
| overconfident (photons × 0.2) | 8 | 12% | **46%** | 19% |
| overconfident (photons × 0.2) | 32 | 0% | 0% | 0% |

At 8–32 photons/pixel the honest posterior got the objects right (median oracle |z| 0.4, the same as elsewhere). At 2
photons/pixel it erased them, but the photons that would reveal this do not exist (median oracle |z| 0.6; even the fit
set could have flagged only 4%). The overconfident model is different in kind: **its own photons held evidence against
46% of its object errors, and it ignored them**.

### 1b. The overconfident model's hallucinations were found globally but not locally

**Diagnosis: its errors are image-wide.** Down-weighting the photons washes out the whole face, so a hallucinated object
is no more wrong *in photon terms* than the rest of the image (median oracle |z| 1.6 on the objects and 1.6
elsewhere). Subtracting each window's neighbourhood median ("local contrast") removes the smooth error and raises the
AUROC for the object windows from 0.46 to **0.67** (8 photons/px) and 0.46 to 0.59 (32 photons/px).

### 1c. "When did it change?" failed with 8 time slices

**Diagnosis: too few held-out photons per space–time cell.** A quarter of the frames split into 8 slices leaves a few
photons per 4×4 window per slice. Pooling in time recovers part of it — at the 99% level:

| time slices × window | covered eye: flagged on it / elsewhere | sticker: on / elsewhere |
|---|---|---|
| 8 × (4×4) | 0% / 0.2% | 0% / 0.7% |
| 4 × (4×4) | 6% / 0.0% | 10% / 1.9% |
| **2 × (4×4)** | **17% / 0.7%** | **21% / 1.0%** |
| 2 × (8×8) | 0% / 2.3% | 0% / 1.3% |

Larger windows *hurt*: they dilute a small change and straddle the moving edges, whose temporal profile carries
posterior uncertainty the statistic ignores (false alarms rise). Caveat: the 2-slice split sits exactly at the true
change time, which is optimistic; an honest change-point scan pays for searching.

### 1d. The sampler was miscalibrated

With 20 Gibbs iterations the 90% intervals covered 60% of the truth (contrast-compressed reconstructions: a chain too
short to leave its grey initialisation); 40 iterations reached 76%; finer prior-step discretisation made it *worse*
(the variance correction then covers a smaller share of the variance). The held-out photons detected this without
ground truth and corrected it.

## 2. Conclusions that go beyond the Lindell group's work

### C1. A detection–correction trade-off: held-out photons cannot catch what an honest posterior would have fixed

With a fraction *f* of the frames held out, the held-out set carries *f* / (1 − *f*) of the fit set's information about
any discrepancy (1/3 at *f* = ¼; z scales by 1/√3). A window error that the held-out photons flag at 3σ was therefore
backed by ≈ 5.2σ — about 13.5 nats — of evidence in the fitted photons. An exact-likelihood posterior keeps such an
error only if its prior favoured the wrong content by more than ~13.5 nats (≈ 700,000 : 1). So:

* for a calibrated prior and an exact likelihood, **detectable hallucinations are essentially self-correcting**: the
  study found 0% flaggable object errors at every photon level;
* held-out photons expose *other* things: the sampler's own miscalibration, errors in the forward model (motion, scene
  changes, sensor), and reconstructions that discard evidence — overconfident priors, approximate samplers,
  feed-forward networks (46% of the overconfident model's object errors were visible to its own photons);
* the split ratio *f* trades verification power against reconstruction quality, with a predictable threshold.

*Beyond the Lindell papers:* Neural Flux Fields and the motion-blur video model reconstruct with strong learned priors
but have no mechanism for telling which outputs the measurements support. C1 says what measurement splitting can and
cannot certify for such models, and predicts when it works (strong priors, approximate inference), which is
directly testable on their pipelines.

### C2. A hierarchy of verifiability: global quantities become verifiable long before local detail

At 2 photons/pixel the per-pixel data share is 0.03 — the face is almost entirely the prior's — yet the held-out frames
decide the motion direction by 28.7 nats (both scenes). The evidence grows roughly in proportion to the photon count
(132.8 nats at 8 photons/px, ×4.6 for ×4 photons). The principle: **the photons needed to verify a quantity scale
inversely with the number of photons that inform it.** One motion parameter is informed by every photon; one pixel by
a handful. In space, the data share crosses ½ only for 8×8 windows at 8–32 photons/px; in time, a change needed 2
slices, not 8.

*Beyond the Lindell papers:* *Inter-Photon-Limited Videography* characterises **recoverability** — which temporal
frequencies a reconstruction can contain — from signal-level theory. This gives the complementary notion of
**verifiability** of a specific reconstruction, per region and per quantity, measured from the photons themselves.
Verification consumes photons, so it is stricter than recoverability: a detail can be recoverable yet unverifiable.

### C3. Photon timing settles the motion ambiguity that generative video models leave open

The motion-blur paper reports that its video diffusion model produces both left-to-right and right-to-left videos and
offers no way to choose. Integrated light cannot distinguish them; single-photon timing can, with ground-truth-free
evidence that scales with the photon count. Extrapolating the linear scaling (28.7 nats at 2 photons/px), the direction
should become genuinely ambiguous (< 5 nats) below about 0.35 photons/pixel for this scene and motion — a concrete,
falsifiable prediction. Below that, a
generative model's chosen direction is a guess, and the method says so.

### C4. Photons as a ground-truth-free judge of reconstructions

* **Calibration:** the held-out residual repaired the posterior's 90% intervals from 34 / 73 / 75% to 77 / 87 / 89%
  (2 / 8 / 32 photons/px; ideal ≈ 87%), and the overconfident model's from 24% to 94%.
* **Model selection:** in 16 of 16 paired reconstructions of the same photons the held-out log score preferred the
  honest model (median margin 290 nats), and its ranking tracked the true PSNR gain (rank correlation 0.79).

This is a validation metric that works on *real* captures, where PSNR cannot be computed. None of the Lindell papers
examined report calibration or a ground-truth-free selection criterion.

### C5. What a generative model leaves unverifiable is structural, not statistical

Content in the sensor's null space — sub-pixel texture under a binned SPAD, colour under a monochrome SPAD — can never
be contradicted (held-out *E*[*z*²] 0.96 and 0.83, as if nothing were wrong) while being wrong (errors 0.29, 0.31). Its
only signature is disagreement among posterior samples. Any honest generative-imaging report must therefore show
the null-space part separately; a single "confidence map" built from measurement consistency cannot cover it.

## 3. Research plan

Ordered by value per GPU-hour on the laptop. Each item states what would confirm or refute it.

| # | question | experiment | cost | confirms / refutes |
|---|---|---|---|---|
| P1 | Does C1's trade-off hold quantitatively? | Vary prior confidence (photon weight β ∈ {1, 0.5, 0.2, 0.1}) × hold-out fraction *f* ∈ {0.1, 0.25, 0.5}; compare the detection rate of object errors with the oracle prediction and the ~13.5-nat threshold | ~2 h | detection should start once the prior's log-odds exceed (1 − *f*)/*f* · t²/2; failure to track it refutes C1 |
| P2 | Can temporal changes be found without knowing when? | Change-point scan over split time × window scale with a permutation-calibrated threshold; cover/sticker at 8 and 32 photons/px | ~30 min GPU | expect detection at 32 photons/px; maps the space–time verification resolution |
| P3 | Can the check avoid wasting photons? | K-fold photon cross-fitting (K = 4): every frame is held out once; display the all-photon reconstruction | 4× sampling | held-out evidence ×√K for borderline errors (the overconfident 8 photons/px case) |
| P4 | Does it survive real photons? | Public colour-SPAD captures (`aRy4n/real-color-SPAD-indoor6`), general image prior, a few frames as the capture, the 6,000-frame sequence as reference | ~1 day | held-out *E*[*z*²] ≈ 1 on real data validates the sensor model; deviations localise hot pixels / crosstalk |
| P5 | Photon-verified video generation | Score samples of a video diffusion model (conditioned on summed photons) with time-sliced held-out frames — the motion-blur paper's setting plus photon timing | lab GPU | C3's predicted ambiguity threshold; which generated videos survive |
| P6 | Theory | Formal detection–correction bound for exponential-family likelihoods under data thinning; optimal *f* | — | a short theory section for a paper |

**Status:** P1 is done (section 3b): all 12 settings are consistent with C1, including 5 of 5 advance predictions.
Next in line: P2 (change-point scan) and P4 (real captures).

## 3b. P1 results: the trade-off holds (all 12 settings; 5 of 5 advance predictions confirmed)

Same 4 scenes with pasted objects, same photons, 8 photons/pixel; the photon weight β sets how much the prior is
over-trusted (1/β) and *f* is the held-out share. "Evidence" is the oracle evidence (nats) the *fitted* photons held
against the model's object errors (90th percentile); the flag line is (1 − *f*)/*f* × 4.5 nats
(`python scripts/analyze_p1.py`, `results/figures/fig10_tradeoff.png`). Settings marked † were predicted in advance
from the first seven.

| prior over-trust | held out | evidence out-voted | flag line | object windows flagged | elsewhere | PSNR |
|---|---|---|---|---|---|---|
| 1× (honest) † | 10% | 0.5 | 40.5 | 0% | 0.4% | 25.6 |
| 2× † | 10% | 4.8 | 40.5 | 0% | 0.1% | 22.5 |
| 5× | 10% | 18.2 | 40.5 | 0% | 0.4% | 16.7 |
| 10× † | 10% | 31.7 | 40.5 | 0% | 12.2% | 12.9 |
| 1× (honest) | 25% | 0.5 | 13.5 | 0% | 0.4% | 24.9 |
| 2× | 25% | 5.4 | 13.5 | 0% | 1.4% | 21.5 |
| 5× | 25% | 13.6 | 13.5 | 19% | 20.0% | 15.6 |
| 10× | 25% | 23.7 | 13.5 | 35% | 38.3% | 12.3 |
| 1× (honest) | 50% | 1.2 | 4.5 | 0% | 4.4% | 23.2 |
| 2× † | 50% | 5.7 | 4.5 | **23%** | 13.9% | 19.2 |
| 5× | 50% | 10.9 | 4.5 | 38% | 43.3% | 13.5 |
| 10× † | 50% | 14.2 | 4.5 | 38% | 55.7% | 11.1 |

* **An honest posterior leaves almost no evidence on the table** (0.5–1.2 nats at every held-out share) — none of its
  object errors is ever flaggable.
* **Over-trusting the prior makes it out-vote more evidence** at every *f* (e.g. 0.5 → 4.8 → 18.2 → 31.7 nats for
  1× → 2× → 5× → 10× at 10% held out).
* **Flags on the objects appear exactly when that evidence passes the split-ratio line**: 0% in all seven settings
  below it, 19–38% in all five above it. The sharpest advance predictions held: 2× over-trust is invisible with a
  quarter of the frames held out but flagged with half (23% on the objects vs 14% elsewhere — here the objects even
  stand out), and 10× over-trust leaves its object errors unflagged with only 10% held out.
* Where the check flags, it reaches the oracle rate (flagged vs flaggable: 19/12, 35/35, 23/19, 38/46, 38/38%).
* Caveats. (i) The line itself is arithmetic (held-out and fitted photons see the same discrepancy in proportion
  to their size); the empirical content is how the out-voted evidence depends on β and *f*, and that the observed flags
  reach the oracle rate. (ii) An over-trusted prior hallucinates the whole face, so its flags are image-wide, not
  specific to the pasted objects — P1 tests the detection of hallucination, not its localisation. (iii) Holding out more
  costs reconstruction quality (24.9 → 23.2 dB for the honest model) and, with more held-out photons, exposes the
  sampler's own residual miscalibration (4.4% background flags at *f* = ½).

## 4. The pitch, in three sentences

Single-photon frames can be split like a train/validation set, so a generative reconstruction can be refereed by
photons it never saw. On a laptop-scale study this calibrates diffusion-model uncertainty without ground truth, picks
the better of two reconstructions every time, and decides which way a scene moved even when the image itself is mostly
the prior's guess. It also exposes a limit worth studying: held-out photons cannot catch hallucinations an exact
posterior would have corrected, so the checks are most powerful exactly where generative pipelines over-trust their
priors.

## Caveats

Four scenes per condition (two for the motion experiment), one prior (FFHQ faces at 64×64), one sampler (PnP-DM), and
a simulated sensor. C1 assumes the sampler targets the true posterior; this one needed photon recalibration to get
there. The oracle analysis uses ground truth and is a simulation-only diagnostic.
