# Refit stability protocol

This document makes Experiment A from `QUASAR_IDS.md` identifiable before compute is spent. It is normative for the refit-stability experiment where it is more specific than the earlier sketch.

## Why the naive comparison is not valid

Two artifacts can make raw digit agreement look like semantic stability when it is not.

First, `HierarchicalKMeansEncoder` currently emits the integer labels assigned by `sklearn.cluster.KMeans` as address digits. Cluster labels are exchangeable: two equivalent fits may discover the same partition while numbering its children differently. Raw digit equality therefore understates plain-hkmeans stability unless labels are aligned or the comparison is permutation-invariant.

Second, a quasar signature computed from one fixed observer, one fixed calibration and one fixed anchor set is itself fixed across codebook refits. Counting that fixed prefix as refit stability would make the coarse quasar arm win partly by construction. Experiment A must perturb the estimated reference frame, not merely refit the local tree underneath an unchanged coarse prefix.

The experiment is about reproducibility of a semantic partition under independent estimation, not reproducibility of arbitrary integer labels and not persistence of constants copied unchanged between replicas.

## Unit of replication

Use one fixed observer and a corpus split once into three disjoint roles:

- **anchor-definition set**: fixes the conceptual quasar identities and is shared across all replicas;
- **calibration pool**: used to estimate each quasar representation in the observer space;
- **evaluation set**: never used to fit calibration or hierarchy and is encoded by every replica.

A replica independently samples its calibration subset and its hierarchy-fit subset from the allowed pools using a preregistered seed. Quasar *identity* is shared; quasar *estimated representation* is refit independently. Plain hkmeans receives the same hierarchy-fit sample sizes and seeds.

No evaluation object may participate in calibration, hierarchy fitting or label alignment.

## Arms

### Plain hierarchical k-means

Fit the existing hierarchy independently in each replica. Do not compare its raw child integers across fits.

For diagnostics that require literal paths, align sibling labels between a pair of fits by minimum-cost bipartite matching of centroids at each already-aligned parent, using only fit/calibration data. Report these as **aligned path agreement**, never raw path agreement.

### Quasar-anchored hierarchy

For each replica, estimate the fixed conceptual quasars independently from that replica's calibration sample. Derive the coarse signature from ordinal/normalized relations to those independently estimated anchors, then fit local hierarchical refinement independently inside the resulting regions.

Anchor identities may be canonical labels because their identity is fixed before fitting. Any learned quantizer or local child labels remain exchangeable and require the same alignment/permutation-invariant treatment as plain hkmeans.

### Controls

Run the same protocol with:

- shuffled mapping between conceptual quasar identities and their calibration stimuli;
- random anchors matched in count and calibration sample size;
- plain hkmeans.

A fixed-signature/no-recalibration arm may be reported only as an **upper-bound control**. It is not evidence for refit stability.

## Matching retrieval quality

Stability is compared only at matched retrieval utility.

For each arm and replica, choose depth/probe settings on a validation subset to match a preregistered recall target and candidate-budget tolerance. Freeze those settings before measuring stability on the evaluation set.

Report the achieved recall and candidate fraction beside every stability estimate. If arms cannot be matched within the tolerance, the comparison is inconclusive rather than rescued by comparing unmatched operating points.

## Primary estimand

The primary estimand is the change in **permutation-invariant partition agreement** between independent replicas at the coarse semantic depth:

`delta = agreement(quasar replicas) - agreement(plain-hkmeans replicas)`

Use Adjusted Rand Index (ARI) as the primary partition-agreement statistic because it is invariant to cluster-label permutation and adjusted for chance. Report normalized mutual information (NMI) as a secondary robustness statistic.

Compute the metric for every replica pair on the same held-out evaluation objects, then report the distribution and bootstrap confidence interval over replica pairs/objects. Do not treat object-level rows from the same fitted pair as independent experimental replicas.

## Secondary estimands

After label alignment where necessary, report:

- per-level aligned address agreement;
- longest common semantic span;
- neighborhood consistency among objects sharing each semantic region;
- occupancy entropy/balance;
- achieved recall and candidate fraction.

Raw digit agreement from exchangeable learned cluster labels is forbidden as evidence.

## Minimum replication

Use at least 8 independent replicas per arm and all pairwise replica comparisons, while uncertainty is resampled at the replica level. Seeds are fixed in the result artifact before inspecting outcomes.

If runtime forces fewer than 8 replicas, mark the run pilot-only and do not apply the kill condition.

## Success and kill rule

Experiment A supports continuing the persistent-ID thesis only if, at matched retrieval quality:

1. quasar anchoring improves the primary ARI estimand over plain hkmeans by at least **0.10 absolute**;
2. the 95% bootstrap confidence interval for the replica-level delta excludes zero;
3. the improvement also exceeds shuffled-quasar and random-anchor controls;
4. the effect is visible at the coarse semantic depth rather than arising only after exchangeable local refinement.

If the point estimate is below 0.10, or does not beat the matched controls, Experiment A fails its stability claim. If the confidence interval crosses zero with a point estimate at or above 0.10, the result is inconclusive and may justify one higher-powered rerun with the same protocol, not a changed threshold.

## Result artifact

The executable experiment must write one machine-readable artifact containing at least:

- corpus/split digest;
- observer identifier;
- conceptual quasar-set digest;
- all replica seeds and sampled-set digests;
- calibration method and sample size;
- arm and control definitions;
- retrieval target/tolerance and achieved operating point per replica;
- pairwise ARI/NMI;
- aligned secondary metrics;
- bootstrap procedure and interval;
- preregistered threshold and final decision (`continue`, `kill`, or `inconclusive`).

The artifact, not a narrative summary, is the evidence used by the next round.
