# Quasar-anchored HILDA IDs

## Motivation

HILDA should remain an identifier scheme, not merely an auxiliary ANN structure. The identifier should carry a compact, persistent, multiresolution semantic address that can be compared, indexed and routed with ordinary scalar machinery.

The current hierarchical k-means ablation shows that a hierarchical address can preserve useful locality. The next question is whether the *coarse* part of that address can be made more stable across retraining, corpus growth and embedding observers by anchoring it to a shared semantic reference frame.

The Semantic Atlas programme already proposes **semantic quasars** as macroscopic reference points. This experiment imports only that operational role: quasars are fixed anchors used to define observer-relative position. They are not assumed to have intrinsic physical meaning.

## Core representation

Treat the 128-bit identifier as 32 hexadecimal nibbles. Each semantic nibble chooses one of up to 16 children at a level of a hierarchical partition.

Unlike a conventional prefix tree, the first released HILDA version need not begin at the leftmost nibble. It may occupy an interior interval:

```text
[ future-generalisation ][ current address ][ future-specialisation ][ local identity ]
```

Both sides are semantic headroom:

- **left headroom** is reserved for future, more general classifications discovered above the current root;
- **right headroom** is reserved for future, more specific subdivisions below the current leaves;
- a small terminal identity budget may be retained when two objects can legitimately share the same semantic leaf.

The allocation is intentionally asymmetric. A semantic item can have only a short chain of useful ancestors but potentially a very large descendant/refinement space, so specialisation should receive substantially more capacity than generalisation.

## Monotonicity rule

A released **semantic** nibble is immutable.

A future HILDA model may only:

1. fill previously unassigned nibbles to the left with more general structure;
2. fill previously unassigned nibbles to the right with finer structure;
3. replace nibbles that were explicitly declared to be **jitter**, because jitter is not a semantic assertion;
4. leave unknown levels unassigned.

It must not reinterpret an already emitted semantic nibble.

This is the central compatibility hypothesis. If useful future models require changing old semantic nibbles rather than extending them, the persistent-ID thesis fails.

Equivalently, a later version must consume the previous version as part of its input and preserve every semantic assertion already made. Refinement to the right is a split of an existing cell. Generalisation to the left groups existing cells without repartitioning their contents. A new release must never perform an unconstrained global repartition and silently assign new meanings to old digits.

## Default jitter for unresolved specialisation

HILDA distinguishes three states in the specialisation tail:

1. **semantic** — this nibble is a classification assertion and is immutable;
2. **unassigned** — no value is being asserted at this resolution;
3. **jitter** — provisional entropy deliberately occupying unresolved specialisation capacity without claiming semantic meaning.

**Jitter is the default policy.** When the known semantic path ends and capacity remains, a normal HILDA encoding appends tagged deterministic jitter. The user chooses how many jitter nibbles are desired. A zero-jitter encoding is an explicit opt-out rather than the default state.

Jitter is useful when an application wants dispersion or local uniqueness inside a known semantic region before the ontology has enough resolution to classify that region further. It must never masquerade as semantic precision.

The amount of jitter is therefore an explicit **user-selected parameter** with a non-zero implementation default. The classifier determines the known semantic span; the caller determines how many provisional jitter nibbles follow it, subject to the remaining 128-bit budget. Implementations must expose the selected/default jitter length rather than silently choosing an uninspectable amount.

Conceptually:

```text
[ generalisation headroom ][ known semantic address ][ J ][ jitter... ][ future semantic headroom ][ identity ]
```

where `J` is a control nibble stating that the following payload is provisional jitter rather than classification.

### Jitter framing

The prototype should not reserve one of the 16 semantic child values globally for `J`, because doing so would silently reduce every semantic level from radix 16 to radix 15 and would make a literal semantic value ambiguous with control syntax.

Instead, the 128-bit format must reserve an explicit **control-nibble position or framed control region** whose interpretation is not a cluster decision. One control nibble identifies the presence/mode of jitter. The concrete UUID packing is deferred until the abstract payload grammar is validated.

A minimal abstract grammar is:

```text
semantic-span | jitter-control | jitter-payload | remaining-headroom
```

The jitter control must distinguish at least:

```text
0 = explicit zero-jitter opt-out
1 = default deterministic jitter present
```

Later framing may use additional control values for caller-supplied jitter or other policies without changing the semantic path.

The **length** of jitter should be explicit rather than inferred from random-looking digits. A practical prototype can pair the control nibble with a length nibble (`0..15` jitter nibbles); longer encodings can be considered only if an application demonstrates a need for them.

### Determinism and replacement

Jitter is intentionally outside the semantic monotonicity invariant. A future HILDA version may replace jitter positions with newly discovered semantic refinements while preserving all previously asserted semantic nibbles.

Default jitter itself must be deterministic from stable inputs, for example:

```text
jitter = PRF(namespace, object-fingerprint, hilda-version, user-jitter-length)
```

Thus the default HILDA encoder remains deterministic: the same object, previous HILDA state, namespace, version and jitter length produce the same identifier. The user controls **how much** jitter is used; the implementation controls its deterministic derivation. An API may additionally permit caller-supplied jitter when explicitly requested, but it remains tagged as non-semantic.

This gives a monotone information rule:

```text
future_version = previous_semantic_information + new_semantic_information
```

not:

```text
future_version = reinterpret(previous_semantic_information)
```

Replacing tagged jitter with semantic digits adds knowledge; it does not contradict an earlier classification because jitter never claimed to be classification.

### Jitter must not affect semantic comparisons

Semantic prefix/span comparisons, quasar-region agreement and hierarchy metrics must ignore tagged jitter. Two objects that share the same semantic path but have different jitter remain in the same HILDA semantic region.

Jitter may be useful for physical B-tree dispersion, collision management or application-level identity, but those are separate claims to benchmark. If jitter degrades range locality enough to erase HILDA's retrieval benefit, it should be restricted to positions outside the range-key portion for that application; this changes placement, not the default existence of jitter.

## Quasar anchoring

Let an embedding observer produce `z = O(x)` and let calibrated quasars be `Q = {q_1, ..., q_m}` in that observer's space.

Define a quasar signature from relations between `z` and the anchors. Candidate signatures include:

- ordered nearest-quasar ranks;
- normalized distance ratios;
- coarse Voronoi region;
- ordinal comparisons `d(z,q_i) < d(z,q_j)`;
- a learned quantizer over the vector of normalized anchor distances.

The first experiment should favor **ordinal and normalized** relations over raw coordinates because they are more likely to survive rotations, scale changes and observer-specific calibration.

The quasar signature defines a coarse HILDA region. Hierarchical local quantization then refines inside that region:

```text
object
  -> observer embedding
  -> quasar-relative signature
  -> coarse anchored region
  -> hierarchical local subclusters
  -> HILDA semantic address
  -> default tagged jitter
  -> UUID-compatible 128-bit encoding
```

## Why this is different from plain hierarchical k-means

A plain hierarchical k-means code such as `a3f7` has meaning only with respect to one fitted tree. Refit the tree and the same nibble path can mean something else.

Quasar anchoring attempts to make the most significant *currently known* semantic levels refer to a stable external frame. Local branches may still differ across observers or versions, but their parent region should be more reproducible.

The key measurable claim is therefore not that quasars improve nearest-neighbor recall. It is that they improve **address stability at matched retrieval quality**.

## Cross-observer interpretation

Different embedding models should be treated as semantic observers, not as identical coordinate systems. After calibrating the same conceptual quasars into each observer, the experiment asks whether the observers agree on increasingly fine HILDA levels.

For two observers `A` and `B`, define prefix agreement at semantic depth `d`:

```text
agreement(d) = P[first d assigned semantic nibbles match]
```

More generally, report per-level agreement rather than only complete-prefix agreement, because one observer may preserve coarse structure while disagreeing locally.

The expected pattern, if the Semantic Observers hypothesis is useful here, is:

- high agreement at coarse/general levels;
- lower agreement as the address becomes more specific;
- stronger observers maintaining useful agreement deeper into the address.

Unknown/unassigned nibbles are not padding. They explicitly mean that the current scheme does not claim a classification at that resolution. Tagged jitter is also not a classification and must be excluded from semantic agreement metrics.

## Experiment A — refit stability

Use one observer and one fixed corpus.

1. Select a fixed quasar set before fitting any HILDA tree.
2. Fit multiple HILDA codebooks from independent samples/seeds.
3. Encode the same held-out objects with:
   - plain hierarchical k-means;
   - quasar-anchored hierarchy.
4. At matched recall/candidate budget, measure:
   - per-level address agreement;
   - longest common semantic span;
   - neighborhood consistency among objects sharing each prefix;
   - occupancy balance.

Success means quasar anchoring materially improves stable coarse-address agreement without a large retrieval penalty.

## Experiment B — corpus-growth stability

Fit a codebook and quasar calibration on an initial snapshot, freeze them, and append corpus waves.

Measure at each wave:

- recall/candidate budget;
- prefix occupancy and entropy;
- fraction of new objects routed into previously empty cells;
- address stability of existing objects;
- whether a proposed refinement can extend rightward without changing assigned semantic nibbles;
- how often tagged jitter can be replaced by new semantic refinement without touching older semantic digits.

This connects directly to the incremental-growth experiment: the desired property is cheap append plus rare need to invalidate an address.

## Experiment C — cross-observer stability

Choose at least two independent embedding observers and a shared stimulus corpus.

1. Define the same conceptual quasar set.
2. Calibrate the quasars separately in each observer.
3. Build quasar-anchored HILDA addresses independently.
4. Compare held-out objects level by level.
5. Compare against shuffled-quasar, random-anchor and plain-hkmeans controls.

Do not require raw coordinate alignment. The claim is only that quasar-relative macroscopic relations can provide a more reproducible discrete address.

## Experiment D — jitter semantics and locality

For fixed semantic addresses, vary the user-selected jitter length over a preregistered ladder centered on the implementation default and including an explicit zero-jitter control.

Measure separately:

- collision/duplicate rate for the application identity policy;
- B-tree physical size and insertion cost;
- semantic range-query locality when jitter is inside versus outside the searchable range key;
- deterministic reproducibility of default jitter;
- successful replacement of tagged jitter by later semantic refinements without mutation of older semantic nibbles.

The experiment must not count jitter bits as additional semantic resolution.

## UUID compatibility

The first prototype should model an abstract 128-bit payload before claiming conformance to a particular UUID version. RFC 9562 reserves version/variant semantics that must not be overwritten casually.

A later implementation can evaluate UUIDv8, whose application-defined payload is the natural candidate for a custom semantic identifier. Until then, report semantic-bit budget, control/jitter overhead and UUID framing overhead separately.

## Kill conditions

The quasar-anchored direction should be rejected or narrowed if any of these hold:

1. quasar anchoring does not improve address stability over plain hierarchical codes at matched retrieval quality;
2. coarse address agreement is highly seed- or corpus-dependent;
3. cross-observer agreement does not exceed matched random-anchor controls;
4. incremental growth routinely requires changing already assigned semantic nibbles;
5. the number of bits required for useful semantic structure leaves too little identity entropy for the target applications;
6. jitter cannot be cleanly separated from semantic ordering or makes ordinary range indexing impractical at useful default lengths.

## Near-term decision

Do not optimize query latency first. The next HILDA question is whether a persistent semantic address exists at all:

> Can fixed semantic quasars make the coarse portion of a hierarchical 128-bit identifier reproducible across refits, corpus growth and independent embedding observers while preserving useful retrieval locality?

If yes, later work can optimize storage, indexing, jitter policy and UUID framing. If no, HILDA remains a corpus-local compressed retrieval code rather than a persistent semantic identifier.
