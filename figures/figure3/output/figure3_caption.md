# Figure 3 caption and reading guide

## Caption

**Figure 3 | Hidden design mixing and reviewer payoff.** Both columns use a
standard Normal prior. B1 sets
\(c_p=5/6\) and \(\delta_0=15\); B2 sets \(c_p=5/6\) and
\(\delta_0=20{,}000\). The first row shows the two support designs in each
tolerance-qualified common-cutoff candidate under mixed traditional review
(TR).
The labels \(\pi_L=\lambda\) and \(\pi_H=1-\lambda\) give the population
probabilities of the low and high designs. The second row shows the
cutoff-preserving and reviewer-payoff-preserving pure-TR benchmarks. The third
row shows preregistration review (PR). The final row reports the design
ranking, TR and PR acceptance probabilities, and the reviewer-payoff
difference.

Each smooth curve is the conditional pre-adjustment density
\(f_Z(z\mid d)\), where \(z=\alpha d+\theta\). Curve height does not encode
the mixed-TR probability; the corresponding mixture weight \(\pi\) is printed
beside each mixed-TR curve. The diamond marks the reach boundary
\(c-Y(d)\). The dark shaded baseline mass satisfies
\(z\in(c-Y(d),c]\) and is mapped by strategic adjustment to the cutoff, so it
becomes an atom at \(m=c\) in the submitted-manuscript distribution. The pale
region satisfies \(z>c\), passes naturally, and retains \(m=z\).

Across the two cases, PR raises reviewer payoff in B1 and lowers it in B2. The
design-rigour comparison is \(d^{PR}>d_{pp}^{TR}>d_{cp}^{TR}\) in B1 and
\(d_{pp}^{TR}>d^{PR}>d_{cp}^{TR}\) in B2.

## Common normalization

\[
\alpha=\beta=V=1,\qquad c_a=200,\qquad c_d=\frac{4}{25},
\qquad r=\frac{15}{7}.
\]

## Displayed numerical comparisons

| Case | TR type | TR acceptance | PR acceptance | \(U_R^{PR}-U_R^{TR,mix}\) |
|---|---|---:|---:|---:|
| B1 | mixed | 0.573 | 0.641 | +0.0381 |
| B2 | mixed | 0.603 | 0.632 | -0.0120 |
