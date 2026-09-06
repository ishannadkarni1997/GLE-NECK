# Bulk Method

The reference system is a homogeneous bulk liquid with solute species `A` and `B` in an explicit solvent bath. The GLE and GLE-NECK simulations retain only the solute coordinates.

## 1. All-Atom Reference Targets

The explicit-solvent reference model is sampled at equilibrium and under steady external driving. Equilibrium sampling provides:

- retained-solute RDFs for `A-A`, `B-B`, and `A-B`;
- a solvent-solvent `W-W` RDF diagnostic;
- the retained-solute VACF used for memory reconstruction.

Driven NEMD sampling provides the AA mobility curve, measured as solute drift relative to the solvent reference frame.

## 2. Equilibrium Coarse-Graining

The retained-solute RDFs are inverted into tabulated conservative interactions:

```text
U_ij(r) = -kBT log g_ij(r)
```

with short-range regularization and tail shifting. These interactions define the conservative part of the retained GLE.

The equilibrium memory kernel is reconstructed from the normalized AA VACF through a Volterra solve. Plots report lag time in ps and memory in `ps^-2`.

## 3. Baseline GLE

The baseline GLE combines:

- tabulated retained-solute conservative forces;
- the fitted equilibrium memory kernel;
- colored noise consistent with the equilibrium memory;
- external forcing for transport validation.

The baseline is compared with AA RDF and VACF targets before training the non-equilibrium correction.

## 4. Corrective Kernel

GLE-NECK adds a field-conditioned memory correction:

```text
M_total(E, tau) = M_eq(tau) + Delta M(E, tau)
Delta M(E, tau) = E^2 N_theta(E, tau)
```

The `E^2` factor makes the correction vanish at zero field. The bulk multi-point model is trained on `E = 0.5, 1.0, 2.0` and evaluated against the AA mobility curve.
