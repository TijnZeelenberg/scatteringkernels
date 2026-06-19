# Recursive equilibrium of the MDN scattering kernel

Companion notes for `analysis/recursive_equilibrium.py` and the figure it produces,
`results/plots/recursive_equilibrium.png`. Two short blocks are written for direct
reuse in the thesis: one for the figure description, one for the discussion.

## What is being plotted

The figure shows what happens when the trained MDN scattering kernel is applied
*recursively*, the way it is during a DSMC simulation, rather than as the single
pre-to-post-collision map it was trained on. Each panel (H$_2$, O$_2$) tracks the
mean relative translational energy fraction
$\eta_\mathrm{trans} = E_\mathrm{rel}/(E_\mathrm{rel}+E_{\mathrm{rot},A}+E_{\mathrm{rot},B})$
of an ensemble of colliding pairs as a function of the mean number of collisions per
molecule. Pairs are repeatedly drawn under the NTC acceptance weighting (collision
probability $\propto g \propto \sqrt{\eta_\mathrm{trans}}$) and their post-collision
state is sampled from the MDN; the collision energy is held at a fixed shell
$E_c \approx 5000$ K, inside the dense CTC training region so that the one-shot fit is
well constrained there. Six trajectories are started from different initial fractions
$\eta_0 \in \{0.15,\dots,0.90\}$. Regardless of where they start, all six collapse onto
a single value — the kernel's **recursive fixed point** $\eta^\ast_\mathrm{MDN}$. The
dashed line marks thermal equipartition, $\eta_\mathrm{trans}=3/7$, which a
microscopically reversible (detailed-balance) kernel must relax to. The trained MDN
instead settles a measurable distance away from $3/7$.

## Interpretation (for the discussion)

The recursive fixed point is the quantity that actually governs the steady state of a
DSMC run, yet it is never part of the one-shot training objective: the loss constrains
the conditional map $p(\eta'\mid\eta)$ pointwise under the training distribution, but
says nothing about the stationary distribution of the iterated operator. A kernel can
therefore reproduce single collisions accurately and still fail detailed balance, and
that small per-collision reversibility error compounds over the millions of collisions
in a relaxation run. The figure confirms this: every starting fraction converges to one
path-independent $\eta^\ast_\mathrm{MDN}\neq 3/7$, so the bias is a property of the
learned kernel itself, not of the initial condition or the transport. For H$_2$ the
fixed point sits below equipartition ($\eta^\ast_\mathrm{MDN}\approx 0.39$ vs. $3/7
\approx 0.429$), i.e. the kernel persistently drains translation into rotation, while
for O$_2$ it sits above ($\eta^\ast_\mathrm{MDN}\approx 0.43$) — the opposite sign. This
matches the direction of the residual the full DSMC relaxation experiments converge to:
H$_2$ settles at $T_\mathrm{trans}\approx 216$ K $< T_\mathrm{rot}\approx 230$ K
($\eta\approx 0.41$) and O$_2$ at $T_\mathrm{trans}\approx 226$ K $> T_\mathrm{rot}
\approx 215$ K ($\eta\approx 0.44$), both off the equipartition value of $220$ K. The
sign of the bias is thus reproduced for both species, which identifies the kernel's
broken detailed balance as the cause of the wrong relaxation endpoint. The magnitudes
do not match exactly (e.g. H$_2$: $0.39$ vs. $0.41$): this single-shell, mean-field
iteration probes the kernel at one fixed energy and re-randomises the rotational
partition each step, whereas DSMC averages the kernel over the full distribution of
collision energies it visits at equilibrium (a lower, energy-averaged shell). The plot
should therefore be read as a *qualitative, mechanism-level* demonstration that the
one-shot-trained kernel carries a sign-definite detailed-balance violation; the precise
endpoint is set by the energy-averaged operating point of the real simulation.
