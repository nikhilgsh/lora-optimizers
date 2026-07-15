# EMA Log-Det State: Online Tracking

This note proves a sufficient online-state tracking theorem. The assumptions
are stated on the observed gradient second moments, the drift of their
diagonal-Kronecker factors, factor norms, and the exact EMA tail. No assumption
says that the online state is already close to the target.

## Definitions

Let

$$
w_{u,s}=(1-\beta_2)\beta_2^{u-s},
\qquad
0\le s\le u.
$$

Write $g_s=\operatorname{vec}(G_s)$ and $D(v)=\operatorname{diag}(v)$. For a
positive semidefinite matrix $C$ with $\lambda_{\max}(C)>0$, define

$$
R_\delta(C)=C+\delta\lambda_{\max}(C)I,
\qquad
K_\delta(C)=R_\delta(C)^{-1}.
$$

For nonnegative vectors $p,q$, define

$$
K_A(s;p)=K_\delta(B_s^\top D(p)B_s),
\qquad
K_B(s;q)=K_\delta(A_sD(q)A_s^\top).
$$

The online observations are

$$
\hat q_s(p)
=
\frac1r\operatorname{diag}\!\left(G_{A,s}^\top K_A(s;p)G_{A,s}\right),
\qquad
\hat p_s(q)
=
\frac1r\operatorname{diag}\!\left(G_{B,s}K_B(s;q)G_{B,s}^\top\right).
$$

The raw and normalized EMA vectors are

$$
q_{u+1}=\beta_2q_u+(1-\beta_2)\hat q_u(\bar p_u),
\qquad
p_{u+1}=\beta_2p_u+(1-\beta_2)\hat p_u(\bar q_u),
$$

$$
\bar p_u=N(p_u),
\qquad
\bar q_u=N(q_u),
\qquad
N(v)=\frac{v}{\|v\|_\infty}.
$$

Thus $P_u=D(\bar p_u)$ and $Q_u=D(\bar q_u)$.

The current-rescored coordinate fits are

$$
F^q_u(p)
=
\frac1r\operatorname{diag}\!\left(
  \sum_{s\le u}w_{u,s}G_{A,s}^\top K_A(u;p)G_{A,s}
\right),
$$

$$
F^p_u(q)
=
\frac1r\operatorname{diag}\!\left(
  \sum_{s\le u}w_{u,s}G_{B,s}K_B(u;q)G_{B,s}^\top
\right).
$$

The log-det target is a fixed point of

$$
T_u(p,q)=
\left(
  N(F^p_u(q)),
  N(F^q_u(p))
\right):
$$

$$
(p^\circ_u,q^\circ_u)=T_u(p^\circ_u,q^\circ_u).
$$

## Assumptions

Fix the terminal step $t$.

**A1. Per-step diagonal-Kronecker second moment.** For each $0\le s\le t$,
there are vectors
$p^\star_s\in\mathbb R^{d_{\mathrm{out}}}$,
$q^\star_s\in\mathbb R^{d_{\mathrm{in}}}$, a scalar $\kappa_s\ge0$, and a
symmetric residual $E_s$ such that

$$
g_sg_s^\top
=
\kappa_s\bigl(D(q^\star_s)\otimes D(p^\star_s)\bigr)+E_s.
$$

The factors are normalized and coordinate-positive:

$$
\|p^\star_s\|_\infty=\|q^\star_s\|_\infty=1,
\qquad
c_{\min}\le(p^\star_s)_i,(q^\star_s)_j\le1.
$$

Let

$$
\mathcal D_p=\{p:c_{\min}/2\le p_i\le1\},
\qquad
\mathcal D_q=\{q:c_{\min}/2\le q_j\le1\}.
$$

For positive semidefinite $r\times r$ matrices $M_A,M_B$, define

$$
\lambda_{q,s}(M_A)
=
\frac{\kappa_s}{r}
\operatorname{Tr}\!\left(M_AB_s^\top D(p^\star_s)B_s\right),
$$

$$
\lambda_{p,s}(M_B)
=
\frac{\kappa_s}{r}
\operatorname{Tr}\!\left(M_BA_sD(q^\star_s)A_s^\top\right).
$$

For every $0\le u\le t$,

$$
\kappa_u:=\sum_{s\le u}w_{u,s}\kappa_s\ge\kappa_{\min}>0,
$$

and the weighted residuals are small after the same tests used below. For the
online $q$-side,

$$
\frac{
  \sum_{s\le u}w_{u,s}\|E_s\|_2
  \operatorname{Tr}\!\left(K_A(s;\bar p_s)B_s^\top B_s\right)
}{
  \sum_{s\le u}w_{u,s}\lambda_{q,s}(K_A(s;\bar p_s))
}
\le
\epsilon_E.
$$

For the online $p$-side,

$$
\frac{
  \sum_{s\le u}w_{u,s}\|E_s\|_2
  \operatorname{Tr}\!\left(K_B(s;\bar q_s)A_sA_s^\top\right)
}{
  \sum_{s\le u}w_{u,s}\lambda_{p,s}(K_B(s;\bar q_s))
}
\le
\epsilon_E.
$$

For the current-rescored $q$-side, uniformly over $p\in\mathcal D_p$,

$$
\frac{
  \sum_{s\le u}w_{u,s}\|E_s\|_2
  \operatorname{Tr}\!\left(K_A(u;p)B_s^\top B_s\right)
}{
  \sum_{s\le u}w_{u,s}\lambda_{q,s}(K_A(u;p))
}
\le
\epsilon_E.
$$

For the current-rescored $p$-side, uniformly over $q\in\mathcal D_q$,

$$
\frac{
  \sum_{s\le u}w_{u,s}\|E_s\|_2
  \operatorname{Tr}\!\left(K_B(u;q)A_sA_s^\top\right)
}{
  \sum_{s\le u}w_{u,s}\lambda_{p,s}(K_B(u;q))
}
\le
\epsilon_E.
$$

**A2. Weighted drift of the Kronecker factors.** For every $u$, the following
four weighted drifts are at most $\Delta$.

Online $q$-side:

$$
\frac{
  \sum_{s\le u}w_{u,s}
  \lambda_{q,s}(K_A(s;\bar p_s))
  \|q^\star_s-q^\star_u\|_\infty
}{
  \sum_{s\le u}w_{u,s}
  \lambda_{q,s}(K_A(s;\bar p_s))
}
\le \Delta.
$$

Online $p$-side:

$$
\frac{
  \sum_{s\le u}w_{u,s}
  \lambda_{p,s}(K_B(s;\bar q_s))
  \|p^\star_s-p^\star_u\|_\infty
}{
  \sum_{s\le u}w_{u,s}
  \lambda_{p,s}(K_B(s;\bar q_s))
}
\le \Delta.
$$

Current-rescored $q$-side, uniformly over $p\in\mathcal D_p$:

$$
\frac{
  \sum_{s\le u}w_{u,s}
  \lambda_{q,s}(K_A(u;p))
  \|q^\star_s-q^\star_u\|_\infty
}{
  \sum_{s\le u}w_{u,s}
  \lambda_{q,s}(K_A(u;p))
}
\le \Delta.
$$

Current-rescored $p$-side, uniformly over $q\in\mathcal D_q$:

$$
\frac{
  \sum_{s\le u}w_{u,s}
  \lambda_{p,s}(K_B(u;q))
  \|p^\star_s-p^\star_u\|_\infty
}{
  \sum_{s\le u}w_{u,s}
  \lambda_{p,s}(K_B(u;q))
}
\le \Delta.
$$

These are drift conditions on the observed Kronecker factors. They do not
assume that $\bar p_s,\bar q_s$ are close to $p^\circ_u,q^\circ_u$.

**A3. Factor bounds and initialization.** For all $u\le t$,

$$
a_{\min}\le\|A_u\|_2\le a_{\max},
\qquad
b_{\min}\le\|B_u\|_2\le b_{\max}.
$$

The initial raw EMA vectors $p_0,q_0$ have positive coordinates. This makes
$P_u,Q_u$ well-defined, and their magnitudes enter only through the exact EMA
tail.

Let

$$
\Theta=
\left(
  \beta_2,\delta,r,d_{\mathrm{in}},d_{\mathrm{out}},
  c_{\min},\kappa_{\min},
  a_{\min},a_{\max},b_{\min},b_{\max},
  \|p_0\|_\infty,\|q_0\|_\infty
\right).
$$

Throughout the note, $C(\Theta)$ denotes a positive constant depending only on
the entries of $\Theta$. Its value may change from line to line.

**A4. Small error and small drift.** Assume

$$
\epsilon_E+\Delta\le\epsilon_\star(\Theta).
$$

## Lemma 1: Testing A Step Gives A Ray

For every $s$ and every positive semidefinite $M_A$,

$$
\frac1r\operatorname{diag}\!\left(G_{A,s}^\top M_AG_{A,s}\right)
=
\lambda_{q,s}(M_A)q^\star_s+e_{q,s}(M_A),
$$

where

$$
\|e_{q,s}(M_A)\|_\infty
\le
C(\Theta)\|E_s\|_2
\operatorname{Tr}(M_AB_s^\top B_s).
$$

For every positive semidefinite $M_B$,

$$
\frac1r\operatorname{diag}\!\left(G_{B,s}M_BG_{B,s}^\top\right)
=
\lambda_{p,s}(M_B)p^\star_s+e_{p,s}(M_B),
$$

where

$$
\|e_{p,s}(M_B)\|_\infty
\le
C(\Theta)\|E_s\|_2
\operatorname{Tr}(M_BA_sA_s^\top).
$$

**Proof.** Test A1 against $M_A$:

$$
\frac1r\operatorname{diag}\!\left(G_{A,s}^\top M_AG_{A,s}\right)
=
\lambda_{q,s}(M_A)q^\star_s+e_{q,s}(M_A).
$$

The residual is the same test applied to $E_s$, which gives the displayed
operator-norm bound. The $p$ side is identical.

## Lemma 2: Weighted Tested Sums Follow The Current Shape

For every $u$,

$$
\sum_{s\le u}w_{u,s}\hat q_s(\bar p_s)
=
\Lambda_{q,u}q^\star_u+d_{q,u},
$$

where

$$
\Lambda_{q,u}
=
\sum_{s\le u}w_{u,s}\lambda_{q,s}(K_A(s;\bar p_s)),
\qquad
\|d_{q,u}\|_\infty
\le
C(\Theta)(\epsilon_E+\Delta)\Lambda_{q,u}.
$$

Similarly,

$$
\sum_{s\le u}w_{u,s}\hat p_s(\bar q_s)
=
\Lambda_{p,u}p^\star_u+d_{p,u},
$$

where

$$
\Lambda_{p,u}
=
\sum_{s\le u}w_{u,s}\lambda_{p,s}(K_B(s;\bar q_s)),
\qquad
\|d_{p,u}\|_\infty
\le
C(\Theta)(\epsilon_E+\Delta)\Lambda_{p,u}.
$$

**Proof.** Sum Lemma 1 with
$M_A=K_A(s;\bar p_s)$. The residual terms sum to
$C(\Theta)\epsilon_E\Lambda_{q,u}$ by A1 and the trace comparisons from A3.
The replacement of $q^\star_s$ by $q^\star_u$ costs
$\Delta\Lambda_{q,u}$ by A2. This proves the $q$ side. The $p$ side is the
same.

## Lemma 3: The Log-Det Target Exists And Is Close To The Current Shape

The map $T_u$ has a fixed point in $\mathcal D_p\times\mathcal D_q$. Any fixed
point in this set satisfies

$$
\|p^\circ_u-p^\star_u\|_\infty
+
\|q^\circ_u-q^\star_u\|_\infty
\le
C(\Theta)(\epsilon_E+\Delta).
$$

**Proof.** Repeat Lemma 2 with the current-rescored matrices
$M_A=K_A(u;p)$ and $M_B=K_B(u;q)$. A2 gives the corresponding drift bound
uniformly for $p\in\mathcal D_p$ and $q\in\mathcal D_q$. Hence

$$
N(F^q_u(p))=q^\star_u+O_\Theta(\epsilon_E+\Delta),
\qquad
N(F^p_u(q))=p^\star_u+O_\Theta(\epsilon_E+\Delta)
$$

uniformly on $\mathcal D_p\times\mathcal D_q$. Since the coordinates of
$p^\star_u,q^\star_u$ are at least $c_{\min}$, A4 makes
$T_u(\mathcal D_p\times\mathcal D_q)\subseteq\mathcal D_p\times\mathcal D_q$.
Brouwer's theorem gives a fixed point, and the same display gives the
distance bound.

## Lemma 4: Normalizing A Perturbed Ray

If $\|y\|_\infty=1$ and

$$
z=\lambda y+e+h,
\qquad
\lambda>0,
$$

then, whenever $\|e\|_\infty+\|h\|_\infty\le\lambda/4$,

$$
\|N(z)-y\|_\infty
\le
C\frac{\|e\|_\infty+\|h\|_\infty}{\lambda}.
$$

**Proof.** Since $N(\lambda y)=y$ and $\|z\|_\infty\ge\lambda/2$, the
Lipschitz bound for $N(v)=v/\|v\|_\infty$ gives the claim.

## Theorem: Online EMA Tracks The Log-Det Target

Assume A1--A4. For every $0\le u\le t$,

$$
\|\bar p_{u+1}-p^\circ_u\|_\infty
+
\|\bar q_{u+1}-q^\circ_u\|_\infty
\le
C(\Theta)
\left[
  \epsilon_E+\Delta
  +
  \beta_2^{u+1}
  \bigl(\|p_0\|_\infty+\|q_0\|_\infty\bigr)
\right].
$$

After the displayed EMA tail is at most $\epsilon_E+\Delta$, the right side is
$C(\Theta)(\epsilon_E+\Delta)$.

**Proof.** Unroll the recurrence:

$$
q_{u+1}
=
\beta_2^{u+1}q_0
+
\sum_{s\le u}w_{u,s}\hat q_s(\bar p_s),
$$

$$
p_{u+1}
=
\beta_2^{u+1}p_0
+
\sum_{s\le u}w_{u,s}\hat p_s(\bar q_s).
$$

Lemma 2 gives

$$
q_{u+1}
=
\Lambda_{q,u}q^\star_u+d_{q,u}+\beta_2^{u+1}q_0,
\qquad
\|d_{q,u}\|_\infty
\le
C(\Theta)(\epsilon_E+\Delta)\Lambda_{q,u}.
$$

A1--A3 give $\Lambda_{q,u}\ge C(\Theta)^{-1}$: relative damping gives

$$
K_A(s;\bar p_s)\succeq \frac1{(1+\delta)b_{\max}^2}I,
$$

and A3 gives

$$
\operatorname{Tr}\!\left(B_s^\top D(p^\star_s)B_s\right)
\ge
c_{\min}b_{\min}^2.
$$

Lemma 4 therefore gives

$$
\|\bar q_{u+1}-q^\star_u\|_\infty
\le
C(\Theta)
\left[
  \epsilon_E+\Delta+\beta_2^{u+1}\|q_0\|_\infty
\right].
$$

The same argument gives

$$
\|\bar p_{u+1}-p^\star_u\|_\infty
\le
C(\Theta)
\left[
  \epsilon_E+\Delta+\beta_2^{u+1}\|p_0\|_\infty
\right].
$$

Lemma 3 bounds $(p^\circ_u,q^\circ_u)$ by the same current shape
$(p^\star_u,q^\star_u)$. The triangle inequality proves the theorem.
