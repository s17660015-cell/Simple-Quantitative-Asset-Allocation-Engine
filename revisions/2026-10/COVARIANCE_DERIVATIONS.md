# Covariance estimation and minimum-variance allocation

This note develops the standard mathematical results underlying the covariance
estimation extension. It makes no claim of original theoretical results.

## Global minimum-variance objective

Maximum Sharpe uses both estimated means and covariance. To isolate the covariance
input, this extension studies the fully invested global minimum-variance problem:

$$\min_{w\in\mathbb R^p}w^T\Sigma w\quad\text{subject to }1^Tw=1.$$

Here Sigma is symmetric positive definite and 1 is the vector of ones. Negative
weights are allowed. This differs from the earlier long-only maximum-Sharpe
problem; clipping negative weights does not generally produce a constrained
optimum. Simulated returns have mean zero because the experiment evaluates risk,
not expected profit or Sharpe.

## Optimal allocation and uniqueness

The Lagrangian is L(w,a)=w^T Sigma w-2a(1^Tw-1). Stationarity gives Sigma w=a1, hence
w=a Sigma inverse 1. Enforcing the budget gives

$$w_* = \frac{\Sigma^{-1}1}{1^T\Sigma^{-1}1},\qquad
v_* = \frac{1}{1^T\Sigma^{-1}1}.$$

The denominator is positive because Sigma inverse is positive definite. It
normalizes the allocation. The implementation solves Sigma v=1 and normalizes v,
avoiding an explicit matrix inverse.

The first-order condition makes all components of Sigma w_* equal. The marginal
variance derivatives, 2 Sigma w_*, are therefore equal across assets. This does
not equalize asset variances or the total risk contributions w_j(Sigma w_*)_j.

Every feasible allocation can be written as w=w_*+h, where 1^Th=0. Thus

$$w^T\Sigma w=v_*+2h^T\Sigma w_*+h^T\Sigma h
=v_*+h^T\Sigma h.$$

The cross term vanishes because Sigma w_*=a1. The remaining quadratic term is
positive for nonzero h. The stationary point is therefore the unique minimum.
The same identity supplies an oracle check for the experiment:

$$\frac{w^T\Sigma w}{v_*}=1+\frac{h^T\Sigma h}{v_*}\geq1.$$


## Shrinkage and eigenvalues

For sample covariance S and m=tr(S)/p, define

$$S_\lambda=(1-\lambda)S+\lambda mI,\qquad0\leq\lambda\leq1.$$

Assume S is positive semidefinite and m>0. Its eigenvectors remain eigenvectors
of S_lambda, with eigenvalues

$$\gamma_i(\lambda)=(1-\lambda)\gamma_i+\lambda m.$$

Every eigenvalue is positive for lambda>0, even if S is singular. This pulls
variances toward one common value and off-diagonal covariances toward zero; it
changes the statistical estimate as well as the numerical problem.

For positive lambda, the spectral condition number is

$$\kappa_2(S_\lambda)=
\frac{(1-\lambda)\gamma_{\max}+\lambda m}
{(1-\lambda)\gamma_{\min}+\lambda m}.$$

Its derivative is

$$\frac{d\kappa_2}{d\lambda}=
\frac{m(\gamma_{\min}-\gamma_{\max})}
{[(1-\lambda)\gamma_{\min}+\lambda m]^2}\leq0.$$

At lambda=1, the condition number is one. A singular S has infinite condition
number at zero; the derivative concerns positive lambda. Better conditioning
helps the solve, but does not prove that the estimated matrix or allocation is
statistically better.


## Better conditioning can increase true risk

Take a perfect estimate S=Sigma=diag(1,9). The allocation is (0.9,0.1), true
variance is 0.9, and the condition number is 9. With lambda=1/2, m=5 and the
shrunk estimate is diag(3,7). Its allocation is (0.7,0.3), with condition number
7/3. Under the original true covariance, those weights have variance:

$$0.7^2+9(0.3)^2=1.30.$$

True risk increases by a factor 13/9, about 1.444, while conditioning improves.
Shrinkage has introduced bias into an already correct estimate.

This idealized example refutes a universal implication. It does not establish
that shrinkage usually harms finite-sample estimates. Repeated sampling will
examine how that bias trades against estimation noise.


## Allocation sensitivity

Let A be Sigma inverse and perturb Sigma by a small symmetric E while retaining
positive definiteness. Differentiating the normalized solution gives

$$\delta w=-AEw_*+w_*\bigl(1^TAEw_*\bigr)+O(\|E\|^2).$$

The first term changes the unnormalized solution; the second restores the budget.
The linear term sums to zero. Small eigenvalues can matter because A
amplifies some perturbations, but direction and normalization also matter.

If E=epsilon Sigma, the two linear terms cancel. GMV weights are exactly invariant
under a positive scalar rescaling of Sigma. Thus the common covariance
scale in the simulation changes absolute variance but not weights or relative risk.


## Connection to the HRP split

For fixed cluster portfolios with variances v_L, v_R and covariance c,

$$V(\alpha)=\alpha^2v_L+(1-\alpha)^2v_R+2\alpha(1-\alpha)c.$$

If the two-portfolio covariance is positive definite, its unconstrained minimizer
is

$$\alpha_* = \frac{v_R-c}{v_L+v_R-2c}.$$

Under 0<=alpha<=1, the optimum is the unconstrained solution clamped to that interval. The HRP
split uses v_R/(v_L+v_R). They agree when c=0 or when v_L=v_R, but not in general.
That local rule is not a proof of global minimum variance.

For v_L=1, v_R=4, c=1.5, HRP assigns 0.8 to the left cluster; the
unconstrained optimum is 1.25 and the long-only optimum is 1. This is a mathematical comparison of rules, not an
empirical ranking.

Sources for method context: [PyPortfolioOpt HRP](https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/hierarchical_portfolio.html)
and [Ledoit and Wolf (2004)](https://www.sciencedirect.com/science/article/pii/S0047259X03000964).
The simulation design uses fixed shrinkage strengths, not fitted Ledoit-Wolf intensity.
