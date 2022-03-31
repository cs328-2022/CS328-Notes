# Chernoff Bound

Let $X_{i}$ be the random variable such that probability of $X_{i}=1$ is $p$ and $X_{i}=0$ with probability $(1-p)$ and all $X_{i}$s are independent of each other.
Let

$$S = \Sigma_{i=1}^{n} X_{i} $$

Then expectation of $S$,   $E[S]= \Sigma_{i=1}^{n} E[X_{i}] = np $

Note

$$E[e^{tX_{i}}] = pe^t + (1-p) = 1 + p(e^t-1) \leq e^{ p(e^t-1)}$$

for any $y$, $1+y \leq e^y$, here $y=1 + p_{i}(e^{t}-1)$

<b> 1. Upper tail </b>

$$ P(S > (1+\delta)np) \leq \Big(\frac{e^\delta}{(1+\delta)^{(1+\delta)}}\Big)^{np}$$

```{math}
:label: property_4
P(S \geq (1+\delta)np) \leq  e^{-\frac{\delta^2}{2+\delta}np}
```

<b>2. Lower tail </b>
```{math}
:label: property_5
P(S\leq(1-\gamma)np) \leq e^{-\gamma^2 \frac{ np}{2}}
```

A schematic plot of upper and lower tail with respect to $\delta$ or $\gamma$ is shown in figure below.

```{figure} ../assets/2022_01_14_chernoff_bound/chernoff_tails.png
---
height: 250px
name: chernoff-tails-fig
---
A Schematic of upper and lower tails of chernoff bound
```

```{note}
The upper tail is bounded by a factor of $e^{-\frac{\delta^2}{2+\delta}np}$, for small $\delta$ it will be proportional to $e^{-\frac{\delta^2}{2}np}$ and for large $\delta$ it is proportional to $e^{-\delta np}$. Therefore the upper tail decreases slowly for large $\delta$. The lower tail always decreases by factor of $e^{-\gamma^2 \frac{ np}{2}}$, which is similar to a Gaussian distribution.
```

````{prf:proof}
Apply Markov's inequality, for any $t> 0$, we have

$$P(S \geq (1+\delta)\mu) = P(e^{tS} \geq e^{t(1+\delta)\mu)} ) \leq \frac{E[e^{tS}]}{e^{t(1+\delta)\mu)}}$$

$$\frac{E[e^{tS}]}{e^{t(1+\delta)\mu}} = \frac{E[e^{t\Sigma_{i=1}^{n} X_{i}]}  }{e^{t(1+\delta)\mu} } = \frac{  E[ \prod_{i=1}^n e^{t X_{i}}]  }{e^{t(1+\delta)\mu} } = \frac{  \prod_{i=1}^n E[e^{t X_{i}}]  }{e^{t(1+\delta)\mu} } \leq \frac{\prod_{i=1}^n e^{ p(e^t-1)} }{e^{t(1+\delta)\mu}} = \frac{ e^{np(e^t-1)} }{e^{t(1+\delta)np}} $$

For any $\delta > 0$, set $t=ln(1+\delta)$

$$ \frac{ e^{np(e^t-1)} }{e^{t(1+\delta)np}} = \frac{ e^{np(1+\delta-1)} }{e^{(1+\delta)ln(1+\delta)np}} = \Big(\frac{ e^{\delta} }{(1+\delta)^{1+\delta}}\Big)^{np} $$

$$ P(S \geq (1+\delta)\mu) \leq  \Big(\frac{ e^{\delta} }{(1+\delta)^{1+\delta}}\Big)^{np} $$

Now taking natural logarithm on the right side of above equation and using the inequality

$$ln(1+x) > \frac{x}{1+x} \geq \frac{x}{1+x/2}$$

$$np(\delta - (1+\delta)ln(1+\delta)) \leq np\Big(\delta - (1+\delta)\frac{\delta}{1+\delta/2}\Big) = np\Big(\frac{(\delta^2 + 2\delta) - (2\delta + 2\delta^2}{2+\delta}\Big) = -\frac{\delta^2}{2+\delta}np$$

Taking exponent on the sides

$$ \Big(\frac{ e^{\delta} }{(1+\delta)^{1+\delta}}\Big)^{np} \leq e^{-\frac{\delta^2}{2+\delta}np} $$

$$ P(S \geq (1+\delta)np) \leq  e^{-\frac{\delta^2}{2+\delta}np} $$

The bound for lower tail can be derived similarly.
````

<footer>
Author(s): Devanshu Thakar, Hardik Mahur, Jayesh Salunkhe
</footer>
