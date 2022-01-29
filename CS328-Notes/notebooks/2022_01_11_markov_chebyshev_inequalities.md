# Markov and Chebyshev Inequalities

Let $X$ be a non-negative random variable and $a>0$ then the probablity that $X$ is atleast $a$ is less than equal to the Expectation of $X$ divided by $a$:

$$P(x \geqslant a) \leqslant \frac{E(x)}{a} \hspace{0.5cm} \text { For any } a>0$$



$$ E(x)=\int_{0}^{\infty} x p(x) d x$$


$$ E(x)=\int_{0}^{a} x p(x) d x+\int_{a}^{\infty} x p(x) d x $$

We have
```{math}
x \geqslant 0, p(x) \geqslant 0  \hspace{1cm}    \text{and} \hspace{1cm}  x \geqslant a, p(x \geqslant a) \hspace{0.5cm} \text{respectively}
```

```{tip}
Markov's inequality is not useful when  $ a < E(x) $
```
