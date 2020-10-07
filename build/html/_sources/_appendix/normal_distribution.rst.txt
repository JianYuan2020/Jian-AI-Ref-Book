.. _normal-distribution-label:

Normal Distribution
===================

A Normal (or Gaussian or Gauss or Laplace-Gauss) distribution, Notation :math:`{\mathcal {N}}(\mu ,\sigma ^{2})`, 
is a type of continuous probability distribution for a real-valued random variable. The general form of its 
probability density function is

	:math:`f(x)={\frac {1}{\sigma {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x-\mu }{\sigma }}\right)^{2}}`

Mean
----
	:math:`\mu = \frac{1}{N}\sum_{i=1}^{N} x_{i}`

Variance
--------
	:math:`\sigma^{2} = \frac{1}{N}\sum_{i=1}^{N} (x_{i} - \mu)^2`

	Note: here we use :math:`\frac{1}{N}` instead of :math:`\frac{1}{N - 1}`

Standard Deviation
------------------
	:math:`\sigma = \sqrt {\sigma^{2}} = \sqrt {\frac{1}{N}\sum_{i=1}^{N} (x_{i} - \mu)^2}`

Here :math:`x = \{ x_{1}, x_{2}, ..., x_{i}, ..., x_{N} \}` and :math:`x_{i}, \mu, \sigma^{2}, \sigma \in \mathbb {R}`.

Octave Code
-----------

.. code-block:: octave 

	mu = mean(X, 1)'; % mean all rows for each columns
	sigma2 = var(X, 1, 1)'; % variance all rows (/N) for each columns
