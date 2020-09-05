.. _better-practices-label:

Better Practices
================

Anomaly Detection vs. Supervised Learning
-----------------------------------------

Anomaly Detection
^^^^^^^^^^^^^^^^^
	* Very small number of positive examples (:math:`y = 1`). (0-20 is common).
	* Large number of negative (:math:`y = 0`) examples.
	* Many different "types" of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like.
	* Future anomalies may look nothing like any of the anomalous examples we've seen so far.

	Examples:

	* Fraud detection
	* Manufacturing (e.g. aircraft engines)
	* Monitoring machines in a data center

Supervised Learning
^^^^^^^^^^^^^^^^^^^
	* Large number of positive and negative examples.
	* Enough positive examples for algorithm to get a sense of what positive examples are like, future positive examples likely to be similar to ones in training set.

	Examples:

	* Email spam classification
	* Weather prediction (sunny/rainy/etc)
	* Cancer classification

Anomaly Detection Original Model vs. Multivariate Gaussian
----------------------------------------------------------

Original Model
^^^^^^^^^^^^^^

	* :math:`p(x) = p(x_{1}; \mu_{1}, \sigma _{1}^{2})` * ... * :math:`p(x_{n}; \mu_{n}, \sigma _{n}^{2})`
	* Manually create features to capture anomalies where :math:`x_{1}, x_{2}` take unusual combinations of values, i.e. :math:`x_{3} = \frac {x_{1}} {x_{2}}`
	* Computationally cheaper (alternatively, scales better to large :math:`n = 10,000`, :math:`n = 100,000`) 
	* OK even if :math:`m` (training set size) is small

Multivariate Gaussian
^^^^^^^^^^^^^^^^^^^^^

	* :math:`p(x; \mu, \Sigma) = \frac {1}{\sqrt {(2\pi)^{n} |\Sigma|}} \exp {(-\frac {1}{2} (x -\mu)^{T} \Sigma^{-1} (x -\mu))}`
	* Automatically captures correlations between features
	* :math:`\Sigma \in \mathbb {R^{nxn}}`, :math:`\Sigma^{-1}` computationally more expensive
	* Must have :math:`m > n`, or else :math:`\Sigma` is non-invertible.
	* Should use :math:`m >= 10 n`
