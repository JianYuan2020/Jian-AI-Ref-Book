Supervised learning
===================

:math:`\alpha > \beta`

.. math::

    n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k

Linear Regression with Multiple Variables/Features
--------------------------------------------------

Let's define:
	* :math:`n` = number of features.
	* :math:`m` = number of training examples.
	* :math:`x^{(i)}` = input (features) of :math:`i^{th}` training example.
	* :math:`x^{(i)}_{j}` = value of feature :math:`j` in :math:`i^{th}` training example.
	* :math:`x^{(i)} = [ x^{(i)}_{1}; x^{(i)}_{2}; ... x^{(i)}_{j}; ... x^{(i)}_{n} ]` - :math:`n * 1` column vector.
	* :math:`X = [ (x^{(1)})^{T}; (x^{(2)})^{T}; ... (x^{(i)})^{T}; ... (x^{(m)})^{T} ]` - :math:`m * n` matrix.

1. Hypothesis
^^^^^^^^^^^^^
	* :math:`h_\theta (x) = \theta_{0} + \theta_{1} * x_{1} + \theta_{2} * x_{2} + ... + \theta_{j} * x_{j} + ... \theta_{n} * x_{n}`.
	* Let: :math:`x_{0} = 1` and :math:`x^{(i)}_{0} = 1`.
	* :math:`x = [ x_{0}; x_{1}; x_{2}; ... x_{j}; ... x_{n} ]` - :math:`(n + 1) * 1` column vector.

The Parameters
^^^^^^^^^^^^^^
	* :math:`\Theta = [ \theta_{0}; \theta_{1}; \theta_{2}; ... \theta_{j}; ... \theta_{n} ]` - :math:`(n + 1) * 1` column vector.

Therefore:

* :math:`h_\theta (x) = \theta_{0} * x_{0} + \theta_{1} * x_{1} + \theta_{2} * x_{2} + ... + \theta_{j} * x_{j} + ... + \theta_{n} * x_{n}`.
* :math:`h_\theta (x) = \Theta^{T} * x`.

2. Cost Function
^^^^^^^^^^^^^^^^
* J(Theta) = sum((h_theta(x(i)) - y(i))^2 where i = 1:m)/(2*m)

3. Gradient Descent
^^^^^^^^^^^^^^^^^^^
* theta_j = theta_j - alpha*sum((h_theta(x(i)) - y(i))*x(i)j where i = 1:m)/m
* Note here, x(i)0 = 1; j = 0:n; and Theta is a (n + 1) x 1 column vector;

Unsupervised learning
---------------------

* Clusters and Segmentation