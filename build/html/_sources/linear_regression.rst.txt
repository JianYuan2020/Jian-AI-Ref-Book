.. _choose-linear-regression-label:

Linear Regression
=================

Linear Regression with Multiple Variables/Features.

Let's define:
	* :math:`n` = number of features.
	* :math:`m` = number of training examples.
	* :math:`x^{(i)}` = input (features) of :math:`i^{th}` training example.
	* :math:`x^{(i)}_{j}` = value of feature :math:`j` in :math:`i^{th}` training example.
	* :math:`x^{(i)} = [ x^{(i)}_{1}; x^{(i)}_{2}; ... x^{(i)}_{j}; ... x^{(i)}_{n} ]` - :math:`n * 1` column vector.
	* :math:`X = [ (x^{(1)})^{T}; (x^{(2)})^{T}; ... (x^{(i)})^{T}; ... (x^{(m)})^{T} ]` - :math:`m * n` matrix.

Hypothesis
----------
	:math:`h_\theta (x) = \theta_{0} + \theta_{1} * x_{1} + \theta_{2} * x_{2} + ... + \theta_{j} * x_{j} + ... \theta_{n} * x_{n}`.

	Let: :math:`x_{0} = 1` and :math:`x^{(i)}_{0} = 1`.

	:math:`x = [ x_{0}; x_{1}; x_{2}; ... x_{j}; ... x_{n} ]` - :math:`(n + 1) * 1` column vector.

Parameters
----------
	:math:`\Theta = [ \theta_{0}; \theta_{1}; \theta_{2}; ... \theta_{j}; ... \theta_{n} ]` - :math:`(n + 1) * 1` column vector.

	Therefore:

	:math:`h_\theta (x) = \theta_{0} * x_{0} + \theta_{1} * x_{1} + \theta_{2} * x_{2} + ... + \theta_{j} * x_{j} + ... + \theta_{n} * x_{n}`.

	:math:`h_\theta (x) = \Theta^{T} * x`.

Cost Function
-------------
	:math:`J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`.

Gradient Descent
----------------
	:math:`\theta_{j} = \theta_{j} - \alpha * \frac{\partial }{\partial \theta_{j}} J(\Theta)`.

	:math:`\theta_{j} = \theta_{j} - \alpha * \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) * x^{(i)}_{j}`.

	Note here:

	:math:`\alpha` = learning rate. :ref:`choose-learning-rate-label`.

	:math:`x^{(i)}_{0} = 1`; :math:`j = 0 ... n`; and :math:`\Theta` is a :math:`(n + 1) * 1` column vector.

Normal Equation
---------------
	Method to solve for :math:`\Theta` analytically.