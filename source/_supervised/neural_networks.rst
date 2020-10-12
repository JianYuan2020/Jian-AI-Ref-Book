.. _neural-networks-label:

Neural Networks
===============
	* Non-linear Hypotheses
	* Non-linear Classification
	* Computer Vision: Car detection

Neurons and the Brain
---------------------
Neural Networks
^^^^^^^^^^^^^^^
	* Origins: Algorithms that try to mimic the brain
	* Was very widely used in 80s and early 90s; popularity diminished in late 90s
	* Recent resurgence: State-of-the-art technique for many applications

	The "one learning algorithm" hypothesis

.. image:: ../_images/nn_auditory_cortex_learns_to_see.png
	:align: center
	:width: 300pt


TODO: week 4

Let's define:
	* :math:`n` = number of features.
	* :math:`m` = number of training examples.
	* :math:`x^{(i)}` = input (features) of :math:`i^{th}` training example.
	* :math:`x^{(i)}_{j}` = value of feature :math:`j` in :math:`i^{th}` training example.
	* :math:`x^{(i)} = [ x^{(i)}_{1}; x^{(i)}_{2}; ...; x^{(i)}_{j}; ...; x^{(i)}_{n} ]` - :math:`n * 1` column vector.
	* :math:`X = [ (x^{(1)})^{T}; (x^{(2)})^{T}; ...; (x^{(i)})^{T}; ...; (x^{(m)})^{T} ]` - :math:`m * n` matrix.

Hypothesis
----------
	:math:`h_\theta (x) = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{j} x_{j} + ... + \theta_{n} x_{n}`

	Let: :math:`x_{0} = 1` and :math:`x^{(i)}_{0} = 1`:

	:math:`x = [ x_{0}; x_{1}; x_{2}; ...; x_{j}; ...; x_{n} ]` - :math:`(n+1) * 1` column vector.

Parameters
----------
	:math:`\Theta = [ \theta_{0}; \theta_{1}; \theta_{2}; ...; \theta_{j}; ...; \theta_{n} ]` - :math:`(n + 1) * 1` column vector.

	Therefore:

	:math:`h_\theta (x) = \theta_{0} x_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{j} x_{j} + ... + \theta_{n} x_{n}`

	:math:`h_\theta (x) = \Theta^{T} x`

Cost Function
-------------
	:math:`J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

Gradient Descent
----------------
	Also called Batch Gradient Descent for it's processing all training examples in one batch at every iteration. 

	:math:`\theta_{j} = \theta_{j} - \alpha \frac{\partial }{\partial \theta_{j}} J(\Theta)`

	Repeat for each iteration {

		:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Here :math:`x^{(i)}_{0} = 1`, :math:`j = 0, ..., n`

	}

	:math:`\alpha` = :ref:`learning-rate-label`.

Normal Equation
---------------
	Method to solve for :math:`\Theta` analytically.