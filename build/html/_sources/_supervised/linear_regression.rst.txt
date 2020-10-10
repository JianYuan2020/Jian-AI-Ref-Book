.. _linear-regression-label:

Linear Regression
=================

	Linear regression with multiple variables/features

Let's define:
	* :math:`n` = number of features
	* :math:`m` = number of training examples
	* :math:`x^{(i)}` = input (features) of :math:`i^{th}` training example
	* :math:`x^{(i)}_{j}` = value of feature :math:`j` in :math:`i^{th}` training example
	* :math:`x^{(i)} = [ x^{(i)}_{1}; x^{(i)}_{2}; ...; x^{(i)}_{j}; ...; x^{(i)}_{n} ] \in \mathbb {R^{n}}`
	* :math:`X = [ (x^{(1)})^{T}; (x^{(2)})^{T}; ...; (x^{(i)})^{T}; ...; (x^{(m)})^{T} ] \in \mathbb {R^{m * n}}`

Hypothesis
----------
	:math:`h_\theta (x) = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{j} x_{j} + ... + \theta_{n} x_{n}`

	Let: :math:`x_{0} = 1` (:math:`x^{(i)}_{0} = 1` for every :math:`i = 1, ..., m`):

	:math:`x = [ x_{0}; x_{1}; x_{2}; ...; x_{j}; ...; x_{n} ] \in \mathbb {R^{n + 1}}`

Parameters
----------
	:math:`\Theta = [ \theta_{0}; \theta_{1}; \theta_{2}; ...; \theta_{j}; ...; \theta_{n} ] \in \mathbb {R^{n + 1}}`

	Therefore:

	:math:`h_\theta (x) = \theta_{0} x_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{j} x_{j} + ... + \theta_{n} x_{n}`

	:math:`h_\theta (x) = \Theta^{T} x \in \mathbb {R}`

Octave Code
-----------

.. code-block:: octave 

	h_theta_x = theta' * x;

Cost Function
-------------
	:math:`J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

	Adding :ref:`regularization-label`:

	:math:`J(\Theta) = \frac{1}{2m} [ \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} (\theta_{j})^2 ]`
		- Exclude :math:`\theta_{0}` for regularization

Gradient Descent
----------------
	Also called Batch Gradient Descent for it's processing all training examples in one batch at every iteration

	:math:`\theta_{j} = \theta_{j} - \alpha \frac{\partial }{\partial \theta_{j}} J(\Theta)`

	Repeat for each iteration {

		:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}` (for every :math:`j = 0, ..., n`)

	}

	:math:`\alpha` = :ref:`learning-rate-label`

:ref:`feature-scaling-label`
----------------------------

Features and Polynomial Regression
----------------------------------

	When adding features: :math:`x^{2}, x^{3}, ...` and/or :math:`x_{1}^{2}, x_{1} x_{2}, x_{2}^{2}, ...` 
	we can extend linear into complex shapes to better fit our training examples.

	Let :math:`x_{n + 1} = x_{1}^{2}, x_{n + 2} = x_{1} x_{2}, x_{n + 3} = x_{2}^{2}, ...` 
	We can continuously call this linear regression.

Normal Equation
---------------
	Method to solve for :math:`\Theta` analytically

		* Set :math:`\frac{\partial }{\partial \theta_{j}} J(\Theta) = 0` (for every :math:`j = 0, ..., n`)
		* Solve for :math:`\theta_{0}, \theta_{1}, \theta_{2}, ..., \theta_{n}`

	Let :math:`X \in \mathbb {R^{m * (n + 1)}}` and :math:`y \in \mathbb {R^{m}}`

		:math:`\theta = (X^{T} X)^{-1} X^{T} y \in \mathbb {R^{n + 1}}`

Octave Code
-----------

.. code-block:: octave 

	theta = pinv(X' * X) * X' * y

Gradient Descent vs. Normal Equation
------------------------------------

	For :math:`m` training examples, :math:`n` features

	Gradient Descent:

	* Need to choose :math:`\alpha`
	* Needs many iterations
	* Works well even when :math:`n` is large (:math:`n = 10^{6}`)

	Normal Equation:

	* No need to choose :math:`\alpha`
	* Don't need to iterate
	* Need to compute :math:`(X^{T} X)^{-1} \in \mathbb {R^{n * n}}`, ~ :math:`O(n^{3})`
	* Slow if :math:`n` is very large, OK with :math:`n = 100; n = 1000`, move to Gradient Descent when :math:`n = 10000`

Non-invertible?
---------------

	What if :math:`X^{T} X` is non-invertible? (singular/degenerate)

	* Redundant features (linearly dependent)

		E.g.
			* :math:`x_{1} =` size in feet
			* :math:`x_{2} =` size in meter -> **need to delete this feature**

	* Too many features (e.g. :math:`m <= n`)

		Delete some features, or use regularization
