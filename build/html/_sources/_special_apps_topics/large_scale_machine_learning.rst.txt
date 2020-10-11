.. _large-scale-machine-learning-label:

Large Scale Machine Learning
============================
"It's not who has the best algorithm that wins. It's who has the most data."

Learning with large datasets
----------------------------
	* :math:`m = 300,000,000`
	* :math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

Linear Regression with Batch Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	:math:`h_{\theta}(x) = \sum_{j=0}^{n} \theta_{j} x_{j}`

	:math:`J_{Train}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

	Repeat for each iteration {

		:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Here :math:`x^{(i)}_{0} = 1`, for every :math:`j = 0, ..., n`

	}

	Here all :math:`m` training examples are being used (therefore Batch) to learn gradient descent for every 
	:math:`j` for just one learning iteration. When :math:`m` is very large, i.e. :math:`m = 300,000,000`, 
	computation becomes very expensive and time consuming. 
	
	All :math:`m` training examples need to be computed for every :math:`j` => :math:`m*(n+1)` times for 
	just one learning iteration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   stochastic_gradient_descent
   online_learning
   map_reduce

:ref:`checking-for-convergence-label`

