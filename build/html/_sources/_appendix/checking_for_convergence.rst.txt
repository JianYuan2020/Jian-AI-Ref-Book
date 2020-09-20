.. _checking-for-convergence-label:

Checking for Convergence
========================

Batch Gradient Descent:
^^^^^^^^^^^^^^^^^^^^^^^
	Plot :math:`J_{Train}(\theta)` as a function of the number of iterations of gradient descent.

	:math:`J_{Train}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

Stochastic Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^
	:math:`cost(\theta, (x^{(i)}, y^{(i)})) = \frac{1}{2} (h_\theta (x^{(i)}) - y^{(i)})^2`

	During learning, compute :math:`cost(\theta, (x^{(i)}, y^{(i)}))` before updating :math:`\theta` 
	using :math:`(x^{(i)}, y^{(i)})`.

	Every :math:`1000` iterations (say), plot :math:`cost(\theta, (x^{(i)}, y^{(i)}))` averaged over 
	the last :math:`1000` examples processed by algorithm.

	The plot would be noisy (not as smooth as Batch). Sometimes too noisy to see the convergence. 
	Increasing from :math:`1000` to :math:`5000` could smooth the plot and/or help see the convergence. 
	If the plot is not converge, use smaller :math:`\alpha`.

	:ref:`learning-rate-label` :math:`\alpha` is typically held constant. Can slowly decrease over time if we want 
	:math:`\theta` to converge. (E.g. :math:`\alpha = \frac{const1}{iterationNumber + const2}`), 
	:math:`\alpha` -> :math:`0`
