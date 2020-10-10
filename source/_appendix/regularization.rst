.. _regularization-label:

Regularization
==============
	To solve the problem of overfitting

.. image:: ../_images/under_good_over_fit.png
	:align: center
	:width: 400pt

Under Fitting
-------------
	Replace :math:`x_{i}` with :math:`x_{i} - \mu_{i}` to make features have approximately zero mean
	(Do not apply to :math:`x_{0} = 1`)

Appropriate Fitting
-------------------
	:math:`x_{i} = (x_{i} - \mu_{i}) / s_{i}`

	:math:`\mu_{i}` = average value of :math:`x_{i}`

	:math:`s_{i}` = range, max - min or standard deviation of :math:`x_{i}`
	
Over Fitting
------------
	If we have too many features, the learned hypothesis may fit the training set very well 
	(:math:`J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2` -> :math:`0`), 
	but fail to generalize to new examples (predict prices on new examples).

Addressing Overfitting:
^^^^^^^^^^^^^^^^^^^^^^^
	* :math:`x_{1} =` size of house
	* :math:`x_{2} =` no. of bedrooms
	* :math:`x_{3} =` no. of floors
	* :math:`x_{4} =` age of house
	* :math:`x_{5} =` average income in neighborhood
	* :math:`x_{6} =` kitchen size
	* ...
	* :math:`x_{100}`

Options:
^^^^^^^^
	#. Reduce number of features
		- Manually select which features to keep
		- Model selection algorithm (later in course)

	#. Regularization
		- Keep all the features, but reduce magnitude/values of parameters :math:`\theta_{j}`
		- Works well when we have a lot of features, each of which contributes a bit to predicting :math:`y`

Regularization
^^^^^^^^^^^^^^
	Small values for parameters :math:`\theta_{1}, \theta_{2}, ..., \theta_{n}`
		- "Simpler" hypothesis
		- Less prone to overfitting

	:math:`J(\theta) = \frac{1}{2m} [ \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} (\theta_{j})^2 ]`
		- Exclude :math:`\theta_{0}` for regularization

TODO: mid Week3_2

