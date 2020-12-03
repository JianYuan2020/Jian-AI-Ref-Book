Model Selection
===============

	Model selection and training/cross validation/test sets.

Overfitting Example
-------------------
	Once parameters :math:`\theta_{0}, \theta_{1}, ..., \theta_{n}` were fit to some set of data (training 
	set), the error of the parameters as measured on that data (the training error :math:`J(\theta)`) is 
	likely to be lower than the actual generalization error.

Model Selection
---------------
	:math:`d` = degree of polynomial
	
	#. :math:`d = 1` -> :math:`h_{\theta}(x) = \theta_{0} + \theta_{1}x` -> :math:`\Theta^{(1)}` -> :math:`J_{test}(\Theta^{(1)})`
	#. :math:`d = 2` -> :math:`h_{\theta}(x) = \theta_{0} + \theta_{1}x + \theta_{2}x^{2}` -> :math:`\Theta^{(2)}` -> :math:`J_{test}(\Theta^{(2)})`
	
	--

	10. :math:`d = 10` -> :math:`h_{\theta}(x) = \theta_{0} + \theta_{1}x + ... + \theta_{10}x^{10}` -> :math:`\Theta^{(10)}` -> :math:`J_{test}(\Theta^{(10)})`

	Choose :math:`\theta_{0} + \theta_{1}x + ... + \theta_{5}x^{5}` with the min :math:`J_{test}(\Theta^{(i)})` where :math:`i = 1, 2, ..., 10`

	How well does the model generalize? Report test set error :math:`J_{test}(\Theta^{(5)})`

	Problem: :math:`J_{test}(\Theta^{(5)})` is likely to be an optimistic estimate of 
	generalization error. I.e. our extra parameter (:math:`d` = degree of polynomial) is 
	fit to test set.

Divide the Dataset
------------------
	* Training set (60%) :math:`(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})`
	* Cross validation set (20%) :math:`(x_{cv}^{(1)}, y_{cv}^{(1)}), ..., (x_{cv}^{(m_{cv})}, y_{cv}^{(m_{cv})})`
	* Test set (20%) :math:`(x_{test}^{(1)}, y_{test}^{(1)}), ..., (x_{test}^{(m_{test})}, y_{test}^{(m_{test})})`

Training/Cross Validation/Test Error
------------------------------------
	Training error:

		:math:`J_{train}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

	Cross Validation error:

		:math:`J_{cv}(\theta) = \frac{1}{2m_{cv}} \sum_{i=1}^{m_{cv}} (h_\theta (x_{cv}^{(i)}) - y_{cv}^{(i)})^2`

	Test error:

		:math:`J_{test}(\theta) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (h_\theta (x_{test}^{(i)}) - y_{test}^{(i)})^2`

Model Selection
---------------	
	#. :math:`d = 1` -> :math:`h_{\theta}(x) = \theta_{0} + \theta_{1}x` -> :math:`\Theta^{(1)}` -> :math:`J_{cv}(\Theta^{(1)})`
	#. :math:`d = 2` -> :math:`h_{\theta}(x) = \theta_{0} + \theta_{1}x + \theta_{2}x^{2}` -> :math:`\Theta^{(2)}` -> :math:`J_{cv}(\Theta^{(2)})`
	
	--

	10. :math:`d = 10` -> :math:`h_{\theta}(x) = \theta_{0} + \theta_{1}x + ... + \theta_{10}x^{10}` -> :math:`\Theta^{(10)}` -> :math:`J_{cv}(\Theta^{(10)})`

	Choose :math:`\theta_{0} + \theta_{1}x + ... + \theta_{4}x^{4}` with the min :math:`J_{cv}(\Theta^{(i)})` where :math:`i = 1, 2, ..., 10`

	Estimate generalization error for test set :math:`J_{test}(\Theta^{(4)})`
