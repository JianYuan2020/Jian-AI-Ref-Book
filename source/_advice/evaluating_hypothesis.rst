Evaluating a Hypothesis
=======================

	When the hypothesis fails to generalize to new examples not in training set.

	Divide the Dataset into:

	* Training set (70%) :math:`(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})`
	* Test set (30%) :math:`(x_{test}^{(1)}, y_{test}^{(1)}), ..., (x_{test}^{(m_{test})}, y_{test}^{(m_{test})})`

	Training/testing procedure for linear regression:

	* Learn parameter :math:`\theta` from training data (minimizing training error :math:`J(\theta)`)
	* Compute test set error:
		:math:`J_{test}(\theta) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (h_\theta (x_{test}^{(i)}) - y_{test}^{(i)})^2`

	Training/testing procedure for logistic regression:

	* Learn parameter :math:`\theta` from training data
	* Compute test set error:
		:math:`J_{test}(\theta) = - \frac{1}{m_{test}} [ \sum_{i=1}^{m_{test}} y_{test}^{(i)} \log(h_\theta (x_{test}^{(i)})) + (1 - y_{test}^{(i)}) \log(1 - h_\theta (x_{test}^{(i)})) ]`
	* Misclassification error (0/1 misclassification error)
