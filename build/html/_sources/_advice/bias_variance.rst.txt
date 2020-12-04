Bias vs Variance
================
	The bias-variance tradeoff is a central problem in supervised learning. 

	.. image:: ../_images/bias_vs_variance.png
		:scale: 70%
		:align: center

	Training error: :math:`J_{train}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

	Cross validation error: :math:`J_{cv}(\theta) = \frac{1}{2m_{cv}} \sum_{i=1}^{m_{cv}} (h_\theta (x_{cv}^{(i)}) - y_{cv}^{(i)})^2` 
	(or :math:`J_{test}(\theta)`)

	Here is a normal pattern for :math:`J_{train}(\theta), J_{cv}(\theta)` (or :math:`J_{test}(\theta)`) 
	over :math:`d` and therefore an effective way to choose the proper :math:`d`:

	.. image:: ../_images/bias_variance_errors.png
		:scale: 70%
		:align: center

