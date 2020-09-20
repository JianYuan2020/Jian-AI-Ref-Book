.. _anomaly-multivariate-gaussian-label:

Multivariate Gaussian Distribution
==================================

	* Parameters: :math:`x, \mu \in \mathbb {R^{n}}`, :math:`\Sigma \in \mathbb {R^{nxn}}` (covariance matrix)
	* :math:`p(x; \mu, \Sigma) = \frac {1}{\sqrt {(2\pi)^{n} |\Sigma|}} \exp {(-\frac {1}{2} (x -\mu)^{T} \Sigma^{-1} (x -\mu))}`
	* Here :math:`|\Sigma|` is the determinant of :math:`\Sigma`.

Parameter fitting:
------------------

	:math:`\mu = \frac {1}{m} \sum_{i=1}^{m} x^{(i)}`

	:math:`\Sigma = \frac {1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu) (x^{(i)} - \mu)^{T}`

Anomaly detection with the multivariate Gaussian

	#. Fit model :math:`p(x)` by setting :math:`\mu, \Sigma`
	#. Given a new example :math:`x`, compute

		:math:`p(x) = \frac {1}{\sqrt {(2\pi)^{n} |\Sigma|}} \exp {(-\frac {1}{2} (x -\mu)^{T} \Sigma^{-1} (x -\mu))}`

	Flag an anomaly if :math:`p(x) < \epsilon`

Relationship to Original Model
------------------------------

	* Original model: :math:`p(x) = p(x_{1}; \mu_{1}, \sigma _{1}^{2})` * ... * :math:`p(x_{n}; \mu_{n}, \sigma _{n}^{2})`
	* Corresponds to multivariate Gaussian 
	
	:math:`p(x; \mu, \Sigma) = \frac {1}{\sqrt {(2\pi)^{n} |\Sigma|}} \exp {(-\frac {1}{2} (x -\mu)^{T} \Sigma^{-1} (x -\mu))}`

	* where all elements in :math:`\Sigma` is zero except on the diagonal line
