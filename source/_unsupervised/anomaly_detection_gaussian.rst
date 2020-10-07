.. _anomaly-gaussian-label:

Gaussian Distribution
=====================

Let's assume:

	* Training set: :math:`X = \{ x^{(1)}, x^{(2)}, ..., x^{(i)}, ..., x^{(m)} \}` and :math:`x^{(i)} \in \mathbb {R^{n}}`
	* All features are distributed as :ref:`normal-distribution-label`
	* That is: :math:`x_{1}` ~ :math:`{\mathcal {N}}(\mu_{1}, \sigma _{1}^{2})`,  :math:`x_{2}` ~ :math:`{\mathcal {N}}(\mu_{2}, \sigma _{2}^{2})`, ..., :math:`x_{n}` ~ :math:`{\mathcal {N}}(\mu_{n}, \sigma _{n}^{2})` 

Then:

	* :math:`p(x) = p(x_{1}; \mu_{1}, \sigma _{1}^{2})` * :math:`p(x_{2}; \mu_{2}, \sigma _{2}^{2})` * ... * :math:`p(x_{n}; \mu_{n}, \sigma _{n}^{2}) = \Pi_{j=1}^{n} p(x_{j}; \mu_{j}, \sigma _{j}^{2})`

Anomaly Detection Algorithm
---------------------------

	#. Choose features :math:`x_{i}` that you think might be indicative of anomalous examples :math:`\{ x^{(1)}, x^{(2)}, ..., x^{(m)} \}`
	#. Fit parameters :math:`\mu_{1}`, :math:`\mu_{2}`, ..., :math:`\mu_{n}` and :math:`\sigma _{1}^{2}`, :math:`\sigma _{2}^{2}`, ..., :math:`\sigma _{n}^{2}` using :ref:`normal-distribution-label`
	#. Given new example :math:`x`, compute :math:`p(x)`:

		:math:`p(x) = \Pi_{j=1}^{n} p(x_{j}; \mu_{j}, \sigma _{j}^{2}) = \Pi_{j=1}^{n} {\frac {1}{\sigma_{j} {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x_{j}-\mu_{j} }{\sigma_{j} }}\right)^{2}}`

	Anomaly if :math:`p(x) < \epsilon`

Note:
^^^^^
	
	Sometimes, i.e. for monitoring computers in a data center case:

	* :math:`x_{3}` = CPU load
	* :math:`x_{4}` = network traffic

	Adding the following will help the detection:

	* :math:`x_{5} = \frac {CPULoad}{networkTraffic}`
	* :math:`x_{6} = \frac {(CPULoad)^2}{networkTraffic}`
