.. _anomaly-detection-label:

Anomaly Detection
=================

Anomaly detection can be used for detecting low probability cases:

	* Fraud detections
	* Manufacturing defect detections
	* Monitoring computers in a data center, deteriorating computer detections
	* and more

Given the dataset with the majority data as normal:

	* Dataset: :math:`X = \{ x^{(1)}, x^{(2)}, ..., x^{(i)}, ..., x^{(m)} \}` with :math:`{\displaystyle x^{(i)} \in \mathbb {R^{n}}}`
	* Is :math:`x_{test}` anomalous?
	* Model (probability): :math:`p(x_{test})`
		* :math:`p(x_{test}) < \epsilon`, flag anomaly
		* :math:`p(x_{test}) >= \epsilon` is normal (OK)

Gaussian (Normal) Distribution
------------------------------

Let's assume:

	* Training set: :math:`X = \{ x^{(1)}, x^{(2)}, ..., x^{(i)}, ..., x^{(m)} \}` and :math:`{\displaystyle x^{(i)} \in \mathbb {R^{n}}}`
	* All features are distributed as :ref:`normal-distribution-label`
	* That is: :math:`x_{1}` ~ :math:`{\mathcal {N}}(\mu_{1}, \sigma _{1}^{2})`,  :math:`x_{2}` ~ :math:`{\mathcal {N}}(\mu_{2}, \sigma _{2}^{2})`, ..., :math:`x_{n}` ~ :math:`{\mathcal {N}}(\mu_{n}, \sigma _{n}^{2})` 

Then:

	* :math:`p(x) = p(x_{1}; \mu_{1}, \sigma _{1}^{2})` * :math:`p(x_{2}; \mu_{2}, \sigma _{2}^{2})` * ... * :math:`p(x_{n}; \mu_{n}, \sigma _{n}^{2}) = \Pi_{j=1}^{n} p(x_{j}; \mu_{j}, \sigma _{j}^{2})`

Anomaly Detection Algorithm
---------------------------

	#. Choose features :math:`x_{i}` that you think might be indicative of anomalous examples :math:`\{ x^{(1)}, x^{(2)}, ..., x^{(m)} \}`
	#. Fit parameters :math:`\mu_{1}`, :math:`\mu_{2}`, ..., :math:`\mu_{n}` and :math:`\sigma _{1}^{2}`, :math:`\sigma _{2}^{2}`, ..., :math:`\sigma _{n}^{2}` using :ref:`normal-distribution-label`
	#. Given new example :math:`x`, compute :math:`p(x)`:

		:math:`p(x) = \Pi_{j=1}^{n} p(x_{j}; \mu_{j}, \sigma _{j}^{2}) = {\displaystyle \Pi_{j=1}^{n} {\frac {1}{\sigma_{j} {\sqrt {2\pi }}}}e^{-{\frac {1}{2}}\left({\frac {x_{j}-\mu_{j} }{\sigma_{j} }}\right)^{2}}}`

	Anomaly if :math:`p(x) < \epsilon`

Better Practice
---------------

	* Assume we have some labeled data, of anomalous and non-anomalous examples (:math:`y = 0` if normal, :math:`y = 1` if anomalous)
	* Training set: :math:`x^{(1)}, x^{(2)}, ..., x^{(m)}` (assume normal examples/not anomalous)
	* Cross validation set: :math:`(x_{cv}^{(1)}, y_{cv}^{(1)})`, ..., :math:`(x_{cv}^{(m_{cv})}, y_{cv}^{(m_{cv})})` with some :math:`y = 1` examples
	* Test set: :math:`(x_{test}^{(1)}, y_{test}^{(1)})`, ..., :math:`(x_{test}^{(m_{test})}, y_{test}^{(m_{test})})` with some :math:`y = 1` examples

Specifically
	
	For 10000 good (normal) engines with 20 flawed engines (anomalous):

	* Training set: 6000 good engines
	* CV set: 2000 good engines (:math:`y = 0`), 10 anomalous (:math:`y = 1`)
	* Test set: 2000 good engines (:math:`y = 0`), 10 anomalous (:math:`y = 1`)

It is not a good practice to use CV set + Test set as one set.

Non-gaussian Features
---------------------

	* Each :math:`x_{j}` should be ploted to confirm the Gaussian distribution.
	* For the one that is not, some simple math transformation could fix it. like :math:`x_{1} = \log(x_{1})`, :math:`x_{2} = \sqrt{x_{2}}`, ...

Algorithm Evaluation
--------------------

	* Fit model :math:`p(x)` on training set :math:`\{ x^{(1)}, x^{(2)}, ..., x^{(i)}, ..., x^{(m)} \}`
	* On a cross validation/test example :math:`x`, predict :math:`y = 0` if :math:`p(x) >= \epsilon`, :math:`y = 1` if :math:`p(x) < \epsilon`
	* Possible evaluation metrics:
		- True positive, false positive, false negative, true negative
		- Precision/Recall
		- :math:`F_{1}`-score
	* Can also use cross validation set to choose parameter :math:`\epsilon`

Octave Code
-----------

.. code-block:: octave 

	% Choosing the best epsilon and best F1 score using cross validation set's yval and pval

	bestEpsilon = 0;
	bestF1 = 0;
	F1 = 0;

	stepsize = (max(pval) - min(pval)) / 1000;
	for epsilon = min(pval):stepsize:max(pval)
    
		cvPredictions = (pval < epsilon);
		truePos = sum((cvPredictions == 1) & (yval == 1));
		falsePos = sum((cvPredictions == 1) & (yval == 0));
		falseNeg = sum((cvPredictions == 0) & (yval == 1));
    
		if (truePos != 0)
		  prec = truePos/(truePos + falsePos); % Precision
		  rec = truePos/(truePos + falseNeg); % Recall   
		  F1 = 2*prec*rec/(prec + rec); % F1 score
		endif

		if F1 > bestF1
		   bestF1 = F1;
		   bestEpsilon = epsilon;
		endif

	endfor

