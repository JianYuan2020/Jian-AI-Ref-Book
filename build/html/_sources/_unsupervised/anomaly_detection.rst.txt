.. _anomaly-detection-label:

Anomaly Detection
=================

Anomaly detection can be used for detecting low probability cases:

	* Fraud detections
	* Manufacturing defect detections
	* Monitoring computers in a data center, deteriorating computer detections
	* and more

Given the dataset with the majority data as normal:

	* Dataset: :math:`X = \{ x^{(1)}, x^{(2)}, ..., x^{(i)}, ..., x^{(m)} \}` with :math:`x^{(i)} \in \mathbb {R^{n}}`
	* Is :math:`x_{test}` anomalous?
	* Model (probability): :math:`p(x_{test})`
		* :math:`p(x_{test}) < \epsilon`, flag anomaly
		* :math:`p(x_{test}) >= \epsilon` is normal (OK)

Non-gaussian Features
---------------------

	* Each :math:`x_{j}` should be ploted to confirm the Gaussian distribution.
	* For the one that is not, some simple math transformation could fix it. like :math:`x_{1} = \log(x_{1})`, :math:`x_{2} = \sqrt{x_{2}}`, ...

Better Practice
---------------

	* Assume we have some labeled data, of anomalous and non-anomalous examples (:math:`y = 0` if normal, :math:`y = 1` if anomalous)
	* Training set: :math:`x^{(1)}, x^{(2)}, ..., x^{(m)}` (assume normal examples/not anomalous)
	* Cross validation set: :math:`(x_{cv}^{(1)}, y_{cv}^{(1)})`, ..., :math:`(x_{cv}^{(m_{cv})}, y_{cv}^{(m_{cv})})` with some :math:`y = 1` examples
	* Test set: :math:`(x_{test}^{(1)}, y_{test}^{(1)})`, ..., :math:`(x_{test}^{(m_{test})}, y_{test}^{(m_{test})})` with some :math:`y = 1` examples

Specifically
------------
	
	For 10000 good (normal) engines with 20 flawed engines (anomalous):

	* Training set: 6000 good engines
	* CV set: 2000 good engines (:math:`y = 0`), 10 anomalous (:math:`y = 1`)
	* Test set: 2000 good engines (:math:`y = 0`), 10 anomalous (:math:`y = 1`)

It is not a good practice to use CV set + Test set as one set.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   anomaly_detection_gaussian
   anomaly_detection_multivariate_gaussian

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

Original Model vs. Multivariate Gaussian
----------------------------------------

Original Model
^^^^^^^^^^^^^^

	* :math:`p(x) = p(x_{1}; \mu_{1}, \sigma _{1}^{2})` * ... * :math:`p(x_{n}; \mu_{n}, \sigma _{n}^{2})`
	* Manually create features to capture anomalies where :math:`x_{1}, x_{2}` take unusual combinations of values, i.e. :math:`x_{3} = \frac {x_{1}} {x_{2}}`
	* Computationally cheaper (alternatively, scales better to large :math:`n = 10,000`, :math:`n = 100,000`) 
	* OK even if :math:`m` (training set size) is small

Multivariate Gaussian
^^^^^^^^^^^^^^^^^^^^^

	* :math:`p(x; \mu, \Sigma) = \frac {1}{\sqrt {(2\pi)^{n} |\Sigma|}} \exp {(-\frac {1}{2} (x -\mu)^{T} \Sigma^{-1} (x -\mu))}`
	* Automatically captures correlations between features
	* :math:`\Sigma \in \mathbb {R^{nxn}}`, :math:`\Sigma^{-1}` computationally more expensive
	* Must have :math:`m > n`, or else :math:`\Sigma` is non-invertible.
	* We use :math:`m >= 10 n`
