.. _logistic-regression-label:

Logistic Regression
===================
	Classification:
		* Email: Spam / Not Spam?
		* Online Transactions: Fraudulent (Yes / No)?
		* Tumor: Malignant / Benign?

	Here, we have:
		* :math:`y \in` { :math:`0, 1` }
		* :math:`0`: "Negative Class" (e.g., benign tumor)
		* :math:`1`: "Positive Class" (e.g., malignant tumor)

	Often, we could have:
		* :math:`y \in` { :math:`0, 1, 2, 3, 4, ...` }

Threshold Classifier
--------------------
	Threshold classifier output :math:`h_\theta (x)` at :math:`0.5`:

	* If :math:`h_\theta (x) >= 0.5`, predict :math:`y = 1`
	* If :math:`h_\theta (x) < 0.5`, predict :math:`y = 0`

Hypothesis Representation
-------------------------
	For Linear Regression, we have

		* :math:`h_\theta (x) = \theta^{T} x`
		* where :math:`-\infty < h_\theta (x) < \infty`

	Logistic Regression Model

		* Want :math:`0 <= h_\theta (x) <= 1`
		* :math:`h_\theta (x) = g (\theta^{T} x)`

	Let :math:`z = \theta^{T} x`

		* :math:`g(z) = \frac{1}{1 + e^{-z}}`
		* :math:`h_\theta (x) = \frac{1}{1 + e^{-\theta^{T} x}}`
		* This is Sigmoid function (Logistic function)
		* :math:`g(z)` -> :math:`1` as :math:`z` -> :math:`\infty` 
		* :math:`g(z) = 0.5` as :math:`z = 0`
		* :math:`g(z)` -> :math:`0` as :math:`z` -> :math:`-\infty` 

Interpretation of Hypothesis Output
-----------------------------------
	:math:`h_\theta (x)` = estimated probability that :math:`y = 1` on input :math:`x`

	Example:

		* If :math:`x = {\begin{bmatrix}x_{0}\\x_{1}\end{bmatrix}} = {\begin{bmatrix}1\\tumorSize\end{bmatrix}}`
		* :math:`h_\theta (x) = 0.7` interpreted as :math:`y = 1`
		* Tell patient that 70% chance of tumor being malignant

	* :math:`y \in` { :math:`0, 1` }
	* :math:`h_\theta (x) = P(y = 1|x; \theta)` is the probability that :math:`y = 1`, given :math:`x`, parameterized by :math:`\theta`
	* :math:`P(y = 0|x; \theta) + P(y = 1|x; \theta) = 1`
	* :math:`P(y = 0|x; \theta) = 1 - P(y = 1|x; \theta)`

	Or simply

	* :math:`P(y = 0) + P(y = 1) = 1`
	* :math:`P(y = 0) = 1 - P(y = 1)`

Decision Boundary
-----------------
	Logistic Regression

		* :math:`h_\theta (x) = g (\theta^{T} x)` and :math:`z = \theta^{T} x`
		* :math:`g(z) = \frac{1}{1 + e^{-z}}`

	Plot :math:`g(z)` vs. :math:`z` curve, we get:

		* predict :math:`y = 1` if :math:`h_\theta (x) >= 0.5` therefore :math:`z = \theta^{T} x >= 0`
		* predict :math:`y = 0` if :math:`h_\theta (x) < 0.5` therefore :math:`z = \theta^{T} x < 0`

	Solve for :math:`\theta^{T} x >= 0`, we can get linear or non-linear decision boundaries.

Cost Function
-------------
	* Training set, :math:`m` examples: :math:`{ (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)}) }`
	* :math:`x = {\begin{bmatrix}x_{0}\\x_{1}\\...\\x_{n}\end{bmatrix}} \in \mathbb {R^{n + 1}}`, :math:`x_{0} = 1`, :math:`y \in` { :math:`0, 1` }
	* For :math:`h_\theta (x) = \frac{1}{1 + e^{-\theta^{T} x}}`, how to choose parameters :math:`\theta`?
	
	Logistic Regression
	
		* :math:`J(\theta) = \frac{1}{m} \sum_{i=1}^{m} cost(h_\theta (x^{(i)}), y^{(i)})`
		* :math:`Cost(h_\theta (x^{(i)}), y^{(i)}) = \frac{1}{2} (h_\theta (x^{(i)}) - y^{(i)})^2`
		
	Or simply

		* :math:`Cost(h_\theta (x), y) = \frac{1}{2} (h_\theta (x) - y)^2`
		* :math:`h_\theta (x) = \frac{1}{1 + e^{-\theta^{T} x}}`

Logistic Regression Cost Function
---------------------------------
	:math:`Cost(h_\theta (x), y) = {\begin{cases}- \log(h_\theta (x))&y = 1\\- \log(1 - h_\theta (x))&y = 0\end{cases}}`

	Plot :math:`Cost(h_\theta (x), y = 1)` vs. :math:`h_\theta (x)` curve

		* :math:`Cost = 0`, if :math:`h_\theta (x) = 1`
		* But as :math:`h_\theta (x)` -> :math:`0`, :math:`Cost` -> :math:`\infty`
		* Captures intuition that if :math:`h_\theta (x) = 0`, (predict :math:`h_\theta (x) = P(y = 1|x; \theta) = 0`), but :math:`y = 1`, we will penalize the learning algorithm by a very large cost.

	Plot :math:`Cost(h_\theta (x), y = 0)` vs. :math:`h_\theta (x)` curve

		* :math:`Cost = 0`, if :math:`h_\theta (x) = 0`
		* But as :math:`h_\theta (x)` -> :math:`1`, :math:`Cost` -> :math:`\infty`
		* Captures intuition that if :math:`h_\theta (x) = 1`, (predict :math:`h_\theta (x) = P(y = 0|x; \theta) = 1`), but :math:`y = 0`, we will penalize the learning algorithm by a very large cost.

Simplified Cost Function and Gradient Descent
---------------------------------------------
	Cost Function

		* :math:`J(\theta) = \frac{1}{m} \sum_{i=1}^{m} cost(h_\theta (x^{(i)}), y^{(i)})`
		* :math:`Cost(h_\theta (x), y) = {\begin{cases}- \log(h_\theta (x))&y = 1\\- \log(1 - h_\theta (x))&y = 0\end{cases}}`

	Therefore

		* :math:`Cost(h_\theta (x), y) = -y \log(h_\theta (x)) -(1 - y) \log(1 - h_\theta (x))`, :math:`y \in` { :math:`0, 1` }
		* :math:`J(\theta) = - \frac{1}{m} [ \sum_{i=1}^{m} y^{(i)} \log(h_\theta (x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta (x^{(i)})) ]`

	* To fit parameters :math:`\theta`: minimize :math:`J(\theta)`
	* To make a prediction given new: :math:`x`, compute output :math:`h_\theta (x) = \frac{1}{1 + e^{-\theta^{T} x}}`

	Gradient Descent

	TODO: mid week 3

	* :math:`y_{n+1}={\begin{cases}2y_{n}&0\leq y_{n}<{\tfrac {1}{2}}\\2y_{n}-1&{\tfrac {1}{2}}\leq y_{n}<1,\end{cases}}`

	:math:`x = {\begin{bmatrix}1&9&-13\\20&5&-6\end{bmatrix}}`




	Also called Batch Gradient Descent for it's processing all training examples in one batch at every iteration. 

	:math:`\theta_{j} = \theta_{j} - \alpha \frac{\partial }{\partial \theta_{j}} J(\Theta)`

	Repeat for each iteration {

		:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Here :math:`x^{(i)}_{0} = 1`, for every :math:`j = 0, ..., n`

	}

	:math:`\alpha` = :ref:`learning-rate-label`.

Normal Equation
---------------
	Method to solve for :math:`\Theta` analytically.