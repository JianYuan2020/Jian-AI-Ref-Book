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
	Logistic Regression Model

		* Want :math:`0 <= h_\theta (x) <= 1`
		* :math:`h_\theta (x) = g (\Theta^{T} x)`

	Let :math:`z = \Theta^{T} x`

		* :math:`g(z) = \frac{1}{1 + e^{-z}}`
		* :math:`h_\theta (x) = \frac{1}{1 + e^{-\Theta^{T} x}}`
		* This is Sigmoid function (Logistic function)
		* :math:`g(z)` -> :math:`1` as :math:`z` -> :math:`+\infty` 
		* :math:`g(z) = 0.5` as :math:`z = 0`
		* :math:`g(z)` -> :math:`0` as :math:`z` -> :math:`-\infty` 

Interpretation of Hypothesis Output
-----------------------------------
	:math:`h_\theta (x)` = estimated probability that :math:`y = 1` on input :math:`x`

	Example:

	:math:`{\begin{bmatrix}1&9&-13\\20&5&-6\end{bmatrix}}`

	{\begin{bmatrix}1&9&-13\\20&5&-6\end{bmatrix}}


	TODO: starting week 3


Cost Function
-------------
	:math:`J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

Gradient Descent
----------------
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