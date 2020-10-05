.. _reinforcement-learning-label:

Reinforcement Learning
======================

Content Based Recommendations
-----------------------------

	Example: Predicting movie ratings. Let's define:

	* :math:`n_{u}` = number of users
	* :math:`n_{m}` = number of movies
	* :math:`n` = number of features for movies
	* :math:`r(i, j)` = 1 if user :math:`j` has rated movie :math:`i` (0 otherwise)
	* :math:`y^{(i, j)}` = rating given by user :math:`j` to movie :math:`i` (defined only if :math:`r(i, j)` = 1)
	* :math:`\theta^{(j)}` = parameter vector for user :math:`j`, :math:`\theta^{(j)} \in \mathbb {R^{n + 1}}`
	* :math:`x^{(i)}` = feature vector for movie :math:`i`, :math:`x^{(i)} \in \mathbb {R^{n + 1}}`
	* For user :math:`j`, movie :math:`i`, predicted rating: :math:`(\theta^{(j)})^{T} (x^{(i)})`

	* These are essentially the same as :ref:`linear-regression-label`.
	* As usual, we can also add the regularization term to prevent the features from becoming too big.

Given :math:`x^{(1)}, ..., x^{(n_{m})}` (and movie ratings), to learn :math:`\theta^{(1)}, ..., \theta^{(n_{u})}`:
------------------------------------------------------------------------------------------------------------------

Cost Function
^^^^^^^^^^^^^

	:math:`J(\Theta) = \frac{1}{2} \sum_{j=1}^{n_{u}} \sum_{i:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)})^2 + 
	\frac{\lambda}{2} \sum_{j=1}^{n_{u}} \sum_{k=1}^{n} (\theta_{k}^{(j)})^2`

Gradient Descent
^^^^^^^^^^^^^^^^

	:math:`\theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha \frac{\partial }{\partial \theta_{k}^{(j)}} J(\Theta)`

	:math:`\theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha \sum_{i:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)}) x^{(i)}_{k}` (for :math:`k = 0`)

	:math:`\theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha (\sum_{i:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)}) x^{(i)}_{k} + \lambda \theta^{(j)}_{k} )` (for :math:`k <> 0`)

Collaborative Filtering
-----------------------

	* Given :math:`x^{(1)}, ..., x^{(n_{m})}` (and movie ratings), to learn :math:`\theta^{(1)}, ..., \theta^{(n_{u})}`
	* Given :math:`\theta^{(1)}, ..., \theta^{(n_{u})}`, can estimate :math:`x^{(1)}, ..., x^{(n_{m})}`
	* Guess :math:`\Theta` -> :math:`x` -> :math:`\Theta` -> :math:`x` -> ...

	.. note:: The nice things with collaborative filtering are:

		* features are learnt automatically
		* k = 0 is not included in the computation

Given :math:`\theta^{(1)}, ..., \theta^{(n_{u})}`, to learn :math:`x^{(1)}, ..., x^{(n_{m})}`:
----------------------------------------------------------------------------------------------

Cost Function
^^^^^^^^^^^^^

	:math:`J(x) = \frac{1}{2} \sum_{i=1}^{n_{m}} \sum_{j:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)})^2 + 
	\frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n} (x_{k}^{(i)})^2`

Gradient Descent
^^^^^^^^^^^^^^^^

	:math:`x_{k}^{(i)} = x_{k}^{(i)} - \alpha \frac{\partial }{\partial x_{k}^{(i)}} J(x)`

	:math:`x_{k}^{(i)} = x_{k}^{(i)} - \alpha (\sum_{j:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)}) \theta^{(j)}_{k} + \lambda x_{k}^{(i)} )` (for :math:`k <> 0`)

Minimizing :math:`x^{(1)}, ..., x^{(n_{m})}` and :math:`\theta^{(1)}, ..., \theta^{(n_{u})}` Simultaneously:
------------------------------------------------------------------------------------------------------------

Cost Function
^^^^^^^^^^^^^

	:math:`J(x^{(1)}, ..., x^{(n_{m})}, \theta^{(1)}, ..., \theta^{(n_{u})}) = \frac{1}{2} \sum_{(i,j):r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)})^2 + 
	\frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n} (x_{k}^{(i)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_{u}} \sum_{k=1}^{n} (\theta_{k}^{(j)})^2`

Gradient Descent
^^^^^^^^^^^^^^^^

	:math:`x_{k}^{(i)} = x_{k}^{(i)} - \alpha (\sum_{j:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)}) \theta^{(j)}_{k} + \lambda x_{k}^{(i)} )`
	:math:`\theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha (\sum_{i:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)}) x^{(i)}_{k} + \lambda \theta^{(j)}_{k} )`

Collaborative Filtering Algorithm
---------------------------------

	#. Initialize :math:`x^{(1)}, ..., x^{(n_{m})}, \theta^{(1)}, ..., \theta^{(n_{u})}` to small random values
	#. Minimize :math:`J(x^{(1)}, ..., x^{(n_{m})}, \theta^{(1)}, ..., \theta^{(n_{u})})` using gradient descent (or an advanced optimization algorithm). E.g. for every :math:`j = 1`, ..., :math:`n_{u}, i = 1`, ..., :math:`n_{m}`:

		:math:`x_{k}^{(i)} = x_{k}^{(i)} - \alpha (\sum_{j:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)}) \theta^{(j)}_{k} + \lambda x_{k}^{(i)} )`
		:math:`\theta_{k}^{(j)} = \theta_{k}^{(j)} - \alpha (\sum_{i:r(i, j)=1}^{} ((\theta^{(j)})^{T} x^{(i)} - y^{(i, j)}) x^{(i)}_{k} + \lambda \theta^{(j)}_{k} )`

	#. For a user with parameters :math:`\theta^{(j)}` and a movie with (learned) features :math:`x^{(i)}`, predict a star rating of :math:`(\theta^{(j)})^{T} x^{(i)}`

Octave Code
-----------

.. code-block:: octave 

	% Cost function
	J = sum(sum(R .* (X*Theta' - Y).^2))/2 + lambda*(sum(sum(Theta.^2)) + sum(sum(X.^2)))/2;

	% Gradient descent
	temp = (R .* (X*Theta' - Y));
	X_grad = temp*Theta + lambda*X;
	Theta_grad = temp'*X + lambda*Theta;

Finding Related Movies
----------------------

	* For each product :math:`i`, we learn a feature vector :math:`x^{(i)} \in \mathbb {R^{n}}`
	* How to find movies related to movie :math:`i`?

		5 most similar movies to movie :math:`i`:

		Find the 5 movies with the smallest :math:`\left\|x^{(i)} - x^{(j)}\right\|`

Implementational Detail: Mean Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	To help with new user with no ratings, for user :math:`j`, on movie :math:`i` predict: :math:`(\theta^{(j)})^{T} (x^{(i)}) + \mu_{i}`
