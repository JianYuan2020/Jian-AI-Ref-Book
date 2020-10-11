.. _stochastic-gradient-descent-label:

Stochastic Gradient Descent
===========================

	One way to optimize the above situation is:

	* :math:`cost(\theta, (x^{(i)}, y^{(i)})) = \frac{1}{2} (h_\theta (x^{(i)}) - y^{(i)})^2`

	* :math:`J_{Train}(\theta) = \frac{1}{m} \sum_{i=1}^{m} cost(\theta, (x^{(i)}, y^{(i)}))`

	* :math:`\theta_{j} = \theta_{j} - \alpha \frac{\partial }{\partial \theta_{j}} cost(\theta, (x^{(i)}, y^{(i)})) = \theta_{j} - \alpha (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

	#. Randomly shuffle (reorder) training examples
	#. Repeat for each iteration {

			for :math:`i = 1, ..., m` {
		
				:math:`\theta_{j} = \theta_{j} - \alpha (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

				Here :math:`x^{(i)}_{0} = 1`, for every :math:`j = 0, ..., n`
	
			} for each training examples :math:`(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})`

		}
	   
	Here :math:`1` training example is being used to learn gradient descent for every :math:`j`. Repeat the learning 
	over the rest of the :math:`m - 1` training examples. For one iteration, each training examples is computed once.

	The learning parameters are not as accurate as Batch, but this is extremely efficient for large dataset and good 
	enough accuracy. Normally after :math:`1 - 10` of iterations, it can reach desired parameters.

Mini-batch Gradient Descent
===========================

	A smoother Stochastic gradient descent.

	* Batch gradient descent: Use all :math:`m` examples in each iteration
	* Stochastic gradient descent: Use :math:`1` example in each iteration
	* Mini-batch gradient descent: Use :math:`b` examples in each iteration

	Say :math:`b = 10, m = 1000`.

	Repeat for each iteration {

			for :math:`i = 1, 11, 21, 31, ..., 991` {
		
				:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{10} \sum_{k=i}^{i + 9} (h_\theta (x^{(k)}) - y^{(k)}) x^{(k)}_{j}`

				Here :math:`x^{(k)}_{0} = 1`, for every :math:`j = 0, ..., n`
	
			} for each training examples :math:`(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})`

	}

	Here :math:`b` training examples are being used (therefore Mini-batch) to learn gradient descent for every :math:`j`. 
	Repeat the learning over the rest of the training examples. For one iteration, each training examples is computed once.
