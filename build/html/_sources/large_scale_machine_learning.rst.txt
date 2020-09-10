.. _large-scale-machine-learning-label:

Large Scale Machine Learning
============================
"It's not who has the best algorithm that wins. It's who has the most data."

Learning with large datasets
----------------------------
	* m = 100,000,000
	* :math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

Linear Regression with Batch Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	:math:`h_{\theta}(x) = \sum_{j=0}^{n} \theta_{j} x_{j}`

	:math:`J_{Train}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

	Repeat for each iteration {

		:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Here :math:`x^{(i)}_{0} = 1`, for every :math:`j = 0, ..., n`

	}

	Here all m training examples are being used (Batch) to learn gradient descent for every :math:`j` for one iteration. 
	When m is very large, i.e. m = 300,000,000, computation becomes very expensive/time consuming. 
	For one iteration, all m training examples need to be computed for every j => (n + 1) times.

Stochastic Gradient Descent
---------------------------
	One Optimization (over the above situation)

	:math:`cost(\theta, (x^{(i)}, y^{(i)})) = \frac{1}{2} (h_\theta (x^{(i)}) - y^{(i)})^2`

	:math:`J_{Train}(\theta) = \frac{1}{m} \sum_{i=1}^{m} cost(\theta, (x^{(i)}, y^{(i)}))`

	:math:`\theta_{j} = \theta_{j} - \alpha \frac{\partial }{\partial \theta_{j}} cost(\theta, (x^{(i)}, y^{(i)})) = \theta_{j} - \alpha (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

	#. Randomly shuffle (reorder) training examples
	#. Repeat for each iteration {

			for i = 1, ..., m {
		
				:math:`\theta_{j} = \theta_{j} - \alpha (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

				Here :math:`x^{(i)}_{0} = 1`, for every :math:`j = 0, ..., n`
	
			} for each training examples :math:`(x^{(1)}, y^{(1)}), ..., (x^{(m)}, y^{(m)})`

		}
	   
	Here one training example is being used to learn gradient descent for every :math:`j`. 
	Repeat the learning over the rest of the training examples. 
	For one iteration, each training examples is computed once.

	The learning parameters are not as accurate as Batch, but this is very efficient and accurate enough. 
	Normally 1 - 10 times of iteration is good enough.
