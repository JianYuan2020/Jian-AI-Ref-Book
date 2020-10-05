.. _learning-rate-label:

Learning Rate
=============

	If learning rate :math:`\alpha` is too small, gradient descent can be slow to converge. If :math:`\alpha` 
	is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge. 

	Gradient descent can converge to a local minimum, even with the learning rate :math:`\alpha` fixed.

	:math:`\theta_{j} = \theta_{j} - \alpha \frac{\partial }{\partial \theta_{j}} J(\Theta)`

	As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to 
	decrease :math:`\alpha` over time.

How to Choose Learning Rate :math:`\alpha`?
-------------------------------------------

	Plot cost function :math:`J(\Theta)` vs. No. of iterations curve. Cost function :math:`J(\Theta)` should 
	decrease after every iteration. Declare convergence if decreases by less than :math:`\epsilon = 10^{-3}` 
	in one iteration.

Summary
-------

	For sufficiently small :math:`\alpha`, :math:`J(\Theta)` should decrease on every iteration. To choose 
	:math:`\alpha`, try: ... :math:`0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1`, ...

	Learning rate :math:`\alpha` is typically held constant. Can slowly decrease over time if we want 
	:math:`\theta` to converge. (E.g. :math:`\alpha = \frac{const1}{iterationNumber + const2}`), 
	:math:`\alpha` -> :math:`0`
