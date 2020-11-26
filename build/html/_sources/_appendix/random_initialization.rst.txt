.. _random-initialization-label:

Random Initialization
=====================

Initial Value of :math:`\Theta`
-------------------------------
	For gradient descent and advanced optimization method, need initial value for :math:`\Theta`.

	:math:`optTheta = fminunc(@costFunction, initialTheta, options)`

	Consider gradient descent 
	Set :math:`initialTheta = zeros(n, 1)`?

Zero Initialization
-------------------

.. image:: ../_images/nn_zero_initialization.png
	:align: center
	:width: 400pt
	
Problem
^^^^^^^
	After each update, parameters corresponding to inputs going into each of two hidden units are identical, 
	:math:`a^{(2)}_{1} = a^{(2)}_{2}`.

Random Initialization: Symmetry Breaking
----------------------------------------
	Initialize each :math:`\Theta^{(l)}_{ij}` to a random value in [:math:`-\epsilon, \epsilon`] 
	(i.e. :math:`-\epsilon \leq \Theta^{(l)}_{ij} \leq \epsilon`)

	E.g.

.. code-block:: octave 

	% rand(10, 11) ~ ranom 10x11 matrix (between 0 and 1)
	% [-INIT_EPSILON, INIT_EPSILON]

	Theta1 = rand(10, 11) * (2*INIT_EPSILON) - INIT_EPSILON;
	Theta1 = rand(1, 11) * (2*INIT_EPSILON) - INIT_EPSILON;
