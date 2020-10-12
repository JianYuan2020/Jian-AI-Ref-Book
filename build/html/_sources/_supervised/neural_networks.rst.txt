.. _neural-networks-label:

Neural Networks
===============
	* Origins: Algorithms that try to mimic the brain
	* Was very widely used in 80s and early 90s; popularity diminished in late 90s
	* Recent resurgence: State-of-the-art technique for many applications

Neural Networks: Representation
-------------------------------
	* Non-linear Hypotheses
	* Non-linear Classification
	* Computer Vision: Car detection

Neurons and the Brain
^^^^^^^^^^^^^^^^^^^^^
	The "one learning algorithm" hypothesis

.. |nn1| image:: ../_images/nn_auditory_cortex_learns_to_see.png
	:scale: 70%
.. |nn2| image:: ../_images/nn_somatosensory_cortex_learns_to_see.png
	:scale: 70%

+---------+---------+
|  |nn1|  |  |nn2|  |
+---------+---------+

Model Representation I
----------------------

Neuron and the Brain
^^^^^^^^^^^^^^^^^^^^

.. image:: ../_images/nn_neuron.png
	:align: center

Neuron Model: Logistic Unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_images/nn_input_output.png
	:align: center

Let's define:
	* :math:`a^{(j)}_{i}` = "activation" of unit :math:`i` in layer :math:`j`
	* :math:`\Theta^{(j)}` = matrix of weights controlling function mapping from layer :math:`j` to layer :math:`j + 1`

.. image:: ../_images/nn_input_hidden_output.png
	:align: center

Here:
	* Input layer 1 has 3 units: :math:`x_{1}, x_{2}, x_{3}`; :math:`\Theta^{(1)} \in \mathbb {R^{3*4}}` 
	* Hidden layer 2 has 3 units: :math:`a^{(2)}_{1}, a^{(2)}_{2}, a^{(2)}_{3}`; :math:`\Theta^{(2)} \in \mathbb {R^{4}}` 
	* Output layer 3 has 1 unit: :math:`a^{(3)}_{1} = y = h_{\Theta}(x)`

Add "bias unit", :math:`x_{0} = 1`, compute for Hidden layer 2:
	* :math:`a^{(2)}_{1} = g(\Theta^{(1)}_{10} x_{0} + \Theta^{(1)}_{11} x_{1} + \Theta^{(1)}_{12} x_{2} + \Theta^{(1)}_{13} x_{3})` 
	* :math:`a^{(2)}_{2} = g(\Theta^{(1)}_{20} x_{0} + \Theta^{(1)}_{21} x_{1} + \Theta^{(1)}_{22} x_{2} + \Theta^{(1)}_{23} x_{3})` 
	* :math:`a^{(2)}_{3} = g(\Theta^{(1)}_{30} x_{0} + \Theta^{(1)}_{31} x_{1} + \Theta^{(1)}_{32} x_{2} + \Theta^{(1)}_{33} x_{3})` 

Add "bias unit", :math:`a^{(2)}_{0} = 1`, compute for Output layer 3:
	* :math:`h_{\Theta}(x) = a^{(3)}_{1} = g(\Theta^{(2)}_{10} a^{(2)}_{0} + \Theta^{(2)}_{11} a^{(2)}_{1} + \Theta^{(2)}_{12} a^{(2)}_{2} + \Theta^{(2)}_{13} a^{(2)}_{3})` 

If network has :math:`s_{j}` units in layer :math:`j`, :math:`s_{j+1}` units in layer :math:`j+1`, then :math:`\Theta^{(j)}` 
will be of dimension :math:`s_{j+1}*(s_{j}+1)`. :math:`\Theta^{(j)} \in \mathbb {R^{s_{j+1}*(s_{j}+1)}}`

Model Representation II
-----------------------

Forward Propagation: Vectorized Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	* :math:`a^{(1)} = x = {\begin{bmatrix}x_{0}\\x_{1}\\x_{2}\\x_{3}\end{bmatrix}}`
	* :math:`z^{(2)} = {\begin{bmatrix}z^{(2)}_{1}\\z^{(2)}_{2}\\z^{(2)}_{3}\end{bmatrix}} = \Theta^{(1)} a^{(1)}`
	* :math:`a^{(2)} = g(z^{(2)}) \in \mathbb {R^{3}}`, add :math:`a^{(2)}_{0} = 1`, :math:`a^{(2)} \in \mathbb {R^{4}}`
	* :math:`z^{(3)} = \Theta^{(2)} a^{(2)}`
	* :math:`h_{\Theta}(x) = a^{(3)}_{1} = g(z^{(3)})`

* Neural Network learning its own features
* Other network architectures can have many hidden layers between the input layer and the output layer

.. image:: ../_images/nn_multiple_hidden_layers.png
	:scale: 50%
	:align: center

Multi-class Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^
	Multiple output units: One-vs-all

.. image:: ../_images/nn_multiple_output_units.png
	:align: center
	
TODO: week 5


Hypothesis
----------
	:math:`h_\theta (x) = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{j} x_{j} + ... + \theta_{n} x_{n}`

	Let: :math:`x_{0} = 1` and :math:`x^{(i)}_{0} = 1`:

	:math:`x = [ x_{0}; x_{1}; x_{2}; ...; x_{j}; ...; x_{n} ]` - :math:`(n+1) * 1` column vector.

Parameters
----------
	:math:`\Theta = [ \theta_{0}; \theta_{1}; \theta_{2}; ...; \theta_{j}; ...; \theta_{n} ]` - :math:`(n + 1) * 1` column vector.

	Therefore:

	:math:`h_\theta (x) = \theta_{0} x_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + ... + \theta_{j} x_{j} + ... + \theta_{n} x_{n}`

	:math:`h_\theta (x) = \Theta^{T} x`

Cost Function
-------------
	:math:`J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2`

Gradient Descent
----------------
	Also called Batch Gradient Descent for it's processing all training examples in one batch at every iteration. 

	:math:`\theta_{j} = \theta_{j} - \alpha \frac{\partial }{\partial \theta_{j}} J(\Theta)`

	Repeat for each iteration {

		:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Here :math:`x^{(i)}_{0} = 1`, :math:`j = 0, ..., n`

	}

	:math:`\alpha` = :ref:`learning-rate-label`.

Normal Equation
---------------
	Method to solve for :math:`\Theta` analytically.