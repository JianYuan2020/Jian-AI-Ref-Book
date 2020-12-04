
Deciding What to Try Next
=========================

Debugging a Learning Algorithm
------------------------------
	Suppose you have implemented regularized linear regression to predict housing prices.
		:math:`J(\theta) = \frac{1}{2m} [ \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_{j}^2 ]`

	However, when you test your hypothesis on a new set of houses, you find that it makes 
	unacceptably large errors in its predictions. What should you try next?

	* Get more training examples
	* Try smaller sets of features
	* Try getting additional features
	* Try adding polynomial features :math:`(x_{1}^{2}, x_{2}^{2}, x_{1} x_{2}, etc.)`
	* Try decreasing :math:`\lambda`
	* Try increasing :math:`\lambda`

Machine Learning Diagnostic
---------------------------
	Diagnostic: A test that you can run to gain insight what is/isn't working with a learning algorithm, and 
	gain guidance as to how best to improve its performance.

	Diagnostics can take time to implement, but doing so can be a very good use of your time.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   evaluating_hypothesis
   model_selection
   diagnosing_bias_vs_variance

TODO: week 6

	Symbols used in LaTeX markup.

	:math:`\approx`, :math:`\not \approx`, :math:`\simeq`, :math:`\sim`, :math:`\cong` is congruence (一致)

