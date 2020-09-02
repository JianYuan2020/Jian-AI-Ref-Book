.. _choose-recommender-systems-label:

Recommender Systems
===================

Content Based Recommendations
-----------------------------

	Example: Predicting movie ratings. Let's define:

	* :math:`n_{u}` = number of users.
	* :math:`n_{m}` = number of movies.
	* :math:`r(i, j)` = 1 if user :math:`j` has rated movie :math:`i` (0 otherwise).
	* :math:`y^{(i, j)}` = rating given by user :math:`j` to movie :math:`i` (defined only if :math:`r(i, j)` = 1).
	* :math:`\theta^{(j)}` = parameter vector for user :math:`j`.
	* :math:`x^{(i)}` = feature vector for movie :math:`i`.

	For user :math:`j`, movie :math:`i`, predicted rating: :math:`(\theta^{(j)})^{T} * x^{(i)}`



	These are essentially the same as :ref:`choose-linear-regression-label`.

