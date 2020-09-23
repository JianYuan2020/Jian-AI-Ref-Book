.. _feature-scaling-label:

Feature Scaling
===============

	Making sure features are on a similar scale.

	Get every feature into approximately a :math:`-1 <= x_{i} <= 1` range.

Mean Normalization
------------------
	Replace :math:`x_{i}` with :math:`x_{i} - \mu_{i}` to make features have approximately zero mean
	(Do not apply to :math:`x_{0} = 1`).

Feature Scaling
---------------
	:math:`x_{i} = (x_{i} - \mu_{i}) / s_{i}`.

	:math:`\mu_{i}` = average value of :math:`x_{i}`.

	:math:`s_{i}` = range, max - min or standard deviation of :math:`x_{i}`.
	
Octave Code
-----------

.. code-block:: octave 

	mu = mean(X, 1); % mean all rows for each columns
	sigma2 = var(X, 1, 1); % variance all rows (/N) for each columns
	X = (X .- mu) ./ sigma2; % feature scaling
