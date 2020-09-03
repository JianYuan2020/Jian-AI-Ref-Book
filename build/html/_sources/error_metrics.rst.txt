.. _error-metrics-label:

Error Metrics
=============

Classification Accuracy
-----------------------
	Of all our predictions, what fraction actually is correct?

	:math:`Accuracy = \frac{numberPredictedCorrectly}{totalPredictionsMade}`

Skewed Classes
--------------
	For skewed classes, i.e. only 0.50% of patients have cancer.
	Classification accuracy is not a good indicator for prediction accuracy.

	:math:`y = 1` in presence of rare class that we want to detect.

	+-------------+---------------------------+
	|             |          Actual           |
	+-------------+-------------+-------------+
	| Predicted   |      1      |      0      |
	+-------------+-------------+-------------+
	|      1      |  True Pos.  |  Fake Pos.  |
	+-------------+-------------+-------------+
	|      0      |  Fake Neg.  |  True Neg.  |
	+-------------+-------------+-------------+

Precision
---------
	Of all patients where we predicted :math:`y = 1`, what fraction actually has cancer?

	:math:`P = \frac{truePos}{predictedPos} = \frac{truePos}{truePos + fakePos}`

Recall
------
	Of all patients that actually have cancer, what fraction did we correctly detect as having cancer?

	:math:`R = \frac{truePos}{actualPos} = \frac{truePos}{truePos + fakeNeg}`

:math:`F_{1}` Score (:math:`F` Score)
-------------------------------------

	:math:`F_{1} Score = 2 \frac{P R}{P + R}`