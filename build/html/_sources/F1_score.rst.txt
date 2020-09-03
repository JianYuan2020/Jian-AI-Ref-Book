.. _f1-score-label:

:math:`F_{1}` Score
===================

For skewed classes, i.e. only 0.50% of patients have cancer.

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