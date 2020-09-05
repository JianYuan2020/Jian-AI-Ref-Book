.. _better-practices-label:

Better Practices
================

Anomaly Detection vs. Supervised Learning
-----------------------------------------

Anomaly Detection
^^^^^^^^^^^^^^^^^
	* Very small number of positive examples (:math:`y = 1`). (0-20 is common).
	* Large number of negative (:math:`y = 0`) examples.
	* Many different "types" of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like.
	* Future anomalies may look nothing like any of the anomalous examples we've seen so far.

	Examples:

	* Fraud detection
	* Manufacturing (e.g. aircraft engines)
	* Monitoring machines in a data center

Supervised Learning
^^^^^^^^^^^^^^^^^^^
	* Large number of positive and negative examples.
	* Enough positive examples for algorithm to get a sense of what positive examples are like, future positive examples likely to be similar to ones in training set.

	Examples:

	* Email spam classification
	* Weather prediction (sunny/rainy/etc)
	* Cancer classification

