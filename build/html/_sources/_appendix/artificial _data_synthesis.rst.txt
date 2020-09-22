.. _artificial-data-synthesis-label:

Artificial Data Synthesis
=========================

	* Getting lots of data
	* Sometimes, when we don't have enough Real data, we can generate Synthetic data

Letter Photos
-------------

	* Manually capture random letter of various fonts from different photos
	* Synthesizing data by introducing distortions

Audio
-----

	* Original audio, A
	* A + bad cellphone connection
	* A + noisy background, i.e. crowd
	* A + noisy background, i.e. machinery

* Distortion introduced should be representation of the type of noise/distortions in the test set
* Usually does not help to add purely random/meaningless noise to your data

Discussion on Getting More Data
-------------------------------

	#. Make sure you have a low bias classifier (so more data will help) before expending the effort. (Plot learning curves). E.g. keep increasing the number of features/number of hidden units in neural network until you have a low bias classifier.
	#. "How much work would it be to get 10x as much data as we currently have?"

		* Artificial data synthesis
		* Collect/label it yourself
		* "Crowd source" (E.g. Amazon Mechanical Turk)
