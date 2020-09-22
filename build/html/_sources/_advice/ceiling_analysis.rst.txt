
Ceiling Analysis
================

	* What part of the pipeline to work on next
	* Estimating the errors due to each component (ceiling analysis)

Photo OCR
---------

	* Here is the Photo OCR ML pipeline:
	* Image -> Text detection -> Character segmentation -> Character recognition
	* The Overall system has an Accuracy of 72%
	* What part of the pipeline should you spend the most time trying to improve?

	========================  ========    =================================================
	        Component         Accuracy                     Action
	========================  ========    =================================================
	 Overall system            72%
	 Text detection            89%        manually ensure 100% accuracy for this component
	 Character segmentation    90%        manually ensure 100% accuracy for this component
	 Character recognition     100%       manually ensure 100% accuracy for this component
	========================  ========    =================================================

Text detection
^^^^^^^^^^^^^^

	So we should work on Text detection component next since it reduces the Overall system's 
	accuracy by 17% while the other two components each reduces the Overall system's accuracy by 10%.

