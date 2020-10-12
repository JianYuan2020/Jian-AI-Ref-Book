.. _online-learning-label:

Online Learning
===============

	Examples:

	* Shipping service
	* Product (phone) search, show user the 10 phones they're most likely to click on
	* Choosing special offers to show user
	* Customized selection of news articles
	* Product recommendation, ...

	Shipping service website where user comes, specifies origin and destination, you offer to ship their package 
	for some asking price, and users sometimes choose to use your shipping service (:math:`y = 1`), sometimes not 
	(:math:`y = 0`).

	Features :math:`x` capture properties of user, of origin/destination and asking price. We want to learn 
	:math:`p(y = 1|x;\theta)` to optimize price.

	:ref:`logistic-regression-label`

	Repeat forever {

		Get :math:`(x, y)` corresponding to user

		Update :math:`\theta` using :math:`(x, y)` {
		
			:math:`\theta_{j} = \theta_{j} - \alpha (h_\theta (x) - y) x_{j}`

			Here :math:`x_{0} = 1`, :math:`j = 0, ..., n`
	
		} throw out :math:`(x, y)` after computation

	}

	Can adapt to changing user preference.
