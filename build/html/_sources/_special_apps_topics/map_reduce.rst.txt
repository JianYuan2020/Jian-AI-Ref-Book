.. _map-reduce-label:

Map-reduce
==========

Map-reduce and data parallelism.

	Let :math:`m = 400`, Batch gradient descent: :math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{400} \sum_{i=1}^{400} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

	Distribute into 4 machines:

		Machine 1: Use :math:`(x^{(1)}, y^{(1)}), ..., (x^{(100)}, y^{(100)})`

			:math:`temp^{(1)}_{j} = \sum_{i=1}^{100} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Machine 2: Use :math:`(x^{(101)}, y^{(101)}), ..., (x^{(200)}, y^{(200)})`

			:math:`temp^{(2)}_{j} = \sum_{i=101}^{200} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Machine 3: Use :math:`(x^{(201)}, y^{(201)}), ..., (x^{(300)}, y^{(300)})`

			:math:`temp^{(3)}_{j} = \sum_{i=201}^{300} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

		Machine 4: Use :math:`(x^{(301)}, y^{(301)}), ..., (x^{(400)}, y^{(400)})`

			:math:`temp^{(4)}_{j} = \sum_{i=301}^{400} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

	Combine:

		:math:`\theta_{j} = \theta_{j} - \alpha \frac{1}{400} (temp^{(1)}_{j} + temp^{(2)}_{j} + temp^{(3)}_{j} + temp^{(4)}_{j})`

		:math:`j = 0, ..., n`

Many learning algorithms can be expressed as computing sums of functions over the training set.

E.g. for advanced optimization, with logistic regression, need:

	:math:`J_{Train}(\theta) = - \frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log (h_\theta (x^{(i)})) - (1 - y^{(i)}) \log (1 - h_\theta (x^{(i)}))`

	:math:`\frac{\partial }{\partial \theta_{j}} J_{Train}(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)}) x^{(i)}_{j}`

	:math:`j = 0, ..., n`
