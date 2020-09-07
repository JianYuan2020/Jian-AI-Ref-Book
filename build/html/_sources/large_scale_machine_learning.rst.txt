.. _large-scale-machine-learning-label:

Large Scale Machine Learning
============================

How to choose learning rate :math:`\alpha`? 

If :math:`\alpha` is too small, gradient descent can be slow. If :math:`\alpha` is too large, 
gradient descent can overshoot the minimum. It may fail to converge, or even diverge. 

Gradient descent can converge to a local minimum, even with the learning rate :math:`\alpha` fixed.

For sufficiently small :math:`\alpha`, :math:`J(\Theta)` should decrease on every iteration.
To choose :math:`\alpha`, try: ... 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...