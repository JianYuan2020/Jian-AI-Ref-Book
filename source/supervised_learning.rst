Supervised learning
===================


Linear Regression with Multiple Variables/Features
--------------------------------------------------

* n = number of features.
* m = number of training examples.
* x(i) = input (features) of ith training example.
* x(i)j = value of feature j in ith training example.
* x(i) = [ x(i)1; x(i)2; ... x(i)j; ... x(i)n ] -- n x 1 column vector.
* X = [ x(1)'; x(2)'; ... x(i)'; ... x(m)' ] -- m x n matrix.

1. Hypothesis
^^^^^^^^^^^^^
* h_theta(x) = theta_0 + theta_1*x1 + theta_2*x2 + ... + theta_j*xj + ... theta_n*xn

* Define: x0 = 1 (x(i)0 = 1)
* x = [ x0; x1; x2; ... xj; ... xn ] -- (n + 1) x 1 column vector.

Parameters
^^^^^^^^^^
* Theta = [ theta_0; theta_1; theta_2; ... theta_j; ... theta_n ] -- (n + 1) x 1 column vector.

Therefore:

* h_theta(x) = theta_0*x0 + theta_1*x1 + theta_2*x2 + ... + theta_j*xj + ... theta_n*xn
* h_theta(x) = Theta_transpose*x

2. Cost Function
^^^^^^^^^^^^^^^^
* J(Theta) = sum((h_theta(x(i)) - y(i))^2 where i = 1:m)/(2*m)

3. Gradient Descent
^^^^^^^^^^^^^^^^^^^
* theta_j = theta_j - alpha*sum((h_theta(x(i)) - y(i))*x(i)j where i = 1:m)/m
* Note here, x(i)0 = 1; j = 0:n; and Theta is a (n + 1) x 1 column vector;

Unsupervised learning
---------------------

* Clusters and Segmentation