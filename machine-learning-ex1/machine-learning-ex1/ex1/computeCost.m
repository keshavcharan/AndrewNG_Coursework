function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h_rms = zeros(size(X, 1), 1);

for i = 1 : 2
	th = theta(i, 1) * X(:, i);
	h_rms = h_rms + th;
end

h_rms = h_rms - y;
h_rms = h_rms .^ 2;

for i = 1 : m
	J = J + h_rms(i, 1);
end

J = J/(2 * m);

% =========================================================================

end