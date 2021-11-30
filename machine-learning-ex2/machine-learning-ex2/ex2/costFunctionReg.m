function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);
z = X * theta;

for i = 1 : m
  h_theta = sigmoid(z(i, 1));
  y_val = y(i,1);
  sigma_val = (y_val * log(h_theta))+((1 - y_val) * log(1 - h_theta)) ;
  J = J - (sigma_val/m);
end

t_sq = 0;
for j = 2 : n
    t_sq = t_sq + (theta(j, 1)^2);
end
J = J + ((lambda/(2*m))* t_sq) ;

for i = 1 : m
  h_theta = sigmoid(z(i, 1));
  y_val = y(i,1);
  sigma_val = (h_theta - y_val);
  X_T = (sigma_val/m) * X(i, :);
  grad = grad + (X_T');
end

tmp = grad(1,1);
grad = grad + ((lambda/m) * theta);
grad(1,1) = tmp;

% =============================================================

end
