function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	h_vector = zeros(size(X, 1), 1);
	h_theta = zeros(2, 1);
	
	for i = 1 : 2
		th = theta(i, 1) * X(:, i);
		h_vector = h_vector + th;			
	end				

	h_vector = h_vector  - y;

	for j = 1 : m
		for i = 1 : 2
			h_theta(i, 1) = h_theta(i, 1) + (h_vector(j, 1) * X(j, i));
		end
	end
	h_theta = h_theta/m;
	h_theta = h_theta * alpha;
	% h_theta
	theta = theta - h_theta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
