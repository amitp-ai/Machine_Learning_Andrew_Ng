function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hTheta = X * theta; %m by 1
hThetay = hTheta - y; %m by 1
J = 1/2/m * (hThetay'*hThetay) + lambda/2/m*(theta(2:end))'*(theta(2:end));
%for a vector A=>m by 1, sum(A.^2) = A'*A

grad(1) = 1/m*hThetay'*X(:,1); %m by 1 times 1 by m = scalar
grad(2:end) = 1/m*(hThetay'*X(:,2:end))' + lambda/m*theta(2:end);








% =========================================================================

grad = grad(:);

end
