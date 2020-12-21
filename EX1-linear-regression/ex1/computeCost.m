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

% Ver1 is using for loop (non-vectorized)
#sigma_temp = 0;

#for i = 1:m
#  sigma_temp = sigma_temp + ((theta(1,1)*X(i,1) + theta(2,1)*X(i,2)) - y(i,1))^2;
#endfor

#J = 1/(2*m)*sigma_temp;
% End Ver1

%Ver2 is vectorized implementation
prediction_h = X * theta; %(theta' * X')';
error_squared = (prediction_h - y) .^ 2;
J = 1/(2*m)*sum(error_squared);
%End Ver2

% =========================================================================

end
