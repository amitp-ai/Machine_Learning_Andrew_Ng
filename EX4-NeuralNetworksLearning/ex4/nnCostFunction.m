function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


yMap = zeros(m, num_labels); %5000 by 10 matrix

%code to map labelled vector of y into a binary vector (actually matrix) of y
for ii = 1:m
  yMap(ii, y(ii)) = 1;
end

%code to map a binary vector (actually matrix) of y to a labelled vector of y
%yBack = zeros(m:1);
%for ii=1:m
%  [val, indx] = find(yMap(ii, :) == 1);
%  yBack(ii) = indx;
%end

aOne = X;
aOne_wBias = [ones(m,1), X]; %5000 by 401 matrix
aTwo = sigmoid (Theta1 * aOne_wBias'); %25 by 5000 matrix
aTwo_wBias = [ones(1, m); aTwo]; %26 by 5000 matrix
aThree = sigmoid(Theta2 * aTwo_wBias); %10 by 5000 matrix
hTheta = aThree'; %5000 by 10 matrix

%Theta1 is 25 by 401 matrix and Theta2 is 10 by 26 matrix
%Remove Theta0_for_bias_units for regularization.
Theta1_Reg = Theta1(:, 2:end); %25 by 400
Theta2_Reg = Theta2(:, 2:end); %10 by 25
Theta_Reg = [Theta1_Reg(:); Theta2_Reg(:)];


J = -1/m*sum(sum(yMap .* log(hTheta) + (1-yMap).*log(1-hTheta))) ...
     + lambda/2/m*sum(Theta_Reg .^ 2); 
%sum(sum(matrix)) sums all the elements of a matrix.

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

Delta1 = zeros(hidden_layer_size,input_layer_size+1); %same size as Theta1
Delta2 = zeros(num_labels,hidden_layer_size+1); %same size as Theta2

%Note: Theta1 is 25 by 401 matrix and Theta2 is 10 by 26 matrix
%Note: yMap is 5000 by 10 matrix
%Note: Variable_wBias means that variable with bias units
%Note: Variable_woBias means that variable without bias units

for ii = 1:m
  %forward propagation
  aOne = X(ii, :);
  aOne_wBias = [1, aOne]; %1 by 401 vector  
  aTwo = sigmoid(Theta1 * aOne_wBias'); %25 by 1 vector
  aTwo_wBias = [1;aTwo]; %26 by 1 vector
  aThree = sigmoid(Theta2 * aTwo_wBias); %10 by 1 vector
  hTheta = aThree'; %1 by 10 vector
  
  %back propagation
  dThree = hTheta - yMap(ii,:); %1 by 10 vector
  dTwo = (Theta2' * dThree') .* (aTwo_wBias .* (1-aTwo_wBias)); %26 by 1 vector
  dTwo_woBias = dTwo(2:end); %Remove dTwo_0. So now it's 25 by 1 vector
  %dOne = (Theta1' * dTwo_woBias)' .* (aOne_wBias .* (1-aOne_wBias)); %1 by 401 vector
  %dOne_woBias = dOne(2:end); %Remove dOne_0. So now it's 400 by 1 vector
  %Don't need to generate dOne. But have include it to get an idea for how
  %the code will look like if have more than 3 layers neural network
 
  Delta1 = Delta1 + dTwo_woBias * aOne_wBias; %25 by 401 matrix
  Delta2 = Delta2 + dThree' * aTwo_wBias'; %10 by 26 matrix

end

Theta1_grad = 1/m*Delta1;
Theta2_grad = 1/m*Delta2;
  
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
