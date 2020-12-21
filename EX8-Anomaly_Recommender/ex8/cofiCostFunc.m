function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%for loop implementation of cost function
% for ii=1:num_movies
%     for jj=1:num_users
%         if R(ii,jj)
%             J = J + 1/2*(X(ii, :) * (Theta(jj,:))' - Y(ii,jj))^2;
%         end
%     end
% end

%vectorized implementation of cost function
squaredErrorMatrix = (X * Theta' - Y).^2;
squaredErrorMatrix = squaredErrorMatrix .* R; %it will zero out errors where R=0
J = 1/2 * sum(sum(squaredErrorMatrix)) + lambda/2*(sum(sum(Theta .* Theta)) + sum(sum(X .* X)));

%for loop implementation of gradient
% for ii=1:num_movies
%     for jj=1:num_users
%         for kk=1:num_features
%             if R(ii,jj)
%                 X_grad(ii, kk) = X_grad(ii, kk) + (X(ii, :) * (Theta(jj,:))' - Y(ii,jj)) * Theta(jj,kk);
%                 Theta_grad(jj,kk) = Theta_grad(jj,kk) + (X(ii, :) * (Theta(jj,:))' - Y(ii,jj)) * X(ii,kk);
%             end
%         end
%     end
% end

%Partially vectorized implementation for gradient I
% for ii = 1:num_movies
%     X_grad(ii,:) = ((X(ii,:) * (Theta') - Y(ii,:))) .* R(ii,:)  * Theta; %zero out any predictions for non-rated movie by user jj
% end
% 
% for jj = 1:num_users
%     Theta_grad(jj,:) = ((X * (Theta(jj,:))' - Y(:,jj)) .* R(:,jj))' * X; %zero out any predictions for non-rated movie i by user j
% end

%Partially vectorized implementation for gradient II (This is the approach
%sugegsted in the programming exercise)
for ii = 1:num_movies
    idx = find(R(ii,:) == 1); %find users who rated movie ii
    ThetaTemp = Theta(idx, :); %create ThetaTemp of only the users who rated movie ii
    YTemp = Y(ii,idx); %Ratings of movie ii which are actually rated
    X_grad(ii,:) = (X(ii,:) * (ThetaTemp') - YTemp) * ThetaTemp;
end
X_grad = X_grad + lambda*X;

for jj = 1:num_users
    idx = find(R(:, jj) == 1); %find movies rated by user jj
    XTemp = X(idx, :); %create XTemp of only movies rated by user jj
    YTemp = Y(idx, jj); %Ratings of user jj which are actually rated
    Theta_grad(jj,:) = (XTemp * (Theta(jj,:)') - YTemp)' * XTemp;
end
Theta_grad = Theta_grad + lambda*Theta;

%Fully Vectorized implementation for gradient
% X_grad = ((X * Theta' - Y) .* R)  * Theta; %zero out any predictions for non-rated movie i by user j
% Theta_grad = ((X * Theta' - Y) .* R)' * X; %zero out any predictions for non-rated movie i by user j

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
