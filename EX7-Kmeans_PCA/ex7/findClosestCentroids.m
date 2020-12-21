function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

    
n = size(X,2);
m = size(X,1);
Ones_Kby1 = ones(K, 1);
val = 0;    
indx = 0;

%Using only 1 for loop (vectorized implementation)
for ii = 1:m
    %multiX = Ones_Kby1 * X(ii,:); %make a copy of ii'th X K times. Will be a K by n array
    multiX = X(ii*ones(K,1), :); %This also works. It works as ii is within the 
    %dimensions of X (i.e. ii is less than m)
    temp1 = multiX - centroids;
    normSquared = sum(temp1 .^ 2, 2); %norm squared of temp1 (for each row, i.e. each centroid)
    [val indx] = min(normSquared);
    idx(ii,1) = indx;
end

%Using 2 for loops
% for ii = 1:m
%     temp = zeros(K,1);
%     val = 0;
%     indx = 0;
%     for jj = 1:K
%         temp(jj,1) = (norm(X(ii,:) - centroids(jj,:)))^2;
%     end
%     [val indx] = min(temp);
%     idx(ii,1) = indx;
% end

% =============================================================

end

