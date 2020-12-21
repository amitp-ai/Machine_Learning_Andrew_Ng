function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

## my version. Due to for loop, this is slow to run. Prolly cause I am calling plot function so many times
#len = length(X(:,1));


#for ii = 1:len
#  if y(ii) == 1
#    plot(X(ii,1), X(ii,2), 'k+', 'MarkerSize', 10);
#  else
#    plot(X(ii,1), X(ii,2), 'yo', 'MarkerSize', 10);
#  end
#
#end
## end my version  

## provided version. It runs fast.
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);
## end provided version

% =========================================================================


hold off;

end
