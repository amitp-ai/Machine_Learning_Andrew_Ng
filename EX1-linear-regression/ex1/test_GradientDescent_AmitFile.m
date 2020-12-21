ex1_multi;

figure();
theta = zeros(3, 1); 
alpha = 1;
[theta, J1] = gradientDescentMulti(X, y, theta, alpha, num_iters);

theta = zeros(3, 1); 
alpha = 0.01;
[theta, J2] = gradientDescentMulti(X, y, theta, alpha, num_iters);

theta = zeros(3, 1); 
alpha = 0.001;
[theta, J3] = gradientDescentMulti(X, y, theta, alpha, num_iters);

plot(1:50, J1(1:50), 'b');
hold on;
plot(1:50, J2(1:50), 'r');
plot(1:50, J3(1:50), 'k');
