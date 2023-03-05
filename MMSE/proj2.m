%% Husam Almanakly 
% ECE302 Project 2 - Minimum Mean Square Error

% This project simulates an MMSE and Linear estimator from the MIT
% notes example in class. Using the worked out example, this script
% confirms the theoretical results through simulation, as explained
% throughout the assignment. 


%% Scenario 1 - 

% Bayes MMSE - 8.5 example X = Y + W
% Y ~ U(-1,1)
% W ~ U(-2, 2)

N = 10000;
Y = -1 + (2) * rand(N, 1);
W = -2 + (4) * rand(N, 1);

X = Y + W;

%% MMSE Estimator:

% From calculations:
% E[Y|X] = {
%               0.5 + 0.5 * X       -3 <= X <= 1
%               0                   -1 <= X <= 1
%               -0.5 + 0.5 * X      1 <= X <= 3
%          }

mmse = zeros(size(X));

% First region
idx1 = X>=-3 & X<=-1;
mmse(idx1) = 0.5 + 0.5 * X(idx1);

% Second region (all zeros anyway, but for consistency)
idx2 = X>=-1 & X<=1;
mmse(idx2) = 0;

% Third region
idx3 = X>=1 & X<=3;
mmse(idx3) = -0.5 + 0.5 * X(idx3);

%% Estimator MSE

% Calculated to be:
% E[(Y-~Y)^2|X] = {
%                       (3 + x)^2 / 12        -3 <= X <= 1
%                       1/3                   -1 <= X <= 1
%                       (3 - x)^2 / 12        1 <= X <= 3
%                 }


mse = zeros(size(X));

% Repeat: calculate for each region
mse(idx1) = (3 + X(idx1)).^2 ./ 12;
mse(idx2) = 1/3;
mse(idx3) = (3 - X(idx3)).^2 ./ 12;

% Now we can calculate our estimators average MSE:
avg_mse = mean(mse);

disp("Average MSE of MMSE Estimator: " + avg_mse)
disp("Theoretical MSE of MMSE Estimator from Calculations: 0.25")

%% Linear MMSE 

% We can also simulate the linear MMSE
% ~y = mu_y + px_y * std(y) / std(x) * ( X - mu_x) 

mu_y = mean(Y);
mu_x = mean(X);
p_x_y = corrcoef(X,Y);
p_x_y = p_x_y(1,2);

mmse_l = mu_y + p_x_y * std(Y) ./ std(X) .* ( X - mu_x);

% Calculate error 
% MSE = var(y) * ( 1 - p^2) (from notes)
mse_l = var(Y) * (1 - p_x_y^2);

theoretical_mse = 1/4; 
disp("MSE of Linear MMSE Estimator: " + mse_l)
disp("MSE using simply mu_y = 0 as our estimator: " + 1/3)
results = table(avg_mse, mse_l, theoretical_mse)


%% Scenario 2 - 

