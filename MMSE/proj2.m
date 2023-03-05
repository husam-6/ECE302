%% Husam Almanakly 
% ECE302 Project 2 - Minimum Mean Square Error

% This project simulates an MMSE and Linear estimator from the MIT
% notes example in class. Using the worked out example, this script
% confirms the theoretical results through simulation, as explained
% throughout the assignment. 

clc
clear
close all

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
mse_l = mean((Y - mmse_l).^2);

theoretical_mse = 1/4; 
disp("MSE of Linear MMSE Estimator: " + mse_l)
disp("MSE using simply mu_y = 0 as our estimator: " + 1/3)


% Set up table for results
sz = [2 3];
varTypes = ["string", "double", "double"];
varNames = ["Method", "Simulated MSE", "Theoretical MSE"];
results = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
results(1, :) = {"MMSE Estimator", avg_mse, 1/4};
results(2, :) = {"Linear MMSE Estimator", mse_l, 4/15};
results



%% Scenario 2 - 

% Linear estimator for multiple observations:

% X1 = Y + R1
% X2 = Y + R2
% ...
% XN = Y + RN

var_y = 1;
var_r = 0.2;
N = 1000;

figure
plot_mses(var_y, var_r, N)
plot_mses(var_y+0.5, var_r + 0.1, N)


% Function to plot the mses given a variance for Y and R and number of
% samples
function r = plot_mses(var_y, var_r, N)
    num_experiments = 7;
    mses = zeros(1, num_experiments);
    t_mses = zeros(1, num_experiments);
    for i = 1:num_experiments
        Y = random('Normal', 1, sqrt(var_y), N, 1);
        Rs = random('Normal', 0, sqrt(var_r), N, i);
        
        Xs = Y + Rs;
        
        [mse, theoretical_mse] = mult_noisy_mmse(Y, Xs, Rs);
        mses(i) = mse; 
        t_mses(i) = theoretical_mse;
    end

    % plot results
    t = 1:num_experiments;
    plot(t, mses, 'DisplayName', "Simulation, var_Y = " + var_y + ", var_R = " + var_r)
    hold on;
    plot(t, t_mses, 'DisplayName', "Theoretical, var_Y = " + var_y + ", var_R = " + var_r)
    legend
    title("Multiple Noisy Observations")
    xlabel("Number of Observations")
    ylabel("MSE")
end


% Function to calculate mse of a noisy observation
function [mse, theoretical_mse] = mult_noisy_mmse(Y, Xs, Rs)
    % Linear estimator (from notes)
    % Use average varainces of both R's...
    num_obs = length(Xs(1, :));
    var_R = mean(var(Rs));
    
    % Calculate var(Y) * X1 + var(Y) * X2 + var(Y) * X3 + ... + var(Y) * XN
    obs = sum(var(Y) * Xs, 2);
    
    % Estimator
    % mu_y = 1;
    Y_l = 1 ./ (num_obs * var(Y) + var_R) * (var_R * 1 + obs);
    mse = mean((Y - Y_l).^2);

    % Theoretical MSE (again from notes)
    theoretical_mse = (var(Y) * var_R) / (num_obs * var(Y) + var_R);

end





















