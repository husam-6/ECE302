%% Husam Almanakly 
% ECE302 Project 3 - Max Likelihood Estimation

% This project implements two maximum likelihood estimators and further
% compares their performance. Samples are drawn from a Rayleigh and an
% Exponential distribution, from which we apply Max Likelihood. The
% estimators are evaluated based on their bias and variance, and the
% experiment is repeated for multiple parameter values (scale parameter for
% Rayleigh distribution, and lambda for exponential dist).

clc
clear
close all


%% Part 1

% Generating random draws from both Rayleigh and Exponential RV's

num_obs = 20;
num_trials = 10000; 
N = 1:num_obs;


% ML Estimators for both Rayleigh and Exponential Distribution

% Derived by hand, see derivation below
figure
derivation = imread("ml_derivation.png");
imshow(derivation, "InitialMagnification",1000)

%% Mean Square Error

% Experiment with multiple values of sigma
sigma = [2 3 4];
sigma_lab = ["\sigma= " + sigma(1), "\sigma= " + sigma(2), "\sigma= " + sigma(1)];
[ray_1, ray_ml_1] = rayleigh_ml_sim(num_trials, num_obs, sigma(1));
[ray_2, ray_ml_2] = rayleigh_ml_sim(num_trials, num_obs, sigma(2));
[ray_3, ray_ml_3] = rayleigh_ml_sim(num_trials, num_obs, sigma(3));

% Putting mse data in matrix
ray_ml = [ray_1', ray_2', ray_3']; 

% Repeat same process for exponential distribution
% Multiple lambda experiments
lambs = [2 4 8];
lambs_lab = ["\lambda = " + lambs(1), "\lambda = " + lambs(2), "\lambda = " + lambs(1)];
[exp_1, exp_ml_1] = exp_ml_sim(num_trials, num_obs, lambs(1));
[exp_2, exp_ml_2] = exp_ml_sim(num_trials, num_obs, lambs(2));
[exp_3, exp_ml_3] = exp_ml_sim(num_trials, num_obs, lambs(3));

% Putting mse data in matrix
exp_ml = [exp_1', exp_2', exp_3'];

graph(ray_ml, exp_ml, "MSE", sigma_lab, lambs_lab)

%% Estimator Bias

% Calculate bias for each case of lambda / sigma
ray_bias = [bias(ray_ml_1, sigma(1))', bias(ray_ml_2, sigma(2))', bias(ray_ml_3, sigma(3))'];
exp_bias = [bias(exp_ml_1, lambs(1))', bias(exp_ml_2, lambs(2))', bias(exp_ml_3, lambs(3))'];
graph(ray_bias, exp_bias, "Bias", sigma_lab, lambs_lab)


%% Estimator Variance

% Calculate variance for each case of lambda / sigma
ray_var = [var(ray_ml_1)', var(ray_ml_2)', var(ray_ml_3)'];
exp_var = [var(exp_ml_1)', var(exp_ml_2)', var(exp_ml_3)'];
graph(ray_var, exp_var, "Variance", sigma_lab, lambs_lab)



%% Part 2

data = load("data.mat").data;

% Compute ML for Rayleigh and Exponential Distribution
data_ray_ml = sqrt(1./(2*size(data, 2)) .* sum(data.^2));
data_exp_ml = size(data,2) ./ sum(data);

disp("Rayleigh Max Likelihood of given data: " + data_ray_ml);
disp("Exponential Max Likelihood of given data: " + data_exp_ml);
disp("Exponential Max Likelihood value > Rayleigh Max Likelihood =>")
disp("We can assume the data is Exponential")



%% Functions

% Takes in 2 matrices representing both Rayleigh and Exponential data and
% graphs on a subplot
function r = graph(ray, exp, y_title, sigma, lambs)
    figure
    subplot(1, 2, 1);

    hold on
    plot(ray)
    xlabel("Number of Observations")
    ylabel(y_title)
    title("Rayleigh ML Estimator " + y_title)
    xlim([5, 20])
    legend(sigma)
    
    % For exponential distribution
    subplot(1, 2, 2);
    
    hold on
    plot(exp)
    xlabel("Number of Observations")
    ylabel(y_title)
    title("Exponential ML Estimator " + y_title)
    legend(lambs)
    xlim([5, 20])
    
end


% Function to simulate Rayleigh Dist Max Likelihood
% We use cumsum instead of a regular summation to get the ml estimation for
% each number of observations (ie for 1 observation, 2, 3, etc)
function [ray_mse, rayleigh_ml] = rayleigh_ml_sim(num_trials, num_obs, sig)
    % Scale parameter (sigma in Rayleigh Dist)
    ray = raylrnd(sig, [num_trials, num_obs]);
    
    % Rayleigh ML Estimator
    % Theta = 1 / (2 * N) * ∑xi^2       NOTE: theta = sigma^2
    N = 1:num_obs; 
    rayleigh_ml = sqrt(1./(2*N) .* cumsum(ray.^2, 2));
    ray_mse = mean((rayleigh_ml - sig).^2);
end

% Function to simulate Exponential Dist Max Likelihood
function [exp_mse, exp_ml] = exp_ml_sim(num_trials, num_obs, lamb)

    % Mean of Exp RV = 1/lambda
    N = 1:num_obs; 
    mu = 1 / lamb;
    exp = exprnd(mu, [num_trials, num_obs]);
    
    % Exponential ML Estimator
    % theta = N / ∑xi
    exp_ml = N ./ cumsum(exp, 2);
    exp_mse = mean((exp_ml - lamb).^2);
end

% Helper function for bias
function b= bias(ml_est, true)
    b = mean(ml_est) - true;
end
