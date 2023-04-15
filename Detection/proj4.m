%% Husam Almanakly 
% ECE302 Project 4 - Detection

% This project implements a MAP Decision detector. Using different models
% for the generated data, this assignment calculates the optimal threshold
% value based on the max likelihood ratio test and then simualtes detector.
% Simulations are conducted to explore different SNR cases and plot what
% the simulated ROC curve results in

clc
clear
close all

%% Part 1 - Radar Detection


% Decision rule derived
figure
derivation = imread("derivation.png");
imshow(derivation, "InitialMagnification",1000)


%% Generate data - part a

params = gen_parameters(100000, 5, 2, 0.8, 0.2);
[t_error, error, best_guesses] = map_detector(params);

disp("Theoretical Error of MAP Detector: " + t_error);
disp("Experimental Error with " + params.N + " simulations: " + error);

%% Part b - ROC curves

% Generate set of parameters for a few SNR tests
params2 = gen_parameters(100000, 3, 1.5, 0.8, 0.2);
params3 = gen_parameters(100000, 1, 1, 0.8, 0.2);

[PF1, PD1] = roc_sim(params);
[PF2, PD2] = roc_sim(params2);
[PF3, PD3] = roc_sim(params3);

% Plot ROC curves
figure
plot(PF1, PD1, 'DisplayName', "A = " + params.A + ", \sigma = " + params.sigma);
hold on;
plot(PF2, PD2, 'DisplayName', "A = " + params2.A + ", \sigma = " + params2.sigma);
plot(PF3, PD3, 'DisplayName', "A = " + params3.A + ", \sigma = " + params3.sigma);
title("ROC Curve of Simulated MAP Detector")
ylabel("Probability of Detection")
xlabel("Probability of False Alarms")
legend()

%% Part c



%% Functions

function [PF, PD] = roc_sim(params)
    % Sweep possible params.gammavalues 
    gamma_sweep = linspace(-5 * params.sigma, 5 * params.sigma, 1000);
    
    PF = zeros(1, 1000);
    PD = zeros(1, 1000);
    % Run our detector sweeping through each possible gamma
    for i=1:1000
        params.gamma = gamma_sweep(i);
        [~, ~, guesses] = map_detector(params);
    
        % False alarm when we were wrong and the guess was a 1
        false_alarms = (guesses ~= params.labels) .* guesses;
           
        % Detect when we were right and its a 1
        detects = (guesses == params.labels) .* guesses;
        
        % Save probabilities
        PF(i) = mean(false_alarms);
        PD(i) = mean(detects);
    end
end


% Function to simulate a MAP detector
% Takes in the generated observations, the params.labels, parameters of the data,
% and threshold point to be used
function [t_error, error, guesses] = map_detector(params)
    % Make our estimation, calculate how many times we were correct 
    guesses = params.obs > params.gamma;
    correct = guesses == params.labels;
    
    % Error
    error = sum(~correct) / params.N;
    
    % Theoretical Error
    Pf = qfunc(params.gamma/ params.sigma);
    Pm = 1 - qfunc((params.gamma- params.A) / params.sigma);
    t_error = Pf * params.H0 + Pm * params.H1;
end


% Function to generate data and store parameters in returned struct
function params = gen_parameters(N, A, sigma, H0, H1)
    % Save parameters in struct to pass into functions
    params.N = N; 
    params.A = A; 
    params.sigma = sigma;
    params.H0 = H0;
    params.H1 = H1;
    
    
    % Figure out threshold and plot class conditionals...
    figure
    x = linspace(-params.sigma * 2, params.A + params.sigma * 2, 1000);
    pdf_present = normpdf(x, params.A, params.sigma) * params.H1;
    pdf_not_present = normpdf(x, 0, params.sigma) * params.H0;
    plot(x, pdf_not_present);
    hold on; 
    plot(x, pdf_present);
    
    % Intersection point
    params.gamma = intersect(pdf_present, pdf_not_present, x);

    % Class conditionals (prior of 0.2 and 0.8) 
    present = normrnd(params.A, params.sigma, 1, params.N * params.H1);
    not_present = normrnd(0, params.sigma, 1, params.N * params.H0);
    
    % Label is the index (present if idx is less than H1 * N, not present if greater
    params.obs = [present, not_present];
    params.labels= 1:params.N <= params.N * params.H1;
end


% Function to help find intersection of two pdfs
function gamma = intersect(pdf1, pdf2, x)
    % Find points where the plot intersects
    possible_idx = find(abs(pdf1- pdf2) < 0.001);

    % Remove points at either ends (gaussians approaching 0)
    idx = (possible_idx > 200).*(possible_idx < 800);
    idx = idx(find(idx, 1));

    % Return x value / intersection point
    gamma= x(possible_idx(idx));
end
