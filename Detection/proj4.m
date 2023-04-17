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

N = 100000;
params = gen_parameters(N, 5, 2, 0, 0.8, 0.2, 1, 1);
[t_error, error, best_guesses] = map_detector(params);

disp("Theoretical Error of MAP Detector: " + t_error);
disp("Experimental Error with " + params.N + " simulations: " + error);

%% Part b - ROC curves

% Generate set of parameters for a few SNR tests
params2 = gen_parameters(N, 3, 1.5, 0, 0.8, 0.2, 1, 1);
params3 = gen_parameters(N, 1, 1, 0, 0.8, 0.2, 1, 1);

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

% Same parameters as beginning, just change cost now
cost_params = gen_parameters(100000, 5, 2, 0, 0.8, 0.2, 1, 10);

% Calculate Pf and Pd using this structure
[~, ~, guesses] = map_detector(cost_params);
false_alarms = (guesses ~= cost_params.labels) .* guesses;
detects = (guesses == cost_params.labels) .* guesses;

% Save probabilities
c_PF = mean(false_alarms);
c_PD = mean(detects);

% Replot first ROC curve with the optimal decision threshold for this cost
figure
plot(PF1, PD1, 'DisplayName', "ROC Curve for A = " + params.A + ", \sigma = " + params.sigma);
hold on;
plot(c_PF, c_PD, "ro", 'DisplayName', "C10 / C01 = 1/10")
title("ROC Curve of Simulated MAP Detector")
ylabel("Probability of Detection")
xlabel("Probability of False Alarms")
legend


%% Part d



%% Part e
% Generate new model parameters
new_model1 = gen_parameters(N, 0, 1, 5.5, 0.8, 0.2, 1, 1);
[t_error, error, guesses] = map_detector(new_model1);

disp("Using new Y = A + Z model for P(Y|H0) and Y = A + X for P(Y|H1)");
disp("Theoretical Error of MAP Detector: " + t_error);
disp("Experimental Error with " + new_model1.N + " simulations: " + error);

%% ROC curves

% Generate set of parameters for a few SNR tests
new_model2 = gen_parameters(N, 0, 0.5, 3, 0.8, 0.2, 1, 1);
new_model3 = gen_parameters(N, 0, 0.25, 4, 0.8, 0.2, 1, 1);

[PF1, PD1] = roc_sim(new_model1);
[PF2, PD2] = roc_sim(new_model2);
[PF3, PD3] = roc_sim(new_model3);

% Plot ROC curves
figure
plot(PF1, PD1, 'DisplayName', "\sigma = " + new_model1.sigma + ", \sigma_z = " + new_model1.sigma_z);
hold on;
plot(PF2, PD2, 'DisplayName', "\sigma = " + new_model2.sigma + ", \sigma_z = " + new_model2.sigma_z);
plot(PF3, PD3, 'DisplayName', "\sigma = " + new_model3.sigma + ", \sigma_z = " + new_model3.sigma_z);
title("ROC Curve of Simulated MAP Detector")
ylabel("Probability of Detection")
xlabel("Probability of False Alarms")
legend()


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
    
    if (params.sigma_z ~= 0)
        guesses_gt = params.obs > params.gamma;
        guesses_lt = params.obs < (-1) * params.gamma;
        guesses = guesses_gt .* guesses_lt;
        correct = guesses == params.labels;
    else
        % Make our estimation, calculate how many times we were correct 
        guesses = params.obs > params.gamma;
        correct = guesses == params.labels;

    end
    
    % Error
    error = sum(~correct) / params.N;
    
    % Theoretical Error
    if params.sigma_z == 0
        Pf = qfunc(params.gamma/ params.sigma);
        Pm = 1 - qfunc((params.gamma- params.A) / params.sigma);
        t_error = Pf * params.H0 + Pm * params.H1;
    else
        Pf = qfunc(params.gamma / params.sigma_z) - qfunc(-1 * params.gamma / params.sigma_z);
        Pm = 2 * qfunc((-1 * params.gamma) / params.sigma);
        t_error = Pf * params.H0 + Pm * params.H1;
    end
end


% Function to generate data and store parameters in returned struct
function params = gen_parameters(N, A, sigma, sigma_z, H0, H1, C10, C01)
    % Save parameters in struct to pass into functions
    params.N = N; 
    params.A = A; 
    params.sigma = sigma;
    params.H0 = H0;
    params.H1 = H1;
    params.sigma_z = sigma_z;
    
    
    % Figure out threshold and plot class conditionals...
    figure
    bound = max(params.sigma, params.sigma_z);
    x = linspace(-bound * 2 + params.A, params.A + bound * 2, 10000);
    pdf_present = normpdf(x, params.A, params.sigma);
    scale = (C10 / C01) * (H0 / H1);
    if sigma_z == 0
        pdf_not_present = normpdf(x, 0, params.sigma) * scale;
        not_present = normrnd(0, params.sigma, 1, params.N * params.H0);
    else
        pdf_not_present = normpdf(x, A, params.sigma_z) * scale;
        not_present = normrnd(A, params.sigma_z, 1, params.N * params.H0);
    end
    plot(x, pdf_not_present, 'DisplayName', "P(Y|H0)");
    hold on; 
    plot(x, pdf_present, 'DisplayName', "P(Y|H1)");
    title("Class Conditional PDF's when A = " + A + " and \sigma = " + sigma);
    legend()

    % Intersection point
    params.gamma = intersect(pdf_present, pdf_not_present, x);

    % Class conditionals (prior of 0.2 and 0.8) 
    present = normrnd(params.A, params.sigma, 1, params.N * params.H1);
    
    % Label is the index (present if idx is less than H1 * N, not present if greater
    params.obs = [present, not_present];
    params.labels= 1:params.N <= params.N * params.H1;
end


% Function to help find intersection of two pdfs, returns first
% intersection point
function gamma = intersect(pdf1, pdf2, x)
    % Find points where the plot intersects
    possible_idx = find(abs(pdf1- pdf2) < 0.001);

    % Remove points at either ends (gaussians approaching 0)
    idx = (possible_idx > 500).*(possible_idx < 9500);
    idx = idx(find(idx, 1));

    % Return x value / intersection point
    gamma= x(possible_idx(idx));
end
