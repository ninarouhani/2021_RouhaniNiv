function [lik,lr] = sims_model(model,stim,reward,lr_emp,X)
% Input:
% model = model specified
% stim = 1 or 2 (the scene category presented on this trial)
% reward = the reward given in this trial
% lr_emp = model-generated learning rate from simulation
% Likelihood is calculated assuming that guesses are linear in prediction
%   plus gaussian noise (linear regression likelihood)
Ntrials = length(stim);

% set parameters
switch model
    case 'RW' % RW
        alpha = X;
        lr = nan(Ntrials,1);
    case 'expDecay' % RW-D
        eta = X(1);
        nu = X(2);
        lambda = X(3);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2); 
        count(1,:) = 0;
    case 'absPE' % RW-PH
        eta = X(1);
        k = X(2);
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
    case 'cuePE' % RW-M
        eta = X(1);
        g = X(2);
        cueLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
    case 'absCue' % RW-PH-M
        eta = X(1);
        g = X(2);
        k = X(3);
        cueLR = nan(Ntrials,1);
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
    case 'absD' % RW-PH-D
        eta = X(1);
        nu = X(2);
        lambda = X(3);
        k = X(4);
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
    case 'cueD' % RW-M-D
        eta = X(1);
        nu = X(2);
        lambda = X(3);
        g = X(4);
        cueLR = nan(Ntrials,1);
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
    case 'absCueD' % RW-PH-M-D
        eta = X(1);
        nu = X(2);
        lambda = X(3);
        k = X(4);
        g = X(5);
        cueLR = nan(Ntrials,1);
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
end

% initial values are 50
Vdeck = nan(Ntrials,2); % values of the stimuli
Vdeck(1,:) = 50; % start with average value of 50

for t = 1:Ntrials-1
    % the internal value of the stimulus on this trial (this is the mean
    % of the Gaussian from which the guess valEst(t) is drawn)
    V(t) = Vdeck(t,stim(t)); % the value of the stimulus on this trial
    
    % copy the current values to the next trial
    Vdeck(t+1,:) = Vdeck(t,:);
    
    % calculate signed and unsigned model-generated outcome RPEs
    PEsigned = reward(t) - V(t);
    absPE = abs(PEsigned)/100;
    
    % calculate the signed cue RPE (contingent on current deck)
    if stim(t)==1
        cuePE = (Vdeck(t,stim(t)) - Vdeck(t,2))/100;
    else
        cuePE = (Vdeck(t,stim(t)) - Vdeck(t,1))/100;
    end
    
    switch model
        case 'RW' % RW
            
            Vdeck(t+1,stim(t)) = V(t) + alpha*PEsigned;
            lr(t) = alpha;
            
        case 'expDecay' % RW-D
            
            stim_count = count(t,stim(t));
            decay = eta + nu*exp(-lambda*stim_count);
            
            lr(t) = sigmoid(decay); % dynamic learning rates
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
        case 'absPE' % RW-PH
            
            outLR(t) = k*absPE;
            
            act = eta + outLR(t);
            lr(t) = sigmoid(act); % dynamic learning rate
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
        case 'cuePE' % RW-M
            
            cueLR(t) = g*cuePE;
            
            act = eta + cueLR(t);
            lr(t) = sigmoid(act); % dynamic learning rate
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
        case 'absCue' % RW-PH-M
            
            cueLR(t) = g*cuePE;
            outLR(t) = k*absPE;
            
            act = eta + cueLR(t) + outLR(t);
            lr(t) = sigmoid(act); % dynamic learning rate
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
        case 'absD' % RW-PH-D
            
            stim_count = count(t,stim(t));
            decay = eta + nu*exp(-lambda*stim_count);
            
            outLR(t) = k*absPE;
            
            act = decay + outLR(t);
            lr(t) = sigmoid(act);  % dynamic learning rate
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
        case 'cueD' % RW-M-D
            
            stim_count = count(t,stim(t));
            decay = eta + nu*exp(-lambda*stim_count);
            
            cueLR(t) = g*cuePE;
            
            act = decay + cueLR(t);
            lr(t) = sigmoid(act); % dynamic learning rate
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
        case 'absCueD' % RW-PH-M-D
            
            stim_count = count(t,stim(t));
            decay = eta + nu*exp(-lambda*stim_count);
            
            cueLR(t) = g*cuePE;
            outLR(t) = k*absPE;
            
            act = cueLR(t) + outLR(t) + decay;
            lr(t) = sigmoid(act); % dynamic learning rate
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
    end % end case
end % loop over trials
V(t+1) = Vdeck(t+1,stim(t+1));

% compute likelihood
var = (nansum((lr_emp-lr).^2))/(Ntrials); % the sample variance
lik = Ntrials*(log(sqrt(2*pi*var))+.5); % This is already the negative log lik

end
