function [lik,V,var,PEsigned,lr] = exp2_models(model,absPE,cuePE,stim,valEst,reward,modelVal,decayFun,X)
% Input:
% model = model specified 
% absPE = empirical unsigned outcome RPE
% cuePE = empirical signed cue RPE
% stim = 1 or 2 (the scene category presented on this trial)
% valEst = the value the subject estimated for this stimulus
% reward = the reward given in this trial
% modelVal = either use model or participant-generated RPEs
% decayFun = either use linear or exponential decay
% Likelihood is calculated assuming that guesses are linear in prediction
%   plus gaussian noise (linear regression likelihood)
Ntrials = length(stim); % this includes NaNs 

% set parameters
switch model
    case 'RW' % RW
        alpha = X;
    case 'expDecay' % RW-D
        if decayFun == 1 % linear decay
            eta = X(1);
            lambda = X(2);
        else % exponential decay
            eta = X(1);
            nu = X(2);
            lambda = X(3);
        end
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
        if decayFun == 1 % linear decay
            eta = X(1);
            lambda = X(2);
            k = X(3);
        else % exponential decay
            eta = X(1);
            nu = X(2);
            lambda = X(3);
            k = X(4);
        end
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
    case 'cueD' % RW-M-D
        if decayFun == 1 % linear decay
            eta = X(1);
            lambda = X(2);
            g = X(3);
        else % exponential decay
            eta = X(1);
            nu = X(2);
            lambda = X(3);
            g = X(4);
        end
        cueLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
    case 'absCueD' % RW-PH-M-D
        if decayFun == 1 % linear decay
            eta = X(1);
            lambda = X(2);         
            k = X(3);
            g = X(4);
        else % exponential decay
            eta = X(1);
            nu = X(2);
            lambda = X(3);
            k = X(4);
            g = X(5);
        end
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

    % model-generated outcome RPEs
    PEsigned = reward(t) - V(t);
    absPE_mod = abs(PEsigned)/100;
    
    % model-generated cue RPEs (contingent on current deck)
    if stim(t)==1
        cuePE_mod = (Vdeck(t,stim(t)) - Vdeck(t,2))/100;
    else
        cuePE_mod = (Vdeck(t,stim(t)) - Vdeck(t,1))/100;
    end

    switch model
        case 'RW' % RW
            
            Vdeck(t+1,stim(t)) = V(t) + alpha*PEsigned;
            
        case 'expDecay' % RW-D
            
            stim_count = count(t,stim(t));
            
            if decayFun == 1 % linear decay
                decay = eta + (-lambda*stim_count);
            else % exponential decay
                decay = eta + nu*exp(-lambda*stim_count);
            end
            
            lr(t) = sigmoid(decay);
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
        case 'absPE' % RW-PH
            
            if modelVal == 1 % model-generated values
                outLR(t) = k*absPE_mod;
                
            else % participant-generated values             
                if isnan(absPE(t))
                    outLR(t) = 0;
                else
                    absRPE = absPE(t)/100;
                    outLR(t) = k*absRPE;
                end
            end
            
            act = eta + outLR(t);
            lr(t) = sigmoid(act);
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
        case 'cuePE' % RW-M
            
            if modelVal == 1 % model-generated values
                cueLR(t) = g*cuePE_mod;
                
            else % participant-generated values         
                if isnan(cuePE(t))
                    cueLR(t) = 0;
                else
                    cRPE = cuePE(t)/100;
                    cueLR(t) = g*cRPE;
                end
            end
            
            act = eta + cueLR(t);
            lr(t) = sigmoid(act);
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
        case 'absCue' % RW-PH-M
            
            if modelVal == 1 % model-generated values
                cueLR(t) = g*cuePE_mod;
                outLR(t) = k*absPE_mod;
                
            else % participant-generated values
                if isnan(cuePE(t))
                    cueLR(t) = 0;
                else
                    cRPE = cuePE(t)/100;
                    cueLR(t) = g*cRPE;
                end
                
                if isnan(absPE(t))
                    outLR(t) = 0;
                else
                    absRPE = absPE(t)/100;
                    outLR(t) = k*absRPE;
                end
            end
            
            act = eta + cueLR(t) + outLR(t);
            lr(t) = sigmoid(act);
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
        case 'absD' % RW-PH-D
            
            stim_count = count(t,stim(t));
            
            if decayFun == 1 % linear decay
                decay = eta + (-lambda*stim_count);
            else % exponential decay
                decay = eta + nu*exp(-lambda*stim_count);
            end
            
            if modelVal == 1 % model-generated values
                outLR(t) = k*absPE_mod;
                
            else % participant-generated values
                if isnan(absPE(t))
                    outLR(t) = 0; 
                else
                    absRPE = absPE(t)/100;
                    outLR(t) = k*absRPE;
                end
            end
            
            act = decay + outLR(t);
            lr(t) = sigmoid(act);
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
        case 'cueD' % RW-M-D
            
            stim_count = count(t,stim(t));
            
            if decayFun == 1 % linear decay
                decay = eta + (-lambda*stim_count);
            else % exponential decay
                decay = eta + nu*exp(-lambda*stim_count);
            end
            
            if modelVal == 1 % model-generated values
                cueLR(t) = g*cuePE_mod;
                
            else % participant-generated values
                if isnan(cuePE(t))
                    cueLR(t) = 0; % learning rate is decay
                else
                    cRPE = cuePE(t)/100;
                    cueLR(t) = g*cRPE;
                end
            end
            
            act = decay + cueLR(t);
            lr(t) = sigmoid(act);
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
        case 'absCueD' % RW-PH-M-D
            
            stim_count = count(t,stim(t));
            
            if decayFun == 1 % linear decay
                decay = eta + (-lambda*stim_count);
            else % exponential decay
                decay = eta + nu*exp(-lambda*stim_count);
            end  
            
            if modelVal == 1 % model-generated values
                cueLR(t) = g*cuePE_mod;
                outLR(t) = k*absPE_mod;
                
            else % participant-generated values
                if isnan(cuePE(t))
                    cueLR(t) = 0;
                else
                    cRPE = abs(cuePE(t))/100;
                    cueLR(t) = g*cRPE;
                end
                
                if isnan(absPE(t))
                    outLR(t) = 0;
                else
                    absRPE = absPE(t)/100;
                    outLR(t) = k*absRPE;
                end
            end
            
            act = cueLR(t) + outLR(t) + decay;  
            lr(t) = sigmoid(act);
            
            Vdeck(t+1,stim(t)) = V(t) + lr(t)*(PEsigned);
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts

    end % end case
end % loop over trials
V(t+1) = Vdeck(t+1,stim(t+1));
% PEsigned(t+1) = reward(t+1) - V(t+1); % for PE vectors

% check to see if NaNs in model values
check = sum(isnan(V));
if check > 0
    fprintf('there is a NaN')
    error('stopped because of NaN')
end

nTrials = sum(~isnan(valEst)); % excluding NaNs

% compute likelihood
var = (nansum((valEst-V').^2))/nTrials; % the sample variance 
lik = nTrials*(log(sqrt(2*pi*var))+.5); % This is already the negative log lik

end
