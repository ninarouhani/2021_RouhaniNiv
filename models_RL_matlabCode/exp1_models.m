function [lik,V,var,PEsigned,lr] = exp1_models(model,absPE,valEst,reward,room,stim,modelVal,X)
% Input:
% model = model specified 
% absPE = empirical unsigned outcome RPE
% valEst = the value the subject estimated for this stimulus
% reward = the reward given in this trial
% room = indicates a new room (binary, to reset values) 
% stim = indicates high or low risk/variance 
% modelVal = either use model-generated or empirical RPEs 
% Likelihood is calculated assuming that guesses are linear in prediction
% plus gaussian noise (linear regression likelihood)
Ntrials = length(reward);

% set parameters
switch model
    case 'RW' % (RW)
        alpha = X;
    case 'expDecay' % (RW-D)
        eta = X(1);
        nu = X(2);
        lambda = X(3);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
    case 'absPE' % (RW-PH)
        eta = X(1);
        k = X(2);
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
    case 'absD' % (RW-PH-D)
        eta = X(1);
        nu = X(2);
        lambda = X(3);
        k = X(4);
        outLR = nan(Ntrials,1);
        lr = nan(Ntrials,1);
        count = nan(Ntrials,2);
        count(1,:) = 0;
end

V = nan(Ntrials,1);

V(1) = 50; % start with average value of 50

for t = 1:Ntrials-1
    
    % model-generated RPEs
    PEsigned = reward(t) - V(t); 
    absPE_mod = abs(PEsigned)/100;
    
    switch model
        
        case 'RW' % RW
            if room(t) % reset values with each new 'room' or context
                V(t) = 50;
            end
            
            V(t+1) = V(t) + alpha*PEsigned;
            
        case 'expDecay' % RW-D
            
            if room(t)
                V(t) = 50;
            end
            
            stim_count = count(t,stim(t)); % variance count
            decay = eta + nu*exp(-lambda*stim_count);
            lr(t) = sigmoid(decay);
                   
            V(t+1) = V(t) + lr(t)*PEsigned;
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts
            
        case 'absPE' % RW-PH
            
            if room(t)
                V(t) = 50;
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
            
            act = eta + outLR(t);
            lr(t) = sigmoid(act);
            
            V(t+1) = V(t) + lr(t)*PEsigned;
            
        case 'absD' % RW-PH-D
            
            if room(t)
                V(t) = 50;
            end
            
            stim_count = count(t,stim(t));
            decay = eta + nu*exp(-lambda*stim_count);
            
            if modelVal == 1 % model-generated values
                outLR(t) = k*absPE_mod;
                
            else % participant-generated values
                if isnan(absPE(t))
                    outLR(t) = 0; % learning rate is decay
                else
                    absRPE = absPE(t)/100;
                    outLR(t) = k*absRPE;
                end
            end

            act = decay + outLR(t);
            lr(t) = sigmoid(act);
            
            V(t+1) = V(t) + lr(t)*PEsigned;
            
            count(t+1,:) = count(t,:); % copy over counts
            count(t+1,stim(t)) = stim_count + 1; % update counts

    end % switch model   
end % loop over trials
% PEsigned(t+1) = reward(t+1) - V(t+1); % for PE vectors

% check to see if NaNs in model values 
check = sum(isnan(V));
if check > 0
    fprintf('there is a NaN')    
    error('stopped because of NaN')
end

nTrials = sum(~isnan(valEst)); % excluding NaNs

% compute likelihood
var = (nansum((valEst-V).^2))/(nTrials); % the sample variance 
lik = nTrials*(log(sqrt(2*pi*var))+.5); % This is already the negative log lik

end
