%% load behavior
load('exp1_learning.mat') % loads 'cueOut' each cell contains each subject's learning data in a table format

%% fit options
clear Fit
Fit.Nsubjects = length(cueOut);

% different models to fit (fits each model one at a time) 
Fit.Model    = 'RW'; % (RW)
% Fit.Model    = 'expDecay'; % (RW-D)
% Fit.Model    = 'absPE'; % (RW-PH)
% Fit.Model    = 'absD'; % (RW-PH-D)

modelVal = 1; % use model values (1) or participant-generated values (0)

Fit.Nonlinoption = 'fmincon'; % minimization algorithm: fmincon or fminsearch
Fit.Niter = 30; % number of iterations to run for each fit

switch Fit.Model
    case 'RW' % basic RL model (RW)
        Fit.parms   = {'alpha'};
        Fit.LB      = 0; % lower bound
        Fit.UB      = 1; % upper bound
    case 'expDecay' % exponential decay model (RW-D)
        Fit.parms   = {'eta','nu','lambda'};
        Fit.LB      = [-10, -15, 0]; % lower bound
        Fit.UB      = [10, 15, 10]; % upper bound
    case 'absPE' % unsigned outcome RPE (RW-PH)
        Fit.parms   = {'eta','k'};
        Fit.LB      = [-10, -20]; % lower bound
        Fit.UB      = [10, 20]; % upper bound
    case 'absD' % unsigned outcome RPE + decay model (RW-PH-D)
        Fit.parms   = {'eta','nu','lambda','k'};
        Fit.LB      = [-10, -15, 0, -20]; % lower bound
        Fit.UB      = [10, 15, 10, 20]; % upper bound
    otherwise
        error('called with unsupported model %s',Fit.Model)
end
Fit.Nparms = length(Fit.parms); % number of parameters to fit (we will fit these per room so it is really x2 in practice)

Fit.init   = NaN(Fit.Nsubjects,Fit.Niter,Fit.Nparms); % initial values for fitting

% fit each subject separately
for s = 1:Fit.Nsubjects
    fprintf('Subject %d...\n',s);
    
    reward = cueOut{s,2}.reward; % reward outcome
    valEst = cueOut{s,2}.estimate; % value estimate
    room = cueOut{s,2}.room; % binary (new room or context = 1, same room = 0)
    stim = grp2idx(cueOut{s,2}.risk); % high or low variance/risk (1 = high, 2 = low) 
    absPE = cueOut{s,2}.absPE; % empirical unsigned outcome RPE
    
    % function to fit
    switch Fit.Model
        case {'RW','expDecay','absPE','absD'}
            fitfunction = @(X) exp1_models(Fit.Model,absPE,valEst,reward,room,stim,modelVal,X);
    end
    
    for iter = 1:Fit.Niter
        
        % random positive initial parameter values within the range
        Fit.init(s,iter,:) = abs(rand(1,length(Fit.LB)).*(Fit.UB-Fit.LB)+Fit.LB);
        
        % run nonlin minimization
        switch Fit.Nonlinoption
            case 'fmincon'
                [res,lik] = ...
                    fmincon(fitfunction,squeeze(Fit.init(s,iter,:)),[],[],[],[],Fit.LB,Fit.UB,[],...
                    optimset('maxfunevals',5000,'Display','off','maxiter',2000,'GradObj','off','DerivativeCheck','off','LargeScale','on','Algorithm','interior-point','Hessian','off'));
                
            case 'fminsearch'
                [res,lik] = ...
                    fminsearch(fitfunction,squeeze(Fit.init(s,iter,:)),...
                    optimset('maxfunevals',5000,'maxiter',3000,'GradObj','off','DerivativeCheck','off','LargeScale','on','Algorithm','active-set','Hessian','off'));
                
            otherwise
                error('Called with a non-supported minimization function %s',Fit.Nonlinoption)
        end
        
        % save parameter fits (# will depend on model) and likelihood
        Fit.result.parms(s,:,iter) = res;
        Fit.result.lik(s,iter) = lik;
        
    end % fitting iterations
    
    % extract best fit across all iterations
    
    [a,b] = min(Fit.result.lik(s,:),[],2);
    
    Fit.result.bestFit(s,:) = [s,...
        Fit.result.parms(s,:,b),...
        Fit.result.lik(s,b)];
    
end % subjects
