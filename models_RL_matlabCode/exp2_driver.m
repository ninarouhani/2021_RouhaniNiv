
%% load data 
load('exp2_learning.mat'); % loads 'cueOut' each cell contains each subject's learning data in a table 
models = {'RW','expDecay','absPE','cuePE','absCue','absD','cueD','absCueD'}; % fits all models in a loop

%% fit
for m = 1:length(models)
    
    fprintf('modFit %d...\n',m);
    
    clear Fit
    
    % fit options
    Fit.Nsubjects = length(cueOut);
    Fit.Nonlinoption = 'fmincon'; % minimization algorithm: fmincon or fminsearch
    Fit.Niter = 30; % number of iterations to run for each fit 
    Fit.Model = char(models(m));
    
    modelVal = 1; % use model values (1) or participant-generated values (0)
    decayFun = 0; % decay is either linear (1) or exponential (0)
    
    switch Fit.Model
        case 'RW' % basic RL model (RW)
            Fit.parms   = {'alpha'};
            Fit.LB      = 0; % lower bound
            Fit.UB      = 1; % upper bound
        case 'expDecay' % exponential decay model (RW-D)
            if decayFun == 1 % linear decay
                Fit.parms   = {'eta','lambda'}; 
                Fit.LB      = [-10, 0]; % lower bound
                Fit.UB      = [10, 10]; % upper bound
            else % exponential decay
                Fit.parms   = {'eta','nu','lambda'}; 
                Fit.LB      = [-10, -15, 0]; % lower bound
                Fit.UB      = [10, 15, 10]; % upper bound
            end
        case 'absPE' % unsigned outcome RPE (RW-PH) 
            Fit.parms   = {'eta','k'};
            Fit.LB      = [-10, -20]; % lower bound
            Fit.UB      = [10, 20]; % upper bound
        case 'cuePE' % signed cue RPE (RW-M) 
            Fit.parms   = {'eta','g'}; 
            Fit.LB      = [-10, -20]; % lower bound
            Fit.UB      = [10, 20]; % upper bound
        case 'absCue' % unsigned outcome and signed cue RPE (RW-PH-M) 
            Fit.parms   = {'eta','g','k'}; 
            Fit.LB      = [-10, -20, -20]; % lower bound
            Fit.UB      = [10, 20, 20]; % upper bound
        case 'absD' % unsigned outcome RPE + decay (RW-PH-D) 
            if decayFun == 1 % linear decay
                Fit.parms   = {'eta','lambda','k'};
                Fit.LB      = [-10, 0, -20]; % lower bound
                Fit.UB      = [10, 10, 20]; % upper bound
            else % exponential decay
                Fit.parms   = {'eta','nu','lambda','k'}; 
                Fit.LB      = [-10, -15, 0, -20]; % lower bound
                Fit.UB      = [10, 15, 10, 20]; % upper bound
            end
        case 'cueD' % signed cue RPE + decay (RW-M-D)
            if decayFun == 1 % linear decay
                Fit.parms   = {'eta','lambda','g'};
                Fit.LB      = [-10, 0, -20]; % lower bound
                Fit.UB      = [10, 10, 20]; % upper bound
            else % exponential decay
                Fit.parms   = {'eta','nu','lambda','g'}; 
                Fit.LB      = [-10, -15, 0, -20]; % lower bound
                Fit.UB      = [10, 15, 10, 20]; % upper bound
            end
        case 'absCueD' % unsigned outcome and signed cue RPE + decay (RW-PH-M-D) 
            if decayFun == 1 % linear decay      
                Fit.parms   = {'eta','lambda','k','g'};
                Fit.LB      = [-10, 0, -20, -20]; % lower bound
                Fit.UB      = [10, 10, 20, 20]; % upper bound
            else % exponential decay
                Fit.parms   = {'eta','nu','lambda','k','g'};
                Fit.LB      = [-10, -15, 0, -20, -20]; % lower bound
                Fit.UB      = [10, 15, 10, 20, 20]; % upper bound
            end
        otherwise
            error('called with unsupported model %s',Fit.Model)
    end
    Fit.Nparms = length(Fit.parms); % number of parameters to fit (we will fit these per room so it is really x2 in practice)
    
    Fit.init   = NaN(Fit.Nsubjects,Fit.Niter,Fit.Nparms); % initial values for fitting
    
    % fit each subject separately
    for s = 1:Fit.Nsubjects
        fprintf('Subject %d...\n',s);
        
        reward = cueOut{s,2}.reward; % reward outcomes
        valEst = cueOut{s,2}.estimate; % subject value estimates 
        stim = grp2idx(cueOut{s,2}.deck); % scene category (1 or 2)
        cuePE = cueOut{s,2}.deckSplit; % empirical signed cue RPE
        absPE = cueOut{s,2}.absPE; % empirical unsigned outcome RPE

        % function to fit
        fitfunction = @(X) exp2_models(Fit.Model,absPE,cuePE,stim,valEst,reward,modelVal,decayFun,X);
        
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
    
    likConfMat(:,m) = Fit.result.bestFit(:,size(Fit.result.bestFit,2));
    bestFit{m} = Fit.result.bestFit;
    
end % end models
