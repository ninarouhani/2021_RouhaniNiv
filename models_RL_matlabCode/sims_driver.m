%% generate simulated data

% load fit parameters, reward sequences, stim sequences, value estimates
% and learning rates (rows are trial numbers, columns are subjects) 
load('paramsConfMat.mat') % parameters fit by value model
load('rewDF.mat') % reward outcomes 
load('stimDF.mat') % scene category sequences
load('valDF.mat') % value estimates
load('lrDF.mat') % empirical (subject) learning rates 

models = {'RW','expDecay','absPE','cuePE','absCue','absD','cueD','absCueD'};
nSample = 100; % number to be sampled

% assign parameters to models 
params_RW = paramsConfMat(:,2);
params_expDecay = paramsConfMat(:,3:5);
params_absPE = paramsConfMat(:,6:7);
params_cuePE = paramsConfMat(:,8:9);
params_absCue = paramsConfMat(:,10:12);
params_absD = paramsConfMat(:,13:16);
params_cueD = paramsConfMat(:,17:20);
params_absCueD = paramsConfMat(:,21:25);

% extract stim and reward sets using random sample
nSubs = size(rewDF,2);
nTrials= size(rewDF,1);
randSamples = randsample(nSubs,nSample);
stimS = stimDF(:,randSamples);
rewS = rewDF(:,randSamples);
valS = valDF(:,randSamples);
lrS = lrDF(:,randSamples);

% generate learning-rate data from fit parameters 
for m = 1:length(models)
    
    model = char(models(m));
    params = eval(['params_' char(model)]);
    
    % sample 100 params (from previous random selection)
    sampleParams = table2array(params(randSamples,:));
    
    for s = 1:nSample
        
        reward = rewS(:,s); % reward sequence
        stim = stimS(:,s); % scene category sequence
        lr_emp = lrS(:,s); % empirical learning rates for checking likelihood
        
        g = sampleParams(s,:);
        
        [lik,lr] = sims_model(model,stim,reward,lr_emp,g);
        
        % add noise to the learning rate 
        mu = 0.05; % empirical noise
        sigma = 0.025;
        noise = normrnd(mu,sigma,[length(lr),1]);
        lr = lr + noise;
        
        lrmat(:,s,m) = lr; % model-generated learning rates (trial, subject, model)   
        
    end
end

%% fit generated learning rates 

for m = 1:length(models) % model fit on all simulated datasets 
    
    fprintf('modFit %d...\n',m);
    
    clear Fit
    
    Fit.Model = char(models(m)); % the model being run on the output of all the other models
    
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
        case 'cuePE' % signed cue RPE (RW-M)
            Fit.parms   = {'eta','g'};
            Fit.LB      = [-10, -20]; % lower bound
            Fit.UB      = [10, 20]; % upper bound
        case 'absCue' % unsigned outcome and signed cue RPE (RW-PH-M)
            Fit.parms   = {'eta','g','k'};
            Fit.LB      = [-10, -20, -20]; % lower bound
            Fit.UB      = [10, 20, 20]; % upper bound
        case 'absD' % unsigned outcome RPE + decay (RW-PH-D)
            Fit.parms   = {'eta','nu','lambda','k'};
            Fit.LB      = [-10, -15, 0, -20]; % lower bound
            Fit.UB      = [10, 15, 10, 20]; % upper bound 
        case 'cueD' % signed cue RPE + decay (RW-M-D)
            Fit.parms   = {'eta','nu','lambda','g'};
            Fit.LB      = [-10, -15, 0, -20]; % lower bound
            Fit.UB      = [10, 15, 10, 20]; % upper bound
        case 'absCueD' % unsigned outcome and signed cue RPE + decay (RW-PH-M-D)
            Fit.parms   = {'eta','nu','lambda','k','g'};
            Fit.LB      = [-10, -15, 0, -20, -20]; % lower bound
            Fit.UB      = [10, 15, 10, 20, 20]; % upper bound
        otherwise
            error('called with unsupported model %s',Fit.Model)
    end
    Fit.Nparms = length(Fit.parms); % number of parameters to fit 
    Fit.init   = NaN(nSample,Fit.Niter,Fit.Nparms); % initial values for fitting
    
    for m2 = 1:length(models) % tested dataset
        
        fprintf('modData %d...\n',m2);

        % fit each subject separately
        for s = 1:nSample
            
            reward = rewS(:,s); % reward sequence
            stim = stimS(:,s); % scene category sequence
            lr_emp = lrmat(:,s,m2); % model-generated learning rates 

            % function to fit
            fitfunction = @(X) sims_model(Fit.Model,stim,reward,lr_emp,X);
            
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
        
        end % subject
        
        likConfMat(:,m2,m) = Fit.result.bestFit(:,size(Fit.result.bestFit,2));
        bicConfMat(:,m2,m) = likConfMat(:,m2,m)+(Fit.Nparms/2)*log(nTrials);
        bestFit{m,m2} = Fit.result.bestFit;
        
    end % simulated model
    
end % fit model 
