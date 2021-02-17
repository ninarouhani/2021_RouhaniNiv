data {
	int nSubj;
	int nTrials[nSubj];
	int maxTrials;

	int cue[nSubj,maxTrials];
  int out[nSubj,maxTrials];
	int reward[nSubj,maxTrials];
	int estimate[nSubj,maxTrials];
  real out_absPE[nSubj,maxTrials];
  real out_PE[nSubj,maxTrials];
  int confHit[nSubj,maxTrials];
	int trialNumLearn[nSubj,maxTrials];
}

parameters {
  real beta_cueOut_pop;
  real beta_absPE_cue_pop;
  real beta_PE_cue_pop;
  real beta_absPE_out_pop;
  real beta_PE_out_pop;
	real alpha_pop;
	
  real beta_cueOut_subj[nSubj];
  real beta_absPE_cue_subj[nSubj];
  real beta_PE_cue_subj[nSubj];
  real beta_absPE_out_subj[nSubj];
  real beta_PE_out_subj[nSubj];
	real alpha_s[nSubj];

	real<lower=0> sigma; //Variance
  real<lower=0> noise;
}

model {
	beta_cueOut_pop ~ normal(0,1);
  beta_absPE_cue_pop ~ normal(0,1);
  beta_PE_cue_pop ~ normal(0,1);
  beta_absPE_out_pop ~ normal(0,1);
  beta_PE_out_pop ~ normal(0,1);
	alpha_pop ~ normal(0,1);

  sigma ~ gamma(1,0.5);
  noise ~ normal(2,1);

	for (s in 1:nSubj) {

    beta_cueOut_subj[s]~normal(beta_cueOut_pop,sigma);    
    beta_absPE_cue_subj[s]~normal(beta_absPE_cue_pop,sigma);
    beta_PE_cue_subj[s]~normal(beta_PE_cue_pop,sigma);
    beta_absPE_out_subj[s]~normal(beta_absPE_out_pop,sigma);
    beta_PE_out_subj[s]~normal(beta_PE_out_pop,sigma);
		alpha_s[s]~normal(alpha_pop,sigma);
    
		for (t in 1:(nTrials[s])) {
      
      confHit[s,t] ~ normal(alpha_s[s] + beta_cueOut_subj[s]*cue[s,t] + beta_absPE_out_subj[s]*(out_absPE[s,t]*out[s,t]) + beta_absPE_cue_subj[s]*(out_absPE[s,t]*cue[s,t]) + beta_PE_out_subj[s]*(out_PE[s,t]*out[s,t]) + beta_PE_cue_subj[s]*(out_PE[s,t]*cue[s,t]), noise);
      
		}
	} 
}

generated quantities {
  matrix[nSubj,maxTrials] lik = rep_matrix(99, nSubj, maxTrials);

    for (s in 1:nSubj) {
      
       for (t in 1:nTrials[s]) {
          lik[s,t] = normal_lpdf(confHit[s,t] | alpha_s[s] + beta_cueOut_subj[s]*cue[s,t] + beta_absPE_out_subj[s]*(out_absPE[s,t]*out[s,t]) + beta_absPE_cue_subj[s]*(out_absPE[s,t]*cue[s,t]) + beta_PE_out_subj[s]*(out_PE[s,t]*out[s,t]) + beta_PE_cue_subj[s]*(out_PE[s,t]*cue[s,t]), noise);
            
    }
  }
}
