# set working directory
dir = paste(getwd(),'/datafiles/',sep="")
setwd(dir)

# load packages 
library(reshape2)
library(ggplot2)
library(lme4)
library(rstan)
library(loo)
library(plyr)
library(ggsignif)

# EXPERIMENT 1

# load data (no NANs)
memory <- read.csv("exp1_old") # just old items

# prepare data
nSubj<-length(unique(memory$subj_idx))

subs = unique(memory$subj_idx)
nTrials = as.data.frame(table(memory$subj_idx))
nTrials = nTrials$Freq
maxTrials = max(nTrials)

cue<-array(0,c(nSubj,maxTrials)) # binary (1,0: cue event or not)
out<-array(0,c(nSubj,maxTrials)) # binary (1,0: outcome event or not)
reward<-array(0,c(nSubj,maxTrials))
estimate<-array(0,c(nSubj,maxTrials))
out_absPE<-array(0,c(nSubj,maxTrials))
out_PE<-array(0,c(nSubj,maxTrials))
confHit<-array(0,c(nSubj,maxTrials))
trialNumLearn<-array(0,c(nSubj,maxTrials))

for (i in 1:nSubj) {
    cue[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$cueD;
    out[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$outD;
    reward[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$reward;
    estimate[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$estimate;
    out_absPE[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$out_absPE;
    out_PE[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$out_PE;
    confHit[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$confHit;
    trialNumLearn[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$trialNumLearn;
}

# data structure
standata = list(nSubj=nSubj, nTrials=nTrials, maxTrials=maxTrials, out=out, cue=cue, reward=reward, estimate=estimate, out_absPE=out_absPE, out_PE=out_PE, confHit=confHit, trialNumLearn=trialNumLearn)

# run model
options(mc.cores = parallel::detectCores())
fit<- stan(file = 'memory_exp1.stan', data = standata, iter = 1500, warmup=500, chains = 4)

# EXPERIMENT 2

# load data (no NANs)
memory <- read.csv("exp2_old") # just old items (explicit and implicit memory version run separately)

# prepare data
nSubj<-length(unique(memory$subj_idx))

subs = unique(memory$subj_idx)
nTrials = as.data.frame(table(memory$subj_idx))
nTrials = nTrials$Freq
maxTrials = max(nTrials)

cue<-array(0,c(nSubj,maxTrials)) # binary (1,0: cue event or not)
out<-array(0,c(nSubj,maxTrials)) # binary (1,0: outcome event or not)
reward<-array(0,c(nSubj,maxTrials))
estimate<-array(0,c(nSubj,maxTrials))
out_absPE<-array(0,c(nSubj,maxTrials))
out_PE<-array(0,c(nSubj,maxTrials))
confHit<-array(0,c(nSubj,maxTrials))
cue_absPE<-array(0,c(nSubj,maxTrials))
cue_PE<-array(0,c(nSubj,maxTrials))
trialNumLearn<-array(0,c(nSubj,maxTrials))

for (i in 1:nSubj) {
  cue[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$cueD;
  out[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$outD;
  reward[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$reward;
  estimate[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$estimate;
  out_absPE[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$out_absPE;
  out_PE[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$out_PE;
  confHit[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$confHit;
  cue_absPE[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$cue_absPE;
  cue_PE[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$cue_PE;
  trialNumLearn[i,1:nTrials[i]] = subset(memory,subj_idx==subs[i])$trialNumLearn;
}

# data structure
standata = list(nSubj=nSubj, nTrials=nTrials, maxTrials=maxTrials, out=out, cue=cue, reward=reward, estimate=estimate, out_absPE=out_absPE, out_PE=out_PE, confHit=confHit, cue_absPE=cue_absPE, cue_PE=cue_PE, trialNumLearn=trialNumLearn)

# run model
options(mc.cores = parallel::detectCores())
fit<- stan(file = 'memory_exp2.stan', data = standata, iter = 1500, warmup=500, chains = 4)
