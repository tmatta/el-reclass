//============================================================================//
// Simultanious growth and survival models 
//  submodel 1:  mixed effects regression with random intercept and slope  
//  submodel 2:  discrete time survival model
//============================================================================//

data {
  int<lower=0> N_g;               // no. repeated measures observations
  int<lower=0> N_s;               // no. survival observations  
  int<lower=0> J;                 // no. subjects
  int<lower=0> K;                 // no. schools
  int<lower=0> n_j[J];            // no. repeated measures per unit j 
  int<lower=1> p_g;               // no. growth fixed effects
  int<lower=1> q_g2;              // no. growth lvl-2 random effects
  int<lower=1> q_g3;              // no. growth lvl-3 random effects
  int<lower=1> p_s;               // no. survival fixed effects
  int<lower=0> q_s;               // no. survival lvl-2 random effects

  vector[N_g] Y_g;                // repeated measures outcome
  int<lower=0, upper=1> Y_s[N_s]; // discrete survival outcome

  row_vector[p_g] X_g[N_g];       // fixed effects design matrix, growth
  row_vector[p_s] X_s[N_s];       // fixed effects design matrix, survival
  row_vector[q_g2] Z_g2[N_g];     // random effects design matrix, growth lvl-2 
  row_vector[q_g3] Z_g3[N_g];     // random effects design matrix, growth lvl-3    
  row_vector[q_s] Z_s[N_s];       // random effects design matrix, survival 

  int studid_g[N_g];              // growth model student ID 
  int studid_s[N_s];              // survival model student ID   
  int schid_g[N_g];               // growth model school ID 
  int schid_s[N_s];               // survival model school ID 

  matrix[p_g, 2] pr_beta;         // prior means and SDs for beta
  matrix[p_s, 2] pr_alpha;        // prior means and SDs for alpha
  vector[2] pr_lambda;            // prior mean and SDs for lambda
}

parameters {
  vector[p_g] beta;                
  cholesky_factor_corr[q_g2] L2;     
  cholesky_factor_corr[q_g3] L3;   
  vector[q_g2] z_zeta_g2[J];        
  vector[q_g3] z_zeta_g3[K];         
  vector<lower=0>[q_g2] zeta_g2_sd;  
  vector<lower=0>[q_g3] zeta_g3_sd;  
  real<lower=0> resid_sd;              

  vector[p_s] alpha;           
  real lambda;                  
  vector[q_s] zeta_s[K];       
  real<lower = 0> zeta_s_sd;   
}

transformed parameters {
  vector[q_g2] zeta_g2[J];                 
  vector[q_g3] zeta_g3[K];               
  matrix[q_g2, q_g2] Psi2;                
  matrix[q_g3, q_g3] Psi3;                
  
  Psi2 = diag_pre_multiply(zeta_g2_sd, L2);   
  Psi3 = diag_pre_multiply(zeta_g3_sd, L3);  
  
  for (jj in 1:J)
    zeta_g2[jj] = Psi2 * z_zeta_g2[jj];   
  
  for (kk in 1:K)
    zeta_g3[kk] = Psi3 * z_zeta_g3[kk];      
}

model {
  vector[N_g] mu;
  vector[N_s] haz;
  
  L2 ~ lkj_corr_cholesky(1.5);
  L3 ~ lkj_corr_cholesky(1.5);

  for (j in 1:J)
    z_zeta_g2[j] ~ normal(0,1);

  for (kk in 1:K){  
    z_zeta_g3[kk] ~ normal(0,1);
    zeta_s[kk] ~ normal(0, zeta_s_sd);
  }

  for (pp in 1:p_g) 
    beta[pp] ~ normal(pr_beta[pp, 1], pr_beta[pp, 2]);  

  for (pp in 1:p_s) 
    alpha[pp] ~ normal(pr_alpha[pp, 1], pr_alpha[pp, 2]);  

    lambda ~ normal(pr_lambda[1], pr_lambda[2]);  

  for (nn in 1:N_g)     
    mu[nn] = X_g[nn] * beta + Z_g2[nn] * zeta_g2[studid_g[nn]] + Z_g3[nn] * zeta_g3[schid_g[nn]];

  for (nn in 1:N_s)     
    haz[nn] = X_s[nn] * alpha + zeta_g2[studid_s[nn], 1]' * lambda + Z_s[nn] * zeta_s[schid_s[nn]];

  Y_g ~ normal(mu, resid_sd); 
  Y_s ~ bernoulli_logit(haz);
}  
