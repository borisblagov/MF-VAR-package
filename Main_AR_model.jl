using Revise
using LinearAlgebra
using Distributions
using DelimitedFiles
using Plots
using BenchmarkTools
using Profile
# using MyPkg


#-----------
# parameters
nm = 1; # no. of monthly vars
nq = 1; # no. of quarterly vars
nex = 1; # no. of exogenous variables
const p  = 6; # no. of lags
hyp = [0.40;    0.10;    1.00;    2.30;    5.3368]; 

# Gibbs setup
nsim     = 12;# number of draws from Posterior Density
nburn    = 10;


# data
const YDATA = readdlm("us.txt",',');


include("AR_functions.jl")  
(YYdraws) = test.mf_var(YDATA,nm,nq,nex,p,hyp,nsim,nburn);
@time test.mf_var(YDATA,nm,nq,nex,p,hyp,nsim,nburn);


YY_mean = dropdims(mean(YYdraws;dims=3);dims=3);

plot(YY_mean,layout=(2,1),legend=:bottomright)