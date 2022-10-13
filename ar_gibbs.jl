

Σ₀  = Matrix(I,5,5)*4000;
β₀       = zeros(5,1);        # For mu,phi1,phi2,phi3,phi4
sig2_d  = 0.5;             # Initial value for the Gibbs sampler
nu0      = 0.0;
d0      = 0.0;               # For SIG2_V OF INDIVIDUAL COMPONENT
XprimX  = Xlag'*Xlag;       # used repeatedly in the calculation
Σ₀Inv   = Σ₀\I
XprimY  = Xlag'*Y;


N0 = 2000;                  # Burn-in phase (number of draws to leave-out)
MM = 5000;                  # Number to save
CAPN = N0 + MM;             # Total number of draws
                            # In the original program, they use N0 = 500 and MM = 2000 and from
                            # the 2000 save every 5th iteration. I follow the book with 1K and 4K

(beta_dist, sigma_dist) = test.gibbs_ols(Xlag,Y,Σ₀Inv,sig2_d,nu0,d0,T,β₀,CAPN,N0,MM);
[mean(beta_dist,dims=2) std(beta_dist,dims=2)]
#[mean(sigma_dist,dims=2) std(sigma_dist,dims=2)] 