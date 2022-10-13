module test
using LinearAlgebra
using Distributions
using DelimitedFiles
export mlag, ols, ols_gibbs

"""
    mlag(X::Array,p::Int64; cnst=1)

Creates a lagged matrix of X with p lags.

Returns **Xlag** and **Y**, such that size(Xlag,1) = T-p and **Y** = X[p+1:end,:]

Adds a constant to the last column by default, can be disabled by cnst=0.

"""
function mlag(X::Array,p::Int64; cnst=1)
    (T,N) = size(X)
    if cnst == 1
        Xlag = ones(T-p,N*p+1);
    else
        Xlag = zeros(T-p,N*p);
    end
    for i = 1:p
        Xlag[:,1+N*(i-1):N+N*(i-1)] = X[1+p-i:end-i,:]
    end
    
    Y = X[1+p:end,:]
    return Xlag, Y, T, N
end




"""
    ols(X::Array{Float64},Y::Array{Float64})

Performs Ordinary Least Squares on data matrix X and dependent array Y.
Returns a vector of parameters B
"""
function ols(X::Array{Float64},Y::Array{Float64})
    B = (X'*X)\(X'*Y)
    return(B)
end

function gibbs_ols(Xlag,Y,Σ₀Inv,sig2_d,nu0,d0,T,β₀,CAPN,N0,MM)
    beta_dist  = zeros(5,MM);   # Matrix to store the empirical distribution of beta
    sigma_dist = zeros(1,MM);   # Matrix to store the empirical distribution of sigma


    for ii = 1:CAPN
        V = Hermitian((Σ₀Inv + (sig2_d\1)*(Xlag'*Xlag))\I)
        β₁ =  V*(Σ₀Inv*β₀ + sig2_d\1*(Xlag'*Y));
        C = cholesky(V)

        β_d = β₁ + C.L*randn(5,1);            # Actual beta draw

        d1  = d0 .+ (Y-Xlag*β_d)'*(Y-Xlag*β_d);

        nu1 = T + nu0;

        sig2inv_d = rand(Gamma(nu1/2,2/d1[1]),1)
        sig2_d = 1/sig2inv_d[1];

        if ii>N0
            beta_dist[:,ii-N0] .= β_d;
            sigma_dist[:,ii-N0] .= sig2_d;
        end 
    
    end

    return beta_dist, sigma_dist
end


function mf_kf(At,Pt,Tb,T0,Zm,Ym,Yq,GAMMAs,GAMMAz,GAMMAc,GAMMAu,sig_qq,sig_mm,sig_mq,sig_qm,LAMBDAz,LAMBDAs,LAMBDAc,LAMBDAu,LAMBDAc_t,LAMBDAs_t,LAMBDAu_t,LAMBDAz_t,p)
    nq = size(Yq,2);
    nm = size(Ym,2);
    T = size(Ym,1);
    At_mat    = zeros(Tb,nq*(p+1));
    Pt_mat_alt = zeros((nq*(p+1)),(nq*(p+1)),Tb);
    for t = 1:Tb
        if ((t+T0)/3)-floor((t+T0)/3)==0
            At1 = At;
            Pt1 = Pt;
            
            # Forecasting
            alphahat = GAMMAs * At1 + GAMMAz * transpose(Zm[t:t,:]) + GAMMAc
            Phat     = GAMMAs * Pt1 * transpose(GAMMAs) + GAMMAu * sig_qq * transpose(GAMMAu);
            Phat     = 0.5*(Phat+transpose(Phat));
            
            yhat     = LAMBDAs * alphahat + LAMBDAz * Zm[t,:] + LAMBDAc;
            nut      = [Ym[t:t,:]'; Yq[t:t,:]'] - yhat
            
            Ft       = LAMBDAs * Phat * transpose(LAMBDAs) + LAMBDAu * sig_mm * transpose(LAMBDAu) + LAMBDAs*GAMMAu*sig_qm*transpose(LAMBDAu) + LAMBDAu*sig_mq*transpose(GAMMAu)*transpose(LAMBDAs)
            Ft       = 0.5*(Ft+transpose(Ft))
            Xit      = LAMBDAs * Phat + LAMBDAu * sig_mq * GAMMAu'
            
            At       = alphahat + Xit'/Ft * nut
            Pt       = Phat     - Xit'/Ft * Xit
            
            At_mat[t:t,:]  = transpose(At)
            Pt_mat_alt[:,:,t:t]  = Pt
    
        else
            At1 = At;
            Pt1 = Pt;
                    
            # Forecasting
            alphahat = GAMMAs * At1 + GAMMAz * Zm[t,:] + GAMMAc
            Phat     = GAMMAs * Pt1 * GAMMAs' + GAMMAu * sig_qq * GAMMAu'
            Phat     = 0.5*(Phat+Phat')
            
            yhat     = LAMBDAs_t * alphahat + LAMBDAz_t * Zm[t,:] + LAMBDAc_t
            nut      = Ym[t:t,:] - yhat
            
            Ft       = LAMBDAs_t * Phat * LAMBDAs_t' + LAMBDAu_t * sig_mm * LAMBDAu_t' + LAMBDAs_t*GAMMAu*sig_qm*LAMBDAu_t' + LAMBDAu_t*sig_mq*GAMMAu'*LAMBDAs_t'
            Ft       = 0.5*(Ft+Ft');
            Xit      = LAMBDAs_t * Phat + LAMBDAu_t * sig_mq * GAMMAu'
            
            At       = alphahat + Xit'/Ft * nut
            Pt       = Phat     - Xit'/Ft * Xit
            
            At_mat[t:t,:]  = transpose(At);
            Pt_mat_alt[:,:,t:t]  = Pt
        end
    end
    return(At_mat,Pt_mat_alt)

end


"""
    mf_ks(Tb,At_mat,Pt_mat_alt,GAMMAs,GAMMAu,GAMMAz,GAMMAc,sig_qq,At_draw,Zm,p,nq)

Performs the Kalman Smoother and returns the smoothed values
"""
function mf_ks(Tb,At_mat,Pt_mat_alt,GAMMAs,GAMMAu,GAMMAz,GAMMAc,sig_qq,AT_draw,Zm,p,nq,nm)
    
    At_draw = zeros(Tb,nq*(p+1));

    for kk=1:p+1
        # this loop selects only the quarterly variables from AT_draw and puts them in At_draw
        At_draw[Tb,(kk-1)*nq+1:kk*nq] = AT_draw[1,kk*nm+(kk-1)*nq+1:kk*(nm+nq)];
    end

    Pmean = zeros(size(Pt_mat_alt[:,:,Tb]));
    for ii = 1:Tb-1
        Att  = At_mat[Tb-ii,:]
        Ptt  = Pt_mat_alt[:,:,Tb-ii]
        Phat = GAMMAs * Ptt * GAMMAs' + GAMMAu * sig_qq * GAMMAu'
        Phat = 0.5.*(Phat+Phat')
        FPhat = svd(Phat);

        inv_Phat= FPhat.U*Diagonal( (1.0./FPhat.S).*(FPhat.S.>1e-12))*FPhat.Vt;
        nut  = At_draw[Tb-ii+1,:]-GAMMAs * Att - GAMMAz * Zm[Tb-ii,:]- GAMMAc
        Amean = Att + (Ptt*GAMMAs')*inv_Phat*nut
        Pmean = Ptt - (Ptt*GAMMAs')*inv_Phat*(Ptt*GAMMAs')'

        # singular value decomposition for the draw
        FPmean = svd(Pmean)
        Pmchol = FPmean.U*Diagonal(sqrt.(FPmean.S))
        At_draw[Tb-ii,:] = (Amean+Pmchol*randn(nq*(p+1),1))'
    end
    return(At_draw,Pmean)
end



    
 
function varprior(nv,nlags,nex,hyp,premom)   
    # Generate matrices with dummy observations
    lambda1 = hyp[1];
    lambda2 = hyp[2];
    lambda3 = convert(Int,hyp[3]);
    lambda4 = hyp[4];
    lambda5 = hyp[5];
    
    # initializations
    dsize = nex + (nlags+lambda3+1)*nv;
    breakss = [0; 0; 0; 0; 0];
    ydu = zeros(dsize,nv);
    xdu = zeros(dsize,nv*nlags+nex);
    
    # dummies for the coefficients of the first lag
    sig = diagm(premom[:,2])
    ydu[1:nv,:] = lambda1*sig
    xdu[1:nv,:] = [lambda1*sig zeros(nv,(nlags-1)*nv+nex)]
    breakss[1] = nv
    
    # dummies for the coefficients of the remaining lags
    if nlags > 1
        ydu[breakss[1]+1:nv*nlags,:] = zeros((nlags-1)*nv,nv)
        j = 1;
        while j <= nlags-1
            xdu[breakss[1]+(j-1)*nv+1:breakss[1]+j*nv,:] = [zeros(nv,j*nv) lambda1*sig*(j+1)^lambda2 zeros(nv,(nlags-1-j)*nv+nex)];
            j = j+1;
        end
        breakss[2] = breakss[1]+(nlags-1)*nv;
    else
        breakss[2] = breakss[1];
    end 
    
    
    # dummies for the covariance matrix of error terms
    ydu[breakss[2]+1:breakss[2]+lambda3*nv,:] = kron(ones(lambda3,1),sig);
    breakss[3] = breakss[2]+lambda3*nv;
    
    # dummies for the coefficents of the constant term
    if nex != 0
        lammean = lambda4*premom[:,1:1]';
        ydu[breakss[3]+1,:] = lammean;
        xdu[breakss[3]+1,:] = [kron(ones(1,nlags),lammean) lambda4];
        breakss[4] = breakss[3]+nex;
    end
    
    # dummies for the covariance matrix of coefficients of different lags
    mumean = diagm(lambda5*premom[:,1]);
    ydu[breakss[4]+1:breakss[4]+nv,:] = mumean;
    xdu[breakss[4]+1:breakss[4]+nv,:] = [kron(ones(1,nlags),mumean) zeros(nv,nex)];
    breakss[5] = breakss[4]+nv;
    
    return(ydu,xdu)

end


"""
    drawPhiSigMinn(YY,hyp,p,T0,nex,n,nobs_)

Uses the Minnesota prior to draw from the Minnesota prior
"""
function drawPhiSigMinn(YY,hyp,p,T0,nex,n,nobs_)
# spec    = [p T0 nex n nobs_];


## Dummy Observations

# nlags   = spec[1];      # number of lags
# T0      = spec[2];      # size of pre-sample
# nex    = spec[3];      # number of exogenous vars; 1 means intercept only
# nv      = spec[4];      # number of variables
# nobs    = spec[5];      # number of observations


# Obtain mean and standard deviation from expandend pre-sample data

YY0     =   YY[1:min(T0+20,size(YY,1)-T0),:];
ybar    =   mean(YY0,dims=1)';
sbar    =   std(YY0,dims=1)';
premom  =   [ybar sbar];

(YYdum,XXdum) = test.varprior(n,p,nex,hyp,premom);


# Actual observations
(XXact,YYact) = test.mlag(YY,p;cnst=nex);

# draws from posterior distribution
Tdummy     = size(YYdum,1);
XXminn     = [XXact; XXdum];
YYminn     = [YYact; YYdum];
TMinn      = nobs_+Tdummy;             # Number of observations with dummies

FMinn     = svd(XXminn);
di        = 1.0./FMinn.S;
B         = FMinn.U'*YYminn;
xxi       = FMinn.V.*di';
inv_x     = xxi*xxi';
Phi_tilde = xxi*B;

# Draw Sigma
SigmaMinn     = (YYminn-XXminn*Phi_tilde)'*(YYminn-XXminn*Phi_tilde);
dist_sigma    = InverseWishart(TMinn-n*p-1,SigmaMinn);
sigma_draw    = rand(dist_sigma);

# Draw Phi
dist_phi      = MvNormal(vec(Phi_tilde),kron(sigma_draw,inv_x));
phi_vec_draw  = rand(dist_phi);

Phi_draw     = reshape(phi_vec_draw,n*p+1,n);

return(Phi_draw,sigma_draw,YYact,XXact)

end




"""
    def_Γ_Λ(Phi,sigma,nm,nq,Im,Iq,p)

Creates the Gamma and Lambda matrices
"""
function def_Γ_Λ(Phi,sigma,nm,nq,Im,Iq,p)
# Define phi(qm), phi(qq), phi(qc)
phi_qm = zeros(nm*p,nq);
for i=1:p
    phi_qm[nm*(i-1)+1:nm*i,:]=Phi[(i-1)*(nm+nq)+1:(i-1)*(nm+nq)+nm,nm+1:end];
end
phi_qq = zeros(nq*p,nq);
for i=1:p
    phi_qq[nq*(i-1)+1:nq*i,:]=Phi[(i-1)*(nm+nq)+nm+1:i*(nm+nq),nm+1:end];
end
phi_qc = Phi[end:end,nm+1:end];

# Define phi(mm), phi(mq), phi(mc)
phi_mm = zeros(nm*p,nm);
for i=1:p
    phi_mm[nm*(i-1)+1:nm*i,:]=Phi[(i-1)*(nm+nq)+1:(i-1)*(nm+nq)+nm,1:nm];
end
phi_mq = zeros(nq*p,nm);
for i=1:p
    phi_mq[nq*(i-1)+1:nq*i,:]=Phi[(i-1)*(nm+nq)+nm+1:i*(nm+nq),1:nm];
end
phi_mc = Phi[end:end,1:nm];

# Define Covariance Term sig_mm, sig_mq, sig_qm, sig_qq
sig_mm  = sigma[1:nm,1:nm];
sig_mq  = 0.5*(sigma[1:nm,nm+1:end]+transpose(sigma[nm+1:end,1:nm]));
sig_qm  = 0.5*(sigma[nm+1:end,1:nm]+transpose(sigma[1:nm,nm+1:end]));
sig_qq  = sigma[nm+1:end,nm+1:end];

# Define Transition Equation Matrices
GAMMAs  = [[transpose(phi_qq) zeros(nq,nq)];[Matrix(I,p*nq,p*nq) zeros(p*nq,nq)]];
GAMMAz  = [transpose(phi_qm); zeros(p*nq,p*nm)];
GAMMAc  = [transpose(phi_qc); zeros(p*nq,1)];
GAMMAu  = [Iq; zeros(p*nq,nq)];

 
#if growthRates == 1    
#    LAMBDAs = [[zeros(nm,nq) phi_mq'];...
#    [(1/3)*eye(nq) (2/3)*eye(nq) eye(nq) (2/3)*eye(nq) (1/3)*eye(nq)]];
#else
    LAMBDAs = [[zeros(nm,nq) phi_mq'];  (1/3)*[Iq Iq Iq zeros(nq,nq*(p-2))]];
#end

LAMBDAz = [transpose(phi_mm); zeros(nq,p*nm)];
LAMBDAc = [transpose(phi_mc); zeros(nq,1)];
LAMBDAu = [Im; zeros(nq,nm)];


Wmatrix   = [Im zeros(nm,nq)];
LAMBDAs_t = Wmatrix * LAMBDAs;
LAMBDAz_t = Wmatrix * LAMBDAz;
LAMBDAc_t = Wmatrix * LAMBDAc;
LAMBDAu_t = Wmatrix * LAMBDAu;
return(LAMBDAz,LAMBDAz_t,LAMBDAc,LAMBDAc_t,LAMBDAs,LAMBDAs_t,LAMBDAu,LAMBDAu_t,GAMMAs,GAMMAz,GAMMAc,GAMMAu,sig_qq,sig_mm,sig_mq,sig_qm)
end # end of def_Γ_Λ



"""
    assignfull_Pt(Pt_mat_end,kn,nq,nm)

Creates the full Pt matrix for the Kalman Filter/Smoother from the small KF matrix Pt_mat_end
"""
function assignfull_Pt(Pt_mat_end,kn,nq,nm,p)    
    BPt = zeros(kn,kn);
    for rr=1:p+1
        for vv=1:p+1
            BPt[rr*nm+(rr-1)*nq+1:rr*(nm+nq),vv*nm+(vv-1)*nq+1:vv*(nm+nq)]=
                Pt_mat_end[(rr-1)*nq+1:rr*nq,(vv-1)*nq+1:vv*nq];
        end
    end
    return(BPt)
end


"""
    mf_var()

TBW
"""
function mf_var(YDATA,nm,nq,nex,p,hyp,nsim,nburn)
    n       = nm + nq;
    kn      = n*(p+1);
    YM      = YDATA[:,1:1];
    YQ      = YDATA[:,2:2];
    Tstar   = size(YDATA,1);        # T star is the length of the M dataset from the beginning (like T but without accounting for the n_lags that have to be discarded)
    T0      = p;
    T       = Tstar-T0;             # This is T in the paper. The full dataset of monthly observations wihtout the lagged values
    Tb_full = size(YDATA,1);        # T is the lenght of 3*Q from the beginning (like T_b but without accounting for the n_lags that have to be discarded)
    Tb      = Tb_full-T0;           # dropping the lags for the pre-sample (=nlags)
    Tnow    = Tstar-Tb_full;        # The nowcasting number of observations
    Yq      = YQ[T0+1:Tb+T0,:];           
    Ym      = YM[T0+1:Tb+T0,:];


    
    # Saving matrices
    Atildemat = zeros(nsim-nburn,nq*(p+1));
    Ptildemat = zeros(nsim,nq*(p+1),nq*(p+1));
    YYsave    = zeros(T,n,nsim-nburn)

    # Matrices for collecting draws from Posterior Density

    Sigmap    = zeros(nsim-nburn,n,n);       # SIGMA in eq (1)
    Phip      = zeros(nsim-nburn,n*p+1,n);   # PHI in eq (2)
    Cons      = zeros(nsim-nburn,n);          # ???
    lstate    = zeros(nsim-nburn,nq,T);    # Tnobs = usable monthly observations
    YYactsim  = zeros(nsim-nburn,4,n);
    XXactsim  = zeros(nsim-nburn,4,n*p+1);

    ## Defining the phi matrices
    Im      = Matrix(I,nm,nm)*1.0;
    Iq      = Matrix(I,nq,nq)*1.0;
    Imq     = Matrix(I,nm+nq,nm+nq)*1.0;


    Phi    = [0.95*Imq;zeros((nm+nq)*(p-1)+1,(nm+nq))];
    sigma   = (1e-4).*Imq



    (LAMBDAz,LAMBDAz_t,LAMBDAc,LAMBDAc_t,LAMBDAs,LAMBDAs_t,LAMBDAu,LAMBDAu_t,
    GAMMAs,GAMMAz,GAMMAc,GAMMAu,sig_qq,sig_mm,sig_mq,sig_qm)    = test.def_Γ_Λ(Phi,sigma,nm,nq,Im,Iq,p)


        
    ## Initialization
    At   = zeros(nq*(p+1),1)                   # At will contain the latent states, s_t
    Pt   = zeros(nq*(p+1),nq*(p+1))

    for kk=1:5
        Pt = GAMMAs * Pt * transpose(GAMMAs) + GAMMAu * sig_qq * transpose(GAMMAu);
    end

    # Lagged Monthly Observations
    Zm   = zeros(Tb,nm*p)
    for i=1:p
        Zm[:,(i-1)*nm+1:i*nm] = YM[T0-(i-1):T0+Tb-i,:];
    end

     # Define Companion Form Matrix PHIF
     PHIF         = zeros(kn,kn);
     for i=1:p
         PHIF[i*n+1:(i+1)*n,(i-1)*n+1:i*n] = Imq;
     end

     SIGF = zeros(kn,kn)

     
    # Measurement Equation
    Z1         = zeros(nm,kn);
    Z1[:,1:nq] = Iq

    Z2         = zeros(nq,kn)
    for bb=1:nq
        # if growthRates
        #     for ll=1:5
        #         Z2(bb,ll*Nm+(ll-1)*Nq+bb)=weights(ll);
        #     end
        # else
            for ll=1:3
                Z2[bb,ll*nm+(ll-1)*nq+bb]=1/3;
            end
        # end
    end
    ZZ         = [Z1;Z2]

    ## -------------------------------------------------
    # beginning of the main for loop
    #    
    for j = 1:nsim

        (At_mat,Pt_mat_alt) = test.mf_kf(At,Pt,Tb,T0,Zm,Ym,Yq,GAMMAs,GAMMAz,GAMMAc,GAMMAu,sig_qq,sig_mm,sig_mq,sig_qm,LAMBDAz,LAMBDAs,LAMBDAc,LAMBDAu,LAMBDAc_t,LAMBDAs_t,LAMBDAu_t,LAMBDAz_t,p)

        Pt_mat_end = Pt_mat_alt[:,:,Tb]

        # unbalanced dataset
        PHIF[1:n,1:n*p] = transpose(Phi[1:end-1,:])

        # Define Constant Term CONF
        CONF = [transpose(Phi[end:end,:]);zeros(n*p,1)];      # ??? DO we really need this defined every time?

        # Define Covariance Term SIGF
        SIGF[1:n,1:n] = sigma

        # Switching to the full Kalman filter
        # zₜ = F₁ (Φ) z₍t-1₎ + Fc (Φ) + νₜ     


        BAt = vec(transpose([Ym[end:-1:end-p,:] transpose(At_mat[Tb:Tb,:])]))
        BPt = assignfull_Pt(Pt_mat_end,kn,nq,nm,p);  



        BAt_mat=zeros(Tnow+1,kn);

        BAt_mat[1,:] = BAt;
        BPt_mat_alt = zeros(kn,kn,Tnow+1);
        BPt_mat_alt[:,:,1] = BPt

        # Forward nowcasting filter
        # for kkk = 1:Tnow
        #     % Define New Data (ND) and New Z matrix (NZ)
        #     ND  = [delif(YDATA(Tb+T0+kkk,:)',index_NY(:,kkk))];       % deletes the missing obs, the NaN values
        #     NZ  = delif(ZZ,index_NY(:,kkk));
            
        #     BAt1 = BAt;
        #     BPt1 = BPt;
            
        #     % Forecasting
        #     Balphahat = PHIF * BAt1 + CONF;
        #     BPhat     = PHIF * BPt1 * PHIF' + SIGF;
        #     BPhat     = 0.5*(BPhat+BPhat');
            
        #     Byhat = NZ*Balphahat;
        #     Bnut  = ND - Byhat;
            
        #     BFt = NZ*BPhat*NZ';
        #     BFt = 0.5*(BFt+BFt');
            
        #     % Updating
        #     BAt = Balphahat + (BPhat*NZ')/BFt*Bnut;
        #     BPt = BPhat - (BPhat*NZ')/BFt*(BPhat*NZ')';
        #     BAt_mat(1+kkk,:)  = BAt';
        
        #     BPt_mat_alt(:,:,1+kkk)  = BPt;
            
        # end

        AT_draw = zeros(Tnow+1,kn);

        # singular value decomposition for the last observation
        F = svd(BPt_mat_alt[:,:,end])
        Pchol = F.U*Diagonal(sqrt.(F.S))
        AT_draw[end:end,:] = BAt_mat[end:end,:]+transpose(Pchol*randn(kn,1))

        # Kalman Smoother nowcasting
        if Tnow > 0
        # for i = 1:Tnow
        #     BAtt  = BAt_mat(end-i,:)';
        #     BPtt  = BPt_mat_alt(:,:,end-i);
            
        #     BPhat = PHIF * BPtt * PHIF' + SIGF;
        #     BPhat = 0.5*(BPhat+BPhat');
            
        #     [up, sp, vp] = svd(BPhat);
        #     inv_sp = zeros(size(sp,1),size(sp,1));
        #     inv_sp(sp>1e-12) = 1./sp(sp>1e-12);
            
        #     inv_BPhat = up*inv_sp*vp';
            
        #     Bnut  = AT_draw(end-i+1,:)'-PHIF*BAtt -CONF;
            
        #     Amean = BAtt + (BPtt*PHIF')*inv_BPhat*Bnut;
        #     Pmean = BPtt - (BPtt*PHIF')*inv_BPhat*(BPtt*PHIF')';
            
        #     % singular value decomposition
        #     [um, sm, ~] = svd(Pmean);
        #     Pmchol = um*sqrt(sm);
        #     AT_draw(end-i,:) = (Amean+Pmchol*randn(kn,1))';
            
        # end
        end

        ## Switching back to the small KF

        ## mf_ks
        (At_draw,Pmean) = test.mf_ks(Tb,At_mat,Pt_mat_alt,GAMMAs,GAMMAu,GAMMAz,GAMMAc,sig_qq,AT_draw,Zm,p,nq,nm);

        At   = copy(transpose(At_draw[1:1,:]));
        Pt   = Pmean;

        ## update dataset
        YY = [[Ym At_draw[:,1:nq]];  AT_draw[2:end,1:(nm+nq)]]

        nobs_   = size(YY,1)-T0;

        (Phi,sigma,YYact,XXact) = test.drawPhiSigMinn(YY,hyp,p,T0,nex,n,nobs_);

        if j>nburn
            Atildemat[j-nburn:j-nburn,:]    = At_mat[Tb:Tb,:]
            YYactsim[j-nburn:j-nburn,:,:]   = YYact[end-3:end,:];
            XXactsim[j-nburn:j-nburn,:,:]   = XXact[end-3:end,:];
            Sigmap[j-nburn:j-nburn,:,:]     = sigma;
            Phip[j-nburn:j-nburn,:,:]       = Phi;
            YYsave[:,:,j-nburn:j-nburn]     = YY;
        end

        (LAMBDAz,LAMBDAz_t,LAMBDAc,LAMBDAc_t,LAMBDAs,LAMBDAs_t,LAMBDAu,LAMBDAu_t,
        GAMMAs,GAMMAz,GAMMAc,GAMMAu,sig_qq,sig_mm,sig_mq,sig_qm)    = test.def_Γ_Λ(Phi,sigma,nm,nq,Im,Iq,p);


        end
   return YYsave
end  # end of function mf_var()



end # end of file