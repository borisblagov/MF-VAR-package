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
j = 1

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
    BPt = test.assignfull_Pt(Pt_mat_end,kn,nq,nm,p);  



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
        
            Atildemat[j-nburn:j-j-nburn,:] = At_mat[Tb:Tb,:]
            YYactsim[j-nburn,:,:]   = YYact[end-3:end,:];
            XXactsim[j-nburn,:,:]   = XXact[end-3:end,:];
            Sigmap[j-nburn,:,:]     = sigma;
            Phip[j-nburn,:,:]       = Phi;
            YYsave[:,:,j-nburn]     = YY;
    end

    (LAMBDAz,LAMBDAz_t,LAMBDAc,LAMBDAc_t,LAMBDAs,LAMBDAs_t,LAMBDAu,LAMBDAu_t,
    GAMMAs,GAMMAz,GAMMAc,GAMMAu,sig_qq,sig_mm,sig_mq,sig_qm)    = test.def_Γ_Λ(Phi,sigma,nm,nq,Im,Iq,p);


    