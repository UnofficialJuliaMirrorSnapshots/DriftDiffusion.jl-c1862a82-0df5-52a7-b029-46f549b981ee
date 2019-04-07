module DriftDiffusion

using ForwardDiff

export adapt_clicks, compute_LL, VGHfun

function compute_LL(bup_times::Array{<:AbstractFloat},  bup_side::AbstractArray,
    stim_dur::Array{<:AbstractFloat}, poke_r::Array{Bool}, input_params::Any ;
    nan_times=Array{Bool}(0), use_param=fill(true,1,9), param_default=[0 1 1 0 1 0.1 0 0.01 1],
    use_prior=zeros(1,9), prior_mu=zeros(1,9), prior_var=zeros(1,9), window_dt=0.01,
    adaptation_scales_perclick="var", nt=[])

    # validate inputs
    @assert size(bup_times,1)==size(bup_side,1) "bup_times and bup_side must have same 1st dimension length (i.e. number of maxbups)"
    @assert size(bup_side,2)==size(stim_dur,2) && size(bup_side,2)==size(poke_r,2) "bup_side, stim_dur, and poke_r must have the same 2nd dimension length (i.e. number of trials)"
    @assert length(use_param)==9 &&  length(use_param)==length(param_default) &&
        length(use_param)==length(use_prior) && length(use_param)==length(prior_mu) &&
        length(use_param)==length(prior_var) "use_param, param_default, use_prior, prior_mu, and prior_var must all have length 9 (because this is currently a 9-parameter model) ";
    if ~all(prior_var[use_prior.==0] .== 0)
        warn("some unused priors have nonzero variance")
    end
    if ~all(use_prior[prior_var.==0] .== 0)
        error("cannot use a prior with 0 variance")
    end

    # initialize variables
    # (Adrian: initializing some variables to be the same type as input_params is crucial, because
    # when you are using compute_LL to compute the Hessian, these must all be duals.)
    params_full              = zeros(eltype(input_params), 1,9)
    params_full[.!use_param] = param_default[.!use_param]
    params_full[use_param]   = input_params
    if params_full[end]<1 & isempty(nt)
        sz=size(bup_times);
        bup_times_normalized = broadcast(/,bup_times,stim_dur);
        nt = Int(round((1-params_full[end])/window_dt))+1;
        bup_times = repmat(bup_times,1,nt);
        in_window = Array{Bool}(size(bup_times));
        for t=1:nt
            inds=(1:sz[2])+sz[2]*(t-1);
            time_window = [0 params_full[end]] + (t-1)*window_dt ;
            in_window[:,inds] = (bup_times_normalized.>time_window[1]) .& (bup_times_normalized.<=time_window[2]);
        end
        bup_times[.~in_window] = NaN;
        nan_times = isnan.(bup_times);
        stim_dur  = stim_dur.*params_full[end];
    elseif isempty(nt)
        nt=1;
    end
    nansum(x::Array{Float64}) = sum(x.*.!isnan.(x),1);
    nTrials = length(poke_r);
    prob_poked_r = zeros(eltype(input_params),nt,nTrials);
    NLL = zeros(eltype(input_params),1,nTrials);
    adapted = ones(eltype(bup_side),size(bup_side))
    init_var = input_params[4];
    a_var = input_params[2];
    c_var = input_params[3];
    lambda = input_params[1];
    bias    = params_full[7]
    lapse   = params_full[8]
    bups_lambda_applied=1;
    if lambda ==0
        s2 = init_var + a_var*stim_dur
    else
        s2 = init_var*exp.(2*lambda*stim_dur) + (a_var./(2*lambda))*(exp.(2*lambda*stim_dur)-1)
    end
    # calculate LL simultaneously for all trials, looping over time points
    for t=1:nt
        inds=(1:nTrials)+nTrials*(t-1);
        curr_buptimes = bup_times[:,inds];
       if isempty(nan_times)
           curr_nantimes = isnan.(curr_buptimes);
       else
           curr_nantimes = nan_times[:,inds];
       end
        # adapt those clicks
        adapted = adapt_clicks(curr_buptimes, curr_nantimes, params_full[5],params_full[6])
        # apply integration timescale
        if lambda!=0
            bups_lambda_applied = exp(lambda * time_from_end[:,inds])
        end
        # compute mean of distribution
        mean_a = nansum(adapted.*bups_lambda_applied.*bup_side,1);
        # add per click variance
        if adaptation_scales_perclick=="std"
            var_a      = s2 + nansum(c_var .* adapted.^2 .* bups_lambda_applied .^ 2,1);
        elseif adaptation_scales_perclick=="var"
            var_a      = s2 + nansum(c_var .* adapted .* bups_lambda_applied .^ 2 ,1);
        elseif adaptation_scales_perclick=="none"
            var_a      = s2 + nansum(c_var .* bups_lambda_applied .^ 2,1);
        end
        no_noise_trials = (var_a .==0);
        if any(no_noise_trials)
            warn( @sprintf("Total noise variance is zero for %g trials. ",sum(no_noise_trials)),
                "Could be due to a model with no noise terms or only stimulus noise but no bups on a trial. ",
                "Adding eps to avoid NaNs but you may want to do some sanity checks.");
            var_a[no_noise_trials]=eps;
        end
        erfTerm[:,inds] =  (mean_a-bias)./sqrt.(2*var_a);
    end
    erfTerm = erf(erfTerm);
    if any(abs(erfTerm).==1)
        erfTerm[erfTerm.==1]=1-eps();
        erfTerm[erfTerm.==-1]=eps()-1;
    end
    prob_poked_r=((1-lapse).*(1+erfTerm)+lapse)/2 ;
    prob_poked_r = mean(prob_poked_r,1);
    NLL[poke_r] = - ( log.( prob_poked_r[poke_r] ) );
    NLL[.!poke_r] = - ( log.(1- prob_poked_r[.!poke_r] ) );
    NLL_total = sum(NLL)
    # increment likelihood for priors
    for pp = find(use_prior)
        NLL_total += -(params_full[pp]-prior_mu[pp])^2/(2*prior_var[pp])
    end
    return NLL_total
end


function adapt_clicks(bup_times, nan_times, phi,tau_phi) #phi, tau_phi)
    adapt   = zeros(eltype(phi),size(bup_times)); # this must be the same type as the params because these could be duals
    if phi<0 | tau_phi<0
        error("adaptation parameters must be non-negative")
    end
    adapt[.!nan_times] = 1
    ici     = diff(bup_times)
    for i = 2:size(bup_times,1)
        adapt[i, :] = 1 + exp.(-ici[i-1,:]./tau_phi).*(adapt[i-1,:]*phi-1)
        adapt[i, ici[i-1, :] .<= 0] = 0
        adapt[i-1, ici[i-1,:] .<= 0] = 0
    end
    return adapt
end

function VGHfun(bup_times,  bup_side, stim_dur, poke_r, input_params; nan_times=fill(Bool,0), use_param=fill(true,1,9),
    param_default=[0 1 1 0 1 0.1 0 0.01 1],use_prior=zeros(1,9), prior_mu=zeros(1,9), prior_var=zeros(1,9),
    window_dt=0.01, adaptation_scales_perclick="var", nt=[])
    # compute hessian using autodiff
    # (Adrian: For reasons I don't understand, you need to call compute_LL with explicit keyword declaration for ForwardDiff to run
    optimFun(x) = DriftDiffusion.compute_LL(bup_times, bup_side, stim_dur, poke_r, x;nan_times=nan_times, use_param=use_param,
        param_default=param_default,use_prior=use_prior, prior_mu=prior_mu, prior_var=prior_var,
        window_dt=window_dt, adaptation_scales_perclick=adaptation_scales_perclick, nt=nt);
    out = DiffResults.HessianResult(zeros(size(input_params)));
    out = ForwardDiff.hessian!(out,optimFun,input_params);
    return DiffResults.value(out),DiffResults.gradient(out),DiffResults.hessian(out)
end

end # module
