# Shared unweighted fits; SEM is used by the plotting code only.
using Statistics

# The centered ordinary least-squares formula used by the original slope plot.
function fit_line(x, y)
    length(x) == length(y) >= 2 || error("Linear regression needs at least two points")
    dx = x .- mean(x)
    denominator = sum(abs2, dx)
    denominator > 0 || error("Linear regression needs distinct coordinates")
    slope = sum(dx .* (y .- mean(y))) / denominator
    (; intercept=mean(y)-slope*mean(x), slope)
end

function autocorrelation_fit_mask(t, y; tmin=0.0, tmax=Inf, floor=0.0)
    length(t) == length(y) || error("Mismatched data lengths")
    0 <= tmin < tmax || error("Require 0 <= tmin < tmax")
    isfinite(floor) && floor >= 0 || error("Require a nonnegative finite C floor")
    isfinite.(t) .& (t .> 0) .& (t .>= tmin) .& (t .<= tmax) .&
        isfinite.(y) .& (y .> 0) .& (y .>= floor)
end

function autocorrelation_log_slope(times, y)
    keep = autocorrelation_fit_mask(times, y)
    # Effective slopes are interpretable as 1/z only in a power-law window.
    abs(fit_line(log.(times[keep]), log.(y[keep])).slope)
end

function fit_decay(t, y, model)
    model in (:power, :exponential) || error("Unknown decay model: $model")
    # x=ln(tau): power uses x; exponential uses exp(x)=tau.
    predictor = model == :power ? log.(t) : Float64.(t)
    logC = log.(y)
    line = fit_line(predictor, logC)
    rate = max(0.0, -line.slope)
    logA = rate == 0 ? mean(logC) : line.intercept
    # Historical CSV names: these are log-space SSE and SSE/(n-2), with unit errors.
    chi_square = sum(abs2, logC .- (logA .- rate .* predictor))
    (; model, A=exp(logA), logA, rate, chi_square,
        reduced_chi_square=chi_square/(length(t)-2))
end

function fit_autocorrelation(t, y; tmin=0.0, tmax=Inf, floor=0.0)
    keep = autocorrelation_fit_mask(t, y; tmin, tmax, floor)
    count(keep) >= 3 || error("Fewer than three valid fit points")
    power = fit_decay(t[keep], y[keep], :power)
    exponential = fit_decay(t[keep], y[keep], :exponential)
    best = power.reduced_chi_square <= exponential.reduced_chi_square ? power : exponential
    (; power, exponential, best, keep, n=count(keep),
        tmin=minimum(t[keep]), tmax=maximum(t[keep]))
end

decay_prediction(fit, t) = exp.(fit.logA .- fit.rate .* (fit.model == :power ? log.(t) : t))

# Intersections of the plotted piecewise-linear curves in (log(h0), chi2_red).
function model_fit_crossings(fields, power, exponential)
    length(fields) == length(power) == length(exponential) || error("Mismatched scan lengths")
    all(h -> isfinite(h) && h > 0, fields) && all(diff(fields) .> 0) ||
        error("Crossings require strictly increasing positive fields")
    crossings = []
    delta = power .- exponential
    for i in eachindex(fields)
        isfinite(delta[i]) || continue
        if delta[i] == 0
            push!(crossings, (; h0=fields[i], reduced_chi_square=power[i], left=fields[i], right=fields[i]))
        elseif i < length(fields) && isfinite(delta[i+1]) && delta[i+1] != 0 &&
                signbit(delta[i]) != signbit(delta[i+1])
            fraction = abs(delta[i]) / (abs(delta[i]) + abs(delta[i+1]))
            h0 = exp((1-fraction)*log(fields[i]) + fraction*log(fields[i+1]))
            chi = (1-fraction)*power[i] + fraction*power[i+1]
            push!(crossings, (; h0, reduced_chi_square=chi, left=fields[i], right=fields[i+1]))
        end
    end
    crossings
end

# 用分位数匹配整条分布；对数差异避免所有曲线缩向零时产生虚假的重合。
function fit_time_collapse(times, samples)
    @assert length(times) == length(samples) >= 2
    @assert all(t -> isfinite(t) && t > 0, times)
    valid = [any(isfinite, y) for y in samples]
    count(valid) >= 2 || return (; mu=NaN, rms_before=NaN, rms_after=NaN)
    times, samples = times[valid], samples[valid]
    quantiles = hcat([quantile(filter(isfinite, y), 0.05:0.05:0.95) for y in samples]...)
    keep = vec(all(quantiles .> 0; dims=2))
    any(keep) || error("Time collapse needs positive quantiles")
    q = log.(quantiles[keep, :])
    q .-= mean(q; dims=2)
    dt = log.(times) .- mean(log.(times))
    denominator = size(q, 1) * sum(abs2, dt)
    denominator > 0 || error("Time collapse needs distinct times")
    mu = sum(q .* dt') / denominator
    (; mu, rms_before=sqrt(mean(abs2, q)), rms_after=sqrt(mean(abs2, q .- mu .* dt')))
end

