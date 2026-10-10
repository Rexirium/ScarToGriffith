using LinearAlgebra

# Log-sum-exp keeps tiny positive tails usable without clipping C.
function offset_log_prediction(p, x)
    u = p[1] .- p[2] .* x
    m = max.(u, p[3])
    m .+ log.(exp.(u .- m) .+ exp.(p[3] .- m))
end

function fit_offset_decay(t, y, model)
    base = fit_decay(t, y, model)
    predictor = model == :power ? log.(t) : Float64.(t)
    origin, scale = minimum(predictor), maximum(predictor)-minimum(predictor)
    scale > 0 || error("Offset fits require distinct times")
    x, target = (predictor .- origin) ./ scale, log.(y)
    # B=0 is an explicit boundary candidate, not an arbitrary positive floor.
    best = (; p=[base.logA-base.rate*origin, base.rate*scale, -Inf],
        sse=base.chi_square, converged=true, iterations=0)
    starts = unique([base.rate*scale, 1.0, scale])
    for fraction in (0.01, 0.5, 1.0), initial_rate in starts
        initial_A = max(y[argmin(predictor)]-fraction*minimum(y), 0.01*maximum(y))
        p = [log(initial_A), initial_rate, log(minimum(y))+log(fraction)]
        damping, converged, iterations = 1e-3, false, 0
        for iteration in 1:400
            iterations = iteration
            prediction = offset_log_prediction(p, x)
            residual = prediction .- target
            sse = sum(abs2, residual)
            w = exp.(p[1] .- p[2].*x .- prediction)
            jacobian = hcat(w, -x.*w, 1 .- w)
            gradient = jacobian' * residual
            projected = copy(gradient)
            p[2] == 0 && (projected[2] = min(0.0, projected[2]))
            if norm(projected, Inf) <= 1e-8 * max(1.0, sqrt(sse))
                converged = true
                break
            end
            normal = jacobian' * jacobian
            step = -(normal + damping*Diagonal(max.(diag(normal), 1e-12))) \ gradient
            trial = p + step
            trial[2] = max(0.0, trial[2])
            score = sum(abs2, offset_log_prediction(trial, x) .- target)
            if isfinite(score) && score < sse
                stable = sse-score <= 1e-12*max(sse, 1e-12) &&
                    norm(trial-p) <= 1e-6*(1+norm(p))
                p = trial
                damping = max(damping/3, 1e-12)
                if stable
                    converged = true
                    break
                end
            else
                damping *= 10
                damping > 1e16 && break
            end
        end
        score = sum(abs2, offset_log_prediction(p, x) .- target)
        # Prefer a converged restart when objectives agree to numerical precision.
        tolerance = 1e-12*max(score, best.sse, 1e-20)
        if score < best.sse-tolerance ||
                (abs(score-best.sse) <= tolerance && converged && !best.converged)
            best = (; p, sse=score, converged, iterations)
        end
    end
    rate = best.p[2]/scale
    logA = best.p[1]+rate*origin
    (; model, A=exp(logA), logA, B=exp(best.p[3]), logB=best.p[3], rate,
        alpha=model == :power ? -rate : NaN, chi_square=best.sse,
        reduced_chi_square=best.sse/(length(t)-3), best.converged, best.iterations)
end

function fit_autocorrelation_offset(t, y; tmin=0.0, tmax=Inf)
    keep = autocorrelation_fit_mask(t, y; tmin, tmax)
    count(keep) >= 4 || error("Fewer than four valid offset-fit points")
    power = fit_offset_decay(t[keep], y[keep], :power)
    exponential = fit_offset_decay(t[keep], y[keep], :exponential)
    best = power.chi_square <= exponential.chi_square ? power : exponential
    (; power, exponential, best, keep, n=count(keep),
        tmin=minimum(t[keep]), tmax=maximum(t[keep]))
end

offset_prediction(fit, t) = exp.(offset_log_prediction(
    [fit.logA, fit.rate, fit.logB], fit.model == :power ? log.(t) : t))
