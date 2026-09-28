# Select existing scan points nearest the six representative plotting fields.
select_sample_fields(fields) = [fields[argmin(abs.(fields .- h))] for h in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)]

function write_correlations(group, result, times)
    RandomTFIM.time_scale(times, :imaginary)
    group["correlation_mean"] = result.C_mean
    group["correlation_sem"] = result.C_sem
    group["log_correlation_mean"] = result.logC_mean
    group["log_correlation_sem"] = result.logC_sem
    group["imaginary_time"] = times
    group["real_time"] = times
    # Both time domains provide real-valued sample means and SEM upstream.
    group["autocorrelation_mean"] = result.Ct_mean
    group["autocorrelation_sem"] = result.Ct_sem
    group["log_autocorrelation_mean"] = result.logCt_mean
    group["log_autocorrelation_sem"] = result.logCt_sem
    group["real_autocorrelation_mean"] = result.real_Ct_mean
    group["real_autocorrelation_sem"] = result.real_Ct_sem
    group["real_log_autocorrelation_mean"] = result.real_logCt_mean
    group["real_log_autocorrelation_sem"] = result.real_logCt_sem
end


function write_observables(group, result, times)
    for field in (:gap_mean, :gap_sem, :log_gap_mean, :log_gap_sem)
        group[string(field)] = getproperty(result, field)
    end
    write_correlations(group, result, times)
    if hasproperty(result, :gaps)
        group["gap_samples"] = result.gaps
        group["gap_resolved"] = result.resolved
        for field in (:sample_C, :sample_Ct, :sample_real_Ct)
            group[string(field)] = getproperty(result, field)
        end
    end
end
