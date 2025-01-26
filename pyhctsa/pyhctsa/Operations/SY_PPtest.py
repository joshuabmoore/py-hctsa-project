import numpy as np
from arch.unitroot import PhillipsPerron

def SY_PPtest(y, lags = range(0, 6)):
    """
    Gives differing outputs for p-value related features compared to MATLAB, 
    however, all of the stats and model selection related features align well.
    """
    if isinstance(lags, range):
        # evaluate test statistic at each lag
        models = []
        for l in lags:
            pp = PhillipsPerron(y, lags=l, trend='n')
            models.append(pp)
        # extract output summaries
        pVals = [models[i].pvalue for i in range(len(models))]
        stats = [models[i].stat for i in range(len(models))]
        lls = [models[i].regression.llf for i in range(len(models))]
        aics = [models[i].regression.aic for i in range(len(models))]
        bics = [models[i].regression.bic for i in range(len(models))]
        rmses = np.sqrt([models[i].regression.mse_resid for i in range(len(models))])

        out = {}
        # p-value related 
        out['maxpValue'] = np.max(pVals)
        out['minpValue'] = np.min(pVals)
        out['meanpValue'] = np.mean(pVals)
        out['stdpValue'] = np.std(pVals, ddof=1)
        out['lagmaxp'] = lags[np.argmax(pVals)]
        out['lagminp'] = lags[np.argmin(pVals)]
        
        # test statistic related
        out['meanstat'] = np.mean(stats)
        out['maxstat'] = np.max(stats)
        out['minstat'] = np.min(stats)

        out['meanloglikelihood'] = np.mean(lls)
        out['minAIC'] = np.min(aics)
        out['minBIC'] = np.min(bics)
        # No HQC available in arch
        
        out['minrmse'] = np.min(rmses)
        out['maxrmse'] = np.max(rmses)

    elif isinstance(lags, int):
        # evaluate test statistic at a single lag
        pp = PhillipsPerron(y, lags=lags, trend='n') # no trend components
        out = {}
        out['pvalue'] = pp.pvalue
        out['stat'] = pp.stat
        out['coeff1'] = pp.regression.params.values[0]
        out['loglikelihood'] = pp.regression.llf
        out['AIC'] = pp.regression.aic
        out['BIC'] = pp.regression.bic
        out['rmse'] = np.sqrt(pp.regression.mse_resid)
    else:
        raise TypeError(f"Invalid type: {lags}. Pass in either a range of lags, or a single lag as an int.")

    return out
