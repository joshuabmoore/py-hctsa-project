from Operations.EN_SampEn import EN_SampEn
from PeripheryFunctions.BF_PreProcess import BF_PreProcess
from PeripheryFunctions.BF_zscore import BF_zscore
from PeripheryFunctions.BF_MakeBuffer import BF_MakeBuffer
import numpy as np 

def EN_mse(y, scaleRange = None, m = 2, r = 0.15, preProcessHow = None):
    """
    """
    if scaleRange is None:
        scaleRange = range(1, 11)
    minTsLength = 20
    numScales = len(scaleRange)

    if preProcessHow is not None:
        y = BF_zscore(BF_PreProcess(y, preProcessHow))
    
    # Coarse-graining across scales
    y_cg = []
    for i in range(numScales):
        buffer_size = scaleRange[i]
        y_buffer = BF_MakeBuffer(y, buffer_size)
        y_cg.append(np.mean(y_buffer, axis=1))
    
    # Run sample entropy for each m and r value at each scale
    samp_ens = np.zeros(numScales)
    for si in range(numScales):
        if len(y_cg[si]) >= minTsLength:
            samp_en_struct = EN_SampEn(y_cg[si], m, r)
            samp_ens[si] = samp_en_struct[f'sampen{m}']
        else:
            samp_ens[si] = np.nan

    # Outputs: multiscale entropy
    if np.all(np.isnan(samp_ens)):
        if preProcessHow:
            pp_text = f"after {preProcessHow} pre-processing"
        else:
            pp_text = ""
        print(f"Warning: Not enough samples ({len(y)} {pp_text}) to compute SampEn at multiple scales")
        return {'out': np.nan}

    # Output raw values
    out = {f'sampen_s{scaleRange[i]}': samp_ens[i] for i in range(numScales)}

     # Summary statistics of the variation
    max_samp_en = np.nanmax(samp_ens)
    max_ind = np.nanargmax(samp_ens)
    min_samp_en = np.nanmin(samp_ens)
    min_ind = np.nanargmin(samp_ens)

    out.update({
        'maxSampEn': max_samp_en,
        'maxScale': scaleRange[max_ind],
        'minSampEn': min_samp_en,
        'minScale': scaleRange[min_ind],
        'meanSampEn': np.nanmean(samp_ens),
        'stdSampEn': np.nanstd(samp_ens, ddof=1),
        'cvSampEn': np.nanstd(samp_ens, ddof=1) / np.nanmean(samp_ens),
        'meanch': np.nanmean(np.diff(samp_ens))
    })

    return out
