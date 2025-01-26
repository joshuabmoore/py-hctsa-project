from Operations.IN_Initialize_MI import IN_Initialize_MI
import jpype as jp

def IN_MutualInfo(y1, y2, estMethod = 'kernel', extraParam = None):
    """
    Compute the mutual information of two data vectors using the information dynamics toolkit implementation.

    Parameters:
    -----------
    y1 : array-like
        Input time series 1.
    y2 : array-like
        Input time series 2.
    estMethod : str
        The estimation method used to compute the mutual information. Options are:
        - 'gaussian'
        - 'kernel'
        - 'kraskov1'
        - 'kraskov2'

        Refer to:
        Kraskov, A., Stoegbauer, H., Grassberger, P. (2004). Estimating mutual information.
        Physical Review E, 69(6), 066138. DOI: http://dx.doi.org/10.1103/PhysRevE.69.066138

    Returns:
    --------
    float
        The estimated mutual information between the two input time series.
    """
    # Initialize miCalc object (don't add noise!):
    miCalc = IN_Initialize_MI(estMethod=estMethod, extraParam=extraParam, addNoise=False)
    # Set observations to two time series:
    y1_jp = jp.JArray(jp.JDouble)(y1) # convert observations to java double
    y2_jp = jp.JArray(jp.JDouble)(y2) # convert observations to java double
    miCalc.setObservations(y1_jp, y2_jp)

    # Compute mutual information
    out = miCalc.computeAverageLocalOfObservations()

    return out
