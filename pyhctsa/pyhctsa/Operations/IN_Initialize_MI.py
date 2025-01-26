import jpype as jp
import os
import logging
import warnings

def IN_Initialize_MI(estMethod, extraParam=None, addNoise=False):
    """
    Parameters
    ----------
    estMethod : str
        The estimation method used to compute the mutual information. Options include:
            - 'gaussian'
            - 'kernel'
            - 'kraskov1'
            - 'kraskov2'
    extraParam : optional
        An extra parameter used in some estimation methods. For 'kraskov1' and 'kraskov2', this specifies
        the number of nearest neighbors (k). Default is None (sets k = 3).
    addNoise : bool, optional
        Whether to add noise to the signal. By default, noise is added. Set to False to make the computation
        deterministic for 'kraskov1' and 'kraskov2'. Default is False.

    Returns
    -------
    miCalc : object
        An initialized mutual information calculator object based on the specified estimation method.
    """

    # Check to see whether a jpype JVM has been started.
    if not jp.isJVMStarted():
        jarloc = (
            os.path.dirname(os.path.abspath(__file__)) + "/../Toolboxes/infodynamics-dist/infodynamics.jar"
        )
        # change to debug info
        logging.debug(f"Starting JVM with java class {jarloc}.")
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarloc)


    if estMethod == 'gaussian':
        implementingClass = 'infodynamics.measures.continuous.gaussian'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateGaussian()
    elif estMethod == 'kernel':
        implementingClass = 'infodynamics.measures.continuous.kernel'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKernel()
    elif estMethod == 'kraskov1':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov1()
    elif estMethod == 'kraskov2':
        implementingClass = 'infodynamics.measures.continuous.kraskov'
        miCalc = jp.JPackage(implementingClass).MutualInfoCalculatorMultiVariateKraskov2()
    else:
        raise ValueError(f"Unknown mutual information estimation method '{estMethod}'")

    # Add neighest neighbor option for KSG estimator
    if estMethod in ['kraskov1', 'kraskov2']:
        if extraParam != None:
            if isinstance(extraParam, int):
                warnings.warn("Number of nearest neighbors needs to be a string. Setting this for you...")
                extraParam = str(extraParam)
            miCalc.setProperty('k', extraParam) # 4th input specifies number of nearest neighbors for KSG estimator
        else:
            miCalc.setProperty('k', '3') # use 3 nearest neighbors for KSG estimator as default
        
    # Make deterministic if kraskov1 or 2 (which adds a small amount of noise to the signal by default)
    if (estMethod in ['kraskov1', 'kraskov2']) and (addNoise == False):
        miCalc.setProperty('NOISE_LEVEL_TO_ADD','0')
    
    # Specify a univariate calculation
    miCalc.initialise(1,1)

    return miCalc
