/* Close returns code by M. Little (c) 2006 */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define REAL double

/* Create embedded version of given sequence */
static void embedSeries(
    unsigned long embedDims,
    unsigned long embedDelay,
    unsigned long embedElements,
    const REAL *x,
    REAL *y
)
{
    unsigned int i, d, inputDelay;
    for (d = 0; d < embedDims; d++) {
        inputDelay = (embedDims - d - 1) * embedDelay;
        for (i = 0; i < embedElements; i++) {
            y[i * embedDims + d] = x[i + inputDelay];
        }
    }
}

/* Search for first close returns in the embedded sequence */
static void findCloseReturns(
    const REAL *x,
    REAL eta,
    unsigned long embedElements,
    unsigned long embedDims,
    unsigned long *closeRets
)
{
    REAL eta2 = eta * eta;
    REAL diff, dist2;
    unsigned long i, j, d, timeDiff, etaFlag;

    for (i = 0; i < embedElements; i++) {
        closeRets[i] = 0;
    }

    for (i = 0; i < embedElements; i++) {
        j = i + 1;
        etaFlag = 0;
        while ((j < embedElements) && !etaFlag) {
            dist2 = 0.0f;
            for (d = 0; d < embedDims; d++) {
                diff = x[i * embedDims + d] - x[j * embedDims + d];
                dist2 += diff * diff;
            }
            if (dist2 > eta2) {
                etaFlag = 1;
            }
            j++;
        }

        etaFlag = 0;
        while ((j < embedElements) && !etaFlag) {
            dist2 = 0.0f;
            for (d = 0; d < embedDims; d++) {
                diff = x[i * embedDims + d] - x[j * embedDims + d];
                dist2 += diff * diff;
            }
            if (dist2 <= eta2) {
                timeDiff = j - i;
                closeRets[timeDiff]++;
                etaFlag = 1;
            }
            j++;
        }
    }
}

/* Plain C entry point */
void close_ret(
    const double *x,           // Input sequence
    unsigned long vectorElements, // Length of input sequence
    unsigned long embedDims,   // Embedding dimension
    unsigned long embedDelay,  // Embedding delay
    double eta,                // Close return distance
    unsigned long *closeRets   // Output: close return time histogram (length = embedElements)
)
{
    unsigned long embedElements;
    REAL *embedSequence;

    if (embedDims < 1 || embedDelay < 1 || vectorElements < embedDims * embedDelay) {
        // Invalid parameters
        return;
    }

    embedElements = vectorElements - ((embedDims - 1) * embedDelay);
    embedSequence = (REAL *)calloc(embedElements * embedDims, sizeof(REAL));
    if (!embedSequence) return;

    embedSeries(embedDims, embedDelay, embedElements, x, embedSequence);
    findCloseReturns(embedSequence, eta, embedElements, embedDims, closeRets);

    free(embedSequence);
}
