#include <math.h>
#include <stdlib.h>
#include <stddef.h>   /* ptrdiff_t */
#include <float.h>    /* DBL_MAX  */

void nearest(const double *x,
             int          *ind,      /* <- zero-based indices for Python     */
             const double *avect,
             int           tau,
             int           m,
             int           n)
{
    ptrdiff_t oi = 0;                           /* offset to column i */
    for (int i = 0; i < n; ++i, oi += m) {

        double bestdist = DBL_MAX;              /* current best distance  */
        int    closest  = -1;                   /* column index of best   */

        ptrdiff_t oj = 0;                       /* offset to column j */
        for (int j = 0; j < n; ++j, oj += m) {

            if (abs(i - j) <= tau)              /* Theiler window        */
                continue;

            double dist = 0.0;
            for (int k = 0; k < m; ++k) {       /* weighted squared diff */
                double diff = x[oi + k] - x[oj + k];
                dist += diff * diff * avect[k];
            }

            if (dist < bestdist) {              /* update running best   */
                bestdist = dist;
                closest  = j;
            }
        }
        ind[i] = closest;                       /* store zero-based idx  */
    }
}
