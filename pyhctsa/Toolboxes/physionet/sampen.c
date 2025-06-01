/* file: sampen.c	Doug Lake	2 August 2002
			Last revised:	1 November 2004 (by george@mit.edu) 1.2
-------------------------------------------------------------------------------
sampen: calculate Sample Entropy
Copyright (C) 2002-2004 Doug Lake

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place - Suite 330, Boston, MA 02111-1307, USA.  You may also view the agreement
at http://www.fsf.org/copyleft/gpl.html.

You may contact the author via electronic mail (dlake@virginia.edu).  For
updates to this software, please visit PhysioNet (http://www.physionet.org/).

_______________________________________________________________________________

Revision history:
  1.0 (2 August 2002, Doug Lake)	Original version
  1.1 (6 January 2004, George Moody)	Removed limits on input series length
  1.2 (1 November 2004, George Moody)	Merged bug fixes from DL (normalize
					by standard deviation, detect and
					avoid divide by zero); changed code to
					use double precision, to avoid loss of
					precision for small m and large N

Compile this program using any standard C compiler, linking with the standard C
math library.  For example, if your compiler is gcc, use:
    gcc -o sampen -O sampen.c -lm

For brief instructions, use the '-h' option:
    sampen -h

Additional information is available at:
    http://www.physionet.org/physiotools/sampen/.

*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Computes Sample Entropy for m = 0..M-1 and writes results to result[0..M-1]
// y: input array (length n)
// M: maximum embedding dimension (calculates for m=0..M-1)
// r: similarity threshold
// n: length of y
// result: output array of length M (must be allocated by caller)
void sampen(double *y, int M, double r, int n, double *result)
{
    double *p = NULL;
    long *run = NULL, *lastrun = NULL, N;
    double *A = NULL, *B = NULL;
    int M1, j, nj, jj, m;
    int i;
    double y1;

    M++; // Compute for m = 0..M
    run = (long *) calloc(n, sizeof(long));
    lastrun = (long *) calloc(n, sizeof(long));
    A = (double *) calloc(M, sizeof(double));
    B = (double *) calloc(M, sizeof(double));
    p = (double *) calloc(M, sizeof(double));

    // Main computation
    for (i = 0; i < n - 1; i++) {
        nj = n - i - 1;
        y1 = y[i];
        for (jj = 0; jj < nj; jj++) {
            j = jj + i + 1;
            if (((y[j] - y1) < r) && ((y1 - y[j]) < r)) {
                run[jj] = lastrun[jj] + 1;
                M1 = M < run[jj] ? M : run[jj];
                for (m = 0; m < M1; m++) {
                    A[m]++;
                    if (j < n - 1)
                        B[m]++;
                }
            }
            else
                run[jj] = 0;
        }
        for (j = 0; j < nj; j++)
            lastrun[j] = run[j];
    }

    N = (long) (n * (n - 1) / 2);
    p[0] = A[0] / N;
    result[0] = (p[0] > 0) ? -log(p[0]) : INFINITY;

    for (m = 1; m < M; m++) {
        p[m] = (B[m - 1] > 0) ? (A[m] / B[m - 1]) : 0.0;
        result[m] = (p[m] > 0) ? -log(p[m]) : INFINITY;
    }

    free(A);
    free(B);
    free(p);
    free(run);
    free(lastrun);
}
