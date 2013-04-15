#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/resource.h>

double **allocmatrix(int nrow, int ncol);
void freematrix(double **m, int nrow);
void nerror(char *error_text);
double second(int nmode);
double rand_gen(double fmin, double fmax);
void SetSeed(int flag);

int main(int argc, char* argv[])
{
      int l,m,n,k;
      int i,j;
      double temp;
      double **A, **B, **C;
      double tstart, tend;

     /*  ****************************************************
     //  * The following allows matrix parameters to be     *
     //  * entered on the command line to take advantage    *
     //  * of dynamically allocated memory.  You may modify *
     //  * or remove it as you wish.                        *
     //  ****************************************************/

     if (argc != 4) {
       nerror("Usage:  <executable> <m-value> <n-value> <k-value>");
     }
      m = atoi((const char *)argv[1]);
      n = atoi((const char *)argv[2]);
      k = atoi((const char *)argv[3]);

      /*SetSeed(1); */

      /* *********************************************************
      // * Call the allocmatrix() subroutine to dynamically allocate *
      // * storage for the matrix sizes specified by m, n, and k *  
      // *********************************************************/

      A=(double **) allocmatrix(m,k);
      B=(double **) allocmatrix(k,n);
      C=(double **) allocmatrix(m,n);

      /* *********************************************************
      //  * Initialize matrix elements so compiler does not      *
      //  * optimize out                                         *
      // *********************************************************/

      for(i=0;i<m;i++) {
        for(j=0;j<k;j++) {
          A[i][j] = i * 3.2  + j * 2.21;
        }
      }

      for(i=0;i<k;i++) {
        for(j=0;j<n;j++) {
          B[i][j] = j * 1.3  + j * 3.1;
        }
      }

      for(i=0;i<m;i++) {
        for(j=0;j<n;j++) {
          C[i][j] = 0.0;
        }
      }
          
      /* **********************************
      // * Perform simple matrix multiply *
      // **********************************/
//      tstart = second(RUSAGE_SELF);
      for(j=0;j<n;j++) {
        for(l=0;l<k;l++) {
          for(i=0;i<m;i++) {
            C[i][j] = C[i][j] + B[l][j]*A[i][l];
          }
        }
      }
//      tend = second(RUSAGE_SELF);
      /* **************************************************
      // * Print out results for testing only    *
      // *                         *
      // **************************************************/
//#if 0
      fprintf(stdout, "Here is the matrix A:\n\n");
      for(i=0;i<m;i++) {
        for(j=0;j<k;j++) {
          fprintf(stdout, "%10.2f ",A[i][j]);
        }
        fprintf(stdout, "\n");
      }
      fprintf(stdout, "Here is the matrix B:\n\n");
      for(i=0;i<k;i++) {
        for(j=0;j<n;j++) {
          fprintf(stdout, "%10.2f",B[i][j]);
        }
        fprintf(stdout, "\n");
      }
      fprintf(stdout, "Here is the matrix C:\n\n");
      for(i=0;i<m;i++) {
        for(j=0;j<n;j++) {
          fprintf(stdout, "%10.2f",C[i][j]);
        }
        fprintf(stdout, "\n");
      }
//#endif
//      fprintf(stdout, "The total CPU time is: %f seconds \n\n", tend - tstart);

      /* **************************************************
      // * Free the memory allocated for the matrices  *
      // *                         *
      // **************************************************/

      freematrix(A, m);
      freematrix(B, n);
      freematrix(C, m);
}

double **allocmatrix(int nRow,int nCol)
{
 double **m;
  int i;

  m=(double **) malloc((unsigned)nRow*sizeof(double*));

  if (!m) nerror("allocation failure 1 in matrix()");

  for(i=0;i<nRow;i++) {
    m[i]=(double *) malloc((unsigned)nCol * sizeof(double));
    if (!m[i]) nerror("allocation failure 2 in matrix()");
  }
  return m;
}

void freematrix(double **m, int nRow)
{
    int i;
    int err = 0;

	for (i = 0; i < nRow; ++i)
		free ((double *)m[i]);

	free ((double **)m);
}

void nerror(char *error_text)
{
  fprintf(stderr, "Run-time error...\n");
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"Exiting...\n");
  exit(1);
}


double second(int nmode)
  /****************************************************************************
 *              Returns the total cpu time used in seconds.
 ****************************************************************************/
{
  struct rusage buf;
  double temp;
  
  getrusage( nmode, &buf );
  
  /* Get system time and user time in micro-seconds.*/
  temp = (double)buf.ru_utime.tv_sec*1.0e6 + (double)buf.ru_utime.tv_usec +
    (double)buf.ru_stime.tv_sec*1.0e6 + (double)buf.ru_stime.tv_usec;
  
  /* Return the sum of system and user time in SECONDS.*/
  return( temp*1.0e-6 );
}                              
#if 0
void SetSeed(int flag)
/* if flag == 1, set fixed random seed
// else 
//       set random seed based on current time */
{
  unsigned short seed[3];
  long ltime;
  
  if (flag)
  {
  seed[0] = (unsigned short) 0;
  seed[0] = (unsigned short) 1 ;
  seed[0] = (unsigned short) 2;	
  seed48(seed);

  return;
  }

  time(&ltime);
  seed[0] = (unsigned short)ltime;
  
  time(&ltime);
  seed[1] = (unsigned short)ltime;
  
  time(&ltime);
  seed[2] = (unsigned short)ltime;
  
  seed48(seed);            
}

double rand_gen(double fmin, double fmax)
{
  return fmin + (fmax - fmin) * drand48();
}
#endif

