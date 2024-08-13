#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
// #include <mpi.h> //For the Next part

extern void matToImage(char *filename, int *mat, int *dims);
extern void matToImageColor(char *filename, int *mat, int *dims);

int main(int argc, char **argv)
{

    int rank,numranks;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numranks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int rowStart, rowEnd;
    MPI_Status stat;
    int nextRowStart;
    int CHUNK_SIZE = 10;
    int done = 0;
    int doneRanks = 0;

    int numThreads = 12;
    omp_set_num_threads(numThreads);
    double progTimeStart = MPI_Wtime();

    int nx = 1200;
    int ny = 800;
    int *matrix = (int *)malloc(nx * ny * sizeof(int));

    for (int i = 0; i<(nx*ny); i++){
        matrix[i]=0;
    }

    int numOutside=0;
    
    int maxIter = 255;
    double xStart = -2;
    double xEnd = 1;
    double yStart = -1;
    double yEnd = 1;

    //Master Region
    if (rank==0){
        nextRowStart=0;

        int tot_numOutside=0;

        for(int i=1; i<numranks; i++){//Initial Sends
            rowStart=nextRowStart;
            rowEnd=rowStart+CHUNK_SIZE;
            nextRowStart+=CHUNK_SIZE;

            MPI_Send(&rowStart, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&rowEnd, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        while(done==0){
            MPI_Recv(&numOutside, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
            tot_numOutside+=numOutside;

            if(nextRowStart>ny){
                rowStart=-1;
                doneRanks++;
            }else{
                rowStart=nextRowStart;
                rowEnd=rowStart+CHUNK_SIZE;
                nextRowStart+=CHUNK_SIZE;
            }

            if(rowEnd>ny){
                rowEnd=ny;
            }

            //printf("Start: %d, End: %d\n", rowStart, rowEnd);

            MPI_Send(&rowStart, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
            MPI_Send(&rowEnd, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
            if(doneRanks==numranks-1){
                done=1;
            }
        }
        //printf("Total Outside: %d\n", tot_numOutside);
    }

    if (rank!=0){
        double workerStTime= MPI_Wtime();
        int keepWorking=1;
        while(keepWorking==1){
            MPI_Recv(&rowStart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
            MPI_Recv(&rowEnd, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
            if(rowStart==-1){
                keepWorking=0;
            }else{
                #pragma omp for schedule(dynamic) nowait
                for (int i = rowStart; i < rowEnd; i++)
                { // row
                    for (int j = 0; j < nx; j++)
                    { // col
                        int index = i * nx + j;
                        // C=x0+iy0
                        double x0 = xStart + (1.0 * j / nx) * (xEnd - xStart);
                        double y0 = yStart + (1.0 * i / ny) * (yEnd - yStart);
                        // Z0=0
                        double x = 0;
                        double y = 0;
                        double iter = 0;

                        while (iter < maxIter)
                        {
                            double temp = x * x - y * y + x0;
                            y = 2 * x * y + y0;
                            x = temp;
                            iter++;
                            if (x * x + y * y > 4)
                            {
                                numOutside=numOutside+1;
                                break;
                            }
                        } // end of while
                        //printf("Index: %d\n", index);
                        matrix[index] = iter;
                    }
                }

                MPI_Send(&numOutside, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            }
        }
        double workerEdTime=MPI_Wtime();
        double workerElapse = workerEdTime - workerStTime;
        printf("Worker #: %d, Time: %.5f\n", rank, workerElapse);
    }
    int *out_matrix = (int *)malloc(nx * ny * sizeof(int));

    for (int i = 0; i<(nx*ny); i++){
        out_matrix[i]=0;
    }
    MPI_Reduce(matrix, out_matrix, nx*ny, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    int dims[2];
    dims[0] = ny;
    dims[1] = nx;
    if(rank==0){
        double progTimeEnd = MPI_Wtime();
        double progTimeElapse = progTimeEnd - progTimeStart;

        printf("Total Time: %.10f\n", progTimeElapse);
        matToImage("mandelbrot.jpg", out_matrix, dims);
    }
    MPI_Finalize();
    return 0;
}
