#include "pch.h"
#include "mpi.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#define MAX_ITERATIONS 100

double Distance(double *X_Old, double *X_New, int N);
int main(int argc, char** argv) {
	MPI_Status status;
	int N, amountROwBloc, NoofRows, NoofCols;
	int size, rank;
	int irow, jrow, icol, index, Iteration, ClobalRowNo;
	double **Matrix_A, *Input_A, *Input_B, *ARecv, *BRecv, *Bloc_XX, *X_New, *X_Old, *Bloc_X;
	double Time_Spent;
	clock_t begin, finish;
	FILE *fp;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	N = 8;

	X_New = (double *)malloc(sizeof(double) * N);
	if (rank == 0) {
		if ((fp = fopen("./marix_A.txt", "r")) == NULL) {
			std::cout << "Can`t open input matrix_A";
			exit(-1);
		}
		fscanf(fp, "%d %d", &NoofRows, &NoofCols);
		N = NoofRows;
		amountROwBloc = N / size;
		Matrix_A = (double **)malloc(sizeof(double *) * N);
		for (irow = 0; irow < N; irow++) {
			Matrix_A[irow] = (double *)malloc(sizeof(double) * N);
			for (icol = 0; icol < N; icol++) {
				fscanf(fp, "%lf", &Matrix_A[irow][icol]);
			}
		}
		fclose(fp);
		if ((fp = fopen("./vector_B.txt", "r")) == NULL) {
			std::cout << "Can`t open input vector_B";
			exit(-1);
		}

		fscanf(fp, "%d", &NoofRows);
		N = NoofRows;
		Input_B = (double *)malloc(sizeof(double) * N);

		for (irow = 0; irow < N; irow++) {
			fscanf(fp, "%lf", &Input_B[irow]);
		}
		fclose(fp);

		if (NoofRows != NoofCols) {
			std::cout << "Input MAtrix Should Be Square Matrix...." << std::endl;
			exit(-1);
		}


		Input_A = (double *)malloc(sizeof(double) * N * N);
		index = 0;
		for (irow = 0; irow < N; irow++) {
			for (icol = 0; icol < N; icol++) {
				Input_A[index++] = Matrix_A[irow][icol];
			}
		}
	}

	begin = clock();
	MPI_Bcast(&NoofRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&NoofCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	amountROwBloc = N / size;
	ARecv = (double *)malloc(sizeof(double) * amountROwBloc * N);
	BRecv = (double *)malloc(sizeof(double) * amountROwBloc);

	MPI_Scatter(Input_A, amountROwBloc * N, MPI_DOUBLE, ARecv, amountROwBloc *N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(Input_B, amountROwBloc, MPI_DOUBLE, BRecv, amountROwBloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	Bloc_X = (double *)malloc(sizeof(double) * N);
	X_New = (double *)malloc(sizeof(double) * N);
	X_Old = (double *)malloc(sizeof(double) * N);

	for (irow = 0; irow < amountROwBloc; irow++) {
		Bloc_X[irow] = BRecv[irow];
	}

	MPI_Allgather(Bloc_X, amountROwBloc, MPI_DOUBLE, X_New, amountROwBloc, MPI_DOUBLE, MPI_COMM_WORLD);
	Iteration = 0;

	for (irow = amountROwBloc * size; irow < N; irow++) {
		MPI_Allgather(&Input_B[irow], 1, MPI_DOUBLE, &X_New[irow], 1, MPI_DOUBLE, MPI_COMM_WORLD);
	}

	Bloc_XX = (double *)malloc(sizeof(double) * N);
	do {
		for (irow = 0; irow < N; irow++) {
			X_Old[irow] = X_New[irow];
		}

		for (irow = 0; irow < amountROwBloc; irow++) {
			ClobalRowNo = (rank * amountROwBloc) + irow;
			Bloc_X[irow] = BRecv[irow];
			index = irow * N;

			for (icol = 0; icol < N; icol++) {
				if (icol != ClobalRowNo) {
					Bloc_X[irow] -= X_Old[icol] * ARecv[index + icol];
				}
			}
			Bloc_X[irow] = Bloc_X[irow] / ARecv[index + ClobalRowNo];
		}
		MPI_Allgather(Bloc_X, amountROwBloc, MPI_DOUBLE, X_New, amountROwBloc, MPI_DOUBLE, MPI_COMM_WORLD);

		if (rank == 0) {
			for (irow = amountROwBloc * size; irow < N; irow++) {
				ClobalRowNo = irow;
				Bloc_XX[irow] = Input_B[irow];
				index = irow * N;
				for (icol = 0; icol < N; icol++) {
					if (icol != ClobalRowNo) {
						Bloc_XX[irow] -= X_Old[icol] * Input_A[index + icol];
					}
				}
				Bloc_XX[irow] = Bloc_XX[irow] / Input_A[index + ClobalRowNo];
			}
		}

		for (irow = amountROwBloc * size; irow < N; irow++) {
			MPI_Allgather(&Bloc_XX[irow], 1, MPI_DOUBLE, &X_New[irow], 1, MPI_DOUBLE, MPI_COMM_WORLD);
		}

		Iteration++;
	} while ((Iteration < MAX_ITERATIONS) && (Distance(X_Old, X_New, N) >= 1.0E-24));
	finish = clock();

	if (rank == 0) {
		std::cout << "------------------------------------" << std::endl;
		std::cout << "Matrix A" << std::endl;
		for (irow = 0; irow < N; irow++) {
			for (icol = 0; icol < N; icol++) {
				std::cout << Matrix_A[irow][icol] << ".00  ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		std::cout << "Matrix B" << std::endl;
		for (irow = 0; irow < N; irow++) {
			std::cout << Input_B[irow] << ".0 ";
		}
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "Matrix Output" << std::endl;
		std::cout << "Number of iterations =" << Iteration << std::endl;
		std::cout << std::endl;
		for (irow = 0; irow < N; irow++) {
			std::cout << std::fixed << X_New[irow] << std::endl;
		}
		Time_Spent = (double)(finish - begin) / CLOCKS_PER_SEC;
		std::cout << "Time to use :" << std::fixed << Time_Spent << std::endl;
		std::cout << "-----------------------------------------" << std::endl;
	} 

	MPI_Finalize();
}

double Distance(double *X_Old, double *X_New, int N)
{
	int index;
	double Sum;
	Sum = 0.0;
	for (index = 0; index < N; index++) {
		Sum += (X_New[index] - X_Old[index]) * (X_New[index] - X_Old[index]);
	}

	return(Sum);
}
