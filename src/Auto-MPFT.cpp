/*****************************************************************************
*    This is an implementation of Fast Multidimensional Partial              *
*    Fourier Transform with Automatic Hyperparameter Selection               *
*    (submitted to KDD 2024).                                                *
*                                                                            *
*    Note that this code example is specifically designed for                *
*    2d real-valued input with a target range centered at zero               *
*    for the best performance. A slight modification removes the             *
*    constraint; please refer to the original paper for more details.        *
*                                                                            *
*    This code also contains the implementation of MKL DFTI.                 *
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "ipps.h"
#include "ipp.h"
#include "mkl_dfti.h"
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

int main()
{
	const int N1 = 8192;
	const int N2 = 8192;
	const int M = 128;
	const int p1 = 128;
	const int p2 = 128;
	const MKL_LONG shape_F[2] = { N1, N2 };
	const MKL_LONG shape_P[2] = { p1, p2 };

	const string error = "e-7";
	// One of e-1, e-2, e-3, e-4, e-5, e-6, e-7 is allowed

	const int num_D1 = ceil(M * (0.5 / p1));
	const int num_D2 = ceil(M * (1.0 / p2));
	const int q1 = N1 / p1;
	const int q2 = N2 / p2;
	
	/* Pointers */
	double* W1, * W2, * XI;
	W1 = (double*)mkl_malloc(27 * sizeof(double), 64);
	W2 = (double*)mkl_malloc(27 * sizeof(double), 64);
	XI = (double*)mkl_malloc(26 * sizeof(double), 64);

	/* Load precomputed xi */
	string file_name = "precomputed/" + error + ".csv";
	ifstream ip(file_name);
	string sd;
	int count = 0;

	while (count < 26)
	{
		getline(ip, sd, ',');
		XI[count] = stod(sd);
		count += 1;
	}

	/* Find r */
	int temp_r1 = 0;
	while (XI[temp_r1] < ((1.0 * M) / p1))
	{
		temp_r1 += 1;
		if (temp_r1 == 25)
			break;
	}
	const int r1 = temp_r1 + 2;

	int temp_r2 = 0;
	while (XI[temp_r2] < ((1.0 * M) / p2))
	{
		temp_r2 += 1;
		if (temp_r2 == 25)
			break;
	}
	const int r2 = temp_r2 + 2;

	/* Load precomputed w */
	for (int op = 0; op < r1 - 2; op++)
	{
		getline(ip, sd);
	}
	for (int op = 0; op < r1; op++)
	{
		getline(ip, sd, ',');
		W1[op] = stod(sd);
	}

	ifstream ip2(file_name);
	for (int op = 0; op < r2 - 1; op++)
	{
		getline(ip2, sd);
	}
	for (int op = 0; op < r2; op++)
	{
		getline(ip2, sd, ',');
		W2[op] = stod(sd);
	}
	
	const int p1ho = p1 / 2 + 1;
	const int p1hod = p1ho * 2;
	const int r1h = r1 / 2;
	const int p2ho = p2 / 2 + 1;
	const int p2hod = p2ho * 2;
	const int r2h = r2 / 2;

	/* Initialize pointers */
	const Ipp32fc complex_i = { 0.0, 1.0 };
	Ipp32fc* FOUT = (Ipp32fc*)mkl_malloc(N1 * (N2 / 2 + 1) * sizeof(Ipp32fc), 32);
	Ipp32fc* COUT = (Ipp32fc*)mkl_malloc(p1 * p2ho * r1 * r2 * sizeof(Ipp32fc), 32);
	Ipp32fc* TEMP = (Ipp32fc*)mkl_malloc(r2 * p1 * p2hod * sizeof(Ipp32fc), 32);
	Ipp32fc* TWDL = (Ipp32fc*)mkl_malloc(p1 * p2ho * num_D1 * num_D2 * sizeof(Ipp32fc), 32);
	Ipp32fc* DS = (Ipp32fc*)mkl_malloc(r1 * r2 * p1 * p2ho * num_D1 * num_D2 * sizeof(Ipp32fc), 32);

	float* A = (float*)mkl_malloc(N1 * N2 * sizeof(float), 32);
	float* AA = (float*)mkl_malloc(p1 * p2 * q1 * q2 * sizeof(float), 32);
	float* BT = (float*)mkl_malloc(r1 * q1 * sizeof(float), 32);
	float* B2 = (float*)mkl_malloc(r2 * q2 * sizeof(float), 32);
	float* B = (float*)mkl_malloc(q2 * r2 * sizeof(float), 32);
	float* C1 = (float*)mkl_malloc(p1 * p2 * r1 * q2 * sizeof(float), 32);
	float* C2 = (float*)mkl_malloc(p1 * p2 * r1 * r2 * sizeof(float), 32);
	float* C2T = (float*)mkl_malloc(r1 * r2 * p1 * p2 * sizeof(float), 32);

	int n, l, i, j;

	/* Generate A */
	std::srand(std::time(0));
	for (n = 0; n != N1 * N2; n++)
	{
		A[n] = (float)(((float)std::rand()) / RAND_MAX); // random 0 ~ 1
	}

	/* Rearrange A */
	for (i = 0; i != p1 * p2; i++)
		for (j = 0; j != q1 * q2; j++)
		{
			AA[q1 * q2 * i + j] = A[p2 * q2 * (q1 * (i / p2) + j / q2) + q2 * (i % p2) + j % q2];
		}

	/* Precompute B^T and B */
	for (j = 0; j != r1; j++)
		for (l = 0; l != q1; l++)
		{
			BT[q1 * j + l] = (float)(W1[j] * pow(-2.0 * (l - (q1 / 2.0)) * (1.0 / q1), j));
		}

	for (j = 0; j != r2; j++)
		for (l = 0; l != q2; l++)
		{
			B2[q2 * j + l] = (float)(W2[j] * pow(-2.0 * (l - (q2 / 2.0)) * (1.0 / q2), j));
		}
	mkl_somatcopy('R', 'T', r2, q2, 1.0, B2, q2, B, r2);

	for (int s1 = 0; s1 != num_D1; s1++)
		for (int s2 = 0; s2 != num_D2; s2++)
			for (int j1 = 0; j1 != r1; j1++)
				for (int j2 = 0; j2 != r2; j2++)
					for (int k1 = 0; k1 != p1; k1++)
						for (int k2 = 0; k2 != p2ho; k2++)
						{
							if (((j1 % 4 < 2) and (j2 % 4 < 2)) or ((j1 % 4 > 1) and (j2 % 4 > 1)))
							{
								if (s2 % 2 == 0)
									DS[num_D2 * r1 * r2 * p1 * p2ho * s1 + r1 * r2 * p1 * p2ho * s2 + r2 * p1 * p2ho * j1 + p1 * p2ho * j2 + p2ho * k1 + k2]
									= { (float)(pow(((s1 * p1 + k1) * 1.0) * (1.0 / p1), j1) * pow(((s2 * p2 + k2) * 1.0) * (1.0 / p2), j2)), 0.0};
								else
									DS[num_D2 * r1 * r2 * p1 * p2ho * s1 + r1 * r2 * p1 * p2ho * s2 + r2 * p1 * p2ho * j1 + p1 * p2ho * j2 + p2ho * k1 + p2ho - 1 - k2]
									= { (float)(pow(((s1 * p1 + k1) * 1.0) * (1.0 / p1), j1) * pow(((s2 * p2 + k2) * 1.0) * (1.0 / p2), j2)), 0.0};
							}
							else
							{
								if (s2 % 2 == 0)
									DS[num_D2 * r1 * r2 * p1 * p2ho * s1 + r1 * r2 * p1 * p2ho * s2 + r2 * p1 * p2ho * j1 + p1 * p2ho * j2 + p2ho * k1 + k2]
									= { (float)(-pow(((s1 * p1 + k1) * 1.0) * (1.0 / p1), j1) * pow(((s2 * p2 + k2) * 1.0) * (1.0 / p2), j2)), 0.0};
								else
									DS[num_D2 * r1 * r2 * p1 * p2ho * s1 + r1 * r2 * p1 * p2ho * s2 + r2 * p1 * p2ho * j1 + p1 * p2ho * j2 + p2ho * k1 + p2ho - 1 - k2]
									= { (float)(-pow(((s1 * p1 + k1) * 1.0) * (1.0 / p1), j1) * pow(((s2 * p2 + k2) * 1.0) * (1.0 / p2), j2)), 0.0};
							}
						}

	/* Precompute Twiddle factors */
	for (int s1 = 0; s1 != num_D1; s1++)
		for (int s2 = 0; s2 != num_D2; s2++)
			for (int k1 = 0; k1 != p1; k1++)
				for (int k2 = 0; k2 != p2ho; k2++)
				{
					float theta = (p1 * s1 + k1) * M_PI * (1.0 / p1) + (p2 * s2 + k2) * M_PI * (1.0 / p2);
					TWDL[num_D2 * p1 * p2ho * s1 + p1 * p2ho * s2 + p2ho * k1 + k2] = { cos(theta), -sin(theta) };
				}

	DFTI_DESCRIPTOR_HANDLE hand;

	/* 2d FFT */
	const MKL_LONG strides_out_F[3] = { 0, N2 / 2 + 1, 1 };
	DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 2, shape_F);
	DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	DftiSetValue(hand,DFTI_OUTPUT_STRIDES, strides_out_F);
	DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	DftiCommitDescriptor(hand);
	DftiComputeForward(hand, A, FOUT);

	/* 2d Auto-MPFT */
	const MKL_LONG strides_out_P[3] = { 0, p2ho, 1 };
	DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 2, shape_P);
	DftiSetValue(hand, DFTI_NUMBER_OF_TRANSFORMS, r1 * r2);
	DftiSetValue(hand, DFTI_INPUT_DISTANCE, p1 * p2);
	DftiSetValue(hand, DFTI_OUTPUT_DISTANCE, p1 * p2ho);
	DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	DftiSetValue(hand, DFTI_OUTPUT_STRIDES, strides_out_P);
	DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	DftiCommitDescriptor(hand);

	/* Matrix Multiplication */
	CBLAS_TRANSPOSE transA = CblasNoTrans;
	CBLAS_TRANSPOSE transB = CblasNoTrans;
	const float alpha = 1.0;
	const float beta = 0.0;
	const long long batch = p1 * p2;
	const long long group_q1 = q1;
	const long long group_q2 = q2;
	const long long group_r1 = r1;
	const long long group_r2 = r2;

	const float* AA_array[batch];
	const float* B_array[batch];
	const float* BT_array[batch];
	float* C1_array[batch];
	float* C2_array[batch];
	const float* CC_array[batch];

	for (i = 0; i < batch; ++i) {
		AA_array[i] = AA + i * q1 * q2;
		B_array[i] = B;
		BT_array[i] = BT;
		C1_array[i] = C1 + i * r1 * q2;
		C2_array[i] = C2 + i * r1 * r2;
		CC_array[i] = C1 + i * r1 * q2; 
	}

	cblas_sgemm_batch(
		CblasRowMajor, &transA, &transB,
		&group_r1, &group_q2, &group_q1,
		&alpha,
		BT_array, &group_q1,
		AA_array, &group_q2,
		&beta,
		C1_array, &group_q2,
		1, &batch
	);
	cblas_sgemm_batch(
		CblasRowMajor, &transA, &transB,
		&group_r1, &group_r2, &group_q2,
		&alpha,
		CC_array, &group_q2,
		B_array, &group_r2,
		&beta,
		C2_array, &group_r2,
		1, &batch
	);

	/* Transpose */
	mkl_somatcopy('R', 'T', batch, r1 * r2, 1.0, C2, r1 * r2, C2T, batch);

	/* Batch 2d FFTs */
	DftiComputeForward(hand, C2T, COUT);

	/* Post-processing */
	for (int s1 = 0; s1 != num_D1; s1++)
		for (int s2 = 0; s2 != num_D2; s2++)
		{
			ippsMul_32fc(DS + r1 * r2 * p1 * p2ho * (num_D2 * s1 + s2), COUT, TEMP, r2 * p1 * p2hod); 
			for (int i = 1; i != (r1 / 2); ++i) {
				ippsAddProduct_32fc(DS + r2 * p1 * p2hod * i, COUT + r2 * p1 * p2hod * i, TEMP, r2 * p1 * p2hod);
			}
			if (r1 % 2 == 1)
				ippsAddProduct_32fc(DS + r2 * p1 * p2hod * r1h, COUT + r2 * p1 * p2hod * r1h, TEMP, r2 * p1 * p2ho);
			ippsMulC_32fc_I(complex_i, TEMP + r2 * p1 * p2ho, r2 * p1 * p2ho);
			ippsAdd_32fc_I(TEMP + r2 * p1 * p2ho, TEMP, r2 * p1 * p2ho);

			for (int i = 1; i != (r2 / 2); ++i) {
				ippsAdd_32fc_I(TEMP + p1 * p2hod * i, TEMP, p1 * p2hod);
			}
			if (r2 % 2 == 1)
				ippsAdd_32fc_I(TEMP + p1 * p2hod * r2h, TEMP, p1 * p2ho);
			ippsMulC_32fc_I(complex_i, TEMP + p1 * p2ho, p1 * p2ho);
			ippsAdd_32fc_I(TEMP + p1 * p2ho, TEMP, p1 * p2ho);
			ippsMul_32fc_I(TWDL + num_D2 * p1 * p2ho * s1 + p1 * p2ho * s2, TEMP, p1 * p2ho);
		}

	DftiFreeDescriptor(&hand);

	/* Print results */
	printf(">>> Show the first few Fourier coefficients\n");
	for (int op = 0; op < 6; op++)
	{
		printf("%e  ", FOUT[op].re);
	}
	std::cout << "<-- FFT.re\n";
	for (int op = 0; op < 6; op++)
	{
		printf("%e  ", TEMP[op].re);
	}
	std::cout << "<-- Auto-MPFT.re\n\n";

	for (int op = 0; op < 6; op++)
	{
		printf("%e  ", FOUT[op].im);
	}
	std::cout << "<-- FFT.im\n";
	for (int op = 0; op < 6; op++)
	{
		printf("%e  ", TEMP[op].im);
	}
	std::cout << "<-- Auto-MPFT.im\n";

	mkl_free(W1);
	mkl_free(W2);
	mkl_free(XI);
	mkl_free(A);
	mkl_free(AA);
	mkl_free(BT);
	mkl_free(B2);
	mkl_free(B);
	mkl_free(C1);
	mkl_free(C2);
	mkl_free(C2T);
	mkl_free(DS);
	mkl_free(TWDL);
	mkl_free(TEMP);
	mkl_free(COUT);
	mkl_free(FOUT);
}
