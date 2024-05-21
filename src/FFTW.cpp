// http://www.fftw.org/pruned.html

#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fftw3.h>
#include <ctime>
#include <time.h>

using namespace std;

int main()
{
	fftwf_init_threads();
	fftwf_plan_with_nthreads(32);
	int N1 = 6000;
	int N2 = 4000;

	float* in;
	fftwf_complex* out;
	fftwf_plan my_plan;

	in = (float*)fftwf_malloc(sizeof(float) * N1 * N2);
	out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N1 * (N2 / 2 + 1));

	std::srand(std::time(0));
	for (int n = 0; n != N1 * N2; n++)
	{
		in[n] = (float)(((float)std::rand()) / RAND_MAX);
	}

	my_plan = fftwf_plan_dft_r2c_2d(N1, N2, in, out, FFTW_MEASURE);
	fftwf_execute(my_plan);
	fftwf_destroy_plan(my_plan);

	for (int k = 0; k < 8; k++)
	{
		cout << out[k][0] << " " << out[k][1] << " \n";
	}

	fftwf_free(in);
	fftwf_free(out);

	return 0;
}
