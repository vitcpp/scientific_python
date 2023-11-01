#include <stdio.h>

int main(void)
{
	double d;
	double dmin = 9007199254740990;
	double dmax = dmin + 10;
	printf("dmin = %lf\n", dmin);
	printf("dmax = %lf\n", dmax);
	for (d = dmin; d < dmax; d += 1)
		printf("%lf\n", d);
	return 0;
}
