#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define IMAGE_DIM 2048
#define MAX_SPHERES 16

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

void output_image_file(uchar4* image);
void checkCUDAError(const char *msg);

struct Sphere {
	float   r, b, g;
	float   radius;
	float   x, y, z;
};

/* Device Code */


__device__ float sphere_intersect(Sphere *s, float ox, float oy, float *n) {
	float dx = ox - s->x;
	float dy = oy - s->y;
	float radius = s->radius;
	if (dx*dx + dy*dy < radius*radius) {
		float dz = sqrtf(radius*radius - dx*dx - dy*dy);
		*n = dz / sqrtf(radius * radius);
		return dz + s->z;
	}
	return -INF;
}

// Ex 1.2.1 (1/2)
__device__ float sphere_intersect_read_only(Sphere* s, float ox, float oy, float *n) {
	float dx = ox - s->x;
	float dy = oy - s->y;
	float radius = s->radius;
	if (dx*dx + dy*dy < radius*radius) {
		float dz = sqrtf(radius*radius - dx*dx - dy*dy);
		*n = dz / sqrtf(radius * radius);
		return dz + s->z;
	}
	return -INF;
}


// Ex 1.2.2 (1/3), La memoire constante se declare de la meme facon qu'une variable ou fonction device (cf. CM2), 
// et sera alloue au temps de compilation.
// Ecrivez le code correspondant a l'exercice 1.2.2 ici:

__constant__ unsigned int d_sphere_count;
// Fin du code pour l'exercice 1.2.2

__global__ void ray_trace(uchar4 *image, Sphere *d_s) {
	// associe les threadIdx/BlockIdx au position des pixels.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - IMAGE_DIM / 2.0f);
	float   oy = (y - IMAGE_DIM / 2.0f);

	float   r = 0, g = 0, b = 0;
	float   maxz = -INF;
	for (int i = 0; i<d_sphere_count; i++) {
		Sphere *s = &d_s[i];
		float   n;
		float   t = sphere_intersect(s, ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}

	image[offset].x = (int)(r * 255);
	image[offset].y = (int)(g * 255);
	image[offset].z = (int)(b * 255);
	image[offset].w = 255;
}

// Ex 1.2.2 (2/3)
__global__ void ray_trace_const(uchar4 *image, Sphere *d_s) {
	// associe les threadIdx/BlockIdx au position des pixels.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - IMAGE_DIM / 2.0f);
	float   oy = (y - IMAGE_DIM / 2.0f);

	float   r = 0, g = 0, b = 0;
	float   maxz = -INF;
	for (int i = 0; i<d_sphere_count; i++) {
		Sphere *s = &d_s[i];
		float   n;
		float   t = sphere_intersect(s, ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}

	image[offset].x = (int)(r * 255);
	image[offset].y = (int)(g * 255);
	image[offset].z = (int)(b * 255);
	image[offset].w = 255;
}

// Ex 1.2.1 (2/2)
__global__ void ray_trace_read_only(uchar4 *image, Sphere *d_s) {
	// associe les threadIdx/BlockIdx au position des pixels.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - IMAGE_DIM / 2.0f);
	float   oy = (y - IMAGE_DIM / 2.0f);

	float   r = 0, g = 0, b = 0;
	float   maxz = -INF;
	for (int i = 0; i<d_sphere_count; i++) {
		Sphere *s = &d_s[i];
		float   n;
		float   t = sphere_intersect(s, ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s->r * fscale;
			g = s->g * fscale;
			b = s->b * fscale;
			maxz = t;
		}
	}

	image[offset].x = (int)(r * 255);
	image[offset].y = (int)(g * 255);
	image[offset].z = (int)(b * 255);
	image[offset].w = 255;
}
/* Host code */

int main(void) {
	unsigned int image_size, spheres_size;
	uchar4 *d_image; // Donnees sur GPU correspondant a l'image qui sera genere par le kernel
	uchar4 *h_image; // Donnes sur CPU correspondant a l'image qui sera genere par le kernel
	cudaEvent_t start, stop;
	Sphere h_s[MAX_SPHERES]; // Donnees des spheres sur GPU
	Sphere *d_s; // Donnees des spheres sur GPU
	float3 timing_data; //donnees pour le timing [0]=normal, [1]=read-only, [2]=const

	// taille de l'image en octets
	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar4);
	// taille des spheres en octets
	spheres_size = sizeof(Sphere)*MAX_SPHERES;

	// creation des timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Ex 1.1.1 Allocation de la memoire pour l'image et les spheres sur le GPU
	cudaMalloc(A completer);
	cudaMalloc(A completer);
	checkCUDAError("CUDA malloc");

	// Creation random de spheres
	for (int i = 0; i<MAX_SPHERES; i++) {
		h_s[i].r = rnd(1.0f);
		h_s[i].g = rnd(1.0f);
		h_s[i].b = rnd(1.0f);
		h_s[i].x = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_s[i].y = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_s[i].z = rnd((float)IMAGE_DIM) - (IMAGE_DIM / 2.0f);
		h_s[i].radius = rnd(100.0f) + 20;
	}

	// Ex 1.2.2 (3/3), Copie les donnees dans la memoire constante
	// cudaMemcpyToSymbol(A completer, h_s, spheres_size);

	// 1.1.2 Copie de la memoire du CPU vers le GPU pour les spheres
	cudaMemcpy(A completer);
	checkCUDAError("CUDA memcpy to device");

	// Allocation de la memoire pour l'image host
	h_image = (uchar4*)malloc(image_size);

	// definition du nombre de thread par blocs et de bloc par grille
	dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3    threadsPerBlock(16, 16);

	unsigned int sphere_count = MAX_SPHERES;
	cudaMemcpyToSymbol(d_sphere_count, &sphere_count, sizeof(unsigned int));
	checkCUDAError("CUDA copy sphere count to device");

	// On genere une image a partir de nos sphere en lancant le kernel
	cudaEventRecord(start, 0);
	ray_trace << <blocksPerGrid, threadsPerBlock >> >(d_image, d_s);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timing_data.x, start, stop);
	checkCUDAError("kernel (normal)");

	// 1.2.1 (3/3) On genere une image a partir de nos sphere en lancant le kernel (avec le cache lecture seule)
	// cudaEventRecord(start, 0);
	// ray_trace_read_only << <blocksPerGrid, threadsPerBlock >> >(d_image, d_s);
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&timing_data.y, start, stop);
	// checkCUDAError("kernel (read-only)");

	// 1.2.2 (4/4) On genere une image a partir de nos sphere en lancant le kernel (avec le cache constant)
	// cudaEventRecord(start, 0);
	// ray_trace_const << <blocksPerGrid, threadsPerBlock >> >(d_image);
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&timing_data.z, start, stop);
	// checkCUDAError("kernel (const)");


	// 1.1.3 Copy le resultat d_image sur h_image du GPU vers le CPU
	cudaMemcpy(A completer);
	checkCUDAError("CUDA memcpy from device");

	//Temps en sortie
	printf("Timing Data Table\n Spheres | Normal | Read-only | Const\n");
	printf(" %-7i | %-6.3f | %-9.3f | %.3f\n", sphere_count, timing_data.x, timing_data.y, timing_data.z);

	// Image en sortie
	output_image_file(h_image);

	// On supprime les allocations memoire
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_s);
	free(h_image);

	return 0;
}

void output_image_file(uchar4* image)
{
	FILE *f; //Permet de contenir le fichier output.ppm

	//ouvre le fichier output.ppm et ecris des info en en-tete
	f = fopen("output.ppm", "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# Programmation GPU CUDA\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f); //only write rgb (ignoring a)
		}
	}
	
	fclose(f);
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
