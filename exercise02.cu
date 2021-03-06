#ifndef __CUDACC__
#define __CUDACC__
#endif
#pragma warning(disable : 4996)
#pragma diag_suppress = deprecated_entity
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"

#define IMAGE_DIM 2048
#define SAMPLE_SIZE 6
#define NUMBER_OF_SAMPLES (((SAMPLE_SIZE*2)+1)*((SAMPLE_SIZE*2)+1))

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

void output_image_file(uchar4* image);
void input_image_file(const char* filename, uchar4* image);
void checkCUDAError(const char *msg);

// Ex 2.1.1, (1/3)

// Ex 2.2.1, (2/3)


__global__ void image_blur(uchar4 *image, uchar4 *image_output) {
	// associe les threadIdx/BlockIdx au position des pixels.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int output_offset = x + y * blockDim.x * gridDim.x;
	uchar4 pixel;
	float4 average = make_float4(0, 0, 0, 0);

	for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++){
		for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++){
			int x_offset = x + i;
			int y_offset = y + j;
			// On fait attention aux conditions de bords
			if (x_offset < 0)
				x_offset += IMAGE_DIM;
			if (x_offset >= IMAGE_DIM)
				x_offset -= IMAGE_DIM;
			if (y_offset < 0)
				y_offset += IMAGE_DIM;
			if (y_offset >= IMAGE_DIM)
				y_offset -= IMAGE_DIM;
			int offset = x_offset + y_offset * blockDim.x * gridDim.x;
			pixel = image[offset];

			// On somme les valeurs
			average.x += pixel.x;
			average.y += pixel.y;
			average.z += pixel.z;
		}
	}
	// On cacul la moyenne
	average.x /= (float)NUMBER_OF_SAMPLES;
	average.y /= (float)NUMBER_OF_SAMPLES;
	average.z /= (float)NUMBER_OF_SAMPLES;

	image_output[output_offset].x = (unsigned char)average.x;
	image_output[output_offset].y = (unsigned char)average.y;
	image_output[output_offset].z = (unsigned char)average.z;
	image_output[output_offset].w = 255;
}

// Ex 2.1.2, (2/3)
__global__ void image_blur_texture1D(uchar4 *image_output) {
	// associe les threadIdx/BlockIdx au position des pixels.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int output_offset = x + y * blockDim.x * gridDim.x;
	uchar4 pixel;
	float4 average = make_float4(0, 0, 0, 0);

	for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++){
		for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++){
			int x_offset = x + i;
			int y_offset = y + j;
			// On fait attention aux conditions de bords
			if (x_offset < 0)
				x_offset += IMAGE_DIM;
			if (x_offset >= IMAGE_DIM)
				x_offset -= IMAGE_DIM;
			if (y_offset < 0)
				y_offset += IMAGE_DIM;
			if (y_offset >= IMAGE_DIM)
				y_offset -= IMAGE_DIM;
			int offset = x_offset + y_offset * blockDim.x * gridDim.x;
			// 2.1.2 Modifier l'instruction suivante
			// Afin d'utiliser la texture 1D
			pixel = 0;

			// On somme les valeurs
			average.x += pixel.x;
			average.y += pixel.y;
			average.z += pixel.z;
		}
	}
	// On cacul la moyenne
	average.x /= (float)NUMBER_OF_SAMPLES;
	average.y /= (float)NUMBER_OF_SAMPLES;
	average.z /= (float)NUMBER_OF_SAMPLES;

	image_output[output_offset].x = (unsigned char)average.x;
	image_output[output_offset].y = (unsigned char)average.y;
	image_output[output_offset].z = (unsigned char)average.z;
	image_output[output_offset].w = 255;
}

// Ex 2.2.2, (2/3)
__global__ void image_blur_texture2D(uchar4 *image_output) {
	// associe les threadIdx/BlockIdx au position des pixels.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int output_offset = x + y * blockDim.x * gridDim.x;
	uchar4 pixel;
	float4 average = make_float4(0, 0, 0, 0);

	for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++){
		for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++){
			int x_offset = x + i;
			int y_offset = y + j;
			// 2.2.2 On recupere la texture 2D
			pixel = 0;

			// On somme les valeurs
			average.x += pixel.x;
			average.y += pixel.y;
			average.z += pixel.z;
		}
	}
	// On calcule la moyenne
	average.x /= (float)NUMBER_OF_SAMPLES;
	average.y /= (float)NUMBER_OF_SAMPLES;
	average.z /= (float)NUMBER_OF_SAMPLES;

	image_output[output_offset].x = (unsigned char)average.x;
	image_output[output_offset].y = (unsigned char)average.y;
	image_output[output_offset].z = (unsigned char)average.z;
	image_output[output_offset].w = 255;
}


/* Host code */

int main(void) {
	unsigned int image_size;
	uchar4 *d_image, *d_image_output;
	uchar4 *h_image;
	cudaEvent_t start, stop;
	float3 ms; //[0]=normal,[1]=tex1d,[2]=tex2d

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar4);

	// creation des timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocation de la memoire sur le GPU pour l'image en sortie
	cudaMalloc((void**)&d_image, image_size);
	cudaMalloc((void**)&d_image_output, image_size);
	checkCUDAError("CUDA malloc");

	// allocate and load host image
	h_image = (uchar4*)malloc(image_size);
	input_image_file("input.ppm", h_image);

	// copy image to device memory
	cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	// Ex 2.2.3
	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	//sample2D.addressMode[0] = cudaAddressModeWrap;
	//sample2D.addressMode[1] = cudaAddressModeWrap;

	// definition du nombre de thread par blocs et de bloc par grille
	dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3    threadsPerBlock(16, 16);

	// normal version
	cudaEventRecord(start, 0);
	image_blur << <blocksPerGrid, threadsPerBlock >> >(d_image, d_image_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms.x, start, stop);
	checkCUDAError("kernel normal");

	// Ex 2.1.3, (3/3) cudaBindTexture() est utilis?? pour li?? la texture CUDA ?? la m??moire que l'on ?? allou??
	// cudaBindTexture(0, sample1D, d_image, image_size);
	// checkCUDAError("tex1D bind");
	// cudaEventRecord(start, 0);
	// image_blur_texture1D << <blocksPerGrid, threadsPerBlock >> >(d_image_output);
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&ms.y, start, stop);
	// cudaUnbindTexture(sample1D);
	// checkCUDAError("kernel tex1D");

	// Ex 2.2.3, (3/3) cudaBindTexture2D() est utilis?? pour lier la texture CUDA ?? la m??moire que nous avons allou??e.
	// Il y a beaucoup plus d'options ici, pour sp??cifier la fa??on dont la m??moire est r??partie.
	// Par exemple, certains formats de stockage d'images allouent la m??moire avec une largeur sup??rieure au nombre de pixels.
	// cudaBindTexture2D(0, sample2D, d_image, desc, IMAGE_DIM, IMAGE_DIM, IMAGE_DIM*sizeof(uchar4));
	// checkCUDAError("tex2D bind");
	// cudaEventRecord(start, 0);
	// image_blur_texture2D << <blocksPerGrid, threadsPerBlock >> >(d_image_output);
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&ms.z, start, stop);
	// checkCUDAError("kernel tex2D");
	// cudaUnbindTexture(sample2D);


	// Copie l'image resultante du GPU vers le CPU
	cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Affichage des temps
	printf("Execution times:\n");
	printf("\tNormal version: %f\n", ms.x);
	printf("\ttex1D version: %f\n", ms.y);
	printf("\ttex2D version: %f\n", ms.z);

	// image en sortie
	output_image_file(h_image);

	// On supprime les allocation memoire
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_image_output);
	free(h_image);

	return 0;
}

void output_image_file(uchar4* image)
{
	FILE *f; //Permet de gerer le fichier output.ppm

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
			fwrite(&image[i], sizeof(unsigned char), 3, f);
		}
	}

	fclose(f);
}

void input_image_file(const char* filename, uchar4* image)
{
	FILE *f; 
	char temp[256];
	unsigned int x, y, s;

	f = fopen("input.ppm", "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'input.ppm' input file\n");
		exit(1);
	}
	fscanf(f, "%s\n", &temp);
	fscanf(f, "%d %d\n", &x, &y);
	fscanf(f, "%d\n", &s);
	if ((x != y) && (x != IMAGE_DIM)){
		fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
		exit(1);
	}

	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fread(&image[i], sizeof(unsigned char), 3, f);
			//image[i].w = 255;
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
