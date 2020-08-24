#include "Includes.cuh"

#ifndef GPU_GENERAL_H
#define GPU_GENERAL_H

__device__ void D_unit_vector(float *start, float *stop, float *vec); // Gives the unit vector which points between two locations
__device__ float D_angle_between (float *A, float *B); // Given 2 unit vectors it returns the angle between them in radians
__device__ float D_gaussian ( float x, float mean, float sigma );
__device__ float Find_Intersection( float *conelist_1D_d, unsigned voxID, unsigned J, float x_start, float y_start, float z_start, float delx, float dely, float delz );

__global__ void Find_Max( float *f_d, float *voxel_max_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned ITHREADSPB );
__global__ void Cull( float *f_d, float *voxel_max_d, float CUTOFF );
__global__ void Remove_Cube_Corners( float *f_d, unsigned XDIVI, float x_start, float delx );

__global__ void Back_Project( float *f_d, float *conelist_1D_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned CONES,
                              float x_start, float y_start, float z_start, float delx, float dely, float delz );
__global__ void Interior_Sum( float *f_d, float *conelist_1D_d, float *lambda_vector_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned CONES,
                              float x_start, float y_start, float z_start, float delx, float dely, float delz );
__global__ void Iterate(      float *f_d, float *conelist_1D_d, float *lambda_vector_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned CONES,
                              float x_start, float y_start, float z_start, float delx, float dely, float delz );

#endif
