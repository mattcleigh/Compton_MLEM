#include "Includes.cuh"

#ifndef GPU_GENERAL_H
#define GPU_GENERAL_H

__device__ void D_unit_vector(float *start, float *stop, float *vec); // Gives the unit vector which points between two locations
__device__ float D_angle_between (float *A, float *B); // Given 2 unit vectors it returns the angle between them in radians
__device__ float D_distance_between (float *start , float *stop); // Gives the distance between two 3D locations

__device__ float D_gaussian ( float x, float mean, float sigma );
__device__ float D_test_gaussian_gaussian_sine ( float e1, float e2, float E1, float E2, float sigma, float alpha, float gamma );
__device__ float D_gaussian_double_integrate ( float E1, float E2, float sigma, float alpha, float gamma, unsigned INTSTEP );

__global__ void Find_Max( float *f_d, float *voxel_max_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned ITHREADSPB );

#endif

#ifndef GPU_REFORM_H
#define GPU_REFORM_H

__global__ void Find_Intersecting( float *conelist_1D_d , unsigned char *voxel_cone_interaction_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, long unsigned CONES,
                                   float delx, float dely, float delz, float x_start, float y_start, float z_start, unsigned INTSTEP );
__global__ void Interior_Sum( float *f_d, unsigned char *voxel_cone_interaction_d, float *lambda_vector_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, long unsigned CONES);
__global__ void Iterate( float *f_d, unsigned char *voxel_cone_interaction_d, float *lambda_vector_d, long unsigned CONES );
__global__ void Cull( float *f_d, unsigned char *voxel_cone_interaction_d, float *voxel_max_d, long unsigned CONES, float CUTOFF );

#endif
