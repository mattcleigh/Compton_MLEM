#include "../Headers/Includes.cuh"

/////////////// General GPU Functions ///////////////

__device__ void D_unit_vector(float *start, float *stop, float *vec){ // Gives the unit vector which points between two locations
  float magsq = 0;

  for (unsigned i = 0; i < 3; i++) {
      vec[i] = stop[i] - start[i];
      magsq += vec[i] * vec[i];
  }

  for (unsigned i = 0; i < 3; i++) {
    vec[i] = vec[i] * rsqrtf(magsq);
  }
}

__device__ float D_distance_between (float *start , float *stop){
  float magsq = 0;

  for (unsigned i = 0; i < 3; i++) {
      magsq += (stop[i] - start[i]) * (stop[i] - start[i]);
  }
  return sqrtf(magsq);
}

__device__ float D_angle_between (float *A, float *B){ // Given 2 unit vectors it returns the angle between them in radians
  return fabsf( acosf( A[0]*B[0] + A[1]*B[1] + A[2]*B[2] ) );
}



/////////////// Functions for Double Integration ///////////////

__device__ float D_gaussian ( float x, float mean, float sigma ){
  return expf( -(x-mean)*(x-mean)/(2*sigma*sigma) );
}

__device__ float D_gaussian_double_integrate ( float E1, float E2, float sigma, float alpha, float gamma, unsigned INTSTEP ){

  float MeCsq = 0.5109989461;

  float E1_min   = fmaxf(0, E1-3*sigma);
  float E1_max   = E1+3*sigma;
  float E1_step  = (E1_max-E1_min)/INTSTEP;

  float integral = 0;

  for (unsigned i = 0; i < INTSTEP; i++) {
    // For a given e1 value
    float e1 = E1_min + i * E1_step;

    // We find the limits of E2 to keep the influence inside the voxel
    float E2_min = - (e1/2) + ( sqrtf(e1) * sqrtf( e1 + 4*MeCsq - e1*cosf(alpha+gamma) ) ) / ( 2*sqrtf( 1 - cosf(alpha+gamma) ) );
    float E2_max = - (e1/2) + ( sqrtf(e1) * sqrtf( e1 + 4*MeCsq - e1*cosf(alpha-gamma) ) ) / ( 2*sqrtf( 1 - cosf(alpha-gamma) ) );

    float E2_step  = (E2_max-E2_min)/INTSTEP;
    float base     = E1_step*E2_step;

    for (unsigned j = 0; j < INTSTEP; j++) {

      float e2 = E2_min + j * E2_step;

      if ( fabsf( 1.0 + MeCsq * ( 1.0/(e1+e2) - 1.0/(e2) ) ) >= 1.0 ) {
        continue;
      }

      float theta1 = acosf( 1.0 + MeCsq * ( 1.0/(e1+e2) - 1.0/(e2) ) );

      if ( theta1<alpha-gamma || theta1>alpha+gamma ){
        continue;
      }

      integral += D_gaussian(e1, E1, sigma) * D_gaussian(e2, E2, sigma) * sinf(theta1) * base;

    }
  }

  return  integral;

}

__global__ void Find_Intersecting( float *conelist_1D_d, unsigned char *voxel_cone_interaction_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, long unsigned CONES,
                                   float delx, float dely, float delz, float x_start, float y_start, float z_start, unsigned INTSTEP ){

  unsigned J = threadIdx.x + blockIdx.x * blockDim.x; // The identity of the cone that we are working on

  float amp_max = 0;

  float theta = conelist_1D_d [ 6 + J * 11 ]; // The scattering angle
  float sigma = conelist_1D_d [ 7 + J * 11 ]; // Scattering angle uncertainty
  float kn    = conelist_1D_d [ 8 + J * 11 ]; // First part of the Klein-Nishina coefficient
  float E1    = conelist_1D_d [ 9 + J * 11 ]; // First energy deposition
  float E2    = conelist_1D_d [ 10 + J * 11 ]; // Second energy deposition

  for (unsigned run = 0; run < 2; run++) {

    for (unsigned i = 0; i < XDIVI; i++) {
      for (unsigned j = 0; j < YDIVI; j++) {
        for (unsigned k = 0; k < ZDIVI; k++) {

          float voxel_center[3] = { x_start + delx * (float)(i+0.5) , y_start + dely * (float)(j+0.5) , z_start + delz * (float)(k+0.5) };
          float line_between[3]{};

          D_unit_vector( &conelist_1D_d [ 0 + J * 11 ] , voxel_center , line_between );

          float alpha = D_angle_between ( &conelist_1D_d [ 3 + J * 11 ] , line_between ); // The angle from the cone axis to the cente of the voxel
          float R = D_distance_between( &conelist_1D_d [ 0 + J * 11 ] , voxel_center ); // The distance from the cone apex to the centre of the voxel
          float gamma = fabsf ( asinf ( delx / ( 2 * R ) ) ); // The angular radius of the voxel for the theta direction

          if ( fminf ( fabsf(theta-alpha+gamma) , fabsf(theta-alpha-gamma) ) < 3*sigma || (alpha-gamma-theta)*(alpha+gamma-theta) < 0 ) { //if the voxel is close enough to the gaussian of the cone, then we integrate

            float delta = fabsf ( asinf ( delx / ( 2 * R * sinf(alpha) ) ) ); // The angular radius of the voxel for the phi direction

            float integral =  D_gaussian_double_integrate( E1, E2, 0.1, alpha, gamma, INTSTEP );
            float term = ( ( R + delx/2 )*( R + delx/2 )*( R + delx/2 )/3 - ( R - delx/2 )*( R - delx/2 )*( R - delx/2 )/3 )*2*delta;

            float final_value = term * integral * ( kn - sinf(alpha)*sinf(alpha) );

            if ( run == 0 && final_value > amp_max ) amp_max = final_value;
            if ( run == 1 ) voxel_cone_interaction_d[J + i*CONES + j*XDIVI*CONES + k*XDIVI*YDIVI*CONES] = __float2int_rd(255*final_value/amp_max+0.5); // Stores the interger value into the GPU global memory

          }

        }
      }
    }

  }

}

/////////////// GPU functions for Iteration ///////////////

__global__ void Find_Max( float *f_d, float *voxel_max_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned ITHREADSPB ){

  // In this kernel we only have a single block.
  unsigned index = threadIdx.x;

  extern __shared__ float cache[];

  unsigned offset = 0;
  float temp_max = 0.0;

  while ( index + offset < (XDIVI*YDIVI*ZDIVI) ) {
    temp_max = fmaxf ( temp_max , f_d[index + offset] );
    offset += ITHREADSPB;

  }

  cache[index] = temp_max;

  __syncthreads();

  // Only the first thread will then look for the maximum within the block

  if ( index == 0 ) {

    float block_max = 0.0;

    for ( unsigned i = 0; i < ITHREADSPB; i++ ) {
      block_max = fmaxf ( block_max , cache[i] );
    }

    *voxel_max_d = block_max;

  }

}

__global__ void Interior_Sum( float *f_d, unsigned char *voxel_cone_interaction_d, float *lambda_vector_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, long unsigned CONES ){
  unsigned J = threadIdx.x + blockIdx.x * blockDim.x; // The identity of the cone that we are working on
  float sum = 0;

  for (unsigned voxel = 0; voxel < XDIVI*YDIVI*ZDIVI; voxel++) { // Now we iterate through all the voxels
    if( f_d[voxel]!=0 && voxel_cone_interaction_d[J + voxel*CONES] != 0 ){ // Which touched the current cone and is alive

      sum += voxel_cone_interaction_d[J + voxel*CONES] * f_d[voxel];

    }
  }

  lambda_vector_d[J] = sum;

}

__global__ void Iterate( float *f_d, unsigned char *voxel_cone_interaction_d, float *lambda_vector_d, long unsigned CONES ){

	unsigned voxID = threadIdx.x + blockIdx.x * blockDim.x; //The voxel this thread is currently working on

    if (f_d[voxID]!=0) { // Exclude all voxels where f is already 0, as it can never increase again

      float first_sum = 0;

      for (unsigned C = 0; C < CONES; C++){ // Now we iterate through all the cones

        if( voxel_cone_interaction_d[C + voxID*CONES] != 0 ){ // Which touched the current voxel

        first_sum += (float)(voxel_cone_interaction_d[C + voxID*CONES])  / lambda_vector_d[C];

        }
      }

    f_d[voxID] *= first_sum;

    }

}

__global__ void Cull( float *f_d, unsigned char *voxel_cone_interaction_d, float *voxel_max_d, long unsigned CONES, float CUTOFF ){
  unsigned trID = threadIdx.x + blockIdx.x * blockDim.x;

  if ( (f_d[ trID ] > 0) && (f_d[ trID ] < *voxel_max_d * CUTOFF) ) { // If the f value of a voxel gets too small

    f_d[ trID ] = 0; // We set the value straight to zero

    for (unsigned C = 0; C < CONES; C++){ // We also declare the voxel as dead
      voxel_cone_interaction_d[C + trID * CONES] = 0;

    }

  }
}


//
