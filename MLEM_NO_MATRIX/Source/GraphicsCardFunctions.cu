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

__device__ float D_angle_between (float *A, float *B){ // Given 2 unit vectors it returns the angle between them in radians
  return fabsf( acosf( A[0]*B[0] + A[1]*B[1] + A[2]*B[2] ) );
}

__device__ float D_gaussian ( float value , float mean , float sigma ){
  return expf ( - ( value - mean ) * ( value - mean ) / ( 2 * sigma * sigma ) );
}

__device__ float Find_Intersection( float *conelist_1D_d, unsigned voxID, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned J,
                                   float x_start, float y_start, float z_start, float delx, float dely, float delz ){

  unsigned k = voxID / (XDIVI*YDIVI);
  unsigned j = voxID / XDIVI - k*YDIVI;
  unsigned i = voxID - j*XDIVI - k*XDIVI*YDIVI;

  float theta = conelist_1D_d [ 6 + J * 9 ]; // The scattering angle
  float sigma = conelist_1D_d [ 7 + J * 9 ]; // Scattering angle uncertainty
  float KN    = conelist_1D_d [ 8 + J * 9 ]; // The first part of the Klein Nishina coefficiant

  if (sigma>0.09) sigma = 0.09;

  float voxel_center[3] = { x_start + delx * ( (float)(i) + 0.5f) , y_start + dely * ( (float)(j) + 0.5f) , z_start + delz * ( (float)(k) + 0.5f) };
  float line_between[3]{};

  D_unit_vector( &conelist_1D_d [ 0 + J * 9 ] , voxel_center , line_between );

  float alpha = D_angle_between ( &conelist_1D_d [ 3 + J * 9 ] , line_between ); // The angle from the cone axis to the cente of the voxel

  if ( fabsf(theta-alpha) < 3*sigma ) { // If the voxel is close enough to the gaussian of the cone, then we evaluate
    return D_gaussian(alpha, theta, sigma) * ( KN - sinf(alpha)*sinf(alpha) );
  } else {
    return 0;
  }

}


/////////////// GPU functions for Iteration ///////////////

__global__ void Find_Max( float *f_d, float *voxel_max_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned ITHREADSPB ){

  // In this kernel we only have a single block.
  unsigned index = threadIdx.x;

  extern __shared__ float cache[];

  unsigned offset = 0;
  float temp_max = 0;

  unsigned k;
  unsigned j;
  unsigned i;
  unsigned voxID;

  while ( index + offset < (XDIVI*YDIVI*ZDIVI) ) {
    voxID = index + offset;

    k = voxID / (XDIVI*XDIVI);
    j = voxID / XDIVI - k*XDIVI;
    i = voxID - j*XDIVI - k*XDIVI*XDIVI;

    // We do not include the edges our search for the max
    if ( i!=(XDIVI-1) && i!= 0 && j!=(YDIVI-1) && j!= 0 && k!=(ZDIVI-1) && k!= 0 ) {
      temp_max = fmaxf ( temp_max , f_d[index + offset] );
    }

    offset += ITHREADSPB;

  }

  cache[index] = temp_max;

  __syncthreads();

  // Only the first thread will then look for the maximum within the block

  if ( index == 0 ) {

    float block_max = 0;

    for ( unsigned i = 0; i < ITHREADSPB; i++ ) {
      block_max = fmaxf ( block_max , cache[i] );
    }

    *voxel_max_d = block_max;

  }

}

__global__ void Back_Project( float *f_d, float *conelist_1D_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned CONES,
                              float x_start, float y_start, float z_start, float delx, float dely, float delz ){

	unsigned voxID = threadIdx.x + blockIdx.x * blockDim.x; //The voxel this thread is currently working on

  float project = 0;
  for (unsigned J = 0; J < CONES; J++){ // Now we iterate through all the cones
    project += Find_Intersection( conelist_1D_d, voxID, XDIVI, YDIVI, ZDIVI, J, x_start, y_start, z_start, delx, dely, delz );
  }

  f_d[voxID] = project;

}

__global__ void Interior_Sum( float *f_d, float *conelist_1D_d, float *lambda_vector_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned CONES,
                              float x_start, float y_start, float z_start, float delx, float dely, float delz ){
  unsigned J = threadIdx.x + blockIdx.x * blockDim.x; // The identity of the cone that we are working on
  float sum = 0;
  float interaction;

  for (unsigned voxID = 0; voxID < XDIVI*YDIVI*ZDIVI; voxID++) { // Now we iterate through all the voxels
    if( f_d[voxID] != 0 ){

      interaction = Find_Intersection( conelist_1D_d, voxID, XDIVI, YDIVI, ZDIVI, J, x_start, y_start, z_start, delx, dely, delz );

      if( interaction != 0 ){ // Which touched the current cone and is alive

        sum += interaction * f_d[voxID];

      }

    }

  }

  lambda_vector_d[J] = sum;

}

__global__ void Iterate( float *f_d, float *conelist_1D_d, float *lambda_vector_d, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned CONES,
                         float x_start, float y_start, float z_start, float delx, float dely, float delz ){

	unsigned voxID = threadIdx.x + blockIdx.x * blockDim.x; //The voxel this thread is currently working on

    if ( f_d[voxID] != 0 ) { // Exclude all voxels where f is already 0, as it can never increase again

      float first_sum = 0;
      float interaction;

      for (unsigned J = 0; J < CONES; J++){ // Now we iterate through all the cones

        interaction = Find_Intersection( conelist_1D_d, voxID, XDIVI, YDIVI, ZDIVI, J, x_start, y_start, z_start, delx, dely, delz );

        if( interaction != 0 ){ // Which touched the current voxel

        first_sum += interaction / lambda_vector_d[J];

        }
      }

    f_d[voxID] *= first_sum;

    }

}

__global__ void Remove_Cube_Corners( float *f_d, unsigned XDIVI, float x_start, float delx ){
  // This is reduce the effects or corner hotspots by rounding the edges of the space.
  // At the moment it only works for cubic spaces cenetered on 0 (XDIVI=YDIVI+ZDIVI, xstart=ystart...)
  unsigned voxID = threadIdx.x + blockIdx.x * blockDim.x; //The voxel this thread is currently working on

  unsigned k = voxID / (XDIVI*XDIVI);
  unsigned j = voxID / XDIVI - k*XDIVI;
  unsigned i = voxID - j*XDIVI - k*XDIVI*XDIVI;

  float voxel_center[3] = { x_start + delx * (float)(i+0.5) , x_start + delx * (float)(j+0.5) , x_start + delx * (float)(k+0.5) };
  float roun_val = powf(voxel_center[0],4) + powf(voxel_center[1],4) + powf(voxel_center[2],4);

  if ( roun_val >= (powf(XDIVI*delx,4)/16) ) { // If the voxel lies outside the rounded cube

    f_d[ voxID ] = 0; // We set the value straight to zero

  }
}

__global__ void Cull( float *f_d, float *voxel_max_d, float CUTOFF ){
  unsigned voxID = threadIdx.x + blockIdx.x * blockDim.x; //The voxel this thread is currently working on

  if ( (f_d[ voxID ] != 0) && (f_d[ voxID ] < *voxel_max_d * CUTOFF) ) { // If the f value of a voxel gets too small

    f_d[ voxID ] = 0; // We set the value straight to zero

  }
}











//
