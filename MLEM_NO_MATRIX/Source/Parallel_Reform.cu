// My Header Files
#include "../Headers/Includes.cuh"
#include "../Headers/CPUHeaders.cuh"
#include "../Headers/GraphicsCardHeaders.cuh"


int main() {
  float time_start = clock();


  /////////////// Defining initial parameters and variables ///////////////
  // We start by delcaring the input parameters
  string NAME, OUTPUTMOD;
  float x_start, x_end, y_start, y_end, z_start, z_end, CUTOFF, UE;
  unsigned XDIVI, YDIVI, ZDIVI, CONES, ITHREADSPB, IBLOCKS, CTHREADSPB, CBLOCKS, TOTALIT, IT, SAVEEVERY;
  bool CORE_OUT, MATT_OUT;

  // Next we import the input parameters from the setup file
  InputSetup( NAME, OUTPUTMOD, IT, x_start, x_end, y_start, y_end, z_start, z_end, XDIVI, YDIVI, ZDIVI,
              ITHREADSPB, IBLOCKS, CONES, CTHREADSPB, CBLOCKS, TOTALIT, SAVEEVERY, CUTOFF, UE,
              CORE_OUT, MATT_OUT );

  // The strings of the input and output files we are going to use
  string input = "../Input/Filtered/" + NAME + ".csv";
  string output =  "../Output/" + NAME + OUTPUTMOD + "_C" + std::to_string(CONES) + "_x" + std::to_string(XDIVI) + "y" + std::to_string(YDIVI) + "z" + std::to_string(ZDIVI) + "_I"; // Output file name

  // Now we must check that the input parameters are acceptible
  if ( InputDebugger( XDIVI, YDIVI, ZDIVI, ITHREADSPB, IBLOCKS, CONES, CTHREADSPB, CBLOCKS, CORE_OUT, MATT_OUT, IT, output, input  ) != 0 ) return 0;

  // Now we can define the widths of the voxel
  const float delx = (x_end - x_start)/XDIVI;
  const float dely = (y_end - y_start)/YDIVI;
  const float delz = (z_end - z_start)/ZDIVI;

  // This contains the locations of the centres of each voxel
  vector<vector<vector<vector<float> > > > position_matrix(XDIVI, vector<vector<vector<float> > >( YDIVI, vector<vector<float> >( ZDIVI , {0,0,0} ) ) );

  // A flattened list of all the cones, with 10 elements per cone, a single cone has elements:
  // {location of first colision x ,y ,z} , {axis of the cone x ,y ,z} , angle , uncertainty , kn-coefficient
  float* conelist_1D = new float [ CONES * 9 ]{};

  // The f distribution which we want to calculate
  float* f = new float [XDIVI*YDIVI*ZDIVI]{};

  DefinePositions( position_matrix, f, output, IT, XDIVI, YDIVI, ZDIVI, x_start, y_start, z_start, delx, dely, delz);
  CreateCones( conelist_1D, input, CONES, UE);

  /////////////// We start the data transfer to the GPU ///////////////
  cout << "Proceeding with Iteration:" << '\n';
  float time_section_start = clock();
  cudaDeviceReset();

  float *conelist_1D_d;
  cudaMalloc( &conelist_1D_d, CONES * 9 * sizeof(float) );
  cudaMemcpy( conelist_1D_d, conelist_1D, CONES * 9 * sizeof(float), cudaMemcpyHostToDevice );

  float *f_d; // This is the radioactive distribution of the volume, it is what is iterated and what we want to esitmate
  cudaMalloc( &f_d , XDIVI * YDIVI * ZDIVI * sizeof(float) );
  cudaMemcpy( f_d, f, XDIVI * YDIVI * ZDIVI * sizeof(float), cudaMemcpyHostToDevice );

  float *lambda_vector_d; // This is the vector which will contain the interior sum of the MLEM for each cone
  cudaMalloc( &lambda_vector_d, CONES * sizeof(float) );
  cudaMemset( lambda_vector_d, 0, CONES * sizeof(float) );

  float *voxel_max_d; // This is the vector which will contain the interior sum of the MLEM for each cone
  cudaMalloc( &voxel_max_d, sizeof(float) );
  cudaMemset( voxel_max_d, 0, sizeof(float) );

  if ( MemDebugger( XDIVI, YDIVI, ZDIVI ) != 0 ) return 0; // We chack the device memory


  // If we arent loading, then we do a back projection
  if (IT == 0) {
    Back_Project <<< IBLOCKS, ITHREADSPB >>> ( f_d, conelist_1D_d, XDIVI, YDIVI, ZDIVI, CONES, x_start, y_start, z_start, delx, dely, delz );
    cudaMemcpy(f, f_d, XDIVI*YDIVI*ZDIVI*sizeof(float), cudaMemcpyDeviceToHost);
    if (MATT_OUT) StoreF_MATT( f, 0, output, position_matrix, XDIVI, YDIVI, ZDIVI );
    if (CORE_OUT) StoreF_CORE( f, 0, output, XDIVI, YDIVI, ZDIVI, x_start, y_start, z_start, delx, dely, delz );
  }

  // Now the iteration process itself
  float time_iter;
  while (IT <= TOTALIT) {
    time_iter = clock();

    // GPU Functions are called with the parameters << blocks , threads_per_block >>
    Interior_Sum      <<< CBLOCKS, CTHREADSPB >>> ( f_d, conelist_1D_d, lambda_vector_d, XDIVI, YDIVI, ZDIVI, CONES, x_start, y_start, z_start, delx, dely, delz );
    Iterate           <<< IBLOCKS, ITHREADSPB >>> ( f_d, conelist_1D_d, lambda_vector_d, XDIVI, YDIVI, ZDIVI, CONES, x_start, y_start, z_start, delx, dely, delz );
    Find_Max          <<< 1, ITHREADSPB, ITHREADSPB*sizeof(float) >>> ( f_d, voxel_max_d, XDIVI, YDIVI, ZDIVI, ITHREADSPB ); // This function has the extra parameter. The amount of shared memory.
    Cull              <<< IBLOCKS, ITHREADSPB >>> ( f_d, voxel_max_d, CUTOFF );
    cudaDeviceSynchronize();
    IT++;

    // For printing the remaining time of the iterations
    Print_Time_Remaining( time_iter, clock(), IT, TOTALIT );

    // Now we copy back and save the f values at regular intervals
    if (IT%SAVEEVERY == 0 || IT == TOTALIT ) {
      cudaMemcpy(f, f_d, XDIVI*YDIVI*ZDIVI*sizeof(float), cudaMemcpyDeviceToHost);
      if (MATT_OUT) StoreF_MATT( f, IT, output, position_matrix, XDIVI, YDIVI, ZDIVI );
      if (CORE_OUT) StoreF_CORE( f, IT, output, XDIVI, YDIVI, ZDIVI, x_start, y_start, z_start, delx, dely, delz );

    }

  }

  Print_Time_Complete( time_section_start, clock() );

  // Now we can free the memory on the GPU and CPU completely
  cudaFree(f_d);
  cudaFree(lambda_vector_d);
  cudaFree(voxel_max_d);
  delete [] f;
  delete [] conelist_1D;

  Print_Time_Complete( time_start, clock(), 1 );

  cudaDeviceReset();
  return 0;
}








//
