#include "../Headers/Includes.cuh"

/////////////// Importing the Setup Paramaters ///////////////
void InputSetup( string &NAME, string &OUTPUTMOD, unsigned &IT, float &x_start, float &x_end, float &y_start,
                float &y_end, float &z_start, float &z_end, unsigned &XDIVI, unsigned &YDIVI,
                unsigned &ZDIVI, unsigned &ITHREADSPB, unsigned &IBLOCKS, unsigned &CONES,
                unsigned &CTHREADSPB, unsigned &CBLOCKS, unsigned &TOTALIT, unsigned &SAVEEVERY,
                float &CUTOFF, float &UE, bool &CORE_OUT, bool &MATT_OUT ){

  cout << "\nReading Setup File:\n";

  fstream file ( "Setup.txt" );
  string line;

  vector<float> all_numbers;

  while ( getline(file, line) ) {
    char start = line[0];
    string entry;

    stringstream sep(line);
    string cell;

    while ( getline ( sep, cell, '=') ) {
      entry = cell.c_str();
    }

    // The input file name
    if ( start == '1' ) NAME = entry.substr(1);

    // The output file modifier
    else if ( start == '2' ){
      OUTPUTMOD = entry.substr(1);
      if ( OUTPUTMOD[0] == ')' || OUTPUTMOD.length() == 0 ) OUTPUTMOD = "";
      else OUTPUTMOD = "_" + OUTPUTMOD;
    }

    else if ( start == '3' ){
      string check = entry.substr(1);
      if ( check[0] == ')' || check.length() == 0 ) IT = 0;
      else IT = atoi( check.c_str() );
    }


    // The setup numbers
    else if ( start != ' ' ) {
        stringstream sep(entry);
        string piece;

        while ( getline( sep, piece, ',' ) ) {
          all_numbers.push_back( atof( piece.c_str() ) );
        }
    }

  }

  x_start      = all_numbers[0];
  x_end        = all_numbers[1];
  y_start      = all_numbers[2];
  y_end        = all_numbers[3];
  z_start      = all_numbers[4];
  z_end        = all_numbers[5];
  XDIVI        = all_numbers[6];
  YDIVI        = all_numbers[7];
  ZDIVI        = all_numbers[8];
  ITHREADSPB   = all_numbers[9];
  IBLOCKS      = all_numbers[10];
  CONES        = all_numbers[11];
  CTHREADSPB   = all_numbers[12];
  CBLOCKS      = all_numbers[13];
  TOTALIT      = all_numbers[14];
  SAVEEVERY    = all_numbers[15];
  CUTOFF       = all_numbers[16];
  UE           = all_numbers[17];
  CORE_OUT     = all_numbers[18];
  MATT_OUT     = all_numbers[19];

  cout << " -- Done\n";
}



/////////////// Debuggers ///////////////
int InputDebugger( unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned ITHREADSPB, unsigned IBLOCKS,
                   unsigned CONES, unsigned CTHREADSPB, unsigned CBLOCKS, bool CORE_OUT, bool MATT_OUT, unsigned IT, string output, string input ){
  int errors = 0;
  int thread_check = XDIVI*YDIVI*ZDIVI - ITHREADSPB*IBLOCKS;
  if ( thread_check != 0 ){
    cout << "\nWhoops!!! Number of VOXELS does not equal the number of called threads!!!\n";
    errors++;
  }

  thread_check = CONES - CTHREADSPB*CBLOCKS;
  if ( thread_check != 0 ){
    cout << "\nWhoops!!! Number of CONES does not equal the number of called threads!!!\n";
    errors++;
  }

  if ( thread_check != 0 ){
    cout << "\nWhoops!!! Number of CONES does not equal the number of called threads!!!\n";
    errors++;
  }

  if ( CORE_OUT == 0 && MATT_OUT == 0){
    cout << "\nWhoops!!! You aren't saving any data!!! Change this an run again!!!\n";
    errors++;
  }

  ifstream infile ( input );
  if ( !infile.good() ){
      cout << "\nWhoops!!! Input data-file not found in Input/Filtered. Please try again!!!\n";
    errors++;
  }
  infile.close();

  if ( IT != 0 ){
    ifstream file ( output + to_string(IT) + string(".csv") );
    if ( !file.good() ) {
      cout << "\nCant locate: " <<  output + to_string(IT) + string(".csv\n");
      cout << "\nWhoops!!! You are trying to load previously iterated file that doesnt exist in the Output folder!!!\n";
      errors++;
    }
    file.close();
  }

  if ( errors != 0 ) {
    cout << "\nProgram Aborted :(\n\n";
  }

  return errors;
}

int MemDebugger( unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI ){
  size_t free_mem;
  size_t total_mem;

  cudaMemGetInfo ( &free_mem, &total_mem );

  size_t used_mem = total_mem - free_mem;
  size_t matrix_mem = XDIVI * YDIVI * ZDIVI * sizeof(float);

  if( used_mem < matrix_mem ){
    cout << "\nWhoops!!! GPU ran out of memory. Reduce the number of VOXELS or CONES\n";
    cout << "\nProgram Aborted :(\n";
    return 1;
  }

  float percent_mem_used = 100*(float)used_mem / total_mem;

  printf (" -- Percentage of GPU memory used = %2.2f%% \n", percent_mem_used);

  return 0;
}

/////////////// Time-Printout ///////////////
void Print_Time_Remaining( float clock_start , float clock_end , unsigned IT, unsigned TOTALIT ){
  unsigned total_time =  (int)( (TOTALIT-IT)*(clock_end - clock_start)/(CLOCKS_PER_SEC) );
  unsigned minutes = total_time/60;
  unsigned seconds = total_time%60;

  if ( minutes > 0 ) printf( " -- %4u -- Time Remaining = %4u minutes and %4u seconds   \r" , IT , minutes , seconds );
  else               printf( " -- %4u -- Time Remaining = %4u seconds                   \r" , IT , seconds );
  cout.flush();
}

void Print_Time_Complete( float clock_start , float clock_end , bool fin = 0 ){
  float total_time =  (clock_end - clock_start)/(CLOCKS_PER_SEC);
  unsigned minutes = total_time/60;
  float seconds = total_time - minutes*60;

  string pref = " -- Time Taken =";
  string post = "            \n -- Done\n";

  if ( fin == 1 ){
    pref = "\nReconstruction Complete \nTotal Runtime =";
    post = "            \n\n";
  }

  if ( minutes > 0 ) printf( "%s %4u minutes and %4.2f seconds %s" , pref.c_str() , minutes , seconds , post.c_str() );
  else               printf( "%s %4.2f seconds %s" , pref.c_str() , seconds , post.c_str() );
}



/////////////// A Couple of Vector Functions ///////////////
vector<float> ScalVec(float c, vector<float> x){ // Just a simple scalar x vector function
  unsigned len = x.size();
  vector<float> z(len);
  for (unsigned i = 0; i < len; i++) {
    z[i] = c * x[i];
  }
  return z;
}
vector<float> unit_vector(vector<float> start, vector<float> stop){ // Gives the unit vector which points between two locations
  unsigned len = start.size();
  float magsq = 0;
  vector<float> vec(len);
  for (unsigned i = 0; i < len; i++) {
      vec[i] = stop[i] - start[i];
      magsq += vec[i] * vec[i];

  }
  return ScalVec( 1.0/sqrt(magsq) , vec );
}

/////////////// Listing functions ///////////////
void DefinePositions( vector<vector<vector<vector<float> > > > &position_matrix, float *f, string output, unsigned IT, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI,
                      float x_start, float y_start, float z_start, float delx, float dely, float delz){ // A function which gives the position values to the position matrix
  cout << "Defining lattice Positions and Values:\n";

  for (unsigned i = 0; i < XDIVI; i++) {
    for (unsigned j = 0; j < YDIVI; j++) {
      for (unsigned k = 0; k < ZDIVI; k++) {

        position_matrix[i][j][k] = { x_start + delx * ( (float)(i) + 0.5f) , y_start + dely * ( (float)(j) + 0.5f) , z_start + delz * ( (float)(k) + 0.5f) };

        f[ i + j * XDIVI + k * XDIVI * YDIVI ] = 0.0; // Initial guess for f is one everywhere

      }
    }
  }



  if (IT != 0){
    ifstream file ( output + to_string(IT) + string(".csv") );
    vector<float> linedata;
    string line;
    unsigned i, j, k;
    float load_f;

    while( getline ( file, line, '\n')){
      stringstream sep(line);
      string cell;

      while (getline ( sep, cell, ',')) {
        linedata.push_back( atof(cell.c_str()) );
      }

      load_f = linedata[0];
      i = linedata[4];
      j = linedata[5];
      k = linedata[6];
      f[ i + j * XDIVI + k * XDIVI * YDIVI ] = load_f;

      linedata.clear();
    }
  }


  cout << " -- Done\n";
}

float UncertaintyDoubleScatter( float E1, float E2, float UE ){
  float MeCsq = 0.5109989461;
    return sqrt( fabs( ( (pow(E1,4) + 4 * pow(E1,3) * E2 + 4 * pow(E1,2) * pow(E2,2) + pow(E2,2)) * MeCsq * pow(UE,2)) / ( E1 * pow(E2,2) * pow(E1+E2,2) * ( 2 * E2 * (E1+E2) - E1 * MeCsq ) ) ) );
}

float ScatteringAngle( float E1, float E2 ){
  float MeCsq = 0.5109989461;
  float value = 1.0 + MeCsq * ( 1.0/(E1+E2) - 1.0/(E2) );

  if ( fabs(value) < 1 ) {
    return acos( value );
  }

  return 0;
}

float KleinNishina( float E1, float E2 ){
  float E0 = E1 + E2;
  return ( E2/E0 + E0/E2 );
}

void CreateCones( float* conelist_1D , string input, unsigned &CONES, float UE ){ // Creating the list of cones by importing from DATAFILE
  cout << "Importing Input Data to Cone-Matrix:\n";

  ifstream file ( input );
  vector<float> linedata;
  string line;
  float E1; // The energy deposited at the first scattering location
  float E2; // The energy deposited at the second scattering location
  float theta; // The scattering angle
  float utheta; // The uncertainty on the scattering angle
  float KN; // The first part of the Klein Nishina coefficiant
  unsigned errors = 0;


  for (unsigned i = 0; i < CONES; i++) {

    getline ( file, line, '\n');
    stringstream sep(line);
    string cell;

    while (getline ( sep, cell, ',')) {
      linedata.push_back( atof(cell.c_str()) );
    }

    E1 = linedata[0];
    E2 = linedata[4];

    theta = ScatteringAngle( E1, E2 );
    utheta = UncertaintyDoubleScatter( E1, E2, UE );
    KN = KleinNishina( E1, E2 );

    if ( theta == 0 ){
      errors++;
      KN = 0; // This effectively kills all weighting of an unphysical cone
    }

    vector<float> axis =  unit_vector ( { linedata[5] , linedata[6] , linedata[7] } , { linedata[1] , linedata[2] , linedata[3] } );
    //The axis is the unit vector which points in the direction from the second scatter to the first

    conelist_1D [ 0 + i * 9 ] = linedata[1]; // First scattering location
    conelist_1D [ 1 + i * 9 ] = linedata[2];
    conelist_1D [ 2 + i * 9 ] = linedata[3];

    conelist_1D [ 3 + i * 9 ] = axis[0]; // Axis of the cone
    conelist_1D [ 4 + i * 9 ] = axis[1];
    conelist_1D [ 5 + i * 9 ] = axis[2];

    conelist_1D [ 6 + i * 9 ] = theta;
    conelist_1D [ 7 + i * 9 ] = utheta;

    conelist_1D [ 8 + i * 9 ] = KN;

    linedata.clear();
  }
  file.close();

  if ( errors ) {
    cout << " -- Warning: " << errors << " cones were found to be be unphysical\n";
  }

  cout << " -- Done\n";
}

/////////////// Printing the data ///////////////
void StoreF_MATT( float *f, unsigned IT, string output, vector<vector<vector<vector<float> > > > position_matrix, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI ){ // How we store the final f values, only need the non-zero voxels
  ofstream outfile;
  string name = string(output) + to_string(IT) + string(".csv");
  outfile.open ( name );
  outfile.precision(7);

  for (unsigned i = 0; i < XDIVI; i++) {
    for (unsigned j = 0; j < YDIVI; j++) {
      for (unsigned k = 0; k < ZDIVI; k++) {
        if (f[i + j*XDIVI + k *XDIVI*YDIVI]!=0){
          outfile << f[i + j*XDIVI + k *XDIVI*YDIVI] << ","
          << position_matrix[i][j][k][0]<< "," << position_matrix[i][j][k][1] << "," << position_matrix[i][j][k][2] << ','
          << i << ',' << j << ',' << k << '\n';
        }
      }
    }
  }
  outfile.close();
}

void StoreF_CORE( float *f, unsigned IT, string output, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI,
                        float x_start, float y_start, float z_start, float delx, float dely, float delz ){
  ofstream outfile;
  string name = string(output) + to_string(IT) + string(".dat");
  outfile.open ( name );
  outfile.precision(7);

  outfile << XDIVI << ' ' << YDIVI << ' ' << ZDIVI << '\n';

  for (unsigned i = 0; i < XDIVI + 1; i++) {
    outfile << x_start + i*(delx) << ' ';
  }
  outfile << '\n';

  for (unsigned j = 0; j < YDIVI + 1; j++) {
    outfile << y_start + j*(dely) << ' ';
  }
  outfile << '\n';

  for (unsigned k = 0; k < ZDIVI + 1; k++) {
    outfile << z_start + k*(delz) << ' ';
  }
  outfile << '\n';

  for (unsigned k = 0; k < ZDIVI; k++) {
    for (unsigned j = 0; j < YDIVI; j++) {
      for (unsigned i = 0; i < XDIVI; i++) {
          outfile << f[i + j*XDIVI + k *XDIVI*YDIVI] << " ";
        }
      }
      outfile << '\n';
    }
  outfile.close();
}



















//
