#include "Includes.cuh"

#ifndef CPU_GENERAL_H
#define CPU_GENERAL_H

void InputSetup( string &NAME, string &OUTPUTMOD, float &x_start, float &x_end, float &y_start,
                float &y_end, float &z_start, float &z_end, unsigned &XDIVI, unsigned &YDIVI,
                unsigned &ZDIVI, unsigned &ITHREADSPB, unsigned &IBLOCKS, long unsigned &CONES,
                unsigned &CTHREADSPB, unsigned &CBLOCKS, unsigned &TOTALIT, unsigned &SAVEEVERY,
                unsigned &INTSTEP, float &CUTOFF, float &UE, bool &CORE_OUT, bool &MATT_OUT );

int MemDebugger( unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, long unsigned CONES );

int InputDebugger( unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI, unsigned ITHREADSPB, unsigned IBLOCKS,
                   long unsigned CONES, unsigned CTHREADSPB, unsigned CBLOCKS, unsigned INTSTEP, bool CORE_OUT, bool MATT_OUT );

void Print_Time_Remaining( float clock_start, float clock_end, unsigned It, unsigned TOTALIT );
void Print_Time_Complete( float clock_start, float clock_end, bool fin = 0 );

vector<float> ScalVec(float c, vector<float> x);
vector<float> unit_vector(vector<float> start, vector<float> stop);

void DefinePositions( vector<vector<vector<vector<float> > > > &position_matrix, float* f, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI,
                      float x_start, float y_start, float z_start, float delx, float dely, float delz);

float UncertaintyDoubleScatter( float E1, float E2, float UE );
float ScatteringAngle( float E1, float E2 );
float KleinNishina( float E1, float E2 );
void CreateCones( float* conelist_1D , string input, long unsigned &CONES, float UE );

void StoreF_MATT( float *f, unsigned It, string output, vector<vector<vector<vector<float> > > > position_matrix, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI );
void StoreF_CORE( float *f, unsigned It, string output, unsigned XDIVI, unsigned YDIVI, unsigned ZDIVI,
                  float x_start, float y_start, float z_start, float delx, float dely, float delz );

#endif
