//    This is an example program for the use of cnn.h = public domain
//    Author: Sebastian Reinolds
//
//    ===================================================================
//
//    This is a simple example program for training cnn.h on the three basic logic gates



#include "stdio.h"
#include "math.h"
#include "time.h"

#define CNN_IMPLEMENTATION 1
#define CNN_FILE_IO 1
#include "nn.h"

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

float _OrGate[][3] = 
{
    { 0, 0,  0},
    { 1, 0,  1},
    { 0, 1,  1},
    { 1, 1,  1},
};

float _AndGate[][3] = 
{
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

float _XorGate[][3] = 
{
    {0,  0, 0},
    {1,  0, 1},
    {0,  1, 1},
    {1,  1, 0},
};

int
main(void)
{
    //NOTE: Random values different every run
    srand(time(0));

    
    //NOTE: Loading Data
    cnn_matrix DataMatrix;
    DataMatrix.Rows   = 4;
    DataMatrix.Cols   = 3;
    DataMatrix.Stride = 3;
    DataMatrix.E = (float *)_XorGate;

    
    //NOTE: Randomizing data
    cnn_randomize_dataset(&DataMatrix);
    
    cnn_matrix In  = cnn_submatrix(&DataMatrix, 4, 2, DataMatrix.E);
    cnn_printmat(&In);
    cnn_matrix Out = cnn_submatrix(&DataMatrix, 4, 1, &CNN_MATRIX_VAL(&DataMatrix, 0, 2));
    cnn_printmat(&Out);

    
    //NOTE: Defining NN
    cnn_u64 Layers[][2] =
    {
	{2, CNN_ACT_SIGMOID},
	{1, CNN_ACT_SIGMOID},
    };

    cnn_nn_info NN = cnn_define_nn(In.Cols, ArrayCount(Layers), Layers);;

    cnn_u64 EpochCount   = 2;
    float   LearningRate = 1.0f;
    cnn_u64 BatchSize    = DataMatrix.Rows;

    
    //NOTE: Test run with full telemetry
    cnn_train(&NN, &In, &Out, BatchSize, EpochCount, LearningRate,
	      TELEMETRYFLAG_ON|TELEMETRYFLAG_VALIDATEGRADIENTS|TELEMETRYFLAG_STOREGRADERRORS);
    cnn_print_telemetry_frames(&NN);


    //NOTE: Full run
    EpochCount = 20000;
    cnn_train(&NN, &In, &Out, BatchSize, EpochCount, LearningRate, 0);
    
    for(cnn_s64 Row = 0;
	Row < DataMatrix.Rows;
	Row++)
    {
	cnn_matrix InputRow = cnn_matrixrow(&In, Row);
	cnn_forward(NN.NN, &InputRow);
	float NNOutput = CNN_NN_OUTPUT(NN.NN).E[0];

	CNN_print("%f, %f : %f\n",
	      CNN_MATRIX_VAL(&InputRow, 0, 0),
	      CNN_MATRIX_VAL(&InputRow, 0, 1),
	      NNOutput);
    }


    //NOTE: Saving weights
    cnn_save_weights(NN.NN);

    
#if 0
    //NOTE: Loading weights if we want
    cnn_neural_network *LoadedNN = cnn_alloc_nn_from_nn(0, NN.NN);    
    cnn_init_nn_zero(LoadedNN);
    cnn_load_weights(LoadedNN, "weights.cnn");
#endif
    return(1);
}
