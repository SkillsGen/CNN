//    This is an example program for the use of cnn.h = public domain
//    Author: Sebastian Reinolds
//
//    ===================================================================
//
//    The dataset to be used with this example program can be found here:
//    https://archive.ics.uci.edu/dataset/52/ionosphere


#include "stdio.h"
#include "math.h"
#include "time.h"

#define CNN_IMPLEMENTATION 1
#define CNN_FILE_IO 1
#include "nn.h"

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

bool
IsPartOfNumber(char *A)
{
    bool Result = false;
    if(*A == '.' ||
       *A == '-' ||
       (*A >= '0' && *A <= '9'))
    {
	Result = true;
    }
       
    return(Result);
}

cnn_matrix
ReadCSVFile(cnn_arena *Arena, char *Filename)
{
    cnn_matrix Result = {0};
    
    FILE *File = fopen(Filename, "r");
    cnn_u64 FileSize = 0;
    if(File)
    {
	fseek(File, 0, SEEK_END);
	FileSize = ftell(File);	
	fseek(File, 0, SEEK_SET);
    }
    else
    {
	CNN_print("Cannot open file %s\n", Filename);

	return(Result);
    }

    char *Data = (char *)CNN_malloc(FileSize);    
    fread(Data, FileSize, 1, File);

    char *Char = Data;
    char *EndOfData = Char + FileSize;
    cnn_u64 RowCount = 0;    
    while(Char != EndOfData)
    {
	if(*Char == '\n' || *Char == '\r')
	{
	    RowCount++;
	}
	Char++;
    }

    Char = Data;
    cnn_u64 ColCount = 1;
    while(*Char != '\n' && *Char != '\r')
    {
	if(*Char == ',')
	{
	    ColCount++;
	}
	else if(!IsPartOfNumber(Char))
	{
	    //NOTE: Non Float column
	}
	Char++;
    }

    CNN_print("File: %s. Rows: %llu, Cols: %llu\n", Filename, RowCount, ColCount);


    cnn_newmatrix(0, &Result, RowCount, ColCount);
    
    Char = Data;
    for(cnn_u64 RowIndex = 0;
	RowIndex < RowCount;
	RowIndex++)
    {
	for(cnn_u64 ColIndex = 0;
	    ColIndex < (ColCount - 1);
	    ColIndex++)
	{
	    CNN_assert(IsPartOfNumber(Char)); //NOTE: non numerical categorical not supported except last column
	    
	    char *NextChar = 0;
	    float ReadValue = strtod(Char, &NextChar);
	    CNN_assert(*NextChar == ',');
	    Char = NextChar + 1;

	    CNN_MATRIX_VAL(&Result, RowIndex, ColIndex) = ReadValue;
//	    CNN_print("Value %llu: %f\n", ColIndex, ReadValue);
	}

	cnn_u64 ColIndex = ColCount - 1;	
	if(!IsPartOfNumber(Char))
	{
	    //NOTE: should i auto categorize/onehot these columns after detecting them? i will decide
	    //      after ingesting more datasets.
	    
	    float ReadValue = 0.0f;
	    if(*Char == 'b')
	    {
		ReadValue = 0.0f;
	    }
	    else if(*Char == 'g')		
	    {
		ReadValue = 1.0f;
	    }
	    else
	    {
		CNN_invalid_code_path;
	    }
	    CNN_MATRIX_VAL(&Result, RowIndex, ColIndex) = ReadValue;
	    
	    Char++;
	    CNN_assert(*Char == '\n' || *Char == '\r');
	    Char++;
//	    CNN_print("Value %llu: %f\n", ColIndex, ReadValue);
	}
	else
	{
	    char *NextChar = 0;
	    float ReadValue = strtod(Char, &NextChar);
	    CNN_assert(*Char == '\n' && *Char == '\r');
	    Char = NextChar + 1;

	    CNN_MATRIX_VAL(&Result, RowIndex, ColIndex) = ReadValue;
//	    CNN_print("Value %llu: %f\n", ColCount, ReadValue);
	}
    }

    CNN_free(Data);

//    cnn_printmat(&Result);
    return(Result);
}

int
main(void)
{
    //NOTE: Random values different every run
    srand(time(0));
//    srand(42);

    //NOTE: Loading Data
    char *DataFilename = "ionosphere.data"; //NOTE: 352 rows, 34 input cols, 1 output col
    cnn_matrix DataMatrix = ReadCSVFile(0, DataFilename);


    //NOTE: Randomizing data
    cnn_randomize_dataset(&DataMatrix);

    cnn_u64 OutputCols = 1;
    cnn_matrix TrainingIn = {0};
    cnn_matrix TrainingOut = {0};
    cnn_matrix TestIn = {0};
    cnn_matrix TestOut = {0};
    cnn_splitdataset(&DataMatrix, OutputCols, 0.33f, &TestIn, &TestOut, &TrainingIn, &TrainingOut);

    //NOTE: Defining NN
    cnn_u64 Layers[][2] =
    {
	{16, CNN_ACT_RELU},
	{16, CNN_ACT_RELU},
	{TrainingOut.Cols, CNN_ACT_SIGMOID},
    };

    cnn_nn_info NN = cnn_define_nn(TrainingIn.Cols, ArrayCount(Layers), Layers);;

    
    //NOTE: Test run with full telemetry    
    cnn_u64 EpochCount = 1;
    float LearningRate = 0.1f;
    cnn_u64 BatchSize = 1;

    cnn_train(&NN, &TrainingIn, &TrainingOut, BatchSize, EpochCount, LearningRate,
	      	      TELEMETRYFLAG_ON|TELEMETRYFLAG_VALIDATEGRADIENTS|TELEMETRYFLAG_STOREGRADERRORS);

    cnn_print_telemetry_frames(&NN);
    cnn_save_telemetry_csv(&NN);    


    //NOTE: Full Run
    EpochCount = 200;
    BatchSize  = 2;
    cnn_train(&NN, &TrainingIn, &TrainingOut, BatchSize, EpochCount, LearningRate, 0);

    
    cnn_u64 Correct = 0;
    for(cnn_s64 TestIndex = 0;
	TestIndex < TestIn.Rows;
	TestIndex++)
    {
	cnn_matrix Input = cnn_submatrix(&TestIn, 1, 0, &CNN_MATRIX_VAL(&TestIn, TestIndex, 0));
	cnn_forward(NN.NN, &Input);

	float GoodOutput = CNN_MATRIX_VAL(&TestOut, TestIndex, 0);
	float NNOutput   = CNN_NN_OUTPUT(NN.NN).E[0];

	CNN_print("NN: %f Correct: %f\n", NNOutput, GoodOutput);
	float Difference = fabsf(GoodOutput - NNOutput);
	if(Difference < 0.5f) Correct++;
    }

    CNN_print("Correct: %llu/%llu %.2f\n", Correct, TestIn.Rows, (float)Correct/(float)TestIn.Rows);

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
