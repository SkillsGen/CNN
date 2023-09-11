//    This is an example program for the use of cnn.h = public domain
//    Author: Sebastian Reinolds
//
//    ===================================================================
//
//    The dataset to be used with this example program can be found here:
//    http://yann.lecun.com/exdb/mnist/
//
//    TELEMETRYFLAG_VALIDATEGRADIENTS will take a very long time for a
//    model this size


#include "stdio.h"
#include "math.h"
#include "time.h"

#define CNN_IMPLEMENTATION 1
#define CNN_FILE_IO 1
#include "nn.h"

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

cnn_u32
EndianFlip(cnn_u32 *Value)
{
    cnn_u8 *Data = (cnn_u8 *)Value;
    cnn_u32 Result = (Data[0]<<24) + (Data[1]<<16) + (Data[2]<<8) + Data[3];
    return(Result);
}

typedef struct MNIST_data_file
{
    cnn_u32 MagicNumber_BigEndian;
    cnn_u32 ImageCount_BigEndian;
    cnn_u32 RowCount_BigEndian;
    cnn_u32 ColumnCount_BigEndian;
} MNIST_data_file;

typedef struct MNIST_label_file
{
    cnn_u32 MagicNumber_BigEndian;
    cnn_u32 LabelCount_BigEndian;
} MNIST_label_file;

void
ReadMNIST(cnn_arena *Arena, cnn_u64 MaxCount, cnn_matrix *Inputs, cnn_matrix *Outputs)
{
    FILE *DataFile = fopen("train-images.idx3-ubyte", "r");
    cnn_u64 DataFileSize = 0;
    if(DataFile)
    {
	fseek(DataFile, 0, SEEK_END);
	DataFileSize = ftell(DataFile);	
	fseek(DataFile, 0, SEEK_SET);
    }
    else
    {
	CNN_print("Couldn't open file\n");
	CNN_invalid_code_path;
    }

    char *Data = (char *)CNN_malloc(DataFileSize);    
    fread(Data, DataFileSize, 1, DataFile);

    MNIST_data_file *MNISTDataFile = (MNIST_data_file *)Data;

    cnn_u32 DataMagicNumber = EndianFlip(&MNISTDataFile->MagicNumber_BigEndian);
    CNN_assert(DataMagicNumber == 2051);
    cnn_u32 ImageCount  = EndianFlip(&MNISTDataFile->ImageCount_BigEndian);
    cnn_u32 RowCount    = EndianFlip(&MNISTDataFile->RowCount_BigEndian);
    cnn_u32 ColumnCount = EndianFlip(&MNISTDataFile->ColumnCount_BigEndian);
    cnn_u8 *FirstImagePixels = (cnn_u8 *)(MNISTDataFile + 1);

    
    if((MaxCount != 0) && (ImageCount > MaxCount))
    {
	ImageCount = MaxCount;
    }
    
    FILE *LabelFile = fopen("train-labels.idx1-ubyte", "r");
    cnn_u64 LabelFileSize = 0;
    if(LabelFile)
    {
	fseek(LabelFile, 0, SEEK_END);
	LabelFileSize = ftell(LabelFile);	
	fseek(LabelFile, 0, SEEK_SET);
    }

    char *Labels = (char *)CNN_malloc(LabelFileSize);    
    fread(Labels, LabelFileSize, 1, LabelFile);

    MNIST_label_file *MNISTLabelFile = (MNIST_label_file *)Labels;
    cnn_u32 LabelMagicNumber = EndianFlip(&MNISTLabelFile->MagicNumber_BigEndian);
    CNN_assert(LabelMagicNumber == 2049);
    cnn_u32 LabelCount       = EndianFlip(&MNISTLabelFile->LabelCount_BigEndian);
    cnn_u8 *FirstImageLabel  = (cnn_u8 *)(MNISTLabelFile + 1);
    cnn_u32 PixelCount = RowCount*ColumnCount;
    
    if((MaxCount != 0) && (LabelCount > MaxCount))
    {
	LabelCount = MaxCount;
    }
    CNN_assert(ImageCount == LabelCount);

    cnn_newmatrix(Arena, Inputs,  ImageCount, PixelCount);
    cnn_newmatrix(Arena, Outputs, ImageCount, 1);

    for(cnn_u32 ImageIndex = 0;
	ImageIndex < ImageCount;
	ImageIndex++)
    {
	cnn_u8 *FirstPixel = FirstImagePixels + (PixelCount*ImageIndex); 
	for(cnn_u32 PixelIndex = 0;
	    PixelIndex < PixelCount;
	    PixelIndex++)
	{
	    cnn_u8 PixelValue = FirstPixel[PixelIndex];
	    float Value = (float)PixelValue/255.0f;
	    CNN_MATRIX_VAL(Inputs, ImageIndex, PixelIndex) = Value;
	}

	cnn_u8 LabelValue = FirstImageLabel[ImageIndex];
	CNN_MATRIX_VAL(Outputs, ImageIndex, 0) = (float)LabelValue;
    }
    
    CNN_free(Data);
    CNN_free(Labels);
}


int
main(void)
{
    //NOTE: Random values different every run
    srand(time(0));
//    srand(42);


    //NOTE: Loading Data
    cnn_matrix MNISTInputs = {0};
    cnn_matrix MNISTOutputs = {0};
    ReadMNIST(0, 0, &MNISTInputs, &MNISTOutputs);

    
    //NOTE: Converting to one hot encoding
    //      https://en.wikipedia.org/wiki/One-hot
    cnn_matrix OneHot = {0};
    cnn_int_to_one_hot(0, &MNISTOutputs, 10, &OneHot);

    //NOTE: Randomizing data
    cnn_randomize_dataset_in_out(&MNISTInputs, &OneHot);

    //NOTE: Small number of samples for telemetry
    cnn_u64 SamplesToTrain = 10;
    float Split = 0.66f;
    cnn_s64 TrainingRows = (cnn_s64)(Split*SamplesToTrain);			
    cnn_matrix TrainingIn  = cnn_submatrix(&MNISTInputs, TrainingRows, MNISTInputs.Cols, MNISTInputs.E);
    cnn_matrix TrainingOut = cnn_submatrix(&OneHot, TrainingRows, OneHot.Cols, OneHot.E);
    
    cnn_s64 TestRows = SamplesToTrain - TrainingRows;
    cnn_matrix TestIn  = cnn_submatrix(&MNISTInputs, TestRows, MNISTInputs.Cols, &CNN_MATRIX_VAL(&MNISTInputs, TrainingRows, 0));
    cnn_matrix TestOut = cnn_submatrix(&OneHot, TestRows, OneHot.Cols, &CNN_MATRIX_VAL(&OneHot, TrainingRows, 0));

    //NOTE: Defining NN
    cnn_u64 Layers[][2] =
    {
	{32, CNN_ACT_RELU},
	{32, CNN_ACT_RELU},
	{TrainingOut.Cols, CNN_ACT_SIGMOID},
    };

    cnn_nn_info NN = cnn_define_nn(TrainingIn.Cols, ArrayCount(Layers), Layers);;

    cnn_u64 EpochCount   = 1;
    float   LearningRate = 0.1f;
    cnn_u64 BatchSize    = 1;

    //NOTE: Test run with full telemetry
    cnn_train(&NN, &TrainingIn, &TrainingOut, BatchSize, EpochCount, LearningRate,
	      TELEMETRYFLAG_ON|TELEMETRYFLAG_VALIDATEGRADIENTS);
    cnn_print_telemetry_frames(&NN);
    cnn_save_telemetry_csv(&NN);


    
    //NOTE: More Samples for full run
    SamplesToTrain = 8000;
    TrainingRows = (cnn_s64)(Split*SamplesToTrain);			
    TrainingIn  = cnn_submatrix(&MNISTInputs, TrainingRows, MNISTInputs.Cols, MNISTInputs.E);
    TrainingOut = cnn_submatrix(&OneHot, TrainingRows, OneHot.Cols, OneHot.E);
    
    TestRows = SamplesToTrain - TrainingRows;
    TestIn  = cnn_submatrix(&MNISTInputs, TestRows, MNISTInputs.Cols, &CNN_MATRIX_VAL(&MNISTInputs, TrainingRows, 0));
    TestOut = cnn_submatrix(&OneHot, TestRows, OneHot.Cols, &CNN_MATRIX_VAL(&OneHot, TrainingRows, 0));

    
    //NOTE: Full Run
    EpochCount = 20;
    BatchSize  = 2;
    cnn_train(&NN, &TrainingIn, &TrainingOut, BatchSize, EpochCount, LearningRate, 0);
        
    cnn_u64 Correct = 0;
    for(cnn_s64 TestIndex = 0;
	TestIndex < TestIn.Rows;
	TestIndex++)
    {
	cnn_matrix Input = cnn_submatrix(&TestIn, 1, 0, &CNN_MATRIX_VAL(&TestIn, TestIndex, 0));
	cnn_forward(NN.NN, &Input);

	cnn_matrix NNOutput   = CNN_NN_OUTPUT(NN.NN);
	cnn_matrix GoodOutput = cnn_matrixrow(&TestOut, TestIndex);

	cnn_s64 NNCategory = -1;
	cnn_s64 CorrectCategory = -1;
	float   HighestScore = 0.0f;
	
	for(cnn_s64 Category = 0;
	    Category < NNOutput.Cols;
	    Category++)
	{
	    float Value = CNN_MATRIX_VAL(&NNOutput, 0, Category);

	    if(Value > HighestScore)
	    {
		HighestScore = Value;
		NNCategory = Category;
	    }

	    if(CNN_MATRIX_VAL(&GoodOutput, 0, Category) == 1.0f) CorrectCategory = Category;
	}
	
	CNN_print("NN: %lld Correct: %lld\n", NNCategory, CorrectCategory);
	if((NNCategory != -1) && NNCategory == CorrectCategory)
	{
	    Correct++;
	}	
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
