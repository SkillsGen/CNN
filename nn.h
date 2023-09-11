//    cnn.h public domain
//    Author: Sebastian Reinolds
//
//    ===================================================================
//
//    This is a basic neural network library running on the cpu. It mostly
//    exists as a learning exercise and perhaps an example for anyone who
//    wants to see a very straightforward implementation of something that
//    is often an opaque if powerful library (or stack of libraries).
//
//
//    Defining the neural network will change as more features are added
//    (recurrance, convolution, etc) but the basic principle is to define
//    the structure of the NN as follows:
//
//    {NEURONS, ACTIVATIONFUNCTION},
//    {OUTPUTNEURONS, ACTIVATIONFUNCTION},
//
//    For example:
//
//        cnn_u64 Layers[][2] =
//        {
//	      {32, CNN_ACT_RELU},
//	      {32, CNN_ACT_RELU},
//	      {Outputs, CNN_ACT_SIGMOID},
//        };
//
//    Activation functions supported:
//        CNN_ACT_RELU,
//        CNN_ACT_SIGMOID,
//        CNN_ACT_TANH,
//
//    Loss functions supported:
//        CNN_LOSS_MEANSQ,
//        //TODO: CNN_LOSS_CROSSENTROPY,
//
//
//    Weights are initialized as follows:
//         RELU:           HE initialization
//         TANH & SIGMOID: Normalized Xavier
//         //NOTE: this can be customized in cnn_init_nn_rand, it's interesting how
//         //      fragile it can be with poor weight initialization
//
//
//    Telemetry:
//         By training with the TELEMETRYFLAG_ON, you will store the loss, gradients that
//         have gone to zero and gradients that have exploded for each batch trained.
//         The flag TELEMETRYFLAG_VALIDATEGRADIENTS will calculate the relative error of
//         each gradient by calculating the finite difference of that weight and performing
//         the following:
//               AbsoluteValue(gradient - finite_Difference)/Max(gradient,finite_Difference)
//         Each batch's telemetry frame with then store the Largest error and it's position
//         in the network as well as the average error.
//         The flag TELEMETRYFLAG_STOREGRADERRORS will store an entire copy of the NN with
//         the calculated relative errors for each weight within each telemetry frame. This
//         can be accessed by the same way as the values in any NN.
//
//         Because the above can take up such a huge amount of space and processing power,
//         it's intended use is to train the full NN on a small portion of your dataset for
//         very few batches. You can validate the structure of your NN with this telemetry and
//         then continue.



#ifdef CNN_IMPLEMENTATION
#include <stdbool.h>
typedef unsigned char   cnn_u8;
typedef signed   char   cnn_s8;
typedef unsigned short  cnn_u16;
typedef signed   short  cnn_s16;
typedef unsigned int    cnn_u32;
typedef signed   int    cnn_s32;

#ifdef _MSC_VER
  typedef unsigned __int64 cnn_u64;
  typedef          __int64 cnn_s64;
#else
  typedef unsigned long long cnn_u64;
  typedef          long long cnn_s64;
#endif

#ifdef CNN_STATIC
#define CNN_DEF static
#else
#define CNN_DEF extern
#endif

#ifndef CNN_malloc
#include <stdlib.h>
#define CNN_malloc(sz)   malloc(sz)
#define CNN_free(ptr)    free(ptr)
#endif

#ifndef CNN_assert
#include <assert.h>
#define CNN_assert(x)    assert(x)
#endif

#ifndef CNN_memcpy
   #include <string.h>
   #define CNN_memcpy       memcpy
   #define CNN_memset       memset
#endif

#ifndef CNN_tanh
#include <math.h>
#define CNN_tanh(x)             tanhf(x)
#define CNN_exp(x)              expf(x)
#define CNN_pow(x, y)           powf(x, y)
#define CNN_log(x)              logf(x)
#define CNN_isnan(x)            isnan(x)
#define CNN_ceil(x)             ceilf(x)
#define CNN_fabs_double(x)      fabs(x)
CNN_DEF float
CNN_sigmoid(float A)
{
    float Result = 1.0f/(1.0f + CNN_exp(-1.0f*A));
    return(Result);
}
#endif

#ifndef CNN_rand
#include <stdlib.h>
CNN_DEF float
CNN_rand(void) //NOTE: 0 - 1
{
    float Result = (float)rand()/(float)RAND_MAX;
    return(Result);
}
CNN_DEF cnn_u64
CNN_rand_u64(void)
{
    cnn_u32 Result[2];
    Result[0] = rand();
    Result[1] = rand();
    
    return(*((cnn_u64 *)Result));
}
#endif


#define CNN_invalid_code_path CNN_assert(!"InvalidCodePath")
#define CNN_invalid_default_case default: {CNN_invalid_code_path;} break

#ifndef CNN_print
#include <stdio.h>
#define CNN_print   printf
#define CNN_snprint snprintf
#endif

typedef struct cnn_arena
{
    cnn_u64 Size;
    cnn_u64 Used;
    cnn_u8 *Base;
} cnn_arena;

CNN_DEF cnn_arena
cnn_new_arena(cnn_u64 Size)
{
    cnn_arena Result;
    Result.Size = Size;
    Result.Used = 0;
    Result.Base = (cnn_u8 *)CNN_malloc(Size);
    
    return(Result);
}

#define CNN_push_array(Arena, Count, Type) (Type *)CNN_push_size(Arena, Count*sizeof(Type), 0)

CNN_DEF void *
CNN_push_size(cnn_arena *Arena, cnn_u64 Size, cnn_u64 RequiredAlignment)
{
    void *Result = 0;
    
    cnn_u64 Alignment = 0;
    if(RequiredAlignment != 0)
    {
	Alignment = RequiredAlignment - ((cnn_u64)(Arena->Base + Arena->Used) % RequiredAlignment);
    }
	
    CNN_assert((Arena->Used + Size + Alignment) <= Arena->Size);

    Result = (void *)(Arena->Base + Arena->Used + Alignment);
    Arena->Used += (Size + Alignment);

    return(Result);
}

#define CNN_zero_struct(Instance) CNN_zero_size(sizeof(Instance), &(Instance))
#define CNN_zero_array(Count, Pointer) CNN_zero_size(Count*sizeof((Pointer)[0]), Pointer)
CNN_DEF void
CNN_zero_size(cnn_u64 Size, void *Ptr)
{
    CNN_memset(Ptr, 0, Size);
}

CNN_DEF float
CNN_rand_bilateral()
{
    float Result = CNN_rand();
    Result *= 2.0f;
    Result -= 1.0f;
    
    return(Result);
}

CNN_DEF float
cnn_boxmuller(float Mean, float StdDev)
{
    float U = 0.0f;
    while(U == 0.0f) U = CNN_rand();

    float Theta = 0.0f;
    while(Theta == 0.0f) Theta = 2.0f * 3.14159265358979323846f * CNN_rand();

    float R = sqrtf(-2.0f * CNN_log(U));
    float X = R * cosf(Theta);

    float Result = (X * StdDev) + Mean;

    return(Result);
}

typedef struct cnn_matrix
{
    cnn_u64 Rows;
    cnn_u64 Cols;
    cnn_u64 Stride;
    float *E;
} cnn_matrix;

#define CNN_MATRIX_VAL(Matrix, Row, Col) (Matrix)->E[(Row)*(Matrix)->Stride + (Col)]

CNN_DEF void
cnn_printmat(cnn_matrix *A)
{
    CNN_print("{\n");
    for(cnn_s64 RowIndex = 0;
	RowIndex < A->Rows;
	RowIndex++)
    {
	CNN_print("    ");
	for(cnn_s64 ColIndex = 0;
	    ColIndex < A->Cols;
	    ColIndex++)
	{
	    CNN_print("%f,    ",CNN_MATRIX_VAL(A, RowIndex, ColIndex));
	}
	CNN_print("\n");
    }
    CNN_print("}\n");
}

CNN_DEF void
cnn_newmatrix(cnn_arena *Arena, cnn_matrix *Matrix, cnn_u64 Rows, cnn_u64 Cols)
{
    CNN_assert(!Matrix->E);
    
    Matrix->Rows   = Rows;
    Matrix->Cols   = Cols;
    Matrix->Stride = Cols;
    if(Arena)
    {
	Matrix->E = CNN_push_array(Arena, Rows*Cols, float);
    }
    else
    {
	cnn_u64 Size = sizeof(float)*Rows*Cols;
	Matrix->E = (float *)CNN_malloc(Size);
    }
//    ZeroArray(Rows*Cols, Matrix->E);
}

CNN_DEF void
cnn_zeromatrix(cnn_matrix *A)
{
    for(cnn_s64 RowIndex = 0;
	RowIndex < A->Rows;
	RowIndex++)
    {
	for(cnn_s64 ColIndex = 0;
	    ColIndex < A->Cols;
	    ColIndex++)
	{
	    CNN_MATRIX_VAL(A, RowIndex, ColIndex) = 0.0f;
	}
    }
}

typedef enum cnn_weight_init
{
    CNN_WEIGHT_INIT_DEFAULT, //TODO: implement

    CNN_WEIGHT_INIT_ZEROS,    
    CNN_WEIGHT_INIT_UNIFORM,
    CNN_WEIGHT_INIT_XAVIER,
    CNN_WEIGHT_INIT_NORMXAVIER,
    CNN_WEIGHT_INIT_HE,
} cnn_weight_init;

CNN_DEF void
cnn_randomize_matrix(cnn_matrix *A, cnn_weight_init WeightInit)
{
    for(cnn_s64 RowIndex = 0;
	RowIndex < A->Rows;
	RowIndex++)
    {
	for(cnn_s64 ColIndex = 0;
	    ColIndex < A->Cols;
	    ColIndex++)
	{
	    float Randomized = 0.0f;
	    switch(WeightInit)
	    {
		case CNN_WEIGHT_INIT_ZEROS:
		{
		    Randomized = 0.0f;
		} break;
		case CNN_WEIGHT_INIT_UNIFORM:
		{
		    Randomized = CNN_rand_bilateral();
		} break;
		case CNN_WEIGHT_INIT_XAVIER:
		{
		    float Value = CNN_rand();
		    float Half = 1.0f/sqrt(A->Rows);
		    Value *= 2*Half;
		    Value -= Half;
		    Randomized = Value;
		} break;
		case CNN_WEIGHT_INIT_NORMXAVIER:
		{
		    float Value = CNN_rand();
		    float Half = sqrt(6)/sqrt(A->Rows + A->Cols);
		    Value *= 2*Half;
		    Value -= Half;
		    Randomized = Value;
		} break;
		case CNN_WEIGHT_INIT_HE:
		{
		    Randomized = cnn_boxmuller(0.0f, sqrtf(2.0f/A->Rows));
		} break;

		CNN_invalid_default_case;
	    }
	    
	    CNN_MATRIX_VAL(A, RowIndex, ColIndex) = Randomized;
	}
    }
}

CNN_DEF cnn_matrix
cnn_submatrix(cnn_matrix *Matrix, cnn_u64 Rows, cnn_u64 Cols, float *FirstValue)
{
    CNN_assert(Rows <= Matrix->Rows);
    CNN_assert(Cols <= Matrix->Cols);

    if(Rows == 0) Rows = Matrix->Rows;
    if(Cols == 0) Cols = Matrix->Cols;
    
    cnn_matrix Result;
    Result.Rows   = Rows;
    Result.Cols   = Cols;
    Result.Stride = Matrix->Stride;
    Result.E      = FirstValue;
    
    return(Result);
}

CNN_DEF cnn_matrix
cnn_matrixrow(cnn_matrix *Matrix, cnn_u64 Row)
{
    CNN_assert(Row < Matrix->Rows);

    cnn_matrix Result;
    Result.Rows   = 1;
    Result.Cols   = Matrix->Cols;
    Result.Stride = Matrix->Stride;
    Result.E      = &CNN_MATRIX_VAL(Matrix, Row, 0);
	
    return(Result);
}

CNN_DEF void
cnn_randomize_dataset(cnn_matrix *Data)
{
    cnn_u64 RowSize = Data->Cols*sizeof(*Data->E);
    void *CopySpace = CNN_malloc(RowSize);

    for(cnn_s64 RowIndex = (Data->Rows - 1);
	RowIndex >= 0;
	RowIndex--)
    {
	float *CurrentRow = &CNN_MATRIX_VAL(Data, RowIndex, 0);
	cnn_u64 RowIndexToSwap = CNN_rand_u64() % (RowIndex + 1);
	float *RowToSwap = &CNN_MATRIX_VAL(Data, RowIndexToSwap, 0);
	
	if(RowIndex != RowIndexToSwap)
	{
	    CNN_memcpy(CopySpace,  RowToSwap,  RowSize);
	    CNN_memcpy(RowToSwap,  CurrentRow, RowSize);
	    CNN_memcpy(CurrentRow, CopySpace,  RowSize);
	}
    }
    
    CNN_free(CopySpace);
}

CNN_DEF void
cnn_randomize_dataset_in_out(cnn_matrix *Input, cnn_matrix *Output)
{
    CNN_assert(Input->Rows == Output->Rows);
    
    cnn_u64 InputRowSize = Input->Cols*sizeof(*Input->E);
    void *InputCopySpace = CNN_malloc(InputRowSize);

    cnn_u64 OutputRowSize = Output->Cols*sizeof(*Output->E);
    void *OutputCopySpace = CNN_malloc(OutputRowSize);

    for(cnn_s64 RowIndex = (Input->Rows - 1);
	RowIndex >= 0;
	RowIndex--)
    {
	cnn_u64 RowIndexToSwap = CNN_rand_u64() % (RowIndex + 1);

	float *InputCurrentRow = &CNN_MATRIX_VAL(Input, RowIndex, 0);
	float *InputRowToSwap  = &CNN_MATRIX_VAL(Input, RowIndexToSwap, 0);

	float *OutputCurrentRow = &CNN_MATRIX_VAL(Output, RowIndex, 0);
	float *OutputRowToSwap  = &CNN_MATRIX_VAL(Output, RowIndexToSwap, 0);

	if(RowIndex != RowIndexToSwap)
	{
	    CNN_memcpy(InputCopySpace,  InputRowToSwap,  InputRowSize);
	    CNN_memcpy(InputRowToSwap,  InputCurrentRow, InputRowSize);
	    CNN_memcpy(InputCurrentRow, InputCopySpace,  InputRowSize);

	    CNN_memcpy(OutputCopySpace,  OutputRowToSwap,  OutputRowSize);
	    CNN_memcpy(OutputRowToSwap,  OutputCurrentRow, OutputRowSize);
	    CNN_memcpy(OutputCurrentRow, OutputCopySpace,  OutputRowSize);
	}
    }
    
    CNN_free(InputCopySpace);
    CNN_free(OutputCopySpace);
}

CNN_DEF void
cnn_splitdataset(cnn_matrix *Data, cnn_u64 OutputCols, float Split,
		 cnn_matrix *TrainingIn, cnn_matrix *TrainingOut,
		 cnn_matrix *TestIn, cnn_matrix *TestOut)
{
    CNN_assert(Split < 1.0f);
    
    cnn_u64 InputRows = (cnn_u64)(Split*(float)Data->Rows);
    cnn_u64 InputCols = Data->Cols - OutputCols;
    *TrainingIn  = cnn_submatrix(Data, InputRows, InputCols, Data->E);
    *TrainingOut = cnn_submatrix(Data, InputRows, OutputCols, &CNN_MATRIX_VAL(Data, 0, InputCols));

    cnn_u64 OutputRows = Data->Rows - InputRows;
    *TestIn  = cnn_submatrix(Data, OutputRows, InputCols, &CNN_MATRIX_VAL(Data, InputRows, 0));
    *TestOut = cnn_submatrix(Data, OutputRows, OutputCols, &CNN_MATRIX_VAL(Data, InputRows, InputCols));
}

CNN_DEF void
cnn_int_to_one_hot(cnn_arena *Arena, cnn_matrix *IntEncoded, cnn_s64 CategoryCount, cnn_matrix *CatEncoded)
{
    CNN_assert(IntEncoded->Cols == 1);
    CNN_assert(!CatEncoded->E);

    cnn_newmatrix(Arena, CatEncoded, IntEncoded->Rows, CategoryCount);
    cnn_zeromatrix(CatEncoded);
    
    for(cnn_s64 Row = 0;
	Row < IntEncoded->Rows;
	Row++)
    {
	cnn_s64 Category = (cnn_s64)CNN_MATRIX_VAL(IntEncoded, Row, 0);
	CNN_assert(Category < CategoryCount);

	CNN_MATRIX_VAL(CatEncoded, Row, Category) = 1.0f;
    }
}

CNN_DEF void
cnn_matrix_add(cnn_matrix *A, cnn_matrix *B)
{
    CNN_assert(A->Rows == B->Rows);
    CNN_assert(A->Cols == B->Cols);

    for(cnn_s64 RowIndex = 0;
	RowIndex < A->Rows;
	RowIndex++)
    {
	for(cnn_s64 ColIndex = 0;
	    ColIndex < A->Cols;
	    ColIndex++)
	{
	    CNN_MATRIX_VAL(B, RowIndex, ColIndex) += CNN_MATRIX_VAL(A, RowIndex, ColIndex);
	}
    }
}

CNN_DEF void
cnn_matrix_mul(cnn_matrix *A, cnn_matrix *B, cnn_matrix *Output)
{
    CNN_assert(A->Cols == B->Rows);
    CNN_assert(Output->Cols == B->Cols);
    CNN_assert(Output->Rows == A->Rows);

    for(cnn_u64 ARow = 0;
	ARow < A->Rows;
	ARow++)
    {
	for(cnn_u64 BCol = 0;
	    BCol < B->Cols;
	    BCol++)
	{
	    float Sum = 0;
	    for(cnn_u64 ACol = 0;
		ACol < A->Cols;
		ACol++)
	    {
		Sum += CNN_MATRIX_VAL(A, ARow, ACol)*CNN_MATRIX_VAL(B, ACol, BCol);
	    }
	    CNN_assert(!CNN_isnan(Sum));
	    CNN_MATRIX_VAL(Output, ARow, BCol) = Sum;
	}
    }
}

typedef enum cnn_loss_func
{
    CNN_LOSS_MEANSQ,
//    CNN_LOSS_CROSSENTROPY,
} cnn_loss_func;

typedef enum cnn_activation_func
{
    CNN_ACT_NONE,
    CNN_ACT_RELU,
    CNN_ACT_SIGMOID,
    CNN_ACT_TANH,

    CNN_ACT_COUNT,
} cnn_activation_func;

typedef struct cnn_layer
{
    cnn_activation_func Activation;
    cnn_matrix Weights;
    cnn_matrix Bias;
    cnn_matrix Outputs;
} cnn_layer;

typedef struct cnn_neural_network
{
    cnn_u64 LayerCount;
    cnn_layer Layers[0];
} cnn_neural_network;

enum
{
    TELEMETRYFLAG_NONE,
    
    TELEMETRYFLAG_ON                      = 0x1,
    TELEMETRYFLAG_VALIDATEGRADIENTS       = 0x2,
    TELEMETRYFLAG_STOREGRADERRORS         = 0x4,
};

typedef struct
{
    cnn_u64 Row;
    cnn_u64 Col;
    bool    IsBias;
} nn_weight_pos;

typedef struct cnn_telemetry_frame
{
    float   Loss;
    cnn_u64 ZeroGradientsCount;
    cnn_u64 ExplodedGradientsCount;

    double AverageError;
    double LargestError;
    nn_weight_pos LargestErrorPos;
    
    struct cnn_neural_network *RelativeGradientErrors;
} cnn_telemetry_frame;

typedef struct cnn_nn_info
{
    cnn_u64 EpochCount;
    cnn_u64 BatchCount;
    cnn_u64 TelemetryFrameCount;
    cnn_telemetry_frame *FirstTelemetryFrame;
    cnn_neural_network *NN;
    cnn_neural_network *G;
} cnn_nn_info;

CNN_DEF void
cnn_printnn(cnn_neural_network *NN)
{
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = NN->Layers + LayerIndex;
	
	CNN_print("Layer %lld, Activation: %u\n", LayerIndex, Layer->Activation);
	cnn_printmat(&Layer->Weights);
	cnn_printmat(&Layer->Bias);
    }
}

#define CNN_NN_OUTPUT(NN) (NN)->Layers[(NN)->LayerCount - 1].Outputs

CNN_DEF cnn_u64
cnn_get_grad_count(cnn_neural_network *NN) //NOTE: not including 
{
    cnn_u64 Result = 0;
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = NN->Layers + LayerIndex;
	Result += (Layer->Weights.Rows*Layer->Weights.Cols);
	Result += Layer->Bias.Rows;
    }
    return(Result);
}

CNN_DEF cnn_u64
cnn_get_nn_size(cnn_u64 InputCount, cnn_u64 LayerCount, cnn_u64 Layers[][2])
{
    cnn_u64 Result = sizeof(cnn_neural_network) + LayerCount*sizeof(cnn_layer);

    cnn_u64 LastLayerOutputs = InputCount;
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < LayerCount;
	LayerIndex++)
    {
	cnn_u64             LayerOutputs    = Layers[LayerIndex][0];	
	cnn_activation_func LayerActivation = (cnn_activation_func)Layers[LayerIndex][1];
	CNN_assert((LayerActivation != CNN_ACT_NONE) && (LayerActivation < CNN_ACT_COUNT));

	Result += (sizeof(float)*LastLayerOutputs*LayerOutputs); //NOTE: Weights
	Result += (sizeof(float)*1*LayerOutputs);                //NOTE: Bias
	Result += (sizeof(float)*1*LayerOutputs);                //NOTE: Outputs
	LastLayerOutputs = LayerOutputs;
    }

    return(Result);
}

CNN_DEF cnn_u64
cnn_get_nn_size_from_nn(cnn_neural_network *NN)
{
    cnn_u64 Result = sizeof(cnn_neural_network) + NN->LayerCount*sizeof(cnn_layer);

    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = NN->Layers + LayerIndex;
	Result += (sizeof(float)*Layer->Weights.Rows*Layer->Weights.Cols);
	Result += (sizeof(float)*Layer->Bias.Rows*Layer->Bias.Cols);
	Result += (sizeof(float)*Layer->Outputs.Rows*Layer->Outputs.Cols);
    }
    return(Result);
}

CNN_DEF cnn_neural_network *
cnn_alloc_nn_from_nn(cnn_arena *Arena, cnn_neural_network *NN)
{
    cnn_arena _Arena;
    if(!Arena)
    {
	cnn_u64 NNSize = cnn_get_nn_size_from_nn(NN);
	_Arena = cnn_new_arena(NNSize);
	Arena = &_Arena;
    }
    
    cnn_u64 HeaderSize = sizeof(cnn_neural_network) + NN->LayerCount*sizeof(cnn_layer);
    cnn_neural_network *Result = (cnn_neural_network *)CNN_push_size(Arena, HeaderSize, 0);    
    CNN_zero_size(HeaderSize, Result);
    Result->LayerCount = NN->LayerCount;
    
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *LayerToCopy = NN->Layers + LayerIndex;
	cnn_layer *NewLayer = Result->Layers + LayerIndex;
	NewLayer->Activation = LayerToCopy->Activation;
	
	cnn_newmatrix(Arena, &NewLayer->Weights, LayerToCopy->Weights.Rows, LayerToCopy->Weights.Cols);
	cnn_newmatrix(Arena, &NewLayer->Bias,    LayerToCopy->Bias.Rows,    LayerToCopy->Bias.Cols);
	cnn_newmatrix(Arena, &NewLayer->Outputs, LayerToCopy->Outputs.Rows, LayerToCopy->Outputs.Cols);	
    }

    return(Result);
}

CNN_DEF cnn_neural_network *
cnn_alloc_nn(cnn_u64 InputCount, cnn_u64 LayerCount, cnn_u64 Layers[][2])
{
    cnn_u64 ArenaSize = cnn_get_nn_size(InputCount, LayerCount, Layers);
    cnn_arena NNArena = cnn_new_arena(ArenaSize);
    
    cnn_u64 HeaderSize = sizeof(cnn_neural_network) + LayerCount*sizeof(cnn_layer);
    cnn_neural_network *Result = (cnn_neural_network *)CNN_push_size(&NNArena, HeaderSize, 0);
    CNN_zero_size(HeaderSize, Result);
    
    Result->LayerCount = LayerCount;
    
    cnn_u64 LastLayerOutputs = InputCount;
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = Result->Layers + LayerIndex;

	cnn_u64             LayerOutputs    = Layers[LayerIndex][0];	
	cnn_activation_func LayerActivation = (cnn_activation_func)Layers[LayerIndex][1];
	CNN_assert((LayerActivation != CNN_ACT_NONE) && (LayerActivation < CNN_ACT_COUNT));
	Layer->Activation = LayerActivation;
	
	cnn_newmatrix(&NNArena, &Layer->Weights, LastLayerOutputs, LayerOutputs);
	cnn_newmatrix(&NNArena, &Layer->Bias,                   1, LayerOutputs);
	cnn_newmatrix(&NNArena, &Layer->Outputs,                1, LayerOutputs);

	LastLayerOutputs = LayerOutputs;
    }
    
    return(Result);
}

CNN_DEF void
cnn_init_nn_rand(cnn_neural_network *NN)
{
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = NN->Layers + LayerIndex;

	cnn_weight_init Init;
	switch(Layer->Activation)
	{
	    case CNN_ACT_RELU:
	    {
		Init = CNN_WEIGHT_INIT_HE;
	    } break;
	    case CNN_ACT_TANH:
	    case CNN_ACT_SIGMOID:
	    {
		Init = CNN_WEIGHT_INIT_NORMXAVIER;
	    } break;
	    
	    CNN_invalid_default_case;
	}
	cnn_randomize_matrix(&Layer->Weights, Init);	
	cnn_randomize_matrix(&Layer->Bias, CNN_WEIGHT_INIT_UNIFORM);	
    }
}

CNN_DEF void
cnn_init_nn_zero(cnn_neural_network *NN)
{
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = NN->Layers + LayerIndex;	
	cnn_zeromatrix(&Layer->Weights);	
	cnn_zeromatrix(&Layer->Bias);	
	cnn_zeromatrix(&Layer->Outputs);	
    }
}

CNN_DEF cnn_nn_info
cnn_define_nn(cnn_u64 InputCount, cnn_u64 LayerCount, cnn_u64 Layers[][2])
{
    cnn_nn_info Result;
    Result.EpochCount = 0;
    Result.BatchCount = 0;
    Result.TelemetryFrameCount = 0;
    Result.FirstTelemetryFrame = 0;
    
    Result.NN = cnn_alloc_nn(InputCount, LayerCount, Layers);
    cnn_init_nn_rand(Result.NN);
    Result.G  = cnn_alloc_nn(InputCount, LayerCount, Layers);
    cnn_init_nn_zero(Result.G);
    
    return(Result);
}

#if CNN_FILE_IO

#pragma pack(push, 1)
char FILE_ID[]  = "CNNF";
char LAYER_ID[] = "CNNL";

typedef struct cnn_weights_file_layer
{
    cnn_u8 LayerID[4]; //NOTE:  "CNNL"
    cnn_u8 Activation;
    cnn_u64 OutputCount;
// float Weights[];
// float Bias[];    
} cnn_weights_file_layer;

#define FILE_LAYOUT_DESCRIPTION_BYTES 316
typedef struct cnn_weights_file_header
{
    cnn_u8 FileID[4]; //NOTE:  "CNNF"
    cnn_u8 FileLayoutDesc[FILE_LAYOUT_DESCRIPTION_BYTES];
    cnn_u64 Inputs;
    cnn_u64 LayerCount;
} cnn_weights_file_header;
#pragma pack(pop)

CNN_DEF void
cnn_save_weights(cnn_neural_network *NN)
{
    //NOTE: need stdio.h
    FILE *OutputFile;
    OutputFile = fopen("weights.cnn", "wb");
    if(!OutputFile)
    {
	CNN_print("Can't open file for writing\n");
	return;
    }

    char FileLayoutDesc[] = "FileID: CNNF\nLayerID: CNNL\nWeights File Layout:\nHeader:\nEXAMPLE_ENTRY(BYTES)\nFileID(4)\nFileLayoutDesc(316)\nInputs(8)\nLayerCount(8)\nLayerArray(x)Layer:\nLayerID(4)\nActivation(1)\nInputs(8)\nWeightMatrix(floats)\nBiasMatrix(floats)\n\nWeightMatrix:\nRowMajor\nRows:Inputs\nCols:Outputs\nBiasMatrix:\nRowMajor\nRows:1\nCols:Outputs";
    cnn_u64 FileLayoutDescBytes = sizeof(FileLayoutDesc) / sizeof(FileLayoutDesc[0]);
    CNN_assert(FileLayoutDescBytes == FILE_LAYOUT_DESCRIPTION_BYTES);

    cnn_weights_file_header Header;
    CNN_memcpy(&Header.FileID, FILE_ID, 4);
    CNN_memcpy(&Header.FileLayoutDesc, FileLayoutDesc, FILE_LAYOUT_DESCRIPTION_BYTES);
    Header.Inputs     = NN->Layers[0].Weights.Rows;
    Header.LayerCount = NN->LayerCount;
    Header.Inputs = 0;
    fwrite(&Header, sizeof(cnn_weights_file_header), 1, OutputFile);        

    cnn_u64 LastLayerOutputs = Header.Inputs;
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = NN->Layers + LayerIndex;

	cnn_weights_file_layer FileLayer;
	CNN_memcpy(&FileLayer.LayerID, LAYER_ID, 4);
	FileLayer.Activation  = Layer->Activation;
	FileLayer.OutputCount = Layer->Weights.Cols;
	
	fwrite(&FileLayer, sizeof(cnn_weights_file_layer), 1, OutputFile);

	for(cnn_u64 Row = 0;
	    Row < Layer->Weights.Rows;
	    Row++)
	{
	    for(cnn_u64 Col = 0;
		Col < Layer->Weights.Cols;
		Col++)
	    {
		float Value = CNN_MATRIX_VAL(&Layer->Weights, Row, Col);
		fwrite(&Value, sizeof(Value), 1, OutputFile);
	    }
	}
	for(cnn_u64 Col = 0;
	    Col < Layer->Bias.Cols;
	    Col++)
	{
	    float Value = CNN_MATRIX_VAL(&Layer->Bias, 0, Col);
	    fwrite(&Value, sizeof(Value), 1, OutputFile);
	}
	
	LastLayerOutputs = FileLayer.OutputCount;
    }

    fflush(OutputFile);
    fclose(OutputFile);
}

CNN_DEF void
cnn_load_weights(cnn_neural_network *NN, char *Filename)
{
    cnn_u64 FileSize = 0;

    FILE *File = fopen(Filename, "r");
    if(File)
    {
	fseek(File, 0, SEEK_END);
	FileSize = ftell(File);	
	fseek(File, 0, SEEK_SET);
    }
    else
    {
	CNN_print("Cannot open weights file: %s\n", Filename);
	return;
    }

    cnn_u8 *Data = (cnn_u8 *)CNN_malloc(FileSize);    
    fread(Data, FileSize, 1, File);

    cnn_weights_file_header *FileHeader = (cnn_weights_file_header *)Data;
    CNN_assert(FileHeader->FileID[0] == FILE_ID[0]);
    CNN_assert(FileHeader->FileID[1] == FILE_ID[1]);
    CNN_assert(FileHeader->FileID[2] == FILE_ID[2]);
    CNN_assert(FileHeader->FileID[3] == FILE_ID[3]);
    CNN_assert(FileHeader->LayerCount == NN->LayerCount);

    cnn_u64 LastLayerOutputs = FileHeader->Inputs;
    for(cnn_u64 LayerIndex = 0;
	LayerIndex < FileHeader->LayerCount;
	LayerIndex++)
    {
	cnn_weights_file_layer *FileLayer = (cnn_weights_file_layer *)(FileHeader + 1);
	CNN_assert(FileLayer->LayerID[0] == LAYER_ID[0]);
	CNN_assert(FileLayer->LayerID[1] == LAYER_ID[1]);
	CNN_assert(FileLayer->LayerID[2] == LAYER_ID[2]);
	CNN_assert(FileLayer->LayerID[3] == LAYER_ID[3]);

	cnn_layer *NNLayer = NN->Layers + LayerIndex;
	NNLayer->Activation = (cnn_activation_func)FileLayer->Activation;

	float *WeightsPtr = (float *)(FileLayer + 1);
	
	for(cnn_u64 Row = 0;
	    Row < NNLayer->Weights.Rows;
	    Row++)
	{
	    for(cnn_u64 Col = 0;
		Col < NNLayer->Weights.Cols;
		Col++)
	    {
		CNN_MATRIX_VAL(&NNLayer->Weights, Row, Col) = *WeightsPtr++;
	    }
	}
	for(cnn_u64 Col = 0;
	    Col < NNLayer->Bias.Cols;
	    Col++)
	{
	    CNN_MATRIX_VAL(&NNLayer->Bias, 0, Col) = *WeightsPtr++;
	}
	
	LastLayerOutputs = FileLayer->OutputCount;
    }
    
    CNN_free(Data);
}

CNN_DEF void
cnn_save_telemetry_csv(cnn_nn_info *NN)
{
    //NOTE: need stdio.h
    FILE *OutputFile = 0;
    OutputFile = fopen("telemetry.csv", "wb");
    if(!OutputFile)
    {
	CNN_print("Can't open file for writing\n");
	return;
    }

    if(NN->FirstTelemetryFrame && NN->TelemetryFrameCount)
    {
	char Buffer[256];
	cnn_s32 Bytes = CNN_snprint(Buffer, 256, "Epoch, Batch, Loss, Zeroed, Exploded, LargestRelGradErr, AverageRelGradErr\n");
	fwrite(Buffer, sizeof(char), Bytes, OutputFile);

	for(cnn_s64 FrameIndex = 0;
	    FrameIndex < NN->TelemetryFrameCount;
	    FrameIndex++)
	{
	    cnn_telemetry_frame *Frame = NN->FirstTelemetryFrame + FrameIndex;
	    Bytes = CNN_snprint(Buffer, 256, "%llu, %llu, %f, %llu, %llu, %f, %f\n",
					FrameIndex / NN->BatchCount,
					FrameIndex % NN->BatchCount,
					Frame->Loss,
					Frame->ZeroGradientsCount,
					Frame->ExplodedGradientsCount,
					Frame->LargestError,
					Frame->AverageError);
	    fwrite(Buffer, sizeof(char), Bytes, OutputFile);
	}        
    }
    else
    {
	CNN_print("No telemetry frames to save\n");
    }

    fflush(OutputFile);
    fclose(OutputFile);
}
#endif


CNN_DEF void
cnn_forward(cnn_neural_network *NN, cnn_matrix *Input)
{
    CNN_assert(Input->Rows == 1);
    
    cnn_matrix *LastLayerOutput = Input;
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *Layer = NN->Layers + LayerIndex;
	cnn_matrix_mul(LastLayerOutput, &Layer->Weights, &Layer->Outputs);
	cnn_matrix_add(&Layer->Bias, &Layer->Outputs);

	for(cnn_u64 ColIndex = 0;
	    ColIndex < Layer->Outputs.Cols;
	    ColIndex++)
	{
	    float Value = CNN_MATRIX_VAL(&Layer->Outputs, 0, ColIndex);
	    CNN_assert(!CNN_isnan(Value));
	    
	    float Activated = 0.0f;
	    switch(Layer->Activation)
	    {
		case CNN_ACT_RELU:
		{
		    Activated = Value > 0.0f ? Value : 0.0f;
		} break;
		case CNN_ACT_SIGMOID:
		{
		    Activated = 1.0f/(1.0f + CNN_exp(-Value));
		} break;
		case CNN_ACT_TANH:
		{
		    Activated = CNN_tanh(Value);
		} break;
		
		CNN_invalid_default_case;
	    }
	    
	    CNN_MATRIX_VAL(&Layer->Outputs, 0, ColIndex) = Activated;
	}
	LastLayerOutput = &Layer->Outputs;
    }
}

CNN_DEF float
cnn_loss(cnn_neural_network *NN, cnn_matrix *InputBatch, cnn_matrix *OutputBatch)
{   
    CNN_assert(InputBatch->Rows == OutputBatch->Rows);
    
    float Result = 0.0f;
    for(cnn_s64 SampleIndex = 0;
	SampleIndex < OutputBatch->Rows;
	SampleIndex++)
    {
	cnn_matrix Input = cnn_matrixrow(InputBatch, SampleIndex);
	cnn_forward(NN, &Input);
	
	cnn_matrix NNOutputMatrix = CNN_NN_OUTPUT(NN);
	float SampleLoss = 0.0f;
	for(cnn_s64 OutputCol = 0;
	    OutputCol < NNOutputMatrix.Cols;
	    OutputCol++)
	{
	    float NNOutput      = CNN_MATRIX_VAL(&NNOutputMatrix,           0, OutputCol);
	    float DesiredOutput = CNN_MATRIX_VAL(OutputBatch,     SampleIndex, OutputCol);

	    float DifferenceSq = (NNOutput - DesiredOutput)*(NNOutput - DesiredOutput);
	    SampleLoss += DifferenceSq;
	}
	SampleLoss /= NNOutputMatrix.Cols;
	
	Result += SampleLoss;
    }

    Result /= OutputBatch->Rows;
    return(Result);
}

CNN_DEF float
cnn_backprop(cnn_neural_network *NN, cnn_neural_network *G, cnn_u64 BatchSize, cnn_matrix *Inputs, cnn_matrix *Outputs)
{
    CNN_assert(Outputs->Cols == CNN_NN_OUTPUT(NN).Cols);
    
    cnn_init_nn_zero(G);

    float Loss = 0.0f;
    
    for(cnn_s64 SampleIndex = 0;
	SampleIndex < BatchSize;
	SampleIndex++)
    {
	for(cnn_s64 LayerIndex = 0;
	    LayerIndex < G->LayerCount;
	    LayerIndex++)
	{
	    cnn_layer *GLayer = G->Layers + LayerIndex;
	    cnn_zeromatrix(&GLayer->Outputs);
	}

	cnn_matrix Input = cnn_matrixrow(Inputs, SampleIndex);
	cnn_forward(NN, &Input);

	
	float SampleLoss = 0.0f;
	cnn_matrix NNOutputMatrix = CNN_NN_OUTPUT(NN);
	for(cnn_s64 OutputCol = 0;
	    OutputCol < NNOutputMatrix.Cols;
	    OutputCol++)
	{
	    float NNOutput      = CNN_MATRIX_VAL(&NNOutputMatrix,           0, OutputCol);
	    float DesiredOutput = CNN_MATRIX_VAL(Outputs,         SampleIndex, OutputCol);

	    //NOTE: Last Layer output's derivative is the derivative of the loss function:
	    //      SUM(((nn_output - desired)^2)/samplecount)
	    //      -> 2(nn_output - desired)
	    //      (samplecount div can be pulled out to the end)
	    
	    CNN_NN_OUTPUT(G).E[OutputCol] = 2.0f*(NNOutput - DesiredOutput);
	    
	    float DifferenceSq = (NNOutput - DesiredOutput)*(NNOutput - DesiredOutput);
	    SampleLoss += DifferenceSq;
	}
	SampleLoss /= NNOutputMatrix.Cols;
	Loss += SampleLoss;

	
	for(cnn_s64 LayerIndex = (NN->LayerCount - 1);
	    LayerIndex >= 0;
	    LayerIndex--)
	{
	    cnn_layer *NNLayer = NN->Layers + LayerIndex;
	    cnn_layer *GLayer  = G->Layers  + LayerIndex;

	    cnn_layer *PrevNNLayer = NN->Layers + (LayerIndex - 1);
	    cnn_layer *PrevGLayer  = G->Layers  + (LayerIndex - 1);

	    cnn_matrix *PrevLayerOutputs = &PrevNNLayer->Outputs;
	    if(LayerIndex == 0)
	    {
		//NOTE: First layer's previous activation function is the input
		PrevLayerOutputs = &Input;
		PrevGLayer = 0;
	    }
	
	    for(cnn_s64 ActivationCol = 0;
		ActivationCol < NNLayer->Outputs.Cols;
		ActivationCol++)
	    {
		float PropDerivative = CNN_MATRIX_VAL(&GLayer->Outputs,  0, ActivationCol); //NOTE: Back propogated derivative
		float Activation     = CNN_MATRIX_VAL(&NNLayer->Outputs, 0, ActivationCol);
		float dActivation    = 0.0f;
		switch(NNLayer->Activation)
		{
		    case CNN_ACT_RELU:
		    {
			if(Activation <= 0.0f)
			{
			    dActivation = 0.0f;
			}
			else
			{
			    dActivation = 1.0f;
			}
		    } break;
		    case CNN_ACT_TANH:
		    {
			//NOTE: 1 - tanh^2	    
			dActivation = 1.0f - Activation*Activation;
		    } break;
		    case CNN_ACT_SIGMOID:
		    {
			//NOTE: sig(1 - sig)
			dActivation = Activation*(1.0f - Activation);
		    } break;

		    CNN_invalid_default_case;
		}
		
		CNN_MATRIX_VAL(&GLayer->Bias, 0, ActivationCol) += (dActivation*PropDerivative);

		for(cnn_s64 PrevActivationCol = 0;
		    PrevActivationCol < PrevLayerOutputs->Cols;
		    PrevActivationCol++)
		{
		    float PrevActivation = CNN_MATRIX_VAL(PrevLayerOutputs,                  0, PrevActivationCol);
		    float Weight         = CNN_MATRIX_VAL(&NNLayer->Weights, PrevActivationCol,     ActivationCol);

		    CNN_MATRIX_VAL(&GLayer->Weights, PrevActivationCol, ActivationCol) +=
			(PrevActivation*dActivation*PropDerivative);

		    if(LayerIndex != 0)
		    {
			//NOTE: First layer's previous activation function is the input
			CNN_MATRIX_VAL(&PrevGLayer->Outputs, 0, PrevActivationCol) +=
			    (Weight*dActivation*PropDerivative);
		    }
		}
	    }
	}
    }
    
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < G->LayerCount;
	LayerIndex++)
    {
	cnn_layer *GLayer = G->Layers + LayerIndex;

	cnn_matrix *Weights = &GLayer->Weights;
	for(cnn_s64 RowIndex = 0;
	    RowIndex < Weights->Rows;
	    RowIndex++)
	{
	    for(cnn_s64 ColIndex = 0;
		ColIndex < Weights->Cols;
		ColIndex++)
	    {
		CNN_MATRIX_VAL(Weights, RowIndex, ColIndex) /= BatchSize;
	    }
	}

	cnn_matrix *Bias = &GLayer->Bias;
	for(cnn_s64 ColIndex = 0;
	    ColIndex < Bias->Cols;
	    ColIndex++)
	{
	    CNN_MATRIX_VAL(Bias, 0, ColIndex) /= BatchSize;
	}
    }
    
    Loss /= BatchSize;
    return(Loss);
}

CNN_DEF double
CNN_max(double A, double B)
{
    double Result = A;
    if(B > A) Result = B;
    return(Result);
}

CNN_DEF double
cnn_grad_rel_error(double Num, double Ana)
{    
    double Result = 0.0f;

    double Diff = CNN_fabs_double(Num - Ana);
    double Denominator = CNN_max(Num, Ana);

    if(Denominator != 0.0f)
    {
	Result = fabs(Diff/Denominator);
    }

    return(Result);
}

CNN_DEF double
cnn_validate_gradients(cnn_nn_info *NN, double h,
		       cnn_matrix *InputBatch, cnn_matrix *OutputBatch,
		       cnn_neural_network *Output, cnn_telemetry_frame *TelemetryFrame,
		       bool StoreErrors)
{
    double LargestRelError = 0.0f;
    double AverageRelError = 0.0f;
    cnn_s64 GradCount = 0;
    nn_weight_pos LargestErrorPos;
	
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *NNLayer   = NN->NN->Layers + LayerIndex;
	cnn_layer *GLayer    = NN->G->Layers  + LayerIndex;
	cnn_layer *OutLayer  = Output->Layers + LayerIndex;
	
	cnn_matrix *Weights         = &NNLayer->Weights;
	cnn_matrix *WeightGradients = &GLayer->Weights;
	cnn_matrix *WeightsErrors   = &OutLayer->Weights;
	
	for(cnn_s64 Col = 0;
	    Col < Weights->Cols;
	    Col++)
	{
	    for(cnn_s64 Row = 0;
		Row < Weights->Rows;
		Row++)
	    {
		float Grad  = CNN_MATRIX_VAL(WeightGradients, Row, Col);
		float Value = CNN_MATRIX_VAL(Weights, Row, Col);

		CNN_MATRIX_VAL(Weights, Row, Col) = (Value - h);
		double LowLoss = cnn_loss(NN->NN, InputBatch, OutputBatch);

		CNN_MATRIX_VAL(Weights, Row, Col) = (Value + h);
		double HighLoss = cnn_loss(NN->NN, InputBatch, OutputBatch);

		double FiniteDiff = (HighLoss - LowLoss) / (2.0f * h);
		double RelError = cnn_grad_rel_error(Grad, FiniteDiff);

		if(StoreErrors) CNN_MATRIX_VAL(WeightsErrors, Row, Col) = RelError;
		
		if(!CNN_isnan(RelError) && Grad != 0.0f && RelError >= LargestRelError)
		{
		    AverageRelError += RelError;
		    LargestRelError = RelError;
		    LargestErrorPos.Row    = Row;
		    LargestErrorPos.Col    = Col;
		    LargestErrorPos.IsBias = false;
		}
		
//		CNN_print("Grad: %lf, FiniteDiff: %lf, RelativeError: %lf\n", Grad, FiniteDiff, RelError);

		CNN_MATRIX_VAL(Weights, Row, Col) = Value;
	    }
	}

	cnn_matrix *Bias          = &NNLayer->Bias;
	cnn_matrix *BiasGradients = &GLayer->Bias;
	cnn_matrix *BiasErrors    = &OutLayer->Bias;
	for(cnn_s64 Col = 0;
	    Col < Weights->Cols;
	    Col++)
	{
	    float Grad = CNN_MATRIX_VAL(BiasGradients, 0, Col);
	    float Value = CNN_MATRIX_VAL(Bias, 0, Col);

	    CNN_MATRIX_VAL(Bias, 0, Col) = (Value - h);
	    float LowLoss = cnn_loss(NN->NN, InputBatch, OutputBatch);

	    CNN_MATRIX_VAL(Bias, 0, Col) = (Value + h);
	    float HighLoss = cnn_loss(NN->NN, InputBatch, OutputBatch);

	    float FiniteDiff = (HighLoss - LowLoss) / (2.0f * h);
	    double RelError = cnn_grad_rel_error(Grad, FiniteDiff);

	    if(StoreErrors) CNN_MATRIX_VAL(BiasErrors, 0, Col) = RelError;

	    if(!CNN_isnan(RelError) && Grad != 0.0f && RelError >= LargestRelError)
	    {
		AverageRelError += RelError;
		LargestRelError = RelError;
		LargestErrorPos.Row    = 1;
		LargestErrorPos.Col    = Col;
		LargestErrorPos.IsBias = true;
	    }
	    
//	    CNN_print("Grad: %lf, FiniteDiff: %lf, Relative Error: %lf\n", Grad, FiniteDiff, RelError);

	    CNN_MATRIX_VAL(Bias, 0, Col) = Value;
	}
	GradCount += (Weights->Rows*Weights->Cols + Bias->Cols);
    }

    if(TelemetryFrame)
    {
	TelemetryFrame->AverageError = AverageRelError / (double)GradCount;
	TelemetryFrame->LargestError    = LargestRelError;
	TelemetryFrame->LargestErrorPos = LargestErrorPos;
    }
    
    return(LargestRelError);
}

CNN_DEF void
cnn_gradient_descent(cnn_nn_info *NN, float LearningRate, cnn_telemetry_frame *TelemetryFrame,
		     cnn_u64 *Vanished, cnn_u64 *Exploded)
{
    *Vanished = 0;
    *Exploded = 0;
    for(cnn_s64 LayerIndex = 0;
	LayerIndex < NN->NN->LayerCount;
	LayerIndex++)
    {
	cnn_layer *NNLayer = NN->NN->Layers + LayerIndex;
	cnn_layer *GLayer  = NN->G->Layers  + LayerIndex;

	cnn_matrix *Weights         = &NNLayer->Weights;
	cnn_matrix *WeightGradients = &GLayer->Weights;
	for(cnn_s64 Col = 0;
	    Col < Weights->Cols;
	    Col++)
	{
	    for(cnn_s64 Row = 0;
		Row < Weights->Rows;
		Row++)
	    {
		float Grad = CNN_MATRIX_VAL(WeightGradients, Row, Col);
		CNN_MATRIX_VAL(Weights, Row, Col) -= LearningRate*Grad;

		if(Grad == 0.0f)    (*Vanished)++;
		if(CNN_isnan(Grad)) (*Exploded)++;
	    }
	}

	cnn_matrix *Bias          = &NNLayer->Bias;
	cnn_matrix *BiasGradients = &GLayer->Bias;
	for(cnn_s64 Col = 0;
	    Col < Weights->Cols;
	    Col++)
	{
	    float Grad = CNN_MATRIX_VAL(BiasGradients, 0, Col);
	    CNN_MATRIX_VAL(Bias, 0, Col) -= LearningRate*Grad;
	    
	    if(Grad == 0.0f)    (*Vanished)++;
	    if(CNN_isnan(Grad)) (*Exploded)++;
	}
    }
}

CNN_DEF void
cnn_train(cnn_nn_info *NN,
	  cnn_matrix *TrainingIn, cnn_matrix *TrainingOut,
	  cnn_u64 BatchSize, cnn_u64 EpochCount, float LearningRate, cnn_u32 TelemetryFlags)
{
    cnn_u64 BatchCount = CNN_ceil((float)TrainingIn->Rows/(float)BatchSize);

    float LastLoss = 1.0f;

    if(TelemetryFlags)
    {
	NN->EpochCount = EpochCount;
	NN->BatchCount = BatchCount;
	
	cnn_u64 FrameCount = EpochCount*BatchCount;

	cnn_u64 FrameArraySize = sizeof(cnn_telemetry_frame)*FrameCount;
	if(TelemetryFlags & TELEMETRYFLAG_STOREGRADERRORS)
	{
	    FrameArraySize += FrameCount*cnn_get_nn_size_from_nn(NN->NN);
	}

	CNN_print("Telemetry ON requires %llu bytes\n", FrameArraySize);
	CNN_print("WARNING: Telemetry significantly slows down training\n");
	CNN_print("         (especially TELEMETRYFLAG_STOREGRADERRORS)\n");
	CNN_print("         the idea is for you to collect a lot of data\n");
	CNN_print("         for a small proportion of your training data\n");
	CNN_print("         before a full send\n");
	
	cnn_arena TelemetryArena = cnn_new_arena(FrameArraySize);
	NN->TelemetryFrameCount = FrameCount;
	NN->FirstTelemetryFrame = CNN_push_array(&TelemetryArena, FrameCount, cnn_telemetry_frame);
	CNN_zero_array(FrameCount, NN->FirstTelemetryFrame);

	if(TelemetryFlags & TELEMETRYFLAG_STOREGRADERRORS)
	{
	    for(cnn_u64 FrameIndex = 0;
		FrameIndex < FrameCount;
		FrameIndex++)
	    {
		cnn_telemetry_frame *Frame = NN->FirstTelemetryFrame + FrameIndex;
		Frame->RelativeGradientErrors = cnn_alloc_nn_from_nn(&TelemetryArena, NN->NN);
	    }
	}

	CNN_assert(TelemetryArena.Size == TelemetryArena.Used);
    }

    CNN_print("Epoch: 0/%llu, Loss: 0.0, Delta: 0.0", EpochCount);
    cnn_u64 FrameIndex = 0;
    for(cnn_s64 Epoch = 0;
	Epoch < EpochCount;
	Epoch++)
    {
	float EpochAverageLoss = 0.0f;
	for(cnn_s64 BatchIndex = 0;
	    BatchIndex < BatchCount;
	    BatchIndex++)
	{
	    if((BatchIndex + BatchSize) > TrainingIn->Rows)
	    {
		BatchSize = TrainingIn->Rows - BatchIndex;
	    }
	    cnn_matrix BatchInput = cnn_submatrix(TrainingIn, BatchSize,
					      0,
					      &CNN_MATRIX_VAL(TrainingIn, BatchIndex*BatchSize, 0));
	    cnn_matrix BatchOutput = cnn_submatrix(TrainingOut, BatchSize,
					       0,
					       &CNN_MATRIX_VAL(TrainingOut, BatchIndex*BatchSize, 0));
	    
	    float Loss = cnn_backprop(NN->NN, NN->G, BatchSize, &BatchInput, &BatchOutput);

	    cnn_telemetry_frame *TelemetryFrame = 0;
	    if(TelemetryFlags & TELEMETRYFLAG_ON)
	    {
		TelemetryFrame = NN->FirstTelemetryFrame + FrameIndex++;
		if(TelemetryFlags & TELEMETRYFLAG_VALIDATEGRADIENTS)
		{
		    cnn_validate_gradients(NN, 0.01f, &BatchInput, &BatchOutput,
					   TelemetryFrame->RelativeGradientErrors, TelemetryFrame,
					   (TelemetryFlags & TELEMETRYFLAG_STOREGRADERRORS));
		}

		EpochAverageLoss += Loss;
		cnn_u64 Vanished = 0;
		cnn_u64 Exploded = 0;
		cnn_gradient_descent(NN, LearningRate, TelemetryFrame, &Vanished, &Exploded);
		TelemetryFrame->Loss                   = Loss;
		TelemetryFrame->ZeroGradientsCount     = Vanished;
		TelemetryFrame->ExplodedGradientsCount = Exploded;
	    }
	    else
	    {
		EpochAverageLoss += Loss;
		cnn_u64 Vanished = 0;
		cnn_u64 Exploded = 0;
		cnn_gradient_descent(NN, LearningRate, TelemetryFrame, &Vanished, &Exploded);
	    }
	}
	EpochAverageLoss /= BatchCount;

	float Delta = EpochAverageLoss - LastLoss;
	printf("\r");
	CNN_print("Epoch: %llu/%llu, Loss: %f, Delta: %f", Epoch + 1, EpochCount, EpochAverageLoss, Delta);

	LastLoss = EpochAverageLoss;
    }
    CNN_print("\n");
}

CNN_DEF void
cnn_print_telemetry_frames(cnn_nn_info *NN)
{
    if(NN->FirstTelemetryFrame && NN->TelemetryFrameCount)
    {
	for(cnn_s64 FrameIndex = 0;
	    FrameIndex < NN->TelemetryFrameCount;
	    FrameIndex++)
	{
	    cnn_telemetry_frame *Frame = NN->FirstTelemetryFrame + FrameIndex;

	    CNN_print("Epoch: %llu, Batch %llu, Loss: %f, Zeroed: %llu, Exploded: %llu, Largest Relative GradError: %f, Average Rel Error: %f\n",
		      FrameIndex / NN->BatchCount,
		      FrameIndex % NN->BatchCount,
		      Frame->Loss,
		      Frame->ZeroGradientsCount,
		      Frame->ExplodedGradientsCount,
		      Frame->LargestError,
		      Frame->AverageError);
	}        
    }
    else
    {
	CNN_print("No telemetry frames to display\n");
    }
}

#endif //CNN implementation
