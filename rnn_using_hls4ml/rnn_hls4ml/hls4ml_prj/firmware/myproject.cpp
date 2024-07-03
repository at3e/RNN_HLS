#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t lstm1_input[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer4_out[N_LAYER_3]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=lstm1_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=lstm1_input,layer4_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 3584>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 4096>(wr2, "wr2.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(br2, "br2.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(w3, "w3.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(b3, "b3.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_OUT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::lstm_stack<input_t, layer2_t, config2>(lstm1_input, layer2_out, w2, wr2, b2, br2); // lstm1

    layer3_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // output

    nnet::softmax<layer3_t, result_t, softmax_config4>(layer3_out, layer4_out); // output_softmax

}
