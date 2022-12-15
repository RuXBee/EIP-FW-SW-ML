#include <Arduino.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "LSM9DS1.h" 
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"


// Predefined MACROS to execute data acquisition program
//#define DATA_ACQUISITION
#define IMU_MSG_CSV_HEADER      "%s, %s, %s, %s, %s, %s\r\n"
#define IMU_MSG_TO_CSV          "%0.03f, %0.03f, %0.03f, %0.03f, %0.03f, %0.03f\r\n" 

// Typedef acceleration_samples_t
typedef struct {
    float accel_x;
    float accel_y;
    float accel_z;
    float gyros_x;
    float gyros_y;
    float gyros_z;
} acceleration_samples_t;

static acceleration_samples_t samples;
static char *msg; 

const float accelerationThreshold = 1; // threshold of significant in G's
const int numSamples = 20;

int samplesRead = numSamples;
// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = (100 * 1024);
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* STATES[] = {
  "normal",
  "falldown"
};


void setup() {
  
  // Initialize UART0 as Serial port
    Serial.begin(9600);
    while (!Serial);
    
    // Initialize Colour, Proximity and Gesture sensor 
    while (!IMU.begin()) {
        Serial.println("[ERROR][IMU] Impossible initialization");
        while (1);
    }

#   ifndef DATA_ACQUISITION
    // get the TFL representation of the model byte array
    tflModel = tflite::GetModel(state_model_tflite);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1);
    }

    // Create an interpreter to run the model
    tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

    // Allocate memory for the model's input and output tensors
    tflInterpreter->AllocateTensors();

    // Get pointers for the model's input and output tensors
    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);

    Serial.println("[INFO][BOARD] Init system successfully");
    delay(10);
#   else
    // Build string message and print it
    msg = (char*)malloc(sizeof(IMU_MSG_CSV_HEADER) + 7*6*sizeof(char));
    sprintf(msg, IMU_MSG_CSV_HEADER,
            "accel_x", 
            "accel_y", 
            "accel_z",
            "gyros_x",
            "gyros_y",
            "gyros_z");
    Serial.print(msg);
    free(msg);
#   endif
}

void loop() {

#   ifndef DATA_ACQUISITION

    // wait for significant motion
    while (samplesRead == numSamples) {
        if (IMU.accelerationAvailable()) {
            // read the acceleration data
            IMU.readAcceleration(samples.accel_x, samples.accel_y, samples.accel_z);

            // sum up the absolutes
            float aSum = fabs(samples.accel_x) + fabs(samples.accel_y) + fabs(samples.accel_z);

            // check if it's above the threshold
            if (aSum >= accelerationThreshold) {
                // reset the sample read count
                samplesRead = 0;
                break;
            }
        }
    }

    // check if the all the required samples have been read since
    // the last time the significant motion was detected
    while (samplesRead < numSamples) {
        // check if new acceleration AND gyroscope data is available
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
            // read the acceleration and gyroscope data
            IMU.readAcceleration(samples.accel_x, samples.accel_y, samples.accel_z);
            IMU.readGyroscope(samples.gyros_x, samples.gyros_y, samples.gyros_z);

            // normalize the IMU data between 0 to 1 and store in the model's
            // input tensor
            tflInputTensor->data.f[samplesRead * 6 + 0] = (samples.accel_x + 4.0) / 8.0;
            tflInputTensor->data.f[samplesRead * 6 + 1] = (samples.accel_y + 4.0) / 8.0;
            tflInputTensor->data.f[samplesRead * 6 + 2] = (samples.accel_z + 4.0) / 8.0;
            tflInputTensor->data.f[samplesRead * 6 + 3] = (samples.gyros_x+ 2000.0) / 4000.0;
            tflInputTensor->data.f[samplesRead * 6 + 4] = (samples.gyros_y+ 2000.0) / 4000.0;
            tflInputTensor->data.f[samplesRead * 6 + 5] = (samples.gyros_z + 2000.0) / 4000.0;


            samplesRead++;

            if (samplesRead == numSamples) {
                // Run inferencing
                TfLiteStatus invokeStatus = tflInterpreter->Invoke();
                if (invokeStatus != kTfLiteOk) {
                    Serial.println("Invoke failed!");
                    while (1);
                    return;
                }

                // Loop through the output tensor values from the model
                for (int i = 0; i < 2; i++) {
                    Serial.print(STATES[i]);
                    Serial.print(": ");
                    if (i == 0) Serial.print(100 - tflOutputTensor->data.f[0]*100, 4);
                    else if (i == 1) Serial.print(tflOutputTensor->data.f[0]*100, 4);
                    Serial.println("%\n");
                }
                Serial.println();
            }
        }
    }
#   else
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {

        // Allocate space for message in heap memory
        msg = (char*)malloc(sizeof(IMU_MSG_TO_CSV) + 3*3);
        
        // Read data from Sensor
        IMU.readAcceleration(samples.accel_x, samples.accel_y, samples.accel_z);
        IMU.readGyroscope(samples.gyros_x, samples.gyros_y, samples.gyros_z);

        // Build string message and print it
        sprintf(msg, IMU_MSG_TO_CSV,
                samples.accel_x, 
                samples.accel_y, 
                samples.accel_z,
                samples.gyros_x,
                samples.gyros_y,
                samples.gyros_z);
        Serial.print(msg);
        
        // Deallocate and free space memory in heap memory
        free(msg);
    }
    delay(200);
#   endif
}