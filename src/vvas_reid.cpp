/*
 * Copyright 2021 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gst/vvas/gstinferencemeta.h>
#include <vvas/vvas_kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vitis/ai/nnpp/reid.hpp>
#include <vitis/ai/reid.hpp>
#include <vitis/ai/reidtracker.hpp>
#include "common.hpp"
#include <sstream>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>


#define MAX_REID 20
#define DEFAULT_REID_THRESHOLD 0.2
#define DEFAULT_REID_DEBUG     0
#define DEFAULT_REID_PORT     1234
#define DEFAULT_MODEL_NAME     "personreid-res18_pt"
#define DEFAULT_MODEL_PATH     "/opt/xilinx/kv260-aibox-reid/share/vitis_ai_library/models"

using namespace std;

struct _Face {
  int last_frame_seen;
  int xctr;
  int yctr;
  int id;
  cv::Mat features;
};

typedef struct _kern_priv {
  uint32_t debug;
  uint32_t port;
  double threshold;
  std::string modelpath;
  std::string modelname;
  std::shared_ptr<vitis::ai::Reid> det;
  std::shared_ptr<vitis::ai::ReidTracker> tracker;
} ReidKernelPriv;

struct _roi {
    uint32_t y_cord;
    uint32_t x_cord;
    uint32_t height;
    uint32_t width;
    double   prob;
	  GstInferencePrediction *prediction;
};

#define MAX_CHANNELS 40
typedef struct _vvas_ms_roi {
    uint32_t nobj;
    struct _roi roi[MAX_CHANNELS];
} vvas_ms_roi;

static int parse_rect(VVASKernel * handle, int start,
      VVASFrame * input[MAX_NUM_OBJECT], VVASFrame * output[MAX_NUM_OBJECT],
      vvas_ms_roi &roi_data
      )
{
    VVASFrame *inframe = input[0];
    GstInferenceMeta *infer_meta = ((GstInferenceMeta *)gst_buffer_get_meta((GstBuffer *)
                                                              inframe->app_priv,
                                                          gst_inference_meta_api_get_type()));
    roi_data.nobj = 0;
    if (infer_meta == NULL)
    {
        return 0;
    }

    GstInferencePrediction *root = infer_meta->prediction;

    /* Iterate through the immediate child predictions */
    GSList *tmp = gst_inference_prediction_get_children(root);
    for (GSList *child_predictions = tmp;
         child_predictions;
         child_predictions = g_slist_next(child_predictions))
    {
        GstInferencePrediction *child = (GstInferencePrediction *)child_predictions->data;

        /* On each children, iterate through the different associated classes */
        for (GList *classes = child->classifications;
             classes; classes = g_list_next(classes))
        {
            GstInferenceClassification *classification = (GstInferenceClassification *)classes->data;
            if (roi_data.nobj < MAX_CHANNELS)
            {
                int ind = roi_data.nobj;
                struct _roi &roi = roi_data.roi[ind];
                roi.y_cord = (uint32_t)child->bbox.y + child->bbox.y % 2;
                roi.x_cord = (uint32_t)child->bbox.x;
                roi.height = (uint32_t)child->bbox.height - child->bbox.height % 2;
                roi.width = (uint32_t)child->bbox.width - child->bbox.width % 2;
                roi.prob = classification->class_prob;
                roi.prediction = child;
                roi_data.nobj++;

            }
        }
    }
    g_slist_free(tmp);
    return 0;
}

extern "C" {
int32_t xlnx_kernel_init(VVASKernel *handle) {
  json_t *jconfig = handle->kernel_config;
  json_t *val; /* kernel config from app */

  handle->is_multiprocess = 1;
  ReidKernelPriv *kernel_priv =
      (ReidKernelPriv *)calloc(1, sizeof(ReidKernelPriv));
  if (!kernel_priv) {
    printf("Error: Unable to allocate reID kernel memory\n");
  }

  /* parse config */
  val = json_object_get(jconfig, "threshold");
  if (!val || !json_is_number(val))
    kernel_priv->threshold = DEFAULT_REID_THRESHOLD;
  else
    kernel_priv->threshold = json_number_value(val);

  val = json_object_get(jconfig, "debug");
  if (!val || !json_is_number(val))
    kernel_priv->debug = DEFAULT_REID_DEBUG;
  else
    kernel_priv->debug = json_number_value(val);

  val = json_object_get(jconfig, "port");
  if (!val || !json_is_number(val))
    kernel_priv->port = DEFAULT_REID_DEBUG;
  else
    kernel_priv->port = json_number_value(val);

  val = json_object_get(jconfig, "model-name");
  if (!val || !json_is_string (val))
    kernel_priv->modelname = DEFAULT_MODEL_NAME;
  else
    kernel_priv->modelname = (char *) json_string_value (val);

  val = json_object_get(jconfig, "model-path");
  if (!val || !json_is_string (val))
    kernel_priv->modelpath = DEFAULT_MODEL_PATH;
  else
    kernel_priv->modelpath = (char *) json_string_value (val);

  std::string xmodelfile = kernel_priv->modelpath + "/" + kernel_priv->modelname + "/" + kernel_priv->modelname + ".xmodel";
  kernel_priv->det = vitis::ai::Reid::create(xmodelfile);
  if (kernel_priv->det.get() == NULL) {
    printf("Error: Unable to create Reid runner with model %s.\n", xmodelfile.c_str());
  }
  kernel_priv->tracker = vitis::ai::ReidTracker::create();

  handle->kernel_priv = (void *)kernel_priv;
  return 0;
}

uint32_t xlnx_kernel_deinit(VVASKernel *handle) {
  ReidKernelPriv *kernel_priv = (ReidKernelPriv *)handle->kernel_priv;
  free(kernel_priv);
  return 0;
}

int32_t xlnx_kernel_start(VVASKernel *handle, int start /*unused */,
                          VVASFrame *input[MAX_NUM_OBJECT],
                          VVASFrame *output[MAX_NUM_OBJECT]) {
  VVASFrame *in_vvas_frame = input[0];
  ReidKernelPriv *kernel_priv = (ReidKernelPriv *)handle->kernel_priv;
  if ( !kernel_priv->det.get() || !kernel_priv->tracker.get() ) {
    return 1;
  }

  static int frame_num = 0;
  frame_num++;
  std::vector<vitis::ai::ReidTracker::InputCharact> input_characts;
  /* get metadata from input */
  cv::Mat tcpimage(input[0]->props.height, input[0]->props.width, CV_8UC3, (char *)in_vvas_frame->vaddr[0]);

  vvas_ms_roi roi_data;
  parse_rect(handle, start, input, output, roi_data);

  // Start timing here
  auto starttime = std::chrono::high_resolution_clock::now();
	
  //tcp connect setting
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in serverAddress;
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_port = htons(kernel_priv->port);
  serverAddress.sin_addr.s_addr = inet_addr("192.168.4.40");
	
  if (connect(sock, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) < 0) {
      std::cerr << "Connection failed." << std::endl;
      return -1;
  } else {
      std::cout << "Connected successfully." << std::endl;
  }
  
  int type = tcpimage.type();
  int rows = tcpimage.rows;
  int cols = tcpimage.cols;
  int channels = tcpimage.channels();

  std::cout << "Type: " << type << ", Rows: " << rows << ", Cols: " << cols << ", Channels: " << channels << std::endl;

  int converted_type = htonl(type);
  int converted_rows = htonl(rows);
  int converted_cols = htonl(cols);
  int converted_channels = htonl(channels);

  send(sock, &converted_type, sizeof(converted_type), 0);
  send(sock, &converted_rows, sizeof(converted_rows), 0);
  send(sock, &converted_cols, sizeof(converted_cols), 0);
  send(sock, &converted_channels, sizeof(converted_channels), 0);

  send(sock, (char*)tcpimage.data, tcpimage.total() * tcpimage.elemSize(), 0);
  std::cout << "Sending bytes: " << tcpimage.total() * tcpimage.elemSize() << std::endl;

  int bbox_count = roi_data.nobj;
  int converted_bbox_count = htonl(bbox_count);
  send(sock, &converted_bbox_count, sizeof(converted_bbox_count), 0);

  for (uint32_t i = 0; i < roi_data.nobj; i++) {
    uint32_t converted_x = htonl(roi_data.roi[i].x_cord);
    uint32_t converted_y = htonl(roi_data.roi[i].y_cord);
    uint32_t converted_width = htonl(roi_data.roi[i].width);
    uint32_t converted_height = htonl(roi_data.roi[i].height);
    std::cout << "x: " << roi_data.roi[i].x_cord << ", y: " << roi_data.roi[i].y_cord << ", width: " << roi_data.roi[i].width << ", height: " << roi_data.roi[i].height << std::endl;

    send(sock, &converted_x, sizeof(converted_x), 0);
    send(sock, &converted_y, sizeof(converted_y), 0);
    send(sock, &converted_width, sizeof(converted_width), 0);
    send(sock, &converted_height, sizeof(converted_height), 0);
  }

  close(sock);

  // Calculate and print the elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - starttime).count();
  std::cout << "Time taken to send image data: " << duration << " milliseconds." << std::endl;
	
  return 0;
}

int32_t xlnx_kernel_done(VVASKernel *handle) {
  /* dummy */
  return 0;
}
}
