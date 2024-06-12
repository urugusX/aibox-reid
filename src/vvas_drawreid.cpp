/*
 * Copyright 2021 Xilinx, Inc.
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
 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vvas/vvas_kernel.h>
#include <gst/vvas/gstinferencemeta.h>
#define __STDC_FORMAT_MACROS 1
#include <stdint.h>
#include <unistd.h>
#include "common.hpp"
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <chrono>

enum
{
  LOG_LEVEL_ERROR,
  LOG_LEVEL_WARNING,
  LOG_LEVEL_INFO,
  LOG_LEVEL_DEBUG
};

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_MESSAGE(level, ...) {\
  do {\
    char *str; \
    if (level == LOG_LEVEL_ERROR)\
      str = (char*)"ERROR";\
    else if (level == LOG_LEVEL_WARNING)\
      str = (char*)"WARNING";\
    else if (level == LOG_LEVEL_INFO)\
      str = (char*)"INFO";\
    else if (level == LOG_LEVEL_DEBUG)\
      str = (char*)"DEBUG";\
    if (level <= log_level) {\
      printf("[%s %s:%d] %s: ",__FILENAME__, __func__, __LINE__, str);\
      printf(__VA_ARGS__);\
      printf("\n");\
    }\
  } while (0); \
}


int log_level = LOG_LEVEL_WARNING;

using namespace cv;
using namespace std;

#define MAX_CLASS_LEN 1024
#define MAX_LABEL_LEN 1024
#define MAX_ALLOWED_CLASS 20
#define MAX_ALLOWED_LABELS 20
#define DEFAULT_REID_PORT     1234

struct color
{
  unsigned int blue;
  unsigned int green;
  unsigned int red;
};

struct vvass_xclassification
{
  color class_color;
  char class_name[MAX_CLASS_LEN];
};


struct vvas_xoverlaypriv
{
  float font_size;
  unsigned int font;
  uint32_t port;
  int line_thickness;
  int y_offset;
  color label_color;
  char label_filter[MAX_ALLOWED_LABELS][MAX_LABEL_LEN];
  unsigned char label_filter_cnt;
  unsigned short classes_count;
  vvass_xclassification class_list[MAX_ALLOWED_CLASS];
};

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

void convertYUVtoRGB(const Mat &lumaImg, const Mat &chromaImg, Mat &outputRGB) {
    // Tách U và V từ ảnh Chroma (UV)
    Mat u, v;
    u.create(lumaImg.rows / 2, lumaImg.cols / 2, CV_8UC1);
    v.create(lumaImg.rows / 2, lumaImg.cols / 2, CV_8UC1);

    for (int i = 0; i < chromaImg.rows; i++) {
        for (int j = 0; j < chromaImg.cols; j++) {
            uint16_t uv = chromaImg.at<uint16_t>(i, j);
            u.at<uchar>(i, j) = uv >> 8;
            v.at<uchar>(i, j) = uv & 0xFF;
        }
    }

    // Thay đổi kích thước của U và V
    Mat u_resized, v_resized;
    resize(u, u_resized, lumaImg.size(), 0, 0, INTER_LINEAR);
    resize(v, v_resized, lumaImg.size(), 0, 0, INTER_LINEAR);

    // Tạo ảnh YUV và chuyển đổi sang RGB
    Mat yuv;
    std::vector<Mat> yuv_channels = { lumaImg, u_resized, v_resized };
    merge(yuv_channels, yuv);
    cvtColor(yuv, outputRGB, COLOR_YUV2RGB);
}

extern "C"
{
  int32_t xlnx_kernel_init (VVASKernel * handle)
  {
    vvas_xoverlaypriv *kpriv =
        (vvas_xoverlaypriv *) malloc (sizeof (vvas_xoverlaypriv));
    memset (kpriv, 0, sizeof (vvas_xoverlaypriv));

    json_t *jconfig = handle->kernel_config;
    json_t *val, *karray = NULL, *classes = NULL;

    /* Initialize config params with default values */
    log_level = LOG_LEVEL_WARNING;
    kpriv->font_size = 0.5;
    kpriv->font = 0;
    kpriv->line_thickness = 1;
    kpriv->y_offset = 0;
    kpriv->label_color = {0, 0, 0};
    strcpy(kpriv->label_filter[0], "class");
    strcpy(kpriv->label_filter[1], "probability");
    kpriv->label_filter_cnt = 2;
    kpriv->classes_count = 0;

      val = json_object_get (jconfig, "debug_level");
    if (!val || !json_is_integer (val))
        log_level = LOG_LEVEL_WARNING;
    else
        log_level = json_integer_value (val);

        val = json_object_get(jconfig, "port");
    if (!val || !json_is_number(val))
        kpriv->port = DEFAULT_REID_PORT;
     else
        kpriv->port = json_number_value(val);

      val = json_object_get (jconfig, "font_size");
    if (!val || !json_is_integer (val))
        kpriv->font_size = 0.5;
    else
        kpriv->font_size = json_integer_value (val);

      val = json_object_get (jconfig, "font");
    if (!val || !json_is_integer (val))
        kpriv->font = 0;
    else
        kpriv->font = json_integer_value (val);

      val = json_object_get (jconfig, "thickness");
    if (!val || !json_is_integer (val))
        kpriv->line_thickness = 1;
    else
        kpriv->line_thickness = json_integer_value (val);

      val = json_object_get (jconfig, "y_offset");
    if (!val || !json_is_integer (val))
        kpriv->y_offset = 0;
    else
        kpriv->y_offset = json_integer_value (val);

    /* get label color array */
      karray = json_object_get (jconfig, "label_color");
    if (!karray)
    {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find label_color");
      return -1;
    } else
    {
      kpriv->label_color.blue =
          json_integer_value (json_object_get (karray, "blue"));
      kpriv->label_color.green =
          json_integer_value (json_object_get (karray, "green"));
      kpriv->label_color.red =
          json_integer_value (json_object_get (karray, "red"));
    }

    karray = json_object_get (jconfig, "label_filter");

    if (!json_is_array (karray)) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "label_filter not found in the config\n");
      return -1;
    }
    kpriv->label_filter_cnt = 0;
    for (unsigned int index = 0; index < json_array_size (karray); index++) {
      strcpy (kpriv->label_filter[index],
          json_string_value (json_array_get (karray, index)));
      kpriv->label_filter_cnt++;
    }

    /* get classes array */
    karray = json_object_get (jconfig, "classes");
    if (!karray) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find key labels");
      return -1;
    }

    if (!json_is_array (karray)) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "labels key is not of array type");
      return -1;
    }
    kpriv->classes_count = json_array_size (karray);
    for (unsigned int index = 0; index < kpriv->classes_count; index++) {
      classes = json_array_get (karray, index);
      if (!classes) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to get class object");
        return -1;
      }

      val = json_object_get (classes, "name");
      if (!json_is_string (val)) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, "name is not found for array %d", index);
        return -1;
      } else {
        strncpy (kpriv->class_list[index].class_name,
            (char *) json_string_value (val), MAX_CLASS_LEN - 1);
        LOG_MESSAGE (LOG_LEVEL_DEBUG, "name %s",
            kpriv->class_list[index].class_name);
      }

      val = json_object_get (classes, "green");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.green = 0;
      else
        kpriv->class_list[index].class_color.green = json_integer_value (val);

      val = json_object_get (classes, "blue");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.blue = 0;
      else
        kpriv->class_list[index].class_color.blue = json_integer_value (val);

      val = json_object_get (classes, "red");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.red = 0;
      else
        kpriv->class_list[index].class_color.red = json_integer_value (val);
    }

    handle->kernel_priv = (void *) kpriv;
    return 0;
  }

  uint32_t xlnx_kernel_deinit (VVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *) handle->kernel_priv;

    if (kpriv)
      free (kpriv);

    return 0;
  }


  uint32_t xlnx_kernel_start (VVASKernel * handle, int start,
      VVASFrame * input[MAX_NUM_OBJECT], VVASFrame * output[MAX_NUM_OBJECT])
  {
    VVASFrame *inframe = input[0];
    vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *)handle->kernel_priv;

    auto starttime = std::chrono::high_resolution_clock::now();
	  
    Mat lumaImg(input[0]->props.height, input[0]->props.stride, CV_8UC1, (char *)inframe->vaddr[0]);
    Mat chromaImg(input[0]->props.height / 2, input[0]->props.stride / 2, CV_16UC1, (char *)inframe->vaddr[1]);
    Mat tcpimage;
	  
    convertYUVtoRGB(lumaImg, chromaImg, tcpimage);

    // Kiểm tra kiểu của bgrImg
    if (tcpimage.type() == CV_8UC3) {
        cout << "The output image is of type CV_8UC3." << endl;
    } else {
        cout << "The output image is not of type CV_8UC3." << endl;
    }

    vvas_ms_roi roi_data;
    parse_rect(handle, start, input, output, roi_data);

    //tcp connect setting
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(kpriv->port);
    serverAddress.sin_addr.s_addr = inet_addr("192.168.4.40");

    if (connect(sock, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) < 0) {
        std::cerr << "Draw: Connection failed." << std::endl;
    } else {
        std::cout << "Draw: Connected successfully." << std::endl;

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
    }
  
    close(sock);
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate and print the elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - starttime).count();
    std::cout << "Time taken to send image data: " << duration << " milliseconds." << std::endl;
	  
      return 0;
  }

  int32_t xlnx_kernel_done (VVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    return 0;
  }
}
