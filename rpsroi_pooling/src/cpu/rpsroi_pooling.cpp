#include <torch/extension.h>
#include <algorithm>

using namespace std;

template <typename T>
void RPSRoIPoolForward(
    const T* bottom_data, // input
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int group_size, // =pooled_height=pooled_width=2
    const int output_dim, // =1
    const T* bottom_rois, // rois
    const int num_rois, // added
    T* top_data) { // output score for Bin[i] = (P/C)[i] 
    //int* mapping_channel, // channel = 1
    //float* areas) { // C[i]
  for (int n = 0; n < num_rois; ++n) {
    // (n, c, ph, pw) is an element in the pooled output

    bottom_rois += n * 9;
    int roi_batch_ind = bottom_rois[0];
    
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        
        int idx = (n*pooled_height + ph)*pooled_width + pw;

        float roi_x1 = static_cast<float>(round(bottom_rois[1])) * spatial_scale;
        float roi_y1 = static_cast<float>(round(bottom_rois[2])) * spatial_scale;
        float roi_x2 = static_cast<float>(round(bottom_rois[3])) * spatial_scale;
        float roi_y2 = static_cast<float>(round(bottom_rois[4])) * spatial_scale;
        float roi_x3 = static_cast<float>(round(bottom_rois[5])) * spatial_scale;
        float roi_y3 = static_cast<float>(round(bottom_rois[6])) * spatial_scale;
        float roi_x4 = static_cast<float>(round(bottom_rois[7])) * spatial_scale;
        float roi_y4 = static_cast<float>(round(bottom_rois[8])) * spatial_scale;

        ////////////////////////////////DEBUG////////////////////////////////////
        //cout << "rois: " << roi_x1 << " " << roi_y1 << " " << roi_x2 << " " << roi_y2 << " " << roi_x3 << " " << roi_y3 << " " << roi_x4 << " " << roi_y4 << endl;
        //printf("rois: %f, %f, %f, %f, %f, %f, %f, %f\n", roi_x1, roi_y1, roi_x2, roi_y2, roi_x3, roi_y3, roi_x4, roi_y4);

        float anchor_x1 = static_cast<float>(pw) * (roi_x2 - roi_x1) / pooled_width + roi_x1;
        float anchor_y1 = static_cast<float>(pw) * (roi_y2 - roi_y1) / pooled_width + roi_y1;
        float anchor_x2 = static_cast<float>(pw+1) * (roi_x2 - roi_x1) / pooled_width + roi_x1;
        float anchor_y2 = static_cast<float>(pw+1) * (roi_y2 - roi_y1) / pooled_width + roi_y1;
        float anchor_x3 = static_cast<float>(pw+1) * (roi_x3 - roi_x4) / pooled_width + roi_x4;
        float anchor_y3 = static_cast<float>(pw+1) * (roi_y3 - roi_y4) / pooled_width + roi_y4;
        float anchor_x4 = static_cast<float>(pw) * (roi_x3 - roi_x4) / pooled_width + roi_x4;
        float anchor_y4 = static_cast<float>(pw) * (roi_y3 - roi_y4) / pooled_width + roi_y4;

        ////////////////////////////////DEBUG////////////////////////////////////
        //cout << "anchor: " << anchor_x1 << " " << anchor_y1 << " " << anchor_x2 << " " << anchor_y2 << " " << anchor_x3 << " " << anchor_y3 << " " << anchor_x4 << " " << anchor_y4 <<endl;
        //printf("anchor: %f, %f, %f, %f, %f, %f, %f, %f\n", anchor_x1, anchor_y1, anchor_x2, anchor_y2, anchor_x3, anchor_y3, anchor_x4, anchor_y4);

        float grid_x1 = static_cast<float>(ph) * (anchor_x4 - anchor_x1) / pooled_height + anchor_x1;
        float grid_y1 = static_cast<float>(ph) * (anchor_y4 - anchor_y1) / pooled_height + anchor_y1;
        float grid_x4 = static_cast<float>(ph + 1) * (anchor_x4 - anchor_x1) / pooled_height + anchor_x1;
        float grid_y4 = static_cast<float>(ph + 1) * (anchor_y4 - anchor_y1) / pooled_height + anchor_y1;
        float grid_x2 = static_cast<float>(ph) * (anchor_x3 - anchor_x2) / pooled_height + anchor_x2;
        float grid_y2 = static_cast<float>(ph) * (anchor_y3 - anchor_y2) / pooled_height + anchor_y2;
        float grid_x3 = static_cast<float>(ph + 1) * (anchor_x3 - anchor_x2) / pooled_height + anchor_x2;
        float grid_y3 = static_cast<float>(ph + 1) * (anchor_y3 - anchor_y2) / pooled_height + anchor_y2;

        ////////////////////////////////DEBUG////////////////////////////////////
        //cout << "grid: " << grid_x1 << " " << grid_y1 << " " << grid_x2 << " " << grid_y2 << " " << grid_x3 << " " << grid_y3 << " " << grid_x4 << " " << grid_y4 << endl;
        // printf("grid: %f, %f, %f, %f, %f, %f, %f, %f\n", grid_x1, grid_y1, grid_x2, grid_y2, grid_x3, grid_y3, grid_x4, grid_y4);

        //printf("min:%f, %f, %f\n", grid_y1, grid_y2,  min(grid_y1, grid_y2));
        //printf("min_grid:%f, %f, %f\n", grid_y1, grid_y2, floor(min(grid_y1, grid_y2)));
        
        int hstart = static_cast<int>(floor(min(min(min(grid_y1, grid_y2) , grid_y3), grid_y4)));
        int hend = static_cast<int>(ceil(max(max(max(grid_y1, grid_y2) , grid_y3), grid_y4)));
        int wstart = static_cast<int>(floor(min(min(min(grid_x1, grid_x2) , grid_x3), grid_x4)));
        int wend = static_cast<int>(ceil(max(max(max(grid_x1, grid_x2) , grid_x3), grid_x4)));

        ///////////////////////////////DEBUG/////////////////////////////////////
        //cout << "start&&end: " << hstart << " " << hend << " " << wstart << " " << wend << endl;
        //printf("start&&end: %d, %d, %d, %d\n", hstart, hend, wstart, wend);
        
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
      	hend = min(max(hend, 0), height);
      	wstart = min(max(wstart, 0), width);
      	wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        /////////////////////////////////////////////////////////////////////
        //cout << "start&&end norm: " << hstart << " " << hend << " " << wstart << " " << wend;
        //printf("start&&end norm: %d, %d, %d, %d\n", hstart, hend, wstart, wend);

        int gw = pw;
      	int gh = ph;
      	int c = (0*group_size + gh)*group_size + gw;
        // printf("c:%d %d %d %d\n", c, channels, height, width);

        bottom_data += (roi_batch_ind * channels + c) * height * width;

        //printf("get value: %d, %d, %d, %f\n", c, 270, 765, bottom_data[270*width + 765]);
        float out_sum = 0;
        float bin_area = 0;
      	for (int h = hstart; h < hend; ++h) {
      	  for (int w = wstart; w < wend; ++w) {
      	    int bottom_index = h*width + w;
            float p1 = (grid_x2 - grid_x1) * (h - grid_y1) - (w - grid_x1) * (grid_y2 - grid_y1);
            float p2 = (grid_x3 - grid_x2) * (h - grid_y2) - (w - grid_x2) * (grid_y3 - grid_y2);
            float p3 = (grid_x4 - grid_x3) * (h - grid_y3) - (w - grid_x3) * (grid_y4 - grid_y3);
            float p4 = (grid_x1 - grid_x4) * (h - grid_y4) - (w - grid_x4) * (grid_y1 - grid_y4);
            if(p1 >= 0 && p2 >= 0 && p3 >= 0 && p4 >= 0){
              out_sum += bottom_data[bottom_index];
              bin_area += 1;
            }
      	  }
      	}

        /////////////////////////////DEBUG//////////////////////////
        //cout << "bin_area: " << bin_area <<" out_sum: " << out_sum << endl;
        //printf("bin_area: %f, out_sum: %f\n", bin_area, out_sum);
      	top_data[idx] = (is_empty || (bin_area ==0)) ? 0. : out_sum/bin_area;
      	//mapping_channel[index] = c;
        //areas[index] = bin_area;
      
      }
    }
  }
}


at::Tensor RPSRoIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int group_size,
    const int output_dim,
    const int pooled_height,
    const int pooled_width) {
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
  AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "RPSROIPool_forward_cpu";
  at::checkAllSameType(c, {input_t, rois_t});

  int num_rois = rois.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "RPSROIPool_forward", [&] {
    RPSRoIPoolForward<scalar_t>(
        input.contiguous().data<scalar_t>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        group_size,
        output_dim,
        rois.contiguous().data<scalar_t>(),
        num_rois,
        output.data<scalar_t>());
  });
  return output;
}

PYBIND11_MODULE(/*TORCH_EXTENSION_NAME*/ "RPSRoIPool_forward_cpu", m) {
  m.def("forward", &RPSRoIPool_forward_cpu, "LLTM forward");
}
