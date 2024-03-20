#include <torch/extension.h>
#include "gSLICr_Lib/gSLICr.h"

void load_image(const int* inimg, gSLICr::UChar4Image* outimg)
{
    gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < outimg->noDims.y;y++)
        for (int x = 0; x < outimg->noDims.x; x++)
        {
            int idx = x + y * outimg->noDims.x;
            outimg_ptr[idx].b = inimg[idx];
            outimg_ptr[idx].g = inimg[idx];
            outimg_ptr[idx].r = inimg[idx];
        }
}

void load_mask(const gSLICr::IntImage* inimg, int* outimg)
{
    const int* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < inimg->noDims.y; y++)
        for (int x = 0; x < inimg->noDims.x; x++)
        {
            int idx = x + y * inimg->noDims.x;
            outimg[idx] = inimg_ptr[idx];
        }
}
void gslic(
    const at::Tensor img, 
    at::Tensor mask, 
    int width, 
    int height,
    int num_segment=2000
    )
{
    gSLICr::objects::settings my_settings;
	my_settings.img_size.x = width;
	my_settings.img_size.y = height;
	my_settings.no_segs = num_segment;
	my_settings.spixel_size = 16;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 10;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step

    // instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

    gSLICr::UChar4Image* inimg = new gSLICr::UChar4Image(my_settings.img_size, true, true);
    const gSLICr::IntImage* outimg = new gSLICr::IntImage(my_settings.img_size, true, true);

    load_image(img.data_ptr<int>(), inimg);
    gSLICr_engine->Process_Frame(inimg);
    outimg = gSLICr_engine->Get_Seg_Res();
    load_mask(outimg, mask.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gslic", &gslic, "gSLICr");
}