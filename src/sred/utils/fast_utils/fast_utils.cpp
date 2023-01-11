#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include "fast_utils.h"
#include "Image.h"
#include "inpainting.h"
#include <iostream>


void hello_test() {
	std::cout << "Hello, World!" << std::endl;
}


void register_rgb_api(
	uint8_t* rgb_data, uint32_t rgb_width, uint32_t rgb_height,
	uint16_t* d_data, uint32_t d_width, uint32_t d_height,
	const CamParams* params_ptr,
	uint8_t* out_data) {

	// input depth image
	SharedImage<uint16_t> d_img{ d_data, d_width, d_height };

	// input rgb image
	SharedImage<Pixel<uint8_t>> rgb_img{ rgb_data, rgb_width, rgb_height };

	// output registered rgb (rgb->d)
	SharedImage<Pixel<uint8_t>> reg_rgb{ out_data, d_width, d_height };

	register_rgb(rgb_img, d_img, *params_ptr, reg_rgb);
}

void register_d_api(
	uint8_t* rgb_data, uint32_t rgb_width, uint32_t rgb_height,
	uint16_t* d_data, uint32_t d_width, uint32_t d_height,
	const CamParams* params_ptr,
	uint16_t* out_data) {

	// input depth image
	SharedImage<uint16_t> d_img{ d_data, d_width, d_height };

	// input rgb image
	SharedImage<Pixel<uint8_t>> rgb_img{ rgb_data, rgb_width, rgb_height };

	// output registered rgb (rgb->d)
	SharedImage<uint16_t> reg_d{ out_data, rgb_width, rgb_height };

	register_d(rgb_img, d_img, *params_ptr, reg_d);
}

void hole_interpolation_api(
	uint16_t* d_data, uint32_t d_width, uint32_t d_height,
	uint16_t* out_data) {

	SharedImage<uint16_t> d_img{ d_data, d_width, d_height };
	SharedImage<uint16_t> out_img{ out_data, d_width, d_height };

	hole_interpolation(d_img, out_img);

}

void smoothen_filled_holes_api(
	uint16_t* d_data, uint16_t* filled_data, uint32_t d_width, uint32_t d_height,
	uint16_t* out_data) {

	SharedImage<uint16_t> d_img{ d_data, d_width, d_height };
	SharedImage<uint16_t> filled_img{ filled_data, d_width, d_height };
	SharedImage<uint16_t> out_img{ out_data, d_width, d_height };

	smoothen_filled_holes(d_img, filled_img, out_img);

}


void inpaint_FMM_api(
	double* img_data, uint32_t width, uint32_t height, uint8_t channels,
	uint8_t* mask_data,
	double radius,
	double* out_data) {

	
	SharedImage<uint8_t> mask{ mask_data, width, height };
	if (channels == 1) {
		SharedImage<double> img{ img_data, width, height };
		SharedImage<double> out{ out_data, width, height };
		inpaint_FMM(img, mask, radius, out);
	}
	else if (channels == 3) {
		SharedImage<Pixel<double>> img{ img_data, width, height };
		SharedImage<Pixel<double>> out{ out_data, width, height };
		inpaint_FMM(img, mask, radius, out);
	}
}

void inpaint_GFMM_api(
	double* img_data, uint32_t width, uint32_t height, uint8_t img_channels,
	uint8_t* mask_data,
	double* guide_data, uint8_t guide_channels,
	double radius,
	double* out_data) {

	
	SharedImage<uint8_t> mask{ mask_data, width, height };
	if (img_channels == 1 && guide_channels == 1) {
		SharedImage<double> img{ img_data, width, height };
		SharedImage<double> guide{ guide_data, width, height };
		SharedImage<double> out{ out_data, width, height };
		inpaint_GFMM(img, mask, guide, radius, out);
	} else if (img_channels == 1 && guide_channels == 3) {
		SharedImage<double> img{ img_data, width, height };
		SharedImage<Pixel<double>> guide{ guide_data, width, height };
		SharedImage<double> out{ out_data, width, height };
		inpaint_GFMM(img, mask, guide, radius, out);
	} else if (img_channels == 3 && guide_channels == 1) {
		SharedImage<Pixel<double>> img{ img_data, width, height };
		SharedImage<double> guide{ guide_data, width, height };
		SharedImage<Pixel<double>> out{ out_data, width, height };
		inpaint_GFMM(img, mask, guide, radius, out);
	} else if (img_channels == 3 && guide_channels == 3) {
		SharedImage<Pixel<double>> img{ img_data, width, height };
		SharedImage<Pixel<double>> guide{ guide_data, width, height };
		SharedImage<Pixel<double>> out{ out_data, width, height };
		inpaint_GFMM(img, mask, guide, radius, out);
	}
}

void cleanup_depth_api(
	uint16_t* img_data, uint32_t width, uint32_t height,
	uint32_t min_size,
	uint16_t* out_data) {

	SharedImage<uint16_t> img{ img_data, width, height };
	SharedImage<uint16_t> out{ out_data, width, height };
	cleanup_depth(img, min_size, out);
}
