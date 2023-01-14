#pragma once

#if defined(_WIN32) || defined(WIN32) || defined(__WIN32__)
#	ifdef FASTUTILS_EXPORTS
#		define FASTUTILS_API __declspec(dllexport)
#	else
#		define FASTUTILS_API __declspec(dllimport)
#	endif
#else
#	define FASTUTILS_API
#endif

#include <cstdint>

// forward decl
struct CamParams;


extern "C" FASTUTILS_API void hello_test();

extern "C" FASTUTILS_API void register_rgb_api(
	uint8_t* rgb_data, uint32_t rgb_width, uint32_t rgb_height,
	uint16_t* d_data, uint32_t d_width, uint32_t d_height,
	const CamParams * params_ptr,
	uint8_t* out_data);

extern "C" FASTUTILS_API void register_d_api(
	uint8_t * rgb_data, uint32_t rgb_width, uint32_t rgb_height,
	uint16_t * d_data, uint32_t d_width, uint32_t d_height,
	const CamParams * params_ptr,
	uint16_t * out_data);

extern "C" FASTUTILS_API void hole_interpolation_api(
	uint16_t* d_data, uint32_t d_width, uint32_t d_height, 
	uint16_t* out_data);

extern "C" FASTUTILS_API void smoothen_filled_holes_api(
	uint16_t * d_data, uint16_t * filled_data, uint32_t d_width, uint32_t d_height,
	uint16_t * out_data);


// only image of doubles with either 1 or 3 channels for simplicity
extern "C" FASTUTILS_API void inpaint_FMM_api(
	double* img_data, uint32_t width, uint32_t height, uint8_t channels,
	uint8_t * mask_data,
	double radius,
	double* out_data);

// only image of doubles with either 1 or 3 channels for simplicity
extern "C" FASTUTILS_API void inpaint_GFMM_api(
	double* img_data, uint32_t width, uint32_t height, uint8_t img_channels,
	uint8_t * mask_data,
	double* guide_data, uint8_t guide_channels,
	double radius,
	double* out_data);


extern "C" FASTUTILS_API void cleanup_depth_api(
	uint16_t* img_data, uint32_t width, uint32_t height,
	uint32_t min_size,
	uint16_t* out_data);