#include "pch.h"
#include "Image.h"
#include <cmath>
#include <algorithm>


void horizontal_hole_interpolation(const Image<uint16_t>& d_img, Image<uint16_t>& out_img) {
    for (uint32_t i = 0; i < d_img.height; i++) {
        int j_start = 0, j_end = 0;
        bool const_fill = false;
        uint16_t fill_val = 0;

        for (uint32_t j = 0; j < d_img.width; j++) {
            if (d_img.at(i, j) != 0) {
                // assumes out_img is empty
                out_img.at(i, j) = d_img.at(i, j);
                continue;
            }

            // set j_start and j_end (and check/set constant fill)
            if (j_end <= j) {
                j_start = j - 1;
                j_end = d_img.width;
                for (uint32_t k = j + 1; k < d_img.width; k++) {
                    if (d_img.at(i, k) != 0) {
                        j_end = k;
                        break;
                    }
                }

                if (j_start < 0 and j_end >= d_img.width) {
                    fill_val = 30000;
                    const_fill = true;
                }
                else if (j_start < 0) {
                    fill_val = d_img.at(i, j_end);
                    const_fill = true;
                }
                else if (j_end >= d_img.width) {
                    fill_val = d_img.at(i, j_start);
                    const_fill = true;
                }
                else {
                    const_fill = false;
                }
            }

            // fill/interpolate
            if (const_fill) {
                out_img.at(i, j) = fill_val;
            }
            else {
                double frac = ((double)j - j_start) / (j_end - j_start);
                out_img.at(i, j) = frac * d_img.at(i, j_end) + (1 - frac) * d_img.at(i, j_start);
            }
        }
    }
}

void vertical_hole_interpolation(const Image<uint16_t>& d_img, Image<uint16_t>& out_img) {
    for (uint32_t j = 0; j < d_img.width; j++) {
        int i_start = 0, i_end = 0;
        bool const_fill = false;
        uint16_t fill_val = 0;

        for (uint32_t i = 0; i < d_img.height; i++) {
            if (d_img.at(i, j) != 0) continue;

            // set i_start and i_end (and check/set constant fill)
            if (i_end <= i) {
                i_start = i - 1;
                i_end = d_img.height;
                for (uint32_t k = i + 1; k < d_img.height; k++) {
                    if (d_img.at(k, j) != 0) {
                        i_end = k;
                        break;
                    }
                }

                if (i_start < 0 and i_end >= d_img.height) {
                    fill_val = 30000;
                    const_fill = true;
                }
                else if (i_start < 0) {
                    fill_val = d_img.at(i_end, j);
                    const_fill = true;
                }
                else if (i_end >= d_img.height) {
                    fill_val = d_img.at(i_start, j);
                    const_fill = true;
                }
                else {
                    const_fill = false;
                }
            }

            // fill/interpolate
            if (const_fill) {
                out_img.at(i, j) = out_img.at(i, j) * 0.5 + fill_val * 0.5;
            }
            else {
                double frac = ((double)i - i_start) / (i_end - i_start);
                out_img.at(i, j) = out_img.at(i, j) * 0.5
                    + (frac * d_img.at(i_end, j) + (1.0 - frac) * d_img.at(i_start, j)) * 0.5;
            }
        }
    }
}

void hole_interpolation(const Image<uint16_t>& d_img, Image<uint16_t>& out_img) {
    horizontal_hole_interpolation(d_img, out_img);
    vertical_hole_interpolation(d_img, out_img);
}


void smoothen_filled_holes(const Image<uint16_t>& d_img, const Image<uint16_t>& filled_img, Image<uint16_t>& out_img) {
    int kernel_size = 7; // 9
    int hs = kernel_size / 2; // half size
    int left, right, top, bottom;
    double sum;

    for (int i = 0; i < d_img.height; i++) {
        for (int j = 0; j < d_img.width; j++) {
            if (d_img.at(i, j) != 0) {
                // assume out_img is empty
                out_img.at(i, j) = d_img.at(i, j);
                continue;
            }

            left = std::fmax(0, j - hs);
            right = std::fmin(j + hs, d_img.width - 1);
            top = std::fmax(0, i - hs);
            bottom = std::fmin(i + hs, d_img.height - 1);
            sum = 0;
            for (int k = top; k <= bottom; k++) {
                for (int l = left; l <= right; l++) {
                    sum += filled_img.at(k, l);
                }
            }
            out_img.at(i, j) = sum / ((right - left + 1) * (bottom - top + 1));
        }
    }
}

/*void smoothen_filled_holes(const Image<uint16_t>& d_img, const Image<uint16_t>& filled_img, Image<uint16_t>& out_img) {
    int kernel_size = 7; // 9
    int hs = kernel_size / 2; // half size
    int left, right, top, bottom;
    double sum;

    for (int i = 0; i < d_img.height; i++) {
        for (int j = 0; j < d_img.width; j++) {
            if (d_img.at(i, j) != 0) {
                // assume out_img is empty
                out_img.at(i, j) = d_img.at(i, j);
                continue;
            }

            top = std::fmax(0, i - hs);
            bottom = std::fmin(i + hs, d_img.height - 1);
            sum = 0;
            for (int k = top; k <= bottom; k++) {
                sum += filled_img.at(k, j);
            }
            out_img.at(i, j) = sum / (bottom - top + 1);
        }
    }
}*/



void transform_point(double& x, double& y, double cos_angle, double sin_angle, double t_x, double t_y) {
    // R = [[cos -sin 0],[sin cos 0],[0 0 1]]
    // R^-1 = [[cos sin 0],[-sin cos 0],[0 0 1]]
    double old_x = x, old_y = y;
    x = (old_x - t_x) * cos_angle + (old_y - t_y) * sin_angle;
    y = (old_x - t_x) * -sin_angle + (old_y - t_y) * cos_angle;
}

bool depth_occluded(const Image<Pixel<uint16_t>>& reg_d, uint32_t i, uint32_t j) {
    int left = std::fmax(0, j - 1);
    int right = std::fmin(j + 1, reg_d.width - 1);
    int top = std::fmax(0, i - 1);
    int bottom = std::fmin(i + 1, reg_d.height - 1);

    int dist = 0;
    int d = reg_d.at(i, j)[0], d_i = reg_d.at(i, j)[1], d_j = reg_d.at(i, j)[2];
    for (uint32_t k = top; k <= bottom; k++) {
        for (uint32_t l = left; l <= right; l++) {
            if (reg_d.at(k, l)[0] == 0 || reg_d.at(k, l)[0] > d)
                continue;
            dist = std::abs((int)reg_d.at(k, l)[1] - d_i) + std::abs((int)reg_d.at(k, l)[2] - d_j);
            if (dist > 2)
                return true;
        }
    }

    return false;
}

// align rgb image to depth image
void register_rgb(const Image<Pixel<uint8_t>>& rgb_img, const Image<uint16_t>& raw_d_img, const CamParams& params, Image<Pixel<uint8_t>>& reg_rgb) {
    //auto inpainted = hole_interpolation(raw_d_img);
    //auto d_img = smoothen_filled_holes(raw_d_img, inpainted);
    const auto& d_img = raw_d_img;

    // Also used to stored positions for occlusion:
    //    red: depth, green: i in d_img, blue: j in d_img
    //auto reg_d_data = new Pixel<uint16_t>[rgb_img.width * rgb_img.height];
    //std::fill((uint16_t*)reg_d_data, (uint16_t*)reg_d_data + (rgb_img.width * rgb_img.height * 3), 0);
    //std::fill(reg_d_data, reg_d_data + (rgb_img.width * rgb_img.height), Pixel<uint16_t>{0,0,0});
    //Image<Pixel<uint16_t>> reg_d{ reg_d_data, rgb_img.width, rgb_img.height };
    TrueImage<Pixel<uint16_t>> reg_d(rgb_img.width, rgb_img.height);
    reg_d.fill(0);
    
    double sin_angle = std::sin(params.angle);
    double cos_angle = std::cos(params.angle);

    uint16_t last_seen = 30000;
    // create sparse registered depth (d->rgb)
    for (uint32_t i = 0; i < d_img.height; i++) {
        for (uint32_t j = 0; j < d_img.width; j++) {
            double y = i, x = j, d = d_img.at(i, j);
            if (d == 0) d = last_seen;
            else        last_seen = d;

            // convert to depth camera coords
            x = (x - params.d_cx) / params.d_fx * d;
            y = (y - params.d_cy) / params.d_fy * d;

            // convert to rgb camera space
            transform_point(x, y, cos_angle, sin_angle, params.t_x, params.t_y);

            // convert to rgb image coords
            x = std::round((x * params.rgb_fx / d) + params.rgb_cx);
            y = std::round((y * params.rgb_fy / d) + params.rgb_cy);

            // assign depth (and depth pos) to color
            if (reg_d.inbounds(y, x) && (reg_d.at(y, x)[0] == 0 || reg_d.at(y, x)[0] > d)) {
            //if (inbounds(reg_d, y, x) && (reg_d.at(y, x).red == 0 || reg_d.at(y, x).blue < j)) {
                //reg_d.at(y, x) = { (png::uint_16)d, (png::uint_16)i, (png::uint_16)j };
                reg_d.at(y, x) = { (uint16_t)d, (uint16_t)i, (uint16_t)j };
            }
        }
    }
    
    // create registered rgb (rgb->d) using registered depth
    for (uint32_t i = 0; i < reg_d.height; i++) {
        for (uint32_t j = 0; j < reg_d.width; j++) {
            uint16_t d = reg_d.at(i, j)[0];
            uint16_t i_d = reg_d.at(i, j)[1];
            uint16_t j_d = reg_d.at(i, j)[2];
            if (d > 0 && rgb_img.inbounds(i, j) && d >= d_img.at(i_d, j_d) && !depth_occluded(reg_d, i, j)) {
                reg_rgb.at(i_d, j_d) = rgb_img.at(i, j);
            }
        }
    }
    
    //delete[] reg_d_data;
}



void register_d(const Image<Pixel<uint8_t>>& rgb_img, const Image<uint16_t>& raw_d_img, const CamParams& params, Image<uint16_t>& reg_d) {
    //auto inpainted = hole_interpolation(raw_d_img);
    //auto d_img = smoothen_filled_holes(raw_d_img, inpainted);
    const auto& d_img = raw_d_img;

    double sin_angle = std::sin(params.angle);
    double cos_angle = std::cos(params.angle);

    // create sparse registered depth (d->rgb)
    for (uint32_t i = 0; i < d_img.height; i++) {
        for (uint32_t j = 0; j < d_img.width; j++) {
            double y = i, x = j;
            uint16_t d = d_img.at(i, j);
            if (d == 0) d = 1;

            // convert to depth camera coords
            x = (x - params.d_cx) / params.d_fx * d;
            y = (y - params.d_cy) / params.d_fy * d;

            // convert to rgb camera space
            transform_point(x, y, cos_angle, sin_angle, params.t_x, params.t_y);

            // convert to rgb image coords
            x = std::round((x * params.rgb_fx / d) + params.rgb_cx);
            y = std::round((y * params.rgb_fy / d) + params.rgb_cy);

            // assign depth to color
            if (reg_d.inbounds(y, x) && (reg_d.at(y, x) == 0 || reg_d.at(y, x) > d)) {
                reg_d.at(y, x) = d;
            }
        }
    }
}