#pragma once


#include "Image.h"
#include <map>
#include <vector>

#define KNOWN 0
#define BAND 1
#define UNKNOWN 2
#define INF 1e6
#define EPS 1e-6

// solves a step of the eikonal equation in order to find closest quadrant
double solve_eikonal(int y1, int x1, int y2, int x2, const Image<double>& dists, const Image<uint8_t>& flags) {
    // check image frame
    if (!flags.inbounds(y1, x1) || !flags.inbounds(y2, x2))
        return INF;

    auto flag1 = flags.at(y1, x1), flag2 = flags.at(y2, x2);

    // both pixels are known
    if (flag1 == KNOWN && flag2 == KNOWN) {
        auto dist1 = dists.at(y1, x1), dist2 = dists.at(y2, x2);
        auto d = 2.0 - (dist1 - dist2) * (dist1 - dist2);
        if (d > 0) {
            auto r = std::sqrt(d);
            auto s = (dist1 + dist2 - r) / 2.0;
            if (s >= dist1 && s >= dist2) return s;
            s += r;
            if (s >= dist1 && s >= dist2) return s;
            // unsolvable
            return INF;
        }
    }

    // only 1st pixel is known
    if (flag1 == KNOWN) return 1.0 + dists.at(y1, x1);
    // only 2d pixel is known
    if (flag2 == KNOWN) return 1.0 + dists.at(y2, x2);
    // no pixel is known
    return INF;
}

template<typename pixel_type>
std::pair<pixel_type, pixel_type> discrete_gradient(int i, int j, const Image<pixel_type>& vals, const Image<uint8_t>& flags){
    std::pair<pixel_type, pixel_type> grad{ 0,0 };
    uint32_t prev = 0, next = 0;

    // horizontal gradient
    prev = i - 1 * (i - 1 >= 0 && flags.at(i - 1, j) != UNKNOWN);
    next = i + 1 * (i + 1 < flags.height && flags.at(i + 1, j) != UNKNOWN);
    grad.first = (vals.at(next, j) - vals.at(prev, j)) / std::fmax(next - prev, 1);

    // vertical gradient
    prev = j - 1 * (j - 1 >= 0 && flags.at(i, j - 1) != UNKNOWN);
    next = j + 1 * (j + 1 < flags.width && flags.at(i, j + 1) != UNKNOWN);
    grad.second = (vals.at(i, next) - vals.at(i, prev)) / std::fmax(next - prev, 1);

    return grad;
}

// compute distances between initial mask contour and pixels outside mask, using FMM (Fast Marching Method)
void compute_dists(Image<double>& dists, const Image<uint8_t>& orig_flags, std::multimap<double, std::pair<uint32_t, uint32_t>> band, bool outside, double radius) {
    uint32_t length = orig_flags.width * orig_flags.height;
    TrueImage<uint8_t> flags{ orig_flags.width, orig_flags.height };

    if (outside) {
        for (uint32_t i = 0; i < length; i++) { // swap INSIDE / OUTSIDE
            switch (orig_flags[i]) {
            case KNOWN: flags[i] = UNKNOWN; break;
            case UNKNOWN: flags[i] = KNOWN; break;
            default: flags[i] = BAND; break;
            }
        }
    } else {
        flags.fill(orig_flags); // copy
    }



    std::pair<int, int> offsets[] = { {-1,0}, {0,-1}, {1,0}, {0,1} };

    double last_dist = 0.0, double_radius = radius * 2;
    while (band.size() > 0) {
        // stop when neighborhood limit is reached
        if (outside && last_dist >= double_radius) break;

        // next band pixel closest to initial band
        auto band_point = band.begin()->second;
        band.erase(band.begin());
        flags.at(band_point.first, band_point.second) = KNOWN;

        // process pixels next to band point
        int i, j;
        double dst;
        for (const auto& off : offsets) {
            i = (int)band_point.first + off.first;
            j = (int)band_point.second + off.second;
            if (!flags.inbounds(i, j) || flags.at(i, j) != UNKNOWN)
                continue;
            
            dst = std::fmin(std::fmin(std::fmin(
                solve_eikonal(i - 1, j, i, j - 1, dists, flags),
                solve_eikonal(i + 1, j, i, j + 1, dists, flags)),
                solve_eikonal(i - 1, j, i, j + 1, dists, flags)),
                solve_eikonal(i + 1, j, i, j - 1, dists, flags)
            );
            dists.at(i, j) = dst;
            last_dist = dst;

            // set as new band point (band shrinking)
            flags.at(i, j) = BAND;
            band.emplace(dst, std::make_pair((uint32_t)i, (uint32_t)j));
        }
    }

    //delete[] f_array_copy;

    if (!outside) return;
    
    // outside dists should be negative
    for (uint32_t i = 0; i < length; i++) {
        if (orig_flags[i] == KNOWN && dists[i] < INF)
            dists[i] = -dists[i];
    }
}

// computes pixels distances to initial mask contour, flags, and narrow band queue
void init_inpainting(const Image<uint8_t>& mask, double radius, Image<double>& dists, Image<uint8_t>& flags, std::multimap<double, std::pair<uint32_t, uint32_t>>& band) {
    uint32_t length = dists.width * dists.height;
    for (uint32_t i = 0; i < length; i++) {
        dists[i] = INF;
        flags[i] = mask[i] == 0 ? KNOWN : UNKNOWN;
    }
    
    std::pair<int, int> offsets[] = { {-1,0}, {0,-1}, {1,0}, {0,1} };
    //holes = np.argwhere(mask == True)
    int k, l;
    for (int i = 0; i < dists.height; i++) {
        for (int j = 0; j < dists.width; j++) {
            if (mask.at(i, j) == 0) continue;
            for (const auto& off : offsets) {
                k = i + off.first;
                l = j + off.second;
                if (!flags.inbounds(k, l) || flags.at(k, l) != KNOWN)
                    continue;
                flags.at(k, l) = BAND;
                dists.at(k, l) = 0;
                band.emplace(0, std::make_pair((uint32_t)k, (uint32_t)l));
            }
        }
    }

    // compute distance to inital mask contour for KNOWN pixels
    // (by inverting mask/flags and running FFM)
    compute_dists(dists, flags, band, true, radius); // outside
    compute_dists(dists, flags, band, false, radius); // inside
}


// returns RGB values for pixel to by inpainted, computed for its neighborhood
template<typename img_type>
img_type inpaint_pixel_FMM(uint32_t i, uint32_t j, const Image<img_type>& img, const Image<double>& dists, const Image<uint8_t>& flags, double radius) {
    auto dist = dists.at(i, j);
    // normal to pixel, ie direction of propagation of the FFM
    auto dist_grad = discrete_gradient(i, j, dists, flags);

    img_type sum_wI = 0;
    double sum_w = 0;

    double dir_i, dir_j, dir_length;
    double w_dir, w_dst, w_lev, confidence, weight;
    for (int k = i - radius; k <= i + radius; k++) {
        for (int l = j - radius; l <= j + radius; l++) {
            // skip invalid and unknown neightbors
            if (!flags.inbounds(k, l) || flags.at(k, l) == UNKNOWN)
                continue;

            // vector q -> p
            dir_i = (int)i - k;
            dir_j = (int)j - l;
            dir_length = std::sqrt(dir_i * dir_i + dir_j * dir_j);
            if (dir_length > radius) continue;

            w_dir = (dir_i * dist_grad.first + dir_j * dist_grad.second) / dir_length;
            if (w_dir == 0) w_dir = EPS;
            //w_dir = 1;
            w_dst = 1 / (dir_length * dir_length);
            w_lev = 1 / (1 + std::abs(dists.at(k, l) - dist));
            
            //confidence = 1.0 / (1 + 2 * std::fmax(0, dists.at(k, l)));
            confidence = 1 + (dists.at(k, l) <= 0);
            weight = w_dir * w_dst * w_lev * confidence;
            //if (weight <= 0) weight = EPS;
            weight = std::fabs(weight);

            sum_wI += weight * img.at(k, l);
            sum_w += weight;
        }
    }

    return sum_wI / sum_w;
}

template<typename img_type, typename guide_type>
std::pair<img_type, double> inpaint_pixel_guided_FMM(uint32_t i, uint32_t j, const Image<img_type>& img, const Image<double>& dists, const Image<uint8_t>& flags, const Image<guide_type>& guide, double guide_var, double radius) {
    auto dist = dists.at(i, j);
    // normal to pixel, ie direction of propagation of the FFM
    //auto dist_grad = discrete_gradient(i, j, dists, flags);

    img_type sum_wI = 0;
    double sum_w = 0;
    double num_neighbors = 0, sum_guide_sim = 0;

    double dir_i, dir_j, dir_length;
    double w_dir, w_dst, w_lev, w_guide, confidence, weight;
    guide_type guide_diff;
    double guide_sim;
    for (int k = i - radius; k <= i + radius; k++) {
        for (int l = j - radius; l <= j + radius; l++) {
            // skip invalid neightbors
            if (!flags.inbounds(k, l))
                continue;

            // vector q -> p
            dir_i = (int)i - k;
            dir_j = (int)j - l;
            dir_length = std::sqrt(dir_i * dir_i + dir_j * dir_j);
            if (dir_length > radius) continue;

            // compute guide weigth and accumulate value for propagation order
            guide_diff = guide.at(k, l) - guide.at(i, j);
            //guide_sim = std::exp(-5 * dot(guide_diff, guide_diff) / (2 * guide_var));
            guide_sim = std::exp(-5* dot(guide_diff, guide_diff) / (2 * guide_var));
            //guide_sim = 1 - np.sum(abs(guide[k, l] - guide[i, j])) / (guide.shape[-1] * guide_max)
            sum_guide_sim += guide_sim;
            num_neighbors += 1;

            if (flags.at(k, l) == UNKNOWN)
                continue;

            //w_dir = (dir_i * dist_grad.first + dir_j * dist_grad.second) / dir_length;
            //if (w_dir == 0) w_dir = EPS;
            w_dst = 1 / (dir_length * dir_length);
            //w_lev = 1 / (1 + std::abs(dists.at(k, l) - dist));
            w_guide = guide_sim;

            confidence = 1.0 / (1 + 2*std::fmax(0, dists.at(k, l)));
            //confidence = 1 + 5*(dists.at(k, l) <= 0);
            weight = w_dst * w_dst * w_guide * confidence;
            //weight = w_dst * w_guide * confidence;
            //weight = w_dir * w_lev * w_dst * w_guide * confidence;
            if (weight <= 0) weight = EPS;
            //weight = std::fabs(weight);

            sum_wI += weight * img.at(k, l);
            sum_w += weight;
        }
    }

    return { sum_wI / sum_w, sum_guide_sim / num_neighbors };
}



template<typename img_type>
void inpaint_FMM(const Image<img_type>& img, const Image<uint8_t>& mask, double radius, Image<img_type>& out_img) {
    TrueImage<double> dists{ img.width, img.height };
    TrueImage<uint8_t> flags{ img.width, img.height };
    std::multimap<double, std::pair<uint32_t, uint32_t>> band;
   
    init_inpainting(mask, radius, dists, flags, band);
    
    out_img.fill(img);

    // FMM

    std::pair<int, int> offsets[] = { {-1,0}, {0,-1}, {1,0}, {0,1} };
    while (band.size() > 0) {
        // next band pixel closest to initial band
        auto band_point = band.begin()->second;
        band.erase(band.begin());
        flags.at(band_point.first, band_point.second) = KNOWN;

        // process pixels next to band point
        int i, j;
        for (const auto& off : offsets) {
            i = band_point.first + off.first;
            j = band_point.second + off.second;
            if (!flags.inbounds(i, j) || flags.at(i, j) != UNKNOWN)
                continue;
            
            out_img.at(i, j) = inpaint_pixel_FMM(i, j, out_img, dists, flags, radius);

            // set as new band point (band shrinking)
            flags.at(i, j) = BAND;
            band.emplace(dists.at(i, j), std::make_pair((uint32_t)i, (uint32_t)j));
        }
    }
}

template<typename img_type, typename guide_type>
void inpaint_GFMM(const Image<img_type>& img, const Image<uint8_t>& mask, const Image<guide_type>& guide, double radius, Image<img_type>& out_img) {
    TrueImage<double> dists{ img.width, img.height };
    TrueImage<uint8_t> flags{ img.width, img.height };
    std::multimap<double, std::pair<uint32_t, uint32_t>> band;

    init_inpainting(mask, radius, dists, flags, band);

    uint32_t f_length = img.width * img.height;

    // compute guide mean
    guide_type guide_avg = 0;
    for (uint32_t i = 0; i < f_length; i++) {
        guide_avg += guide[i];
    }
    guide_avg = guide_avg / f_length;

    // compute guide variance
    guide_type guide_var_sum = 0;
    double guide_var = 0;
    for (uint32_t i = 0; i < f_length; i++) {
        auto diff = guide[i] - guide_avg;
        guide_var_sum += diff * diff; // el-wise mult
    }
    guide_var = inner_sum(guide_var_sum) / (f_length - 1);

    // compute max dist
    double max_dist = 0;
    for (uint32_t i = 0; i < f_length; i++) {
        const auto& val = dists[i];
        if (val < INF && val > max_dist)
            max_dist = val;
    }

    out_img.fill(img);

    // GFMM

    std::pair<int, int> offsets[] = { {-1,0}, {0,-1}, {1,0}, {0,1} };
    double guide_sim, priority;
    while (band.size() > 0) {
        // next band pixel closest to initial band
        auto band_point = band.begin()->second;
        band.erase(band.begin());
        flags.at(band_point.first, band_point.second) = KNOWN;

        // process pixels next to band point
        int i, j;
        for (const auto& off : offsets) {
            i = (int)band_point.first + off.first;
            j = (int)band_point.second + off.second;
            if (!flags.inbounds(i, j) || flags.at(i, j) != UNKNOWN)
                continue;

            auto ret = inpaint_pixel_guided_FMM(i, j, out_img, dists, flags, guide, guide_var, radius);
            out_img.at(i, j) = ret.first;
            guide_sim = ret.second;

            // set as new band point (band shrinking)
            flags.at(i, j) = BAND;
            priority = 0.1 * dists.at(i, j) / max_dist + 0.9 * (1 - guide_sim);
            band.emplace(priority, std::make_pair((uint32_t)i, (uint32_t)j));
        }
    }

}


void cleanup_depth_region(int start_i, int start_j, Image<uint16_t>& img, Image<bool>& seen, uint32_t min_size, std::vector<std::pair<int, int>>& region_arr) {
    seen.at(start_i, start_j) = true;
    
    region_arr[0] = { start_i, start_j };
    uint32_t pos = 0, total = 1;

    std::pair<int, int> offsets[] = { {-1,0}, {0,-1}, {1,0}, {0,1} };
    int i, j, k, l;
    while (pos < total) {
        i = region_arr[pos].first;
        j = region_arr[pos].second;
        
        for (const auto& off : offsets) {
            k = i + off.first;
            l = j + off.second;
            if (img.inbounds(k, l) && !seen.at(k, l)) {
                seen.at(k, l) = true;
                region_arr[total++] = { k, l };
            }
        }

        pos++;
    }
    
    // delete region if too small
    if (total < min_size) {
        for (pos = 0; pos < total; pos++) {
            i = region_arr[pos].first;
            j = region_arr[pos].second;
            img.at(i, j) = 0;
        }
    }
}

void cleanup_depth(const Image<uint16_t>& img, uint32_t min_size, Image<uint16_t>& out) {
    out.fill(img);
    TrueImage<bool> seen{ img.width, img.height };

    uint32_t f_length = img.width * img.height;
    for (uint32_t i = 0; i < f_length; i++) {
        seen[i] = (img[i] == 0);
    }

    std::vector<std::pair<int, int>> region_arr(f_length);

    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            if (!seen.at(i, j))
                cleanup_depth_region(i, j, out, seen, min_size, region_arr);
        }
    }

}
