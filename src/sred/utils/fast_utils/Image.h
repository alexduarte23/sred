#pragma once

#include <iostream>


/*template<typename base_unit, uint32_t N = 3>
struct Pixel {
	base_unit values[N];


	const base_unit& operator[](uint32_t i) const {
		return values[i];
	}

	base_unit& operator[](uint32_t i) {
		return values[i];
	}

	friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
		os << "(";
		for (uint32_t i = 0; i < N-1; i++)
			os << p.values[i] << ", ";
		return os << p.values[N-1] << ")";
	}
};*/

template<typename base_unit>
struct Pixel {
	base_unit r, g, b;

	Pixel() : r{ 0 }, g{ 0 }, b{ 0 } {}

	Pixel(const base_unit& v0, const base_unit& v1, const base_unit& v2)
		: r{ v0 }, g{ v1 }, b{ v2 } {}

	Pixel(const base_unit& v0)
		: r{ v0 }, g{ v0 }, b{ v0 } {}


	Pixel<base_unit> operator+(const Pixel<base_unit>& p) const {
		return Pixel<base_unit>(r + p.r, g + p.g, b + p.b);
	}

	Pixel<base_unit>& operator+=(const Pixel<base_unit>& p) {
		r += p.r;
		g += p.g;
		b += p.b;
		return *this;
	}

	Pixel<base_unit> operator-(const Pixel<base_unit>& p) const {
		return Pixel<base_unit>(r - p.r, g - p.g, b - p.b);
	}

	Pixel<base_unit> operator*(const Pixel<base_unit>& p) const {
		return Pixel<base_unit>(r * p.r, g * p.g, b * p.b);
	}

	Pixel<base_unit> operator*(base_unit v) const {
		return Pixel<base_unit>(r * v, g * v, b * v);
	}

	Pixel<base_unit> operator/(base_unit v) const {
		return Pixel<base_unit>(r / v, g / v, b / v);
	}

	const base_unit& operator[](uint32_t i) const {
		return (&r)[i];
	}

	base_unit& operator[](uint32_t i) {
		return (&r)[i];
	}

	friend Pixel<base_unit> operator*(base_unit v, const Pixel<base_unit>& p) {
		return Pixel<base_unit>(p.r * v, p.g * v, p.b * v);
	}

	friend std::ostream& operator<<(std::ostream& os, const Pixel& p) {
		return os << "(" << p.r << ", " << p.g << ", " << p.b << ")";
	}
};

static double dot(double l, double r) {
	return l * r;
}

static double dot(const Pixel<double>& l, const Pixel<double>& r) {
	return l.r * r.r + l.g * r.g + l.b * r.b;
}

static double inner_sum(double v) {
	return v;
}

static double inner_sum(const Pixel<double>& v) {
	return v.r + v.g + v.b;
}


// just abstracts data, not responsible for allocating or freeing anything
/*template<typename pixel_type>
struct Image {
	pixel_type* data;
	unsigned int width, height;

	Image(void* arr, uint32_t w, uint32_t h)
		: data((pixel_type*)arr), width(w), height(h) {}

	Image(pixel_type* arr, uint32_t w, uint32_t h)
		: data(arr), width(w), height(h) {}

	bool inbounds(int i, int j) const {
		return i >= 0 && i < height&& j >= 0 && j < width;
	}

	const pixel_type& at(uint32_t i, uint32_t j) const {
		return data[i * width + j];
	}

	pixel_type& at(uint32_t i, uint32_t j) {
		return data[i * width + j];
	}
};*/


template<typename pixel_type>
struct Image {
	pixel_type* data;
	uint32_t width, height;

	virtual ~Image() {}

	bool inbounds(int i, int j) const {
		return i >= 0 && i < height&& j >= 0 && j < width;
	}

	const pixel_type& at(uint32_t i, uint32_t j) const {
		return data[i * width + j];
	}

	pixel_type& at(uint32_t i, uint32_t j) {
		return data[i * width + j];
	}

	const pixel_type& operator[](uint32_t i) const {
		return data[i];
	}

	pixel_type& operator[](uint32_t i) {
		return data[i];
	}

	void fill(const Image<pixel_type>& other) {
		uint32_t length = (std::min)(width * height, other.width * other.height);
		std::copy(other.data, other.data + length, data);
	}

	void fill(const pixel_type& pixel_value) {
		uint32_t length = width * height;
		std::fill(data, data + length, pixel_value);
	}

protected:
	Image() {}
	Image(const Image<pixel_type>&) = delete;
	Image(Image<pixel_type>&&) noexcept = delete;
};

// for images with externally owned data
template<typename pixel_type>
struct SharedImage : Image<pixel_type> {
	SharedImage(void* arr, uint32_t w, uint32_t h) {
		this->data = (pixel_type*)arr;
		this->width = w;
		this->height = h;
	}

	SharedImage(pixel_type* arr, uint32_t w, uint32_t h) {
		this->data = arr;
		this->width = w;
		this->height = h;
	}

};

// for images that own their data
template<typename pixel_type>
struct TrueImage : Image<pixel_type> {
	TrueImage(uint32_t w, uint32_t h) {
		this->data = new pixel_type[w * h];
		this->width = w;
		this->height = h;
	}

	~TrueImage() override {
		delete[] this->data;
	}
};



struct CamParams {
	double d_fx = 0, d_fy = 0, d_cx = 0, d_cy = 0, 
		rgb_fx = 0, rgb_fy = 0, rgb_cx = 0, rgb_cy = 0, 
		angle = 0, t_x = 0, t_y = 0;
};


void horizontal_hole_interpolation(const Image<uint16_t>& d_img, Image<uint16_t>& out_img);

void vertical_hole_interpolation(const Image<uint16_t>& d_img, Image<uint16_t>& out_img);

void hole_interpolation(const Image<uint16_t>& d_img, Image<uint16_t>& out_img);

void smoothen_filled_holes(const Image<uint16_t>& d_img, const Image<uint16_t>& filled_img, Image<uint16_t>& out_img);


// Applies transformation: R^-1 * (point - T)
void transform_point(double& x, double& y, double cos_angle, double sin_angle, double t_x, double t_y);

bool depth_occluded(const Image<Pixel<uint16_t>>& reg_d, uint32_t i, uint32_t j);

// align rgb image to depth image
void register_rgb(const Image<Pixel<uint8_t>>& rgb_img, const Image<uint16_t>& raw_d_img, const CamParams& params, Image<Pixel<uint8_t>>& reg_rgb);

// align d image to rgb image (sparse)
void register_d(const Image<Pixel<uint8_t>>& rgb_img, const Image<uint16_t>& raw_d_img, const CamParams& params, Image<uint16_t>& reg_d);