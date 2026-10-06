// One thread per output pixel/channel, with deterministic angular accumulation.
// Matches skimage.iradon(circle=False, interpolation="linear").
template <typename T>
__device__ void backproject_pixel(const double* filtered, const double* trig,
                                 T* output, int detectors, int angles,
                                 int side, int channels) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long pixels = (long long)side * side;
    if (index >= pixels * channels) return;
    const int channel = index / pixels;
    const int pixel = index % pixels;
    const double row = pixel / side - side / 2;
    const double col = pixel % side - side / 2;
    T sum = 0;
    for (int angle = 0; angle < angles; ++angle) {
        const double position = col * trig[2 * angle]
                              - row * trig[2 * angle + 1] + detectors / 2;
        double value = 0;
        // np.interp returns zero beyond either endpoint, with no edge blending.
        if (position >= 0 && position <= detectors - 1) {
            const int lower = __double2int_rd(position);
            const int upper = lower + 1 < detectors ? lower + 1 : lower;
            const long long base = ((long long)channel * angles + angle) * detectors;
            const double left = filtered[base + lower];
            value = left + (position - lower) * (filtered[base + upper] - left);
        }
        // CPU float32 iradon casts after each angle, rather than once at the end.
        sum = (T)((double)sum + value);
    }
    output[index] = (T)(sum * (T)3.14159265358979323846) / (T)(2 * angles);
}

extern "C" __global__ void fbp_float(const double* filtered, const double* trig,
                                    float* output, int detectors, int angles,
                                    int side, int channels) {
    backproject_pixel(filtered, trig, output, detectors, angles, side, channels);
}

extern "C" __global__ void fbp_double(const double* filtered, const double* trig,
                                     double* output, int detectors, int angles,
                                     int side, int channels) {
    backproject_pixel(filtered, trig, output, detectors, angles, side, channels);
}
