//=============================================================================
//
// nldiffusion_functions.cpp
// Author: Pablo F. Alcantarilla
// Institution: University d'Auvergne
// Address: Clermont Ferrand, France
// Date: 27/12/2011
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2012, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file nldiffusion_functions.cpp
 * @brief Functions for non-linear diffusion applications:
 * 2D Gaussian Derivatives
 * Perona and Malik conductivity equations
 * Perona and Malik evolution
 * @date Dec 27, 2011
 * @author Pablo F. Alcantarilla
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "nldiffusion_functions.h"
#include <algorithm>
#include <iostream>

// Namespaces

/* ************************************************************************* */

namespace cv
{
using namespace std;

/* ************************************************************************* */
/**
 * @brief This function smoothes an image with a Gaussian kernel
 * @param src Input image
 * @param dst Output image
 * @param ksize_x Kernel size in X-direction (horizontal)
 * @param ksize_y Kernel size in Y-direction (vertical)
 * @param sigma Kernel standard deviation
 */
void gaussian_2D_convolutionV2(const cv::Mat& src, cv::Mat& dst, int ksize_x, int ksize_y, float sigma) {

    int ksize_x_ = 0, ksize_y_ = 0;

    // Compute an appropriate kernel size according to the specified sigma
    if (sigma > ksize_x || sigma > ksize_y || ksize_x == 0 || ksize_y == 0) {
        ksize_x_ = (int)ceil(2.0f*(1.0f + (sigma - 0.8f) / (0.3f)));
        ksize_y_ = ksize_x_;
    }

    // The kernel size must be and odd number
    if ((ksize_x_ % 2) == 0) {
        ksize_x_ += 1;
    }

    if ((ksize_y_ % 2) == 0) {
        ksize_y_ += 1;
    }

    // Perform the Gaussian Smoothing with border replication
    GaussianBlur(src, dst, Size(ksize_x_, ksize_y_), sigma, sigma, BORDER_REPLICATE);
}

/* ************************************************************************* */
/**
 * @brief This function computes image derivatives with Scharr kernel
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 * @note Scharr operator approximates better rotation invariance than
 * other stencils such as Sobel. See Weickert and Scharr,
 * A Scheme for Coherence-Enhancing Diffusion Filtering with Optimized Rotation Invariance,
 * Journal of Visual Communication and Image Representation 2002
 */
void image_derivatives_scharrV2(const cv::Mat& src, cv::Mat& dst, int xorder, int yorder) {
    Scharr(src, dst, CV_32F, xorder, yorder, 1.0, 0, BORDER_DEFAULT);
}

/* ************************************************************************* */
/**
 * @brief This function computes the Perona and Malik conductivity coefficient g1
 * g1 = exp(-|dL|^2/k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void pm_g1V2(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {

  // Compute: dst = exp((Lx.mul(Lx) + Ly.mul(Ly)) / (-k * k))

  const float neg_inv_k2 = -1.0f / (k*k);

  const int total = Lx.rows * Lx.cols;
  const float* lx = Lx.ptr<float>(0);
  const float* ly = Ly.ptr<float>(0);
  float* d = dst.ptr<float>(0);

  for (int i = 0; i < total; i++)
    d[i] = neg_inv_k2 * (lx[i]*lx[i] + ly[i]*ly[i]);

  exp(dst, dst);
}

/* ************************************************************************* */
/**
 * @brief This function computes the Perona and Malik conductivity coefficient g2
 * g2 = 1 / (1 + dL^2 / k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void pm_g2V2(const cv::Mat &Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {

  // Compute: dst = 1.0f / (1.0f + ((Lx.mul(Lx) + Ly.mul(Ly)) / (k * k)) );

  const float inv_k2 = 1.0f / (k * k);

  const int total = Lx.rows * Lx.cols;
  const float* lx = Lx.ptr<float>(0);
  const float* ly = Ly.ptr<float>(0);
  float* d = dst.ptr<float>(0);

  for (int i = 0; i < total; i++)
    d[i] = 1.0f / (1.0f + ((lx[i] * lx[i] + ly[i] * ly[i]) * inv_k2));
}

/* ************************************************************************* */
/**
 * @brief This function computes Weickert conductivity coefficient gw
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 * @note For more information check the following paper: J. Weickert
 * Applications of nonlinear diffusion in image processing and computer vision,
 * Proceedings of Algorithmy 2000
 */
void weickert_diffusivityV2(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {

  // Compute: dst = 1.0f - exp(-3.315f / ((Lx.mul(Lx) + Ly.mul(Ly)) / (k * k))^4)

  const float inv_k2 = 1.0f / (k * k);

  const int total = Lx.rows * Lx.cols;
  const float* lx = Lx.ptr<float>(0);
  const float* ly = Ly.ptr<float>(0);
  float* d = dst.ptr<float>(0);

  for (int i = 0; i < total; i++) {
    float dL = inv_k2 * (lx[i] * lx[i] + ly[i] * ly[i]);
    d[i] = -3.315f / (dL*dL*dL*dL);
  }

  exp(dst, dst);

  for (int i = 0; i < total; i++)
    d[i] = 1.0f - d[i];
}


/* ************************************************************************* */
/**
* @brief This function computes Charbonnier conductivity coefficient gc
* gc = 1 / sqrt(1 + dL^2 / k^2)
* @param Lx First order image derivative in X-direction (horizontal)
* @param Ly First order image derivative in Y-direction (vertical)
* @param dst Output image
* @param k Contrast factor parameter
* @note For more information check the following paper: J. Weickert
* Applications of nonlinear diffusion in image processing and computer vision,
* Proceedings of Algorithmy 2000
*/
void charbonnier_diffusivityV2(const cv::Mat& Lx, const cv::Mat& Ly, cv::Mat& dst, float k) {

  // Compute: dst = 1.0f / sqrt(1.0f + (Lx.mul(Lx) + Ly.mul(Ly)) / (k * k))

  const float inv_k2 = 1.0f / (k * k);

  const int total = Lx.rows * Lx.cols;
  const float* lx = Lx.ptr<float>(0);
  const float* ly = Ly.ptr<float>(0);
  float* d = dst.ptr<float>(0);

  for (int i = 0; i < total; i++)
    d[i] = 1.0f / sqrtf(1.0f + inv_k2 * (lx[i]*lx[i] + ly[i]*ly[i]));
}


/* ************************************************************************* */
/**
 * @brief This function computes a good empirical value for the k contrast factor
 * given an input image, the percentile (0-1), the gradient scale and the number of
 * bins in the histogram
 * @param img Input image
 * @param perc Percentile of the image gradient histogram (0-1)
 * @param gscale Scale for computing the image gradient histogram
 * @param nbins Number of histogram bins
 * @param ksize_x Kernel size in X-direction (horizontal) for the Gaussian smoothing kernel
 * @param ksize_y Kernel size in Y-direction (vertical) for the Gaussian smoothing kernel
 * @return k contrast factor
 */
float compute_k_percentileV2(const cv::Mat& Lx, const cv::Mat& Ly, float perc, std::vector<int>& hist) {

    const int nbins = (int)hist.size();
    int nbin = 0, nelements = 0, nthreshold = 0, k = 0;
    float kperc = 0.0, modg = 0.0;
    float npoints = 0.0;
    float hmax = 0.0;

    // Clear the histogram by zero-fill
    std::fill(std::begin(hist), std::end(hist), 0);

    // Skip the borders for computing the histogram
    for (int i = 1; i < Lx.rows - 1; i++) {
        const float *lx = Lx.ptr<float>(i);
        const float *ly = Ly.ptr<float>(i);
        for (int j = 1; j < Lx.cols - 1; j++) {
            modg = lx[j]*lx[j] + ly[j]*ly[j];

            // Get the maximum
            if (modg > hmax) {
                hmax = modg;
            }
        }
    }
    hmax = sqrt(hmax);
    // Skip the borders for computing the histogram
    for (int i = 1; i < Lx.rows - 1; i++) {
        const float *lx = Lx.ptr<float>(i);
        const float *ly = Ly.ptr<float>(i);
        for (int j = 1; j < Lx.cols - 1; j++) {
            modg = lx[j]*lx[j] + ly[j]*ly[j];

            // Find the correspondent bin
            if (modg != 0.0) {
                nbin = (int)floor(nbins*(sqrt(modg) / hmax));

                if (nbin == nbins) {
                    nbin--;
                }

                hist[nbin]++;
                npoints++;
            }
        }
    }

    // Now find the perc of the histogram percentile
    nthreshold = (int)(npoints*perc);

    for (k = 0; nelements < nthreshold && k < nbins; k++) {
        nelements = nelements + hist[k];
    }

    if (nelements < nthreshold)  {
        kperc = 0.03f;
    }
    else {
        kperc = hmax*((float)(k) / (float)nbins);
    }

    return kperc;
}


/* ************************************************************************* */
/**
 * @brief Compute Scharr derivative kernels for sizes different than 3
 * @param _kx Horizontal kernel ues
 * @param _ky Vertical kernel values
 * @param dx Derivative order in X-direction (horizontal)
 * @param dy Derivative order in Y-direction (vertical)
 * @param scale_ Scale factor or derivative size
 */
void compute_scharr_derivative_kernelsV2(cv::OutputArray _kx, cv::OutputArray _ky, int dx, int dy, int scale) {

    int ksize = 3 + 2 * (scale - 1);

    // The standard Scharr kernel
    if (scale == 1) {
        getDerivKernels(_kx, _ky, dx, dy, 0, true, CV_32F);
        return;
    }

    _kx.create(ksize, 1, CV_32F, -1, true);
    _ky.create(ksize, 1, CV_32F, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();

    float w = 10.0f / 3.0f;
    float norm = 1.0f / (2.0f*scale*(w + 2.0f));

    std::vector<float> kerI(ksize, 0.0f);

    if (dx == 0) {
        kerI[0] = norm, kerI[ksize / 2] = w*norm, kerI[ksize - 1] = norm;
    }
    else if (dx == 1) {
        kerI[0] = -1, kerI[ksize / 2] = 0, kerI[ksize - 1] = 1;
    }
    Mat(kx.rows, kx.cols, CV_32F, &kerI[0]).copyTo(kx);

    kerI.assign(ksize, 0.0f);

    if (dy == 0) {
        kerI[0] = norm, kerI[ksize / 2] = w*norm, kerI[ksize - 1] = norm;
    }
    else if (dy == 1) {
        kerI[0] = -1, kerI[ksize / 2] = 0, kerI[ksize - 1] = 1;
    }
    Mat(ky.rows, ky.cols, CV_32F, &kerI[0]).copyTo(ky);
}

class Nld_Step_Scalar_InvokerV2 : public cv::ParallelLoopBody
{
public:
    Nld_Step_Scalar_InvokerV2(const cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep)
        : _Ld(&Ld)
        , _c(&c)
        , _Lstep(&Lstep)
    {
    }

    virtual ~Nld_Step_Scalar_InvokerV2()
    {

    }

    void operator()(const cv::Range& range) const
    {
        const cv::Mat& Ld = *_Ld;
        const cv::Mat& c = *_c;
        cv::Mat& Lstep = *_Lstep;

        for (int i = range.start; i < range.end; i++)
        {
            const float *c_prev  = c.ptr<float>(i - 1);
            const float *c_curr  = c.ptr<float>(i);
            const float *c_next  = c.ptr<float>(i + 1);
            const float *ld_prev = Ld.ptr<float>(i - 1);
            const float *ld_curr = Ld.ptr<float>(i);
            const float *ld_next = Ld.ptr<float>(i + 1);

            float *dst  = Lstep.ptr<float>(i);

            for (int j = 1; j < Lstep.cols - 1; j++)
            {
                float xpos = (c_curr[j]   + c_curr[j+1])*(ld_curr[j+1] - ld_curr[j]);
                float xneg = (c_curr[j-1] + c_curr[j])  *(ld_curr[j]   - ld_curr[j-1]);
                float ypos = (c_curr[j]   + c_next[j])  *(ld_next[j]   - ld_curr[j]);
                float yneg = (c_prev[j]   + c_curr[j])  *(ld_curr[j]   - ld_prev[j]);
                dst[j] = (xpos - xneg + ypos - yneg);
            }
        }
    }
private:
    const cv::Mat * _Ld;
    const cv::Mat * _c;
    cv::Mat * _Lstep;
};

/* ************************************************************************* */
/**
* @brief This function computes a scalar non-linear diffusion step
* @param Ld Base image in the evolution
* @param c Conductivity image
* @param Lstep Output image that gives the difference between the current
* Ld and the next Ld being evolved
* @note Forward Euler Scheme 3x3 stencil
* The function c is a scalar value that depends on the gradient norm
* dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
*/
void nld_step_scalarV2(const cv::Mat& Ld, const cv::Mat& c, cv::Mat& Lstep) {

    cv::parallel_for_(cv::Range(1, Lstep.rows - 1), Nld_Step_Scalar_InvokerV2(Ld, c, Lstep), (double)Ld.total()/(1 << 16));

    float xneg, xpos, yneg, ypos;
    float* dst = Lstep.ptr<float>(0);
    const float* cprv = NULL;
    const float* ccur  = c.ptr<float>(0);
    const float* cnxt  = c.ptr<float>(1);
    const float* ldprv = NULL;
    const float* ldcur = Ld.ptr<float>(0);
    const float* ldnxt = Ld.ptr<float>(1);
    for (int j = 1; j < Lstep.cols - 1; j++) {
        xpos = (ccur[j]   + ccur[j+1]) * (ldcur[j+1] - ldcur[j]);
        xneg = (ccur[j-1] + ccur[j])   * (ldcur[j]   - ldcur[j-1]);
        ypos = (ccur[j]   + cnxt[j])   * (ldnxt[j]   - ldcur[j]);
        dst[j] = (xpos - xneg + ypos);
    }

    dst = Lstep.ptr<float>(Lstep.rows - 1);
    ccur = c.ptr<float>(Lstep.rows - 1);
    cprv = c.ptr<float>(Lstep.rows - 2);
    ldcur = Ld.ptr<float>(Lstep.rows - 1);
    ldprv = Ld.ptr<float>(Lstep.rows - 2);

    for (int j = 1; j < Lstep.cols - 1; j++) {
        xpos = (ccur[j] + ccur[j+1]) * (ldcur[j+1] - ldcur[j]);
        xneg = (ccur[j-1] + ccur[j]) * (ldcur[j] - ldcur[j-1]);
        yneg = (cprv[j] + ccur[j])   * (ldcur[j] - ldprv[j]);
        dst[j] = (xpos - xneg - yneg);
    }

    ccur = c.ptr<float>(1);
    ldcur = Ld.ptr<float>(1);
    cprv = c.ptr<float>(0);
    ldprv = Ld.ptr<float>(0);

    int r0 = Lstep.cols - 1;
    int r1 = Lstep.cols - 2;

    for (int i = 1; i < Lstep.rows - 1; i++) {
        cnxt = c.ptr<float>(i + 1);
        ldnxt = Ld.ptr<float>(i + 1);
        dst = Lstep.ptr<float>(i);

        xpos = (ccur[0] + ccur[1]) * (ldcur[1] - ldcur[0]);
        ypos = (ccur[0] + cnxt[0]) * (ldnxt[0] - ldcur[0]);
        yneg = (cprv[0] + ccur[0]) * (ldcur[0] - ldprv[0]);
        dst[0] = (xpos + ypos - yneg);

        xneg = (ccur[r1] + ccur[r0]) * (ldcur[r0] - ldcur[r1]);
        ypos = (ccur[r0] + cnxt[r0]) * (ldnxt[r0] - ldcur[r0]);
        yneg = (cprv[r0] + ccur[r0]) * (ldcur[r0] - ldprv[r0]);
        dst[r0] = (-xneg + ypos - yneg);

        cprv = ccur;
        ccur = cnxt;
        ldprv = ldcur;
        ldcur = ldnxt;
    }
}

/* ************************************************************************* */
/**
* @brief This function downsamples the input image using OpenCV resize
* @param img Input image to be downsampled
* @param dst Output image with half of the resolution of the input image
*/
void halfsample_imageV2(const cv::Mat& src, cv::Mat& dst) {

    // Make sure the destination image is of the right size
    CV_Assert(src.cols / 2 == dst.cols);
    CV_Assert(src.rows / 2 == dst.rows);
    resize(src, dst, dst.size(), 0, 0, cv::INTER_AREA);
}

/* ************************************************************************* */
/**
 * @brief This function checks if a given pixel is a maximum in a local neighbourhood
 * @param img Input image where we will perform the maximum search
 * @param dsize Half size of the neighbourhood
 * @param value Response value at (x,y) position
 * @param row Image row coordinate
 * @param col Image column coordinate
 * @param same_img Flag to indicate if the image value at (x,y) is in the input image
 * @return 1->is maximum, 0->otherwise
 */
bool check_maximum_neighbourhoodV2(const cv::Mat& img, int dsize, float value, int row, int col, bool same_img) {

    bool response = true;

    for (int i = row - dsize; i <= row + dsize; i++) {
        for (int j = col - dsize; j <= col + dsize; j++) {
            if (i >= 0 && i < img.rows && j >= 0 && j < img.cols) {
                if (same_img == true) {
                    if (i != row || j != col) {
                        if ((*(img.ptr<float>(i)+j)) > value) {
                            response = false;
                            return response;
                        }
                    }
                }
                else {
                    if ((*(img.ptr<float>(i)+j)) > value) {
                        response = false;
                        return response;
                    }
                }
            }
        }
    }

    return response;
}

}
