#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "features2d_akaze2.hpp"

#include <functional>	// ref()
#include <atomic>
#include <thread>
#include <chrono>
#include <string>

#include <iostream>

#include "barter.hpp"
#include "fps_stats.hpp"


// The switch to select AKAZE(the original version) or AKAZE2(the local variant)
#define USE_AKAZE2			1


// The number of threads to use; 0 to disable multi-threading
#define OPENCV_THREAD_COUNT		0

// Allow akaze2 thread to run at the pace exceeding the video frame rate
#define	ALLOW_OVERPACE			true

// OpenCV event dispatcher wait time
#define CVWAITKEY_WAIT			30  /* msec */


// Logicool C525 Spec: 640x480 30fps, FOV 69 degree
#define VIDEO_FRAME_WIDTH		640
#define VIDEO_FRAME_HEIGHT		480
#define VIDEO_FRAME_RATE		30  /* frames per second */
#define	CAM_ID				0


// Debug window titles
#define	WIN_TITLE_INPUT			"Video Input"
#define	WIN_TITLE_OUTPUT		"KP Match"


// Akaze parameters
#define AKAZE_DESCRIPTOR_SIZE		486	/* 64 or 256 or 486 bits; 0 means full and 486 bits in case of three channels */
#define AKAZE_DESCRIPTOR_CH		3	/* 1 or 2 or 3; The descriptor size must be <= 162*CH */
#define AKAZE_NUM_OCTAVES		4
#define AKAZE_NUM_OCTAVE_SUBLAYERS	4

#define AKAZE_KPCOUNT_MIN		140
#define AKAZE_KPCOUNT_MAX		160
#define AKAZE_THRESHOLD_MIN		0.00001f
#define AKAZE_THRESHOLD_MAX		0.1f


// Threshold for matching outliers
#define MATCH_HAMMING_RADIUS		121.5f	/* 1/4 of the descriptor size */



enum ThreadState {
    Pause, Running, Quit
};

enum ThreadCommand {
    None, SetReference, SwitchDrawMethod, TrackingOn,
};


#if USE_AKAZE2
void tune_akaze_threshold(cv::AKAZE2 & akaze_, int last_nkp)
#else
void tune_akaze_threshold(cv::AKAZE & akaze_, int last_nkp)
#endif
{
    if (AKAZE_KPCOUNT_MIN <= last_nkp && last_nkp <= AKAZE_KPCOUNT_MAX)
        return;

    /*
      By converting the parameters as y = log10(nkp+1), x = log10(threshold),
      a simple fitting line, y = a * x + b, can be assumed to find out
      the threshold to give the target nkp
    */

    const double target_nkp = 0.5 * (AKAZE_KPCOUNT_MAX + AKAZE_KPCOUNT_MIN);
    const double target_y = log10(target_nkp);

    // Some negative number; closer to 0 means finer and slower to approach the target
    const double slope = -1.0;

    double x = log10(akaze_.getThreshold());
    double y = log10(last_nkp + 1.0);

    x = x + slope * (target_y - y);

    double threshold = exp(x * log(10.0));
    char *s{ threshold > akaze_.getThreshold() ? "n" : "w" };  // Narrower or Wider aperture

    if (threshold > AKAZE_THRESHOLD_MAX)
        threshold = AKAZE_THRESHOLD_MAX, s = "c"; // The aperture is closed
    else
    if (threshold < AKAZE_THRESHOLD_MIN)
        threshold = AKAZE_THRESHOLD_MIN, s = "o"; // The aperture is fully open

    //std::cout << s << " " << last_nkp << "\tdelta:" << (target_y - y) << ": " << threshold << std::endl;
    std::cout << s;

    akaze_.setThreshold(threshold);
}


void remove_outliers_by_distance(const std::vector<cv::KeyPoint> & kp0_,
                                 const std::vector<cv::KeyPoint> & kp1_,
                                 const float threshold_,
                                 std::vector<cv::DMatch> & matches_,
                                 std::vector<cv::DMatch> & outliers_)
{
    if (matches_.empty())
        return;

    // Remove matches if the best-matched distance is too far
    matches_.erase(std::remove_if(std::begin(matches_), std::end(matches_),
        [threshold_, &outliers_](cv::DMatch &m)
        {
            if (m.distance > threshold_) {
                outliers_.push_back(m);
                return true;
            }
            return false;
        }),
        matches_.end());
}



void draw_side_by_side(const cv::Mat & frame0_,
                       const cv::Mat & frame1_,
                       const std::vector<cv::KeyPoint> & kp0_,
                       const std::vector<cv::KeyPoint> & kp1_,
                       const std::vector<cv::DMatch> & matches_,
                       const std::vector<cv::DMatch> & outliers1_,
                       const std::vector<cv::DMatch> & outliers2_,
                       const std::vector<cv::DMatch> & outliers3_,
                       const float fps_,
                       cv::Mat & dst_)
{
    if (frame0_.empty() || frame1_.empty())
        return;

    if (kp0_.empty() || kp1_.empty() || matches_.empty())
        return;

    cv::drawMatches(frame0_, kp0_, frame1_, kp1_, matches_, dst_,
                    /* matchColor */ cv::Scalar{250,250,250},
                    /* singlePointColor */ cv::Scalar::all(-1),
                    /* matchesMask */ std::vector<char>(),
                    /* flags */ cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS |
                                cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    if (!outliers1_.empty()) {
        for (auto & m : outliers1_) {
            cv::Point2f p{ kp1_[m.trainIdx].pt.x + frame0_.cols, kp1_[m.trainIdx].pt.y };
            cv::line(dst_, kp0_[m.queryIdx].pt, p, cv::Scalar{ 100, 20, 20 });
        }
    }

    // Draw the commentary text
    std::vector<std::string> s{ std::to_string(fps_) + " fps",
                                std::to_string(kp0_.size()) + " keypoints",
                                std::to_string(matches_.size()) + " matches",
                                std::to_string(outliers1_.size()) + " outliers(blue)",
                                std::to_string(outliers2_.size()) + " outliers(green)",
                                std::to_string(outliers3_.size()) + " outliers(red)" };

    for (int i = 0; i < s.size(); i++)
        cv::putText(dst_, s[i].c_str(),
                    /* top-left corner */ cv::Point(10, 20 + 20*i),
                    /* font face  */ cv::FONT_HERSHEY_COMPLEX,
                    /* font scale */ 0.5,
                    /* font color */ cv::Scalar(80,220,80),
                    /* thickness  */ 1,
                    /* line type  */ cv::LINE_AA);
}


void draw_frame(const cv::Mat & frame_,
                const std::vector<cv::KeyPoint> & kp0_,
                const std::vector<cv::KeyPoint> & kp1_,
                const std::vector<cv::DMatch> & matches_,
                const std::vector<cv::DMatch> & outliers1_,
                const std::vector<cv::DMatch> & outliers2_,
                const std::vector<cv::DMatch> & outliers3_,
                const float fps_,
                cv::Mat & dst_)
{

    if (kp0_.empty()) {
        frame_.copyTo(dst_);
    }
    else {
        cv::drawKeypoints(frame_, kp0_, dst_, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS |
                          cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    if (!outliers1_.empty()) {
        for (auto & m : outliers1_)
            cv::line(dst_, kp0_[m.queryIdx].pt, kp1_[m.trainIdx].pt, cv::Scalar{ 100, 20, 20 });
    }

    if (!outliers2_.empty()) {
        for (auto & m : outliers2_)
            cv::line(dst_, kp0_[m.queryIdx].pt, kp1_[m.trainIdx].pt, cv::Scalar{ 20, 100, 20 });
    }

    if (!outliers3_.empty()) {
        for (auto & m : outliers3_)
            cv::line(dst_, kp0_[m.queryIdx].pt, kp1_[m.trainIdx].pt, cv::Scalar{ 20, 20, 100 });
    }

    if (!matches_.empty()) {
        for (auto & m : matches_)
            cv::line(dst_, kp0_[m.queryIdx].pt, kp1_[m.trainIdx].pt, cv::Scalar{ 250, 250, 250 }, 2, cv::LINE_AA);
    }


    std::vector<std::string> s{ std::to_string(fps_) + " fps",
                                std::to_string(kp0_.size()) + " keypoints",
                                std::to_string(matches_.size()) + " matches",
                                std::to_string(outliers1_.size()) + " outliers(blue)",
                                std::to_string(outliers2_.size()) + " outliers(green)",
                                std::to_string(outliers3_.size()) + " outliers(red)" };

    for (int i = 0; i < s.size(); i++)
        cv::putText(dst_, s[i].c_str(),
                    /* top-left corner */ cv::Point(10, 20 + 20*i),
                    /* font face  */ cv::FONT_HERSHEY_COMPLEX,
                    /* font scale */ 0.5,
                    /* font color */ cv::Scalar(80,220,80),
                    /* thickness  */ 1,
                    /* line type  */ cv::LINE_AA);

    // Draw the progress bar
    if (!kp0_.empty()) {
        int width = (int)(dst_.cols * 0.6f);
        int gap0 = (int)(dst_.cols * 0.2f);

        int height = std::max(dst_.rows / 20, 4);
        int y0 = std::max(dst_.rows - height - 10, 40);

        int gap = gap0;
        int progress = (int)(width * matches_.size() / kp0_.size());
        cv::rectangle(dst_, cv::Rect(gap0, y0, progress, height), cv::Scalar(100, 100, 100), cv::FILLED);

        gap += progress;
        progress = (int)(width * outliers1_.size() / kp0_.size());
        cv::rectangle(dst_, cv::Rect(gap, y0, progress, height), cv::Scalar(100, 20, 20), cv::FILLED);

        gap += progress;
        progress = (int)(width * outliers2_.size() / kp0_.size());
        cv::rectangle(dst_, cv::Rect(gap, y0, progress, height), cv::Scalar(20, 100, 20), cv::FILLED);

        gap += progress;
        progress = (int)(width * outliers3_.size() / kp0_.size());
        cv::rectangle(dst_, cv::Rect(gap, y0, progress, height), cv::Scalar(20, 20, 100), cv::FILLED);

        // Draw the outer box
        cv::rectangle(dst_, cv::Rect(gap0, y0, width, height), cv::Scalar(50, 50, 50), 1);
    }
}



void run_akaze2(barter<cv::Mat> & frame_barter_, std::atomic_int & t_state_, std::atomic_int & t_cmd_)
{
    // Create an AKAZE detector and the related constructs
#if USE_AKAZE2
    auto detector = cv::AKAZE2::create(cv::AKAZE::DESCRIPTOR_MLDB,
#else
    auto detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,
#endif
                                      AKAZE_DESCRIPTOR_SIZE,
                                      AKAZE_DESCRIPTOR_CH,
                                      AKAZE_THRESHOLD_MAX,
                                      AKAZE_NUM_OCTAVES,
                                      AKAZE_NUM_OCTAVE_SUBLAYERS);

    std::vector<cv::KeyPoint> kp_ref, kp;
    cv::Mat desc_ref, desc;
    int last_nkp = 0;


    // Create a Brute-force matcher and the related constructs
    auto matcher = cv::BFMatcher{ cv::NORM_HAMMING, /* crossCheck */ true };
    std::vector<cv::DMatch> matches, outliers1, outliers2, outliers3;


    // Allocate the frame memory to exchange with the main thread
    auto frame_ref = std::unique_ptr<cv::Mat>(new cv::Mat);
    auto frame     = std::unique_ptr<cv::Mat>(new cv::Mat);


    // Allocate the output image to show on the debug window
    cv::Mat output;


#if USE_AKAZE2
    fps_stats fps{ "AKAZE2" };
#else
    fps_stats fps{ "AKAZE" };
#endif
    bool side_by_side = false, tracking = false;
    for (;; fps.tick()) {

        // Wait for a new frame to arrive
        while (t_state_ != ThreadState::Running) {
            if (t_state_ == ThreadState::Quit)	return;
            std::this_thread::sleep_for(std::chrono::milliseconds(CVWAITKEY_WAIT));
        }
        if (!ALLOW_OVERPACE) t_state_ = ThreadState::Pause;

        frame_barter_.exchange(frame);
        CV_Assert(frame);

        if (ALLOW_OVERPACE && frame->empty())
            continue;

        // Keypoint detection
        tune_akaze_threshold(*detector, last_nkp);
        detector->detectAndCompute(*frame, cv::noArray(), kp, desc);
        last_nkp = (int)kp.size();


        // Keypoint matching
        matches.clear();
        outliers1.clear();
        outliers2.clear();
        outliers3.clear();

        if (last_nkp > 0 && kp_ref.size() > 0) {
            matcher.match(desc, desc_ref, matches);
            remove_outliers_by_distance(kp, kp_ref, MATCH_HAMMING_RADIUS, matches, outliers1);
        }

        // Show the result
        if (!side_by_side || frame_ref->empty())
            draw_frame(*frame, kp, kp_ref, matches, outliers1, outliers2, outliers3, fps.last_fps(), output);
        else
            draw_side_by_side(*frame, *frame_ref, kp, kp_ref, matches, outliers1, outliers2, outliers3, fps.last_fps(), output);
        if (!output.empty())
            cv::imshow(WIN_TITLE_OUTPUT, output);

        if (tracking) {
            std::swap(frame, frame_ref);
            std::swap(kp, kp_ref);
            desc.copyTo(desc_ref);
        }

        // Handle a thread command afterward
        switch (t_cmd_) {
        default:
            std::cout << "Unknown command:" << t_cmd_ << std::endl;
            break;
        case ThreadCommand::None:
            break;
        case ThreadCommand::TrackingOn:
            tracking = !tracking;
            break;
        case ThreadCommand::SwitchDrawMethod:
            side_by_side = !side_by_side;
            break;
        case ThreadCommand::SetReference:
            if (!tracking) {
                std::swap(frame, frame_ref);
                std::swap(kp, kp_ref);
                desc.copyTo(desc_ref);
            }
            tracking = false;
            break;
        }
        t_cmd_ = ThreadCommand::None; /* i.e. handled */
    }

    std::cout << __FUNCTION__ << " is ending." << std::endl;
}



bool handle_command_key(int key_, std::atomic_int & t_cmd_)
{
    // Time-out is handled by ignorance
    if (key_ == -1)
        return true;

    int expected = ThreadCommand::None;

    switch (key_) {
    default:
        return false;  // An unknown key; not handled
    case 't':
        t_cmd_.compare_exchange_strong(expected, ThreadCommand::TrackingOn);
        break;
    case 's':
        t_cmd_.compare_exchange_strong(expected, ThreadCommand::SwitchDrawMethod);
        break;
    case 'r':
        t_cmd_.compare_exchange_strong(expected, ThreadCommand::SetReference);
        break;
    }

    if (expected != ThreadCommand::None)
        std::cout << "Ignoring:" << key_ << "  locked by " << expected << std::endl;

    return true;
}



int main(void)
{
    // Setup the camera
    cv::VideoCapture cap(cv::CAP_DSHOW + CAM_ID);
    if (!cap.isOpened()) {
        std::cerr << "Failed to connect the camera" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_HEIGHT);
    cap.set(cv::CAP_PROP_FPS, VIDEO_FRAME_RATE);


    // Setup the debug windows
    cv::namedWindow(WIN_TITLE_INPUT, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(WIN_TITLE_OUTPUT, cv::WINDOW_AUTOSIZE);


    // Set OpenCV parallelization
    cv::setNumThreads(OPENCV_THREAD_COUNT);


    // Setup the akaze2 thread
    barter<cv::Mat> frame_barter;
    std::atomic_int t_state{ ThreadState::Pause };
    std::atomic_int t_cmd{ ThreadCommand::None };
    std::thread akaze2_thread { run_akaze2,
                                std::ref(frame_barter),
                                std::ref(t_state),
                                std::ref(t_cmd) };


    // Allocate the memory for the input frame to exchange with akaze2_thread
    auto frame = std::unique_ptr<cv::Mat>(new cv::Mat);

    std::this_thread::sleep_for(std::chrono::seconds(1)); /* ramp-up for akaze2_thread */
    std::cout << "cv::getNumThreads(): " << cv::getNumThreads() << std::endl;

    // Start the event loop
    fps_stats fps{ "Video" };
    cap >> *frame;
    while (!frame->empty()) {

        cv::imshow(WIN_TITLE_INPUT, *frame);

        // Put the frame to the shelf, so the akaze2 thread can take it
        frame_barter.exchange(frame);
        t_state = ThreadState::Running;

        // Event dispatcher; appropriately set t_cmd
        if (!handle_command_key(cv::waitKey(CVWAITKEY_WAIT), t_cmd))
            break;

        // Grab the next frame; the waiting time depends on the exposure light and the camera spec
        fps.tick(false);
        cap >> *frame;
    }

    if (akaze2_thread.joinable()) {
        t_state = ThreadState::Quit;
        akaze2_thread.join();
        std::cout << "akaze2_thread is joined" << std::endl;
    }

    cv::destroyAllWindows();

    return 0;
}
