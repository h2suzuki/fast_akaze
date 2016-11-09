#ifndef __FRAME_PER_SECOND_STATS_HPP__
#define __FRAME_PER_SECOND_STATS_HPP__


#include <chrono>
#include <string>  // fps_stats::header
#include <cstdio>  // printf


/*
  Description:
    Measure the statistics of the frame rate.  For every msec_ interval, the stats line will
    be printed out with header_, which are passed through the constructor.
*/
class fps_stats
{
    using hireso_clock = std::chrono::high_resolution_clock;

    std::chrono::milliseconds const _fps_update_interval;
    std::string const _header;

    int   _frame_count;
    float _fps_last;    // The fps of the last tick
    float _fps_ave;	// The average fps of the last interval
    float _fps_ema;	// The exponential moving average of fps

    std::chrono::time_point<hireso_clock, hireso_clock::duration> _t_interval_start, _t_last, _t_current;

public:
    fps_stats(const std::string header_, bool start_now_ = true, int msec_ = 5000)
        : _header(header_), _fps_update_interval(msec_)
    {
        if (start_now_)
            start();
    }

    void start(void)
    {
        _frame_count = 0;
        _fps_last = _fps_ave = _fps_ema = 0;
        _t_interval_start = _t_last = _t_current = hireso_clock::now();
    }

    void tick(bool verbose = true)
    {
        using namespace std::chrono;

        _t_current = hireso_clock::now();	// the first thing to do

        auto delta_last = _t_current - _t_last;
        auto delta_interval = _t_current - _t_interval_start;

        _fps_last = 1000000.0f / duration_cast<microseconds>(delta_last).count();
        _t_last = _t_current;

        _frame_count++;

        if (_fps_update_interval <= delta_interval)
        {
            // Calculate FPS of this interval and its moving average
            _fps_ave = _frame_count * 1000000.0f / duration_cast<microseconds>(delta_interval).count();
            _fps_ema = 0.8f * _fps_ema + 0.2f * _fps_ave;

            // Print the statistics
            if (verbose)
                printf("%s: Last %0.3fms %0.3ffps | Int %ldms %dfr %0.3ffps | EMA %0.3ffps\n",
                        _header.c_str(),
                        0.001f * duration_cast<microseconds>(delta_last).count(),
                        _fps_last,
                        duration_cast<milliseconds>(delta_interval).count(),
                        _frame_count,
                        _fps_ave,
                        _fps_ema);

            _frame_count = 0;
            _t_interval_start = _t_current;
        }
    }

    float last_fps(void) { return _fps_last;	}
    float interval_fps(void) { return _fps_ave; }
    float ema_fps(void) { return _fps_ema; }
};

#endif // !__FRAME_PER_SECOND_STATS_HPP__
