#include "common/plugin.hpp"
#include "common/threadloop.hpp"
#include "common/switchboard.hpp"
#include "common/phonebook.hpp"
#include "common/data_format.hpp"
#include "common/logger.hpp"
#include "common/threadloop.hpp"
#include "hologram.h"
#include <chrono>
#include <memory>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cuda_runtime_api.h>

using namespace ILLIXR;
using std::unique_ptr;
using std::thread;
using std::atomic;

class hologram : public threadloop {
public:
    hologram(std::string name_, phonebook* pb_)
        : threadloop{name_, pb_}
        , sb{pb->lookup_impl<switchboard>()}
        , _m_in{sb->get_reader<hologram_input>("hologram_in")}
        , _m_out{sb->get_writer<hologram_output>("hologram_out")}
    {
        bool ret = HLG_initialize();
        if (!ret) {
            throw std::runtime_error{"Hologram Initialization failed (" + std::to_string(ret) + ")"};
        }

        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    // destructor
    virtual ~hologram() override {
        for (int i = 0; i < _start_durations.size(); ++i) {
            std::cout << "gpu_timer,hologram," << i << ",0,0," << (_stop_durations[i] - _start_durations[i]) * 1000000 << "\n";
        }
    }

    void _p_one_iteration() override {
        _start_durations.push_back(_total_gpu_time);

        cudaEventRecord(_start, 0);
        HLG_process();
        cudaEventRecord(_stop, 0);

        cudaEventSynchronize(_stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, _start, _stop);
        _total_gpu_time += elapsed_time;

        _stop_durations.push_back(_total_gpu_time);
    }

private:
    const std::shared_ptr<switchboard> sb;
    switchboard::reader<hologram_input> _m_in;
    switchboard::writer<hologram_output> _m_out;

    // Timing
    cudaEvent_t _start;
    cudaEvent_t _stop;
    std::vector<float> _start_durations;
    std::vector<float> _stop_durations;
    float _total_gpu_time = 0;
};


PLUGIN_MAIN(hologram)
