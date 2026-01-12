#include "mpmc.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

static double cycles_to_ns(uint64_t delta) noexcept {
    static double factor = [] {
        long khz = 0;
        uint64_t c0 = __rdtsc();
        struct timespec ts{0, 100000000};
        nanosleep(&ts, nullptr);
        uint64_t c1 = __rdtsc();
        khz = long((c1 - c0) / 100);
        return 1e6 / double(khz);
    }();

    return delta * factor;
}

static constexpr size_t QUEUE_SIZE = 1u << 20;
static constexpr uint32_t TOTAL_MSGS = 10'000'000;
template <size_t N>
struct Payload {
    std::array<uint8_t, N> bytes{};
};

template <size_t N>
Payload<N> make_payload(uint64_t producer, uint64_t seq) {
    Payload<N> payload{};
    const uint64_t data[2] = {producer, seq};
    constexpr size_t copy_bytes =
        (N < sizeof(data)) ? N : sizeof(data);
    if constexpr (copy_bytes > 0) {
        std::memcpy(payload.bytes.data(), data, copy_bytes);
    }
    return payload;
}

struct LatencyStats {
    size_t count{0};
    double avg{0.0};
    double p25{0.0};
    double p50{0.0};
    double p75{0.0};
    double p90{0.0};
    double p99{0.0};
};

static double pick_percentile_sorted(const std::vector<double>& samples, double pct) {
    if (samples.empty()) {
        return 0.0;
    }
    const size_t count = samples.size();
    const size_t idx = static_cast<size_t>(pct * (count - 1));
    return samples[idx];
}

static LatencyStats compute_stats_sorted(const std::vector<double>& samples) {
    LatencyStats stats{};
    if (samples.empty()) {
        return stats;
    }

    const size_t count = samples.size();
    stats.count = count;
    stats.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / count;
    stats.p25 = pick_percentile_sorted(samples, 0.25);
    stats.p50 = pick_percentile_sorted(samples, 0.50);
    stats.p75 = pick_percentile_sorted(samples, 0.75);
    stats.p90 = pick_percentile_sorted(samples, 0.90);
    stats.p99 = pick_percentile_sorted(samples, 0.99);
    return stats;
}

static void print_stats(const char* label, const LatencyStats& stats) {
    std::cout << label << ": "
              << "count=" << stats.count
              << " median=" << stats.p50
              << " p25=" << stats.p25
              << " p50=" << stats.p50
              << " p75=" << stats.p75
              << " p90=" << stats.p90
              << " p99=" << stats.p99
              << '\n';
}

static void pin_thread(const std::vector<int>& cpus, size_t thread_index) {
    if (cpus.empty()) {
        return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    const int cpu = cpus[thread_index % cpus.size()];
    CPU_SET(cpu, &cpuset);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

}

template <size_t PayloadBytes, size_t Consumers>
void throughput_case(size_t producers, uint32_t total_msgs) {
    static_assert(sizeof(Payload<PayloadBytes>) == PayloadBytes,
                  "Payload size must match PayloadBytes.");
    auto ring = std::make_unique<MPMC<Payload<PayloadBytes>, Consumers, QUEUE_SIZE>>();
    if (producers == 0) {
        std::cerr << "producers must be > 0\n";
        return;
    }

    const uint32_t base = total_msgs / static_cast<uint32_t>(producers);
    const uint32_t extra = total_msgs % static_cast<uint32_t>(producers);
    std::vector<uint32_t> producer_counts(producers, 0);
    for (size_t pid = 0; pid < producers; ++pid) {
        producer_counts[pid] = base + (pid < extra ? 1u : 0u);
    }

    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> producers_threads;
    std::vector<std::thread> consumers_threads;
    producers_threads.reserve(producers);
    consumers_threads.reserve(Consumers);
    const std::vector<int> pinned_cpus{0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<std::vector<double>> read_latencies(Consumers);
    std::vector<std::vector<double>> write_latencies(producers);
    for (auto& v : read_latencies) {
        v.reserve(total_msgs);
    }
    for (auto& v : write_latencies) {
        v.reserve((total_msgs + producers - 1) / producers);
    }

    for (size_t cid = 0; cid < Consumers; ++cid) {
        consumers_threads.emplace_back([&, cid] {
            pin_thread(pinned_cpus, cid);
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            Payload<PayloadBytes> value{};
            for (uint32_t i = 0; i < total_msgs; ++i) {
                uint64_t t0 = __rdtsc();
                while (!ring->dequeue(cid, &value)) {}
                uint64_t t1 = __rdtsc();
                read_latencies[cid].push_back(cycles_to_ns(t1 - t0));
            }
        });
    }

    for (size_t pid = 0; pid < producers; ++pid) {
        producers_threads.emplace_back([&, pid] {
            pin_thread(pinned_cpus, pid);
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            const uint32_t count = producer_counts[pid];
            for (uint32_t seq = 0; seq < count; ++seq) {
                auto value = make_payload<PayloadBytes>(pid, seq);
                uint64_t t0 = __rdtsc();
                while (!ring->enqueue(value)){}
                uint64_t t1 = __rdtsc();
                write_latencies[pid].push_back(cycles_to_ns(t1 - t0));
            }
        });
    }

    const size_t total_threads = producers + Consumers;
    while (ready.load(std::memory_order_acquire) < total_threads) {
        std::this_thread::yield();
    }

    auto start_time = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);

    for (auto& t : producers_threads) {
        t.join();
    }
    for (auto& t : consumers_threads) {
        t.join();
    }

    (void)start_time;

    std::vector<double> read_all;
    std::vector<double> write_all;
    read_all.reserve(static_cast<size_t>(total_msgs) * Consumers);
    write_all.reserve(total_msgs);
    for (const auto& v : read_latencies) {
        read_all.insert(read_all.end(), v.begin(), v.end());
    }
    for (const auto& v : write_latencies) {
        write_all.insert(write_all.end(), v.begin(), v.end());
    }

    std::sort(read_all.begin(), read_all.end());
    std::sort(write_all.begin(), write_all.end());
    auto read_stats = compute_stats_sorted(read_all);
    auto write_stats = compute_stats_sorted(write_all);
    print_stats("read_ns", read_stats);
    print_stats("write_ns", write_stats);

}

template <size_t PayloadBytes>
void run_payload_case(size_t producers,
                      size_t consumers,
                      uint32_t total_msgs) {
    std::cout << "payload_bytes=" << PayloadBytes << "\n";
    switch (consumers) {
    case 1: throughput_case<PayloadBytes, 1>(producers, total_msgs); break;
    case 2: throughput_case<PayloadBytes, 2>(producers, total_msgs); break;
    case 4: throughput_case<PayloadBytes, 4>(producers, total_msgs); break;
    case 8: throughput_case<PayloadBytes, 8>(producers, total_msgs); break;
    case 16: throughput_case<PayloadBytes, 16>(producers, total_msgs); break;
    case 32: throughput_case<PayloadBytes, 32>(producers, total_msgs); break;
    default:
        std::cerr << "consumers must be 1,2,4,8,16,32\n";
    }
}


int main(int argc, char** argv) {
    size_t producers = 1;
    size_t consumers = 1;
    uint32_t total_msgs = TOTAL_MSGS;

    if (argc >= 2) {
        producers = std::strtoul(argv[1], nullptr, 10);
    }
    if (argc >= 3) {
        consumers = std::strtoul(argv[2], nullptr, 10);
    }
    if (argc >= 4) {
        total_msgs = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    }


    std::cout << "mpmc_bench: producers=" << producers
              << ", consumers=" << consumers
              << ", total_msgs=" << total_msgs
              << "\n";

    run_payload_case<1>(producers, consumers, total_msgs);
    run_payload_case<2>(producers, consumers, total_msgs);
    run_payload_case<4>(producers, consumers, total_msgs);
    run_payload_case<8>(producers, consumers, total_msgs);
    run_payload_case<16>(producers, consumers, total_msgs);
    run_payload_case<32>(producers, consumers, total_msgs);
    run_payload_case<64>(producers, consumers, total_msgs);
    run_payload_case<128>(producers, consumers, total_msgs);
    run_payload_case<256>(producers, consumers, total_msgs);
}
