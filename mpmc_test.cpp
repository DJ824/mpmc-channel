#include "mpmc.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

constexpr size_t QUEUE_SIZE = 1u << 16;
constexpr uint32_t kTotalMsgs = 1'000'000;
constexpr uint32_t kSkewTotalMsgs = 200'000;
constexpr uint32_t kJitterTotalMsgs = 100'000;
constexpr uint32_t kFifoTotalMsgs = static_cast<uint32_t>(QUEUE_SIZE * 2);
constexpr size_t kSmallQueue = 8;

struct alignas(64) Payload {
    uint64_t producer{0};
    uint64_t seq{0};
    uint8_t pad[48]{};
};

static_assert(sizeof(Payload) == 64, "Payload must be 64 bytes.");

struct CountedPayload {
    uint64_t producer{0};
    uint64_t seq{0};

    static std::atomic<int64_t> live;

    CountedPayload() {
        live.fetch_add(1, std::memory_order_relaxed);
    }

    CountedPayload(uint64_t producer_id, uint64_t seq_id)
        : producer(producer_id), seq(seq_id) {
        live.fetch_add(1, std::memory_order_relaxed);
    }

    CountedPayload(const CountedPayload& other)
        : producer(other.producer), seq(other.seq) {
        live.fetch_add(1, std::memory_order_relaxed);
    }

    CountedPayload& operator=(const CountedPayload&) = default;

    ~CountedPayload() {
        live.fetch_sub(1, std::memory_order_relaxed);
    }
};

std::atomic<int64_t> CountedPayload::live{0};

struct DelayConfig {
    bool enable_producer;
    size_t slow_producer_id;
    uint32_t producer_delay_us;
    uint32_t producer_start_delay_us;
    bool enable_consumer;
    size_t slow_consumer_id;
    uint32_t consumer_delay_us;
    uint32_t consumer_start_delay_us;
    uint32_t producer_jitter_us;
    uint32_t consumer_jitter_us;
};

template <size_t Consumers, size_t Capacity>
void run_case(size_t producers, uint32_t total_msgs, const DelayConfig* delays) {
    std::cout << "case (consumers=" << Consumers
              << ", producers=" << producers
              << ", total_msgs=" << total_msgs
              << ", capacity=" << Capacity << ")\n";

    auto ring = std::make_unique<MPMC<Payload, Consumers, Capacity>>();
    if (!producers) {
        return;
    }

    const size_t total_msgs_count = total_msgs;
    std::vector seen(Consumers, std::vector<uint8_t>(total_msgs_count, 0));

    std::vector<uint32_t> producer_counts(producers, 0);
    std::vector<size_t> producer_offsets(producers, 0);
    const uint32_t base = total_msgs / static_cast<uint32_t>(producers);
    const uint32_t extra = total_msgs % static_cast<uint32_t>(producers);
    size_t offset = 0;
    for (size_t pid = 0; pid < producers; ++pid) {
        uint32_t count = base + (pid < extra ? 1u : 0u);
        producer_counts[pid] = count;
        producer_offsets[pid] = offset;
        offset += count;
    }
    assert(offset == total_msgs_count);

    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> producers_threads;
    std::vector<std::thread> consumers_threads;
    producers_threads.reserve(producers);
    consumers_threads.reserve(Consumers);

    for (size_t cid = 0; cid < Consumers; ++cid) {
        consumers_threads.emplace_back([&, cid] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::mt19937_64 jitter_rng(0xd1342543de82ef95ULL ^ cid);
            std::uniform_int_distribution<uint32_t> jitter_dist(
                0u, delays ? delays->consumer_jitter_us : 0u);
            if (delays && delays->enable_consumer && cid == delays->slow_consumer_id &&
                delays->consumer_start_delay_us) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(delays->consumer_start_delay_us));
            }
            Payload value{};
            for (size_t i = 0; i < total_msgs_count; ++i) {
                while (!ring->dequeue(cid, &value)){}
                uint32_t pid = static_cast<uint32_t>(value.producer);
                uint32_t seq = static_cast<uint32_t>(value.seq);
                assert(pid < producers);
                assert(seq < producer_counts[pid]);
                size_t idx = producer_offsets[pid] + seq;
                assert(seen[cid][idx] == 0);
                seen[cid][idx] = 1;
                if (delays && delays->enable_consumer && cid == delays->slow_consumer_id) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(delays->consumer_delay_us));
                }
                if (delays && delays->consumer_jitter_us) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(jitter_dist(jitter_rng)));
                }
            }
        });
    }

    for (uint32_t pid = 0; pid < producers; ++pid) {
        producers_threads.emplace_back([&, pid] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::mt19937_64 jitter_rng(0x9e3779b97f4a7c15ULL ^ pid);
            std::uniform_int_distribution<uint32_t> jitter_dist(
                0u, delays ? delays->producer_jitter_us : 0u);
            if (delays && delays->enable_producer && pid == delays->slow_producer_id &&
                delays->producer_start_delay_us) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(delays->producer_start_delay_us));
            }
            const uint32_t count = producer_counts[pid];
            std::vector<uint32_t> values(count);
            std::iota(values.begin(), values.end(), 0u);
            std::mt19937_64 rng(0x5bf03635d31c8f73ULL ^ pid);
            std::shuffle(values.begin(), values.end(), rng);
            for (uint32_t seq : values) {
                Payload value{};
                value.producer = pid;
                value.seq = seq;
                while (!ring->enqueue(value)){}
                if (delays && delays->enable_producer && pid == delays->slow_producer_id) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(delays->producer_delay_us));
                }
                if (delays && delays->producer_jitter_us) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(jitter_dist(jitter_rng)));
                }
            }
        });
    }

    const size_t total_threads = producers + Consumers;
    while (ready.load(std::memory_order_acquire) < total_threads) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    std::atomic<bool> done{false};
    uint64_t timeout_sec = 20;
    if (delays) {
        const uint64_t max_consumer_delay =
            static_cast<uint64_t>(delays->consumer_delay_us) +
            static_cast<uint64_t>(delays->consumer_jitter_us);
        const uint64_t max_producer_delay =
            static_cast<uint64_t>(delays->producer_delay_us) +
            static_cast<uint64_t>(delays->producer_jitter_us);
        const uint64_t max_delay = std::max(max_consumer_delay, max_producer_delay);
        if (max_delay > 0) {
            const uint64_t start_delay =
                static_cast<uint64_t>(delays->consumer_start_delay_us) +
                static_cast<uint64_t>(delays->producer_start_delay_us);
            const uint64_t est_sec =
                (total_msgs_count * max_delay + start_delay) / 1'000'000;
            timeout_sec = std::max<uint64_t>(timeout_sec, est_sec * 2 + 10);
        }
    }

    std::thread watchdog([&] {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(timeout_sec);
        while (!done.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!done.load(std::memory_order_acquire)) {
            std::cerr << "timeout: consumers/producers did not finish\n";
            std::abort();
        }
    });

    for (auto& t : producers_threads) {
        t.join();
    }
    for (auto& t : consumers_threads) {
        t.join();
    }

    for (size_t cid = 0; cid < Consumers; ++cid) {
        for (size_t i = 0; i < total_msgs_count; ++i) {
            assert(seen[cid][i] == 1);
        }
    }
    done.store(true, std::memory_order_release);
    watchdog.join();
}

void test_invalid_args() {
    MPMC<Payload, 2, kSmallQueue> ring;
    Payload value{};
    assert(!ring.dequeue(2, &value));
    assert(!ring.dequeue(0, nullptr));
}

void test_empty_and_full() {
    MPMC<Payload, 1, 4> ring;
    Payload value{};
    assert(!ring.dequeue(0, &value));
    for (uint32_t i = 0; i < 4; ++i) {
        value.seq = i;
        assert(ring.enqueue(value));
    }
    assert(!ring.enqueue(value));
}

void test_non_trivial_cleanup() {
    CountedPayload::live.store(0, std::memory_order_relaxed);
    {
        constexpr size_t consumers = 2;
        constexpr size_t capacity = 64;
        constexpr uint32_t total_msgs = 50'000;
        auto ring = std::make_unique<MPMC<CountedPayload, consumers, capacity>>();

        std::atomic<size_t> ready{0};
        std::atomic<bool> start{false};
        std::vector<std::thread> producers;
        std::vector<std::thread> consumers_threads;
        producers.reserve(1);
        consumers_threads.reserve(consumers);

        for (size_t cid = 0; cid < consumers; ++cid) {
            consumers_threads.emplace_back([&, cid] {
                ready.fetch_add(1, std::memory_order_acq_rel);
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                CountedPayload value{};
                for (uint32_t i = 0; i < total_msgs; ++i) {
                    while (!ring->dequeue(cid, &value)) {}
                }
            });
        }

        producers.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint32_t seq = 0; seq < total_msgs; ++seq) {
                CountedPayload value{0, seq};
                while (!ring->enqueue(value)) {}
            }
        });

        const size_t total_threads = consumers + 1;
        while (ready.load(std::memory_order_acquire) < total_threads) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);

        for (auto& t : producers) {
            t.join();
        }
        for (auto& t : consumers_threads) {
            t.join();
        }
    }
    auto live = CountedPayload::live.load(std::memory_order_relaxed);
    if (live != 0) {
        std::cerr << "CountedPayload live objects: " << live << "\n";
    }
    assert(live == 0);
}

template <size_t Consumers>
void run_global_fifo_case(size_t producers, uint32_t total_msgs) {
    std::cout << "GLOBAL FIFO case (consumers=" << Consumers
              << ", producers=" << producers
              << ", total_msgs=" << total_msgs << ")\n";

    auto ring = std::make_unique<MPMC<Payload, Consumers, kSmallQueue>>();

    std::atomic<uint64_t> gid{0};
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};

    std::vector<std::thread> prod_threads;
    std::vector<std::thread> cons_threads;
    std::vector<std::vector<uint64_t>> order(Consumers);

    for (size_t cid = 0; cid < Consumers; ++cid) {
        cons_threads.emplace_back([&, cid] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {}

            Payload v{};
            std::vector<uint64_t> local;
            local.reserve(total_msgs);
            for (uint32_t i = 0; i < total_msgs; ++i) {
                while (!ring->dequeue(cid, &v)) {}
                assert(v.seq < total_msgs);
                local.push_back(v.seq);
            }
            order[cid] = std::move(local);
        });
    }

    for (size_t pid = 0; pid < producers; ++pid) {
        prod_threads.emplace_back([&, pid] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {}

            for (;;) {
                uint64_t id = gid.fetch_add(1, std::memory_order_relaxed);
                if (id >= total_msgs) break;

                Payload v{};
                v.producer = pid;
                v.seq = id;

                while (!ring->enqueue(v)) {}
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < producers + Consumers) {}
    start.store(true, std::memory_order_release);

    for (auto& t : prod_threads) t.join();
    for (auto& t : cons_threads) t.join();

    for (size_t cid = 0; cid < Consumers; ++cid) {
        const auto& local = order[cid];
        assert(local.size() == total_msgs);
        std::vector<uint8_t> seen(total_msgs, 0);
        for (uint64_t id : local) {
            assert(id < total_msgs);
            assert(seen[id] == 0);
            seen[id] = 1;
        }
        for (uint32_t i = 0; i < total_msgs; ++i) {
            assert(seen[i] == 1);
        }
    }

    for (size_t cid = 1; cid < Consumers; ++cid) {
        const auto& reference = order[0];
        const auto& local = order[cid];
        for (uint32_t i = 0; i < total_msgs; ++i) {
            assert(local[i] == reference[i]);
        }
    }
}


template <size_t Consumers>
void run_fifo_case(uint32_t total_msgs) {
    std::cout << "fifo case (consumers=" << Consumers
              << ", total_msgs=" << total_msgs << ")\n";

    auto ring = std::make_unique<MPMC<Payload, Consumers, QUEUE_SIZE>>();
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> done{false};

    std::vector<std::thread> producers_threads;
    std::vector<std::thread> consumers_threads;
    producers_threads.reserve(1);
    consumers_threads.reserve(Consumers);

    for (size_t cid = 0; cid < Consumers; ++cid) {
        consumers_threads.emplace_back([&, cid] {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            Payload value{};
            uint64_t expected = 0;
            for (uint32_t i = 0; i < total_msgs; ++i) {
                while (!ring->dequeue(cid, &value)) {}
                assert(value.producer == 0);
                assert(value.seq == expected);
                ++expected;
            }
        });
    }

    producers_threads.emplace_back([&] {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (uint32_t seq = 0; seq < total_msgs; ++seq) {
            Payload value{};
            value.producer = 0;
            value.seq = seq;
            while (!ring->enqueue(value)) {}
        }
    });

    const size_t total_threads = 1 + Consumers;
    while (ready.load(std::memory_order_acquire) < total_threads) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    std::thread watchdog([&] {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(10);
        while (!done.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!done.load(std::memory_order_acquire)) {
            std::cerr << "timeout: fifo test did not finish\n";
            std::abort();
        }
    });

    for (auto& t : producers_threads) {
        t.join();
    }
    for (auto& t : consumers_threads) {
        t.join();
    }
    done.store(true, std::memory_order_release);
    watchdog.join();
}

int main() {
    // run_fifo_case<1>(kFifoTotalMsgs);
    // run_fifo_case<4>(kFifoTotalMsgs);
    // run_fifo_case<8>(kFifoTotalMsgs);
    run_global_fifo_case<1>(1, kFifoTotalMsgs);
    run_global_fifo_case<2>(2, kFifoTotalMsgs);
    run_global_fifo_case<4>(4, kFifoTotalMsgs);


    const size_t counts[] = {1, 2, 4, 8, 16, 32};
    for (size_t producers : counts) {
        run_case<1, QUEUE_SIZE>(producers, kTotalMsgs, nullptr);
        run_case<2, QUEUE_SIZE>(producers, kTotalMsgs, nullptr);
        run_case<4, QUEUE_SIZE>(producers, kTotalMsgs, nullptr);
        run_case<8, QUEUE_SIZE>(producers, kTotalMsgs, nullptr);
        run_case<16, QUEUE_SIZE>(producers, kTotalMsgs, nullptr);
        run_case<32, QUEUE_SIZE>(producers, kTotalMsgs, nullptr);
    }

    std::cout << "running delayed cases" << std::endl;

    DelayConfig skewed{
        true,
        2,
        5,
        0,
        true,
        6,
        5,
        0,
        0,
        0
    };
    run_case<8, QUEUE_SIZE>(8, kSkewTotalMsgs, &skewed);
    run_case<16, QUEUE_SIZE>(16, kSkewTotalMsgs, &skewed);
    run_case<32, QUEUE_SIZE>(32, kSkewTotalMsgs, &skewed);

    DelayConfig slow_producer{
        true,
        0,
        10,
        0,
        false,
        0,
        0,
        0,
        0,
        0
    };
    run_case<8, QUEUE_SIZE>(8, kSkewTotalMsgs, &slow_producer);

    DelayConfig slow_consumer{
        false,
        0,
        0,
        0,
        true,
        0,
        10,
        0,
        0,
        0
    };
    run_case<8, QUEUE_SIZE>(8, kSkewTotalMsgs, &slow_consumer);

    DelayConfig late_consumer{
        false,
        0,
        0,
        0,
        true,
        0,
        0,
        500,
        0,
        0
    };
    run_case<8, QUEUE_SIZE>(8, kSkewTotalMsgs, &late_consumer);

    DelayConfig jitter_all{
        false,
        0,
        0,
        0,
        false,
        0,
        0,
        0,
        10,
        10
    };
    run_case<8, QUEUE_SIZE>(8, kJitterTotalMsgs, &jitter_all);

    std::cout << "running small queue stress cases" << std::endl;
    run_global_fifo_case<2>(2, kSkewTotalMsgs);
    run_global_fifo_case<4>(4, kSkewTotalMsgs);
    run_case<4, 1024>(8, kSkewTotalMsgs, nullptr);
    run_case<8, 1024>(16, kSkewTotalMsgs, &slow_consumer);
    run_case<8, 1024>(16, kSkewTotalMsgs, &late_consumer);

    std::cout << "mpmc broadcast tests passed\n";
    return 0;
}
