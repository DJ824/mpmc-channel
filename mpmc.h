#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <type_traits>
#include <utility>

template <typename T, size_t Consumers, size_t Capacity>
class MPMC {
    static_assert(Consumers > 0, "Consumers must be greater than 0.");
    static_assert(Consumers < 64, "Consumers must fit in the bitmap.");
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two.");
    static_assert(alignof(T) <= 64, "T alignment must be <= 64.");
    static_assert(std::is_copy_constructible_v<T>, "T must be copy constructible.");
    static_assert(std::is_copy_assignable_v<T>, "T must be copy assignable.");

    static constexpr uint64_t bitmap_mask = (1ull << Consumers) - 1;
    static constexpr uint64_t empty_state = bitmap_mask;
    static constexpr size_t idx_mask = Capacity - 1;

    using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

    struct alignas(64) Slot {
        alignas(64) std::atomic<uint64_t> seq{0};
        alignas(64) std::atomic<uint64_t> bitmap{empty_state};
        alignas(64) Storage storage;

        T* value_ptr() {
            return std::launder(reinterpret_cast<T*>(&storage));
        }

        const T* value_ptr() const {
            return std::launder(reinterpret_cast<const T*>(&storage));
        }

        void* storage_ptr() {
            return &storage;
        }
    };

    struct alignas(64) Head {
        uint64_t value{0};
    };

    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) std::array<Head, Consumers> heads_{};
    alignas(64) std::array<Slot, Capacity> slots_{};

    static_assert(sizeof(Head) == 64, "Head must be 64 bytes.");
    static_assert(sizeof(Slot) % 64 == 0, "Slot size must be a multiple of 64 bytes.");

public:
    MPMC() {
        for (size_t i = 0; i < Capacity; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
            slots_[i].bitmap.store(empty_state, std::memory_order_relaxed);
        }
    }

    ~MPMC() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_t i = 0; i < Capacity; ++i) {
                slots_[i].value_ptr()->~T();
            }
        }
    }

    template <typename U,
              typename = std::enable_if_t<std::is_constructible_v<T, U&&>>>
    bool enqueue(U&& value) {
        for (;;) {
            uint64_t seq = tail_.fetch_add(1, std::memory_order_relaxed);
            Slot& slot = slots_[seq & idx_mask];

            while (slot.seq.load(std::memory_order_acquire) != seq) {
                _mm_pause();
            }

            if constexpr (std::is_trivially_copyable_v<T>) {
                if constexpr (std::is_same_v<std::decay_t<U>, T>) {
                    std::memcpy(slot.storage_ptr(), &value, sizeof(T));
                }
                else {
                    T tmp(std::forward<U>(value));
                    std::memcpy(slot.storage_ptr(), &tmp, sizeof(T));
                }
            }
            else {
                ::new(slot.storage_ptr()) T(std::forward<U>(value));
            }

            slot.bitmap.store(0, std::memory_order_relaxed);
            slot.seq.store(seq + 1, std::memory_order_release);
            return true;
        }
    }

    bool dequeue(size_t consumer_id, T* out) {
        if (!out || consumer_id >= Consumers) {
            return false;
        }

        uint64_t bit = 1ull << consumer_id;
        uint64_t seq = heads_[consumer_id].value;

        for (;;) {
            Slot& slot = slots_[seq & idx_mask];
            while (slot.seq.load(std::memory_order_acquire) != seq + 1) {
                _mm_pause();
            }

            *out = *slot.value_ptr();
            uint64_t prev = slot.bitmap.fetch_or(bit, std::memory_order_relaxed);
            if ((prev | bit) == bitmap_mask) {
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    slot.value_ptr()->~T();
                }
                slot.bitmap.store(empty_state, std::memory_order_relaxed);
                slot.seq.store(seq + Capacity, std::memory_order_release);
            }
            heads_[consumer_id].value = seq + 1;
            return true;
        }
    }
};
