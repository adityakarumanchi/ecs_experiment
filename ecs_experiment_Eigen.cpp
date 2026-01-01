#include <chrono>
#include <iostream>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#define EIGEN_RUNTIME_NO_MALLOC
#include "Eigen/Dense"
#include "benchmark/benchmark.h"

template <unsigned int num_vehicles>
class VehiclePool {
 public:
  void add_random_vehicles() {
    posx_vec_ = Eigen::Matrix<double, num_vehicles, 1>::Random();
    posy_vec_ = Eigen::Matrix<double, num_vehicles, 1>::Random();
    velx_vec_ = Eigen::Matrix<double, num_vehicles, 1>::Random();
    vely_vec_ = Eigen::Matrix<double, num_vehicles, 1>::Random();
  }

  void move_all(float dt) {
    // See https://github.com/stulp/eigenrealtime regarding the next 3 lines
    Eigen::internal::set_is_malloc_allowed(false);
    posx_vec_.noalias() += velx_vec_ * dt;
    posy_vec_.noalias() += vely_vec_ * dt;
    // posx_vec_ += velx_vec_ * dt;
    // posy_vec_ += vely_vec_ * dt;
  }

  void move_range(size_t start_idx, size_t end_idx, float dt) {
    Eigen::internal::set_is_malloc_allowed(false);
    posx_vec_.block(start_idx, 0, end_idx - start_idx, 1).noalias() +=
        velx_vec_.block(start_idx, 0, end_idx - start_idx, 1) * dt;
    posy_vec_.block(start_idx, 0, end_idx - start_idx, 1).noalias() +=
        vely_vec_.block(start_idx, 0, end_idx - start_idx, 1) * dt;
  }

 private:
  Eigen::Matrix<double, num_vehicles, 1> posx_vec_, posy_vec_, velx_vec_,
      vely_vec_;
};

template <const unsigned int num_vehicles>
class World {
 public:
  World(const float time_step = 0.01f, const float duration = 1.0f,
        const float curr_time = 0.0f)
      : time_step_(time_step), duration_(duration), curr_time_(curr_time) {
    vehicles_.add_random_vehicles();
  }

  void tick_all() {
    while (curr_time_ < duration_) {
      vehicles_.move_all(time_step_);
      curr_time_ += time_step_;
    }
  }

  void tick_range(size_t start_idx, size_t end_idx) {
    while (curr_time_ < duration_) {
      vehicles_.move_range(start_idx, end_idx, time_step_);
      curr_time_ += time_step_;
    }
  }

 private:
  float time_step_, duration_, curr_time_;
  VehiclePool<num_vehicles> vehicles_;
};

const auto n_threads = std::thread::hardware_concurrency();
constexpr unsigned int scale_factor = 1024;
const auto n_worlds = n_threads * scale_factor;
constexpr float time_step = 0.01f;
constexpr float duration = 100.0f;

template <unsigned int num_vehicles>
static void BM_World_Tick(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<World<num_vehicles>> worlds;
    worlds.reserve(n_worlds);
    for (size_t i = 0; i < n_worlds; ++i) {
      worlds.emplace_back(time_step, duration);
    }
    for (auto& world : worlds) {
      world.tick_all();
    }
  }
  state.SetItemsProcessed(state.iterations() * num_vehicles * n_worlds);
}

template <unsigned int num_vehicles>
static void BM_World_Tick_Threads(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<World<num_vehicles>> worlds;
    worlds.reserve(n_worlds);
    for (size_t i = 0; i < n_worlds; ++i) {
      worlds.emplace_back(time_step, duration);
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t tid = 0; tid < n_threads; ++tid) {
      threads.emplace_back([&worlds, tid]() {
        for (size_t i = tid; i < n_worlds; i += n_threads) {
          worlds[i].tick_all();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  state.SetItemsProcessed(state.iterations() * num_vehicles * n_worlds);
}

BENCHMARK(BM_World_Tick<1000>)->Name("BM_World_Tick/1000");
BENCHMARK(BM_World_Tick<10000>)->Name("BM_World_Tick/10000");
BENCHMARK(BM_World_Tick_Threads<1000>)->Name("BM_World_Tick_Threads/1000");
BENCHMARK(BM_World_Tick_Threads<10000>)->Name("BM_World_Tick_Threads/10000");
BENCHMARK_MAIN();