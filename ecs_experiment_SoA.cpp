#include <iostream>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"

using Vec2d = Eigen::Vector2d;
class VehiclePool {
 public:
  explicit VehiclePool(const unsigned int num_vehicles) {
    posx_vec_.reserve(num_vehicles);
    posy_vec_.reserve(num_vehicles);
    velx_vec_.reserve(num_vehicles);
    vely_vec_.reserve(num_vehicles);
  }

  void add_vehicle(double x, double y, double vx, double vy) {
    posx_vec_.emplace_back(x);
    posy_vec_.emplace_back(y);
    velx_vec_.emplace_back(vx);
    vely_vec_.emplace_back(vy);
  }

  void add_random_vehicle() {
    Vec2d pos = Vec2d::Random();
    Vec2d vel = Vec2d::Random();
    add_vehicle(pos[0], pos[1], vel[0], vel[1]);
  }

  void move_all(float dt) {
    for (size_t i = 0; i < posx_vec_.size(); ++i) {
      posx_vec_[i] += velx_vec_[i] * dt;
      posy_vec_[i] += vely_vec_[i] * dt;
    }
  }

  void move_range(size_t start_idx, size_t end_idx, float dt) {
    for (size_t i = start_idx; i < end_idx; ++i) {
      posx_vec_[i] += velx_vec_[i] * dt;
      posy_vec_[i] += vely_vec_[i] * dt;
    }
  }

  size_t size() const { return posx_vec_.size(); }

 private:
  std::vector<double> posx_vec_, posy_vec_, velx_vec_, vely_vec_;
};

class World {
 public:
  World(const unsigned int num_vehicles, const float curr_time = 0.0f,
        const float time_step = 0.01f, const float duration = 1.0f)
      : vehicles_(num_vehicles),
        curr_time_(curr_time),
        time_step_(time_step),
        duration_(duration) {
    for (int i = 0; i < num_vehicles; ++i) {
      vehicles_.add_random_vehicle();
    }
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

  size_t num_vehicles() const { return vehicles_.size(); }

 private:
  VehiclePool vehicles_;
  float curr_time_, time_step_, duration_;
};

const auto n_threads = std::thread::hardware_concurrency();
constexpr unsigned int scale_factor = 1024;

static void BM_World_Tick_SoA(benchmark::State& state) {
  const int num_vehicles = static_cast<int>(state.range(0));
  // std::cout << "Number of vehicles: " << num_vehicles << std::endl;
  constexpr float time_step = 0.01f;
  constexpr float duration = 1.0f;

  for (auto _ : state) {
    std::vector<World> worlds;
    worlds.reserve(n_threads * scale_factor);
    for (auto i = 0; i < n_threads; ++i) {
      worlds.emplace_back(num_vehicles, time_step, duration);
    }
    for (auto& world : worlds) {
      world.tick_all();
    }
  }
  state.SetItemsProcessed(state.iterations() * num_vehicles);
}

static void BM_World_Tick_Threads_SoA(benchmark::State& state) {
  const int num_vehicles = static_cast<int>(state.range(0));
  // std::cout << "Number of vehicles: " << num_vehicles << std::endl;
  // std::cout << "Number of threads: " << n_threads << std::endl;
  constexpr float time_step = 0.01f;
  constexpr float duration = 1.0f;

  for (auto _ : state) {
    std::vector<World> worlds;
    worlds.reserve(n_threads * scale_factor);
    for (size_t i = 0; i < n_threads; ++i) {
      worlds.emplace_back(num_vehicles, time_step, duration);
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t tid = 0; tid < n_threads; ++tid) {
      threads.emplace_back([&worlds, tid]() {
        for (size_t i = tid; i < worlds.size(); i += n_threads) {
          worlds[i].tick_all();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  state.SetItemsProcessed(state.iterations() * num_vehicles);
}

BENCHMARK(BM_World_Tick_SoA)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);
BENCHMARK(BM_World_Tick_Threads_SoA)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);
BENCHMARK_MAIN();