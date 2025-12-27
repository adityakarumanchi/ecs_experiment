#include <cstddef>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"

using Vec2d = Eigen::Vector2d;
class Vehicle {
 public:
  explicit Vehicle() : posx_(0), posy_(0), velx_(0), vely_(0) {}
  explicit Vehicle(const double x, const double y) : posx_(x), posy_(y) {}
  explicit Vehicle(const Vec2d& pos) : posx_(pos[0]), posy_(pos[1]) {}
  explicit Vehicle(const Vec2d& pos, const Vec2d& vel)
      : posx_(pos[0]), posy_(pos[1]), velx_(vel[0]), vely_(vel[1]) {}
  explicit Vehicle(const double x, const double y, const double vx,
                   const double vy)
      : posx_(x), posy_(y), velx_(vx), vely_(vy) {}

  void move(float dt) {
    posx_ += velx_ * dt;
    posy_ += vely_ * dt;
  }
  void set_velocity(const double vx, const double vy) {
    velx_ = vx;
    vely_ = vy;
  }
  Vec2d get_position() const { return Vec2d(posx_, posy_); }

 private:
  double posx_, posy_, velx_, vely_;
};

class World {
 public:
  World(const unsigned int num_vehicles, const float curr_time = 0.0f,
        const float time_step = 0.01f, const float duration = 1.0f)
      : curr_time_(curr_time), time_step_(time_step), duration_(duration) {
    for (size_t i = 0; i < num_vehicles; ++i) {
      vehicles_.emplace_back(Vec2d::Random() /*For position*/,
                             Vec2d::Random() /*For velocity*/);
    }
  }
  void tick() {
    while (curr_time_ < duration_) {
      for (auto& veh : vehicles_) {
        veh.move(time_step_);
      }
      curr_time_ += time_step_;
    }
  }

 private:
  std::vector<Vehicle> vehicles_;
  float curr_time_, time_step_, duration_;
};

const auto n_threads = std::thread::hardware_concurrency();
constexpr unsigned int scale_factor = 1024;

static void BM_World_Tick(benchmark::State& state) {
  const int num_vehicles = static_cast<int>(state.range(0));
  // std::cout << "Number of vehicles: " << num_vehicles << std::endl;
  constexpr float time_step = 0.01f;
  constexpr float duration = 1.0f;

  for (auto _ : state) {
    std::vector<World> worlds;
    worlds.reserve(n_threads * scale_factor);
    for (size_t i = 0; i < n_threads; ++i) {
      worlds.emplace_back(num_vehicles, time_step, duration);
    }
    for (auto& world : worlds) {
      world.tick();
    }
  }
  state.SetItemsProcessed(state.iterations() * num_vehicles);
}

static void BM_World_Tick_Threads(benchmark::State& state) {
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
          worlds[i].tick();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  state.SetItemsProcessed(state.iterations() * num_vehicles);
}

BENCHMARK(BM_World_Tick)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);
BENCHMARK(BM_World_Tick_Threads)
    ->Arg(1)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);
BENCHMARK_MAIN();
