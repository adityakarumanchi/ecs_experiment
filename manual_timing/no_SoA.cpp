#include <cstddef>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "Eigen/Dense"
// #include "benchmark/benchmark.h"

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

static void BM_World_Tick(unsigned int n_vehicles, const unsigned int n_worlds,
                          const unsigned int n_repetitions,
                          float time_step = 0.01f, float duration = 1.0f) {
  const auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t n = 0; n < n_repetitions; n++) {
    std::vector<World> worlds;
    worlds.reserve(n_worlds);
    for (size_t i = 0; i < n_worlds; ++i) {
      worlds.emplace_back(n_vehicles, time_step, duration);
    }
    for (auto& world : worlds) {
      world.tick();
    }
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time)
          .count();
  std::cout << "Number of vehicles: " << n_vehicles
            << ", number of worlds: " << n_worlds
            << ", time taken (ms): " << duration_ms << std::endl;
}

static void BM_World_Tick_Threads(unsigned int n_vehicles,
                                  const unsigned int n_worlds,
                                  const unsigned int n_threads,
                                  const unsigned int n_repetitions,
                                  float time_step = 0.01f,
                                  float duration = 1.0f) {
  const auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t n = 0; n < n_repetitions; n++) {
    std::vector<World> worlds;
    worlds.reserve(n_worlds);
    for (size_t i = 0; i < n_worlds; ++i) {
      worlds.emplace_back(n_vehicles, time_step, duration);
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t tid = 0; tid < n_threads; ++tid) {
      threads.emplace_back([&worlds, tid, n_worlds, n_threads]() {
        for (size_t i = tid; i < n_worlds; i += n_threads) {
          worlds[i].tick();
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time)
          .count();
  std::cout << "Number of vehicles: " << n_vehicles
            << ", number of worlds: " << n_worlds
            << ", time taken (ms): " << duration_ms << std::endl;
}

int main() {
  const auto num_threads = std::thread::hardware_concurrency();
  constexpr unsigned int scale_factor = 1024;
  const auto num_worlds = num_threads * scale_factor;
  const unsigned int num_repetitions = 1000;
  constexpr float time_step = 0.01f;
  constexpr float duration = 100.0f;
  BM_World_Tick(1, num_worlds, num_repetitions, time_step, duration);
  BM_World_Tick(10, num_worlds, num_repetitions, time_step, duration);
  BM_World_Tick(100, num_worlds, num_repetitions, time_step, duration);
  BM_World_Tick(1000, num_worlds, num_repetitions, time_step, duration);
  BM_World_Tick_Threads(1, num_worlds, num_threads, num_repetitions, time_step,
                        duration);
  BM_World_Tick_Threads(10, num_worlds, num_threads, num_repetitions, time_step,
                        duration);
  BM_World_Tick_Threads(100, num_worlds, num_threads, num_repetitions,
                        time_step, duration);
  BM_World_Tick_Threads(1000, num_worlds, num_threads, num_repetitions,
                        time_step, duration);
  return 0;
}
