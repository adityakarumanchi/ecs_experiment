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
    }
    for (size_t i = 0; i < posy_vec_.size(); ++i) {
      posy_vec_[i] += vely_vec_[i] * dt;
    }
  }

  void move_range(size_t start_idx, size_t end_idx, float dt) {
    for (size_t i = start_idx; i < end_idx; ++i) {
      posx_vec_[i] += velx_vec_[i] * dt;
    }
    for (size_t i = start_idx; i < end_idx; ++i) {
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
    for (size_t i = 0; i < num_vehicles; ++i) {
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

static void BM_World_Tick(unsigned int n_vehicles, const unsigned int n_worlds,
                          const unsigned int n_repetitions, float time_step,
                          float duration) {
  const auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t n = 0; n < n_repetitions; n++) {
    std::vector<World> worlds;
    worlds.reserve(n_worlds);
    for (size_t i = 0; i < n_worlds; ++i) {
      worlds.emplace_back(n_vehicles, time_step, duration);
    }
    for (auto& world : worlds) {
      world.tick_all();
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
                                  float time_step, float duration) {
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
        for (size_t i = tid; i < worlds.size(); i += n_threads) {
          worlds[i].tick_all();
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