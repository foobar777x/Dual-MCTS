// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_HSR_H_
#define OPEN_SPIEL_GAMES_HSR_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Simple game of HSR:
// https://en.wikipedia.org/wiki/hsr
//
// Parameters: none

namespace open_spiel {
namespace hsr {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRows = 1;
inline constexpr int kNumCols = 127;                      // this is the n (number of rungs)
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 1 + kNumPlayers;         // empty, 'x', 'o'.

inline constexpr int kTests = 7;                        // this is the q (number of tests)
inline constexpr int kJars = 7;                         // this is the k (number of jars)

// states calculated based on the possible combinations of actions (number of tests). Note that this will be
// an optimistic estimate
inline constexpr int kNumberStates = 6585;

// State of a cell.
enum class CellState {
  kEmpty,
  kCross,
  kNought
};

// State of an in-play game.
class HSRState : public State {
 public:
  HSRState(std::shared_ptr<const Game> game);

  HSRState(const HSRState&) = default;
  HSRState& operator=(const HSRState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  CellState BoardAt(int cell) const { return board_[cell]; }
  CellState BoardAt(int row, int column) const {
    return board_[row * kNumCols + column];
  }

 protected:
  std::array<CellState, kNumCells> board_;
  void DoApplyAction(Action move) override;

 private:
  bool IsFull() const;                // Is the board full?
  Player current_player_ = 0;         // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int current_part_ = 0;
  int current_tests_ = 0;             // Initialize the number of tests with zero
  int current_jars_ = kJars;          // Initialize the number of jars with the maximum allowed number
  int previous_move_ = kNumCells - 1;     // Previous move is initialized with the last cell value
  int num_moves_ = 0;
};

// Game object.
class HSRGame : public Game {
 public:
  explicit HSRGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumCells; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new HSRState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kCellStates, kNumRows, kNumCols};
  }
  int MaxGameLength() const override { return kNumCells; }
};

CellState PlayerToState(Player player);
std::string StateToString(CellState state);

inline std::ostream& operator<<(std::ostream& stream, const CellState& state) {
  return stream << StateToString(state);
}

}  // namespace hsr
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HSR_H_
