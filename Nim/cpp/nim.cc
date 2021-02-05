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

#include "open_spiel/games/nim.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace nim {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"nim",
    /*long_name=*/"Nim",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new NimGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "o";
    case CellState::kCross:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

void NimState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  int move_count = 0;
  for (int cell = 0; cell <= move; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      board_[cell] = PlayerToState(CurrentPlayer());
      move_count += 1;
//      std::cout << "move " << move << std::endl;
//      std::cout << "move count " << move_count << std::endl;
    }
  }
//  std::cout << "num moves " << num_moves_ << std::endl;
  num_moves_ += move_count;
  if (num_moves_ == 10){
    outcome_ = current_player_;
    }
  current_player_ = 1 - current_player_;
}

std::vector<Action> NimState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  int counter = 0;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
      counter += 1;
//      std::cout << "counter " << counter << std::endl;
//      std::cout << "cell " << cell << std::endl;
      if (counter == 3){
        break;
      }
     }
    }
  return moves;
}

std::string NimState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

//bool NimState::HasLine(Player player) const {
//  CellState c = PlayerToState(player);
//  return (board_[0] == c && board_[1] == c && board_[2] == c) ||
//         (board_[3] == c && board_[4] == c && board_[5] == c) ||
//         (board_[6] == c && board_[7] == c && board_[8] == c) ||
//         (board_[0] == c && board_[3] == c && board_[6] == c) ||
//         (board_[1] == c && board_[4] == c && board_[7] == c) ||
//         (board_[2] == c && board_[5] == c && board_[8] == c) ||
//         (board_[0] == c && board_[4] == c && board_[8] == c) ||
//         (board_[2] == c && board_[4] == c && board_[6] == c);
//}

bool NimState::IsFull() const {return num_moves_ == kNumCells;}

NimState::NimState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string NimState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool NimState::IsTerminal() const {
  return outcome_ != kInvalidPlayer;
}

std::vector<double> NimState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string NimState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string NimState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void NimState::ObservationTensor(Player player,
                                       std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void NimState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
}

std::unique_ptr<State> NimState::Clone() const {
  return std::unique_ptr<State>(new NimState(*this));
}

NimGame::NimGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace nim
}  // namespace open_spiel
