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

#include "open_spiel/games/hsr.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace hsr {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"hsr",
    /*long_name=*/"HSR",
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
  return std::shared_ptr<const Game>(new HSRGame(params));
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
    case CellState::kCross:
      return "x";
    case CellState::kNought:
      return "o";
    default:
      SpielFatalError("Unknown state.");
  }
}

void HSRState::DoApplyAction(Action move) {

  if (current_player_ == 0){

    SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
    board_[move] = PlayerToState(CurrentPlayer());
    if (board_[move - 1] == CellState::kCross && board_[move + 1] == CellState::kCross) {
      outcome_ = Player{0};
      return;
    } else if (board_[move - 1] == CellState::kCross && move == kNumCells - 1) {
      outcome_ = Player{0};
      return;
    } else if (move == 0 && board_[move + 1] == CellState::kCross) {
      outcome_ = Player{0};
      return;
    }
    previous_move_ = move;
    num_moves_ += 1;
    current_tests_ += 1;

  } else if (current_player_ == 1) {
    current_part_ = move;
    if (current_part_ == 1) {
      for (int cell = 0; cell < previous_move_; ++cell) {
        board_[cell] = CellState::kCross;
      }
    } else if (current_part_ == 0) {
      for (int cell = previous_move_ + 1; cell < kNumCells; ++cell) {
        board_[cell] = CellState::kCross;
      }
      current_jars_ -= 1;
    }
  }

  if (IsFull()) { outcome_ = Player{0}; return;}
  if (current_tests_ >= kTests) { outcome_ = Player{1}; return;}
  if (current_jars_ == 0) { outcome_ = Player{1}; return;}

  current_player_ = 1 - current_player_;
}

std::vector<Action> HSRState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
//  int move_op [] = { 0,1 };

  if (current_player_ == 0){
    if (current_part_ == 0) {
      if (previous_move_ == 1) {
        moves.push_back(0);
      } else {
        for (int cell = 1; cell < previous_move_; ++cell) {
          if (board_[cell] == CellState::kEmpty) {
            moves.push_back(cell);
          }
        }
      }
    } else if (current_part_ == 1) {
      if (previous_move_ == kNumCells - 2) {
        moves.push_back(kNumCells - 1);
      } else {
        for (int cell = previous_move_ + 1; cell < kNumCells - 1; ++cell) {
          if (board_[cell] == CellState::kEmpty) {
            moves.push_back(cell);
          }
        }
      }
     }
//     if (previous_move_ == 1 && board_[0] == CellState::kEmpty) {
//        moves.push_back(0);
//     } else if (previous_move_ == kNumCells - 2 && board_[kNumCells - 1] == CellState::kEmpty) {
//        moves.push_back(kNumCells - 1);
//     }

  } else if (current_player_ == 1) {
    if (board_[previous_move_ - 1] == CellState::kEmpty) {
      moves.push_back(0);
    }
    if (board_[previous_move_ + 1] == CellState::kEmpty) {
      moves.push_back(1);
    }
  }
  return moves;
}

std::string HSRState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

bool HSRState::IsFull() const { return num_moves_ == kNumCells; }

HSRState::HSRState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string HSRState::ToString() const {
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

bool HSRState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> HSRState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string HSRState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string HSRState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void HSRState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void HSRState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
}

std::unique_ptr<State> HSRState::Clone() const {
  return std::unique_ptr<State>(new HSRState(*this));
}

HSRGame::HSRGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace hsr
}  // namespace open_spiel
