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

#ifndef OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
#define OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/card_abstraction/card_abstraction.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// This is a wrapper around the Annual Computer Poker Competition bot (ACPC)
// environment. See http://www.computerpokercompetition.org/. The code is
// initially available at https://github.com/ethansbrown/acpc
// It is an optional dependency (see install.md for documentation and
// open_spiel/scripts/global_variables.sh to enable this).
//
// It has not been extensively reviewed/tested by the DeepMind OpenSpiel team.
namespace open_spiel {
namespace universal_poker {

class UniversalPokerGame;

constexpr uint8_t kMaxUniversalPokerPlayers = 10;

enum ActionType {
  kFold = 0,
  kCall = 1,
  kBet = 2
};

enum BettingAbstraction {
  kLimit = 0,
  kNoLimit = 1,
  kDiscreteNoLimit = 2,
};

class UniversalPokerState : public State {
 public:
  explicit UniversalPokerState(std::shared_ptr<const Game> game);

  bool IsTerminal() const override;
  bool IsChanceNode() const override;
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

  // Used to make UpdateIncrementalStateDistribution much faster.
  std::unique_ptr<HistoryDistribution> GetHistoriesConsistentWithInfostate(
      int player_id) const override;

  void ApplyBet(int amount);

  int GetRound() const;
  int NumFolded() const;
  int MaxSpend() const;
  int Pot() const;
  int MaxSpend(int player) const;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  double GetTotalReward(Player player) const;

  void AddHoleCard(uint8_t card) {
    Player p = hole_cards_dealt_ / acpc_game_->GetNbHoleCardsRequired();
    const int card_index =
        hole_cards_dealt_ % acpc_game_->GetNbHoleCardsRequired();
    acpc_state_.AddHoleCard(p, card_index, card);
    ++hole_cards_dealt_;
  }

  void AddBoardCard(uint8_t card) {
    acpc_state_.AddBoardCard(board_cards_dealt_, card);
    ++board_cards_dealt_;
  }

  std::tuple<logic::CardSet, logic::CardSet, uint64_t>
  AbstractedHoleAndBoardCards(Player player) const {
    return GetCardAbstraction()->abstract(HoleCards(player), BoardCards());
  }

  logic::CardSet HoleCards(Player player) const {
    logic::CardSet hole_cards;
    const int num_players = acpc_game_->GetNbPlayers();
    const int num_cards_dealt_to_all = hole_cards_dealt_ / num_players;
    int num_cards_dealt_to_player = num_cards_dealt_to_all;
    // We deal to players in order from 0 to n - 1. So if the number of cards
    // dealt % num_players is > the player, we haven't dealt them a card yet;
    // otherwise we have.
    if (player < (hole_cards_dealt_ % num_players) &&
        num_cards_dealt_to_all < acpc_game_->GetNbHoleCardsRequired()) {
      ++num_cards_dealt_to_player;
    }
    SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
    SPIEL_CHECK_LE(num_cards_dealt_to_player,
                   static_cast<int>(acpc_game_->GetNbHoleCardsRequired()));
    for (int i = 0; i < num_cards_dealt_to_player; ++i) {
      hole_cards.AddCard(acpc_state_.hole_cards(player, i));
    }
    return hole_cards;
  }

  logic::CardSet BoardCards() const {
    logic::CardSet board_cards;
    const int num_board_cards =
        std::min(board_cards_dealt_,
                 static_cast<int>(acpc_game_->GetTotalNbBoardCards()));
    for (int i = 0; i < num_board_cards; ++i) {
      board_cards.AddCard(acpc_state_.board_cards(i));
    }
    return board_cards;
  }

  void AddToActionSequence(uint8_t action, uint32_t size);
  BettingAbstraction GetBettingAbstraction() const;
  std::vector<double> GetBetSet() const;
  std::vector<std::vector<bool>> GetBetSetInitial() const;
  std::vector<std::vector<bool>> GetBetSetAfterRaise() const;
  std::vector<bool> GetBetSetInitial(int round) const;
  std::vector<bool> GetBetSetAfterRaise(int round) const;
  card_abstraction::CardAbstraction* GetCardAbstraction() const;
  bool GetCardAbsIndexOnly() const;
  int CalculateBetSize(uint8_t action_id, Player player) const;
  int BigBlind() const;
  uint8_t AllInActionId() const;

  const acpc_cpp::ACPCGame *acpc_game_;
  mutable acpc_cpp::ACPCState acpc_state_;
  logic::CardSet deck_;  // The remaining cards to deal.
  int hole_cards_dealt_ = 0;
  int board_cards_dealt_ = 0;

  // Action sequence
  struct PokerAction {
    uint8_t round;
    uint8_t player;
    uint8_t action;
    uint32_t size;
  };
  std::vector<PokerAction> action_sequence_;
};

class UniversalPokerGame : public Game {
 public:
  explicit UniversalPokerGame(const GameParameters &params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  int MaxChanceOutcomes() const override;
  double UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int MaxGameLength() const override {
    //this is from acpc/project_acpc_server/game.h#23
    return 64;
  }

  BettingAbstraction GetBettingAbstraction() const {
    return betting_abstraction_;
  }
  card_abstraction::CardAbstraction* card_abstraction() const {
    return card_abstraction_;
  }

  std::vector<double> GetBetSet() const {
    return bet_set_;
  }

  std::vector<std::vector<bool>> GetBetSetInitialMask() const {
    return bet_set_initial_mask_;
  }

  std::vector<std::vector<bool>> GetBetSetAfterRaiseMask() const {
    return bet_set_after_raise_mask_;
  }

  card_abstraction::CardAbstraction* GetCardAbstraction() const {
    return card_abstraction_;
  }

  bool GetCardAbsIndexOnly() const {
      return card_abs_index_only_;
  }

  int BigBlind() const {
    return big_blind_;
  }

private:
  std::string gameDesc_;
  const acpc_cpp::ACPCGame acpc_game_;
  BettingAbstraction betting_abstraction_;
  std::vector<double> bet_set_;
  std::vector<std::vector<bool>> bet_set_initial_mask_;
  std::vector<std::vector<bool>> bet_set_after_raise_mask_;
  card_abstraction::CardAbstraction* card_abstraction_;
  bool card_abs_index_only_ = false;
  int32_t big_blind_ = 0;

  bool CheckStandardDeck() const;
  std::vector<int> CardPerRound() const;

public:
  const acpc_cpp::ACPCGame *GetACPCGame() const { return &acpc_game_; }

  std::string parseParameters(const GameParameters &map);
};

// Only supported for UniversalPoker. Randomly plays an action from a fixed list
// of actions. If none of the actions are legal, checks/calls.
class UniformRestrictedActions : public Policy {
 public:
  // Actions will be restricted to this list when legal. If no such action is
  // legal, checks/calls.
  explicit UniformRestrictedActions(absl::Span<const ActionType> actions)
      : actions_(actions.begin(), actions.end()),
        max_action_(*absl::c_max_element(actions)) {}

  ActionsAndProbs GetStatePolicy(const State &state) const {
    ActionsAndProbs policy;
    policy.reserve(actions_.size());
    const std::vector<Action> legal_actions = state.LegalActions();
    for (Action action : legal_actions) {
      if (actions_.contains(static_cast<ActionType>(action))) {
        policy.emplace_back(action, 1.);
      }
      if (policy.size() >= actions_.size() || action > max_action_) break;
    }

    // It is always legal to check/call.
    if (policy.empty()) {
      SPIEL_DCHECK_TRUE(absl::c_find(legal_actions, ActionType::kCall) !=
                        legal_actions.end());
      policy.push_back({static_cast<Action>(ActionType::kCall), 1.});
    }

    // If we have a non-empty policy, normalize it!
    if (policy.size() > 1) NormalizePolicy(&policy);
    return policy;
  }

 private:
  const absl::flat_hash_set<ActionType> actions_;
  const ActionType max_action_;
};

std::ostream &operator<<(std::ostream &os, const BettingAbstraction &betting);
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
