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

#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <math.h>
#include <tuple>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/card_abstraction/isomorphic.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace {

std::string BettingAbstractionToString(const BettingAbstraction &betting) {
  switch (betting) {
    case BettingAbstraction::kLimit: {
      return "BettingAbstraction: kLimit";
    }
    case BettingAbstraction::kNoLimit: {
      return "BettingAbstraction: kNoLimit";
    }
    case BettingAbstraction::kDiscreteNoLimit: {
      return "BettingAbstraction: kDiscreteNoLimit";
    }
    default:
      SpielFatalError("Unknown betting abstraction.");
  }
}

}  // namespace

const GameType kGameType{
    /*short_name=*/"universal_poker",
    /*long_name=*/"Universal Poker",
                   GameType::Dynamics::kSequential,
                   GameType::ChanceMode::kExplicitStochastic,
                   GameType::Information::kImperfectInformation,
                   GameType::Utility::kZeroSum,
                   GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/

                   {// The ACPC code uses a specific configuration file to describe the game.
                       // The following has been copied from ACPC documentation:
                       //
                       // Empty lines or lines with '#' as the very first character will be
                       // ignored
                       //
                       // The Game definitions should start with "gamedef" and end with
                       // "end gamedef" and can have the fields documented bellow (case is
                       // ignored)
                       //
                       // If you are creating your own game definitions, please note that game.h
                       // defines some constants for maximums in games (e.g., number of rounds).
                       // These may need to be changed for games outside of the what is being run
                       // for the Annual Computer Poker Competition.

                       // The documentation is adapted from project_acpc_server/game.cc.
                       //
                       // Number of Players (up to 10)
                       {"numPlayers", GameParameter(2)},
                       // Betting Type "limit" "nolimit"
                       {"betting", GameParameter(std::string("nolimit"))},
                       // The stack size for each player at the start of each hand (for
                       // no-limit).
                       // INT32_MAX for all players when not provided.
                       {"stack", GameParameter(std::string("1200 1200"))},
                       // The size of the blinds for each player (relative to the dealer)
                       {"blind", GameParameter(std::string("100 100"))},
                       // The size of raises on each round (for limit games only) as numrounds
                       // integers. It will be ignored for nolimite games.
                       {"raiseSize", GameParameter(std::string("100 100"))},
                       // Number of betting rounds per hand of the game
                       {"numRounds", GameParameter(2)},
                       // The player that acts first (relative to the dealer) on each round
                       {"firstPlayer", GameParameter(std::string("1 1"))},
                       // maxraises - the maximum number of raises on each round. If not
                       // specified, it will default to UINT8_MAX.
                       {"maxRaises", GameParameter(std::string(""))},
                       // The number of different suits in the deck
                       {"numSuits", GameParameter(4)},
                       // The number of different ranks in the deck
                       {"numRanks", GameParameter(6)},
                       // The number of private cards to  deal to each player
                       {"numHoleCards", GameParameter(std::string("1"))},
                       // The number of cards revealed on each round
                       {"numBoardCards", GameParameter(std::string("0 1"))},

                       // Game Params

                       {"bettingAbstraction", GameParameter(std::string("discreteNoLimit"))},
                       {"betSet", GameParameter(std::string("0.25 0.5 0.75 1.0 1.25"))},
                       {"cardAbstraction", GameParameter(std::string("noop"))}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new UniversalPokerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// namespace universal_poker
UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game)
    : State(game),
      acpc_game_(
          static_cast<const UniversalPokerGame *>(game.get())->GetACPCGame()),
      acpc_state_(acpc_game_),
      deck_(/*num_suits=*/acpc_game_->NumSuitsDeck(),
          /*num_ranks=*/acpc_game_->NumRanksDeck()) {}

BettingAbstraction UniversalPokerState::GetBettingAbstraction() const {
  return static_cast<const UniversalPokerGame *>(game_.get())->GetBettingAbstraction();
}

std::vector<double> UniversalPokerState::GetBetSet() const {
  return static_cast<const UniversalPokerGame *>(game_.get())->GetBetSet();
}

card_abstraction::CardAbstraction *UniversalPokerState::GetCardAbstraction() const {
  return static_cast<const UniversalPokerGame *>(game_.get())->GetCardAbstraction();
}

std::string UniversalPokerState::ToString() const {
  std::string str =
      absl::StrCat(BettingAbstractionToString(GetBettingAbstraction()), "\n");
  for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
    absl::StrAppend(&str, "P", p, " Cards: ", HoleCards(p).ToString(), "\n");
  }
  absl::StrAppend(&str, "BoardCards ", BoardCards().ToString(), "\n");

  if (IsChanceNode()) {
    absl::StrAppend(&str, "PossibleCardsToDeal ", deck_.ToString(), "\n");
  }
  if (IsTerminal()) {
    for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
      absl::StrAppend(&str, "P", p, " Reward: ", GetTotalReward(p), "\n");
    }
  }
  absl::StrAppend(&str, "Node type?: ");
  if (IsChanceNode()) {
    absl::StrAppend(&str, "Chance node\n");
  } else if (IsTerminal()) {
    absl::StrAppend(&str, "Terminal Node!\n");
  } else {
    absl::StrAppend(&str, "Player node for player ", CurrentPlayer(), "\n");
  }

  auto legal_actions = LegalActions();
  absl::StrAppend(&str, "PossibleActions (", legal_actions.size(), "): [");
  for (Action action : legal_actions) {
    absl::StrAppend(&str, action, ",");
  }

  std::string action_seq = "";
  for (PokerAction pa: action_sequence_) {
    if (pa.action == kFold) {
      absl::StrAppend(&action_seq, "f");
    } else if (pa.action == kCall) {
      absl::StrAppend(&action_seq, "c");
    } else {
      absl::StrAppend(&action_seq, "b", double(pa.size) / BigBlind());
    }
  }
  absl::StrAppend(&str, "]", "\nRound: ", acpc_state_.GetRound(),
                  "\nACPC State: ", acpc_state_.ToString(),
                  "\nAction Sequence: ", action_seq);
  return str;
}

std::string UniversalPokerState::ActionToString(Player player,
                                                Action move) const {
  std::string move_str;
  if (IsChanceNode()) {
    move_str = absl::StrCat("Deal(", move, ")");
  } else if (static_cast<ActionType>(move) == ActionType::kFold) {
    move_str = "Fold";
  } else if (static_cast<ActionType>(move) == ActionType::kCall) {
    move_str = "Call";
  } else if (move == static_cast<const UniversalPokerGame *>(game_.get())->AllInActionId()) {
    move_str = "All In";
  } else if (GetBettingAbstraction() == BettingAbstraction::kNoLimit) {
    SPIEL_CHECK_GE(move, 2);
    const int raise_size = (move - 1) * BigBlind();
    move_str = absl::StrCat("Bet", raise_size);
  } else if (GetBettingAbstraction() == BettingAbstraction::kLimit) {
    move_str = absl::StrCat("Bet");
  } else if (GetBettingAbstraction() == BettingAbstraction::kDiscreteNoLimit) {
    move_str = absl::StrCat("Bet ", GetBetSet()[move - 2], " Pot");
  } else {
    SpielFatalError(absl::StrCat("Unknown action: ", move));
  }
  return absl::StrCat("player=", player, " move=", move_str);
}

bool UniversalPokerState::IsTerminal() const {
  return acpc_state_.IsFinished() && !IsChanceNode();
}

int UniversalPokerState::CalculateBetSize(uint8_t action_id, Player player) const {
  if (GetBettingAbstraction() == BettingAbstraction::kNoLimit) {
    return (action_id - 1) * BigBlind();
  } else if (GetBettingAbstraction() == BettingAbstraction::kDiscreteNoLimit) {
    auto pot_fraction = GetBetSet()[action_id - kBet];
    auto pot_and_amount_to_call = acpc_state_.TotalSpent();
    auto amount_to_call = acpc_state_.MaxSpend() - acpc_state_.CurrentSpent(player);
    return round(amount_to_call + pot_and_amount_to_call * pot_fraction);
  } else if (GetBettingAbstraction() == BettingAbstraction::kLimit) {
    return acpc_game_->GetRaiseSize(acpc_state_.GetRound());
  } else {
    SpielFatalError("Unknown betting abstraction");
  }
}

int UniversalPokerState::BigBlind() const {
  return static_cast<const UniversalPokerGame *>(game_.get())->BigBlind();
}

bool UniversalPokerState::IsChanceNode() const {
  bool need_hole_cards = hole_cards_dealt_ <
                         acpc_game_->GetNbHoleCardsRequired() * acpc_game_->GetNbPlayers();
  bool need_board_cards = board_cards_dealt_ <
                          acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound());
  return need_hole_cards || need_board_cards;
}

Player UniversalPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (IsChanceNode()) {
    return kChancePlayerId;
  }
  return Player(acpc_state_.CurrentPlayer());
}

std::vector<double> UniversalPokerState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(NumPlayers(), 0.0);
  }

  std::vector<double> returns(NumPlayers());
  for (Player player = 0; player < NumPlayers(); ++player) {
    // Money vs money at start.
    returns[player] = GetTotalReward(player) / BigBlind();
  }

  return returns;
}

int UniversalPokerState::PublicObservationTensor(Player player, absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0.);
  // """"""""""""
  // Table State
  // """"""""""""
  int offset = 0;
  // current actor
  values[offset + player] = 1;
  offset += NumPlayers();

  // round
  values[offset + acpc_state_.GetRound()] = 1;
  offset += acpc_game_->NumRounds();

  // """"""""""""
  // Players State
  // """"""""""""

  for (int p = 0; p < acpc_game_->GetNbPlayers(); p++) {
    values[offset] = double(acpc_state_.Money(p)) / BigBlind(); // player stack
    values[offset + 1] = double(acpc_state_.CurrentSpent(p)) / BigBlind(); // pot contribution
    values[offset + 2] = acpc_state_.Folded(p); // player has folded
    values[offset + 3] = acpc_state_.Money(p) == 0; // player is all in
    offset += 4;
  }

  // """"""""""""
  // History
  // Action = action size (call/bet/raise) -1 means action not taken
  // One Round = [p1 action | p2 action | p3 action | p4 action ] * 4
  // History = [Round] * 4
  // """"""""""""
  auto history_len = kMaxActionsPerPlayerPerRound * acpc_game_->GetNbPlayers() * acpc_game_->NumRounds();

  std::fill(values.begin() + offset, values.begin() + offset + history_len, -1);

  std::vector<int> actions_in_round(acpc_game_->GetNbPlayers(), 0);
  int current_round = 0;

  for (PokerAction pa: action_sequence_) {
    if (pa.round > current_round) {
      std::fill(actions_in_round.begin(), actions_in_round.end(), 0);
      current_round = pa.round;
    }
    int round_offset = pa.round * (kMaxActionsPerPlayerPerRound * acpc_game_->GetNbPlayers());
    int player_offset = pa.player;
    int action_offset = actions_in_round[pa.player]++ * kMaxActionsPerPlayerPerRound;
    int total_offset = round_offset + player_offset + action_offset;
    values[offset + total_offset] = double(pa.size) / BigBlind();
  }

  offset += history_len;

  return offset;
}

void UniversalPokerState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorShape()[0]);
  int offset = PublicObservationTensor(player, values);

  // """"""""""""
  // Board and Hole Cards
  // """"""""""""

  auto[holeCards, boardCards] = AbstractedHoleAndBoardCards(player);
  auto boardCardsArr = boardCards.ToCardArray();

  for (int i = 0; i < acpc_game_->GetTotalNbBoardCards(); i++) {
    if (i < boardCardsArr.size()) {
      auto card = boardCardsArr[i];
      values[offset] = card;
    } else {
      values[offset] = kCardNotDealt;
    }
    offset += 1;
  }

  auto holeCardsArr = holeCards.ToCardArray();

  for (int i = 0; i < acpc_game_->GetNbHoleCardsRequired(); i++) {
    if (i < holeCardsArr.size()) {
      auto card = holeCardsArr[i];
      values[offset] = card;
    } else {
      values[offset] = kCardNotDealt;
    }
    offset += 1;
  }

  SPIEL_CHECK_EQ(offset, game_->InformationStateTensorShape()[0]);
}

void UniversalPokerState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorShape()[0]);
  int offset = PublicObservationTensor(player, values);

  // """"""""""""
  // Hole Cards For Each Player
  // """"""""""""
  auto boardCards = BoardCards();
  auto boardCardsArr = boardCards.ToCardArray();

  for (int i = 0; i < acpc_game_->GetTotalNbBoardCards(); i++) {
    if (i < boardCardsArr.size()) {
      auto card = boardCardsArr[i];
      values[offset] = card;
    } else {
      values[offset] = kCardNotDealt;
    }
    offset += 1;
  }


  for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
    auto holeCards = HoleCards(player);
    auto holeCardsArr = holeCards.ToCardArray();

    for (int i = 0; i < acpc_game_->GetNbHoleCardsRequired(); i++) {
      if (i < holeCardsArr.size()) {
        auto card = holeCardsArr[i];
        values[offset] = card;
      } else {
        values[offset] = kCardNotDealt;
      }
      offset += 1;
    }
  }

  SPIEL_CHECK_EQ(offset, game_->ObservationTensorShape()[0]);
}

std::string UniversalPokerState::InformationStateString(Player player) const {
  // TODO: match this with tensor
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  const uint32_t pot = acpc_state_.TotalSpent();
  std::vector<int> money;
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    money.emplace_back(acpc_state_.Money(p));
  }
  std::vector<std::string> sequences;
  for (auto r = 0; r <= acpc_state_.GetRound(); r++) {
    sequences.emplace_back(acpc_state_.BettingSequence(r));
  }

  auto[holeCards, boardCards] = AbstractedHoleAndBoardCards(player);
  return absl::StrFormat(
      "[Round %i][Player: %i][Pot: %i][Money: %s][Private: %s][Public: "
      "%s][Sequences: %s]",
      acpc_state_.GetRound(), CurrentPlayer(), pot, absl::StrJoin(money, " "),
      holeCards.ToString(), boardCards.ToString(),
      absl::StrJoin(sequences, "|"));
}

std::string UniversalPokerState::ObservationString(Player player) const {
  // TODO: match this with tensor
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  std::string result;

  const uint32_t pot = acpc_state_.MaxSpend() *
                       (acpc_game_->GetNbPlayers() - acpc_state_.NumFolded());
  absl::StrAppend(&result, "[Round ", acpc_state_.GetRound(),
                  "][Player: ", CurrentPlayer(), "][Pot: ", pot, "][Money:");
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    absl::StrAppend(&result, " ", acpc_state_.Money(p));
  }
  // Add the player's private cards
  auto[holeCards, boardCards] = AbstractedHoleAndBoardCards(player);
  if (player != kChancePlayerId) {
    absl::StrAppend(&result, "][Private: ", holeCards.ToString(), "]");
  }
  // Adding the contribution of each players to the pot
  absl::StrAppend(&result, "[Ante:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", acpc_state_.Ante(p));
  }
  absl::StrAppend(&result, "]");

  return result;
}

std::unique_ptr<State> UniversalPokerState::Clone() const {
  return std::unique_ptr<State>(new UniversalPokerState(*this));
}

std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes()
const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  auto available_cards = LegalActions();
  const int num_cards = available_cards.size();
  const double p = 1.0 / num_cards;

  // We need to convert std::vector<uint8_t> into std::vector<Action>.
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_cards);
  for (const auto &card : available_cards) {
    outcomes.push_back({Action{card}, p});
  }
  return outcomes;
}

std::vector<Action> UniversalPokerState::LegalActions() const {
  if (IsChanceNode()) {
    const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                   acpc_game_->NumRanksDeck());
    const std::vector<uint8_t> all_cards = full_deck.ToCardArray();
    std::vector<Action> actions;
    actions.reserve(deck_.NumCards());
    for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
      if (deck_.ContainsCards(all_cards[i])) actions.push_back(i);
    }
    return actions;
  }

  std::vector<Action> legal_actions;

  if (IsTerminal()) {
    return legal_actions;
  }

  if (acpc_state_.IsValidAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0)) {
    legal_actions.push_back(kFold);
  }
  if (acpc_state_.IsValidAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0)) {
    legal_actions.push_back(kCall);
  }

  int32_t min_bet_size = 0;
  int32_t max_bet_size = 0;
  bool valid_to_raise = acpc_state_.RaiseIsValid(&min_bet_size, &max_bet_size);


  if (valid_to_raise) {
    if (GetBettingAbstraction() == BettingAbstraction::kLimit) {
      legal_actions.push_back(kBet);
    } else {
      int largest_bet = 0;
      int all_in_size = acpc_state_.Money(CurrentPlayer()) + acpc_state_.CurrentSpent(CurrentPlayer());

      for (int action_id = kBet; action_id < NumDistinctActions() - 1; action_id++) {
        int bet_size = CalculateBetSize(action_id, CurrentPlayer());
        if (bet_size >= min_bet_size && bet_size <= max_bet_size) {
          largest_bet = bet_size;
          legal_actions.push_back(action_id);
        }
      }

      if (all_in_size >= min_bet_size && all_in_size <= max_bet_size && all_in_size > largest_bet) {
        legal_actions.push_back(static_cast<const UniversalPokerGame *>(game_.get())->AllInActionId());
      }
    }
  }
  return legal_actions;
}

// We first deal the cards to each player, dealing all the cards to the first
// player first, then the second player, until all players have their private
// cards.
void UniversalPokerState::DoApplyAction(Action action_id) {
  if (IsChanceNode()) {
    // In chance nodes, the action_id is an index into the full deck.
    uint8_t card =
        logic::CardSet(acpc_game_->NumSuitsDeck(), acpc_game_->NumRanksDeck())
            .ToCardArray()[action_id];
    deck_.RemoveCard(card);

    // Check where to add this card
    if (hole_cards_dealt_ <
        acpc_game_->GetNbPlayers() * acpc_game_->GetNbHoleCardsRequired()) {
      AddHoleCard(card);
      return;
    }

    if (board_cards_dealt_ <
        acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
      AddBoardCard(card);
      return;
    }
  } else {
    int action_int = static_cast<int>(action_id);
    if (action_int == kFold) {
      AddToActionSequence(kFold, 0);
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0);
      return;
    }
    if (action_int == kCall) {
      AddToActionSequence(kCall, 0);
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0);
      return;
    }
    if (action_int == static_cast<const UniversalPokerGame *>(game_.get())->AllInActionId()) {
      auto all_in_size = acpc_state_.Money(CurrentPlayer()) + acpc_state_.CurrentSpent(CurrentPlayer());
      AddToActionSequence(action_int, all_in_size);
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE, all_in_size);
      return;
    }
    if (action_int >= kBet) {
      int bet = CalculateBetSize(action_int, CurrentPlayer());
      AddToActionSequence(action_int, bet);
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE, bet);
      return;
    }
    SpielFatalError(absl::StrFormat("Action not recognized: %i", action_id));
  }
}

double UniversalPokerState::GetTotalReward(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  return acpc_state_.ValueOfState(player);
}

std::unique_ptr<HistoryDistribution>
UniversalPokerState::GetHistoriesConsistentWithInfostate(int player_id) const {
  // This is only implemented for 2 players.
  if (acpc_game_->GetNbPlayers() != 2) return {};

  logic::CardSet is_cards;
  logic::CardSet our_cards = HoleCards(player_id);
  for (uint8_t card : our_cards.ToCardArray()) is_cards.AddCard(card);
  for (uint8_t card : BoardCards().ToCardArray()) is_cards.AddCard(card);
  logic::CardSet fresh_deck(/*num_suits=*/acpc_game_->NumSuitsDeck(),
      /*num_ranks=*/acpc_game_->NumRanksDeck());
  for (uint8_t card : is_cards.ToCardArray()) fresh_deck.RemoveCard(card);
  auto dist = std::make_unique<HistoryDistribution>();
  for (uint8_t hole_card1 : fresh_deck.ToCardArray()) {
    logic::CardSet subset_deck = fresh_deck;
    subset_deck.RemoveCard(hole_card1);
    for (uint8_t hole_card2 : subset_deck.ToCardArray()) {
      if (hole_card1 < hole_card2) continue;
      std::unique_ptr<State> root = game_->NewInitialState();
      if (player_id == 0) {
        for (uint8_t card : our_cards.ToCardArray()) root->ApplyAction(card);
        root->ApplyAction(hole_card1);
        root->ApplyAction(hole_card2);
      } else if (player_id == 1) {
        root->ApplyAction(hole_card1);
        root->ApplyAction(hole_card2);
        for (uint8_t card : our_cards.ToCardArray()) root->ApplyAction(card);
      }
      SPIEL_CHECK_FALSE(root->IsChanceNode());
      dist->first.push_back(std::move(root));
      dist->second.push_back(1.);
    }
  }
  dist->second.resize(dist->first.size(),
                      1. / static_cast<double>(dist->first.size()));
  return dist;
}

void UniversalPokerState::AddToActionSequence(uint8_t action, uint32_t size) {
  auto current_player = acpc_state_.CurrentPlayer();
  auto current_round = acpc_state_.GetRound();

  action_sequence_.push_back({current_round, current_player, action, size});
}

/**
 * Universal Poker Game Constructor
 * @param params
 */
UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
    : Game(kGameType, params),
      gameDesc_(parseParameters(params)),
      acpc_game_(gameDesc_) {
  std::string betting_abstraction =
      ParameterValue<std::string>("bettingAbstraction");
  std::string betting =
      ParameterValue<std::string>("betting");
  if (betting == "limit" || betting_abstraction == "limit") {
    if (betting == "limit" && betting_abstraction == "limit") {
      betting_abstraction_ = BettingAbstraction::kLimit;
    } else {
      SpielFatalError("bettingAbstraction must be limit for limit games");
    }
  } else if (betting_abstraction == "nolimit") {
    betting_abstraction_ = BettingAbstraction::kNoLimit;
  } else if (betting_abstraction.find("discreteNoLimit") == 0) {
    betting_abstraction_ = BettingAbstraction::kDiscreteNoLimit;
    std::vector<std::string> bet_set_s =
        absl::StrSplit(ParameterValue<std::string>("betSet"), ' ');
    if (bet_set_s.empty()) {
      SpielFatalError("You must supply a bet set with discreteNoLimit");
    }
    for (std::string b: bet_set_s) {
      bet_set_.push_back(std::stod(b));
    }
  } else {
    SpielFatalError(absl::StrFormat("bettingAbstraction: %s not supported.",
                                    betting_abstraction));
  }

  std::string card_abstraction =
      ParameterValue<std::string>("cardAbstraction");
  if (card_abstraction == "isomorphic") {
    if (acpc_game_.NumSuitsDeck() != 4 || acpc_game_.NumRanksDeck() != 13) {
      SpielFatalError(
          "Isomorphic card abstraction is only supported standard 52 card deck.");
    }
    std::vector<int> cards_per_round;
    int i;
    for (i = 0; i < acpc_game_.NumRounds(); ++i) {
      cards_per_round.push_back(acpc_game_.GetNbBoardCardsRequired(i));
    }
    cards_per_round[0] += cards_per_round[0] + acpc_game_.GetNbHoleCardsRequired();
    card_abstraction_ = new card_abstraction::IsomorphicCardAbstraction(cards_per_round);
  } else if (card_abstraction == "noop") {
    card_abstraction_ = new card_abstraction::NoopCardAbstraction();
  } else {
    SpielFatalError(absl::StrFormat("cardAbstraction: %s not supported.",
                                    card_abstraction));
  }

  for (int32_t b: acpc_game_.blinds()) {
    if (b > big_blind_) {
      big_blind_ = b;
    }
  }
}

std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
  return absl::make_unique<UniversalPokerState>(shared_from_this());
}

std::vector<int> UniversalPokerGame::InformationStateTensorShape() const {
  // Table State
  //    Current Actor(1 hot), Current Round (1 hot)
  // Players State
  //    Stack, Current Bet, Folded, All In
  // History
  //     Per Round Per Player X 4 (limit max moves by player per round to 4
  //     Action holds the action value (Raise amount) -1 for no action yet
  // Board State
  // Private State
  const int num_players = acpc_game_.GetNbPlayers();

  const int table_state = num_players + acpc_game_.NumRounds();
  const int players_state = 4 * num_players;
  const int history = kMaxActionsPerPlayerPerRound * num_players * acpc_game_.NumRounds();
  const int board_state = acpc_game_.GetTotalNbBoardCards();
  const int private_state = acpc_game_.GetNbHoleCardsRequired();
  return {table_state + players_state + history + board_state + private_state};
}

std::vector<int> UniversalPokerGame::ObservationTensorShape() const {
  // Same as info state but private cards for all players
  const int private_states_diff = (acpc_game_.GetNbPlayers() - 1) * acpc_game_.GetNbHoleCardsRequired();

  return {InformationStateTensorShape()[0] + private_states_diff};
}

double UniversalPokerGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.
  return (double) acpc_game_.StackSize(0) * (acpc_game_.GetNbPlayers() - 1);
}

double UniversalPokerGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  return -1. * (double) acpc_game_.StackSize(0);
}

int UniversalPokerGame::MaxChanceOutcomes() const {
  return acpc_game_.NumSuitsDeck() * acpc_game_.NumRanksDeck();
}

int UniversalPokerGame::NumPlayers() const { return acpc_game_.GetNbPlayers(); }

int UniversalPokerGame::NumDistinctActions() const {
  if (betting_abstraction_ == BettingAbstraction::kNoLimit) {
    // fold, check/call, bet/raise some multiple of BBs, all in
    // covers 100 BB stack depth
    return 104;
  } else if (betting_abstraction_ == BettingAbstraction::kLimit) {
    return 3;
  } else { // disc no limit
    // fold, check/call, bet/raise w.r.t. pot, all in
    return 3 + bet_set_.size();
  }
}

/**
 * Parses the Game Paramters and makes a gameDesc out of it
 * @param map
 * @return
 */
std::string UniversalPokerGame::parseParameters(const GameParameters &map) {
  std::string generated_gamedef = "GAMEDEF\n";

  absl::StrAppend(
      &generated_gamedef, ParameterValue<std::string>("betting"), "\n",
      "numPlayers = ", ParameterValue<int>("numPlayers"), "\n",
      "numRounds = ", ParameterValue<int>("numRounds"), "\n",
      "numsuits = ", ParameterValue<int>("numSuits"), "\n",
      "firstPlayer = ", ParameterValue<std::string>("firstPlayer"), "\n",
      "numRanks = ", ParameterValue<int>("numRanks"), "\n",
      "numHoleCards = ", ParameterValue<std::string>("numHoleCards"), "\n",
      "numBoardCards = ", ParameterValue<std::string>("numBoardCards"), "\n");

  std::string max_raises = ParameterValue<std::string>("maxRaises");
  if (!max_raises.empty()) {
    absl::StrAppend(&generated_gamedef, "maxRaises = ", max_raises, "\n");
  }

  if (ParameterValue<std::string>("betting") == "limit") {
    std::string raise_size = ParameterValue<std::string>("raiseSize");
    if (!raise_size.empty()) {
      absl::StrAppend(&generated_gamedef, "raiseSize = ", raise_size, "\n");
    }
  }

  std::string stack = ParameterValue<std::string>("stack");
  if (!stack.empty()) {
    absl::StrAppend(&generated_gamedef, "stack = ", stack, "\n");
  }

  absl::StrAppend(&generated_gamedef,
                  "blind = ", ParameterValue<std::string>("blind"), "\n");
  absl::StrAppend(&generated_gamedef, "END GAMEDEF\n");

  return generated_gamedef;
}

uint8_t UniversalPokerGame::AllInActionId() const {
  if (betting_abstraction_ == BettingAbstraction::kNoLimit) {
    return 103;
  } else if (betting_abstraction_ == BettingAbstraction::kLimit) {
    // no all in for limit games
    return 3;
  } else { // disc no limit
    return 2 + bet_set_.size();
  }
}

uint8_t UniversalPokerGame::NumSuits() const {
  return acpc_game_.NumSuitsDeck();
}

uint8_t UniversalPokerGame::NumRanks() const{
  return acpc_game_.NumRanksDeck();
}

uint8_t UniversalPokerGame::NumRounds() const{
  return acpc_game_.NumRounds();
}

uint8_t UniversalPokerGame::GetNbOfHoleCards() const {
    return acpc_game_.GetNbHoleCardsRequired();
}

uint8_t UniversalPokerGame::GetTotalNbBoardCards() const {
    return acpc_game_.GetTotalNbBoardCards();
}

uint8_t UniversalPokerGame::GetNbBoardCardsForRound(uint8_t round) const{
    return acpc_game_.GetNbBoardCardsRequired(round);
}

std::unique_ptr<State> UniversalPokerGame::StateFromACPCState(std::string acpc_state) const {
  // TODO: parse and apply state
  return NewInitialState();
}

std::ostream &operator<<(std::ostream &os, const BettingAbstraction &betting) {
  os << BettingAbstractionToString(betting);
  return os;
}

}  // namespace universal_poker
}  // namespace open_spiel
