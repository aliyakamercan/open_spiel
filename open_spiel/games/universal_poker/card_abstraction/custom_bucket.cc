#include "open_spiel/games/universal_poker/card_abstraction/custom_bucket.h"
#include <cstdlib>
#include <iostream>

extern "C" {
#include "open_spiel/games/universal_poker/hand-isomorphism/src/hand_index.h"
}

namespace open_spiel::universal_poker::card_abstraction {

CustomBucketCardAbstraction::CustomBucketCardAbstraction(
  std::vector<int> cards_per_round, std::string label_folder) {

  cards_per_round_.reserve(cards_per_round.size());

  for(int r=0; r < cards_per_round.size(); r++){
      cards_per_round_.push_back(cards_per_round[r]);
      // create indexer for this round
      hand_indexer_t indexer;
      uint8_t array_slice[2];
      array_slice[0] = cards_per_round[0];
      array_slice[1] = 0;
      for (int i = 1; i <= r; i ++) {
        array_slice[1] += cards_per_round[i];
      }
      hand_indexer_init(r > 0 ? 2 : 1, array_slice, &indexer);
      indexers_.push_back(indexer);
  }

  // Load river labels
  FILE * rfin = fopen((label_folder + "/river.labels").c_str(), "rb");
  if (!rfin) {
    std::cerr << "River label file not found." << std::endl;
    std::exit(1);
  }
  fread(river_labels_, sizeof(river_labels_), 1, rfin);
  fclose(rfin);

  // Load turn labels
  FILE * tfin = fopen((label_folder + "/turn.labels").c_str(), "rb");
  if (!tfin) {
    std::cerr << "Turn label file not found." << std::endl;
    std::exit(1);
  }
  fread(turn_labels_, sizeof(turn_labels_), 1, tfin);
  fclose(tfin);

  // Load flop labels
  FILE * ffin = fopen((label_folder + "/flop.labels").c_str(), "rb");
  if (!ffin) {
    std::cerr << "Flop label file not found." << std::endl;
    std::exit(1);
  }
  fread(flop_labels_, sizeof(flop_labels_), 1, ffin);
  fclose(ffin);
}

uint8_t to_iso_card2(uint8_t cs_card) {
    return deck_make_card(cs_card % 4, cs_card / 4);
}

std::tuple<logic::CardSet, logic::CardSet, uint64_t>
CustomBucketCardAbstraction::abstract(logic::CardSet hole_cards,
                                    logic::CardSet board_cards) const {
    std::vector<uint8_t> h_arr = hole_cards.ToCardArray();
    std::vector<uint8_t> b_arr = board_cards.ToCardArray();
    h_arr.insert(h_arr.end(), b_arr.begin(), b_arr.end());

    int total_cards = h_arr.size();
    int round = -1;
    uint8_t cards[total_cards];

    for(int i = 0; i < total_cards; i++) {
        cards[i] = to_iso_card2(h_arr[i]);
    }

    while (total_cards > 0) {
        total_cards -= cards_per_round_[++round];
    }

    hand_indexer_t indexer = indexers_[round];
    uint64_t index = hand_index_last(&indexer, cards);

    // TODO: map index for turn and river
    if (round == 1) { // FLOP
        index = flop_labels_[index];
    } if (round == 2) { // TURN
        index = turn_labels_[index];
    } else if (round == 3) { // RIVER
        index = river_labels_[index];
    }
    return {empty_card_set_, empty_card_set_, index};
}

}
