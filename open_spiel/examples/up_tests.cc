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

#include <string>
#include <iostream>
#include <fstream>
#include "omp.h"

#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/spiel.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel_utils.h"

open_spiel::GameParameters LimitHoldem2PParameters() {
  return {{"betting", open_spiel::GameParameter(std::string("limit"))},
          {"numPlayers", open_spiel::GameParameter(2)},
          {"numRounds", open_spiel::GameParameter(4)},
          {"blind", open_spiel::GameParameter(std::string("10 5"))},
          {"raiseSize", open_spiel::GameParameter(std::string("10 10 20 20"))},
          {"firstPlayer", open_spiel::GameParameter(std::string("2 1 1 1"))},
          {"maxRaises", open_spiel::GameParameter(std::string("3 4 4 4"))},
          {"numSuits", open_spiel::GameParameter(4)},
          {"numRanks", open_spiel::GameParameter(13)},
          {"stack", open_spiel::GameParameter(std::string("1000 1000"))},
          {"numHoleCards", open_spiel::GameParameter(2)},
          {"numBoardCards", open_spiel::GameParameter(std::string("0 3 1 1"))},
          {"bettingAbstraction", open_spiel::GameParameter(std::string("limit"))},
          {"cardAbstraction", open_spiel::GameParameter(std::string("custom"))}
          };
}

open_spiel::GameParameters NLHE6PParameters() {
  return {{"betting", open_spiel::GameParameter(std::string("nolimit"))},
          {"numPlayers", open_spiel::GameParameter(6)},
          {"numRounds", open_spiel::GameParameter(4)},
          {"blind", open_spiel::GameParameter(std::string("2 1 0 0 0 0"))},
          {"firstPlayer", open_spiel::GameParameter(std::string("3 1 1 1"))},
          {"numSuits", open_spiel::GameParameter(4)},
          {"numRanks", open_spiel::GameParameter(13)},
          {"stack", open_spiel::GameParameter(std::string("200 200 200 200 200 200"))},
          {"numHoleCards", open_spiel::GameParameter(2)},
          {"numBoardCards", open_spiel::GameParameter(std::string("0 3 1 1"))},
          {"bettingAbstraction", open_spiel::GameParameter(std::string("discreteNoLimit"))},
          {"betSet", open_spiel::GameParameter(std::string(
          "0.35 0.5 0.65 0.75 1.0 1.1 1.2:0.75 1.0 1.2|0.25 0.33 0.5 0.75 1.0 1.1 1.2:0.5 0.75 1.0 1.2|0.5 1.0:1.0|0.5 1.0:1.0"))},
//          "0.75 0.8 1.0:1.0|0.8 1.0:1.0|0.5 1.0:1.0|0.5 1.0:1.0"))},
          {"cardAbstraction", open_spiel::GameParameter(std::string("custom"))},
          {"cardAbstractionLabelsFolder", open_spiel::GameParameter(std::string("/Users/aliy/IdeaProjects/hand-bucketing/outputs/t1"))}
  };
}

/*
"0.75 0.8 1.0:1.0|0.8 1.0:1.0|0.5 1.0:1.0|0.5 1.0:1.0" non omp
preflop info states: 105738
flop info states: 536802
turn info states: 1390114
river info states: 2693842
./build/examples/up_tests  58.68s user 0.29s system 99% cpu 59.352 total

preflop info states: 105738
flop info states: 536802
turn info states: 1390114
river info states: 2693842
./build/examples/up_tests  65.55s user 0.26s system 167% cpu 39.268 total

"0.35 0.5 0.65 0.75 1.0 1.1 1.2:0.75 1.0 1.2|0.25 0.33 0.5 0.75 1.0 1.1 1.2:0.5 0.75 1.0 1.2|0.5 1.0:1.0|0.5 1.0:1.0"
preflop info states: 2691638
flop info states: 37456916
turn info states: 80927348
river info states: 95454832
./build/examples/up_tests  3030.32s user 8.09s system 99% cpu 50:50.07 total


*/

//std::atomic<long> preflop(0);
//std::atomic<long> flop(0);
//std::atomic<long> turn(0);
//std::atomic<long> river(0);

long preflop = 0;
long flop = 0;
long turn = 0;
long river = 0;

std::ofstream myfile;

void traverse_tree(open_spiel::State& state, int round) {
    if (round > 0) {
        return;
    }
    if (state.IsChanceNode()) {
//        while(state.IsChanceNode()) {
//            state.ApplyAction(state.LegalActions()[0]);
//        }
//        traverse_tree(state, round + 1);
        return;
    }

    if (state.IsTerminal()){
            return;
    }

    if (round == 0) {
        std::string original = state.InformationStateString();
        std::string start = original.substr(0, 4);
        std::string end = original.substr(5);
        auto actions = state.LegalActions();
        std::string actions_str =absl::StrJoin(actions, ",");

        for (int i = 0; i < 169; i ++) {
            myfile << start << i << end << "--" << actions_str <<'\n';
        }
    } else if (round == 1) { flop++;
    } else if (round == 2) { turn++;
    } else if (round == 3) { river++;
    }


//    if (round == 0 && preflop % 10000 == 0) {
//        int ID = omp_get_thread_num();
//        std::cout << "c(t" << ID << ", r" << round << "):" << preflop << std::endl;
//    } else if (round == 1 && flop % 10000 == 0) {
//        int ID = omp_get_thread_num();
//        std::cout << "c(t" << ID << ", r" << round << "):" << flop << std::endl;
//    } else if (round == 2 && turn % 10000 == 0) {
//        int ID = omp_get_thread_num();
//        std::cout << "c(t" << ID << ", r" << round << "):" << turn << std::endl;
//    } else if (round == 3 && river % 10000 == 0) {
//        int ID = omp_get_thread_num();
//        std::cout << "c(t" << ID << ", r" << round << "):" << river << std::endl;
//    }


    auto actions = state.LegalActions();
//    #pragma omp parallel for
    for (int i = 0; i < actions.size(); i++){
       traverse_tree(*state.Child(actions[i]), round);
    }
}

int main(int argc, char** argv) {
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("universal_poker", NLHE6PParameters());
    auto test_state = game->NewInitialState();
    test_state->ApplyAction(0);
    test_state->ApplyAction(1);
    test_state->ApplyAction(2);
    test_state->ApplyAction(3);
    test_state->ApplyAction(10);
    test_state->ApplyAction(11);
    test_state->ApplyAction(4);
    test_state->ApplyAction(5);
    test_state->ApplyAction(6);
    test_state->ApplyAction(7);
    test_state->ApplyAction(8);
    test_state->ApplyAction(9);

    std::cout <<"preflop info states: " << preflop << std::endl;
    std::cout <<"flop info states: " << flop << std::endl;
    std::cout <<"turn info states: " << turn << std::endl;
    std::cout <<"river info states: " << river << std::endl;

    myfile.open ("/Users/aliy/round1_info_states.txt", std::ios::out);
    traverse_tree(*test_state, 0);
    myfile.close();
    std::cout <<"preflop info states: " << preflop << std::endl;
    std::cout <<"flop info states: " << flop << std::endl;
    std::cout <<"turn info states: " << turn << std::endl;
    std::cout <<"river info states: " << river << std::endl;
}
