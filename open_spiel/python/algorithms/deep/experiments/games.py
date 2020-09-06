import pyspiel


kuhn_poker_2p = pyspiel.UniversalPokerGame(
    {"betting": pyspiel.GameParameter("limit"),
     "numPlayers": pyspiel.GameParameter(2),
     "numRounds": pyspiel.GameParameter(1),
     "blind": pyspiel.GameParameter("1 1"),
     "raiseSize": pyspiel.GameParameter("1 "),
     "firstPlayer": pyspiel.GameParameter("1 "),
     "maxRaises": pyspiel.GameParameter("1 "),
     "numSuits": pyspiel.GameParameter(1),
     "numRanks": pyspiel.GameParameter(3),
     "numHoleCards": pyspiel.GameParameter(1),
     "numBoardCards": pyspiel.GameParameter("0 "),
     "bettingAbstraction": pyspiel.GameParameter("limit")})

LEDUC_HYPERS = {
 'TRAVERSALS_ES': 346,  # int(900 / 2.6)
 'TRAVERSALS_OS': 900,
 'BATCHES': 3000,
 'BASELINE_BATCHES': 1000
}

leduc_poker_2p = pyspiel.UniversalPokerGame(
    {"betting": pyspiel.GameParameter("limit"),
     "numPlayers": pyspiel.GameParameter(2),
     "numRounds": pyspiel.GameParameter(2),
     "blind": pyspiel.GameParameter("1 1"),
     "raiseSize": pyspiel.GameParameter("2 4"),
     "firstPlayer": pyspiel.GameParameter("1 1"),
     "maxRaises": pyspiel.GameParameter("2 2"),
     "numSuits": pyspiel.GameParameter(2),
     "numRanks": pyspiel.GameParameter(3),
     "numHoleCards": pyspiel.GameParameter("1 0"),
     "numBoardCards": pyspiel.GameParameter("0 1"),
     "bettingAbstraction": pyspiel.GameParameter("limit")})

FHP = {
 'TRAVERSALS_ES': 10000,
 'TRAVERSALS_OS': 50000,
 'BATCHES': 4000,
 'BATCH_SIZE': 10000,
 'BASELINE_BATCHES': 1000,
}