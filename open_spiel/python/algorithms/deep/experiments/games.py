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
     "numHoleCards": pyspiel.GameParameter("1 "),
     "numBoardCards": pyspiel.GameParameter("0 "),
     "bettingAbstraction": pyspiel.GameParameter("limit")})


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
