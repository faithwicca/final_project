import random
import numpy as np

class Player:
    """
    An abstraction for a bot/real player. It keeps its hand as a numpy array of integers. It's main methods are attack and defend. Other methonds
    are for observing the state of the game.
    """
    def __init__(self, name='player'):
        self.trump = None
        self.nPlayers = None
        self.name = name
        self.moveState = None
        self.valuesOnBoard = None
        self.board = None
        self.trumpcard = None
        self.moveN = None
        self.hand = np.array([], dtype='int')
        self.left = []
        self.isDeckEmpty = None
        self.cardsOnBoard = None

    def dump(self):
        self.trump = None
        self.moveState = None
        self.valuesOnBoard = None
        self.board = None
        self.trumpcard = None
        self.moveN = None
        self.hand = np.array([], dtype='int')
        self.left = []
        self.isDeckEmpty = None
        self.cardsOnBoard = None

    def inputInfo(self, info):
        self.moveN = 0
        self.nPlayers = info['nPlayers']
        self.trumpcard = info['trumpcard']
        self.trump = info['trump']

    # information about the current move updates automatically, because variables are mutable. 
    def newMove(self, board, valuesOnBoard, moveState, cardsOnBoard):
        self.board = board
        self.valuesOnBoard = valuesOnBoard
        self.moveState = moveState
        self.cardsOnBoard = cardsOnBoard

    def inputAtkInfo(self, name, info):
        pass

    def inputMoveInfo(self, info):
        self.moveN = info['moveN']
        self.left += info['left']
        self.isDeckEmpty = info['isDeckEmpty']

    def __repr__(self):
        return f'{self.name}, {vision(self.hand)}'
    
    def attack(self, legalMoves):
        pass
    
    def defend(self, legalMoves):
        pass


class Deck:
    """
    Deck contains order of cards as integers in a numpy array. 
    
    Cards are integers from 0 to 35. Each suite (hearts, spades, etc) starts from
    a multiple of 9. So 0-8 is one suite, 9-17 is the next.
    """
    def __init__(self):
        self.trumpcard = None
        self.order = np.linspace(0, 35, 36, dtype='int')
        self.trash = np.array([])

    def newdeck(self):
        self.order = np.linspace(0, 35, 36, dtype='int')
        self.trash = np.array([])

    def shuffle(self):
        np.random.shuffle(self.order)
        self.trumpcard = self.order[35]
        self.trump = (self.order[35] // 9)

    def deal(self, player: Player, q: int):
        player.hand = np.concatenate((player.hand, self.order[:q]))
        self.order = self.order[q:]


def movesFirst(order: np.array, trump: int, trumpcard: int, numpl: int) -> int:
    """
    This function decides who moves first in a game.
    If the trumpcard is 10 or lower, then the player with the highest card of trump suite plays first. If the trumpcard is higher than 10,
    then the person with the lowest card of trump suite plays first.
    """
    fmovers = list()
    if trumpcard % 9 > 4:
        direction = 1
    else:
        direction = -1
    for i in range(numpl):
        trumps = order[i * 6:(i + 1) * 6]
        trumps = trumps[trump * 9 - 1 < trumps]
        trumps = trumps[(trump + 1) * 9 > trumps]
        if len(trumps) > 0:
            if direction == 1:
                fmovers.append(np.min(trumps))
            else:
                fmovers.append(np.max(trumps))
        else:
            fmovers.append(100 * direction)
    if np.sum(fmovers) == direction * 10 * numpl:
        mf = -1
    else:
        if direction == 1:
            mf = np.argmin(fmovers)
        else:
            mf = np.argmax(fmovers)
    return mf


def addLeft(left: list, player):
    """Adds a player to the list of left player if necessary"""
    if np.size(player.hand) == 0 and player not in left:
        left.append(player)
    return left


class Chain:
    """
    Keeps order of players and decides who must currently attack.
    """
    def __init__(self, players, attacker_n):
        self.players = players
        self.attacker = players[attacker_n]
        self.n = len(self.players)
        self.chain = players[attacker_n:] + players[:attacker_n]

    def queue(self):
        defender = self.chain[1]
        attackers = tuple([self.chain[0]]) + self.chain[2:]
        return attackers, defender

    def cycle(self, left, take):
        self.chain = self.chain[1:] + tuple([self.chain[0]])
        if take:
            self.chain = self.chain[1:] + tuple([self.chain[0]])
        self.chain = tuple(player for player in self.chain if player not in left)
        if len(self.chain) <= 1:
            return None, None
        return self.queue()


def legitAttack(hand, board, valuesOnBoard, defCapability):
    """Creates a list of feasible moves given current state of the game for attacker"""
    if len(board) == 0: return list(hand)
    if len(board) >= 6 or defCapability == 0: return ['pass']
    moves = [card for card in hand if card % 9 in valuesOnBoard]
    moves.append('pass')
    return moves


def applyAttack(move, attacker, board, valuesOnBoard, moveState, cardsOnBoard):
    if move == 'pass':
        moveState['pass'] = True
        return
    moveState['pass'] = False
    attacker.hand = attacker.hand[attacker.hand != move]
    valuesOnBoard |= {move % 9}
    if not moveState['take']:
        board['toBeat'] = move
    else:
        board[move] = None
    cardsOnBoard.append(move)


def legitDefense(hand, board, trump):
    """Creates a list of feasible moves given current state of the game for defender"""
    color, value = divmod(board['toBeat'], 9)
    if color == trump: return [card for card in hand if card // 9 == trump and card % 9 > value] + ['take']
    return [card for card in hand if card // 9 == trump or (card // 9 == color and card % 9 > value)] + ['take']


def applyDefense(move, defender, board, valuesOnBoard, moveState, cardsOnBoard):
    if move == 'take':
        moveState['take'] = True
        return
    valuesOnBoard |= {move % 9}
    cardsOnBoard.append(move)
    toBeat = board['toBeat']
    board.pop('toBeat')
    board |= {toBeat: move}
    defender.hand = defender.hand[defender.hand != move]


def applyTake(defender, cardsOnBoard):
    defender.hand = np.concatenate((defender.hand, np.array(cardsOnBoard)))


class Game:
    def __init__(self, players, dsize=36, manual=False, trump=None):
        self.whoAttack = None
        self.nPlayers, self.players, self.left = len(players), players, list()
        self.deck, self.moveN = Deck(), 0
        self.attacker = None
        self.dsize = dsize
        self.stop = False
        self.manual = manual
        self.deckHistory = []
        self.fixedTrump = trump

    def prepare(self):
        """
        Prepares the game for start: shuffles a deck, decides who must attack first, creates
        a chain and tells the players basic information about current game.
        """
        self.whoAttack = -1
        fixedTrumpCondition = True #mechanism for randomly shuffling cards until trump is specified
        while self.whoAttack == -1 or fixedTrumpCondition:
            self.deck.shuffle()
            self.whoAttack = movesFirst(self.deck.order, self.deck.trump, self.deck.trumpcard, self.nPlayers)
            if self.fixedTrump is not None and self.fixedTrump != self.deck.trump:
                fixedTrumpCondition = True
                self.whoAttack = -1
            else: fixedTrumpCondition = False
        info = {'trump': self.deck.trump, 'trumpcard': self.deck.trumpcard, 'nPlayers': self.nPlayers}
        if self.manual: print(f'{vision(self.deck.order)=},\n{vision(self.deck.trumpcard)=}, {self.players=}')
        for player in self.players:
            self.deck.deal(player, 6)
            player.inputInfo(info)
        self.queuer = Chain(self.players, self.whoAttack)
        self.attackers, self.defender = self.queuer.queue()

    def verbose(self, isAttack, data):
        """Used for debugging"""
        print('|', '-' * 100, sep='')
        if isAttack:
            print(f'|  {vision(self.attacker)} attacks')
        else:
            print(f'|  {vision(self.defender)} defends')
        print(f'|  Board: {vision(self.board)}')
        print(f'|  Move state: {self.moveState}')
        print(f"|  Available moves: {vision(data['moves'])}")
        print(f'|  Move: {vision(data["move"])}')
        if input('| Continue manual? Enter/n?') == 'n': self.manual = False

    def assertion_text(self, move, moves):
        """Used for debugging"""
        return str(f'{vision(self.attacker)}' +
                   f'{self.moveN=}\n{vision(self.board)=}' +
                   f'{vision(move)=}, {vision(moves)=}, {self.defender=}')

    def move(self):
        """
        Main component of the code. It cycles through players in the current move. It manipulates everything happening on board
        and keeps track of everything else, also telling players of what's happening.
        """
        continuePlaying, left = True, list()
        self.board, self.valuesOnBoard, self.moveState, self.cardsOnBoard = dict(), set(), {'take': False,
                                                                                            'pass': False}, []
        for player in self.players: player.newMove(self.board, self.valuesOnBoard, self.moveState, self.cardsOnBoard)
        if self.manual: print('=' * 100, '=' * 100, f'MOVE #{self.moveN}', sep='\n')
        while continuePlaying:
            continuePlaying = False
            for i, self.attacker in enumerate(self.attackers):
                self.moveState['pass'] = False
                while not self.moveState['pass']:
                    moves = legitAttack(self.attacker.hand, self.board, self.valuesOnBoard, np.size(self.defender.hand))
                    move = self.attacker.attack(moves)
                    if self.manual: self.verbose(isAttack=True, data={'move': move, 'moves': moves})
                    assert move in moves, self.assertion_text(move, moves)
                    for p in self.players: p.inputAtkInfo(self.attacker.name, move)
                    applyAttack(move, self.attacker, self.board, self.valuesOnBoard, self.moveState, self.cardsOnBoard)
                    left = addLeft(left, self.attacker)
                    if self.moveState['pass']: break
                    if i != 0: continuePlaying = True

                    if self.moveState['take']: continue
                    defense = legitDefense(self.defender.hand, self.board, self.deck.trump)
                    move = self.defender.defend(defense)
                    if self.manual: self.verbose(isAttack=False, data={'move': move, 'moves': defense})
                    assert move in defense, self.assertion_text(move, moves)
                    applyDefense(move, self.defender, self.board, self.valuesOnBoard, self.moveState, self.cardsOnBoard)
                    left = addLeft(left, self.defender)

        if self.moveState['take']:
            applyTake(self.defender, self.cardsOnBoard)
        else:
            self.deck.trash = np.concatenate((self.deck.trash, self.cardsOnBoard))

        for player in (tuple(self.attackers) + (self.defender,)):
            need = max(6 - np.size(player.hand), 0)
            self.deck.deal(player, need)

        left = [p for p in left if np.size(p.hand) == 0]

        self.left.extend(left)
        self.attackers, self.defender = self.queuer.cycle(left, self.moveState['take'])
        if len(self.left) >= self.nPlayers - 1: self.stop = True
        for p in self.players: p. (
            {'left': left, 'moveN': self.moveN, 'isDeckEmpty': np.size(self.deck.order) == 0})

    def play(self):
        """Conveniently launches everything in the game, so that we don't have to do it manually"""
        self.prepare()
        self.deckHistory.append(self.deck.order)
        while not self.stop:
            self.moveN += 1
            self.move()
            self.deckHistory.append(self.deck.order)

    @property
    def results(self):
        results = {'length': self.moveN, 'fool': [p for p in self.players if p not in self.left],
                   'deckHistory': self.deckHistory, 'trump': self.deck.trump}
        return results


def vision(cards):
    """Transforms numbers representing cards into human-readable symbols"""
    value_reference = '67891JQKA'
    color_reference = '♥♦♣♠'

    # the function below transforms only one card, then it is mapped to the input sequence/dict/player
    def onecard(card):
        if card == -1 or card is None or type(card) is str:
            return card
        value = value_reference[card % 9]
        if value == '1': value = '10'
        color = color_reference[card // 9]
        return (value + color)

    if isinstance(cards, Player):
        return cards.name, vision(cards.hand)
    if isinstance(cards, dict):
        return {onecard(key): onecard(val) for key, val in cards.items()}
    if isinstance(cards, int) or isinstance(cards, np.int64):
        return onecard(cards)
    if isinstance(cards, np.ndarray):
        number_representations = cards.flatten()
        return [onecard(card) for card in number_representations]
    if isinstance(cards, str):
        return cards
    else:
        return 'Not implemented'
    return [onecard(card) for card in cards]


class RealPlayer(Player):
    """This is an UI for a human to play fool for testing or for fun"""
    def __init__(self, name: str):
        super().__init__(name)
        self.print = True

    def attack(self, legalMoves):
        print('-' * 100)
        if self.print:
            print(f'trumpcard is {vision(int(self.trumpcard))}')
            self.print = False
        print(f'{self.name}, your hand: {vision(self.hand)}, the board: {vision(self.board)}')
        if 'pass' in legalMoves:
            print(f'You attack, you can play either of these moves: {vision(legalMoves[:-1])} and PASS')
        else:
            print(f'You attack, you can play either of these moves: {vision(legalMoves)}')
        i = int(input('Please, pick a number of a move'))
        return legalMoves[i - 1]

    def defend(self, legalMoves):
        print('-' * 100)
        if self.print:
            print(f'trumpcard is {vision(int(self.trumpcard))}')
            self.print = False
        print(f'{self.name}, you hand: {vision(self.hand)}, the board: {vision(self.board)}')
        print(f'You defend, you can play either of these moves {vision(legalMoves[:-1])} and TAKE')
        i = int(input('Please, pick a number of a move'))
        return legalMoves[i - 1]


class RS(Player):
    """Plays random legal move"""
    def attack(self, legalMoves):
        move = random.choice(legalMoves)
        return move

    def defend(self, legalDefense):
        move = random.choice(legalDefense)
        return move
    
class RS_Left(Player):
    """
    Plays the left most legal move. 
    Since humans can't really pick randomly, I decided that playing a move on an arbitrary condition fits the puprose well.
    """
    def attack(self, legalMoves):
        move = legalMoves[0]
        return move

    def defend(self, legalDefense):
        move = legalDefense[0]
        return move


class ScorePlayer(Player):
    """Creates a map of scores for every card before the game starts"""
    def inputInfo(self, info):
        super().inputInfo(info)
        self.scores = {card: card % 9 for card in range(36)}
        start = self.trump * 9
        for card in range(start, start + 9):
            self.scores[card] += 9
        self.scores['take'] = 666 #so that players never pick these moves
        self.scores['pass'] = 666

    def score(self, move):
        return self.scores[move]

class PowerPlayer(ScorePlayer):
    """
    Knows power of any set of cards. Power is defined as sum of power of individual cards.
    Cards of non-trump suite have powers (scores) 0-8 and trump cards have scores 9-13 in order of 'strength'
    """
    def power(self, cards):
        return sum([self.score(card) for card in cards])

    def averagePower(self, cards):
        return self.power(cards) / np.size(cards)


class Naive1(ScorePlayer):
    """Always picks move with the smallest score"""
    def attack(self, legalMoves):
        moves = legalMoves
        scores = [self.score(move) for move in moves]
        move = moves[np.argmin(scores)]
        return move

    def defend(self, legalDefense):
        moves = legalDefense
        scores = [self.score(move) for move in moves]
        move = moves[np.argmin(scores)]
        return move
    

class Naive2(Naive1):
    """Picks move with the smallest score, but does not play cards of trump suite until 'cutoff' number of moves have passed"""
    def __init__(self, name: str, cutoff: int):
        super().__init__('name')
        self.cutoff = cutoff
        
    def attack(self, legalMoves):
        move = super().attack(legalMoves)
        if self.score(move) > 8 and 'pass' in legalMoves and self.moveN < self.cutoff:
            return 'pass'
        else: return move
    
    def defend(self, legalDefense):
        move = super().defend(legalDefense)
        if self.score(move) > 8 and self.moveN < self.cutoff:
            return 'take'
        else: return move
