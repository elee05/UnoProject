package src.pas.uno.agents;


// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.Hand;
import edu.bu.pas.uno.Hand.HandView;
import edu.bu.pas.uno.agents.MCTSAgent;
import edu.bu.pas.uno.enums.Color;
import edu.bu.pas.uno.enums.Value;
import edu.bu.pas.uno.moves.Move;
import edu.bu.pas.uno.tree.Node;

import java.util.Random;
import java.util.Set;
import java.util.List;
import java.util.ArrayList;


public class ExpectedOutcomeAgent
    extends MCTSAgent
{

    public static class MCTSNode
        extends Node
    {
        public MCTSNode(final GameView game,
                        final int logicalPlayerIdx,
                        final Node parent)
        {
            super(game, logicalPlayerIdx, parent);
        }

        @Override
        public Node getChild(final Move move)
        {
            Game game = new Game(this.getGameView());
            game.resolveMove(move);
            GameView newGameView = game.getView(this.getLogicalPlayerIdx());
            int nextPlayerIdx = newGameView.getPlayerOrder().getCurrentLogicalPlayerIdx();
            return new MCTSNode(newGameView, nextPlayerIdx, this);
        }
    }

    public ExpectedOutcomeAgent(final int playerIdx,
                                final long maxThinkingTimeInMS)
    {
        super(playerIdx, maxThinkingTimeInMS);
    }

    @Override
    public Node search(final GameView game,
                       final Integer drawnCardIdx)
    {
        int currentPlayer = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        MCTSNode root = new MCTSNode(game, currentPlayer, null);
        
        evaluateNode(root, drawnCardIdx);
        
        return root;
    }
    
    private double evaluateNode(MCTSNode node, Integer drawnCardIdx) {
        GameView game = node.getGameView();
        
        if (game.isOver()) {
            HandView playerHand = game.getHandView(node.getLogicalPlayerIdx());
            return playerHand.size() == 0 ? 1.0 : 0.0;
        }
        
        Node.NodeState state = node.getNodeState();
        
        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            Game tempGame = new Game(game);
            Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
            Set<Integer> legalMoveIndices = hand.getLegalMoves(tempGame);
            
            if (legalMoveIndices == null || legalMoveIndices.isEmpty()) {
                return 0.0;
            }
            
            int moveCount = 0;
            for (int idx : legalMoveIndices) {
                Card card = hand.getCard(idx);
                
                Move move;
                if (card.isWild()) {
                    Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                    Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
                    move = Move.createMove(this, idx, chosenColor);
                } else {
                    move = Move.createMove(this, idx);
                }
                
                Node child = node.getChild(move);
                if (child != null) {
                    double childValue = evaluateNode((MCTSNode) child, drawnCardIdx);
                    node.setQValueTotal(moveCount, (float) childValue);
                    node.setQCount(moveCount, 1);
                }
                moveCount++;
            }
            
            double maxQ = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < moveCount; i++) {
                double q = node.getQValue(i);
                if (q > maxQ) maxQ = q;
            }
            return maxQ;
        }
        else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            // Must draw cards - run rollouts
            return runRolloutsFromState(game, node.getLogicalPlayerIdx());
        }
        else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            // The drawn card is already in hand at this state
            // We need to evaluate both choices: keep it or play it
            
            // Get the drawn card (the last card in hand)
            Game tempGame = new Game(game);
            Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
            int drawnCardIndex = hand.size() - 1;
            Card drawnCard = hand.getCard(drawnCardIndex);
            
            // Check if the drawn card is playable
            boolean hasUnresolved = game.getUnresolvedCards().size() > 0;
            boolean canPlay;
            if (hasUnresolved) {
                canPlay = drawnCard.value() == Value.DRAW_TWO || drawnCard.value() == Value.WILD_DRAW_FOUR;
            } else {
                canPlay = drawnCard.color() == game.getCurrentColor() ||
                          drawnCard.value() == game.getLastPlayedCard().value() ||
                          drawnCard.isWild();
            }
            
            // Option 1: Keep the card - advance to next player with the card remaining in hand
            double keepValue = evaluateKeepCard(game, node.getLogicalPlayerIdx(), drawnCardIndex, drawnCardIdx);
            
            // Option 2: Play the card (if possible)
            double playValue = 0.0;
            if (canPlay) {
                playValue = evaluatePlayDrawnCard(game, node.getLogicalPlayerIdx(), drawnCardIndex, drawnCard, drawnCardIdx);
            }
            
            node.setQValueTotal(0, (float) keepValue);
            node.setQCount(0, 1);
            node.setQValueTotal(1, (float) playValue);
            node.setQCount(1, 1);
            
            return Math.max(keepValue, playValue);
        }
        
        return 0.0;
    }
    
    private double runRolloutsFromState(GameView game, int ourPlayerIdx) {
        int numSimulations = 5;
        double totalUtility = 0.0;
        
        for (int sim = 0; sim < numSimulations; sim++) {
            Game simGame = new Game(game);
            double utility = randomRollout(simGame, ourPlayerIdx);
            totalUtility += utility;
        }
        
        return totalUtility / numSimulations;
    }
    
    private double evaluateKeepCard(GameView game, int playerIdx, int drawnCardIndex, Integer originalDrawnCardIdx) {
        // Keep the card: it remains in hand, we just advance to next player
        // Create a new Game from the current state
        Game newGame = new Game(game);
        
        // The drawn card is already in hand, we just need to advance the turn
        int nextPlayer = (playerIdx + 1) % newGame.getNumPlayers();
        newGame.setCurrentMoveIdx(nextPlayer);
        
        GameView newGameView = newGame.getView(playerIdx);
        MCTSNode childNode = new MCTSNode(newGameView, newGameView.getPlayerOrder().getCurrentLogicalPlayerIdx(), null);
        return evaluateNode(childNode, originalDrawnCardIdx);
    }
    
    private double evaluatePlayDrawnCard(GameView game, int playerIdx, int drawnCardIndex, Card drawnCard, Integer originalDrawnCardIdx) {
        // Play the drawn card
        Move move;
        if (drawnCard.isWild()) {
            Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
            Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
            move = Move.createMove(this, drawnCardIndex, chosenColor);
        } else {
            move = Move.createMove(this, drawnCardIndex);
        }
        
        Game newGame = new Game(game);
        newGame.resolveMove(move);
        
        GameView newGameView = newGame.getView(playerIdx);
        MCTSNode childNode = new MCTSNode(newGameView, newGameView.getPlayerOrder().getCurrentLogicalPlayerIdx(), null);
        return evaluateNode(childNode, originalDrawnCardIdx);
    }
    
    private double randomRollout(Game game, int ourPlayerIdx) {
        int maxDepth = 50;
        int depth = 0;
        
        while (!game.isOver() && depth < maxDepth) {
            int currentPlayer = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
            Hand hand = game.getHand(currentPlayer);
            
            Set<Integer> legalMoveIndices = hand.getLegalMoves(game);
            
            if (legalMoveIndices == null || legalMoveIndices.isEmpty()) {
                // No legal moves - draw a card
                int drawnIdx = game.drawCard(hand);
                if (drawnIdx >= 0) {
                    // After drawing, check if we can play it
                    Set<Integer> newLegalMoves = hand.getLegalMoves(game);
                    if (newLegalMoves != null && newLegalMoves.contains(drawnIdx)) {
                        // Randomly decide to play or keep
                        if (this.getRandom().nextBoolean()) {
                            Card drawnCard = hand.getCard(drawnIdx);
                            Move move;
                            if (drawnCard.isWild()) {
                                Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                                Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
                                move = Move.createMove(game.getCurrentAgent(), drawnIdx, chosenColor);
                            } else {
                                move = Move.createMove(game.getCurrentAgent(), drawnIdx);
                            }
                            game.resolveMove(move);
                        } else {
                            int nextPlayer = (currentPlayer + 1) % game.getNumPlayers();
                            game.setCurrentMoveIdx(nextPlayer);
                        }
                    } else {
                        int nextPlayer = (currentPlayer + 1) % game.getNumPlayers();
                        game.setCurrentMoveIdx(nextPlayer);
                    }
                } else {
                    int nextPlayer = (currentPlayer + 1) % game.getNumPlayers();
                    game.setCurrentMoveIdx(nextPlayer);
                }
            } else {
                // Choose a random legal move
                List<Integer> movesList = new ArrayList<>(legalMoveIndices);
                int randomIdx = this.getRandom().nextInt(movesList.size());
                int cardIdx = movesList.get(randomIdx);
                Card cardToPlay = hand.getCard(cardIdx);
                
                Move move;
                if (cardToPlay.isWild()) {
                    Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                    Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
                    move = Move.createMove(game.getCurrentAgent(), cardIdx, chosenColor);
                } else {
                    move = Move.createMove(game.getCurrentAgent(), cardIdx);
                }
                game.resolveMove(move);
            }
            depth++;
        }
        
        if (game.isOver()) {
            return game.getHand(ourPlayerIdx).isEmpty() ? 1.0f : 0.0f;
        }
        
        int handSize = game.getHand(ourPlayerIdx).size();
        return (7.0f - Math.min(handSize, 7)) / 7.0f;
    }

    @Override
    public Move argmaxQValues(final Node node)
    {
        Node.NodeState state = node.getNodeState();
        
        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            List<Integer> moveIndices = node.getOrderedLegalMoves();
            
            if (moveIndices == null || moveIndices.isEmpty()) {
                return null;
            }
            
            int bestIdx = -1;
            float bestQ = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < moveIndices.size(); i++) {
                float qValue = node.getQValue(i);
                if (qValue > bestQ) {
                    bestQ = qValue;
                    bestIdx = i;
                }
            }
            
            if (bestIdx >= 0) {
                GameView game = node.getGameView();
                Game tempGame = new Game(game);
                Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
                Set<Integer> legalMoves = hand.getLegalMoves(tempGame);
                
                if (legalMoves != null && bestIdx < legalMoves.size()) {
                    List<Integer> movesList = new ArrayList<>(legalMoves);
                    int cardIdx = movesList.get(bestIdx);
                    Card card = hand.getCard(cardIdx);
                    
                    if (card.isWild()) {
                        Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                        Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
                        return Move.createMove(this, cardIdx, chosenColor);
                    } else {
                        return Move.createMove(this, cardIdx);
                    }
                }
            }
            return null;
        }
        else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            return null;
        }
        else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            float keepQ = node.getQValue(0);
            float playQ = node.getQValue(1);
            
            if (playQ > keepQ) {
                // Play the drawn card
                GameView game = node.getGameView();
                Game tempGame = new Game(game);
                Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
                Set<Integer> legalMoveIndices = hand.getLegalMoves(tempGame);
                
                if (legalMoveIndices != null && !legalMoveIndices.isEmpty()) {
                    int cardIdx = legalMoveIndices.iterator().next();
                    Card card = hand.getCard(cardIdx);
                    
                    if (card.isWild()) {
                        Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                        Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
                        return Move.createMove(this, cardIdx, chosenColor);
                    } else {
                        return Move.createMove(this, cardIdx);
                    }
                }
            }
            return null;
        }
        
        return null;
    }
}
