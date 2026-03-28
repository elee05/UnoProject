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


public class UCTAgent
    extends MCTSAgent
{

    // Exploration constant for UCB formula
    private static final double EXPLORATION_CONSTANT = Math.sqrt(2.0);

    public static class UCTNode
        extends Node
    {
        public UCTNode(final GameView game,
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
            return new UCTNode(newGameView, nextPlayerIdx, this);
        }
    }

    public UCTAgent(final int playerIdx,
                    final long maxThinkingTimeInMS)
    {
        super(playerIdx, maxThinkingTimeInMS);
    }

    @Override
    public Node search(final GameView game,
                       final Integer drawnCardIdx)
    {
        int currentPlayer = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        UCTNode root = new UCTNode(game, currentPlayer, null);
        
        // Initialize root's state count to 1
        root.setQCount(0, 1);
        root.setQValueTotal(0, 0.0f);
        
        long startTime = System.currentTimeMillis();
        long maxTime = this.getMaxThinkingTimeInMS();
        
        int iterations = 0;
        while (System.currentTimeMillis() - startTime < maxTime) {
            // 1. Selection + Expansion
            UCTNode selected = select(root);
            
            // 2. Simulation
            double result = simulate(selected);
            
            // 3. Backpropagation
            backpropagate(selected, result);
            
            iterations++;
        }
        
        return root;
    }
    
    /**
     * Select a node using UCB, expanding as we go
     */
    private UCTNode select(UCTNode node) {
        while (true) {
            // If node is terminal, return it
            if (node.isTerminal()) {
                return node;
            }
            
            // Get all possible moves from this node
            GameView game = node.getGameView();
            Game tempGame = new Game(game);
            Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
            Set<Integer> legalMoveIndices = hand.getLegalMoves(tempGame);
            
            if (legalMoveIndices == null || legalMoveIndices.isEmpty()) {
                return node;
            }
            
            // Collect all children
            List<UCTNode> children = new ArrayList<>();
            List<Move> allMoves = new ArrayList<>();
            
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
                allMoves.add(move);
                UCTNode child = (UCTNode) node.getChild(move);
                children.add(child);
            }
            
            // Check if any child has zero visits
            for (UCTNode child : children) {
                if (child.getStateCount() == 0) {
                    return child; // Expand this unvisited child
                }
            }
            
            // Otherwise, select the child with highest UCB value
            UCTNode bestChild = null;
            double bestUCB = Double.NEGATIVE_INFINITY;
            double parentVisits = node.getStateCount();
            
            for (UCTNode child : children) {
                double qValue = child.getQValue(0);
                long childVisits = child.getStateCount();
                
                // childVisits is > 0 here because we checked above
                double ucb = qValue + EXPLORATION_CONSTANT * Math.sqrt(2.0 * Math.log(parentVisits) / childVisits);
                
                if (ucb > bestUCB) {
                    bestUCB = ucb;
                    bestChild = child;
                }
            }
            
            if (bestChild == null) {
                return node;
            }
            node = bestChild;
        }
    }
    
    /**
     * Run a random simulation from a node using heuristic
     */
    private double simulate(UCTNode node) {
        if (node.isTerminal()) {
            GameView game = node.getGameView();
            HandView playerHand = game.getHandView(node.getLogicalPlayerIdx());
            return playerHand.size() == 0 ? 1.0 : 0.0;
        }
        
        // Use heuristic: get the best possible card value from the current state
        GameView game = node.getGameView();
        Game tempGame = new Game(game);
        Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
        
        Set<Integer> legalMoves = hand.getLegalMoves(tempGame);
        if (legalMoves == null || legalMoves.isEmpty()) {
            return 0.0;
        }
        
        double bestValue = 0.0;
        for (int idx : legalMoves) {
            Card card = hand.getCard(idx);
            double value = evaluateCardValue(card);
            if (value > bestValue) {
                bestValue = value;
            }
        }
        
        return bestValue;
    }
    
    /**
     * Evaluate card value for heuristic
     */
    private double evaluateCardValue(Card card) {
        if (card.value() == Value.WILD_DRAW_FOUR) return 1.0;
        if (card.value() == Value.DRAW_TWO) return 0.9;
        if (card.isWild()) return 0.8;
        if (card.value() == Value.SKIP || card.value() == Value.REVERSE) return 0.7;
        return 0.5;
    }
    
    /**
     * Backpropagate result up the tree
     */
    private void backpropagate(UCTNode node, double result) {
        UCTNode current = node;
        while (current != null) {
            long newCount = current.getStateCount() + 1;
            float currentTotal = current.getQValueTotal(0);
            current.setQValueTotal(0, currentTotal + (float) result);
            current.setQCount(0, newCount);
            current = (UCTNode) current.getParent();
        }
    }

    @Override
    public Move argmaxQValues(final Node node)
    {
        Node.NodeState state = node.getNodeState();
        
        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            GameView game = node.getGameView();
            Game tempGame = new Game(game);
            Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
            Set<Integer> legalMoveIndices = hand.getLegalMoves(tempGame);
            
            if (legalMoveIndices == null || legalMoveIndices.isEmpty()) {
                return null;
            }
            
            // Find the move with highest Q-value
            int bestIdx = -1;
            double bestQ = Double.NEGATIVE_INFINITY;
            List<Integer> movesList = new ArrayList<>(legalMoveIndices);
            
            for (int i = 0; i < movesList.size(); i++) {
                int cardIdx = movesList.get(i);
                Card card = hand.getCard(cardIdx);
                
                Move move;
                if (card.isWild()) {
                    Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                    Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
                    move = Move.createMove(this, cardIdx, chosenColor);
                } else {
                    move = Move.createMove(this, cardIdx);
                }
                
                UCTNode child = (UCTNode) node.getChild(move);
                double qValue = child.getQValue(0);
                
                if (!Double.isNaN(qValue) && qValue > bestQ) {
                    bestQ = qValue;
                    bestIdx = i;
                }
            }
            
            if (bestIdx >= 0) {
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
            
            // Fallback: return first legal move
            int firstIdx = movesList.get(0);
            Card firstCard = hand.getCard(firstIdx);
            if (firstCard.isWild()) {
                Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                Color chosenColor = colors[this.getRandom().nextInt(colors.length)];
                return Move.createMove(this, firstIdx, chosenColor);
            } else {
                return Move.createMove(this, firstIdx);
            }
        }
        else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            return null;
        }
        else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            return null;
        }
        
        return null;
    }
}
