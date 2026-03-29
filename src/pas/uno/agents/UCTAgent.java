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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;


public class UCTAgent extends MCTSAgent {

    // -------------------------------------------------------------------------
    // MCTSNode — identical pattern to the working ExpectedOutcomeAgent
    // -------------------------------------------------------------------------

    public static class MCTSNode extends Node {

        public MCTSNode(final GameView game,
                        final int logicalPlayerIdx,
                        final Node parent) {
            super(game, logicalPlayerIdx, parent);
        }

        @Override
        public Node getChild(final Move move) {
            Game gameCopy = new Game(this.getGameView());
            gameCopy.resolveMove(move);
            GameView newView = gameCopy.getOmniscientView();
            int nextPlayerIdx = newView.getCurrentMoveIdx();
            return new MCTSNode(newView, nextPlayerIdx, this);
        }
    }

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    public UCTAgent(final int playerIdx, final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
    }

    // -------------------------------------------------------------------------
    // search  —  UCT main loop
    // -------------------------------------------------------------------------

    @Override
    public Node search(final GameView game, final Integer drawnCardIdx) {

        int myPlayerIdx = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        Game rootGame = new Game(game);
        GameView omniscientView = rootGame.getOmniscientView();
        MCTSNode root = new MCTSNode(omniscientView, myPlayerIdx, null);

        long deadline = System.currentTimeMillis() + this.getMaxThinkingTimeInMS();

        while (System.currentTimeMillis() < deadline) {

            // 1. Selection: walk down with UCB until terminal or not fully expanded.
            MCTSNode current = root;

            while (!current.isTerminal()) {
                List<Integer> actionIndices = getActionIndices(current);

                boolean fullyExpanded = true;
                for (int ai : actionIndices) {
                    if (current.getQCount(ai) == 0) {
                        fullyExpanded = false;
                        break;
                    }
                }

                if (!fullyExpanded) {
                    // 2. Expansion: pick a random untried action.
                    int actionIdx = pickUntriedAction(current, actionIndices);
                    Move move = buildMove(current, actionIdx, drawnCardIdx);
                    MCTSNode child = (MCTSNode) current.getChild(move);

                    // 3. Simulation: random rollout from the new child.
                    float reward = rollout(child, myPlayerIdx);

                    // 4. Backpropagation.
                    backpropagate(current, actionIdx, reward);
                    break;

                } else {
                    // Fully expanded — descend via UCB.
                    int bestAction = selectUCB(current, actionIndices);
                    Move move = buildMove(current, bestAction, drawnCardIdx);
                    current = (MCTSNode) current.getChild(move);
                }
            }
        }

        return root;
    }

    // -------------------------------------------------------------------------
    // argmaxQValues  —  select best move from root after search
    // -------------------------------------------------------------------------

    @Override
    public Move argmaxQValues(final Node node) {

        Node.NodeState state = node.getNodeState();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            List<Integer> moveIndices = node.getOrderedLegalMoves();
            if (moveIndices == null || moveIndices.isEmpty()) {
                return null;
            }

            int bestIdx = -1;
            float bestQ = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < moveIndices.size(); i++) {
                float q = node.getQValue(i);
                if (q > bestQ) {
                    bestQ = q;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) return null;

            Game tempGame = new Game(node.getGameView());
            Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
            Set<Integer> legalMoves = hand.getLegalMoves(tempGame);
            if (legalMoves == null || legalMoves.isEmpty()) return null;

            List<Integer> movesList = new ArrayList<>(legalMoves);
            if (bestIdx >= movesList.size()) return null;

            int cardIdx = movesList.get(bestIdx);
            Card card = hand.getCard(cardIdx);

            if (card.isWild()) {
                return Move.createMove(this, cardIdx, chooseBestColor(hand));
            } else {
                return Move.createMove(this, cardIdx);
            }

        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            // Single q-value at index 0 — must draw the unresolved stack.
            return Move.createMove(this, 0);

        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            // Q-value 0 = keep, Q-value 1 = play.
            float keepQ = node.getQValue(0);
            float playQ = node.getQValue(1);

            if (playQ > keepQ) {
                Game tempGame = new Game(node.getGameView());
                Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
                int drawnIdx = hand.size() - 1;
                Card card = hand.getCard(drawnIdx);
                if (card.isWild()) {
                    return Move.createMove(this, drawnIdx, chooseBestColor(hand));
                } else {
                    return Move.createMove(this, drawnIdx);
                }
            } else {
                return null; // keep the card
            }
        }

        return null;
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Returns valid action indices for a node.
     * HAS_LEGAL_MOVES: 0-based positions into getOrderedLegalMoves().
     * NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT: {0}
     * NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD: {0=keep, 1=play}
     */
    private List<Integer> getActionIndices(Node node) {
        List<Integer> actions = new ArrayList<>();
        Node.NodeState state = node.getNodeState();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            List<Integer> legal = node.getOrderedLegalMoves();
            for (int i = 0; i < legal.size(); i++) {
                actions.add(i);
            }
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            actions.add(0);
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            actions.add(0);
            actions.add(1);
        }

        return actions;
    }

    /** Picks a random untried (Q-count == 0) action. */
    private int pickUntriedAction(Node node, List<Integer> actionIndices) {
        List<Integer> untried = new ArrayList<>();
        for (int ai : actionIndices) {
            if (node.getQCount(ai) == 0) {
                untried.add(ai);
            }
        }
        return untried.get(this.getRandom().nextInt(untried.size()));
    }

    /** UCB selection: argmax_a [ Q̄(s,a) + sqrt( 2 * ln(N(s)) / N(s,a) ) ] */
    private int selectUCB(Node node, List<Integer> actionIndices) {
        double logN = Math.log(node.getStateCount());
        int bestIdx = actionIndices.get(0);
        double bestVal = Double.NEGATIVE_INFINITY;

        for (int ai : actionIndices) {
            long count = node.getQCount(ai);
            if (count == 0) return ai;
            double avgQ = node.getQValue(ai);
            double ucb = avgQ + Math.sqrt(2.0 * logN / count);
            if (ucb > bestVal) {
                bestVal = ucb;
                bestIdx = ai;
            }
        }
        return bestIdx;
    }

    /**
     * Constructs the Move for a given node + action index.
     * Only uses the two documented Move.createMove factory methods.
     */
    private Move buildMove(Node node, int actionIdx, Integer drawnCardIdx) {
        Node.NodeState state = node.getNodeState();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            List<Integer> ordered = node.getOrderedLegalMoves();
            int cardIdx = ordered.get(actionIdx);

            Game tempGame = new Game(node.getGameView());
            Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
            Card card = hand.getCard(cardIdx);

            if (card.isWild()) {
                return Move.createMove(this, cardIdx, chooseBestColor(hand));
            } else {
                return Move.createMove(this, cardIdx);
            }

        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            return Move.createMove(this, 0);

        } else {
            // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD
            if (actionIdx == 0) {
                // Keep — pass 0 to signal keep.
                return Move.createMove(this, 0);
            } else {
                // Play the drawn card.
                if (drawnCardIdx != null) {
                    Game tempGame = new Game(node.getGameView());
                    Hand hand = tempGame.getHand(node.getLogicalPlayerIdx());
                    Card card = hand.getCard(drawnCardIdx);
                    if (card.isWild()) {
                        return Move.createMove(this, drawnCardIdx, chooseBestColor(hand));
                    }
                    return Move.createMove(this, drawnCardIdx);
                }
                return Move.createMove(this, 0); // fallback: keep
            }
        }
    }

    /**
     * Random rollout. Uses only documented API methods:
     * - new Game(GameView), game.resolveMove(Move), game.getView(int),
     *   game.getHand(int), hand.getLegalMoves(Game), hand.getCard(int),
     *   hand.size(), gameView.isOver(), gameView.getCurrentMoveIdx(),
     *   gameView.getNumPlayers(), node.getNodeState(), node.isTerminal(),
     *   node.getUtilityValues().
     * HandView and UnresolvedCardBuffer methods are intentionally avoided.
     */
    private float rollout(MCTSNode startNode, int myPlayerIdx) {
        if (startNode.isTerminal()) {
            return startNode.getUtilityValues();
        }

        GameView currentView = startNode.getGameView();
        Random rand = this.getRandom();

        int maxSteps = 200;
        for (int step = 0; step < maxSteps; step++) {
            if (currentView.isOver()) break;

            int currentPlayerLogicalIdx = currentView.getCurrentMoveIdx();

            // Build a temporary node to read NodeState without calling any
            // undocumented methods on UnresolvedCardBuffer or HandView.
            MCTSNode tempNode = new MCTSNode(
                currentView, currentPlayerLogicalIdx, null);
            Node.NodeState tempState = tempNode.getNodeState();

            Game tempGame = new Game(currentView);
            Hand hand = tempGame.getHand(currentPlayerLogicalIdx);

            if (tempState == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                Move move = Move.createMove(this, 0);
                tempGame.resolveMove(move);
                currentView = tempGame.getOmniscientView();
                continue;
            }

            if (tempState == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
                // Keep the drawn card.
                Move move = Move.createMove(this, 0);
                tempGame.resolveMove(move);
                currentView = tempGame.getOmniscientView();
                continue;
            }

            // HAS_LEGAL_MOVES — pick a random one.
            Set<Integer> legalMoves = hand.getLegalMoves(tempGame);
            if (legalMoves == null || legalMoves.isEmpty()) {
                Move move = Move.createMove(this, 0);
                tempGame.resolveMove(move);
                currentView = tempGame.getOmniscientView();
                continue;
            }

            List<Integer> legalList = new ArrayList<>(legalMoves);
            int cardIdx = legalList.get(rand.nextInt(legalList.size()));
            Card card = hand.getCard(cardIdx);

            Move move;
            if (card.isWild()) {
                Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
                move = Move.createMove(this, cardIdx, colors[rand.nextInt(colors.length)]);
            } else {
                move = Move.createMove(this, cardIdx);
            }

            tempGame.resolveMove(move);
            currentView = tempGame.getOmniscientView();
        }

        // Determine winner: use Hand.size() via Game.getHand() — both documented.
        int numPlayers = currentView.getNumPlayers();
        int minSize = Integer.MAX_VALUE;
        int winnerIdx = -1;
        for (int i = 0; i < numPlayers; i++) {
            Game g = new Game(currentView);
            int sz = g.getHand(i).size();
            if (sz < minSize) {
                minSize = sz;
                winnerIdx = i;
            }
        }

        if (currentView.isOver()) {
            return (winnerIdx == myPlayerIdx) ? 1.0f : 0.0f;
        }

        // Not finished within step budget — heuristic.
        int mySize = new Game(currentView).getHand(myPlayerIdx).size();
        return (mySize == minSize) ? 0.6f : 0.3f;
    }

    /** Updates Q-statistics for one (node, action) pair. */
    private void backpropagate(MCTSNode node, int actionIdx, float reward) {
        long oldCount = node.getQCount(actionIdx);
        float oldTotal = node.getQValueTotal(actionIdx);
        node.setQCount(actionIdx, oldCount + 1);
        node.setQValueTotal(actionIdx, oldTotal + reward);
    }

    /** Picks the color held most in hand; falls back to RED. */
    private Color chooseBestColor(Hand hand) {
        Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
        int[] counts = new int[4];

        for (int i = 0; i < hand.size(); i++) {
            Card c = hand.getCard(i);
            if (!c.isWild() && c.color() != null) {
                for (int j = 0; j < colors.length; j++) {
                    if (c.color() == colors[j]) {
                        counts[j]++;
                        break;
                    }
                }
            }
        }

        int best = 0;
        for (int j = 1; j < 4; j++) {
            if (counts[j] > counts[best]) best = j;
        }
        return colors[best];
    }
}