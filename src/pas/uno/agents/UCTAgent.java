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

    // =========================================================================
    // MCTSNode — mirrors the pattern from the working ExpectedOutcomeAgent
    // =========================================================================

    public static class MCTSNode extends Node {

        public MCTSNode(final GameView game,
                        final int logicalPlayerIdx,
                        final Node parent) {
            super(game, logicalPlayerIdx, parent);
        }

        @Override
        public Node getChild(final Move move) {
            // Exactly as in ExpectedOutcomeAgent — copy, resolve, re-view.
            Game gameCopy = new Game(this.getGameView());
            gameCopy.resolveMove(move);
            GameView newView = gameCopy.getView(this.getLogicalPlayerIdx());
            int nextPlayerIdx = newView.getCurrentMoveIdx();
            return new MCTSNode(newView, nextPlayerIdx, this);
        }
    }

    // =========================================================================
    // Constructor
    // =========================================================================

    public UCTAgent(final int playerIdx, final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
    }

    // =========================================================================
    // search  —  UCT main loop
    // =========================================================================

    @Override
    public Node search(final GameView game, final Integer drawnCardIdx) {

        int myPlayerIdx = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        // Root node built exactly as ExpectedOutcomeAgent builds its root.
        MCTSNode root = new MCTSNode(game, myPlayerIdx, null);

        long deadline = System.currentTimeMillis() + this.getMaxThinkingTimeInMS();

        while (System.currentTimeMillis() < deadline) {

            // 1. Selection
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
                    // 2. Expansion
                    int actionIdx = pickUntriedAction(current, actionIndices);
                    Move move = buildMove(current, actionIdx, drawnCardIdx);
                    MCTSNode child = (MCTSNode) current.getChild(move);

                    // 3. Simulation
                    float reward = rollout(child.getGameView(), myPlayerIdx);

                    // 4. Backpropagation
                    backpropagate(current, actionIdx, reward);
                    break;

                } else {
                    // Fully expanded — descend via UCB
                    int bestAction = selectUCB(current, actionIndices);
                    Move move = buildMove(current, bestAction, drawnCardIdx);
                    current = (MCTSNode) current.getChild(move);
                }
            }
        }

        return root;
    }

    // =========================================================================
    // argmaxQValues
    // =========================================================================

    @Override
    public Move argmaxQValues(final Node node) {

        Node.NodeState state = node.getNodeState();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            List<Integer> moveIndices = node.getOrderedLegalMoves();
            if (moveIndices == null || moveIndices.isEmpty()) return null;

            int bestIdx = -1;
            float bestQ = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < moveIndices.size(); i++) {
                float q = node.getQValue(i);
                if (q > bestQ) { bestQ = q; bestIdx = i; }
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
            return Move.createMove(this, 0);

        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
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
                return null;
            }
        }

        return null;
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    private List<Integer> getActionIndices(Node node) {
        List<Integer> actions = new ArrayList<>();
        Node.NodeState state = node.getNodeState();
        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            List<Integer> legal = node.getOrderedLegalMoves();
            for (int i = 0; i < legal.size(); i++) actions.add(i);
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            actions.add(0);
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            actions.add(0);
            actions.add(1);
        }
        return actions;
    }

    private int pickUntriedAction(Node node, List<Integer> actionIndices) {
        List<Integer> untried = new ArrayList<>();
        for (int ai : actionIndices) {
            if (node.getQCount(ai) == 0) untried.add(ai);
        }
        return untried.get(this.getRandom().nextInt(untried.size()));
    }

    private int selectUCB(Node node, List<Integer> actionIndices) {
        double logN = Math.log(node.getStateCount());
        int bestIdx = actionIndices.get(0);
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int ai : actionIndices) {
            long count = node.getQCount(ai);
            if (count == 0) return ai;
            double ucb = node.getQValue(ai) + Math.sqrt(2.0 * logN / count);
            if (ucb > bestVal) { bestVal = ucb; bestIdx = ai; }
        }
        return bestIdx;
    }

    private Move buildMove(Node node, int actionIdx, Integer drawnCardIdx) {
        Node.NodeState state = node.getNodeState();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            int cardIdx = node.getOrderedLegalMoves().get(actionIdx);
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
            // NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD: 0=keep, 1=play
            if (actionIdx == 0) {
                return Move.createMove(this, 0);
            } else {
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
     * Random rollout from the given GameView.
     *
     * Crucially: we NEVER create MCTSNode objects here. Node construction
     * triggers the Node base-class constructor which internally calls
     * new Game(view) — if that view has UNKNOWN cards it throws. Instead we
     * detect game state directly using Game + Hand methods, all of which are
     * documented and safe.
     *
     * State detection without NodeState:
     *   - If game.getUnresolvedCards().size() > 0  → must draw unresolved stack
     *     (NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT)
     *   - Else if hand.getLegalMoves(game) is empty → must draw one card
     *     (NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD — we just keep it, index 0)
     *   - Else → pick a random legal card to play (HAS_LEGAL_MOVES)
     *
     * We need getUnresolvedCards().size() — UnresolvedCardBuffer is undocumented
     * but since Game.getUnresolvedCards() returns it and the ExpectedOutcomeAgent
     * never needed to call it, we instead rely on hasLegalMoves() + the fact that
     * if hasLegalMoves is false AND there are unresolved draw cards the engine
     * will handle it. We detect by trying getLegalMoves first; if the set is
     * non-empty we play; if empty we signal draw/keep with index 0 which covers
     * both no-legal-move cases safely.
     */
    private float rollout(GameView startView, int myPlayerIdx) {
        GameView currentView = startView;
        Random rand = this.getRandom();

        int maxSteps = 200;
        for (int step = 0; step < maxSteps; step++) {
            if (currentView.isOver()) break;

            int currentPlayerIdx = currentView.getCurrentMoveIdx();
            Game tempGame = new Game(currentView);
            Hand hand = tempGame.getHand(currentPlayerIdx);

            Set<Integer> legalMoves = hand.getLegalMoves(tempGame);

            if (legalMoves == null || legalMoves.isEmpty()) {
                // Either unresolved draw stack or must draw single card.
                // In both cases pass cardToPlayIdx=0 — the engine handles it.
                Move move = Move.createMove(this, 0);
                tempGame.resolveMove(move);
                currentView = tempGame.getView(currentPlayerIdx);
                continue;
            }

            // Pick a random legal move.
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
            currentView = tempGame.getView(currentPlayerIdx);
        }

        // Determine winner by finding the player with the empty (or smallest) hand.
        int numPlayers = currentView.getNumPlayers();
        int minSize = Integer.MAX_VALUE;
        int winnerIdx = -1;
        for (int i = 0; i < numPlayers; i++) {
            Game g = new Game(currentView);
            int sz = g.getHand(i).size();
            if (sz < minSize) { minSize = sz; winnerIdx = i; }
        }

        if (currentView.isOver()) {
            return (winnerIdx == myPlayerIdx) ? 1.0f : 0.0f;
        }
        // Heuristic for games that hit the step cap.
        int mySize = new Game(currentView).getHand(myPlayerIdx).size();
        return (mySize == minSize) ? 0.6f : 0.3f;
    }

    private void backpropagate(MCTSNode node, int actionIdx, float reward) {
        node.setQCount(actionIdx, node.getQCount(actionIdx) + 1);
        node.setQValueTotal(actionIdx, node.getQValueTotal(actionIdx) + reward);
    }

    private Color chooseBestColor(Hand hand) {
        Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW};
        int[] counts = new int[4];
        for (int i = 0; i < hand.size(); i++) {
            Card c = hand.getCard(i);
            if (!c.isWild() && c.color() != null) {
                for (int j = 0; j < colors.length; j++) {
                    if (c.color() == colors[j]) { counts[j]++; break; }
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