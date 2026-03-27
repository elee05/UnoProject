package src.pas.uno.agents;


// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.Game.PlayerOrder;
import edu.bu.pas.uno.Hand;
import edu.bu.pas.uno.Hand.HandView;
import edu.bu.pas.uno.agents.Agent;
import edu.bu.pas.uno.agents.MCTSAgent;
import edu.bu.pas.uno.agents.RandomAgent;
import edu.bu.pas.uno.enums.Color;
import edu.bu.pas.uno.enums.Value;
import edu.bu.pas.uno.moves.Move;
import edu.bu.pas.uno.tree.Node;
import edu.bu.pas.uno.tree.Node.NodeState;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;


// JAVA PROJECT IMPORTS 

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
        //!each node represents a state of the game
        //!please look at my implementation
        @Override
        public Node getChild(final Move move)
        {
            
            Game childGame = new Game(this.getGameView());
            childGame.resolveMove(move);
            GameView childGameView = childGame.getOmniscientView();
            int nextLogicalIdx = childGameView.getCurrentMoveIdx();

            Node childNode = new MCTSNode(childGameView, nextLogicalIdx, this);

            // Node next = new  MCTSNode(childGame.getView(getLogicalPlayerIdx()), this.getLogicalPlayerIdx(),this);
            return childNode;

        }
    }


    public ExpectedOutcomeAgent(final int playerIdx,final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
    }

    private Color getMostCommonColor(Hand.HandView hand) {
        Map<Color, Integer> colorCounts = new HashMap<>();
        for (int i = 0; i < hand.size(); i++) {
            Color c = hand.getCard(i).color();
            if (c != Color.UNKNOWN && c != null) {
                colorCounts.put(c, colorCounts.getOrDefault(c, 0) + 1);
            }
        }
        Color mostCommonColor = null;
        int maxCount = Integer.MIN_VALUE;

        for (Map.Entry<Color, Integer> entry : colorCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostCommonColor = entry.getKey();
            }
        }
        return mostCommonColor;
    }

    private void backpropagate(MCTSNode leaf, float outcome) {
        MCTSNode current = leaf;
        while (current != null) {
            Node parent = current.getParent();
            if (parent == null) break;

            // You need to know which move/index from parent led to current
            // One way: during selection record path
            // Alternative (approximate): assume uniform update — but not correct

            // Better: store chosen index in child or pass it
            current = (MCTSNode) parent;
        }
    }


    /**
     * A method to perform the MCTS search on the game tree
     *
     * @param   game            The {@link GameView} that should be the root of the game tree
     * @param   drawnCardIdx    This will be non-null when this method is being called by the 
     *                          <code>maybePlayDrawnCard</code> method of {@link Agent} and will
     *                          be <code>null</code> when being called by <code>chooseCardToPlay</code>
     *                          method of {@link Agent}
     * @return  The {@link Node} of the root who'se q-values should now be populated and ready to argmax
     */

    

    private double rollout(Node node) { 

        // assigning all agents random
        // ? agentIdx
        // ? logicalPlayerIdx

        int ourPlayerIdx = this.getLogicalPlayerIdx();


        // get all of the "playerIdxs"
        GameView game = node.getGameView();
        int numPlayers = game.getNumPlayers();
        Game.PlayerOrder order = game.getPlayerOrder();

        // To get logical index for every physical player index
        List<Integer> logicalPlayerIdxs = new ArrayList<>();
        for (int i = 0; i < numPlayers; i++) {
            int logicalIdx = order.getLogicalIdx(i);
            logicalPlayerIdxs.add(logicalIdx);
        }

        // want to create new game replacing agents with random agents
        // how to replicate current game state


        Agent[] agents = new Agent[logicalPlayerIdxs.size()];

        for (int i = 0; i < logicalPlayerIdxs.size(); i++) {
            Agent a = new RandomAgent(logicalPlayerIdxs.get(i),5 );
            agents[i] = a;
        }
        

        Game simGame = new Game(game,agents);

        while (!simGame.isOver()) {
            Move nextMove = simGame.getMove();
            simGame.resolveMove(nextMove);
        }

        PlayerOrder finalOrder = simGame.getPlayerOrder();
        int winnerLogPlayIdx = finalOrder.getCurrentLogicalPlayerIdx();

        if (winnerLogPlayIdx == ourPlayerIdx) {
            return 1;
        } else {
            return 0;
        }
    }

    public Node minimax(Node node)
    {
        // uncomment if you want to see the tree being made
        // System.out.println(this.getTabs(node) + "Node(currentPlayer=" + node.getCurrentPlayerType() +
        //      " isTerminal=" + node.isTerminal() + " lastMove=" + node.getLastMove() + ")" + "children: {" + node.getChildren() + "}");

        System.out.println();
        System.out.println("player type is: " + node.getMyPlayerType());
        System.out.println("maxxing player: " + (node.getCurrentPlayerType()==node.getMyPlayerType()));
        System.out.println(); 

        if (node.isTerminal()) {
            node.setUtilityValue(node.getTerminalUtility());
            return node;
        }
        if (node.getDepth() == this.maxDepth) {
            double h = Heuristics.calculateHeuristicValue(node);
            node.setUtilityValue(h);
            return node;
        }
    
        

        if (node.getCurrentPlayerType()==node.getMyPlayerType()) {
            Node bestChild = null;
            double maxUtility = Integer.MIN_VALUE;
            for (Node c : node.getChildren()) {
                Node result = minimax(c);
                if (result.getUtilityValue() > maxUtility) {
                    maxUtility = result.getUtilityValue();
                    bestChild = c;
                }
            }
            node.setUtilityValue(maxUtility);
            return bestChild;

        } else {
            Node bestChild = null;
            double minUtility = Integer.MAX_VALUE;
            for (Node c : node.getChildren()) {
                Node result = minimax(c);
                if (result.getUtilityValue() < minUtility) {
                    minUtility = result.getUtilityValue();
                    bestChild = c;
                }
            }
            node.setUtilityValue(minUtility);
            return bestChild;
        }
    }
    
    @Override
    public Node search(final GameView game, final Integer drawnCardIdx) {
        MCTSNode root = new MCTSNode(game, this.getLogicalPlayerIdx(), null);
    
        int logicalPlayerIdx = this.getLogicalPlayerIdx();
        Hand.HandView hand = game.getHandView(logicalPlayerIdx);
        int numRollouts = 10;

        if (root.getNodeState() == Node.NodeState.HAS_LEGAL_MOVES) {
            List<Integer> legalMoves = root.getOrderedLegalMoves();

            for (int moveIdx : legalMoves) {

                    Card card = hand.getCard(moveIdx);
                    Move aMove = null;
                    int colIdx = this.getRandom().nextInt(4);
                    if (card.isWild()) {
                        aMove = Move.createMove(this, moveIdx, Color.values()[colIdx]);
                    } else {
                        aMove = Move.createMove(this, moveIdx);
                    }  
                    Node child = root.getChild(aMove);
                    float wins = 0;

                    for (int i = 0; i < numRollouts; i++) {
                        wins += rollout(child);
                    }

                    root.setQValueTotal(moveIdx,  wins/numRollouts);
                    root.setQCount(moveIdx, numRollouts);
            }
        } else  if (root.getNodeState() == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            List<Integer> choices =  new ArrayList<>(Arrays.asList(
                    Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX,
                    Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX
            ));

            Move aMove = null;

            int choiceIdx = choices.get(this.getRandom().nextInt(2));
            if (choiceIdx == Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX) { //playing drawn card
                aMove = Move.createMove(this, choiceIdx);
            } else { //doing nothing
                System.out.println("forced null move");
                aMove = null;
            } 

            Node child = root.getChild(aMove);
            float wins = 0;

            for (int i = 0; i < numRollouts; i++) {
                wins += rollout(child);
            }

            root.setQValueTotal(choiceIdx,  wins/numRollouts);
            root.setQCount(choiceIdx, numRollouts);

            
        } else {
            Move aMove = Move.createMove(this, Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs.MOVE_IDX);
            Node child = root.getChild(aMove);
            float wins = 0;

            for (int i = 0; i < numRollouts; i++) {
                wins += rollout(child);
            }

            root.setQValueTotal(Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs.MOVE_IDX,  wins/numRollouts);
            root.setQCount(Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs.MOVE_IDX, numRollouts);
        }

    // System.out.println("Completed " + totalSims + " simulations"); // optional debug
        return root;
    }

    
    
    @Override
    public Move argmaxQValues(final Node node)
    {

        /**
     * A method to argmax the Q values inside a {@link Node}
     *
     * @param   node            The {@link Node} who has populated q-values
     * @return  The {@link Move} corresponding to whichever {@link Move} has the largest q-value. Note
     *          that this can be <code>null</code> if you choose to not play the drawn card (you will
     *          have to detect whether or not you are in that scenario by examining the @{link Node}'s state).
     */

    
        //Node variable establishment
        GameView view = node.getGameView();
        int logicalPlayerIdx = node.getLogicalPlayerIdx();
        Hand.HandView hand = view.getHandView(logicalPlayerIdx);
        Move nextMove = null;

// look at legal cards via game state
// ? store drawn card in instance variable

        // TODO HAS LEGAL MOVES IN HAND CASE

        if (node.getNodeState() == NodeState.HAS_LEGAL_MOVES) {
            Float maxUtil = Float.NEGATIVE_INFINITY;
            Integer idxOfBestCardinHand = 0;
            List<Integer> legalMoves = node.getOrderedLegalMoves();
            for (int cardIdx: legalMoves) {
                if (node.getQValue(cardIdx) > maxUtil) {
                    maxUtil = node.getQValue(cardIdx);
                    idxOfBestCardinHand = cardIdx;
                }
            }
            
                        
            Card card = hand.getCard(idxOfBestCardinHand);
            if (card.isWild()) {
                nextMove = Move.createMove(this, idxOfBestCardinHand, getMostCommonColor(hand));
            } else {
                nextMove = Move.createMove(this, idxOfBestCardinHand);
            }        
        

        // TODO NO LEGAL MOVES MAY PLAY DRAWN CARD
        } else if (node.getNodeState() == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            // !FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFf
            // ? more clarification,  how does choosing the optimal move change whne you draw a card vs not draw a card

            float keepQValue = node.getQValue(Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX); // DrawSingleCardIdxs.KEEP_IDX
            float playQValue = node.getQValue(Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX); // DrawSingleCardIdxs.PLAY_IDX

            if (playQValue > keepQValue) {
                // The drawn card is the last card in hand
                int drawnCardIdx = Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX;
                Card drawnCard = hand.getCard(drawnCardIdx);

                if (drawnCard.isWild()) {
                    nextMove = Move.createMove(this, drawnCardIdx, getMostCommonColor(hand));
                }
                nextMove = Move.createMove(this, drawnCardIdx);
            } else {
                System.out.println("forced null move");
                nextMove = null;
            }

        // TODO DRAW X dropped on player and node valid moves
        } else {
            nextMove = Move.createMove(this, Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs.MOVE_IDX);
        }
        return nextMove;
    }
}

// while (!cur.isTerminal()) {

//             List<Integer> legalMoves = cur.getOrderedLegalMoves();
//             Game.GameView game = cur.getGameView();
//             Hand.HandView hand = game.getHandView(cur.getLogicalPlayerIdx());

//             Move rMove = null;



//             // if (cur.getNodeState() == NodeState.HAS_LEGAL_MOVES) { //! HAS LEGAL MOVES CON

//             //     int listIdx = this.getRandom().nextInt(legalMoves.size());
//             //     int moveIdx = legalMoves.get(listIdx);
//             //     Card card = hand.getCard(moveIdx);
//             //     Move aMove = null; 
//             //     int colIdx = this.getRandom().nextInt(4);


//             //     if (card.isWild()) {
//             //         aMove = Move.createMove(this, moveIdx, Color.values()[colIdx]);
//             //     } else {
//             //         aMove = Move.createMove(this, moveIdx);
//             //     }  

//             // } else if (cur.getNodeState() == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) { //! NO LEGAL MOVES or DRAW CON
                 
//             //     List<Integer> choices =  new ArrayList<>(Arrays.asList(
//             //         Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.KEEP_CARD_MOVE_IDX,
//             //         Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX
//             //     ));

//             //     int choiceIdx = choices.get(this.getRandom().nextInt(2));
//             //     if (choiceIdx == Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX) { //playing drawn card
//             //         rMove = Move.createMove(this, choiceIdx);
//             //     } else { //doing nothing
//             //         System.out.println("forced null move");
//             //         rMove = null;
//             //     } 
//             // } else {
//             //     rMove = Move.createMove(this, Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs.MOVE_IDX);
//             // }

            
            
//             cur = cur.getChild(rMove);

//         }

// Node cur  = node;

//         while (!cur.isTerminal()) {
//             Game.GameView game = cur.getGameView();
//             Game staticGame = new Game(game);
//             if (cur.getNodeState() == NodeState.HAS_LEGAL_MOVES) {
//                 Move rMove = this.chooseCardToPlay(game);
//                 cur = cur.getChild(rMove);


//             } else if (cur.getNodeState() == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
//                 // Move rMove = this.maybePlayDrawnCard(game, cur.getLogicalPlayerIdx());
//                 // cur = cur.getChild(rMove);
//                 Move rMove = null;
//                 boolean play = this.getRandom().nextBoolean();
//                 if (play) {
//                     rMove = Move.createMove(this, Node.NoLegalMovesIdxDefaults.DrawSingleCardIdxs.PLAY_CARD_MOVE_IDX);
//                 } else {
//                     rMove = null; // keep card
//                 }
//                 cur = cur.getChild(rMove);


//             } else {
//                 Move rMove = Move.createMove(staticGame.getCurrentAgent(), Node.NoLegalMovesIdxDefaults.DrawUnresolvedCardsIdxs.MOVE_IDX );
//                 cur = cur.getChild(rMove);
//             }
//         }

//         return cur.getUtilityValues();