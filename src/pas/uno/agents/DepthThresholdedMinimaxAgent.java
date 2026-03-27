package src.labs.rttt.agents;


// SYSTEM IMPORTS
import edu.bu.labs.rttt.agents.SearchAgent;
import edu.bu.labs.rttt.game.CellType;
import edu.bu.labs.rttt.game.PlayerType;
import edu.bu.labs.rttt.game.RecursiveTicTacToeGame;
import edu.bu.labs.rttt.game.RecursiveTicTacToeGame.RecursiveTicTacToeGameView;
import edu.bu.labs.rttt.traversal.Node;
import edu.bu.labs.rttt.utils.Coordinate;
import edu.bu.labs.rttt.utils.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

// JAVA PROJECT IMPORTS
import src.labs.rttt.heuristics.Heuristics;


public class DepthThresholdedMinimaxAgent
    extends SearchAgent
{

    public static final int DEFAULT_MAX_DEPTH = 3;

    private int maxDepth;
    

    public DepthThresholdedMinimaxAgent(PlayerType myPlayerType)
    {
        super(myPlayerType);
        this.maxDepth = DEFAULT_MAX_DEPTH;
    }

    public final int getMaxDepth() { return this.maxDepth; }
    public void setMaxDepth(int i) { this.maxDepth = i; }

    public String getTabs(Node node)
    {
        StringBuilder b = new StringBuilder();
        for(int idx = 0; idx < node.getDepth(); ++idx)
        {
            b.append("\t");
        }
        return b.toString();
    }

    class ChildResult {
        double utility;
        Node node;

        ChildResult(double utility, Node node) {
            this.utility = utility;
            this.node = node;
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

    public Node search(Node node)
    {
        return this.minimax(node);
    }

    @Override
    public void afterGameEnds(final RecursiveTicTacToeGameView game) {}
}
