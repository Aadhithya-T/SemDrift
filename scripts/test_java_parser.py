import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

sample_code = b"""
public class Sample {
    /**
     * Adds two integers together.
     * @param a first number
     * @param b second number
     * @return the sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
}
"""

tree = parser.parse(sample_code)
root = tree.root_node

def walk(node, depth=0):
    print("  " * depth + node.type)
    for child in node.children:
        walk(child, depth + 1)

walk(root)