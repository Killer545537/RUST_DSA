//This contains the implementation for the functions where we can insert a word more than 1 time and it stores the number of times we insert the same word

use std::collections::HashMap;

#[derive(Debug)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    word_endings_count: usize, //Count the number of words ending with the character
    prefix_count: usize, //Count the number of times a word contains the character
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            word_endings_count: 0,
            prefix_count: 0,
        }
    }
}

#[derive(Debug)]
struct Trie {
    root: TrieNode,
}

impl Trie {
    fn new() -> Self {
        Trie {
            root: TrieNode::new()
        }
    }

    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for c in word.chars() {
            node = node.children.entry(c).or_insert_with(TrieNode::new);
            node.prefix_count += 1;
        }

        node.word_endings_count += 1;
    }

    pub fn count_words_equal_to(&self, word: &str) -> usize {
        let mut node = &self.root;
        for c in word.chars() {
            match node.children.get(&c) {
                None => return 0,
                Some(n) => node = n,
            }
        }

        node.prefix_count
    }

    pub fn count_words_starting_with(&self, prefix: &str) -> usize {
        let mut node = &self.root;
        for c in prefix.chars() {
            match node.children.get(&c) {
                None => return 0,
                Some(n) => node = n,
            }
        }

        node.prefix_count
    }

    pub fn erase(&mut self, word: &str) { //Assume the word exists in the Trie
        let mut node = &mut self.root;
        for c in word.chars() {
            match node.children.get_mut(&c) {
                None => return,
                Some(n) => {
                    node = n;
                    node.prefix_count -= 1;
                }
            }
        }
        node.word_endings_count -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trie_test() {
        let mut trie = Trie::new();
        trie.insert("samsung");
        trie.insert("samsung");
        trie.insert("vivo");
        trie.erase("vivo");
        assert_eq!(trie.count_words_equal_to("samsung"), 2);
        assert_eq!(trie.count_words_starting_with("vivo"), 0);
    }
}