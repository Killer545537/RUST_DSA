use std::collections::HashMap;


///This does not need a Box<TrieNode> since HashMap is already heap allocated which can help it to store recursive types by managing memory internally
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end_of_word: bool
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            is_end_of_word: false,
        }
    }
}
pub struct Trie {
    root: TrieNode
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            root: TrieNode::new()
        }
    }

    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for c in word.chars() {
            node = node.children.entry(c).or_insert_with(TrieNode::new);
        }
        node.is_end_of_word = true;
    }

    pub fn search(&self, word: &str) -> bool {
        let mut node = &self.root;
        for c in word.chars() {
            match node.children.get(&c) {
                None => return false, //If the character is not found, the trie does not contain it
                Some(n) => node = n,
            }
        }

        node.is_end_of_word //If the last character is the end of a word
    }

    pub fn starts_with(&self, prefix: &str) -> bool { //Same as search
        let mut node = &self.root;
        for c in prefix.chars() {
            match node.children.get(&c) {
                None => return false,
                Some(n) => node = n,
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trie_test() {
        let mut trie = Trie::new();
        trie.insert("apple");
        assert_eq!(trie.search("apple"), true);
        assert_eq!(trie.search("appl"), false);
        assert_eq!(trie.starts_with("appl"), true);
    }
}