use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// A Trie is a tree-based data structure for efficiently finding the longest
// matching prefix of a string among a vocabulary. This indexed Trie does the
// same thing, but also holds an index for each vocabulary word so they can be
// efficiently indexed during prefixing instead of having to scan through the
// vocabulary after.

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
struct IndexedTrieNode {
    value: Option<char>,
    index: Option<i64>,
    children: HashMap<char, IndexedTrieNode>,
}
#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IndexedTrie {
    root_node: IndexedTrieNode,
}

impl IndexedTrie {
    // Create a TrieStruct
    pub fn new() -> IndexedTrie {
        IndexedTrie {
            root_node: IndexedTrieNode {
                value: None,
                index: None,
                children: HashMap::new(),
            },
        }
    }

    // Insert a string
    pub fn insert(&mut self, new_idx: i64, string_val: &str) {
        let mut current_node = &mut self.root_node;
        let char_list: Vec<char> = string_val.chars().collect();
        let mut last_match = 0;
        for (letter_index, letter) in char_list.iter().enumerate() {
            if current_node.children.contains_key(letter) {
                current_node = current_node.children.get_mut(letter).unwrap();
                last_match = letter_index + 1;
            } else {
                last_match = letter_index;
                break;
            }
        }

        if last_match < char_list.len() {
            for new_counter in last_match..char_list.len() {
                current_node.children.insert(
                    char_list[new_counter],
                    IndexedTrieNode {
                        value: Some(char_list[new_counter]),
                        index: None,
                        children: HashMap::new(),
                    },
                );
                current_node = current_node
                    .children
                    .get_mut(&char_list[new_counter])
                    .unwrap();
            }
        }
        current_node.index = Some(new_idx);
    }

    pub fn longest_prefix<'a>(&self, sentence: &'a str) -> Option<(i64, &'a str)> {
        let mut cur_node = &self.root_node;
        let mut longest_so_far: Option<(i64, &str)> = None;
        for (char_idx, c) in sentence.chars().enumerate() {
            match cur_node.index {
                Some(idx_val) => {
                    longest_so_far = Some((idx_val, &sentence[..char_idx]));
                }
                None => (),
            };
            if cur_node.children.contains_key(&c) {
                cur_node = cur_node.children.get(&c).unwrap();
            } else {
                return longest_so_far;
            }
        }
        match cur_node.index {
            Some(idx_val) => longest_so_far = Some((idx_val, sentence)),
            None => (),
        }
        longest_so_far
    }

    // Find a string
    pub fn find(&mut self, string_val: &str) -> bool {
        let mut current_node = &mut self.root_node;
        let char_list: Vec<char> = string_val.chars().collect();

        for counter in 0..char_list.len() {
            if !current_node.children.contains_key(&char_list[counter]) {
                return false;
            } else {
                current_node = current_node.children.get_mut(&char_list[counter]).unwrap();
            }
        }
        return true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find() {
        // Create Trie
        let mut trie_test = IndexedTrie::new();

        // Insert Stuff
        trie_test.insert(1, "Test");
        trie_test.insert(2, "Tea");
        trie_test.insert(3, "Background");
        trie_test.insert(4, "Back");
        trie_test.insert(5, "Brown");

        // Find Stuff
        assert_eq!(trie_test.find("Testing"), false);
        assert_eq!(trie_test.find("Brown"), true);
    }
    #[test]
    fn test_longest_prefix() {
        // Create Trie
        let mut trie_test = IndexedTrie::new();

        // Insert Stuff
        trie_test.insert(1, "Test");
        trie_test.insert(2, "Tea");
        trie_test.insert(3, "Background");
        trie_test.insert(4, "Back");
        trie_test.insert(5, "Brown");

        // Find Stuff
        assert_eq!(trie_test.longest_prefix("Testing"), Some((1, "Test")));
        assert_eq!(
            trie_test.longest_prefix("Background"),
            Some((3, "Background"))
        );
    }
}
