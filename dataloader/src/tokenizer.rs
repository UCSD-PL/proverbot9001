use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead};

extern crate regex;
use regex::Regex;

pub type Token = i64;

pub struct OpenIndexer<T>
where
    T: Eq + Hash + Clone,
{
    next_idx: i64,
    map: HashMap<T, i64>,
    frozen: bool,
}

pub type PickleableIndexer<T> = (i64, HashMap<T, i64>, bool);

impl<T> OpenIndexer<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        OpenIndexer {
            next_idx: 1,
            map: HashMap::new(),
            frozen: false,
        }
    }
    pub fn freeze(&mut self) {
        self.frozen = true;
    }
    pub fn lookup(&mut self, v: T) -> i64 {
        if !self.map.contains_key(&v) {
            if self.frozen {
                return 0;
            } else {
                self.map.insert(v.clone(), self.next_idx);
                self.next_idx += 1;
            }
        }
        *self.map.get(&v).unwrap()
    }
    pub fn reverse_lookup(&self, i: i64) -> T {
        self.map
            .iter()
            .find(|(_item, idx)| **idx == i)
            .expect("That token doesn't exist!")
            .0
            .clone()
    }
    pub fn to_pickleable(self) -> PickleableIndexer<T> {
        (self.next_idx, self.map, self.frozen)
    }
    pub fn from_pickleable(tup: PickleableIndexer<T>) -> Self {
        OpenIndexer {
            next_idx: tup.0,
            map: tup.1,
            frozen: tup.2,
        }
    }
    pub fn num_indices(&self) -> i64 {
        self.next_idx
    }
}

#[pyclass]
pub struct Tokenizer {
    use_unknowns: bool,
    num_reserved_tokens: usize,
    unknown_token: Token,
    token_dict: HashMap<String, Token>,
}

pub type PickleableTokenizer = (bool, usize, Token, HashMap<String, Token>);

impl Tokenizer {
    pub fn new(use_unknowns: bool, num_reserved_tokens: usize, keywords_filepath: String) -> Self {
        let keywords = io::BufReader::new(File::open(keywords_filepath).expect(&format!(
            "Couldn't open keywords file {}",
            keywords_filepath
        )))
        .lines()
        .collect();
        let first_token = (num_reserved_tokens + if use_unknowns { 1 } else { 0 }) as i64;
        let unknown_token = if use_unknowns { num_reserved_tokens } else { 0 } as i64;
        let mut token_dict = HashMap::new();
        for (idx, keyword) in keywords.into_iter().enumerate() {
            token_dict.insert(keyword, idx as i64 + first_token);
        }
        Tokenizer {
            use_unknowns,
            num_reserved_tokens,
            unknown_token,
            token_dict,
        }
    }
    pub fn tokenize(&self, sentence: &str) -> Vec<Token> {
        let words = get_words(sentence);
        words
            .into_iter()
            .flat_map(|word| match self.token_dict.get(word) {
                None => {
                    if self.use_unknowns {
                        Some(self.unknown_token)
                    } else {
                        None
                    }
                }
                Some(tok) => Some(*tok),
            })
            .collect()
    }
    pub fn to_pickleable(self) -> PickleableTokenizer {
        (
            self.use_unknowns,
            self.num_reserved_tokens,
            self.unknown_token,
            self.token_dict,
        )
    }
    pub fn from_pickleable(tup: PickleableTokenizer) -> Self {
        Tokenizer {
            use_unknowns: tup.0,
            num_reserved_tokens: tup.1,
            unknown_token: tup.2,
            token_dict: tup.3,
        }
    }
    pub fn num_tokens(&self) -> i64 {
        (self.token_dict.len() + self.num_reserved_tokens + if self.use_unknowns { 1 } else { 0 })
            as i64
    }
}
pub fn get_words(string: &str) -> Vec<&str> {
    lazy_static! {
        static ref WORDS: Regex = Regex::new(r"(,|\.+|:\b|:=|\)|\()|(([[:word:]]|')+)").unwrap();
    }
    WORDS.find_iter(string).map(|m| m.as_str()).collect()
}
